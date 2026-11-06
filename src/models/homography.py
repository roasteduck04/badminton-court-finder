"""Differentiable, batched weighted DLT homography estimation (torch).

``fit_homography(src, dst, weights)`` estimates H with dst ~ H @ src for a
batch of point sets that share a fixed number of slots N; points that should
not participate get weight 0, so every sample is solved in one batched call.
Both point sets are Hartley-normalized (weighted centroid at the origin, mean
distance sqrt(2)) before solving, and the result is de-normalized.

Two solvers:
    "solve" (default): inhomogeneous DLT with h33 = 1 in normalized
        coordinates, solved from Tikhonov-regularized normal equations. Its
        gradient is finite even for degenerate inputs (e.g. all predicted
        points collapsed together early in training), so it is the one the
        loss uses. h33 = 1 is safe after normalization: h33 is the projective
        depth of the source centroid, which is non-zero for any court seen by
        a real camera.
    "svd": classic homogeneous DLT, h = right singular vector of the
        smallest singular value. Gradients are only well defined when the
        singular values are distinct (non-degenerate inputs).

All computations are done in float64 unless ``dtype`` says otherwise.
"""

import math

import torch

_EPS = 1e-12


def hartley_normalization(pts, weights):
    """Similarity transforms that normalize each point set.

    Args:
        pts: (B, N, 2)
        weights: (B, N) non-negative

    Returns:
        (pts_normalized (B, N, 2), T (B, 3, 3)) with pts_normalized = T @ pts.
    """
    w = weights / weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    c = (w.unsqueeze(-1) * pts).sum(dim=1)  # (B, 2)
    centred = pts - c.unsqueeze(1)
    dist = torch.sqrt((centred ** 2).sum(dim=-1) + 1e-12)
    mean_dist = (w * dist).sum(dim=-1).clamp_min(1e-9)
    s = math.sqrt(2.0) / mean_dist  # (B,)

    B = pts.shape[0]
    T = torch.zeros(B, 3, 3, dtype=pts.dtype, device=pts.device)
    T[:, 0, 0] = s
    T[:, 1, 1] = s
    T[:, 0, 2] = -s * c[:, 0]
    T[:, 1, 2] = -s * c[:, 1]
    T[:, 2, 2] = 1.0
    return centred * s[:, None, None], T


def _dlt_matrix(src, dst, weights):
    """(B, 2N, 9) weighted DLT design matrix for dst ~ H src."""
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    r1 = torch.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u], dim=-1)
    r2 = torch.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], dim=-1)
    A = torch.stack([r1, r2], dim=2)  # (B, N, 2, 9)
    A = A * weights[..., None, None]
    return A.reshape(src.shape[0], -1, 9)


def _normalize_h(H):
    h22 = H[:, 2:3, 2:3]
    sign = torch.where(h22 >= 0, torch.ones_like(h22), -torch.ones_like(h22))
    denom = sign * h22.abs().clamp_min(_EPS)
    return H / denom


def fit_homography(src, dst, weights=None, method="solve", reg=1e-9, dtype=torch.float64):
    """Batched weighted DLT.

    Args:
        src: (B, N, 2) or (N, 2) source points (e.g. court template, metres)
        dst: (B, N, 2) or (N, 2) destination points (e.g. image pixels)
        weights: (B, N) or (N,) non-negative row weights (None = all 1)
        method: "solve" or "svd" (see module docstring)
        reg: Tikhonov regularization for "solve", relative to the normal
            matrix trace
        dtype: compute dtype

    Returns:
        (B, 3, 3) or (3, 3) homography (H[2, 2] = 1), in ``dtype``.
    """
    squeeze = dst.dim() == 2
    if squeeze:
        dst = dst.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
    if src.dim() == 2:
        src = src.unsqueeze(0)
    if src.shape[0] != dst.shape[0]:
        src = src.expand(dst.shape[0], -1, -1)
    src = src.to(dtype)
    dst = dst.to(dtype)
    if weights is None:
        weights = torch.ones(dst.shape[:2], dtype=dtype, device=dst.device)
    weights = weights.to(dtype)

    src_n, T1 = hartley_normalization(src, weights)
    dst_n, T2 = hartley_normalization(dst, weights)
    A = _dlt_matrix(src_n, dst_n, weights)  # (B, 2N, 9)
    B = A.shape[0]

    if method == "solve":
        A8, a9 = A[..., :8], A[..., 8:]
        M = A8.transpose(1, 2) @ A8  # (B, 8, 8)
        rhs = -(A8.transpose(1, 2) @ a9)  # (B, 8, 1)
        trace = M.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(_EPS)
        eye = torch.eye(8, dtype=dtype, device=A.device)
        M = M + (reg * trace)[:, None, None] * eye
        h8 = torch.linalg.solve(M, rhs).squeeze(-1)
        h = torch.cat([h8, torch.ones(B, 1, dtype=dtype, device=A.device)], dim=-1)
    elif method == "svd":
        if A.shape[1] < 9:  # need a square-or-tall matrix to expose the null vector
            pad = torch.zeros(B, 9 - A.shape[1], 9, dtype=dtype, device=A.device)
            A = torch.cat([A, pad], dim=1)
        _, _, Vh = torch.linalg.svd(A, full_matrices=False)
        h = Vh[:, -1, :]
    else:
        raise ValueError(f"unknown method {method!r}")

    Hn = h.reshape(B, 3, 3)
    H = torch.linalg.inv(T2) @ Hn @ T1
    H = _normalize_h(H)
    return H[0] if squeeze else H


def project(H, pts):
    """Apply (B, 3, 3) homographies to (B, N, 2) points (or unbatched).

    Returns (projected (…, N, 2), w (…, N)) where w is the homogeneous
    depth before division (w <= 0 means the point is behind the camera).
    """
    pts = pts.to(H.dtype)
    ones = torch.ones(pts.shape[:-1] + (1,), dtype=H.dtype, device=pts.device)
    ph = torch.cat([pts, ones], dim=-1)  # (…, N, 3)
    out = ph @ H.transpose(-1, -2)
    w = out[..., 2]
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    denom = sign * w.abs().clamp_min(1e-9)
    return out[..., :2] / denom.unsqueeze(-1), w
