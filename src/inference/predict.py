"""Inference pipeline for CourtVisionNet.

``CourtPredictor`` loads a checkpoint, runs the model on a BGR image and
turns its outputs into a ``CourtDetection``:

1. letterbox the image to the model's square input (the same transform as
   training), build the 7-channel input and keep the first ``in_channels``;
2. one forward pass (segmentation and keypoints share it);
3. decode the heatmaps with argmax + local refinement (src/inference/decode.py)
   or, optionally, use the model's soft-argmax ``offsets``;
4. map keypoints back to normalized ORIGINAL-image coordinates;
5. relabel the keypoints into the canonical convention with
   ``court_geometry.canonicalize`` (the rule the ground-truth labels follow).
   The court looks the same mirrored or rotated 180 degrees, and training uses
   a symmetric (rot180) loss, so the network may output either labeling; the
   canonical one is a function of the image geometry and is applied here;
6. optionally fit a robust template->image homography (MAGSAC/RANSAC) to the
   confident keypoints, sanity-check it, and fill in outliers / unseen
   points by reprojecting the court template.

Optional test-time augmentation (``tta=True``) adds a horizontally flipped
pass; its labels are mirrored, so they are relabeled with
``court_geometry.canonicalize`` (not a fixed left/right swap) before the two
views are fused.

Checkpoints: the v2 format (``format_version`` 2, with ``config``) rebuilds the
exact model from its config. The legacy Run-7 checkpoint (no config, old
visibility head ``kpt_head.visibility_conv.*``) is loaded non-strictly with a
warning: its visibility head is re-initialized, so the predictor reports the
heatmap peak value as visibility instead; it was trained without input
normalization and on squashed (not letterboxed) images, so those are
reproduced.
"""

import inspect
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from src.court_geometry import (
    CORNER_INDICES, COURT_KEYPOINTS_TEMPLATE, NUM_KEYPOINTS, SYMMETRY_PERMS,
    canonicalize, court_orientation, get_court_lines, has_homography_support,
    validate_quadrilateral,
)
from src.inference.decode import decode_heatmaps
from src.models.courtvisionnet import CourtVisionNet
from src.models.keypoint_head import BETA_EPS
from src.preprocessing.channels import generate_channels
from src.preprocessing.letterbox import content_mask, keypoints_from_letterbox, letterbox

# generate_channels(image, valid=...) zeroes derived channels in the padding;
# pass the content mask when supported so inference matches training.
_CHANNELS_TAKE_VALID = "valid" in inspect.signature(generate_channels).parameters

VISIBILITY_THRESHOLD = 0.5
MIN_POINTS_FOR_HOMOGRAPHY = 4
FILL_MODES = ("none", "outliers", "in_frame", "all_projected", "all")


@dataclass
class CourtDetection:
    """Result of ``CourtPredictor.predict()`` on a single image.

    All keypoint coordinates are normalized (x, y) in the ORIGINAL image.
    """

    keypoints: np.ndarray                  # (30, 2) after optional homography refinement
    visibility: np.ndarray                 # (30,) visibility probabilities
    confidence: float = 0.0                # mean visibility of the visible keypoints
    homography: Optional[np.ndarray] = None    # (3, 3) template (metres) -> original pixels
    seg_mask: Optional[np.ndarray] = None      # (S, S) float32 seg probabilities on the model canvas
    projected_lines: Optional[list] = None     # ((x1, y1), (x2, y2)) pixel segments
    raw_keypoints: Optional[np.ndarray] = None  # (30, 2) straight from the decoder
    inlier_mask: Optional[np.ndarray] = None    # (30,) bool RANSAC inliers
    image_size: Optional[list] = None           # [h, w] of the original image
    keypoint_confidence: Optional[np.ndarray] = None  # (30,) decoder peak confidence
    projected_keypoints: Optional[np.ndarray] = None  # (30, 2) template reprojection
    homography_status: str = "not_run"          # "ok" or why no homography was kept
    canonical_perm: Optional[str] = None        # symmetry applied to the decoder labels, or None
    seg_mask_image: Optional[np.ndarray] = None  # (h, w) uint8 binary mask in the original image

    def __post_init__(self):
        if self.raw_keypoints is None:
            self.raw_keypoints = np.array(self.keypoints, dtype=np.float64, copy=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _inverse_softplus(y):
    y = max(float(y), 1e-4)
    return y + math.log(-math.expm1(-y))


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------

def _is_legacy_state_dict(state_dict):
    return any(
        k.startswith("kpt_head.visibility_conv") or k.startswith("kpt_head.vis_reduce.p")
        or k == "kpt_head.beta"
        for k in state_dict
    )


def load_model(checkpoint_path, device="cpu", image_size=None, heatmap_size=None):
    """Build a CourtVisionNet from a checkpoint.

    ``image_size`` / ``heatmap_size`` are only used when the checkpoint has no
    config (legacy or bare state dict); defaults 640 and image_size // 4.

    Returns:
        (model in eval mode on ``device``, info dict with keys "legacy",
        "config", "image_size", "in_channels", "heatmap_kind")
    """
    # weights_only=False: v2 checkpoints carry config/history/RNG state.
    # Only load checkpoints you produced yourself.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config") or {}
        version = ckpt.get("format_version", 1)
    else:
        state_dict, config, version = ckpt, {}, None
    if not isinstance(config, dict):
        config = vars(config)

    legacy = _is_legacy_state_dict(state_dict)
    if legacy:
        state_dict = dict(state_dict)
        in_channels = state_dict["backbone.stem.0.weight"].shape[1]
        num_keypoints = state_dict["kpt_head.heatmap_conv.6.weight"].shape[0]
        size = image_size or 640
        model = CourtVisionNet(
            in_channels=in_channels, num_keypoints=num_keypoints, image_size=size,
            heatmap_size=heatmap_size or size // 4, pretrained=False, normalize_input=False,
        )
        if "kpt_head.beta" in state_dict:
            beta = float(state_dict.pop("kpt_head.beta"))
            state_dict["kpt_head.beta_raw"] = torch.tensor(_inverse_softplus(beta - BETA_EPS))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        warnings.warn(
            f"Loaded legacy (pre-v2) checkpoint {checkpoint_path} non-strictly: "
            f"{len(missing)} missing keys (re-initialized: visibility head), "
            f"{len(unexpected)} unexpected keys ignored. Visibility outputs are UNRELIABLE; "
            "input normalization is disabled and images are squashed to match how it was trained.",
            UserWarning, stacklevel=2,
        )
        info = {"legacy": True, "config": {}, "image_size": size,
                "in_channels": in_channels, "heatmap_kind": "gaussian"}
    else:
        params = inspect.signature(CourtVisionNet.__init__).parameters
        kwargs = {k: v for k, v in config.items() if k in params and k != "self"}
        kwargs["pretrained"] = False
        kwargs.setdefault("image_size", image_size or 640)
        kwargs.setdefault("heatmap_size", heatmap_size or kwargs["image_size"] // 4)
        if "in_channels" not in kwargs and "backbone.stem.0.weight" in state_dict:
            kwargs["in_channels"] = state_dict["backbone.stem.0.weight"].shape[1]
        model = CourtVisionNet(**kwargs)
        model.load_state_dict(state_dict, strict=True)
        if version is not None and version != 2:
            warnings.warn(f"checkpoint format_version={version}, expected 2", UserWarning, stacklevel=2)
        kind = "logits" if config.get("heatmap_loss", "mse") == "js" else "gaussian"
        info = {"legacy": False, "config": config, "image_size": model.image_size,
                "in_channels": model.in_channels, "heatmap_kind": kind}

    model.to(device)
    model.eval()
    return model, info


# ----------------------------------------------------------------------
# Homography fitting
# ----------------------------------------------------------------------

def _project(H, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = ph[:, 2]
    safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return ph[:, :2] / safe[:, None], w


def fit_court_homography(keypoints, candidates, image_hw, threshold_frac=0.01, min_inliers=6):
    """Robustly fit template (metres) -> image (pixels) to candidate keypoints.

    Args:
        keypoints: (30, 2) normalized original-image coords
        candidates: (30,) bool, points allowed in the fit
        image_hw: (h, w)
        threshold_frac: inlier threshold as a fraction of max(h, w)
        min_inliers: minimum inlier count to accept the fit

    Returns:
        (H or None, inlier_mask (30,) bool or None, status str). Sanity
        checks: enough inliers, inliers determine a homography (not all
        collinear), positive handedness (a real camera cannot mirror the
        court), and a convex outer-corner quadrilateral when all four corners
        are in front of the camera.
    """
    h, w = image_hw
    candidates = np.asarray(candidates, dtype=bool)
    idx = np.flatnonzero(candidates)
    need = max(MIN_POINTS_FOR_HOMOGRAPHY, min_inliers)
    if len(idx) < need:
        return None, None, f"too_few_points ({len(idx)} < {need})"
    if not has_homography_support(candidates):
        return None, None, "degenerate_points"

    src = COURT_KEYPOINTS_TEMPLATE[idx].astype(np.float64)
    dst = (np.asarray(keypoints, dtype=np.float64)[idx] * np.array([w, h])).astype(np.float64)
    thr = float(threshold_frac * max(h, w))
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    try:
        H, inl = cv2.findHomography(src, dst, method, thr, maxIters=5000, confidence=0.999)
    except cv2.error:
        H, inl = None, None
    if H is None or inl is None:
        return None, None, "fit_failed"

    inliers = np.zeros(len(candidates), dtype=bool)
    inliers[idx[inl.ravel() > 0]] = True
    if inliers.sum() < need:
        return None, None, f"too_few_inliers ({int(inliers.sum())} < {need})"
    if not has_homography_support(inliers):
        return None, None, "degenerate_inliers"

    orient = court_orientation(keypoints, inliers, [h, w])
    if orient is None or orient[0] <= 0:
        return None, None, "mirrored"

    corners, depth = _project(H, COURT_KEYPOINTS_TEMPLATE[CORNER_INDICES])
    if np.all(depth > 0) and not validate_quadrilateral(corners):
        return None, None, "non_convex"
    return H, inliers, "ok"


def refine_keypoints(raw, H, candidates, inliers, image_hw, fill="in_frame"):
    """Replace keypoints by the template reprojection according to ``fill``.

    fill:
        "none":          keep raw keypoints
        "outliers":      replace candidates rejected by RANSAC
        "in_frame":      outliers + non-candidate points whose reprojection
                         lies inside the image (default)
        "all_projected": every non-inlier point
        "all":           every point (pure template reprojection)
    Points that reproject behind the camera are never filled.

    Returns (keypoints (30, 2), projected (30, 2)).
    """
    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}, got {fill!r}")
    h, w = image_hw
    proj_px, depth = _project(H, COURT_KEYPOINTS_TEMPLATE)
    proj = proj_px / np.array([w, h])
    front = depth > 0
    in_frame = front & np.all((proj >= 0) & (proj <= 1), axis=1)
    outliers = candidates & ~inliers
    if fill == "none":
        replace = np.zeros_like(front)
    elif fill == "outliers":
        replace = outliers
    elif fill == "in_frame":
        replace = outliers | (~candidates & in_frame)
    elif fill == "all_projected":
        replace = ~inliers
    else:
        replace = np.ones_like(front)
    replace = replace & front
    out = np.array(raw, dtype=np.float64, copy=True)
    out[replace] = proj[replace]
    return out, proj


def projected_court_lines(H):
    """Template court lines projected through H as pixel segments (front only)."""
    proj, depth = _project(H, COURT_KEYPOINTS_TEMPLATE)
    lines = []
    for a, b in get_court_lines():
        if depth[a] > 0 and depth[b] > 0:
            lines.append(((float(proj[a, 0]), float(proj[a, 1])),
                          (float(proj[b, 0]), float(proj[b, 1]))))
    return lines


# ----------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------

class CourtPredictor:
    """Loads a CourtVisionNet checkpoint and runs end-to-end court detection.

    Args:
        checkpoint_path: v2 or legacy checkpoint (or a bare state dict)
        device: torch device; None = cuda if available else cpu
        image_size, heatmap_size: only for checkpoints without a config
        decode_method: "local" (argmax + local refinement) or "soft_argmax"
        refine: "dark" or None, local refinement of the argmax
        use_offsets: use the model's soft-argmax ``offsets`` instead of
            decoding the heatmaps
        vis_threshold: visibility probability for a point to be "visible"
        conf_threshold: minimum decoder confidence for a point to enter the
            homography fit
        homography: fit and apply the robust homography refinement
        fill: see ``refine_keypoints``
        ransac_threshold: inlier threshold as a fraction of max(h, w)
        min_inliers: minimum inliers to accept a homography
        tta: horizontal-flip test-time augmentation by default in predict()
        tta_combine: "weighted" (confidence-weighted mean) or "median"
        canonical_output: relabel the decoded keypoints into the canonical
            convention (orientation from the visible points); False keeps the
            network's own labeling
        resize_mode: "auto" (letterbox; "squash" for legacy checkpoints),
            "letterbox" or "squash"
    """

    def __init__(self, checkpoint_path, device=None, image_size=None, heatmap_size=None,
                 decode_method="local", refine="dark", use_offsets=False,
                 vis_threshold=VISIBILITY_THRESHOLD, conf_threshold=0.0, homography=True,
                 fill="in_frame", ransac_threshold=0.01, min_inliers=6, tta=False,
                 tta_combine="weighted", resize_mode="auto", canonical_output=True):
        if device is None or (str(device).startswith("cuda") and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if fill not in FILL_MODES:
            raise ValueError(f"fill must be one of {FILL_MODES}, got {fill!r}")
        self.device = device
        self.model, self.info = load_model(checkpoint_path, device, image_size, heatmap_size)
        self.legacy = self.info["legacy"]
        self.image_size = self.info["image_size"]
        self.in_channels = self.info["in_channels"]
        self.heatmap_kind = self.info["heatmap_kind"]
        self.heatmap_size = self.model.heatmap_size
        self.decode_method = decode_method
        self.refine = refine
        self.use_offsets = use_offsets
        self.vis_threshold = vis_threshold
        self.conf_threshold = conf_threshold
        self.use_homography = homography
        self.fill = fill
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.tta = tta
        self.tta_combine = tta_combine
        self.canonical_output = canonical_output
        if resize_mode == "auto":
            resize_mode = "squash" if self.legacy else "letterbox"
        if resize_mode not in ("letterbox", "squash"):
            raise ValueError(f"resize_mode must be auto/letterbox/squash, got {resize_mode!r}")
        self.resize_mode = resize_mode

    # -- single forward pass --------------------------------------------

    @torch.no_grad()
    def _run(self, image):
        """One forward pass. Returns kps (30, 2) normalized original coords,
        vis (30,), conf (30,), seg prob on the canvas (S, S) and in the
        original image (h, w), float32."""
        h, w = image.shape[:2]
        S = self.image_size
        if self.resize_mode == "letterbox":
            canvas, info = letterbox(image, S)
        else:
            canvas, info = cv2.resize(image, (S, S), interpolation=cv2.INTER_LINEAR), None
        if info is not None and _CHANNELS_TAKE_VALID:
            channels = generate_channels(canvas, valid=content_mask(info))
        else:
            channels = generate_channels(canvas)
        channels = channels[..., :self.in_channels]
        x = torch.from_numpy(np.ascontiguousarray(channels.transpose(2, 0, 1))).unsqueeze(0)
        out = self.model(x.float().to(self.device))

        heatmaps = out["heatmaps"][0].float()
        beta = float(out["beta"]) if "beta" in out else 1.0
        coords, conf = decode_heatmaps(
            heatmaps, method=self.decode_method, refine=self.refine,
            kind=self.heatmap_kind, beta=beta,
        )
        if self.use_offsets:
            coords = out["offsets"][0].float()
        kps_c = coords.cpu().numpy().astype(np.float64)
        conf = conf.cpu().numpy().astype(np.float64)
        if self.legacy:
            # The legacy visibility head is re-initialized (random); the peak
            # of its Gaussian-regressed heatmaps is the best visibility proxy.
            vis = np.clip(conf, 0.0, 1.0)
        else:
            vis = torch.sigmoid(out["visibility"][0].float()).cpu().numpy().astype(np.float64)
        seg_canvas = torch.sigmoid(out["seg_logits"][0, 0].float()).cpu().numpy()
        seg = seg_canvas

        if info is not None:
            kps = keypoints_from_letterbox(kps_c, info)
            seg = seg[info.pad_y:info.pad_y + info.content_h, info.pad_x:info.pad_x + info.content_w]
        else:
            kps = kps_c
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
        return kps, vis, conf, seg_canvas, seg

    # -- TTA ------------------------------------------------------------

    def _align_flipped(self, kps0, vis0, kps_f, vis_f, image_hw):
        """Permutation that relabels the (un-flipped) mirror-view prediction
        like the original view's prediction."""
        mask0 = vis0 > self.vis_threshold
        mask_f = vis_f > self.vis_threshold
        _, _, name_f = canonicalize(kps_f, mask_f, image_hw)
        _, _, name_0 = canonicalize(kps0, mask0, image_hw)
        if name_f is not None:
            perm = np.asarray(SYMMETRY_PERMS[name_f])
            if name_0 is not None:  # original view not canonical: map back to its labels
                perm = perm[np.asarray(SYMMETRY_PERMS[name_0])]
            return perm
        # Orientation undeterminable: pick the relabeling closest to view 0.
        best, best_err = None, np.inf
        for perm in SYMMETRY_PERMS.values():
            perm = np.asarray(perm)
            both = mask0 & mask_f[perm]
            if not both.any():
                continue
            err = np.median(np.linalg.norm(kps0[both] - kps_f[perm][both], axis=1))
            if err < best_err:
                best, best_err = perm, err
        return best if best is not None else np.asarray(SYMMETRY_PERMS["col_flip"])

    def _run_tta(self, image):
        h, w = image.shape[:2]
        k0, v0, c0, canvas0, s0 = self._run(image)
        kf, vf, cf, _, sf = self._run(cv2.flip(image, 1))
        kf = kf.copy()
        kf[:, 0] = 1.0 - kf[:, 0]
        sf = sf[:, ::-1]
        perm = self._align_flipped(k0, v0, kf, vf, (h, w))
        kf, vf, cf = kf[perm], vf[perm], cf[perm]

        kps_views = np.stack([k0, kf])
        if self.tta_combine == "median":
            kps = np.median(kps_views, axis=0)
        else:
            wts = np.stack([c0 * v0, cf * vf]) + 1e-6
            kps = (wts[..., None] * kps_views).sum(0) / wts.sum(0)[:, None]
        # Canvas-space seg comes from the un-flipped view (the flipped
        # letterbox canvas can be offset by a pixel); image-space is averaged.
        return kps, 0.5 * (v0 + vf), 0.5 * (c0 + cf), canvas0, 0.5 * (s0 + sf)

    def _canonicalize(self, kps, vis, conf, image_hw):
        """Relabel per-keypoint arrays into the canonical convention.

        Orientation comes from the points above the visibility threshold (as
        for the ground truth, only points actually in view); returns the
        arrays unchanged and None if it cannot be determined.
        """
        _, _, name = canonicalize(kps, vis > self.vis_threshold, image_hw)
        if name is None or name == "identity":
            return kps, vis, conf, name
        perm = np.asarray(SYMMETRY_PERMS[name])
        return kps[perm], vis[perm], conf[perm], name

    # -- public API -----------------------------------------------------

    @torch.no_grad()
    def predict(self, image, tta=None):
        """Run inference on a single BGR uint8 image of any size.

        Returns:
            CourtDetection (see contract §8).
        """
        use_tta = self.tta if tta is None else tta
        h, w = image.shape[:2]
        if use_tta:
            raw, vis, conf, seg_canvas, seg = self._run_tta(image)
        else:
            raw, vis, conf, seg_canvas, seg = self._run(image)

        perm_name = None
        if self.canonical_output:
            raw, vis, conf, perm_name = self._canonicalize(raw, vis, conf, (h, w))

        keypoints = raw.copy()
        H, inliers, projected, lines, status = None, None, None, None, "disabled"
        if self.use_homography:
            candidates = (vis > self.vis_threshold) & (conf >= self.conf_threshold)
            H, inliers, status = fit_court_homography(
                raw, candidates, (h, w), self.ransac_threshold, self.min_inliers,
            )
            if H is not None:
                keypoints, projected = refine_keypoints(raw, H, candidates, inliers, (h, w), self.fill)
                lines = projected_court_lines(H)

        visible = vis > self.vis_threshold
        return CourtDetection(
            keypoints=keypoints,
            visibility=vis,
            confidence=float(vis[visible].mean()) if visible.any() else 0.0,
            homography=H,
            seg_mask=seg_canvas.astype(np.float32),
            projected_lines=lines,
            raw_keypoints=raw,
            inlier_mask=inliers,
            image_size=[h, w],
            keypoint_confidence=conf,
            projected_keypoints=projected,
            homography_status=status,
            seg_mask_image=(seg > 0.5).astype(np.uint8),
            canonical_perm=perm_name,
        )

    def predict_tta(self, image):
        """``predict`` with horizontal-flip test-time augmentation."""
        return self.predict(image, tta=True)


__all__ = [
    "CourtDetection", "CourtPredictor", "VISIBILITY_THRESHOLD", "fit_court_homography",
    "load_model", "projected_court_lines", "refine_keypoints", "NUM_KEYPOINTS",
]
