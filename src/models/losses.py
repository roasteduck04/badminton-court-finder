"""Multi-task loss for CourtVisionNet.

Terms (all computed in float32, whatever dtype the predictions come in):

seg        BCE + per-image soft Dice on the court-line mask; an optional
           ``valid`` target (1 = real image pixel) excludes letterbox padding.
heatmap    "js": DSNT-style. The predicted heatmap is turned into a spatial
               distribution p = softmax(beta * logits), the Gaussian target
               into q = t / sum(t), and the loss is the Jensen-Shannon
               divergence JS(p, q) in nats (0 .. ln 2 = 0.69), averaged over
               visible keypoints.
           "mse": foreground-weighted squared error summed over the pixels of
               each heatmap, w = 1 + fg_weight * target, divided by that
               heatmap's target energy sum(w * t^2) (a constant for a given
               sigma), averaged over visible keypoints. This is the
               per-heatmap normalization (sum over pixels / num visible) made
               scale-free: it equals 1.0 for an all-zero prediction and 0 for
               a perfect one, so it no longer needs a weight of ~100.
offset     Euclidean distance between the soft-argmax coordinates and the GT
           keypoint, in normalized canvas units (0.01 = 6.4 px at 640),
           averaged over visible keypoints.
visibility BCE on the visibility logits over all keypoints.
reproj     Homography consistency: per sample with homography support, a
           weighted DLT (template metres -> predicted points in canvas px) is
           fit to the visible predicted points, the template is reprojected,
           and the Huber distance (delta = ``reproj_delta`` px) between each
           visible prediction and its reprojection is averaged. Gradients
           flow to the points; the fit itself is detached by default
           (``reproj_detach_fit``) for stability.

symmetric=True takes, per sample, the minimum of the keypoint terms
(heatmap + offset + reproj) over the {identity, rot180} relabelings of the
targets (heatmap channels, keypoints and visibility are all permuted), and
uses the same relabeling for the visibility BCE. This makes the loss
invariant to the 180-degree ambiguity of the court labeling.

Default weights (chosen so each weighted term is O(0.1-1) early in training):
    seg 1.0, heatmap 1.0 (js), offset 5.0, vis 0.5, reproj 0.01
Early on: seg ~1.4, JS ~0.6, offset 5 * ~0.3, reproj 0.01 * (~10-100 px).
Converged: JS ~0.1-0.3, offset 5 * ~0.005 (3 px), reproj 0.01 * ~1-2 px.
For heatmap_loss="mse" use heatmap_weight 1.0 and offset_weight ~1.0 (the
soft-argmax of a regressed Gaussian is centre-biased until beta grows).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS, has_homography_support
from src.models.homography import fit_homography, project


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits, averaged per image.

    ``valid`` (same shape as target, optional) masks out pixels to ignore.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, valid=None):
        prob = torch.sigmoid(pred.float()).flatten(1)
        target = target.float().flatten(1)
        if valid is not None:
            valid = valid.float().flatten(1)
            prob = prob * valid
            target = target * valid
        intersection = (prob * target).sum(-1)
        dice = (2.0 * intersection + self.smooth) / (prob.sum(-1) + target.sum(-1) + self.smooth)
        return (1.0 - dice).mean()


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence (nats) along the last dim of distributions."""
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def _masked_mean_per_sample(values, mask):
    """(B, K) values, (B, K) mask -> (B,) sums and (B,) counts."""
    return (values * mask).sum(-1), mask.sum(-1)


class CourtVisionLoss(nn.Module):
    """Combined multi-task loss for CourtVisionNet (see module docstring).

    Expects:
        pred: dict with "seg_logits" (B,1,H,W), "heatmaps" (B,K,h,w) logits,
              "offsets" (B,K,2), "visibility" (B,K) logits, optional "beta"
        targets: dict with "mask" (B,1,H,W), "heatmaps" (B,K,h,w),
                 "keypoints" (B,K,2), "visibility" (B,K), optional "valid"
                 (B,1,H,W)

    Returns (total, components) where components maps
    seg_loss/heatmap_loss/offset_loss/visibility_loss/reproj_loss to floats
    (unweighted, batch means), plus rot180_fraction when symmetric.
    """

    def __init__(
        self,
        seg_weight=1.0,
        heatmap_weight=1.0,
        offset_weight=5.0,
        vis_weight=0.5,
        reproj_weight=0.01,
        heatmap_loss="js",
        symmetric=False,
        mse_fg_weight=9.0,
        reproj_delta=2.0,
        reproj_detach_fit=True,
        image_size=None,
    ):
        super().__init__()
        if heatmap_loss not in ("js", "mse"):
            raise ValueError(f"heatmap_loss must be 'js' or 'mse', got {heatmap_loss!r}")
        self.seg_weight = seg_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.vis_weight = vis_weight
        self.reproj_weight = reproj_weight
        self.heatmap_loss = heatmap_loss
        self.symmetric = symmetric
        self.mse_fg_weight = mse_fg_weight
        self.reproj_delta = reproj_delta
        self.reproj_detach_fit = reproj_detach_fit
        self.image_size = image_size
        self.dice_loss = DiceLoss()
        self.register_buffer(
            "template", torch.tensor(COURT_KEYPOINTS_TEMPLATE, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "rot180", torch.tensor(SYMMETRY_PERMS["rot180"], dtype=torch.long), persistent=False,
        )

    # ------------------------------------------------------------------
    # Per-keypoint terms, (B, K) each
    # ------------------------------------------------------------------

    def _heatmap_terms(self, logits, target, p=None):
        """``p`` is the precomputed softmax(beta * logits) (needed for "js")."""
        B, K = logits.shape[:2]
        logits = logits.reshape(B, K, -1)
        target = target.reshape(B, K, -1)
        if self.heatmap_loss == "js":
            mass = target.sum(-1, keepdim=True)
            q = target / mass.clamp_min(1e-12)
            return js_divergence(p, q), (mass.squeeze(-1) > 1e-6)
        w = 1.0 + self.mse_fg_weight * target
        energy = (w * target ** 2).sum(-1)
        err = (w * (logits - target) ** 2).sum(-1)
        return err / energy.clamp_min(1e-6), (energy > 1e-6)

    @staticmethod
    def _offset_terms(offsets, keypoints):
        return torch.sqrt(((offsets - keypoints) ** 2).sum(-1) + 1e-12)

    def _reproj_terms(self, offsets, vis, image_size):
        """(B, K) Huber reprojection residuals (zero where not applicable)."""
        B, K = vis.shape
        out = torch.zeros(B, K, dtype=offsets.dtype, device=offsets.device)
        if K != self.template.shape[0]:
            return out, torch.zeros_like(vis)
        vis_cpu = vis.detach().cpu().numpy()
        supported = [b for b in range(B) if has_homography_support(vis_cpu[b] > 0.5)]
        if not supported:
            return out, torch.zeros_like(vis)
        idx = torch.tensor(supported, device=offsets.device)
        pts = offsets[idx] * image_size  # (S, K, 2) canvas px
        w = vis[idx]
        fit_pts = pts.detach() if self.reproj_detach_fit else pts
        template = self.template.to(offsets.device)  # in case the loss was not .to(device)'d
        H = fit_homography(template, fit_pts, w)
        reproj, _ = project(H, template)
        resid = torch.sqrt(((pts.double() - reproj) ** 2).sum(-1) + 1e-12)
        d = self.reproj_delta
        huber = torch.where(resid < d, 0.5 * resid ** 2 / d, resid - 0.5 * d).to(out.dtype)
        mask = torch.zeros_like(vis)
        mask[idx] = (w > 0.5).to(vis.dtype)
        out = out.index_put((idx,), huber * mask[idx])
        return out, mask

    def _keypoint_loss(self, pred, heatmaps_t, keypoints_t, vis_t, p, image_size):
        """Per-sample keypoint terms for one labeling of the targets.

        Returns dict of (B,) per-sample (sum, count) pairs.
        """
        hm, hm_ok = self._heatmap_terms(pred["heatmaps"], heatmaps_t, p)
        hm_mask = vis_t * hm_ok.to(vis_t.dtype)
        off = self._offset_terms(pred["offsets"], keypoints_t)
        terms = {
            "heatmap": _masked_mean_per_sample(hm, hm_mask),
            "offset": _masked_mean_per_sample(off, vis_t),
        }
        if self.reproj_weight != 0.0:
            rp, rp_mask = self._reproj_terms(pred["offsets"], vis_t, image_size)
            terms["reproj"] = _masked_mean_per_sample(rp, rp_mask)
        else:
            z = torch.zeros(vis_t.shape[0], device=vis_t.device)
            terms["reproj"] = (z, z)
        vis_bce = F.binary_cross_entropy_with_logits(
            pred["visibility"], vis_t, reduction="none",
        ).mean(-1)
        terms["visibility"] = (vis_bce, torch.ones_like(vis_bce))
        return terms

    def _selection_score(self, terms):
        """(B,) per-sample weighted keypoint loss used to choose the labeling."""
        score = 0.0
        for name, weight in (("heatmap", self.heatmap_weight), ("offset", self.offset_weight),
                             ("reproj", self.reproj_weight)):
            s, n = terms[name]
            score = score + weight * s / n.clamp_min(1.0)
        return score

    def forward(self, pred, targets):
        pred = {
            k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in pred.items()
        }
        mask_t = targets["mask"].float()
        heatmaps_t = targets["heatmaps"].float()
        keypoints_t = targets["keypoints"].float()
        vis_t = targets["visibility"].float()
        valid = targets.get("valid")
        valid = valid.float() if valid is not None else None
        beta = pred.get("beta", 1.0)
        image_size = self.image_size or pred["seg_logits"].shape[-1]

        # Segmentation
        seg_logits = pred["seg_logits"]
        bce = F.binary_cross_entropy_with_logits(seg_logits, mask_t, reduction="none")
        if valid is not None:
            seg_bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            seg_bce = bce.mean()
        seg_loss = seg_bce + self.dice_loss(seg_logits, mask_t, valid)

        # Keypoints, for the identity labeling and optionally rot180
        p = None
        if self.heatmap_loss == "js":
            B, K = pred["heatmaps"].shape[:2]
            p = F.softmax(pred["heatmaps"].reshape(B, K, -1) * beta, dim=-1)
        terms = self._keypoint_loss(pred, heatmaps_t, keypoints_t, vis_t, p, image_size)
        rot_frac = None
        if self.symmetric:
            K = vis_t.shape[1]
            if K != len(SYMMETRY_PERMS["rot180"]):
                raise ValueError("symmetric loss needs the 30-keypoint layout")
            perm = self.rot180.to(vis_t.device)
            terms_r = self._keypoint_loss(
                pred, heatmaps_t[:, perm], keypoints_t[:, perm], vis_t[:, perm], p, image_size,
            )
            with torch.no_grad():
                use_rot = self._selection_score(terms_r) < self._selection_score(terms)
            rot_frac = use_rot.float().mean().item()
            terms = {
                name: tuple(torch.where(use_rot, r, i) for i, r in zip(terms[name], terms_r[name]))
                for name in terms
            }

        def batch_mean(name):
            s, n = terms[name]
            return s.sum() / n.sum().clamp_min(1.0)

        heatmap_loss = batch_mean("heatmap")
        offset_loss = batch_mean("offset")
        reproj_loss = batch_mean("reproj")
        visibility_loss = batch_mean("visibility")

        total = (
            self.seg_weight * seg_loss
            + self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.vis_weight * visibility_loss
            + self.reproj_weight * reproj_loss
        )

        components = {
            "seg_loss": seg_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
            "offset_loss": offset_loss.item(),
            "visibility_loss": visibility_loss.item(),
            "reproj_loss": reproj_loss.item(),
        }
        if rot_frac is not None:
            components["rot180_fraction"] = rot_frac
        return total, components
