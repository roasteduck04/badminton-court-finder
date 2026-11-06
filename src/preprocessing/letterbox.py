"""Aspect-preserving resize with padding, and the matching keypoint maps.

Training and inference must use the same transform, so both go through
``letterbox`` here. Keypoints are normalized [0,1] in both spaces: relative to
the original (w, h) before, and to the square ``size`` canvas after.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float     # original pixels -> canvas pixels
    pad_x: int       # left padding in canvas pixels
    pad_y: int       # top padding in canvas pixels
    orig_w: int
    orig_h: int
    size: int        # square canvas side

    @property
    def content_w(self):
        return int(round(self.orig_w * self.scale))

    @property
    def content_h(self):
        return int(round(self.orig_h * self.scale))


def letterbox(image, size=640, pad_value=0):
    """Resize ``image`` (H, W[, C]) to fit a ``size`` x ``size`` canvas,
    keeping its aspect ratio, and centre it with constant padding.

    Returns:
        (canvas, LetterboxInfo)
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size) + image.shape[2:], pad_value, dtype=image.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, LetterboxInfo(scale, pad_x, pad_y, w, h, size)


def content_mask(info):
    """(size, size) uint8 mask, 1 inside the image content, 0 in padding."""
    mask = np.zeros((info.size, info.size), dtype=np.uint8)
    mask[info.pad_y:info.pad_y + info.content_h,
         info.pad_x:info.pad_x + info.content_w] = 1
    return mask


def keypoints_to_letterbox(keypoints, info):
    """Map (N, 2) normalized original-image coords to normalized canvas coords.

    Rows equal to [-1, -1] (unlabeled) are passed through unchanged.
    """
    kps = np.asarray(keypoints, dtype=np.float64)
    out = kps.copy()
    valid = ~np.all(kps == -1, axis=-1)
    out[valid, 0] = (kps[valid, 0] * info.orig_w * info.scale + info.pad_x) / info.size
    out[valid, 1] = (kps[valid, 1] * info.orig_h * info.scale + info.pad_y) / info.size
    return out


def keypoints_from_letterbox(keypoints, info):
    """Inverse of keypoints_to_letterbox: canvas coords -> original coords."""
    kps = np.asarray(keypoints, dtype=np.float64)
    out = kps.copy()
    valid = ~np.all(kps == -1, axis=-1)
    out[valid, 0] = (kps[valid, 0] * info.size - info.pad_x) / (info.orig_w * info.scale)
    out[valid, 1] = (kps[valid, 1] * info.size - info.pad_y) / (info.orig_h * info.scale)
    return out
