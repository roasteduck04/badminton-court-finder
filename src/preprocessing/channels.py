import cv2
import numpy as np

NUM_CHANNELS = 7

# Pixels within this distance of the content boundary get no Canny response,
# so the edge between image content and letterbox/affine padding does not
# show up as a fake line.
_EDGE_MARGIN = 3


def generate_channels(image, valid=None):
    """Generate the 7-channel representation from a BGR uint8 image.

    Channels:
        0-2: RGB (normalized to [0,1])
        3:   Grayscale (normalized to [0,1])
        4:   CLAHE enhanced grayscale (normalized to [0,1])
        5:   Canny edge map (0 or 1)
        6:   Court-colour mask: HSV range for a GREEN court surface (0 or 1).
             This prior is green-specific; on blue/red/wooden courts it is
             mostly empty and the network has to rely on the other channels.
             Changing it would invalidate trained checkpoints, so it is left
             as is and documented here.

    Args:
        image: (H, W, 3) BGR uint8 image (normally the letterboxed canvas).
        valid: optional (H, W) content mask (nonzero = real image pixel,
            0 = letterbox/affine padding). When given, the derived channels
            (CLAHE, Canny, court colour) are zeroed in the padding, and Canny
            is also suppressed within a few pixels of the content boundary.
            RGB/gray are left as they are (padding is already black). Pass
            ``content_mask(info)`` at inference to match training.

    Returns:
        (H, W, 7) float32 array, all values in [0, 1]
    """
    h, w = image.shape[:2]
    result = np.zeros((h, w, NUM_CHANNELS), dtype=np.float32)

    # Ch 0-2: RGB normalized
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    result[:, :, :3] = rgb

    # Ch 3: Grayscale normalized
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result[:, :, 3] = gray.astype(np.float32) / 255.0

    # Ch 4: CLAHE enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    result[:, :, 4] = clahe_img.astype(np.float32) / 255.0

    # Ch 5: Canny edge map
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)
    result[:, :, 5] = edges.astype(np.float32) / 255.0

    # Ch 6: Court-colour mask (green surface in HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 60])
    upper_green = np.array([90, 255, 255])
    court_mask = cv2.inRange(hsv, lower_green, upper_green)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    result[:, :, 6] = court_mask.astype(np.float32) / 255.0

    if valid is not None:
        valid_u8 = (np.asarray(valid) > 0).astype(np.uint8)
        if valid_u8.shape != (h, w):
            raise ValueError(f"valid mask shape {valid_u8.shape} != image shape {(h, w)}")
        if not valid_u8.all():
            vf = valid_u8.astype(np.float32)
            k = 2 * _EDGE_MARGIN + 1
            # Erode with a border value of 1 so the canvas edge itself is not
            # treated as a content boundary.
            inner = cv2.erode(
                valid_u8, np.ones((k, k), np.uint8),
                borderType=cv2.BORDER_CONSTANT, borderValue=1,
            ).astype(np.float32)
            result[:, :, 4] *= vf
            result[:, :, 5] *= inner
            result[:, :, 6] *= vf

    return result
