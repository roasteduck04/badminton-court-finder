import cv2
import numpy as np


def generate_channels(image):
    """Generate 7-channel representation from a BGR image.

    Channels:
        0-2: RGB (normalized to [0,1])
        3:   Grayscale (normalized to [0,1])
        4:   CLAHE enhanced (normalized to [0,1])
        5:   Canny edge map (normalized to [0,1])
        6:   Court-color mask — HSV filter for green court surface (normalized to [0,1])

    Args:
        image: (H, W, 3) BGR uint8 image

    Returns:
        (H, W, 7) float32 array, all values in [0, 1]
    """
    h, w = image.shape[:2]
    result = np.zeros((h, w, 7), dtype=np.float32)

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

    # Ch 6: Court-color mask (green surface in HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 60])
    upper_green = np.array([90, 255, 255])
    court_mask = cv2.inRange(hsv, lower_green, upper_green)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    result[:, :, 6] = court_mask.astype(np.float32) / 255.0

    return result
