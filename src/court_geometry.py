import numpy as np
import cv2

COURT_LENGTH = 13.4       # meters
COURT_WIDTH_DOUBLES = 6.1  # meters
COURT_WIDTH_SINGLES = 5.18 # meters
SHORT_SERVICE_LINE = 1.98  # meters from net
NET_POSITION = 6.7         # meters from each end (center)
LONG_SERVICE_LINE_DOUBLES = 0.76  # meters from back boundary

SINGLES_SIDELINE_OFFSET = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2  # 0.46m

NUM_KEYPOINTS = 31

_SS = NET_POSITION - SHORT_SERVICE_LINE   # 4.72m — left short service line
_RS = NET_POSITION + SHORT_SERVICE_LINE   # 8.68m — right short service line
_LS = LONG_SERVICE_LINE_DOUBLES           # 0.76m from back boundary
_RLS = COURT_LENGTH - _LS                 # 12.64m — right long service line
_SO = SINGLES_SIDELINE_OFFSET             # 0.46m
_CY = COURT_WIDTH_DOUBLES / 2            # 3.05m — center

# Horizontal-flip pairs: keypoints whose top/bottom identity swaps when
# the image is flipped horizontally (y=0 ↔ y=CW).
# Points on the center line (K10, K15, K20) have no pair.
FLIP_PAIRS = [
    (0, 3),    # Back-L top-dbl ↔ Back-L bot-dbl
    (1, 2),    # Back-L top-sgl ↔ Back-L bot-sgl
    (4, 7),    # Long-svc-L top-dbl ↔ Long-svc-L bot-dbl
    (5, 6),    # Long-svc-L top-sgl ↔ Long-svc-L bot-sgl
    (8, 12),   # Short-svc-L top-dbl ↔ Short-svc-L bot-dbl
    (9, 11),   # Short-svc-L top-sgl ↔ Short-svc-L bot-sgl
    (13, 17),  # Net top-dbl ↔ Net bot-dbl
    (14, 16),  # Net top-sgl ↔ Net bot-sgl
    (18, 22),  # Short-svc-R top-dbl ↔ Short-svc-R bot-dbl
    (19, 21),  # Short-svc-R top-sgl ↔ Short-svc-R bot-sgl
    (23, 26),  # Long-svc-R top-dbl ↔ Long-svc-R bot-dbl
    (24, 25),  # Long-svc-R top-sgl ↔ Long-svc-R bot-sgl
    (27, 30),  # Back-R top-dbl ↔ Back-R bot-dbl
    (28, 29),  # Back-R top-sgl ↔ Back-R bot-sgl
]

KEYPOINT_NAMES = [
    # Row 0: back boundary left (x=0)
    "Back-L / Top-dbl",       # K0
    "Back-L / Top-sgl",       # K1
    "Back-L / Bot-sgl",       # K2
    "Back-L / Bot-dbl",       # K3
    # Row 1: long service left (x=0.76)
    "LongSvc-L / Top-dbl",    # K4
    "LongSvc-L / Top-sgl",    # K5
    "LongSvc-L / Bot-sgl",    # K6
    "LongSvc-L / Bot-dbl",    # K7
    # Row 2: short service left (x=4.72)
    "ShortSvc-L / Top-dbl",   # K8
    "ShortSvc-L / Top-sgl",   # K9
    "ShortSvc-L / Center",    # K10
    "ShortSvc-L / Bot-sgl",   # K11
    "ShortSvc-L / Bot-dbl",   # K12
    # Row 3: net (x=6.7)
    "Net / Top-dbl",           # K13
    "Net / Top-sgl",           # K14
    "Net / Center",            # K15
    "Net / Bot-sgl",           # K16
    "Net / Bot-dbl",           # K17
    # Row 4: short service right (x=8.68)
    "ShortSvc-R / Top-dbl",   # K18
    "ShortSvc-R / Top-sgl",   # K19
    "ShortSvc-R / Center",    # K20
    "ShortSvc-R / Bot-sgl",   # K21
    "ShortSvc-R / Bot-dbl",   # K22
    # Row 5: long service right (x=12.64)
    "LongSvc-R / Top-dbl",    # K23
    "LongSvc-R / Top-sgl",    # K24
    "LongSvc-R / Bot-sgl",    # K25
    "LongSvc-R / Bot-dbl",    # K26
    # Row 6: back boundary right (x=13.4)
    "Back-R / Top-dbl",        # K27
    "Back-R / Top-sgl",        # K28
    "Back-R / Bot-sgl",        # K29
    "Back-R / Bot-dbl",        # K30
]

# 31 keypoints: every line intersection on a standard doubles court.
# Origin at top-left corner (K0), x along length, y along width.
COURT_KEYPOINTS_TEMPLATE = np.array([
    # Row 0: back boundary left (x=0)
    [0.0, 0.0],                                  # K0
    [0.0, _SO],                                   # K1
    [0.0, COURT_WIDTH_DOUBLES - _SO],             # K2
    [0.0, COURT_WIDTH_DOUBLES],                   # K3
    # Row 1: long service left (x=0.76)
    [_LS, 0.0],                                   # K4
    [_LS, _SO],                                   # K5
    [_LS, COURT_WIDTH_DOUBLES - _SO],             # K6
    [_LS, COURT_WIDTH_DOUBLES],                   # K7
    # Row 2: short service left (x=4.72)
    [_SS, 0.0],                                   # K8
    [_SS, _SO],                                   # K9
    [_SS, _CY],                                   # K10
    [_SS, COURT_WIDTH_DOUBLES - _SO],             # K11
    [_SS, COURT_WIDTH_DOUBLES],                   # K12
    # Row 3: net (x=6.7)
    [NET_POSITION, 0.0],                          # K13
    [NET_POSITION, _SO],                          # K14
    [NET_POSITION, _CY],                          # K15
    [NET_POSITION, COURT_WIDTH_DOUBLES - _SO],    # K16
    [NET_POSITION, COURT_WIDTH_DOUBLES],          # K17
    # Row 4: short service right (x=8.68)
    [_RS, 0.0],                                   # K18
    [_RS, _SO],                                   # K19
    [_RS, _CY],                                   # K20
    [_RS, COURT_WIDTH_DOUBLES - _SO],             # K21
    [_RS, COURT_WIDTH_DOUBLES],                   # K22
    # Row 5: long service right (x=12.64)
    [_RLS, 0.0],                                  # K23
    [_RLS, _SO],                                  # K24
    [_RLS, COURT_WIDTH_DOUBLES - _SO],            # K25
    [_RLS, COURT_WIDTH_DOUBLES],                  # K26
    # Row 6: back boundary right (x=13.4)
    [COURT_LENGTH, 0.0],                          # K27
    [COURT_LENGTH, _SO],                          # K28
    [COURT_LENGTH, COURT_WIDTH_DOUBLES - _SO],    # K29
    [COURT_LENGTH, COURT_WIDTH_DOUBLES],          # K30
], dtype=np.float64)


def get_court_lines():
    """Return list of (start_keypoint_idx, end_keypoint_idx) pairs defining all court lines.

    Each pair connects two keypoints that lie on the same painted court line.
    """
    return [
        # Top doubles sideline (y=0): K0 → K4 → K8 → K13 → K18 → K23 → K27
        (0, 4), (4, 8), (8, 13), (13, 18), (18, 23), (23, 27),
        # Bottom doubles sideline (y=6.1): K3 → K7 → K12 → K17 → K22 → K26 → K30
        (3, 7), (7, 12), (12, 17), (17, 22), (22, 26), (26, 30),
        # Top singles sideline (y=0.46): K1 → K5 → K9 → K14 → K19 → K24 → K28
        (1, 5), (5, 9), (9, 14), (14, 19), (19, 24), (24, 28),
        # Bottom singles sideline (y=5.64): K2 → K6 → K11 → K16 → K21 → K25 → K29
        (2, 6), (6, 11), (11, 16), (16, 21), (21, 25), (25, 29),
        # Center service line (y=3.05, between short service lines only): K10 → K15 → K20
        (10, 15), (15, 20),
        # Left back boundary (x=0): K0 → K1, K1 → K2, K2 → K3
        (0, 1), (1, 2), (2, 3),
        # Right back boundary (x=13.4): K27 → K28, K28 → K29, K29 → K30
        (27, 28), (28, 29), (29, 30),
        # Left long service (x=0.76): K4 → K5, K5 → K6, K6 → K7
        (4, 5), (5, 6), (6, 7),
        # Right long service (x=12.64): K23 → K24, K24 → K25, K25 → K26
        (23, 24), (24, 25), (25, 26),
        # Left short service (x=4.72): K8 → K9, K9 → K10, K10 → K11, K11 → K12
        (8, 9), (9, 10), (10, 11), (11, 12),
        # Right short service (x=8.68): K18 → K19, K19 → K20, K20 → K21, K21 → K22
        (18, 19), (19, 20), (20, 21), (21, 22),
        # Net (x=6.7): K13 → K14, K14 → K15, K15 → K16, K16 → K17
        (13, 14), (14, 15), (15, 16), (16, 17),
    ]


def compute_homography(src_points, dst_points):
    """Compute 3x3 homography from source to destination points.

    Args:
        src_points: (N, 2) array of source points, N >= 4
        dst_points: (N, 2) array of destination points, N >= 4

    Returns:
        3x3 homography matrix

    Raises:
        ValueError: If either input has < 4 points or points are degenerate (collinear).
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    # Validate input dimensions
    if src.shape[0] < 4:
        raise ValueError(f"src_points must have at least 4 points, got {src.shape[0]}")
    if dst.shape[0] < 4:
        raise ValueError(f"dst_points must have at least 4 points, got {dst.shape[0]}")

    H, _ = cv2.findHomography(src, dst, method=0)

    # Check if homography computation failed (degenerate points)
    if H is None:
        raise ValueError("Failed to compute homography: source and destination points are likely degenerate (collinear or duplicate)")

    return H


def project_points(H, points):
    """Apply homography to project points.

    Args:
        H: 3x3 homography matrix
        points: (N, 2) array of points

    Returns:
        (N, 2) array of projected points

    Raises:
        ValueError: If H is None.
    """
    if H is None:
        raise ValueError("Homography matrix H cannot be None")

    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 3)
    projected_h = (H @ pts_h.T).T   # (N, 3)
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    return projected


def validate_quadrilateral(corners):
    """Check if 4 corners form a valid convex quadrilateral.

    Args:
        corners: (4, 2) array of corner points in order

    Returns:
        True if valid convex quadrilateral
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        return False

    # Check convexity via cross products of consecutive edges
    n = len(corners)
    sign = None
    for i in range(n):
        p1 = corners[i]
        p2 = corners[(i + 1) % n]
        p3 = corners[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def generate_line_mask(keypoints, visibility, width=640, height=640, line_thickness=3):
    """Generate a binary mask of court lines from keypoint annotations.

    Args:
        keypoints: (N, 2) array of normalized [0,1] keypoint coordinates
        visibility: list of N ints (1=visible, 0=not visible)
        width: mask width in pixels
        height: mask height in pixels
        line_thickness: line width in pixels

    Returns:
        (height, width) uint8 mask, 255 for line pixels, 0 elsewhere
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    kps = np.asarray(keypoints, dtype=np.float64).copy()

    pixel_kps = np.zeros_like(kps)
    pixel_kps[:, 0] = kps[:, 0] * width
    pixel_kps[:, 1] = kps[:, 1] * height

    for start_idx, end_idx in get_court_lines():
        if visibility[start_idx] and visibility[end_idx]:
            pt1 = tuple(pixel_kps[start_idx].astype(int))
            pt2 = tuple(pixel_kps[end_idx].astype(int))
            cv2.line(mask, pt1, pt2, 255, line_thickness)

    return mask
