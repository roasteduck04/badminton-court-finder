import numpy as np
import cv2

COURT_LENGTH = 13.4       # meters
COURT_WIDTH_DOUBLES = 6.1  # meters
COURT_WIDTH_SINGLES = 5.18 # meters
SHORT_SERVICE_LINE = 1.98  # meters from net
NET_POSITION = 6.7         # meters from each end (center)
LONG_SERVICE_LINE_DOUBLES = 0.76  # meters from back boundary

SINGLES_SIDELINE_OFFSET = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2  # 0.46m

NUM_KEYPOINTS = 30

_SS = NET_POSITION - SHORT_SERVICE_LINE   # 4.72m
_RS = NET_POSITION + SHORT_SERVICE_LINE   # 8.68m
_LS = LONG_SERVICE_LINE_DOUBLES           # 0.76m
_RLS = COURT_LENGTH - _LS                 # 12.64m
_SO = SINGLES_SIDELINE_OFFSET             # 0.46m
_CY = COURT_WIDTH_DOUBLES / 2            # 3.05m
_SB = COURT_WIDTH_DOUBLES - _SO          # 5.64m

# Horizontal-flip pairs: left<->right across center column (y=0 <-> y=6.1).
# Center-column keypoints (K2, K7, K12, K17, K22, K27) have no pair.
FLIP_PAIRS = [
    (0, 4),    # Baseline-L dbl-top <-> dbl-bot
    (1, 3),    # Baseline-L sgl-top <-> sgl-bot
    (5, 9),    # LongSvc-L dbl-top <-> dbl-bot
    (6, 8),    # LongSvc-L sgl-top <-> sgl-bot
    (10, 14),  # ShortSvc-L dbl-top <-> dbl-bot
    (11, 13),  # ShortSvc-L sgl-top <-> sgl-bot
    (15, 19),  # ShortSvc-R dbl-top <-> dbl-bot
    (16, 18),  # ShortSvc-R sgl-top <-> sgl-bot
    (20, 24),  # LongSvc-R dbl-top <-> dbl-bot
    (21, 23),  # LongSvc-R sgl-top <-> sgl-bot
    (25, 29),  # Baseline-R dbl-top <-> dbl-bot
    (26, 28),  # Baseline-R sgl-top <-> sgl-bot
]

# Outer corners in cyclic order: TL -> TR -> BR -> BL
CORNER_INDICES = [0, 25, 29, 4]

KEYPOINT_NAMES = [
    # Row 0: Baseline L (x=0)
    "Baseline-L / Dbl-top",    # K0
    "Baseline-L / Sgl-top",    # K1
    "Baseline-L / Center",     # K2
    "Baseline-L / Sgl-bot",    # K3
    "Baseline-L / Dbl-bot",    # K4
    # Row 1: Long service L (x=0.76)
    "LongSvc-L / Dbl-top",    # K5
    "LongSvc-L / Sgl-top",    # K6
    "LongSvc-L / Center",     # K7
    "LongSvc-L / Sgl-bot",    # K8
    "LongSvc-L / Dbl-bot",    # K9
    # Row 2: Short service L (x=4.72)
    "ShortSvc-L / Dbl-top",   # K10
    "ShortSvc-L / Sgl-top",   # K11
    "ShortSvc-L / Center",    # K12
    "ShortSvc-L / Sgl-bot",   # K13
    "ShortSvc-L / Dbl-bot",   # K14
    # Row 3: Short service R (x=8.68)
    "ShortSvc-R / Dbl-top",   # K15
    "ShortSvc-R / Sgl-top",   # K16
    "ShortSvc-R / Center",    # K17
    "ShortSvc-R / Sgl-bot",   # K18
    "ShortSvc-R / Dbl-bot",   # K19
    # Row 4: Long service R (x=12.64)
    "LongSvc-R / Dbl-top",    # K20
    "LongSvc-R / Sgl-top",    # K21
    "LongSvc-R / Center",     # K22
    "LongSvc-R / Sgl-bot",    # K23
    "LongSvc-R / Dbl-bot",    # K24
    # Row 5: Baseline R (x=13.4)
    "Baseline-R / Dbl-top",    # K25
    "Baseline-R / Sgl-top",    # K26
    "Baseline-R / Center",     # K27
    "Baseline-R / Sgl-bot",    # K28
    "Baseline-R / Dbl-bot",    # K29
]

# 6 rows x 5 columns = 30 keypoints.
# Origin at K0 (top-left corner), x along length, y along width.
COURT_KEYPOINTS_TEMPLATE = np.array([
    # Row 0: Baseline L (x=0)
    [0.0, 0.0],                    # K0
    [0.0, _SO],                    # K1
    [0.0, _CY],                    # K2
    [0.0, _SB],                    # K3
    [0.0, COURT_WIDTH_DOUBLES],    # K4
    # Row 1: Long service L (x=0.76)
    [_LS, 0.0],                    # K5
    [_LS, _SO],                    # K6
    [_LS, _CY],                    # K7
    [_LS, _SB],                    # K8
    [_LS, COURT_WIDTH_DOUBLES],    # K9
    # Row 2: Short service L (x=4.72)
    [_SS, 0.0],                    # K10
    [_SS, _SO],                    # K11
    [_SS, _CY],                    # K12
    [_SS, _SB],                    # K13
    [_SS, COURT_WIDTH_DOUBLES],    # K14
    # Row 3: Short service R (x=8.68)
    [_RS, 0.0],                    # K15
    [_RS, _SO],                    # K16
    [_RS, _CY],                    # K17
    [_RS, _SB],                    # K18
    [_RS, COURT_WIDTH_DOUBLES],    # K19
    # Row 4: Long service R (x=12.64)
    [_RLS, 0.0],                   # K20
    [_RLS, _SO],                   # K21
    [_RLS, _CY],                   # K22
    [_RLS, _SB],                   # K23
    [_RLS, COURT_WIDTH_DOUBLES],   # K24
    # Row 5: Baseline R (x=13.4)
    [COURT_LENGTH, 0.0],                   # K25
    [COURT_LENGTH, _SO],                   # K26
    [COURT_LENGTH, _CY],                   # K27
    [COURT_LENGTH, _SB],                   # K28
    [COURT_LENGTH, COURT_WIDTH_DOUBLES],   # K29
], dtype=np.float64)


def get_court_lines():
    """Return (start_idx, end_idx) pairs for all court line segments."""
    return [
        # Doubles sideline top (y=0): K0->K5->K10->K15->K20->K25
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
        # Doubles sideline bot (y=6.1): K4->K9->K14->K19->K24->K29
        (4, 9), (9, 14), (14, 19), (19, 24), (24, 29),
        # Singles sideline top (y=0.46): K1->K6->K11->K16->K21->K26
        (1, 6), (6, 11), (11, 16), (16, 21), (21, 26),
        # Singles sideline bot (y=5.64): K3->K8->K13->K18->K23->K28
        (3, 8), (8, 13), (13, 18), (18, 23), (23, 28),
        # Center line L half (painted): K2->K7->K12
        (2, 7), (7, 12),
        # Center line R half (painted): K17->K22->K27
        (17, 22), (22, 27),
        # Center line across net (virtual, for homography): K12->K17
        (12, 17),
        # Baseline L: K0->K1->K2->K3->K4
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Baseline R: K25->K26->K27->K28->K29
        (25, 26), (26, 27), (27, 28), (28, 29),
        # Long service L: K5->K6->K7->K8->K9
        (5, 6), (6, 7), (7, 8), (8, 9),
        # Long service R: K20->K21->K22->K23->K24
        (20, 21), (21, 22), (22, 23), (23, 24),
        # Short service L: K10->K11->K12->K13->K14
        (10, 11), (11, 12), (12, 13), (13, 14),
        # Short service R: K15->K16->K17->K18->K19
        (15, 16), (16, 17), (17, 18), (18, 19),
    ]


def get_collinear_groups():
    """Return groups of keypoint indices that must be collinear.

    Each group is a list of indices lying on the same court line.
    6 rows + 5 columns = 11 groups.
    """
    return [
        # 6 rows (horizontal lines, same x)
        [0, 1, 2, 3, 4],           # Baseline L
        [5, 6, 7, 8, 9],           # Long service L
        [10, 11, 12, 13, 14],      # Short service L
        [15, 16, 17, 18, 19],      # Short service R
        [20, 21, 22, 23, 24],      # Long service R
        [25, 26, 27, 28, 29],      # Baseline R
        # 5 columns (vertical lines, same y)
        [0, 5, 10, 15, 20, 25],    # Doubles sideline top
        [1, 6, 11, 16, 21, 26],    # Singles sideline top
        [2, 7, 12, 17, 22, 27],    # Center line
        [3, 8, 13, 18, 23, 28],    # Singles sideline bot
        [4, 9, 14, 19, 24, 29],    # Doubles sideline bot
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
