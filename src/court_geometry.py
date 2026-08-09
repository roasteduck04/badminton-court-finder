import numpy as np
import cv2

COURT_LENGTH = 13.4       # meters
COURT_WIDTH_DOUBLES = 6.1  # meters
COURT_WIDTH_SINGLES = 5.18 # meters
SHORT_SERVICE_LINE = 1.98  # meters from net
NET_POSITION = 6.7         # meters from each end (center)
LONG_SERVICE_LINE_DOUBLES = 0.76  # meters from back boundary

SINGLES_SIDELINE_OFFSET = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2  # 0.46m

# Left/right keypoint pairs whose *indices* must be swapped after a
# horizontal flip augmentation so that keypoint identity (e.g. "K0 = top
# left") is preserved. Albumentations mirrors the x-coordinate of each
# keypoint but keeps it at the same array index, so without this swap a
# flipped "top-left" keypoint would be mislabeled as "top-left" while
# actually sitting at the top-right of the frame.
#
# K8 (net/top) and K9 (net/bottom) lie on the court's center line and have
# no left/right counterpart, so they are intentionally excluded.
FLIP_PAIRS = [
    (0, 1),    # K0 top-left outer <-> K1 top-right outer
    (2, 3),    # K2 bottom-right outer <-> K3 bottom-left outer
    (4, 6),    # K4 left short service/top <-> K6 right short service/top
    (5, 7),    # K5 left short service/bottom <-> K7 right short service/bottom
    (10, 11),  # K10 left center service <-> K11 right center service
    (12, 13),  # K12 net/top singles <-> K13 net/bottom singles
]

KEYPOINT_NAMES = [
    "Top-left corner",       # K0
    "Top-right corner",      # K1
    "Bottom-right corner",   # K2
    "Bottom-left corner",    # K3
    "L short svc / top",     # K4
    "L short svc / bottom",  # K5
    "R short svc / top",     # K6
    "R short svc / bottom",  # K7
    "Net / top",             # K8
    "Net / bottom",          # K9
    "L center svc",          # K10
    "R center svc",          # K11
    "Net / top singles",     # K12
    "Net / bottom singles",  # K13
]

# 14 keypoints in real-world coordinates (meters)
# Origin at top-left corner (K0), x along length, y along width
COURT_KEYPOINTS_TEMPLATE = np.array([
    [0.0, 0.0],                                          # K0:  top-left outer
    [COURT_LENGTH, 0.0],                                 # K1:  top-right outer
    [COURT_LENGTH, COURT_WIDTH_DOUBLES],                  # K2:  bottom-right outer
    [0.0, COURT_WIDTH_DOUBLES],                           # K3:  bottom-left outer
    [NET_POSITION - SHORT_SERVICE_LINE, 0.0],             # K4:  left short service / top
    [NET_POSITION - SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES],  # K5:  left short service / bottom
    [NET_POSITION + SHORT_SERVICE_LINE, 0.0],             # K6:  right short service / top
    [NET_POSITION + SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES],  # K7:  right short service / bottom
    [NET_POSITION, 0.0],                                  # K8:  net / top
    [NET_POSITION, COURT_WIDTH_DOUBLES],                  # K9:  net / bottom
    [NET_POSITION - SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES / 2],  # K10: left short service / center
    [NET_POSITION + SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES / 2],  # K11: right short service / center
    [NET_POSITION, SINGLES_SIDELINE_OFFSET],              # K12: net / top singles
    [NET_POSITION, COURT_WIDTH_DOUBLES - SINGLES_SIDELINE_OFFSET],  # K13: net / bottom singles
], dtype=np.float64)


def get_court_lines():
    """Return list of (start_keypoint_idx, end_keypoint_idx) pairs defining all court lines."""
    return [
        # Outer boundary
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Short service lines
        (4, 5), (6, 7),
        # Net/center line
        (8, 9),
        # Center service line
        (10, 11),
        # Long service lines for doubles (not full lines, but segments)
        # For simplicity, we define the major structural lines:
        # Left half center line
        (4, 10), (10, 5),
        # Right half center line
        (6, 11), (11, 7),
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
        keypoints: (14, 2) array of normalized [0,1] keypoint coordinates
        visibility: list of 14 ints (1=visible, 0=not visible)
        width: mask width in pixels
        height: mask height in pixels
        line_thickness: line width in pixels

    Returns:
        (height, width) uint8 mask, 255 for line pixels, 0 elsewhere
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    kps = np.asarray(keypoints, dtype=np.float64).copy()

    # Convert normalized coords to pixel coords
    pixel_kps = np.zeros_like(kps)
    pixel_kps[:, 0] = kps[:, 0] * width
    pixel_kps[:, 1] = kps[:, 1] * height

    lines = get_court_lines()
    for start_idx, end_idx in lines:
        if start_idx == end_idx:
            continue
        if visibility[start_idx] and visibility[end_idx]:
            pt1 = tuple(pixel_kps[start_idx].astype(int))
            pt2 = tuple(pixel_kps[end_idx].astype(int))
            cv2.line(mask, pt1, pt2, 255, line_thickness)

    # Draw outer boundary if all 4 corners visible
    if all(visibility[i] for i in range(4)):
        for i in range(4):
            pt1 = tuple(pixel_kps[i].astype(int))
            pt2 = tuple(pixel_kps[(i + 1) % 4].astype(int))
            cv2.line(mask, pt1, pt2, 255, line_thickness)

    return mask
