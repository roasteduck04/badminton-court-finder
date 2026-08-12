import numpy as np
import pytest
from src.evaluation.metrics import (
    pck_at_k,
    mean_reprojection_error,
    court_iou,
    segmentation_iou,
)
from src.court_geometry import NUM_KEYPOINTS, COURT_KEYPOINTS_TEMPLATE, CORNER_INDICES


class TestPckAtK:
    def test_perfect_prediction(self):
        kps = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        per_kp, mean_acc = pck_at_k(kps, kps, vis, k=5, image_size=640)
        assert mean_acc == 1.0
        assert per_kp.all()

    def test_all_invisible_returns_zero(self):
        pred = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        gt = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        per_kp, mean_acc = pck_at_k(pred, gt, vis, k=5, image_size=640)
        assert mean_acc == 0.0

    def test_threshold_discriminates(self):
        gt = np.full((NUM_KEYPOINTS, 2), 0.5, dtype=np.float32)
        pred = gt.copy()
        pred[0] = [0.5 + 15.0 / 640, 0.5]  # 15px off
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        _, acc_at_10 = pck_at_k(pred, gt, vis, k=10, image_size=640)
        _, acc_at_20 = pck_at_k(pred, gt, vis, k=20, image_size=640)
        assert acc_at_10 < 1.0  # K0 outside 10px
        assert acc_at_20 == 1.0  # K0 inside 20px


class TestMeanReprojectionError:
    def test_perfect_homography_gives_zero_error(self):
        image_w, image_h = 200, 100
        sx, sy = image_w / 13.4, image_h / 6.1
        H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        from src.court_geometry import project_points
        projected_px = project_points(H, COURT_KEYPOINTS_TEMPLATE)
        kps_norm = projected_px / np.array([image_w, image_h])
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        mre = mean_reprojection_error(kps_norm, kps_norm, vis, image_w, image_h)
        assert mre is not None
        assert mre < 1.0  # sub-pixel

    def test_too_few_visible_returns_none(self):
        pred = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        gt = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        vis[0] = 1.0  # only 1 visible
        mre = mean_reprojection_error(pred, gt, vis, 640, 640)
        assert mre is None


class TestCourtIou:
    def test_identical_corners_gives_one(self):
        kps = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        kps[0] = [0.1, 0.1]   # K0 TL
        kps[4] = [0.1, 0.9]   # K4 BL
        kps[25] = [0.9, 0.1]  # K25 TR
        kps[29] = [0.9, 0.9]  # K29 BR
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        for i in CORNER_INDICES:
            vis[i] = 1.0
        iou = court_iou(kps, kps, vis, vis)
        assert abs(iou - 1.0) < 1e-6

    def test_missing_corner_gives_zero(self):
        kps = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        vis[0] = 1.0  # only 1 corner
        iou = court_iou(kps, kps, vis, vis)
        assert iou == 0.0

    def test_partial_overlap(self):
        kps_a = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        kps_a[0] = [0.0, 0.0]; kps_a[4] = [0.0, 1.0]
        kps_a[25] = [1.0, 0.0]; kps_a[29] = [1.0, 1.0]
        kps_b = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        kps_b[0] = [0.5, 0.0]; kps_b[4] = [0.5, 1.0]
        kps_b[25] = [1.5, 0.0]; kps_b[29] = [1.5, 1.0]
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        for i in CORNER_INDICES:
            vis[i] = 1.0
        iou = court_iou(kps_a, kps_b, vis, vis)
        assert 0.3 < iou < 0.4  # 50% overlap on unit square -> IoU = 1/3


class TestSegmentationIou:
    def test_identical_masks(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 1.0
        assert segmentation_iou(mask, mask) == 1.0

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[:32, :] = 1.0
        gt = np.zeros((64, 64), dtype=np.float32)
        gt[32:, :] = 1.0
        assert segmentation_iou(pred, gt) == 0.0

    def test_empty_masks(self):
        empty = np.zeros((64, 64), dtype=np.float32)
        assert segmentation_iou(empty, empty) == 0.0

    def test_threshold_applied(self):
        pred = np.full((64, 64), 0.4, dtype=np.float32)  # all below 0.5
        gt = np.ones((64, 64), dtype=np.float32)
        assert segmentation_iou(pred, gt) == 0.0
