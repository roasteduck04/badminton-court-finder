import json

import numpy as np
import pytest

from src.court_geometry import (
    CORNER_INDICES,
    COURT_KEYPOINTS_TEMPLATE,
    NUM_KEYPOINTS,
    SYMMETRY_PERMS,
    project_points,
)
from src.evaluation.metrics import (
    aggregate,
    bootstrap_ci,
    clip_convex,
    convex_polygon_iou,
    court_diagonal,
    court_iou,
    court_iou_homography,
    court_plane_errors,
    evaluate_image,
    fit_gt_homography,
    fit_pred_homography,
    keypoint_errors,
    mean_reprojection_error,
    pck_at_k,
    polygon_area,
    projected_court_polygon,
    render_line_mask,
    segmentation_iou,
    summarize_errors,
)

# A perspective view of the full court inside a 720 x 1280 image.
H_IMG = 720
W_IMG = 1280
SIZE = (H_IMG, W_IMG)
H_TRUE = np.array([[40.0, 5.0, 100.0], [2.0, -30.0, 500.0], [0.002, 0.01, 1.0]])


def _gt(vis=None):
    px = project_points(H_TRUE, COURT_KEYPOINTS_TEMPLATE)
    kps = px / np.array([W_IMG, H_IMG])
    v = np.ones(NUM_KEYPOINTS) if vis is None else np.asarray(vis, dtype=float)
    return kps, v


def _shift(kps, dx_px, dy_px=0.0, size=SIZE):
    return kps + np.array([dx_px / size[1], dy_px / size[0]])


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

class TestKeypointErrors:
    def test_identity_is_zero(self):
        kps, vis = _gt()
        err = keypoint_errors(kps, kps, vis, SIZE)
        assert err.shape == (NUM_KEYPOINTS,)
        assert np.allclose(err, 0)

    def test_known_offset_in_original_pixels(self):
        gt = np.full((NUM_KEYPOINTS, 2), 0.5)
        pred = gt.copy()
        pred[0] = [0.5 + 3 / 200, 0.5 + 4 / 100]  # w=200, h=100: 3 px, 4 px
        err = keypoint_errors(pred, gt, np.ones(NUM_KEYPOINTS), (100, 200))
        assert err[0] == pytest.approx(5.0)
        assert np.allclose(err[1:], 0)

    def test_invisible_is_nan(self):
        kps, vis = _gt()
        vis[3] = 0
        err = keypoint_errors(kps + 0.1, kps, vis, SIZE)
        assert np.isnan(err[3]) and np.isfinite(np.delete(err, 3)).all()

    def test_batched_with_per_sample_sizes(self):
        gt = np.full((2, NUM_KEYPOINTS, 2), 0.5)
        pred = gt + np.array([0.1, 0.0])
        err = keypoint_errors(pred, gt, np.ones((2, NUM_KEYPOINTS)), np.array([[100, 100], [100, 300]]))
        assert np.allclose(err[0], 10) and np.allclose(err[1], 30)

    def test_non_finite_prediction_is_inf(self):
        kps, vis = _gt()
        pred = kps.copy()
        pred[2] = np.nan
        err = keypoint_errors(pred, kps, vis, SIZE)
        assert np.isinf(err[2])

    def test_torch_input(self):
        torch = pytest.importorskip("torch")
        kps, vis = _gt()
        err = keypoint_errors(torch.tensor(kps), torch.tensor(kps), torch.tensor(vis), SIZE)
        assert np.allclose(err, 0)


class TestSummarizeErrors:
    def test_hand_computed(self):
        nan = np.nan
        E = np.array([[3, nan, 12], [nan, nan, nan], [1, 30, nan]], dtype=float)
        s = summarize_errors(E, thresholds=(5, 10, 20))
        assert s["n_points"] == 4
        assert s["n_images"] == 3
        assert s["n_images_no_visible"] == 1
        assert s["err_mean"] == pytest.approx(11.5)
        assert s["err_median"] == pytest.approx(7.5)
        assert s["img_err_mean"] == pytest.approx((7.5 + 15.5) / 2)
        assert s["pck@5"] == pytest.approx(0.5)
        assert s["pck@20"] == pytest.approx(0.75)
        assert s["img_pck@5"] == pytest.approx(0.5)
        assert s["img_pck@20"] == pytest.approx((1.0 + 0.5) / 2)

    def test_threshold_inclusive(self):
        s = summarize_errors(np.array([[10.0]]), thresholds=(10,))
        assert s["pck@10"] == 1.0

    def test_empty_is_nan(self):
        s = summarize_errors(np.full((2, 3), np.nan))
        assert np.isnan(s["pck@10"]) and np.isnan(s["err_mean"])
        assert s["n_images_no_visible"] == 2


# ---------------------------------------------------------------------------
# Per-image + aggregate
# ---------------------------------------------------------------------------

def _summary(rows, **kw):
    kw.setdefault("n_boot", 0)
    return aggregate(rows, **kw)


class TestPerfectPredictions:
    def test_all_scores_perfect(self):
        kps, vis = _gt()
        row = evaluate_image(kps, vis, kps, vis, SIZE)
        s = _summary([row])
        st = s["strict"]
        for key in ("pck@5px", "pck@10px", "pck@20px", "pck_norm@0.01", "pck_norm@0.02",
                    "pck_norm@0.05", "kla@0.05", "kla@0.1", "img_pck@10px",
                    "pcr_norm@0.01", "pcr@5px"):
            assert st[key] == 1.0, key
        assert st["auc_norm"] == pytest.approx(1.0)
        assert st["err_px_mean"] == pytest.approx(0, abs=1e-9)
        assert st["err_m_median"] == pytest.approx(0, abs=1e-6)
        assert s["homography"]["success_rate"] == 1.0
        assert s["homography"]["reproj_px_median"] == pytest.approx(0, abs=1e-4)
        assert s["court_iou"]["mean"] == pytest.approx(1.0, abs=1e-6)
        assert s["visibility"]["f1"] == 1.0
        assert row["sym_perm"] == "identity"
        assert row["court_diag_source"] == "homography"


class TestKnownOffsets:
    def test_reference_pixel_pck(self):
        # ref scale = 640 / 1280 = 0.5, so 16 original px = 8 reference px
        kps, vis = _gt()
        row = evaluate_image(_shift(kps, 16), vis, kps, vis, SIZE)
        st = _summary([row])["strict"]
        assert st["err_px_mean"] == pytest.approx(16)
        assert st["err_ref_mean"] == pytest.approx(8)
        assert st["pck@5px"] == 0.0
        assert st["pck@10px"] == 1.0

    def test_normalized_pck_uses_court_diagonal(self):
        kps, vis = _gt()
        px = kps * np.array([W_IMG, H_IMG])
        diag, src = court_diagonal(px, vis, SIZE, fit_gt_homography(px, vis))
        assert src == "homography"
        corners = project_points(H_TRUE, COURT_KEYPOINTS_TEMPLATE[CORNER_INDICES])
        ext = corners.max(0) - corners.min(0)
        assert diag == pytest.approx(np.hypot(*ext))
        row = evaluate_image(_shift(kps, 0.015 * diag), vis, kps, vis, SIZE)
        st = _summary([row])["strict"]
        assert st["pck_norm@0.01"] == 0.0
        assert st["pck_norm@0.02"] == 1.0
        # error 0.015 of the diagonal -> curve is 1 for a >= 0.015
        assert st["auc_norm"] == pytest.approx(1 - 0.015 / 0.1, abs=0.02)

    def test_kla_uses_image_diagonal(self):
        kps, vis = _gt()
        d_img = np.hypot(H_IMG, W_IMG)
        row = evaluate_image(_shift(kps, 0, 0.07 * d_img), vis, kps, vis, SIZE)
        st = _summary([row])["strict"]
        assert st["kla@0.05"] == 0.0
        assert st["kla@0.1"] == 1.0

    def test_court_metres(self):
        kps, vis = _gt()
        pred = kps.copy()
        # K0 predicted where the template point 1 m along the length projects
        pred[0] = project_points(H_TRUE, np.array([[1.0, 0.0]]))[0] / np.array([W_IMG, H_IMG])
        row = evaluate_image(pred, vis, kps, vis, SIZE)
        assert row["err_m"][0] == pytest.approx(1.0, abs=1e-6)
        assert row["err_m"][1] == pytest.approx(0.0, abs=1e-6)

    def test_beyond_horizon_is_inf(self):
        kps, vis = _gt()
        px = kps * np.array([W_IMG, H_IMG])
        H = fit_gt_homography(px, vis)
        line = np.linalg.inv(H)[2]  # vanishing line in the image
        p0 = np.append(px[0], 1.0)
        reflected = px[0] - 2 * (line @ p0) / (line[:2] @ line[:2]) * line[:2]
        pred_px = px.copy()
        pred_px[0] = reflected
        e = court_plane_errors(pred_px, px, vis, H)
        assert np.isinf(e[0]) and np.allclose(e[1:], 0, atol=1e-6)
        row = evaluate_image(pred_px / np.array([W_IMG, H_IMG]), vis, kps, vis, SIZE)
        st = _summary([row])["strict"]
        assert st["n_beyond_horizon"] == 1
        assert np.isfinite(st["err_m_mean"])


class TestSymmetry:
    def test_rot180_relabel_detected(self):
        kps, vis = _gt()
        pred = kps[SYMMETRY_PERMS["rot180"]]
        row = evaluate_image(pred, vis, kps, vis, SIZE)
        assert row["sym_perm"] == "rot180"
        s = _summary([row])
        assert s["strict"]["pck@10px"] < 0.1
        assert s["sym"]["pck@10px"] == 1.0
        assert s["sym"]["auc_norm"] == pytest.approx(1.0)
        # IoU / homography are label-invariant for a court-symmetric relabel
        assert s["court_iou"]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_col_flip_is_not_forgiven(self):
        kps, vis = _gt()
        row = evaluate_image(kps[SYMMETRY_PERMS["col_flip"]], vis, kps, vis, SIZE)
        assert row["sym_perm"] in ("identity", "rot180")
        assert _summary([row])["sym"]["pck@10px"] < 1.0


class TestExclusion:
    def test_invisible_points_excluded(self):
        kps, vis = _gt()
        vis[[1, 2, 3, 27]] = 0
        pred = kps.copy()
        pred[[1, 2, 3, 27]] += 0.3  # wrong but GT-invisible
        row = evaluate_image(pred, vis, kps, vis, SIZE)
        st = _summary([row])["strict"]
        assert st["n_points"] == 26
        assert st["pck@5px"] == 1.0

    def test_zero_visible_images_excluded_from_per_image(self):
        kps, vis = _gt()
        good = evaluate_image(kps, vis, kps, vis, SIZE)
        empty = evaluate_image(kps + 0.2, vis, kps, np.zeros(NUM_KEYPOINTS), SIZE)
        s = _summary([good, empty])
        st = s["strict"]
        assert st["n_images_no_visible"] == 1
        assert st["n_images_with_visible"] == 1
        assert st["img_pck@10px"] == 1.0
        assert st["pck@10px"] == 1.0
        assert s["homography"]["n_attempted"] == 1
        assert s["court_iou"]["n"] == 1

    def test_few_visible_fall_back_to_bbox(self):
        kps, vis = _gt()
        vis[:] = 0
        vis[[0, 4, 12]] = 1
        px = kps * np.array([W_IMG, H_IMG])
        assert fit_gt_homography(px, vis) is None
        diag, src = court_diagonal(px, vis, SIZE)
        assert src == "bbox"
        p = px[[0, 4, 12]]
        assert diag == pytest.approx(np.hypot(*(p.max(0) - p.min(0))))

    def test_single_point_excluded_from_normalized(self):
        kps, vis = _gt()
        vis[:] = 0
        vis[7] = 1
        row = evaluate_image(kps, vis, kps, vis, SIZE)
        st = _summary([row])["strict"]
        assert st["n_points"] == 1 and st["n_points_norm"] == 0
        assert np.isnan(st["pck_norm@0.02"])
        assert st["pck@5px"] == 1.0


class TestHomographyFromPredictions:
    def test_uses_predicted_visibility_only(self):
        kps, vis = _gt()
        row = evaluate_image(kps, np.zeros(NUM_KEYPOINTS), kps, vis, SIZE)
        assert row["h_success"] is False
        s = _summary([row])
        assert s["homography"]["success_rate"] == 0.0
        assert s["court_iou"]["mean"] == 0.0
        assert s["court_iou"]["n"] == 1

    def test_ransac_rejects_outlier(self):
        kps, vis = _gt()
        pred = kps.copy()
        pred[12] += [0.2, 0.1]
        px = pred * np.array([W_IMG, H_IMG])
        H, inl = fit_pred_homography(px, vis, SIZE)
        assert H is not None and not inl[12] and inl.sum() == 29
        row = evaluate_image(pred, vis, kps, vis, SIZE)
        assert row["reproj_px"][12] == pytest.approx(0, abs=1e-3)
        assert row["h_inliers"] == 29

    def test_collinear_predictions_fail(self):
        kps, vis = _gt()
        pvis = np.zeros(NUM_KEYPOINTS)
        pvis[[0, 1, 2, 3, 4]] = 0.9  # one line only
        H, inl = fit_pred_homography(kps * np.array([W_IMG, H_IMG]), pvis, SIZE)
        assert H is None and inl is None


class TestCourtIou:
    def test_identical_homographies(self):
        assert court_iou_homography(H_TRUE, H_TRUE, SIZE) == pytest.approx(1.0)

    def test_hand_computed_shift(self):
        # court -> axis-aligned 134 x 61 rectangle; shift by half its width
        H1 = np.diag([10.0, 10.0, 1.0])
        H2 = H1.copy()
        H2[0, 2] = 67.0
        assert court_iou_homography(H1, H2, (1000, 1000)) == pytest.approx(1 / 3)

    def test_clipped_to_image(self):
        H1 = np.diag([10.0, 10.0, 1.0])
        poly = projected_court_polygon(H1, (30, 50))
        assert polygon_area(poly) == pytest.approx(30 * 50)
        H2 = H1.copy()
        H2[0, 2] = 20.0  # same inside the 30x50 window except a 20 px strip
        assert court_iou_homography(H1, H2, (30, 50)) == pytest.approx(30 / 50)

    def test_behind_camera_part_is_clipped(self):
        # w = 1 - 0.1 x: the far half of the court (x > 10 m) is behind the camera
        H = np.array([[10.0, 0, 0], [0, 10.0, 0], [-0.1, 0, 1.0]])
        poly = projected_court_polygon(H, (10_000, 10_000))
        assert len(poly) >= 3 and np.all(np.isfinite(poly))
        assert court_iou_homography(H, H, (10_000, 10_000)) == pytest.approx(1.0)

    def test_no_gt_homography_is_nan(self):
        assert np.isnan(court_iou_homography(None, H_TRUE, SIZE))

    def test_convex_polygon_helpers(self):
        a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        b = a + [1, 1]
        assert polygon_area(clip_convex(a, b)) == pytest.approx(1.0)
        assert convex_polygon_iou(a, b) == pytest.approx(1 / 7)
        assert convex_polygon_iou(a, a[::-1]) == pytest.approx(1.0)


class TestPcr:
    def test_one_corner_wrong_fails_image(self):
        kps, vis = _gt()
        a = kps.copy()
        a[CORNER_INDICES[0]] = _shift(a[CORNER_INDICES[0]], 100)
        rows = [evaluate_image(a, vis, kps, vis, SIZE), evaluate_image(kps, vis, kps, vis, SIZE)]
        st = _summary(rows)["strict"]
        assert st["pcr_norm@0.05"] == pytest.approx(0.5)
        assert st["pcr_n_images"] == 2

    def test_invisible_corner_ignored(self):
        kps, vis = _gt()
        vis[CORNER_INDICES[0]] = 0
        a = kps.copy()
        a[CORNER_INDICES[0]] += 0.3
        st = _summary([evaluate_image(a, vis, kps, vis, SIZE)])["strict"]
        assert st["pcr@5px"] == 1.0


class TestVisibility:
    def test_counts(self):
        kps, vis = _gt()
        vis[:10] = 0  # 20 positives
        pvis = np.zeros(NUM_KEYPOINTS)
        pvis[5:25] = 0.9  # predicted 20: TP = 15 (10..24), FP = 5 (5..9), FN = 5 (25..29)
        row = evaluate_image(kps, pvis, kps, vis, SIZE)
        v = _summary([row])["visibility"]
        assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (15, 5, 5, 5)
        assert v["precision"] == pytest.approx(0.75)
        assert v["recall"] == pytest.approx(0.75)
        assert v["f1"] == pytest.approx(0.75)
        assert v["accuracy"] == pytest.approx(20 / 30)


class TestSegmentation:
    def test_net_segment_excluded(self):
        kps, _ = _gt()
        vis = np.zeros(NUM_KEYPOINTS)
        vis[[12, 17]] = 1
        assert render_line_mask(kps, vis, 64, 3).sum() == 0
        vis[7] = 1
        assert render_line_mask(kps, vis, 64, 3).sum() > 0

    def test_thickness_matters(self):
        kps, vis = _gt()
        assert render_line_mask(kps, vis, 128, 5).sum() > render_line_mask(kps, vis, 128, 1).sum()

    def test_identical_and_empty(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 1.0
        assert segmentation_iou(mask, mask) == 1.0
        empty = np.zeros((64, 64), dtype=np.float32)
        assert segmentation_iou(empty, empty) == 0.0
        assert np.isnan(segmentation_iou(empty, empty, empty_value=float("nan")))


class TestBootstrapAndBreakdowns:
    def _rows(self):
        kps, vis = _gt()
        rows = []
        for i, dx in enumerate([0, 4, 30, 60, 0, 12]):
            rows.append(evaluate_image(_shift(kps, dx), vis, kps, vis, SIZE,
                                       source=["manual", "roboflow"][i % 2], group=f"g{i % 3}"))
        return rows

    def test_ci_deterministic_with_seed(self):
        rows = self._rows()
        a = aggregate(rows, n_boot=50, seed=3)["ci"]
        b = aggregate(rows, n_boot=50, seed=3)["ci"]
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        lo, hi = a["pck@10px"]
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_degenerate_when_all_equal(self):
        kps, vis = _gt()
        rows = [evaluate_image(kps, vis, kps, vis, SIZE) for _ in range(4)]
        assert aggregate(rows, n_boot=20)["ci"]["pck@10px"] == [1.0, 1.0]

    def test_bootstrap_ci_generic(self):
        vals = np.arange(10, dtype=float)
        ci1 = bootstrap_ci(10, lambda idx: float(vals[idx].mean()), n_boot=200, seed=1)
        ci2 = bootstrap_ci(10, lambda idx: float(vals[idx].mean()), n_boot=200, seed=1)
        assert ci1 == ci2 and ci1[0] < 4.5 < ci1[1]

    def test_breakdowns(self):
        s = aggregate(self._rows(), n_boot=0)
        assert set(s["by_source"]) == {"manual", "roboflow"}
        assert s["by_source"]["manual"]["n_images"] == 3
        assert set(s["by_group"]) == {"g0", "g1", "g2"}
        assert len(s["by_keypoint"]) == NUM_KEYPOINTS
        assert s["by_keypoint"][0]["n"] == 6
        assert s["by_n_visible"]["23-30"]["n_images"] == 6
        assert s["by_n_visible"]["0"]["n_images"] == 0
        # ref px = orig px / 2 -> [0, 2, 15, 30, 0, 6]; three are <= 5
        assert s["strict"]["pck@5px"] == pytest.approx(3 / 6)


# ---------------------------------------------------------------------------
# Legacy API (kept importable for train.py)
# ---------------------------------------------------------------------------

class TestLegacy:
    def test_pck_at_k(self):
        gt = np.full((NUM_KEYPOINTS, 2), 0.5, dtype=np.float32)
        pred = gt.copy()
        pred[0] = [0.5 + 15.0 / 640, 0.5]
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        assert pck_at_k(pred, gt, vis, k=10)[1] < 1.0
        assert pck_at_k(pred, gt, vis, k=20)[1] == 1.0
        assert pck_at_k(pred, gt, np.zeros(NUM_KEYPOINTS), k=5)[1] == 0.0

    def test_mean_reprojection_error(self):
        image_w, image_h = 200, 100
        H = np.diag([image_w / 13.4, image_h / 6.1, 1.0])
        kps = project_points(H, COURT_KEYPOINTS_TEMPLATE) / np.array([image_w, image_h])
        vis = np.ones(NUM_KEYPOINTS)
        assert mean_reprojection_error(kps, kps, vis, image_w, image_h) < 1.0
        vis1 = np.zeros(NUM_KEYPOINTS)
        vis1[0] = 1
        assert mean_reprojection_error(kps, kps, vis1, 640, 640) is None

    def test_court_iou(self):
        a = np.zeros((NUM_KEYPOINTS, 2))
        a[0], a[4], a[25], a[29] = [0, 0], [0, 1], [1, 0], [1, 1]
        b = np.zeros((NUM_KEYPOINTS, 2))
        b[0], b[4], b[25], b[29] = [0.5, 0], [0.5, 1], [1.5, 0], [1.5, 1]
        vis = np.zeros(NUM_KEYPOINTS)
        vis[CORNER_INDICES] = 1
        assert court_iou(a, a, vis, vis) == pytest.approx(1.0)
        assert court_iou(a, b, vis, vis) == pytest.approx(1 / 3)
        one = np.zeros(NUM_KEYPOINTS)
        one[0] = 1
        assert court_iou(a, a, one, one) == 0.0
