# CourtVisionNet improvement plan (2026-11-01)

This plan comes from a literature review of sports-field registration, court detection and keypoint estimation, plus measurements on our own data. Citations marked [U] could not be verified at the source. Every other paper and number below was checked on its arXiv, CVF, publisher or GitHub page.

## Where we stand

- **Model:** ResNet-50 + FPN, 7-channel input, heatmap keypoints with DARK decoding, a line-segmentation head (used only as an extra training signal), and a MAGSAC homography fit.
- **No trustworthy accuracy number yet:**
  - The rewritten pipeline has not been trained on GPU.
  - The old Run 7 model scores 26% / 24% PCK@10px on `test_real` / `test_roboflow`, but it was trained on the old labels and saw these videos.
  - Copying the labels of the most similar training image scores 0.4% / 1.2%.
  - The classical line-fitting baseline (B2, no learning) scores 37.9% / 73.9% with its val-chosen failure threshold, and 44.1% / 98.1% when every fit is accepted (`docs/classical-baseline.md`).
- **Real data is the bottleneck:**
  - Real training images come from about 9 distinct scenes: 2 Roboflow videos and a few phone sessions in one hall.
  - The test sets are one hall plus 4 web photos (`test_real`) and 2 broadcast events (`test_roboflow`).
- **Synthetic camera framing is now fixed (this change):**
  - Before, 40% of renders showed the court covering under 10% of the image. Real images never do.
  - The new per-image auto-FOV in `src/tools/camera_framing.py` brings the median coverage to 0.27 against 0.22–0.50 in real images, with no tiny-court renders. It also produces realistic phone and broadcast focal lengths and mostly-portrait phone framing.

## Guiding rules

1. Measure before changing anything. Every change is an ablation with ≥3 seeds, reported on `test_real` and `test_roboflow` separately, next to the nearest-neighbour and classical baselines.
2. Prefer changes that don't need retraining (geometry at inference) and changes reviewers expect (baselines, metrics, a diverse test set) over model tinkering.
3. Keep `canonicalize` as the only labeling rule: any new data source, augmentation or baseline passes its output through it.

## Track A: evidence and evaluation (do first)

| # | Item | Effort | Why |
|---|---|---|---|
| A1 | **Train the baseline**: `configs/baseline.json`, 3 seeds on RunPod (`docs/runpod.md`). Re-render synthetic data with the new framing first (Track E1). | S + compute | No result exists without it |
| A2 | **Diverse held-out test set**: 150–300 frames from 20+ unseen halls (YouTube amateur matches, club videos, phone photos), labelled at all 30 points with the annotator. Keep entire venues out of training. | M (labelling) | The strongest single fix for credibility. No public badminton set has inner-line keypoints from phone or oblique views, so releasing ours is itself a contribution. |
| A3 | **Cross-dataset check**: evaluate our 4 corner keypoints on the CourtKeyNet badminton data (CC BY 4.0, outer corners only; Raj & G., *ML with Applications* 2026). Optionally test transfer to tennis with TennisCourtDetector's 8,841 frames. | S | Cheap extra test diversity |
| A4 | **Report what reviewers expect**: IoU_whole of the warped template, court-plane error in cm (mean / median / p90), completeness (share of images that get a homography), PCK at several px thresholds, AUC, broken down by view type (end-on, side, high, phone). Most of these are already in `metrics.py`; add the view-type breakdown and make IoU_whole and completeness the headline numbers. | S | Matches SoccerNet, WC14 and TS-WorldCup practice (Magera et al., CVPRW 2024 benchmark protocol) |

## Track B: baselines

| # | Item | Effort |
|---|---|---|
| B1 | Nearest-neighbour label copy (done: `scripts/nn_baseline.py`) | – |
| B2 | Classical line-fitting baseline (done: `scripts/classical_baseline.py`, results in `docs/classical-baseline.md`; recipe below) | – |
| B3 | A field-standard learned baseline: HRNet-W32/48 heatmaps + DLT, as used by the top SoccerNet calibration entries. This is the D1 backbone swap, so it doubles as a baseline. | M |

### B2 recipe: classical court detection (Farin et al. 2004 + MonoTrack 2022)

Sources:
- Farin, Krabbe, de With & Effelsberg, *Robust Camera Calibration for Sport Videos using Court Models*, SPIE 5307, 2004.
- Liu & Wang, *MonoTrack*, CVPRW 2022, which applies this to badminton.
- Parameters from the open implementation github.com/gchlebus/tennis-court-detection.

1. **Line pixels:** luminance ≥ 80 and more than 20 brighter than both neighbours τ px away, horizontally or vertically. Use τ ≈ line width in px, scaled with resolution. For yellow or coloured lines, use the chroma distance to the floor colour instead of luminance.
2. **Texture filter:** structure-tensor eigenvalue ratio > 4 (Sobel 3, 41×41 window). This removes bright textured areas such as spectators and banners.
3. **Line candidates:** Hough transform, then about 5 rounds of least-squares refit on line pixels within 8 px.
4. **Two families:** split the lines into two direction groups with MonoTrack's max-weight bipartite partition, weight (|θ − π/2| + ε)⁻². This replaces Farin's hard angle bins, which fail on off-centre views (Farin failed on about 24% of MonoTrack's videos).
5. **Hypotheses:** each pair of image lines per family × each pair of model lines per family. Our model has 6 transverse × 5 longitudinal lines, i.e. our 6×5 keypoint grid. Compute H from the 4 intersections.
6. **Score:** walk each projected model segment: +1 per line pixel, −0.5 per non-line pixel. Reject degenerate H (projected sides < 30 px, implausible geometry).
7. **Refine:** least squares of line pixels to the projected model lines, then project the 30 keypoints and `canonicalize`.
8. **Expected failures** (worth reporting): adjacent courts and other sports' lines create wrong hypotheses that still score well; also partial views, specular floors, and thin far lines.

MonoTrack reports this kind of improved Farin pipeline raising success (IoU > 0.8) from 73.9% to 85.5% on broadcast badminton. Magera et al. (CVPRW 2025) found geometric methods hold up better out of domain than learned ones. That is a reason to keep this pipeline as a fallback, not just as a baseline.

## Track C: geometry at inference (no retraining; best value)

| # | Item | Effort | Evidence |
|---|---|---|---|
| C1 | **Line-aware homography refinement.** After MAGSAC, minimise keypoint reprojection plus the distance of line pixels from our segmentation head to the projected model lines, with outlier gating. | M | PnLCalib (Gutiérrez-Pérez & Agudo, arXiv 2404.08401) gains IoU_whole 92.4 → 93.4 on WC14 from its point-and-line term. On TennisCourtDetector, line refinement + homography cut median error from 2.83 to 1.83 px. Our segmentation head already predicts lines and nothing uses them at inference. |
| C2 | **Multi-hypothesis fit + verification.** Fit H on several keypoint subsets and RANSAC thresholds, and score each by how well the projected template agrees with the segmentation mask. Low scores mean reject, which gives a principled completeness number. | S | Every top SoccerNet calibration entry does this (2022 Baidu, 2023 Sportlight, NBJW) |
| C3 | **Uncertainty-weighted fit.** Weight keypoints in MAGSAC/DLT by heatmap peak sharpness (or an RLE-style σ). | S | Cheap; fewer bad inliers |
| C4 | **Radial distortion.** Estimate k1 inside C1's refinement, as TVCalib (Theiner & Ewerth, WACV 2023) does. Phone ultra-wide lenses bend lines; NBJW/PnLCalib list fisheye shots as a failure case. | M | Our synthetic augmentation already produces distorted training images |
| C5 | **Video:** Kalman / optical-flow homography tracking (BHITK, Claasen & de Villiers 2024; SoccerNet 2025 entries) | S–M | Only if the paper claims video use |

## Track D: model and training (after A1 gives a reference number)

| # | Item | Effort | Notes |
|---|---|---|---|
| D1 | **HRNet-W32/48 backbone** | M | Standard in field registration (NBJW, PnLCalib, Sportlight); high-resolution features suit precise intersections |
| D2 | **DINOv2 ViT-S/B backbone** with a high-resolution decoder | M | No sports-registration paper uses it yet, so it's a publishable ablation. El Banani et al. (CVPR 2024) found DINOv2 strongest among frozen models for geometry. Plausibly helps with only ~9 scenes. Re-check whether the 7-channel input still helps. |
| D3 | **Line heatmap head** (two endpoint peaks per court line), as in NBJW/Sportlight | M | Feeds C1 directly, and lines remain visible when intersections are occluded |
| D4 | **Occlusion copy-paste**: paste players, nets, posts and bags from real photos as occluders | S | Copy-paste (Ghiasi, CVPR 2021). KaliCalib (480 training images, close to our regime) cut error 18–30% with basic augmentation alone. |
| D5 | **Synthetic protocol**: synthetic pretrain → real fine-tune vs mixed batches (1:1, 3:1) | M (compute) | The literature conflicts (Nowruzi 2019 vs a 2025 replication), so the paper needs its own result |
| D6 | SimCC head (RTMPose) as an alternative to heatmaps + DARK | S–M | Lower priority |

## Track E: data

| # | Item | Effort | Notes |
|---|---|---|---|
| E1 | **Re-render** 2000 images with the auto-FOV generator, augment them, and rebuild `colab_data_v2` (commands in the handover) | S (about 1–2 h locally) | Replaces the old renders, which had the default cube and no net |
| E2 | **Semi-supervised learning on unlabeled badminton video**: homography-equivariance loss (warp image and predictions with a random homography; they must agree, then re-canonicalise) plus mean-teacher pseudo-labels filtered by geometric consistency (MAGSAC inlier ratio, residual, segmentation agreement) | M–L | No published self-training for field registration, so this is a novelty opportunity. Honari et al. (CVPR 2018) beat the previous state of the art on AFLW with 5% of images labelled. Unlabeled amateur video is plentiful. |
| E3 | **Generator realism, round 2**: community-hall lighting (windows, overhead tubes, bright walls) vs BWF event lighting (dark surrounds, side-mounted lights); homography-conditioning reject filter | M | From the camera-prior and venue research (BWF facility specs) |
| E4 | ControlNet photorealism of renders, conditioned on line mask + depth at low strength, with an automatic line-IoU check | L | **Stretch only**: a 2026 paper reports ControlNet unstable on thin structures (guidewires); court lines are 40 mm |

## Suggested order

1. **Week 1:**
   - E1: re-render and rebuild the dataset.
   - A1: baseline × 3 seeds on RunPod.
   - B2: classical baseline (runs in parallel; no GPU needed).
   - Start A2 labelling.
2. **Week 2:**
   - C1 + C2 on the trained baseline, with no retraining, and measure the gain.
   - A4 reporting.
   - Ablations already in `configs/` (rgb_only, no_synthetic, no_reproj, mse_heatmap, no_symmetric, sigma6).
3. **Week 3+:**
   - D1 / D2 backbones.
   - D4 occlusion copy-paste.
   - E2 semi-supervised learning, if A2 shows a large cross-venue gap.

## What the paper could claim

- A canonical labeling scheme for courts that are symmetric under mirroring and 180° rotation, plus a leak-free split protocol. We showed the earlier result was leakage: label copying reached 95% on the old test set.
- A badminton dataset with all 30 line intersections from phone and oblique views, the first we found.
- A method: keypoints + line-aware geometric refinement, with or without semi-supervised video training.
- Honest baselines (label copy, classical, HRNet) and a cross-venue evaluation.
