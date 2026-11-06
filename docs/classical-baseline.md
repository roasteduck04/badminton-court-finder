# Classical line-fitting baseline (B2), results 2026-11-01

`scripts/classical_baseline.py` implements the "B2 recipe" in `docs/improvement-plan.md`: Farin et al. 2004 court-model fitting, with MonoTrack's (2022) max-cut split into two line families and the parameters from github.com/gchlebus/tennis-court-detection. Nothing is learned. It is scored with the same metric code as the model (`run_evaluation`, original-image pixels, canonical labels).

```
python scripts/classical_baseline.py --manifest data/colab_data_v2/manifest.json \
    --eval-split test_real --out results/classical_test_real.json [--debug-dir DIR]
python scripts/classical_baseline.py --manifest ... --eval-split test_roboflow \
    --out results/classical_test_roboflow.json
python scripts/classical_baseline.py --image photo.jpg --debug-dir out/   # one image + overlay
```

It runs on CPU at about 3–5 s per image (1280 px working resolution). Tests: `tests/test_classical_baseline.py` covers synthetic end-on, side, oblique and partial views, distractor lines, yellow lines (chroma mode) and a blank image. Synthetic keypoints are recovered to about 0.2 px.

## Results

Headline numbers from `results/classical_<split>.json` (default threshold) and `results/classical_<split>_nothr.json` (`--score-threshold=-1e9`, every hypothesis accepted). A failed image predicts nothing, so all its GT-visible points count as misses and its court IoU is 0. PCK is in %, at the 640 px reference resolution; brackets give the 95% bootstrap CI over images.

| | NN label copy | Classical, threshold 2.3 | Classical, no threshold |
|---|---|---|---|
| **test_real (41 images)** | | | |
| completeness (has a prediction) | 100 | 46.3 | 97.6 |
| PCK@5px | 0.0 | 37.7 | 43.9 |
| PCK@10px | 0.4 | 37.9 [22.3, 54.8] | 44.1 [27.4, 61.3] |
| sym PCK@10px (rot180-tolerant) | 0.4 | 46.7 | 52.9 |
| PCK_norm@0.02 / @0.05 | 0.9 / 5.5 | 38.1 / 38.3 | 44.3 / 45.0 |
| court-plane error, median (m) | 6.96 | inf (> 50% missed) | 6.55 |
| court IoU, mean (successful fits only) | 0.46 | 0.46 (0.99) | 0.62 (0.77) |
| **test_roboflow (46 images)** | | | |
| completeness | 100 | 73.9 | 100 |
| PCK@5px | 0.8 | 73.9 | 98.1 |
| PCK@10px | 1.2 | 73.9 [60.9, 87.0] | 98.1 [94.4, 100] |
| PCK_norm@0.02 / @0.05 | 1.2 / 39.9 | 73.9 / 73.9 | 98.1 / 98.5 |
| court-plane error, median (m) | 1.19 | 0.018 | 0.017 |
| court IoU, mean | 0.80 | 0.74 (0.99) | 0.99 |

The NN numbers are a re-run of `scripts/nn_baseline.py` (ncc) on the same splits and match the handover (0.4% / 1.2%).

### Takeaways

- **Broadcast views are solved by geometry alone.** On test_roboflow, with no threshold, 45 of 46 frames fit to sub-pixel accuracy (median error 0.7 px, 1.7 cm on the court). A learned model has to beat 98% PCK@10px here, not 1.2%.
- **Close, low phone views are where learning has to earn its keep.** On test_real, 17 of 41 images fit correctly. Those that fail show only 2–4 court lines, or a lot of hall clutter.
- **The failure threshold does not transfer across scales.** The threshold is on the walk score divided by the working long side, i.e. the length of confirmed court line. It was fixed on val as the lowest value at which every accepted val image is correct: on val, all scores ≥ 2.38 are correct fits and all wrong fits score ≤ 2.27. On test it also rejects every frame of the second broadcast event (`output2`, scores 2.04–2.26), whose court is smaller in the frame, although all 11 are correct fits. Pooled PCK counts a rejection as a miss, so rejecting never raises PCK. The threshold only buys precision among accepted images:

  | accepted correct / accepted (median error ≤ 10 px) | none | ≥ 1.5 | ≥ 2.0 | ≥ 2.3 |
  |---|---|---|---|---|
  | val | 24/35 | 24/31 | 24/28 | 24/24 |
  | test_real | 17/40 | 17/24 | 17/22 | 15/19 |
  | test_roboflow | 45/46 | 45/46 | 45/45 | 34/34 |

  This sweep is diagnostic only: the threshold was not changed after looking at test. A scale-free acceptance test, for example the score relative to the projected model length inside the image, is follow-up work. The plain on-line pixel ratio does not separate right from wrong fits on val (wrong 0.44–0.85, right 0.63–0.93).
- **Court IoU is uninformative for close-ups.** When the court covers the whole frame, a wrong fit still gets IoU ≈ 1. For example, val `130837` has IoU 0.98 at a median error of 317 px. Lead with PCK and court-metre error for test_real.

## Failure modes (from the `--debug-dir` overlays)

1. **Adjacent courts and hall clutter.** The test_real hall has several courts side by side, plus rows of seats, ad-board edges and banner stripes that are long, straight and bright. These lines fill the two families, and wrong hypotheses that match them, or a neighbouring court's lines, score as well as the true court. Examples: `201605`, `201559`, `201614`.
2. **Too few court lines.** Low phone views close to the floor show 2–4 lines. Several model-line assignments (e.g. short-service vs long-service line) explain them equally well, so the hypothesis is ambiguous even when the lines are found. `WA0005` (wooden floor in front, one transverse line) does not even yield 2 lines per family.
3. **Other sports' and non-court lines.** The net tape, net posts and the umpire chair give strong transverse or vertical lines. One broadcast frame (`output_0026`) fits one line off (IoU 0.79). In the synthetic test, a diagonal line and an adjacent sideline did not break the fit, but real distractors are more numerous.
4. **Lens distortion.** Phone ultra-wide frames bend straight lines (`201605`). The pinhole homography cannot fit them, and the line fits split. This is improvement-plan item C4.
5. **Score depends on scale** (see above). Far or small courts confirm fewer pixels of line and fall under a fixed threshold.

## Implementation notes and deviations from the recipe

The recipe steps are implemented as written. The following details were added or chosen:

- **Working resolution.** Every image is resized to a 1280 px long side, and all pixel parameters (τ, 8 px refit band, 30 px minimum side) are at that scale. Line width is unknown and varies with distance, so τ ∈ {3, 6, 12, 24} and the masks are OR-ed. Coloured lines: `--line-mode chroma` (Lab a\*b\* distance to the floor colour, taken as the histogram mode of the lower two thirds of the image) or `both`. The default is luminance.
- **Orientation-gated support.** A line pixel supports a line only if its structure-tensor orientation runs along the line (±15°). Without this, Hough finds lines that only connect crossings of other lines. On the synthetic views those lines roughly doubled the number of hypotheses and the runtime.
- **Walk score.**
  - Model samples outside the image count 0. Otherwise every partial view is penalised.
  - Each line pixel counts once. Otherwise a degenerate H that stacks several model lines onto one image line scores multiple times.
  - The centre segment across the net (12, 17) is virtual and is not walked.
- **Search.**
  - One ordering per family is enough, because reversing an ordering is a court symmetry.
  - All hypotheses are ranked by a vectorised coarse score: a 4-sample pass first, the best 4000 then rescored with 12 samples per segment, each sample weighted by its image spacing.
  - The best 100 distinct hypotheses (de-duplicated modulo symmetry) get the exact pixel walk, and the best 3 are refined.
- **Refinement.** Levenberg–Marquardt on 8 homography parameters, Huber loss (2 px), with pixels reassigned to the nearest projected painted segment within 8 px, over 3 rounds. The walk score is quantised to whole pixels, so a slightly-off H can outscore the sub-pixel fit. A refined H is therefore kept unless its score drops by more than 5%.
- **Output.** Keypoints behind the camera are NaN. Predicted visibility means in front of the camera and inside the image. Labels go through `court_geometry.canonicalize`, and the returned homography is relabelled to match.
