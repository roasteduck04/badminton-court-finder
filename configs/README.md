# Training configs

`baseline.json` is the full method. Every other file inherits from it via
`"base": "baseline.json"` and changes one factor:

| Config | Change vs baseline | Question it answers |
|---|---|---|
| `baseline.json` | none | full method |
| `rgb_only.json` | `in_channels: 3` | do the gray/CLAHE/Canny/court-colour channels help? |
| `no_synthetic.json` | `exclude_sources: ["blender"]` | does Blender data help on real images? |
| `no_reproj.json` | `reproj_weight: 0` | does the homography-consistency loss help? |
| `mse_heatmap.json` | `heatmap_loss: "mse"`, `offset_weight: 1.0` | JS (DSNT-style) vs MSE heatmap loss (offset weight follows the loss docstring's recommendation for MSE) |
| `no_symmetric.json` | `symmetric_loss: false` | does the rot180-symmetric loss help? |
| `sigma6.json` | `sigma: 6.0` | narrow (2 px) vs old wide (6 px) heatmap targets |

## Running

```bash
python -m src.training.train --config configs/baseline.json --set seed=0
python -m src.training.train --config configs/rgb_only.json --set seed=1 num_workers=4
# resume (refuses if num_epochs/batch_size/steps per epoch changed, unless --force)
python -m src.training.train --config configs/baseline.json --set seed=0 \
    --resume-from checkpoints/baseline_s0/last.pt
```

Outputs go to `checkpoint_dir` (default `checkpoints/{name}_s{seed}`):
`best_model.pt`, `last.pt`, `checkpoint_epoch_N.pt`, `config.json`, `history.json`.

## Protocol

- Run every config with at least 3 seeds (e.g. 0, 1, 2) and report mean and std.
  Differences smaller than the seed-to-seed spread are not results.
- Model selection uses `val_pck10` (pooled PCK@10 px in the 640 letterbox
  canvas) on the group-disjoint `val` split, which contains real images only
  (`val_exclude_sources: ["blender"]`).
- Report final numbers with `src.evaluation.evaluate` on `test_real` and
  `test_roboflow` separately, never on `val`.
- Data comes from `data/colab_data_v2/manifest.json` (built by
  `scripts/prepare_colab.py`). If that file is missing, training falls back to
  the old `data/colab_data/{train,valid}` directories with a warning; those
  splits leak frames between train and test and must not be used for reported numbers.
