# Training on RunPod

One GPU pod runs every config × seed back to back with `scripts/runpod/run_experiments.sh`, evaluates each run on `test_real` and `test_roboflow`, runs the nearest-neighbour baseline, and writes `summary.md` (mean ± std over seeds).

## Cost and GPU

Prices on runpod.io/pricing (page dated 2026-10-28, on-demand, billed per second):

| GPU | Community | Secure |
|---|---|---|
| RTX 4090 (recommended) | $0.34/h | $0.89/h |
| RTX A5000 | $0.16/h | $0.27/h |
| A100 | $1.19–1.39/h | $1.79/h |

The model (ResNet-50 + FPN, 640 px, batch 8, fp32) fits easily in 24 GB, so an RTX 4090 is the best value and an A100 is not worth the premium. Measure the time per epoch with the pilot run below before launching the full set.

The 2026-11-03 build has about 4,800 training images (about 600 steps per epoch), dominated by the 3,991 renders. As a planning figure, if a 100-epoch run takes 3–5 h, then:
- baseline × 3 seeds is about 10–15 GPU-hours, roughly $4–5 on a Community 4090;
- baseline + 6 ablations × 3 seeds (21 runs) is about 60–100 GPU-hours, roughly $20–35.

Early stopping (patience 15) usually ends runs sooner.

**Spot / interruptible** pods are cheaper but can stop without notice. The script is restart-safe: re-running it skips finished runs and resumes unfinished ones from `last.pt`, and the resumed run reproduces an uninterrupted one. Keep `OUT_DIR` on the volume (`/workspace`) so checkpoints survive.

## Steps

1. **Build and pack the dataset locally.** This writes `data/colab_data_v2.zip`, with `colab_data_v2/` at the zip root:
   ```
   python scripts/prepare_colab.py --out data/colab_data_v2 --overwrite --extra-synthetic data/blender_aug --zip
   ```
   Install `runpodctl` once. On Windows, download `runpodctl-windows-amd64.exe` from github.com/runpod/runpodctl/releases, rename it `runpodctl.exe` and put it on your PATH. Then run `runpodctl config --apiKey <key>` with a key from RunPod console → Settings → API Keys.

2. **Create the pod** in the RunPod console:
   - GPU: RTX 4090
   - Template: the official RunPod PyTorch 2.8 image
   - Container disk: 30 GB
   - Volume disk: 20 GB, mounted at `/workspace`

   Community-cloud pods may have no public IP. If so, use `runpodctl send/receive` (step 3) rather than scp.

3. **Copy the dataset to the pod.**
   - Locally: `runpodctl send data/colab_data_v2.zip`. It prints a one-time code.
   - In the pod's web terminal: `cd /workspace && runpodctl receive <code>`.

   Keep the API key out of the repo and out of shell history files.

4. **Get the code** (in the pod):
   ```
   cd /workspace
   git clone https://github.com/roasteduck04/badminton-court-finder.git
   cd badminton-court-finder
   git checkout <commit SHA or branch>
   ```
   Pin a SHA for any run you report. The SHA is also stored in every checkpoint and in `eval_done`.

5. **Pilot** (2 epochs, a few minutes). This confirms data, CUDA and evaluation work, and shows the time per epoch:
   ```
   CONFIGS=baseline SEEDS=0 EXTRA_SET="num_epochs=2" OUT_DIR=/workspace/pilot bash scripts/runpod/run_experiments.sh
   ```

6. **Full runs** inside `tmux`, so they survive a closed browser tab:
   ```
   tmux new -s train
   AUTO_STOP=1 bash scripts/runpod/run_experiments.sh
   AUTO_STOP=1 CONFIGS="rgb_only no_synthetic no_reproj mse_heatmap no_symmetric sigma6" bash scripts/runpod/run_experiments.sh
   ```
   Detach with `Ctrl-b d` and reattach with `tmux attach -t train`. With `AUTO_STOP=1`, the pod stops itself when the script ends, so GPU billing stops; the volume is still billed until you terminate the pod. Run the two commands one after the other (or put both config lists in one `CONFIGS`) so the first one doesn't stop the pod before the second starts.

7. **Collect results** by restarting the pod if it stopped, then:
   ```
   cd /workspace && tar czf runs.tgz runs/*/eval_* runs/*/history.json runs/*/config.json runs/*/best_model.pt runs/summary.md runs/nn_baseline_*
   runpodctl send runs.tgz
   ```
   Receive locally with `runpodctl receive <code>`.

8. **Terminate the pod** in the console once results are safe. A stopped pod still bills its volume.

## Outputs

`$OUT_DIR/<config>_s<seed>/` contains:
- `best_model.pt`, `last.pt`, `config.json`, `history.json`
- `eval_<split>.json` / `_rows.jsonl` / `.md`
- `eval_done`, which holds the commit SHA

`$OUT_DIR/summary.md` is the table to paste into the paper draft, next to `nn_baseline_<split>.json`. Rebuild the table locally with:
```
python scripts/summarize_runs.py --runs <dir>
```

## Synthetic data

Render locally rather than on RunPod. On the development machine EEVEE takes about 1–4 s per image at 16 samples, so 1000 images is roughly an hour. EEVEE headless rendering on cloud GPUs is not well documented; Cycles with CUDA/OptiX works headless but looks different.
