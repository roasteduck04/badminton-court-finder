# Auto-exported from notebooks/train_colab.ipynb (py:percent format).
# Open in Colab via jupytext, or keep the .ipynb as the source of truth.

# %% [markdown]
# # CourtVisionNet Training on Google Colab
#
# Trains CourtVisionNet on the leak-free, group-split dataset and evaluates it on the
# held-out `test_real` and `test_roboflow` splits.
#
# **Prerequisites:**
# 1. Build the dataset locally: `python scripts/prepare_colab.py --zip` (writes `data/colab_data_v2.zip`)
# 2. Upload `colab_data_v2.zip` to your Google Drive (adjust `DRIVE_ZIP_PATH` below)
# 3. Runtime → Change runtime type → **GPU**
# 4. Run all cells in order

# %% [markdown]
# ## 1. Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# === CONFIGURE THESE ===
DRIVE_ZIP_PATH = '/content/drive/MyDrive/colab_data_v2.zip'
REPO_URL = 'https://github.com/roasteduck04/badminton-court-finder.git'
REPO_REF = 'main'                      # pin a commit SHA for reported runs
CONFIG = 'configs/baseline.json'
RUN_NAME = 'baseline'
SEEDS = [0, 1, 2]
EXTRA_OVERRIDES = []                   # e.g. ['num_epochs=50', 'batch_size=16']
RESUME_FROM = ''                       # e.g. f'{OUT_DIR}/baseline_s0/last.pt'
NUM_WORKERS = 2

import os
REPO_DIR = '/content/badminton-court-finder'
OUT_DIR = '/content/drive/MyDrive/cvn_runs'   # on Drive so runs survive disconnects
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# - `REPO_REF`: pin to a commit SHA for any run you report (the SHA is also stored in every checkpoint).
# - `CONFIG`: `configs/baseline.json` or an ablation from `configs/` (see `configs/README.md`).
# - `SEEDS`: use at least 3 seeds per config for reported numbers.

# %%
# Clone the repo and check out the pinned ref (commit SHA, tag or branch)
import os
if not os.path.exists(REPO_DIR):
    get_ipython().system(f"git clone -q {REPO_URL} {REPO_DIR}")  # noqa: F821
get_ipython().system(f"git -C {REPO_DIR} fetch -q --all --tags")  # noqa: F821
get_ipython().system(f"git -C {REPO_DIR} checkout -q {REPO_REF}")  # noqa: F821
get_ipython().system(f"git -C {REPO_DIR} log -1 --format='Using commit %H (%s)'")  # noqa: F821
get_ipython().run_line_magic('cd', REPO_DIR)  # noqa: F821

# %%
# torch/torchvision are preinstalled; the code needs the albumentations 2.x API
get_ipython().system("pip install -q 'albumentations>=2.0.8,<3' 'shapely>=2.0,<3'")  # noqa: F821

# %%
# Unzip the dataset to local disk (much faster than reading from Drive)
import zipfile
DATA_DIR = '/content/colab_data_v2'
if not os.path.isfile(os.path.join(DATA_DIR, 'manifest.json')):
    print(f'Unzipping {DRIVE_ZIP_PATH} ...')
    with zipfile.ZipFile(DRIVE_ZIP_PATH) as zf:
        zf.extractall('/content')
print(f'Using data from {DATA_DIR}')

# %%
# Verify the manifest and its splits
import json
MANIFEST = os.path.join(DATA_DIR, 'manifest.json')
with open(MANIFEST, encoding='utf-8') as f:
    manifest = json.load(f)
for split, entries in manifest['splits'].items():
    by_source = {}
    for e in entries:
        by_source[e['source']] = by_source.get(e['source'], 0) + 1
    groups = len({e['group'] for e in entries})
    print(f'  {split:<14} {len(entries):5d} images, {groups:3d} groups  {by_source}')

# %% [markdown]
# ## 2. Verify GPU

# %%
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# %% [markdown]
# ## 3. Train

# %%
# Train one run per seed. Each run writes best_model.pt, last.pt,
# config.json and history.json to {OUT_DIR}/{name}_s{seed}.
def run_dir(seed):
    return f'{OUT_DIR}/{RUN_NAME}_s{seed}'

for seed in SEEDS:
    overrides = [f'seed={seed}', f'manifest={MANIFEST}', f'name={RUN_NAME}',
                 f'checkpoint_dir={run_dir(seed)}', f'num_workers={NUM_WORKERS}'] + EXTRA_OVERRIDES
    resume = f'--resume-from {RESUME_FROM}' if RESUME_FROM else ''
    print(f'===== {CONFIG} seed={seed} =====')
    get_ipython().system(f"python -m src.training.train --config {CONFIG} {resume} --set {' '.join(overrides)}")  # noqa: F821

# %% [markdown]
# ## 4. Evaluate on held-out test splits

# %%
# Evaluate each run's best checkpoint on the two held-out test splits
# separately (never on val, which was used for model selection).
for seed in SEEDS:
    ckpt = f'{run_dir(seed)}/best_model.pt'
    for split in ['test_real', 'test_roboflow']:
        print(f'===== seed={seed} {split} =====')
        get_ipython().system(f"python -m src.evaluation.evaluate --checkpoint {ckpt} --manifest {MANIFEST} --split {split} --out {run_dir(seed)}/eval_{split}.json")  # noqa: F821

# %%
# Headline test numbers per seed (full tables are in eval_*.md next to each checkpoint)
import glob
for path in sorted(glob.glob(f'{OUT_DIR}/{RUN_NAME}_s*/eval_*.json')):
    with open(path, encoding='utf-8') as f:
        summary = json.load(f)
    print(path)
    print(json.dumps(summary.get('headline', summary), indent=1)[:2000])

# %% [markdown]
# ## 5. Training history

# %%
import json
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for seed in SEEDS:
    path = f'{run_dir(seed)}/history.json'
    if not os.path.exists(path):
        continue
    with open(path, encoding='utf-8') as f:
        history = json.load(f)
    ep = [h['epoch'] for h in history]
    axes[0].plot(ep, [h['train_loss'] for h in history], label=f'train s{seed}')
    axes[0].plot(ep, [h['val_loss'] for h in history], '--', label=f'val s{seed}')
    axes[1].plot(ep, [h['val_pck10'] for h in history], label=f'PCK@10 s{seed}')
    axes[1].plot(ep, [h['val_pck5'] for h in history], ':', label=f'PCK@5 s{seed}')
    axes[2].plot(ep, [h['val_mean_px'] for h in history], label=f's{seed}')
axes[0].set_title('Loss'); axes[1].set_title('Val PCK (640 canvas px)'); axes[2].set_title('Val mean error (px)')
axes[1].set_ylim(0, 1)
for ax in axes:
    ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/training_history.png', dpi=150)
plt.show()

# %% [markdown]
# ## 6. Outputs
#
# Runs are written to `OUT_DIR` on Google Drive (`best_model.pt`, `last.pt`, `config.json`,
# `history.json`, `eval_test_real.*`, `eval_test_roboflow.*`). If the runtime disconnects,
# set `RESUME_FROM` to the run's `last.pt` and re-run the training cell.
