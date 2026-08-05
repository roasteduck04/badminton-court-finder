# %% [markdown]
# # CourtVisionNet — Colab Training Notebook
#
# GPU training notebook for the CourtVisionNet badminton court detection
# model. Upload this file to Colab (or use `jupytext`/`File > Upload notebook`
# with the "pair py:percent" format) and run top to bottom.
#
# Runtime > Change runtime type > **GPU** (T4 or better) before running.
#
# Steps:
# 1. Mount Google Drive + get the repo
# 2. Install dependencies
# 3. Link annotated data (from Drive)
# 4. Configure `TrainConfig`
# 5. Run training with `train(config)`
# 6. Visualize loss curves
# 7. Save the best checkpoint back to Drive

# %% [markdown]
# ## 1. Mount Google Drive + clone/copy the repo
#
# Two options are supported — pick whichever matches your setup:
#   (a) clone from a git remote (fill in REPO_URL below), or
#   (b) the project already lives in Drive and you just `%cd` into it.
#
# Either way, this repo (and your annotated `data/` directory) must end up
# somewhere under `/content` so relative imports (`src.training...`) work.

# %%
from google.colab import drive  # noqa: E402

drive.mount("/content/drive")

# %%
import os  # noqa: E402

REPO_URL = ""  # e.g. "https://github.com/<you>/badminton-court-finder.git"
REPO_DIR = "/content/badminton-court-finder"
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/badminton-court-finder"  # option (b)

if REPO_URL:
    if not os.path.exists(REPO_DIR):
        get_ipython().system(f"git clone {REPO_URL} {REPO_DIR}")  # noqa: F821
    get_ipython().run_line_magic("cd", REPO_DIR)  # noqa: F821
elif os.path.exists(DRIVE_PROJECT_DIR):
    get_ipython().run_line_magic("cd", DRIVE_PROJECT_DIR)  # noqa: F821
else:
    raise RuntimeError(
        "Set REPO_URL to a git remote, or place the project at "
        f"{DRIVE_PROJECT_DIR} in your Drive."
    )

print("Working directory:", os.getcwd())

# %% [markdown]
# ## 2. Install dependencies

# %%
get_ipython().system("pip install -q -r requirements.txt")  # noqa: F821

# %%
import torch  # noqa: E402

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 3. Upload / link annotated data
#
# The training script expects:
#   - `train_images/`, `train_annotations/` — training frames + JSON labels
#   - `val_images/`, `val_annotations/` — validation frames + JSON labels
#
# Easiest path: keep your annotated `data/` folder (frames + annotations,
# produced by `src/tools/annotator.py`) in Drive and point the config at it
# directly (Drive paths shown below). Swap in your own paths as needed.
#
# If you don't have a pre-made train/val split, the helper below makes one
# from a single annotations directory using an 80/20 split.

# %%
DATA_ROOT = "/content/drive/MyDrive/badminton-court-finder/data"

# If you already have separate train/ and val/ annotation folders, set:
TRAIN_ANNOTATIONS = f"{DATA_ROOT}/annotations/train"
VAL_ANNOTATIONS = f"{DATA_ROOT}/annotations/val"
TRAIN_IMAGES = f"{DATA_ROOT}/frames"
VAL_IMAGES = f"{DATA_ROOT}/frames"

print("Train annotations:", TRAIN_ANNOTATIONS, os.path.exists(TRAIN_ANNOTATIONS))
print("Val annotations:  ", VAL_ANNOTATIONS, os.path.exists(VAL_ANNOTATIONS))


# %%
def make_train_val_split(annotations_dir, images_dir, out_dir, val_fraction=0.2, seed=42):
    """One-time helper: split a flat annotations directory into train/ and
    val/ subfolders (by copying JSON files) for use with CourtDataset.

    Only needed if your annotations aren't already split. Images stay in
    place; only the annotation JSONs (which reference absolute image
    paths) are partitioned.
    """
    import glob
    import random
    import shutil

    random.seed(seed)
    paths = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    random.shuffle(paths)

    n_val = max(1, int(len(paths) * val_fraction))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]

    train_out = os.path.join(out_dir, "train")
    val_out = os.path.join(out_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    for p in train_paths:
        shutil.copy(p, train_out)
    for p in val_paths:
        shutil.copy(p, val_out)

    print(f"Split {len(paths)} annotations -> {len(train_paths)} train / {len(val_paths)} val")
    return train_out, val_out


# Uncomment to generate a split from a single annotations directory:
# TRAIN_ANNOTATIONS, VAL_ANNOTATIONS = make_train_val_split(
#     f"{DATA_ROOT}/annotations", TRAIN_IMAGES, f"{DATA_ROOT}/annotations_split"
# )

# %% [markdown]
# ## 4. Configure TrainConfig

# %%
from src.training.config import TrainConfig  # noqa: E402

config = TrainConfig(
    train_annotations=TRAIN_ANNOTATIONS,
    val_annotations=VAL_ANNOTATIONS,
    train_images=TRAIN_IMAGES,
    val_images=VAL_IMAGES,
    in_channels=7,
    num_keypoints=14,
    image_size=640,
    heatmap_size=160,
    pretrained=True,
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=10,
    freeze_backbone_epochs=5,
    checkpoint_dir="/content/drive/MyDrive/badminton-court-finder/checkpoints",
    num_workers=2,  # safe to raise on Colab's Linux runtime
)
config

# %% [markdown]
# ## 5. Run training

# %%
from src.training.train import train  # noqa: E402

result = train(config)
print("Best val loss:", result["best_val_loss"])
print("Final epoch:  ", result["final_epoch"])

# %% [markdown]
# ## 6. Visualize loss curves

# %%
import matplotlib.pyplot as plt  # noqa: E402

history = result["history"]
epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="train_loss")
plt.plot(epochs, val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CourtVisionNet training curves")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
component_keys = [k for k in history[0].keys() if k.startswith("val_") and k != "val_loss"]

plt.figure(figsize=(8, 5))
for key in component_keys:
    plt.plot(epochs, [h[key] for h in history], label=key)
plt.xlabel("Epoch")
plt.ylabel("Loss component")
plt.title("Validation loss components")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 7. Save the best model to Drive
#
# `train()` already checkpoints to `config.checkpoint_dir`, which above
# points straight at Drive — so `best_model.pt` is already saved there.
# This cell just confirms it and prints a summary you can copy elsewhere.

# %%
best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")
print("Best checkpoint exists:", os.path.exists(best_ckpt_path))

if os.path.exists(best_ckpt_path):
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    print("Saved epoch:   ", ckpt["epoch"])
    print("Saved val_loss:", ckpt["val_loss"])
    print("Path:          ", best_ckpt_path)
