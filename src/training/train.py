"""Training loop for CourtVisionNet.

Provides:
    train_one_epoch(model, dataloader, loss_fn, optimizer, device) -> float
    validate(model, dataloader, loss_fn, device) -> (float, dict)
    train(config: TrainConfig) -> dict

`train()` wires up the dataset, model, loss, optimizer and scheduler
described by a `TrainConfig`, runs the full training loop with a
backbone-freeze warm-up, cosine annealing LR schedule, best-checkpoint
saving, periodic checkpoints, and early stopping on validation loss.
"""

import os

import numpy as np
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device="cuda"):
    """Run one training epoch. Returns the mean loss over all samples."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        targets = {
            "mask": batch["mask"].to(device),
            "heatmaps": batch["heatmaps"].to(device),
            "keypoints": batch["keypoints"].to(device),
            "visibility": batch["visibility"].to(device),
        }

        optimizer.zero_grad()
        pred = model(images)
        loss, _ = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss / max(count, 1)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device="cuda"):
    """Run validation over the full dataloader (no gradient updates).

    Returns (avg_loss, avg_components, avg_metrics) where avg_components is
    a dict of the individual loss terms (seg_loss, heatmap_loss,
    offset_loss, visibility_loss) averaged over the dataset, and
    avg_metrics is a dict with "pck_at_10" and "mre" keys.
    """
    from src.evaluation.metrics import mean_reprojection_error, pck_at_k

    model.eval()
    total_loss = 0.0
    total_components = {}
    count = 0
    all_pck = []
    all_mre = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        targets = {
            "mask": batch["mask"].to(device),
            "heatmaps": batch["heatmaps"].to(device),
            "keypoints": batch["keypoints"].to(device),
            "visibility": batch["visibility"].to(device),
        }

        pred = model(images)
        loss, components = loss_fn(pred, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size
        for k, v in components.items():
            total_components[k] = total_components.get(k, 0.0) + v * batch_size

        # Per-sample metrics
        pred_kps = pred["offsets"].cpu().numpy()
        gt_kps = targets["keypoints"].cpu().numpy()
        gt_vis = targets["visibility"].cpu().numpy()

        for i in range(batch_size):
            _, pck_mean = pck_at_k(pred_kps[i], gt_kps[i], gt_vis[i], k=10)
            all_pck.append(pck_mean)
            mre = mean_reprojection_error(pred_kps[i], gt_kps[i], gt_vis[i], 640, 640)
            if mre is not None:
                all_mre.append(mre)

    avg_loss = total_loss / max(count, 1)
    avg_components = {k: v / max(count, 1) for k, v in total_components.items()}
    avg_metrics = {
        "pck_at_10": float(np.mean(all_pck)) if all_pck else 0.0,
        "mre": float(np.mean(all_mre)) if all_mre else 0.0,
    }
    return avg_loss, avg_components, avg_metrics


def train(config):
    """Full training loop driven by a TrainConfig.

    Builds the train/val datasets and loaders, the model, loss and
    optimizer, then runs `config.num_epochs` epochs with:
      - backbone frozen for the first `config.freeze_backbone_epochs` epochs
      - AdamW + cosine annealing LR over the full run
      - best-validation-loss checkpointing
      - periodic checkpoints every 10 epochs
      - early stopping once `config.patience` epochs pass without a new
        best validation loss

    Returns a dict with "best_val_loss" and "final_epoch".
    """
    from torch.utils.data import DataLoader

    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
    from src.training.dataset import CourtDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    train_transform = get_train_transforms(config.image_size)
    val_transform = get_val_transforms(config.image_size)

    train_ds = CourtDataset(
        config.train_annotations, config.train_images,
        transform=train_transform, image_size=config.image_size,
        heatmap_size=config.heatmap_size,
    )
    val_ds = CourtDataset(
        config.val_annotations, config.val_images,
        transform=val_transform, image_size=config.image_size,
        heatmap_size=config.heatmap_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device == "cuda"),
    )

    # Model
    model = CourtVisionNet(
        in_channels=config.in_channels,
        num_keypoints=config.num_keypoints,
        image_size=config.image_size,
        heatmap_size=config.heatmap_size,
        pretrained=config.pretrained,
    ).to(device)

    loss_fn = CourtVisionLoss(
        seg_weight=config.seg_weight,
        heatmap_weight=config.heatmap_weight,
        offset_weight=config.offset_weight,
        vis_weight=config.vis_weight,
        collinear_weight=config.collinear_weight,
        ratio_weight=config.ratio_weight,
        convex_weight=config.convex_weight,
    )

    # Optimizer + cosine annealing LR over the full run.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    final_epoch = 0

    for epoch in range(config.num_epochs):
        final_epoch = epoch + 1

        # Backbone freeze/unfreeze schedule.
        if epoch < config.freeze_backbone_epochs:
            model.freeze_backbone()
            frozen = True
        else:
            model.unfreeze_backbone()
            frozen = False

        print(f"\nEpoch {final_epoch}/{config.num_epochs} {'(backbone frozen)' if frozen else ''}")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_components, val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        for k, v in val_components.items():
            print(f"    {k}: {v:.4f}")
        print(f"  PCK@10: {val_metrics['pck_at_10']:.4f}")
        if val_metrics['mre'] > 0:
            print(f"  MRE:    {val_metrics['mre']:.2f} px")

        history.append({
            "epoch": final_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_components.items()},
        })

        # Save best model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": final_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(config.checkpoint_dir, "best_model.pt"))
            print("  Saved best model")
        else:
            patience_counter += 1

        # Periodic checkpoint.
        if final_epoch % 10 == 0:
            torch.save({
                "epoch": final_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{final_epoch}.pt"))

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {final_epoch}")
            break

    return {
        "best_val_loss": best_val_loss,
        "final_epoch": final_epoch,
        "history": history,
    }


if __name__ == "__main__":
    from src.training.config import TrainConfig

    cfg = TrainConfig()
    train(cfg)
