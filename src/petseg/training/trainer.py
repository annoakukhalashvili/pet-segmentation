"""Training loop with validation, checkpointing, resuming and early stopping."""

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from petseg.config import TrainConfig
from petseg.data.annotations import NUM_CLASSES, remap_trimap
from petseg.data.dataset import PetSegmentationDataset
from petseg.data.splits import read_split
from petseg.data.transforms import build_transforms
from petseg.env import select_device
from petseg.evaluation.evaluate import evaluate
from petseg.models.deeplab import build_model

logger = logging.getLogger(__name__)

SELECTION_METRIC = "mean_iou"  # the validation metric used to pick the best checkpoint


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


def median_frequency_weights(data_dir: Path, image_ids: list[str]) -> torch.Tensor:
    """Class weights = median class frequency / class frequency (computed on TRAIN masks only).

    Rare classes (boundary) get weights above 1, frequent ones (background) below 1,
    so mistakes on rare classes cost more in the loss.
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for image_id in tqdm(image_ids, desc="Counting class pixels", leave=False):
        trimap = np.array(Image.open(Path(data_dir) / "annotations" / "trimaps" / f"{image_id}.png"))
        counts += np.bincount(remap_trimap(trimap).ravel(), minlength=NUM_CLASSES)
    frequencies = counts / counts.sum()
    return torch.tensor(np.median(frequencies) / frequencies, dtype=torch.float32)


def freeze_backbone(model: nn.Module) -> None:
    """Stop gradient updates for the pretrained feature extractor."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with a smaller learning rate for the pretrained backbone than for the head.

    The backbone already holds useful features, so it should change slowly; the head
    (partly brand-new) needs larger steps.
    """
    groups = [{"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": config.lr_head}]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": config.lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


class Trainer:
    """Runs one experiment described by a TrainConfig."""

    def __init__(self, config: TrainConfig, max_batches: int | None = None):
        """
        Args:
            config: experiment settings.
            max_batches: if set, use only this many batches per epoch (quick smoke tests).
        """
        self.config = config
        self.max_batches = max_batches
        self.out_dir = config.experiment_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        set_seed(config.seed)
        self.device = select_device(config.device)
        self.use_amp = config.mixed_precision and self.device.type == "cuda"

        splits_dir = Path(config.splits_dir)
        train_ids = read_split(splits_dir / "train.txt")
        val_ids = read_split(splits_dir / "val.txt")
        train_set = PetSegmentationDataset(config.data_dir, train_ids, build_transforms(True, config.image_size))
        val_set = PetSegmentationDataset(config.data_dir, val_ids, build_transforms(False, config.image_size))
        self.train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,  # avoids a tiny last batch, which makes BatchNorm statistics noisy
            generator=torch.Generator().manual_seed(config.seed),
        )
        self.val_loader = DataLoader(
            val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        )

        self.model = build_model(NUM_CLASSES, pretrained=config.pretrained)
        if config.freeze_backbone:
            freeze_backbone(self.model)
        self.model.to(self.device)

        weights = None
        if config.class_weighting == "median_frequency":
            weights = median_frequency_weights(Path(config.data_dir), train_ids)
            logger.info("Class weights (background, pet, boundary): %s", [round(w, 3) for w in weights.tolist()])
        self.loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device) if weights is not None else None)

        self.optimizer = build_optimizer(self.model, config)
        # Cosine schedule: the learning rate decreases smoothly to ~0 by the last epoch.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_epoch = 1
        self.best_score = -1.0
        self.epochs_without_improvement = 0
        self.history: list[dict] = []

        with open(self.out_dir / "config.json", "w", encoding="utf-8") as file:
            json.dump(config.to_dict(), file, indent=2)

    # ---------- checkpoints ----------

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_score": self.best_score,
                "epochs_without_improvement": self.epochs_without_improvement,
                "history": self.history,
                "config": self.config.to_dict(),
            },
            path,
        )

    def resume(self) -> None:
        """Continue from last.pt, e.g. after a Kaggle/Colab session was interrupted."""
        path = self.out_dir / "last.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume from: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.best_score = state["best_score"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.history = state["history"]
        self.start_epoch = state["epoch"] + 1
        logger.info("Resumed from epoch %d (best %s so far: %.4f)", state["epoch"], SELECTION_METRIC, self.best_score)

    # ---------- training ----------

    def train_one_epoch(self, epoch: int) -> float:
        """One pass over the training data. Returns the average loss."""
        self.model.train()
        if self.config.freeze_backbone:
            # Frozen layers must also keep their BatchNorm statistics fixed.
            self.model.backbone.eval()

        total_loss, num_batches = 0.0, 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False)
        for batch_index, (images, masks) in enumerate(progress):
            if self.max_batches is not None and batch_index >= self.max_batches:
                break
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)  # clear gradients from the previous step
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)["out"]  # (B, num_classes, H, W)
                loss = self.loss_fn(logits, masks)  # masks: (B, H, W) class indices

            self.scaler.scale(loss).backward()  # backpropagation: compute gradients
            self.scaler.step(self.optimizer)  # update the weights
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()

        def predict(images: torch.Tensor) -> torch.Tensor:
            return self.model(images)["out"].argmax(dim=1)

        loader = self.val_loader
        if self.max_batches is not None:
            loader = [batch for _, batch in zip(range(self.max_batches), self.val_loader)]
        return evaluate(predict, loader, self.device, NUM_CLASSES).compute()

    def fit(self) -> dict:
        """Train for config.epochs epochs (or until early stopping). Returns the best val metrics."""
        logger.info(
            "Experiment %s on %s: %d train / %d val images",
            self.config.name, self.device, len(self.train_loader.dataset), len(self.val_loader.dataset),
        )
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            start = time.perf_counter()
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            score = val_metrics[SELECTION_METRIC]
            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "seconds": round(time.perf_counter() - start, 1),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(record)
            logger.info(
                "epoch %d | loss %.4f | val mIoU %.4f | val fg IoU %.4f | val boundary IoU %.4f | %.0fs%s",
                epoch, train_loss, score, val_metrics["foreground_iou"], val_metrics["iou_boundary"] or 0.0,
                record["seconds"], " | best" if improved else "",
            )

            if improved:
                self.save_checkpoint(self.out_dir / "best.pt", epoch)
            self.save_checkpoint(self.out_dir / "last.pt", epoch)
            self.write_history()

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info("Early stopping: no improvement for %d epochs", self.epochs_without_improvement)
                break

        best = max(self.history, key=lambda r: r[f"val_{SELECTION_METRIC}"])
        with open(self.out_dir / "best_val_metrics.json", "w", encoding="utf-8") as file:
            json.dump(best, file, indent=2)
        logger.info("Best epoch %d: val %s = %.4f", best["epoch"], SELECTION_METRIC, best[f"val_{SELECTION_METRIC}"])
        return best

    def write_history(self) -> None:
        with open(self.out_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
