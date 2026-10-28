"""Segmentation metrics computed from a confusion matrix.

The confusion matrix C has shape (num_classes, num_classes):
    C[t, p] = number of pixels whose TRUE class is t and PREDICTED class is p.
From it, for each class c:
    TP (true positives)  = C[c, c]          pixels correctly predicted as c
    FP (false positives) = column sum - TP  pixels wrongly predicted as c
    FN (false negatives) = row sum - TP     pixels of class c that were missed
    IoU  = TP / (TP + FP + FN)
    Dice = 2 TP / (2 TP + FP + FN)
"""

import math

import torch

from petseg.data.annotations import BACKGROUND, CLASS_NAMES, NUM_CLASSES


class SegmentationMetrics:
    """Accumulates a confusion matrix over many batches, then computes metrics."""

    def __init__(self, num_classes: int = NUM_CLASSES, class_names: tuple[str, ...] = CLASS_NAMES):
        self.num_classes = num_classes
        self.class_names = class_names
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Add one batch. Both tensors hold class indices and must have the same shape."""
        if predictions.shape != targets.shape:
            raise ValueError(f"shape mismatch: predictions {predictions.shape}, targets {targets.shape}")

        predictions = predictions.detach().flatten().cpu().long()
        targets = targets.detach().flatten().cpu().long()
        for name, values in (("predictions", predictions), ("targets", targets)):
            if values.numel() and (values.min() < 0 or values.max() >= self.num_classes):
                raise ValueError(f"{name} contain values outside 0..{self.num_classes - 1}")

        # Each (true, predicted) pair becomes one number: true * num_classes + predicted.
        # Counting those numbers fills the whole confusion matrix in one fast operation.
        pair_index = targets * self.num_classes + predictions
        counts = torch.bincount(pair_index, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def _tp_fp_fn(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        confusion = self.confusion.double()
        tp = confusion.diag()
        fp = confusion.sum(dim=0) - tp
        fn = confusion.sum(dim=1) - tp
        return tp, fp, fn

    def iou_per_class(self) -> torch.Tensor:
        """IoU for each class. NaN for a class that appears in neither predictions nor targets."""
        tp, fp, fn = self._tp_fp_fn()
        return tp / (tp + fp + fn)  # 0 / 0 gives NaN, which nanmean skips

    def dice_per_class(self) -> torch.Tensor:
        tp, fp, fn = self._tp_fp_fn()
        return 2 * tp / (2 * tp + fp + fn)

    def foreground_iou(self) -> float:
        """IoU of "any non-background class" vs background (pet and boundary merged).

        Lets us compare models that cannot predict the boundary class (the zero-shot
        baseline) with models that can.
        """
        fg = [c for c in range(self.num_classes) if c != BACKGROUND]
        tp = self.confusion[fg][:, fg].sum().item()
        fp = self.confusion[BACKGROUND, fg].sum().item()
        fn = self.confusion[fg, BACKGROUND].sum().item()
        denominator = tp + fp + fn
        return tp / denominator if denominator else math.nan

    def pixel_accuracy(self) -> float:
        total = self.confusion.sum().item()
        return self.confusion.diag().sum().item() / total if total else math.nan

    def compute(self) -> dict:
        """Return all metrics as plain Python numbers (NaN becomes None, for JSON)."""
        iou = self.iou_per_class()
        dice = self.dice_per_class()

        def clean(value: float) -> float | None:
            return None if math.isnan(value) else round(value, 4)

        results = {
            "mean_iou": clean(torch.nanmean(iou).item()),
            "mean_dice": clean(torch.nanmean(dice).item()),
            "foreground_iou": clean(self.foreground_iou()),
            "pixel_accuracy": clean(self.pixel_accuracy()),
        }
        for index, name in enumerate(self.class_names):
            results[f"iou_{name}"] = clean(iou[index].item())
            results[f"dice_{name}"] = clean(dice[index].item())
        return results
