"""Run a model over a DataLoader and accumulate metrics."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from petseg.evaluation.metrics import SegmentationMetrics


@torch.no_grad()  # no gradients needed for evaluation: faster and uses less memory
def evaluate(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> SegmentationMetrics:
    """Evaluate predictions over a whole DataLoader.

    Args:
        predict_fn: takes a batch of images (B, 3, H, W) on `device` and returns
            predicted class indices of shape (B, H, W).
        loader: yields (images, masks) batches.
        device: where the model runs.
        num_classes: number of classes in the masks.
    """
    metrics = SegmentationMetrics(num_classes)
    for images, masks in tqdm(loader, desc="Evaluating", leave=False):
        predictions = predict_fn(images.to(device))
        metrics.update(predictions, masks)
    return metrics
