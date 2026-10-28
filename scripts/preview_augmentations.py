"""Save a grid of augmented training samples to check that masks stay aligned.

Each row is one image; each column is a different random augmentation of it.

Usage:
    python scripts/preview_augmentations.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from petseg.data.dataset import PetSegmentationDataset
from petseg.data.splits import read_split
from petseg.data.transforms import build_transforms, denormalize
from petseg.visualization import overlay_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview training augmentations.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--split-file", default="splits/train.txt", type=Path)
    parser.add_argument("--out", default="artifacts/data_inspection/augmentations.png", type=Path)
    parser.add_argument("--num-images", default=4, type=int)
    parser.add_argument("--num-augmentations", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # v2 transforms draw their randomness from torch
    ids = read_split(args.split_file)[: args.num_images]
    dataset = PetSegmentationDataset(args.data_dir, ids, build_transforms(train=True))

    fig, axes = plt.subplots(args.num_images, args.num_augmentations, figsize=(3 * args.num_augmentations, 3 * args.num_images))
    for row in range(args.num_images):
        for col in range(args.num_augmentations):
            image, mask = dataset[row]  # a new random augmentation every call
            image_np = (denormalize(image).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            axes[row, col].imshow(overlay_mask(image_np, mask.numpy()))
            axes[row, col].axis("off")
        axes[row, 0].set_title(ids[row], fontsize=9, loc="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    plt.close(fig)
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()