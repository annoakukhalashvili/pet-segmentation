"""Experiment E0: evaluate the pretrained VOC model without any fine-tuning.

Pixels predicted as VOC "cat" or "dog" count as pet; everything else as background.
The model cannot predict the boundary class, so its boundary IoU is 0 by construction;
foreground_iou (pet + boundary vs background) is the fair comparison metric.

Usage:
    python scripts/evaluate_baseline.py                          # validation split
    python scripts/evaluate_baseline.py --split-file splits/test.txt   # only for the final report
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from petseg.data.annotations import NUM_CLASSES
from petseg.data.dataset import PetSegmentationDataset
from petseg.data.splits import read_split
from petseg.data.transforms import build_transforms
from petseg.env import select_device
from petseg.evaluation.evaluate import evaluate
from petseg.models.deeplab import build_pretrained_voc_model, voc_logits_to_pet_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the zero-shot baseline (E0).")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--split-file", default="splits/val.txt", type=Path)
    parser.add_argument("--image-size", default=320, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="artifacts/results", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    model = build_pretrained_voc_model().to(device).eval()

    dataset = PetSegmentationDataset(
        args.data_dir, read_split(args.split_file), build_transforms(train=False, image_size=args.image_size)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info("Evaluating E0 on %s: %d images, device=%s", args.split_file, len(dataset), device)

    def predict(images: torch.Tensor) -> torch.Tensor:
        return voc_logits_to_pet_labels(model(images)["out"])

    start = time.perf_counter()
    metrics = evaluate(predict, loader, device, NUM_CLASSES)
    elapsed = time.perf_counter() - start

    results = {
        "experiment": "E0_zero_shot",
        "split": args.split_file.stem,
        "num_images": len(dataset),
        "image_size": args.image_size,
        "device": str(device),
        "seconds_total": round(elapsed, 1),
        **metrics.compute(),
    }
    for key, value in results.items():
        logger.info("%-18s %s", key, value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"e0_zero_shot_{args.split_file.stem}.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
