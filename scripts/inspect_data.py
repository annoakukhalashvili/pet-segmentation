"""Inspect and validate the Oxford-IIIT Pet dataset.

Checks every image/trimap pair, collects statistics and saves:
    <out-dir>/data_report.json     statistics and the list of problem files
    <out-dir>/samples.png          example images with their masks

Usage:
    python scripts/inspect_data.py
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from petseg.data.annotations import CLASS_NAMES, NUM_CLASSES, parse_split_file, remap_trimap
from petseg.data.validate import check_sample
from petseg.visualization import overlay_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("trainval", "test")


def inspect_split(base_dir: Path, split: str) -> dict:
    """Validate every sample in one split and collect statistics."""
    records = parse_split_file(base_dir / "annotations" / f"{split}.txt")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    widths, heights = [], []
    errors, warnings, usable_ids = {}, {}, []

    for record in tqdm(records, desc=f"Checking {split}"):
        image_path = base_dir / "images" / f"{record.image_id}.jpg"
        mask_path = base_dir / "annotations" / "trimaps" / f"{record.image_id}.png"
        report = check_sample(record.image_id, image_path, mask_path)

        if report.warnings:
            warnings[record.image_id] = report.warnings
        if not report.is_usable:
            errors[record.image_id] = report.errors
            continue

        usable_ids.append(record.image_id)
        widths.append(report.width)
        heights.append(report.height)
        labels = remap_trimap(np.array(Image.open(mask_path)))
        pixel_counts += np.bincount(labels.ravel(), minlength=NUM_CLASSES)

    pixel_fractions = pixel_counts / pixel_counts.sum()
    return {
        "num_records": len(records),
        "num_usable": len(usable_ids),
        "species_counts": dict(Counter(r.species for r in records)),
        "num_breeds": len({r.breed for r in records}),
        "breed_counts": dict(sorted(Counter(r.breed for r in records).items())),
        "image_width": {"min": min(widths), "median": int(np.median(widths)), "max": max(widths)},
        "image_height": {"min": min(heights), "median": int(np.median(heights)), "max": max(heights)},
        "pixel_fraction": {name: round(float(f), 4) for name, f in zip(CLASS_NAMES, pixel_fractions)},
        "errors": errors,
        "warnings": warnings,
        "_usable_ids": usable_ids,
    }


def save_examples(base_dir: Path, image_ids: list[str], out_path: Path) -> None:
    """Save a grid: each row shows the original image and the image with its mask."""
    fig, axes = plt.subplots(len(image_ids), 2, figsize=(8, 3.2 * len(image_ids)))
    for row, image_id in enumerate(image_ids):
        image = np.array(Image.open(base_dir / "images" / f"{image_id}.jpg").convert("RGB"))
        trimap = np.array(Image.open(base_dir / "annotations" / "trimaps" / f"{image_id}.png"))
        labels = remap_trimap(trimap)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(image_id, fontsize=9)
        axes[row, 1].imshow(overlay_mask(image, labels))
        axes[row, 1].set_title("green = pet, yellow = boundary", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Oxford-IIIT Pet dataset.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--out-dir", default="artifacts/data_inspection", type=Path)
    parser.add_argument("--num-examples", default=6, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if not (args.data_dir / "annotations").is_dir():
        raise SystemExit(f"Dataset not found in {args.data_dir}. Run scripts/download_data.py first.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {split: inspect_split(args.data_dir, split) for split in SPLITS}

    # Leakage check: no image may appear in both the trainval and the test split.
    overlap = set(report["trainval"]["_usable_ids"]) & set(report["test"]["_usable_ids"])
    report["trainval_test_overlap"] = sorted(overlap)

    for split in SPLITS:
        info = report[split]
        logger.info(
            "%-8s records=%d usable=%d breeds=%d errors=%d warnings=%d",
            split, info["num_records"], info["num_usable"], info["num_breeds"],
            len(info["errors"]), len(info["warnings"]),
        )
        logger.info("%-8s pixel fractions: %s", split, info["pixel_fraction"])
    logger.info("Images in both trainval and test: %d", len(overlap))

    random.seed(args.seed)
    example_ids = random.sample(report["trainval"]["_usable_ids"], args.num_examples)
    save_examples(args.data_dir, example_ids, args.out_dir / "samples.png")

    for split in SPLITS:
        del report[split]["_usable_ids"]  # too long to be useful in the JSON file
    with open(args.out_dir / "data_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    logger.info("Saved report and examples to %s", args.out_dir)


if __name__ == "__main__":
    main()
