"""Show the near-duplicate pairs found by make_splits.py side by side.

Usage:
    python scripts/show_duplicates.py
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise near-duplicate image pairs.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--duplicates", default="splits/duplicates.json", type=Path)
    parser.add_argument("--out", default="artifacts/data_inspection/duplicates.png", type=Path)
    args = parser.parse_args()

    with open(args.duplicates, encoding="utf-8") as file:
        duplicates = json.load(file)
    pairs = [(key, p["a"], p["b"], p["distance"]) for key, items in duplicates.items() for p in items]
    if not pairs:
        logger.info("No near-duplicate pairs to show.")
        return

    fig, axes = plt.subplots(len(pairs), 2, figsize=(6, 2.6 * len(pairs)), squeeze=False)
    for row, (key, id_a, id_b, distance) in enumerate(pairs):
        split_a, split_b = key.split("-")
        for col, (split, image_id) in enumerate(((split_a, id_a), (split_b, id_b))):
            with Image.open(args.data_dir / "images" / f"{image_id}.jpg") as image:
                axes[row, col].imshow(image.convert("RGB"))
            axes[row, col].set_title(f"{split}: {image_id} (d={distance})", fontsize=8)
            axes[row, col].axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    plt.close(fig)
    logger.info("Saved %d pairs to %s", len(pairs), args.out)


if __name__ == "__main__":
    main()
