"""Download the trained weights from the GitHub release.

Usage:
    python scripts/download_weights.py
"""

import argparse
import logging
import urllib.request
from pathlib import Path

from petseg.models.checkpoint import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URL = (
    "https://github.com/annoakukhalashvili/pet-segmentation/releases/download/v1.0/petseg_deeplabv3_mbv3.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download trained weights.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="artifacts/weights/petseg_deeplabv3_mbv3.pt", type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", args.url)
    urllib.request.urlretrieve(args.url, args.out)
    _, meta = load_model(args.out)  # verifies the file is a valid model
    logger.info("Saved %s (%s, val mean IoU %s)", args.out, meta.get("experiment"), meta.get("val_mean_iou"))


if __name__ == "__main__":
    main()
