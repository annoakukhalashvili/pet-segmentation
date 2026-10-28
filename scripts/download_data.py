"""Download the Oxford-IIIT Pet dataset (images + trimaps) using torchvision.

Usage:
    python scripts/download_data.py --root data

The files end up in data/oxford-iiit-pet/ (about 800 MB).
"""

import argparse
import logging

from torchvision.datasets import OxfordIIITPet

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", default="data", help="folder to download into")
    args = parser.parse_args()

    for split in ("trainval", "test"):
        dataset = OxfordIIITPet(root=args.root, split=split, target_types="segmentation", download=True)
        logger.info("Split %-8s ready: %d samples", split, len(dataset))


if __name__ == "__main__":
    main()
