"""Train one experiment.

Usage:
    python scripts/train.py --config configs/e2_full_finetune.yaml
    python scripts/train.py --config configs/e2_full_finetune.yaml --resume
    python scripts/train.py --config configs/e1_head_only.yaml --smoke-test   # quick pipeline check

Outputs go to artifacts/experiments/<name>/:
    best.pt                 checkpoint with the best validation mean IoU
    last.pt                 latest checkpoint (used by --resume)
    history.csv             loss and validation metrics for every epoch
    best_val_metrics.json   validation metrics of the best epoch
    config.json             the exact settings used
"""

import argparse
import logging
from pathlib import Path

from petseg.config import load_config
from petseg.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    parser.add_argument("--smoke-test", action="store_true", help="2 batches, 1 epoch: checks the pipeline runs")
    args = parser.parse_args()

    config = load_config(args.config)
    max_batches = None
    if args.smoke_test:
        config.name = f"{config.name}_smoke"
        config.epochs = 1
        max_batches = 2

    trainer = Trainer(config, max_batches=max_batches)
    if args.resume:
        trainer.resume()
    trainer.fit()


if __name__ == "__main__":
    main()
