"""Export the best experiment (highest VALIDATION mean IoU) to a compact weights file.

Usage:
    python scripts/export_weights.py
"""

import argparse
import json
import logging
from pathlib import Path

from petseg.models.checkpoint import export_weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS = ("e1_head_only", "e2_full_finetune", "e3_weighted_loss")


def best_experiment(experiments_dir: Path) -> str:
    """Pick the experiment with the best validation mean IoU (never the test set)."""
    scores = {}
    for name in EXPERIMENTS:
        metrics_file = experiments_dir / name / "best_val_metrics.json"
        if metrics_file.is_file():
            scores[name] = json.loads(metrics_file.read_text())["val_mean_iou"]
    if not scores:
        raise SystemExit(f"No experiment results found in {experiments_dir}")
    for name, score in scores.items():
        logger.info("%-18s val mean IoU %.4f", name, score)
    return max(scores, key=scores.get)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model weights for inference.")
    parser.add_argument("--experiments-dir", default="artifacts/experiments", type=Path)
    parser.add_argument("--out", default="artifacts/weights/petseg_deeplabv3_mbv3.pt", type=Path)
    args = parser.parse_args()

    name = best_experiment(args.experiments_dir)
    meta = export_weights(args.experiments_dir / name / "best.pt", args.out)
    size_mb = args.out.stat().st_size / 1e6
    logger.info("Exported %s (epoch %d) to %s (%.1f MB)", name, meta["epoch"], args.out, size_mb)


if __name__ == "__main__":
    main()
