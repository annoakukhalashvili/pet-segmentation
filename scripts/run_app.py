"""Launch the Gradio demo locally.

Usage:
    python scripts/run_app.py
    python scripts/run_app.py --weights artifacts/weights/petseg_deeplabv3_mbv3.pt --device cpu
Then open http://127.0.0.1:7860 in a browser.
"""

import argparse
import logging
from pathlib import Path

from petseg.app.gradio_app import build_app
from petseg.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pet segmentation demo.")
    parser.add_argument("--weights", default="artifacts/weights/petseg_deeplabv3_mbv3.pt", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(
            f"Weights not found: {args.weights}\n"
            "Create them with scripts/export_weights.py or download them with scripts/download_weights.py."
        )
    predictor = Predictor(args.weights, device=args.device)
    logger.info("Loaded %s on %s", predictor.meta.get("experiment"), predictor.device)
    build_app(predictor).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
