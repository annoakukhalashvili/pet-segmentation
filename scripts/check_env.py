"""Print environment information and run a tiny computation on the selected device.

Usage:
    python scripts/check_env.py
"""

import logging

import torch

from petseg.env import collect_env_info, select_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    for key, value in collect_env_info().items():
        logger.info("%-16s %s", key, value)

    device = select_device()
    x = torch.randn(256, 256, device=device)
    y = x @ x  # matrix multiplication: a small smoke test on the device
    logger.info("Matrix multiply on %s OK, result shape %s", device, tuple(y.shape))


if __name__ == "__main__":
    main()