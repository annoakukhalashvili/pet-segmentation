"""Creating, saving and loading reproducible dataset splits."""

import random
from collections import defaultdict
from pathlib import Path

from petseg.data.annotations import PetRecord


def stratified_split(
    records: list[PetRecord], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split records into train and validation image ids, keeping breed proportions.

    Every breed contributes the same fraction of its images to validation, so both
    splits contain all 37 breeds in roughly the same proportions.

    Returns:
        (train_ids, val_ids), each sorted alphabetically.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    ids_by_breed: dict[int, list[str]] = defaultdict(list)
    for record in records:
        ids_by_breed[record.class_id].append(record.image_id)

    rng = random.Random(seed)  # own generator: not affected by other random calls
    train_ids, val_ids = [], []
    for class_id in sorted(ids_by_breed):
        ids = sorted(ids_by_breed[class_id])  # sort first so the result never depends on input order
        rng.shuffle(ids)
        num_val = round(len(ids) * val_fraction)
        val_ids.extend(ids[:num_val])
        train_ids.extend(ids[num_val:])

    return sorted(train_ids), sorted(val_ids)


def write_split(path: Path, image_ids: list[str]) -> None:
    """Save one image id per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")


def read_split(path: Path) -> list[str]:
    """Load image ids saved by write_split."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]