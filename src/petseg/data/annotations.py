"""Parsing of Oxford-IIIT Pet annotation files and trimap label remapping."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Values stored in the raw trimap PNG files.
RAW_PET = 1
RAW_BACKGROUND = 2
RAW_BOUNDARY = 3

# Labels used by our model. Background is 0 by convention.
BACKGROUND = 0
PET = 1
BOUNDARY = 2
CLASS_NAMES = ("background", "pet", "boundary")
NUM_CLASSES = len(CLASS_NAMES)

SPECIES_NAMES = {1: "cat", 2: "dog"}


@dataclass(frozen=True)
class PetRecord:
    """One line of an Oxford-IIIT Pet split file."""

    image_id: str  # file name without extension, e.g. "Abyssinian_100"
    class_id: int  # breed index from 1 to 37
    species: str  # "cat" or "dog"
    breed: str  # e.g. "Abyssinian"


def breed_from_image_id(image_id: str) -> str:
    """Extract the breed name: "american_pit_bull_terrier_12" -> "american_pit_bull_terrier"."""
    breed, separator, number = image_id.rpartition("_")
    if not separator or not breed or not number.isdigit():
        raise ValueError(f"Unexpected image id format: {image_id!r}")
    return breed


def parse_split_file(path: Path) -> list[PetRecord]:
    """Parse a split file such as annotations/trainval.txt.

    Each non-comment line has four fields: image_id class_id species_id breed_id.
    Lines starting with '#' are comments.
    """
    records = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected 4 fields, got {len(fields)}")

            image_id, class_id, species_id, _breed_id = fields
            species_id = int(species_id)
            if species_id not in SPECIES_NAMES:
                raise ValueError(f"{path}:{line_number}: unknown species id {species_id}")

            records.append(
                PetRecord(
                    image_id=image_id,
                    class_id=int(class_id),
                    species=SPECIES_NAMES[species_id],
                    breed=breed_from_image_id(image_id),
                )
            )
    return records


def remap_trimap(trimap: np.ndarray) -> np.ndarray:
    """Convert raw trimap values (1, 2, 3) to model labels (PET, BACKGROUND, BOUNDARY).

    Raises:
        ValueError: if the trimap contains any value other than 1, 2 or 3.
    """
    unexpected = ~np.isin(trimap, [RAW_PET, RAW_BACKGROUND, RAW_BOUNDARY])
    if unexpected.any():
        bad_values = np.unique(trimap[unexpected]).tolist()
        raise ValueError(f"Unexpected trimap values: {bad_values}")

    labels = np.empty(trimap.shape, dtype=np.uint8)
    labels[trimap == RAW_PET] = PET
    labels[trimap == RAW_BACKGROUND] = BACKGROUND
    labels[trimap == RAW_BOUNDARY] = BOUNDARY
    return labels
