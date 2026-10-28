"""Finding near-duplicate images with a perceptual hash (dHash)."""

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(image_path: Path, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash.

    The image is shrunk to a tiny grayscale version, and each bit records whether a
    pixel is brighter than its right neighbour. Resized, re-compressed or slightly
    edited copies of a photo get the same or a very similar hash.
    """
    with Image.open(image_path) as image:
        small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def find_near_duplicates(
    ids_a: list[str], hashes_a: list[int], ids_b: list[str], hashes_b: list[int], max_distance: int
) -> list[tuple[str, str, int]]:
    """Return (id_a, id_b, distance) for every pair whose hashes differ in <= max_distance bits."""
    array_b = np.array(hashes_b, dtype=np.uint64)
    pairs = []
    for id_a, hash_a in zip(ids_a, hashes_a):
        distances = np.bitwise_count(np.uint64(hash_a) ^ array_b)  # Hamming distance
        for index in np.nonzero(distances <= max_distance)[0]:
            pairs.append((id_a, ids_b[index], int(distances[index])))
    return pairs