"""Tests for petseg.data.duplicates."""

import numpy as np
from PIL import Image

from petseg.data.duplicates import dhash, find_near_duplicates


def save_gradient(path, size=(64, 48), flip=False):
    row = np.linspace(0, 255, size[0], dtype=np.uint8)
    pixels = np.tile(row[::-1] if flip else row, (size[1], 1))
    Image.fromarray(pixels).convert("RGB").save(path)


def test_resized_copy_has_same_hash(tmp_path):
    save_gradient(tmp_path / "a.png", size=(64, 48))
    save_gradient(tmp_path / "b.png", size=(128, 96))  # same picture, bigger
    assert dhash(tmp_path / "a.png") == dhash(tmp_path / "b.png")


def test_different_images_are_far_apart(tmp_path):
    save_gradient(tmp_path / "a.png")
    save_gradient(tmp_path / "b.png", flip=True)
    pairs = find_near_duplicates(
        ["a"], [dhash(tmp_path / "a.png")], ["b"], [dhash(tmp_path / "b.png")], max_distance=4
    )
    assert pairs == []


def test_find_near_duplicates_reports_distance():
    pairs = find_near_duplicates(["a"], [0b1011], ["b", "c"], [0b1011, 0b0000], max_distance=1)
    assert pairs == [("a", "b", 0)]