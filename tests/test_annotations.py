"""Tests for petseg.data.annotations."""

import numpy as np
import pytest

from petseg.data.annotations import (
    BACKGROUND,
    BOUNDARY,
    PET,
    breed_from_image_id,
    parse_split_file,
    remap_trimap,
)


def test_parse_split_file_skips_comments_and_reads_fields(tmp_path):
    split_file = tmp_path / "trainval.txt"
    split_file.write_text(
        "#Image CLASS-ID SPECIES BREED ID\n"
        "Abyssinian_100 1 1 1\n"
        "\n"
        "american_pit_bull_terrier_12 3 2 2\n"
    )
    records = parse_split_file(split_file)

    assert len(records) == 2
    assert records[0].image_id == "Abyssinian_100"
    assert records[0].species == "cat"
    assert records[0].breed == "Abyssinian"
    assert records[1].species == "dog"
    assert records[1].breed == "american_pit_bull_terrier"
    assert records[1].class_id == 3


def test_parse_split_file_rejects_malformed_line(tmp_path):
    split_file = tmp_path / "bad.txt"
    split_file.write_text("Abyssinian_100 1 1\n")
    with pytest.raises(ValueError):
        parse_split_file(split_file)


def test_breed_from_image_id_rejects_missing_number():
    with pytest.raises(ValueError):
        breed_from_image_id("Abyssinian")


def test_remap_trimap_maps_each_raw_value():
    trimap = np.array([[1, 2], [3, 1]], dtype=np.uint8)
    expected = np.array([[PET, BACKGROUND], [BOUNDARY, PET]], dtype=np.uint8)
    np.testing.assert_array_equal(remap_trimap(trimap), expected)


def test_remap_trimap_rejects_unexpected_values():
    trimap = np.array([[1, 0], [255, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="Unexpected trimap values"):
        remap_trimap(trimap)
