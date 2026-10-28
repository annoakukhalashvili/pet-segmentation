"""Tests for petseg.data.splits."""

import pytest

from petseg.data.annotations import PetRecord
from petseg.data.splits import read_split, stratified_split, write_split


def make_records(num_breeds=3, per_breed=20):
    return [
        PetRecord(image_id=f"breed{b}_{i}", class_id=b, species="cat", breed=f"breed{b}")
        for b in range(1, num_breeds + 1)
        for i in range(1, per_breed + 1)
    ]


def test_split_has_no_overlap_and_keeps_every_image():
    records = make_records()
    train, val = stratified_split(records, val_fraction=0.25, seed=42)
    assert not set(train) & set(val)
    assert len(train) + len(val) == len(records)


def test_split_is_stratified_per_breed():
    train, val = stratified_split(make_records(per_breed=20), val_fraction=0.25, seed=42)
    for breed in ("breed1", "breed2", "breed3"):
        assert sum(i.startswith(breed + "_") for i in val) == 5  # 25% of 20


def test_split_is_reproducible_and_seed_dependent():
    records = make_records()
    assert stratified_split(records, 0.25, seed=42) == stratified_split(records, 0.25, seed=42)
    assert stratified_split(records, 0.25, seed=42) != stratified_split(records, 0.25, seed=7)


def test_split_does_not_depend_on_input_order():
    records = make_records()
    assert stratified_split(records, 0.25, 42) == stratified_split(records[::-1], 0.25, 42)


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        stratified_split(make_records(), val_fraction=1.5, seed=42)


def test_write_then_read_round_trip(tmp_path):
    ids = ["a_1", "b_2", "c_3"]
    write_split(tmp_path / "splits" / "train.txt", ids)
    assert read_split(tmp_path / "splits" / "train.txt") == ids