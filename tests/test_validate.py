"""Tests for petseg.data.validate, using small synthetic files."""

import numpy as np
from PIL import Image

from petseg.data.validate import check_sample


def make_image(path, size=(8, 6), mode="RGB"):
    Image.new(mode, size).save(path)


def make_mask(path, size=(8, 6), value=1):
    width, height = size
    Image.fromarray(np.full((height, width), value, dtype=np.uint8)).save(path)


def test_valid_sample_has_no_problems(tmp_path):
    make_image(tmp_path / "a.jpg")
    make_mask(tmp_path / "a.png")
    report = check_sample("a", tmp_path / "a.jpg", tmp_path / "a.png")
    assert report.is_usable
    assert report.warnings == []
    assert (report.width, report.height) == (8, 6)


def test_missing_files_are_errors(tmp_path):
    report = check_sample("a", tmp_path / "a.jpg", tmp_path / "a.png")
    assert not report.is_usable
    assert len(report.errors) == 2


def test_corrupted_image_is_an_error(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"this is not an image")
    make_mask(tmp_path / "a.png")
    report = check_sample("a", tmp_path / "a.jpg", tmp_path / "a.png")
    assert not report.is_usable
    assert "unreadable image" in report.errors[0]


def test_size_mismatch_is_an_error(tmp_path):
    make_image(tmp_path / "a.jpg", size=(8, 6))
    make_mask(tmp_path / "a.png", size=(4, 6))
    report = check_sample("a", tmp_path / "a.jpg", tmp_path / "a.png")
    assert not report.is_usable
    assert "size mismatch" in report.errors[0]


def test_unexpected_mask_value_is_an_error(tmp_path):
    make_image(tmp_path / "a.jpg")
    make_mask(tmp_path / "a.png", value=7)
    report = check_sample("a", tmp_path / "a.jpg", tmp_path / "a.png")
    assert not report.is_usable
    assert "unexpected mask values" in report.errors[0]


def test_grayscale_image_is_only_a_warning(tmp_path):
    make_image(tmp_path / "a.jpg", mode="L")
    make_mask(tmp_path / "a.png")
    report = check_sample("a", tmp_path / "a.jpg", tmp_path / "a.png")
    assert report.is_usable
    assert "will be converted to RGB" in report.warnings[0]
