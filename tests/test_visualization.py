"""Tests for petseg.visualization."""

import numpy as np
import pytest

from petseg.data.annotations import BACKGROUND, PET
from petseg.visualization import overlay_mask


def test_overlay_keeps_shape_and_background():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    labels = np.full((4, 4), BACKGROUND, dtype=np.uint8)
    labels[0, 0] = PET

    result = overlay_mask(image, labels)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[1:, :], image[1:, :])  # background untouched
    assert not np.array_equal(result[0, 0], image[0, 0])  # pet pixel coloured


def test_overlay_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        overlay_mask(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((3, 4), dtype=np.uint8))
