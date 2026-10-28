"""Tests for the Dataset and transforms, using a tiny synthetic dataset."""

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.data.annotations import BACKGROUND, PET
from petseg.data.dataset import PetSegmentationDataset
from petseg.data.transforms import build_transforms, denormalize


@pytest.fixture
def tiny_dataset_dir(tmp_path):
    """One 80x60 image: a red pet rectangle on a blue background, with a matching trimap."""
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations" / "trimaps").mkdir(parents=True)

    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)  # blue background
    trimap = np.full((60, 80), 2, dtype=np.uint8)  # raw 2 = background
    trimap[15:45, 20:60] = 3  # raw 3 = boundary band
    trimap[17:43, 22:58] = 1  # raw 1 = pet
    image[trimap != 2] = (255, 0, 0)  # pet and boundary area is red

    Image.fromarray(image).save(tmp_path / "images" / "cat_1.jpg", quality=100)
    Image.fromarray(trimap).save(tmp_path / "annotations" / "trimaps" / "cat_1.png")
    return tmp_path


@pytest.mark.parametrize("train", [True, False])
def test_output_shapes_dtypes_and_labels(tiny_dataset_dir, train):
    dataset = PetSegmentationDataset(tiny_dataset_dir, ["cat_1"], build_transforms(train, image_size=32))
    image, mask = dataset[0]

    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert mask.shape == (32, 32) and mask.dtype == torch.int64
    assert set(mask.unique().tolist()) <= {0, 1, 2}  # nearest-neighbour keeps labels valid


def test_mask_stays_aligned_with_image_after_augmentation(tiny_dataset_dir):
    torch.manual_seed(0)
    dataset = PetSegmentationDataset(tiny_dataset_dir, ["cat_1"], build_transforms(train=True, image_size=64))

    for _ in range(10):  # several different random crops and flips
        image, mask = dataset[0]
        rgb = denormalize(image)
        is_red = rgb[0] > rgb[2]
        # Pet pixels must be red and background pixels blue, whatever the crop/flip was.
        assert is_red[mask == PET].float().mean() > 0.95
        assert (~is_red[mask == BACKGROUND]).float().mean() > 0.95


def test_eval_transform_is_deterministic(tiny_dataset_dir):
    dataset = PetSegmentationDataset(tiny_dataset_dir, ["cat_1"], build_transforms(train=False, image_size=32))
    image_1, mask_1 = dataset[0]
    image_2, mask_2 = dataset[0]
    assert torch.equal(image_1, image_2) and torch.equal(mask_1, mask_2)