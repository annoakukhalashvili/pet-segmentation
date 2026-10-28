"""Tests for petseg.models.deeplab. No weights are downloaded."""

import torch

from petseg.data.annotations import BACKGROUND, NUM_CLASSES, PET
from petseg.models.deeplab import VOC_CAT, VOC_DOG, build_model, voc_logits_to_pet_labels


def test_voc_class_indices():
    assert (VOC_CAT, VOC_DOG) == (8, 12)


def test_model_output_shape_matches_input():
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))["out"]
    assert output.shape == (2, NUM_CLASSES, 64, 64)  # one score per class for every pixel


def test_voc_logits_to_pet_labels():
    logits = torch.zeros(1, 21, 2, 2)
    logits[0, VOC_CAT, 0, 0] = 1.0  # cat   -> pet
    logits[0, VOC_DOG, 0, 1] = 1.0  # dog   -> pet
    logits[0, 15, 1, 0] = 1.0  # person -> background
    # pixel (1, 1): all zeros -> argmax is class 0 (VOC background) -> background
    expected = torch.tensor([[[PET, PET], [BACKGROUND, BACKGROUND]]])
    assert torch.equal(voc_logits_to_pet_labels(logits), expected)
