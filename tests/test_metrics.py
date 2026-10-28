"""Tests for petseg.evaluation.metrics, checked against values computed by hand."""

import math

import pytest
import torch

from petseg.data.annotations import BACKGROUND, BOUNDARY, PET
from petseg.evaluation.metrics import SegmentationMetrics


def test_perfect_prediction_scores_one():
    target = torch.tensor([[0, 1], [2, 1]])
    metrics = SegmentationMetrics()
    metrics.update(target.clone(), target)
    results = metrics.compute()
    assert results["mean_iou"] == 1.0
    assert results["mean_dice"] == 1.0
    assert results["pixel_accuracy"] == 1.0


def test_hand_computed_example():
    # 4 pixels. Truth: bg, bg, pet, pet. Prediction: bg, pet, pet, pet.
    target = torch.tensor([BACKGROUND, BACKGROUND, PET, PET])
    prediction = torch.tensor([BACKGROUND, PET, PET, PET])
    metrics = SegmentationMetrics()
    metrics.update(prediction, target)

    iou = metrics.iou_per_class()
    # background: TP=1, FP=0, FN=1 -> 1/2.   pet: TP=2, FP=1, FN=0 -> 2/3.
    assert iou[BACKGROUND].item() == pytest.approx(1 / 2)
    assert iou[PET].item() == pytest.approx(2 / 3)
    assert math.isnan(iou[BOUNDARY].item())  # boundary absent everywhere: undefined, not 0

    dice = metrics.dice_per_class()
    assert dice[BACKGROUND].item() == pytest.approx(2 / 3)  # 2*1 / (2*1 + 0 + 1)
    assert dice[PET].item() == pytest.approx(4 / 5)  # 2*2 / (2*2 + 1 + 0)

    results = metrics.compute()
    assert results["mean_iou"] == pytest.approx((1 / 2 + 2 / 3) / 2, abs=1e-4)  # NaN is skipped
    assert results["iou_boundary"] is None
    assert results["pixel_accuracy"] == 0.75


def test_foreground_iou_merges_pet_and_boundary():
    # Pet and boundary are swapped, but the foreground region is exactly right.
    target = torch.tensor([BACKGROUND, BOUNDARY, PET, PET])
    prediction = torch.tensor([BACKGROUND, PET, PET, BOUNDARY])
    metrics = SegmentationMetrics()
    metrics.update(prediction, target)
    assert metrics.foreground_iou() == 1.0
    assert metrics.compute()["iou_boundary"] == 0.0


def test_updates_accumulate_over_batches():
    metrics = SegmentationMetrics()
    metrics.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    metrics.update(torch.tensor([1, 1]), torch.tensor([0, 1]))
    assert metrics.confusion.sum().item() == 4
    assert metrics.confusion[0, 1].item() == 1  # one background pixel predicted as pet


def test_invalid_inputs_raise():
    metrics = SegmentationMetrics()
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.update(torch.zeros(3, dtype=torch.long), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        metrics.update(torch.tensor([5]), torch.tensor([0]))
