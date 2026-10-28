"""End-to-end training tests on a tiny synthetic dataset (CPU, no downloads)."""

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.config import TrainConfig
from petseg.data.splits import write_split
from petseg.training.trainer import Trainer, median_frequency_weights


@pytest.fixture
def tiny_project(tmp_path):
    """6 random images with simple trimaps, split 4 train / 2 val."""
    data_dir = tmp_path / "data"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "annotations" / "trimaps").mkdir(parents=True)
    rng = np.random.default_rng(0)
    ids = [f"cat_{i}" for i in range(6)]
    for image_id in ids:
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(data_dir / "images" / f"{image_id}.jpg")
        trimap = np.full((48, 48), 2, dtype=np.uint8)
        trimap[10:38, 10:38] = 3
        trimap[12:36, 12:36] = 1
        Image.fromarray(trimap).save(data_dir / "annotations" / "trimaps" / f"{image_id}.png")
    write_split(tmp_path / "splits" / "train.txt", ids[:4])
    write_split(tmp_path / "splits" / "val.txt", ids[4:])
    return tmp_path


def make_config(root, **overrides):
    values = dict(
        name="tiny", data_dir=str(root / "data"), splits_dir=str(root / "splits"),
        output_dir=str(root / "out"), image_size=32, batch_size=2, num_workers=0,
        device="cpu", pretrained=False, epochs=1,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_fit_writes_checkpoints_and_history(tiny_project):
    trainer = Trainer(make_config(tiny_project))
    best = trainer.fit()
    out = tiny_project / "out" / "tiny"
    for name in ("best.pt", "last.pt", "history.csv", "best_val_metrics.json", "config.json"):
        assert (out / name).is_file(), name
    assert best["epoch"] == 1
    assert 0.0 <= best["val_mean_iou"] <= 1.0


def test_resume_continues_from_next_epoch(tiny_project):
    Trainer(make_config(tiny_project, epochs=1)).fit()
    trainer = Trainer(make_config(tiny_project, epochs=2))
    trainer.resume()
    assert trainer.start_epoch == 2
    trainer.fit()
    assert [r["epoch"] for r in trainer.history] == [1, 2]


def test_frozen_backbone_does_not_change(tiny_project):
    trainer = Trainer(make_config(tiny_project, freeze_backbone=True))
    backbone_before = {k: v.clone() for k, v in trainer.model.backbone.state_dict().items()}
    head_before = {k: v.clone() for k, v in trainer.model.classifier.state_dict().items()}
    trainer.train_one_epoch(1)

    # Backbone weights AND BatchNorm running statistics must be untouched.
    for key, value in trainer.model.backbone.state_dict().items():
        assert torch.equal(value, backbone_before[key]), key
    # The head must have learned something.
    assert any(not torch.equal(v, head_before[k]) for k, v in trainer.model.classifier.state_dict().items())


def test_median_frequency_weights_favour_rare_classes(tiny_project):
    weights = median_frequency_weights(tiny_project / "data", ["cat_0"])
    background, pet, boundary = weights.tolist()
    # In the fixture: background 1520 px, pet 576 px, boundary 208 px -> boundary is rarest.
    assert boundary > pet > background
    assert pet == pytest.approx(1.0)  # pet has the median frequency
