"""Tests for model loading, the Predictor, ONNX export and the app logic (tiny model, no downloads)."""

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.app.gradio_app import segment_image
from petseg.data.annotations import NUM_CLASSES
from petseg.inference.onnx_export import LogitsOnly, create_session, export_onnx
from petseg.inference.predictor import Predictor
from petseg.models.checkpoint import export_weights, load_model
from petseg.models.deeplab import build_model


@pytest.fixture
def training_checkpoint(tmp_path):
    """A file shaped like the trainer's best.pt."""
    model = build_model(NUM_CLASSES, pretrained=False)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "epoch": 3, "best_score": 0.5,
                "config": {"image_size": 64, "name": "tiny"}}, path)
    return path


@pytest.fixture
def weights_file(training_checkpoint, tmp_path):
    out = tmp_path / "weights.pt"
    export_weights(training_checkpoint, out)
    return out


def test_load_model_from_both_formats(training_checkpoint, weights_file):
    for path in (training_checkpoint, weights_file):
        model, meta = load_model(path)
        assert meta["image_size"] == 64
        assert not model.training  # eval mode


def test_predictor_output_matches_original_size(weights_file):
    predictor = Predictor(weights_file, device="cpu")
    result = predictor.predict(Image.new("RGB", (50, 70)))  # PIL size is (width, height)
    assert result.labels.shape == (70, 50)
    assert set(np.unique(result.labels)) <= {0, 1, 2}
    assert sum(result.coverage.values()) == pytest.approx(100, abs=0.1)


def test_predictor_accepts_grayscale_and_rejects_tiny_images(weights_file):
    predictor = Predictor(weights_file, device="cpu")
    assert predictor.predict(Image.new("L", (40, 40))).labels.shape == (40, 40)
    with pytest.raises(ValueError, match="too small"):
        predictor.predict(Image.new("RGB", (10, 10)))


def test_segment_image_returns_overlay_mask_and_details(weights_file):
    predictor = Predictor(weights_file, device="cpu")
    overlay, mask, details = segment_image(predictor, Image.new("RGB", (64, 48)))
    assert overlay.shape == (48, 64, 3) and mask.shape == (48, 64, 3)
    assert {"coverage_percent", "pet_confidence", "latency_ms"} <= set(details)
    with pytest.raises(ValueError):
        segment_image(predictor, None)


def test_onnx_export_matches_pytorch(weights_file, tmp_path):
    model, _ = load_model(weights_file)
    onnx_path = export_onnx(model, tmp_path / "model.onnx", image_size=64)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = LogitsOnly(model)(x).numpy()
    actual = create_session(onnx_path, num_threads=1).run(None, {"image": x.numpy()})[0]
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-3
