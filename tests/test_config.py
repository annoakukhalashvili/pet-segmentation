"""Tests for petseg.config."""

import pytest

from petseg.config import load_config


def test_load_config_applies_defaults(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: demo\nfreeze_backbone: true\n")
    config = load_config(path)
    assert config.name == "demo"
    assert config.freeze_backbone is True
    assert config.image_size == 320  # default


def test_unknown_key_raises(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: demo\nlearning_rate: 0.1\n")  # typo-like key
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_invalid_class_weighting_raises(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: demo\nclass_weighting: magic\n")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("name", ["e1_head_only", "e2_full_finetune", "e3_weighted_loss"])
def test_project_configs_are_valid(name):
    config = load_config(f"configs/{name}.yaml")
    assert config.name == name
