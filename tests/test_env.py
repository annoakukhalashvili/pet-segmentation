"""Tests for petseg.env."""

import pytest
import torch

from petseg.env import collect_env_info, select_device


def test_auto_returns_a_torch_device():
    assert isinstance(select_device("auto"), torch.device)


def test_cpu_is_always_available():
    assert select_device("cpu") == torch.device("cpu")


def test_unknown_device_raises_value_error():
    with pytest.raises(ValueError):
        select_device("tpu")


def test_env_info_has_expected_keys():
    info = collect_env_info()
    for key in ("python", "torch", "mps_available", "selected_device"):
        assert key in info