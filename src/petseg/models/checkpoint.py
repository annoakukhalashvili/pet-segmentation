"""Loading trained models and exporting compact weight files."""

from pathlib import Path

import torch
from torch import nn

from petseg.data.annotations import CLASS_NAMES, NUM_CLASSES
from petseg.models.deeplab import build_model


def load_model(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Load a model from a training checkpoint (best.pt) or an exported weights file.

    Returns:
        (model in eval mode on `device`, metadata dict with at least "image_size").
    """
    state = torch.load(path, map_location="cpu", weights_only=True)  # weights_only: never runs pickled code
    if "state_dict" in state:  # exported weights file
        state_dict, meta = state["state_dict"], dict(state.get("meta", {}))
    elif "model" in state:  # full training checkpoint
        state_dict = state["model"]
        meta = {"image_size": state["config"]["image_size"], "experiment": state["config"]["name"]}
    else:
        raise ValueError(f"{path} is neither a training checkpoint nor an exported weights file")

    model = build_model(NUM_CLASSES, pretrained=False)
    model.load_state_dict(state_dict)  # strict: fails loudly if the architecture does not match
    meta.setdefault("image_size", 320)
    return model.to(device).eval(), meta


def export_weights(checkpoint_path: Path, out_path: Path) -> dict:
    """Save only the model weights + metadata (no optimizer state): smaller, inference-only file."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = {
        "experiment": state["config"]["name"],
        "epoch": state["epoch"],
        "val_mean_iou": state["best_score"],
        "image_size": state["config"]["image_size"],
        "classes": list(CLASS_NAMES),
        "architecture": "deeplabv3_mobilenet_v3_large",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state["model"], "meta": meta}, out_path)
    return meta
