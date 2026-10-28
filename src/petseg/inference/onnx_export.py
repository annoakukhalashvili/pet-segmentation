"""Export the model to ONNX and run it with ONNX Runtime."""

from pathlib import Path

import onnxruntime as ort
import torch
from torch import nn


class LogitsOnly(nn.Module):
    """torchvision segmentation models return a dict; ONNX export needs a plain tensor output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["out"]


def export_onnx(model: nn.Module, out_path: Path, image_size: int, opset: int = 17) -> Path:
    """Export with a fixed input shape (1, 3, image_size, image_size)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = LogitsOnly(model).cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        wrapper, (dummy,), str(out_path),
        input_names=["image"], output_names=["logits"], opset_version=opset, dynamo=False,
    )
    return out_path


def create_session(onnx_path: Path, num_threads: int = 4) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # operator fusion etc.
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
