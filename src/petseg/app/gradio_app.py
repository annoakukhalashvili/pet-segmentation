"""Gradio web interface: upload a pet photo, see the predicted segmentation."""

import gradio as gr
import numpy as np
from PIL import Image

from petseg.data.annotations import BOUNDARY, PET
from petseg.inference.predictor import Predictor
from petseg.visualization import overlay_mask

MASK_COLORS = {PET: (0, 200, 0), BOUNDARY: (255, 200, 0)}  # background stays black


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in MASK_COLORS.items():
        mask[labels == label] = color
    return mask


def segment_image(predictor: Predictor, image: Image.Image | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Core logic, independent of Gradio (so it can be unit-tested)."""
    if image is None:
        raise ValueError("Please upload an image first.")
    result = predictor.predict(image)
    overlay = overlay_mask(np.array(image.convert("RGB")), result.labels)
    details = {
        "coverage_percent": result.coverage,
        "pet_confidence": result.pet_confidence,
        "pet_detected": result.coverage["pet"] > 1.0,
        "latency_ms": result.latency_ms,
        "model": predictor.meta.get("experiment", "unknown"),
        "device": str(predictor.device),
    }
    return overlay, colorize_labels(result.labels), details


def build_app(predictor: Predictor) -> gr.Blocks:
    def run(image: Image.Image | None):
        try:
            return segment_image(predictor, image)
        except ValueError as error:  # invalid input: show a friendly message in the UI
            raise gr.Error(str(error)) from error
        except Exception as error:  # unexpected failure: still report it instead of crashing
            raise gr.Error(f"Inference failed: {error}") from error

    with gr.Blocks(title="Pet Segmentation") as demo:
        gr.Markdown(
            "# Pet Segmentation\n"
            "Upload a photo of a cat or dog. The model (DeepLabV3-MobileNetV3 fine-tuned on "
            "Oxford-IIIT Pet) labels every pixel: **green = pet**, **yellow = boundary**, rest = background."
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Input image")
            overlay_output = gr.Image(label="Prediction overlay")
        with gr.Row():
            mask_output = gr.Image(label="Predicted mask")
            details_output = gr.JSON(label="Details")
        gr.Button("Segment", variant="primary").click(
            run, inputs=image_input, outputs=[overlay_output, mask_output, details_output]
        )
    return demo
