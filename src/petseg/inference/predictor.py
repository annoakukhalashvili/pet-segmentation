"""Run the trained model on a single new image."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from petseg.data.annotations import CLASS_NAMES, PET
from petseg.data.transforms import build_transforms
from petseg.env import select_device
from petseg.models.checkpoint import load_model

MIN_IMAGE_SIDE = 32


@dataclass
class Prediction:
    labels: np.ndarray  # (H, W) uint8 at the ORIGINAL image size: 0 background, 1 pet, 2 boundary
    coverage: dict[str, float]  # percentage of the image covered by each class
    pet_confidence: float  # mean softmax probability over pixels predicted as pet (0-1)
    latency_ms: float  # model + pre/post-processing time


class Predictor:
    """Loads the model once, then segments any number of images."""

    def __init__(self, weights_path: Path, device: str = "auto"):
        self.device = select_device(device)
        self.model, self.meta = load_model(weights_path, self.device)
        self.transform = build_transforms(train=False, image_size=self.meta["image_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL image.")
        width, height = image.size
        if min(width, height) < MIN_IMAGE_SIDE:
            raise ValueError(f"Image is too small ({width}x{height}); each side must be at least {MIN_IMAGE_SIDE} px.")

        start = time.perf_counter()
        rgb = image.convert("RGB")  # handles grayscale, RGBA (PNG with transparency), CMYK
        batch = self.transform(rgb).unsqueeze(0).to(self.device)  # (1, 3, S, S)
        logits = self.model(batch)["out"]  # (1, 3, S, S)
        # Resize the scores (not the labels) back to the original size for smooth edges.
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        probabilities = logits.softmax(dim=1)[0]
        labels = probabilities.argmax(dim=0)
        latency_ms = (time.perf_counter() - start) * 1000

        labels_np = labels.cpu().numpy().astype(np.uint8)
        counts = np.bincount(labels_np.ravel(), minlength=len(CLASS_NAMES))
        coverage = {name: round(100 * c / labels_np.size, 2) for name, c in zip(CLASS_NAMES, counts)}
        pet_pixels = labels == PET
        pet_confidence = probabilities[PET][pet_pixels].mean().item() if pet_pixels.any() else 0.0
        return Prediction(labels_np, coverage, round(pet_confidence, 4), round(latency_ms, 1))
