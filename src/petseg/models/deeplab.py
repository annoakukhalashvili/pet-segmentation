"""DeepLabV3 with a MobileNetV3-Large backbone (torchvision)."""

import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from petseg.data.annotations import BACKGROUND, NUM_CLASSES, PET

# Weights trained on a subset of COCO using the 20 Pascal VOC categories + background.
VOC_WEIGHTS = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
VOC_CATEGORIES = VOC_WEIGHTS.meta["categories"]
VOC_CAT = VOC_CATEGORIES.index("cat")  # 8
VOC_DOG = VOC_CATEGORIES.index("dog")  # 12


def build_pretrained_voc_model() -> nn.Module:
    """The pretrained model exactly as published: 21 VOC output classes.

    Used for the zero-shot baseline (E0). Downloads ~45 MB of weights the first
    time and caches them in ~/.cache/torch.
    """
    return deeplabv3_mobilenet_v3_large(weights=VOC_WEIGHTS)


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Model for fine-tuning: pretrained weights with a new final layer for our classes.

    Everything is reused except the last 1x1 convolution of the head, which maps
    256 features to 21 VOC classes. It is replaced by a fresh layer that maps to
    `num_classes` (background, pet, boundary). The auxiliary head, used only to help
    training of the original model, is removed to keep the model simple.

    pretrained=False builds a randomly initialised model without downloading anything
    (useful for tests).
    """
    if not pretrained:
        return deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False
        )

    model = build_pretrained_voc_model()
    final_layer = model.classifier[4]  # Conv2d(256, 21, kernel_size=1)
    model.classifier[4] = nn.Conv2d(final_layer.in_channels, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model


def voc_logits_to_pet_labels(logits: torch.Tensor) -> torch.Tensor:
    """Zero-shot mapping: VOC "cat" or "dog" -> PET, every other VOC class -> BACKGROUND.

    Args:
        logits: (B, 21, H, W) raw scores from the VOC model.
    Returns:
        (B, H, W) labels. The boundary class is never predicted.
    """
    voc_classes = logits.argmax(dim=1)
    is_pet = (voc_classes == VOC_CAT) | (voc_classes == VOC_DOG)
    return torch.where(is_pet, PET, BACKGROUND)
