"""Image + mask transforms for training and evaluation."""

import torch
from torchvision.transforms import v2

# The pretrained backbone was trained on images normalised with these ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, image_size: int = 320) -> v2.Compose:
    """Build the transform pipeline.

    torchvision's v2 transforms accept (image, mask) together and apply the SAME random
    crop and flip to both, so the mask stays aligned with the image. Masks are resized
    with nearest-neighbour interpolation automatically, so labels stay 0, 1 or 2.

    Train: random crop of 50-100% of the image, resized to image_size, plus random flip.
    Eval:  plain resize to image_size (deterministic, no randomness).
    """
    size = (image_size, image_size)
    if train:
        geometric = [
            v2.RandomResizedCrop(size=size, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    else:
        geometric = [v2.Resize(size=size, antialias=True)]

    return v2.Compose(
        [
            v2.ToImage(),  # PIL image -> tensor of shape (3, H, W), uint8
            *geometric,
            v2.ToDtype(torch.float32, scale=True),  # images only: uint8 0-255 -> float 0.0-1.0
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # images only
        ]
    )


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Undo Normalize for display: returns a float tensor (3, H, W) with values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)