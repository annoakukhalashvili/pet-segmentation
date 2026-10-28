"""Drawing segmentation masks on top of images."""

import numpy as np

from petseg.data.annotations import BACKGROUND, BOUNDARY, PET

# RGB colour for each label. Background is not drawn.
LABEL_COLORS = {
    PET: (0, 200, 0),  # green
    BOUNDARY: (255, 200, 0),  # yellow
}


def overlay_mask(image: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend coloured labels onto an RGB image.

    Args:
        image: uint8 array of shape (H, W, 3).
        labels: integer array of shape (H, W) with values BACKGROUND, PET or BOUNDARY.
        alpha: opacity of the colour, between 0 (invisible) and 1 (solid).

    Returns:
        A new uint8 array of shape (H, W, 3). Background pixels are unchanged.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if labels.shape != image.shape[:2]:
        raise ValueError(f"labels shape {labels.shape} does not match image {image.shape[:2]}")

    result = image.astype(np.float32)
    for label, color in LABEL_COLORS.items():
        region = labels == label
        result[region] = (1 - alpha) * result[region] + alpha * np.array(color, dtype=np.float32)

    assert BACKGROUND not in LABEL_COLORS
    return result.round().astype(np.uint8)
