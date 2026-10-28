"""Validation of individual image/trimap pairs."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from petseg.data.annotations import RAW_BACKGROUND, RAW_BOUNDARY, RAW_PET

EXPECTED_TRIMAP_VALUES = {RAW_PET, RAW_BACKGROUND, RAW_BOUNDARY}


@dataclass
class SampleReport:
    """Result of checking one sample.

    errors: problems that make the sample unusable (it must be excluded).
    warnings: problems we can fix automatically (e.g. converting to RGB).
    """

    image_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def is_usable(self) -> bool:
        return not self.errors


def check_sample(image_id: str, image_path: Path, mask_path: Path) -> SampleReport:
    """Check that an image and its trimap exist, decode, match in size and have valid labels."""
    report = SampleReport(image_id=image_id)

    if not image_path.is_file():
        report.errors.append(f"missing image file: {image_path.name}")
    if not mask_path.is_file():
        report.errors.append(f"missing mask file: {mask_path.name}")
    if report.errors:
        return report

    try:
        with Image.open(image_path) as image:
            image.load()  # forces a full decode, which catches truncated files
            report.width, report.height = image.size
            image_mode = image.mode
    except (OSError, SyntaxError) as error:
        report.errors.append(f"unreadable image: {error}")
        return report

    try:
        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image)
    except (OSError, SyntaxError) as error:
        report.errors.append(f"unreadable mask: {error}")
        return report

    if image_mode != "RGB":
        report.warnings.append(f"image mode is {image_mode}, will be converted to RGB")

    if mask.ndim != 2:
        report.errors.append(f"mask should be 2D, got shape {mask.shape}")
        return report

    mask_height, mask_width = mask.shape
    if (mask_width, mask_height) != (report.width, report.height):
        report.errors.append(
            f"size mismatch: image {report.width}x{report.height}, mask {mask_width}x{mask_height}"
        )

    values = set(np.unique(mask).tolist())
    if not values <= EXPECTED_TRIMAP_VALUES:
        report.errors.append(f"unexpected mask values: {sorted(values - EXPECTED_TRIMAP_VALUES)}")

    return report
