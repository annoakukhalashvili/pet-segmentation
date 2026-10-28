"""PyTorch Dataset for Oxford-IIIT Pet three-class segmentation."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from petseg.data.annotations import remap_trimap


class PetSegmentationDataset(Dataset):
    """Returns (image, mask) pairs.

    image: float32 tensor of shape (3, H, W), normalised.
    mask:  int64 tensor of shape (H, W) with values 0 (background), 1 (pet), 2 (boundary).
    """

    def __init__(self, data_dir: Path, image_ids: list[str], transforms: v2.Compose):
        self.images_dir = Path(data_dir) / "images"
        self.trimaps_dir = Path(data_dir) / "annotations" / "trimaps"
        self.image_ids = list(image_ids)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        trimap = np.array(Image.open(self.trimaps_dir / f"{image_id}.png"))
        mask = tv_tensors.Mask(torch.from_numpy(remap_trimap(trimap)))

        image, mask = self.transforms(image, mask)
        # Mask is a tensor subclass; convert to a plain int64 tensor, as the loss function expects.
        return image, mask.as_subclass(torch.Tensor).long()