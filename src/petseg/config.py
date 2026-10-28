"""Experiment configuration loaded from YAML files in configs/."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """All settings of one training experiment. Defaults are shared by every experiment."""

    name: str  # experiment name, also the output folder name
    data_dir: str = "data/oxford-iiit-pet"
    splits_dir: str = "splits"
    output_dir: str = "artifacts/experiments"

    seed: int = 42
    image_size: int = 320
    batch_size: int = 16
    num_workers: int = 2
    device: str = "auto"

    pretrained: bool = True
    freeze_backbone: bool = False  # True = train only the segmentation head

    epochs: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    class_weighting: str = "none"  # "none" or "median_frequency"
    early_stopping_patience: int = 5  # stop after this many epochs without val improvement
    mixed_precision: bool = True  # used only on CUDA GPUs

    def __post_init__(self) -> None:
        if self.class_weighting not in {"none", "median_frequency"}:
            raise ValueError(f"class_weighting must be 'none' or 'median_frequency', got {self.class_weighting!r}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1")

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> TrainConfig:
    """Read a YAML file into a TrainConfig. Unknown keys raise an error (catches typos)."""
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}

    known = {f.name for f in fields(TrainConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return TrainConfig(**values)
