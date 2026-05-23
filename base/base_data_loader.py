import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class BaseFusionDataset(Dataset, ABC):
    """
    Abstract base for all fusion datasets.
    Subclasses implement _load_pairs() and declare ir_channels / vi_channels.
    __getitem__ always returns {'ir': Tensor, 'vi': Tensor, 'name': str}.
    """

    ir_channels: int
    vi_channels: int

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.pairs: list[tuple[Path, Path, str]] = self._load_pairs()

    @abstractmethod
    def _load_pairs(self) -> list[tuple[Path, Path, str]]:
        """Return sorted list of (ir_path, vi_path, name) tuples."""

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        ir_path, vi_path, name = self.pairs[idx]
        ir = self._load_image(ir_path)
        vi = self._load_image(vi_path)

        if self.transform is not None:
            ir, vi = self.transform(ir, vi)

        return {'ir': ir, 'vi': vi, 'name': name}

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        """Load image as float tensor in [0, 1], preserving original channels."""
        return T.ToTensor()(Image.open(path))

    @staticmethod
    def _load_mask(path: Path) -> torch.Tensor:
        """Load integer class mask (e.g. segmentation labels), preserving raw IDs."""
        import numpy as np
        return torch.from_numpy(np.array(Image.open(path))).long()


# ------------------------------------------------------------------ #
# Dataset registry                                                     #
# ------------------------------------------------------------------ #

DATASET_REGISTRY: dict[str, type['BaseFusionDataset']] = {}


def register_dataset(name: Optional[str] = None):
    """
    Class decorator that registers a BaseFusionDataset subclass.

    Usage:
        @register_dataset('MSRS')
        class MSRSDataset(BaseFusionDataset): ...

        @register_dataset()
        class TNODataset(BaseFusionDataset): ...   # key = 'TNODataset'
    """
    def decorator(cls: type[BaseFusionDataset]) -> type[BaseFusionDataset]:
        key = name if name is not None else cls.__name__
        if key in DATASET_REGISTRY:
            raise KeyError(f"Dataset '{key}' is already registered.")
        DATASET_REGISTRY[key] = cls
        return cls
    return decorator


def build_dataset(name: str, **kwargs) -> BaseFusionDataset:
    """
    Instantiate a registered dataset by name.

    Unknown kwargs are silently filtered out so callers can always pass
    the full set of options (e.g. split='train') and datasets that do
    not support them simply ignore them.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    cls = DATASET_REGISTRY[name]
    params = inspect.signature(cls.__init__).parameters
    valid = {k: v for k, v in kwargs.items() if k in params}
    return cls(**valid)
