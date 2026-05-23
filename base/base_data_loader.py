from abc import ABC, abstractmethod
from pathlib import Path

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
