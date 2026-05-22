from pathlib import Path

from ..base_dataset import BaseFusionDataset


class TNODataset(BaseFusionDataset):
    """
    TNO dataset — all 37 pairs used as test set (from meta/pred.txt).
    IR: grayscale (1-channel) | VI: grayscale (1-channel)
    Both modalities are single-channel; method wrappers must handle vi_channels=1.
    """

    ir_channels = 1
    vi_channels = 1  # grayscale, unlike most other datasets

    def __init__(self, root: str | Path, transform=None):
        super().__init__(root, transform)

    def _load_pairs(self) -> list[tuple[Path, Path, str]]:
        ir_dir = self.root / 'ir'
        vi_dir = self.root / 'vi'
        names = (self.root / 'meta' / 'pred.txt').read_text().strip().splitlines()
        return [(ir_dir / n, vi_dir / n, Path(n).stem) for n in names]
