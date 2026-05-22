from pathlib import Path

from ..base_dataset import BaseFusionDataset


class RoadSceneDataset(BaseFusionDataset):
    """
    RoadScene dataset — all 42 pairs used as test set (from meta/pred.txt).
    IR: grayscale (1-channel) | VI: RGB (3-channel)
    """

    ir_channels = 1
    vi_channels = 3

    def __init__(self, root: str | Path, transform=None):
        super().__init__(root, transform)

    def _load_pairs(self) -> list[tuple[Path, Path, str]]:
        ir_dir = self.root / 'ir'
        vi_dir = self.root / 'vi'
        names = (self.root / 'meta' / 'pred.txt').read_text().strip().splitlines()
        return [(ir_dir / n, vi_dir / n, Path(n).stem) for n in names]
