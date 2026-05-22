from pathlib import Path

from ..base_dataset import BaseFusionDataset


class M3FDDataset(BaseFusionDataset):
    """
    M3FD dataset — all 300 pairs used as test set.
    IR: RGB (3-channel, stored as colour PNG) | VI: RGB (3-channel)
    Method wrappers are responsible for converting IR to grayscale if needed.
    """

    ir_channels = 3  # stored as RGB despite being infrared
    vi_channels = 3

    def __init__(self, root: str | Path, transform=None):
        super().__init__(root, transform)

    def _load_pairs(self) -> list[tuple[Path, Path, str]]:
        ir_dir = self.root / 'Ir'
        vi_dir = self.root / 'Vis'
        names = sorted(p.name for p in ir_dir.iterdir() if p.suffix == '.png')
        return [(ir_dir / n, vi_dir / n, Path(n).stem) for n in names]
