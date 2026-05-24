from pathlib import Path

from base.base_data_loader import BaseFusionDataset, register_dataset


@register_dataset('M3FD')
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


@register_dataset('MSRS')
class MSRSDataset(BaseFusionDataset):
    """
    MSRS dataset with official train/test folder split.
    IR: grayscale (1-channel) | VI: RGB (3-channel)
    Optionally returns segmentation labels (integer class IDs) when include_seg=True.
    """

    ir_channels = 1
    vi_channels = 3

    def __init__(
        self,
        root: str | Path,
        split: str = 'test',
        include_seg: bool = False,
        transform=None,
    ):
        assert split in ('train', 'test'), f"split must be 'train' or 'test', got '{split}'"
        self.split = split
        self.include_seg = include_seg
        super().__init__(root, transform)

    def _load_pairs(self) -> list[tuple[Path, Path, str]]:
        ir_dir = self.root / self.split / 'ir'
        vi_dir = self.root / self.split / 'vi'
        names = sorted(p.name for p in ir_dir.iterdir() if p.suffix == '.png')
        return [(ir_dir / n, vi_dir / n, Path(n).stem) for n in names]

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        if self.include_seg:
            _, _, name = self.pairs[idx]
            seg_dir = self.root / self.split / 'Segmentation_labels'
            sample['seg_label'] = self._load_mask(seg_dir / f'{name}.png')
        return sample


@register_dataset('RoadScene')
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

@register_dataset('TNO')
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

@register_dataset('LLVIP')
class LLVIPDataset(BaseFusionDataset):
    """
    LLVIP dataset with official train/test folder split.
    IR: RGB (3-channel, grayscale content stored as colour JPG) | VI: RGB (3-channel)
    Method wrappers are responsible for converting IR to grayscale if needed.
    """

    ir_channels = 3  # stored as RGB despite being grayscale infrared
    vi_channels = 3

    def __init__(self, root: str | Path, split: str = 'test', transform=None):
        assert split in ('train', 'test'), f"split must be 'train' or 'test', got '{split}'"
        self.split = split
        super().__init__(root, transform)

    def _load_pairs(self) -> list[tuple[Path, Path, str]]:
        ir_dir = self.root / 'infrared' / self.split
        vi_dir = self.root / 'visible' / self.split
        names = sorted(p.name for p in ir_dir.iterdir() if p.suffix == '.jpg')
        return [(ir_dir / n, vi_dir / n, Path(n).stem) for n in names]
