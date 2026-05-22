from pathlib import Path

from ..base_dataset import BaseFusionDataset


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
