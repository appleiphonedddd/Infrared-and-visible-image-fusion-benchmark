from torch.utils.data import DataLoader

from .base_dataset import BaseFusionDataset


class FusionDataLoader(DataLoader):
    """
    DataLoader for benchmark evaluation.
    Always deterministic (shuffle=False) so all methods run on identical image order.
    Default batch_size=1 to handle datasets with varying resolutions without padding.
    """

    def __init__(
        self,
        dataset: BaseFusionDataset,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
