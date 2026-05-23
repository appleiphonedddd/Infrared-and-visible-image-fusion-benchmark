from base.base_data_loader import BaseFusionDataset
from base.base_fusion_loader import FusionDataLoader
from .data_loaders import M3FDDataset, MSRSDataset, RoadSceneDataset, TNODataset

REGISTRY: dict = {
    'MSRS': MSRSDataset,
    'M3FD': M3FDDataset,
    'RoadScene': RoadSceneDataset,
    'TNO': TNODataset,
}


def build_dataset(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ['BaseFusionDataset', 'FusionDataLoader', 'build_dataset', 'REGISTRY',
           'M3FDDataset', 'MSRSDataset', 'RoadSceneDataset', 'TNODataset']
