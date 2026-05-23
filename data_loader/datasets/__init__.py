from .data_loaders import M3FDDataset, MSRSDataset, RoadSceneDataset, TNODataset

REGISTRY: dict = {
    'MSRS': MSRSDataset,
    'M3FD': M3FDDataset,
    'RoadScene': RoadSceneDataset,
    'TNO': TNODataset,
}


def build_dataset(name: str, **kwargs):
    """Instantiate a dataset by name. kwargs are forwarded to the dataset constructor."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ['REGISTRY', 'build_dataset', 'MSRSDataset', 'M3FDDataset', 'RoadSceneDataset', 'TNODataset']
