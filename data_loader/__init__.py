import importlib
from pathlib import Path

from base.base_data_loader import BaseFusionDataset, DATASET_REGISTRY, register_dataset, build_dataset
from base.base_fusion_loader import FusionDataLoader

# Auto-import every *.py module in this package to trigger @register_dataset decorators.
# Adding a new dataset file here requires zero changes to existing files.
for _f in sorted(Path(__file__).parent.glob('*.py')):
    if _f.stem != '__init__':
        importlib.import_module(f'.{_f.stem}', package=__name__)

__all__ = [
    'BaseFusionDataset', 'FusionDataLoader',
    'DATASET_REGISTRY', 'register_dataset', 'build_dataset',
]
