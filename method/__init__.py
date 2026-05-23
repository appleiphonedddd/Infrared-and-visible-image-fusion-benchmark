import importlib
from pathlib import Path

from base.base_method import build_method, METHOD_REGISTRY
from base.base_trainer import build_trainer, TRAINER_REGISTRY

# Auto-import every subpackage (directory with __init__.py) to trigger
# @register_method and @register_trainer decorators.
# Adding a new method directory requires zero changes to existing files.
for _pkg in sorted(Path(__file__).parent.iterdir()):
    if _pkg.is_dir() and (_pkg / '__init__.py').exists():
        importlib.import_module(f'.{_pkg.name}', package=__name__)

__all__ = [
    'build_method', 'METHOD_REGISTRY',
    'build_trainer', 'TRAINER_REGISTRY',
]
