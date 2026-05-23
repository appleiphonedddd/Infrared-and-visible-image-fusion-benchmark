from __future__ import annotations
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import ClassVar


def _to_numpy_hwc(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.ndim == 4:
            x = x.squeeze(0)
        x = x.permute(1, 2, 0).cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    return x


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 1:
        return img[:, :, 0]
    return (0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2]).astype(np.float32)


class BaseMetric(ABC):
    name: ClassVar[str]
    higher_is_better: ClassVar[bool] = True

    def __call__(
        self,
        fused: np.ndarray | torch.Tensor,
        ir:    np.ndarray | torch.Tensor,
        vi:    np.ndarray | torch.Tensor,
    ) -> float:
        f = _to_gray(_to_numpy_hwc(fused))
        i = _to_gray(_to_numpy_hwc(ir))
        v = _to_gray(_to_numpy_hwc(vi))
        return float(self.compute(f, i, v))

    @abstractmethod
    def compute(self, fused: np.ndarray, ir: np.ndarray, vi: np.ndarray) -> float:
        """All inputs: (H, W) float32 greyscale in [0, 1]."""


METRIC_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(name: str | None = None):
    """
    Class decorator that registers a BaseMetric subclass.

    Usage
    -----
        @register_metric()
        class MyMetric(BaseMetric): ...          # registered as 'MyMetric'

        @register_metric('alias')
        class AnotherMetric(BaseMetric): ...     # registered as 'alias'
    """
    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        key = name if name is not None else cls.__name__
        if key in METRIC_REGISTRY:
            raise KeyError(f"Metric '{key}' is already registered.")
        METRIC_REGISTRY[key] = cls
        return cls
    return decorator


def build_metric(name: str, **kwargs) -> BaseMetric:
    """Instantiate a registered metric by name. kwargs forwarded to __init__."""
    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY)}"
        )
    return METRIC_REGISTRY[name](**kwargs)
