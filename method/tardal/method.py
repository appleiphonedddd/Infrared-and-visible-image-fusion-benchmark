import torch

from base.base_method import BaseMethod, register_method
from .model import TarDALNet


@register_method('TarDAL')
class TarDALMethod(BaseMethod):
    """
    TarDAL evaluation wrapper.

    Input:  IR  — 1-channel grayscale, [0, 1]
            VI  — 3-channel RGB,       [0, 1]
    Output: fused 1-channel grayscale, [0, 1]
    """

    def _build_model(self) -> TarDALNet:
        return TarDALNet()

    def _fuse(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        return self.model(ir, vi)
