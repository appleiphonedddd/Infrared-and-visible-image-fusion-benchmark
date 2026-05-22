import torch

from base.base_method import BaseMethod, register_method
from .model import SeAFusionNet


@register_method('SeAFusion')
class SeAFusionMethod(BaseMethod):
    """
    SeAFusion evaluation wrapper.

    Input:  IR  — 1-channel grayscale, [0, 1]
            VI  — 3-channel RGB,       [0, 1]
    Output: fused 1-channel grayscale, [0, 1]

    If the method expects YCbCr input, override preprocess/postprocess here.
    """

    def _build_model(self) -> SeAFusionNet:
        return SeAFusionNet()

    def _fuse(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        return self.model(ir, vi)
