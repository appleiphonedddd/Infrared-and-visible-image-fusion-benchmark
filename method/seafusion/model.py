import torch
import torch.nn as nn

from base import BaseModel


class SeAFusionNet(BaseModel):
    """
    SeAFusion: Image Fusion in the Loop of High-Level Vision Tasks
    INFFUS 2022  —  https://doi.org/10.1016/j.inffus.2021.12.004

    TODO: implement architecture.
    """

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
