import torch
import torch.nn as nn

from base import BaseModel


class TarDALNet(BaseModel):
    """
    TarDAL: Target-aware Dual Adversarial Learning
    CVPR 2022  —  https://doi.org/10.1109/CVPR52688.2022.00570

    TODO: implement architecture.
    """

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
