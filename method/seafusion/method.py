import torch

from base.base_method import BaseMethod, register_method
from .model import SeAFusionNet


def rgb_to_ycbcr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """B×3×H×W [0,1] → (Y: B×1×H×W, CbCr: B×2×H×W) [0,1]."""
    r, g, b = x[:, :1], x[:, 1:2], x[:, 2:3]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, torch.cat([cb, cr], dim=1)


def ycbcr_to_rgb(y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
    """(Y: B×1×H×W, CbCr: B×2×H×W) [0,1] → B×3×H×W [0,1]."""
    cb, cr = cbcr[:, :1] - 0.5, cbcr[:, 1:] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


@register_method('SeAFusion')
class SeAFusionMethod(BaseMethod):
    """
    SeAFusion evaluation wrapper.

    Input:  IR  — 1-channel grayscale, [0, 1]
            VI  — 3-channel RGB,       [0, 1]
    Output: 3-channel RGB fused image, [0, 1]

    Internally fuses the Y channel (YCbCr). postprocess merges fused Y
    back with the Cb/Cr from the original visible image.
    """

    def _build_model(self) -> SeAFusionNet:
        return SeAFusionNet()

    def preprocess(
        self, ir: torch.Tensor, vi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ir = ir.to(self.device)
        # M3FD stores IR as 3-channel RGB PNG; model expects 1-channel grayscale
        if ir.shape[1] == 3:
            ir = 0.299 * ir[:, :1] + 0.587 * ir[:, 1:2] + 0.114 * ir[:, 2:3]
        vi = vi.to(self.device)
        # TNO VI is grayscale (1-ch); use directly as Y. Otherwise extract Y from RGB.
        vi_y = vi if vi.shape[1] == 1 else rgb_to_ycbcr(vi)[0]
        return ir, vi_y

    def _fuse(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        return self.model(ir, vi)

    def postprocess(self, fused: torch.Tensor, vi_original: torch.Tensor) -> torch.Tensor:
        # TNO: VI is grayscale, no CbCr to merge back — return grayscale result
        if vi_original.shape[1] == 1:
            return fused.clamp(0.0, 1.0).cpu()
        _, vi_cbcr = rgb_to_ycbcr(vi_original.to(fused.device))
        return ycbcr_to_rgb(fused, vi_cbcr).cpu()
