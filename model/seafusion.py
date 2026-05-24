import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class _ConvLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size // 2)


class _Sobelxy(nn.Module):
    """Learnable depthwise conv initialised with Sobel kernels (original repo style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.convx = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.convy = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.convx.weight.data = kx.repeat(channels, 1, 1, 1)
        self.convy.weight.data = ky.repeat(channels, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.convx(x) ** 2 + self.convy(x) ** 2 + 1e-8)


class _DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 16):
        super().__init__()
        self.conv1 = _ConvLayer(in_channels, growth_rate)
        self.conv2 = _ConvLayer(in_channels + growth_rate, growth_rate)


class RGBD(nn.Module):
    """Gradient Residual Dense Block — naming matches original SeAFusion checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 16):
        super().__init__()
        self.dense = _DenseBlock(in_channels, growth_rate)
        self.convdown = _ConvLayer(in_channels + 2 * growth_rate, out_channels, kernel_size=1)
        self.sobelconv = _Sobelxy(in_channels)
        self.convup = _ConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.dense.conv1.conv(x), 0.2, inplace=True)
        x2 = F.leaky_relu(self.dense.conv2.conv(torch.cat([x, x1], dim=1)), 0.2, inplace=True)
        main = self.convdown.conv(torch.cat([x, x1, x2], dim=1))
        residual = self.convup.conv(self.sobelconv(x))
        return main + residual


class _DecodeLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_c)


class SeAFusionNet(BaseModel):
    """SeAFusion inference network.

    Parameter names mirror the original repository so that ``fusionmodel_final.pth``
    loads without any key remapping.

    Inputs
    ------
    ir : (B, 1, H, W)  infrared Y-channel, float32 [0, 1]
    vi : (B, 1, H, W)  visible  Y-channel, float32 [0, 1]

    Output
    ------
    (B, 1, H, W) fused Y-channel, float32 [0, 1]

    Reference: Tang et al., Information Fusion 82 (2022) 28-42
    """

    def __init__(self):
        super().__init__()
        self.vis_conv = _ConvLayer(1, 16)
        self.vis_rgbd1 = RGBD(16, 32, growth_rate=16)
        self.vis_rgbd2 = RGBD(32, 48, growth_rate=32)

        self.inf_conv = _ConvLayer(1, 16)
        self.inf_rgbd1 = RGBD(16, 32, growth_rate=16)
        self.inf_rgbd2 = RGBD(32, 48, growth_rate=32)

        self.decode4 = _DecodeLayer(96, 64)
        self.decode3 = _DecodeLayer(64, 32)
        self.decode2 = _DecodeLayer(32, 16)
        self.decode1 = _DecodeLayer(16, 1, kernel_size=3)

    def _extract(self, x, stem, rgbd1, rgbd2):
        x = F.leaky_relu(stem.conv(x), 0.2, inplace=True)
        x = rgbd1(x)
        x = rgbd2(x)
        return x

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        f_vi = self._extract(vi, self.vis_conv, self.vis_rgbd1, self.vis_rgbd2)
        f_ir = self._extract(ir, self.inf_conv, self.inf_rgbd1, self.inf_rgbd2)
        x = torch.cat([f_vi, f_ir], dim=1)
        x = F.leaky_relu(self.decode4.bn(self.decode4.conv(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.decode3.bn(self.decode3.conv(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.decode2.bn(self.decode2.conv(x)), 0.2, inplace=True)
        x = torch.tanh(self.decode1.bn(self.decode1.conv(x)))
        return (x + 1.0) / 2.0
