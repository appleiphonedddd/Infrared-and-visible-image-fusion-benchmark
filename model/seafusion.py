import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class GRDB(nn.Module):
    """Gradient Residual Dense Block (Section 3.3, Fig. 4).

    Main dense stream: two 3×3 conv layers with LReLU and dense connections,
    followed by 1×1 conv for channel projection.
    Residual gradient stream: Sobel magnitude + 1×1 conv.
    Output = main + residual.
    """

    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + growth_rate, growth_rate, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Conv2d(in_channels + 2 * growth_rate, out_channels, 1)

        self.grad_proj = nn.Conv2d(in_channels, out_channels, 1)

        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.expand(in_channels, -1, -1, -1))
        self.register_buffer('sobel_y', sobel_y.expand(in_channels, -1, -1, -1))
        self.in_channels = in_channels

    def _sobel_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        main = self.proj(torch.cat([x, x1, x2], dim=1))
        residual = self.grad_proj(self._sobel_magnitude(x))
        return main + residual


class FeatureExtractor(nn.Module):
    """Single-modality feature extraction stream (Fig. 3).

    3×3 Conv+LReLU → GRDB (16→32) → GRDB (32→48)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.grdb1 = GRDB(16, 32, growth_rate=16)
        self.grdb2 = GRDB(32, 48, growth_rate=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.grdb1(x)
        x = self.grdb2(x)
        return x


class SeAFusionNet(BaseModel):
    """SeAFusion inference network (Section 3.3, Fig. 3).

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
        self.ir_extractor = FeatureExtractor(in_channels=1)
        self.vi_extractor = FeatureExtractor(in_channels=1)

        self.reconstructor = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        f_ir = self.ir_extractor(ir)
        f_vi = self.vi_extractor(vi)
        fused = self.reconstructor(torch.cat([f_ir, f_vi], dim=1))
        return (fused + 1.0) / 2.0
