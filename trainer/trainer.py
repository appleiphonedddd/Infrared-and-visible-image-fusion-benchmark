import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base.base_method import BaseMethod
from base.base_trainer import BaseFusionTrainer


def _to_gray(t: torch.Tensor) -> torch.Tensor:
    """B×C×H×W → B×1×H×W via luminance weights (pass-through if already 1-ch)."""
    if t.shape[1] == 1:
        return t
    return 0.299 * t[:, :1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]


def _sobel_magnitude(t: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude, B×1×H×W → B×1×H×W."""
    kx = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=t.dtype, device=t.device
    ).unsqueeze(0)
    ky = kx.transpose(-1, -2)
    return (F.conv2d(t, kx, padding=1) ** 2 + F.conv2d(t, ky, padding=1) ** 2).sqrt()


class FusionLoss(nn.Module):
    """
    Standard pixel + gradient loss for image fusion training.

    L_intensity = L1(fused_gray, max(ir_gray, vi_gray))
    L_gradient  = L1(∇fused_gray, max(∇ir_gray, ∇vi_gray))
    L_total     = w_intensity * L_intensity + w_gradient * L_gradient

    Both IR and VI inputs are converted to grayscale before loss computation,
    so the loss works regardless of channel counts (1-ch or 3-ch).
    """

    def __init__(self, w_intensity: float = 1.0, w_gradient: float = 1.0):
        super().__init__()
        self.w_intensity = w_intensity
        self.w_gradient = w_gradient

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vi: torch.Tensor,
    ) -> torch.Tensor:
        fused_g = _to_gray(fused)
        ir_g = _to_gray(ir)
        vi_g = _to_gray(vi)

        loss_intensity = F.l1_loss(fused_g, torch.max(ir_g, vi_g))

        grad_fused = _sobel_magnitude(fused_g)
        grad_target = torch.max(_sobel_magnitude(ir_g), _sobel_magnitude(vi_g))
        loss_gradient = F.l1_loss(grad_fused, grad_target)

        return self.w_intensity * loss_intensity + self.w_gradient * loss_gradient


class FusionTrainer(BaseFusionTrainer):
    """
    Concrete trainer for image fusion baselines.

    Uses FusionLoss (intensity + gradient) by default; pass a custom loss_fn
    to override.  All keyword arguments beyond loss_fn are forwarded to
    BaseFusionTrainer (epochs, save_dir, save_period, resume).

    Example
    -------
    method = build_method('SeAFusion', device='cuda')
    optimizer = torch.optim.Adam(method.model.parameters(), lr=1e-4)
    train_ds = MSRSDataset('dataset/MSRS', split='train')
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    trainer = FusionTrainer(
        method, optimizer, train_loader,
        epochs=50, save_dir='saved/seafusion',
    )
    trainer.train()
    """

    def __init__(
        self,
        method: BaseMethod,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        *,
        loss_fn: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(method, optimizer, train_loader, **kwargs)
        self.loss_fn = FusionLoss() if loss_fn is None else loss_fn

    def _train_epoch(self, epoch: int) -> dict:
        self.method.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            ir, vi = self.method.preprocess(batch['ir'], batch['vi'])

            self.optimizer.zero_grad()
            fused = self.method._fuse(ir, vi)
            loss = self.loss_fn(fused, ir, vi)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info('Epoch %d | loss: %.4f', epoch, avg_loss)
        return {'loss': avg_loss}
