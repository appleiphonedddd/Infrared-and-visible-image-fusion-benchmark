from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from base.base_trainer import BaseFusionTrainer, register_trainer
from loss.seafusion import SeAFusionLoss, SemanticLoss
from .method import SeAFusionMethod, rgb_to_ycbcr, ycbcr_to_rgb


class SeAFusionTrainer(BaseFusionTrainer):

    def __init__(
        self,
        method: SeAFusionMethod,
        seg_net: nn.Module,
        fusion_optimizer: torch.optim.Optimizer,
        seg_optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        *,
        M: int = 4,
        p: int = 2700,
        q: int = 20000,
        gamma: float = 1.0,
        save_dir: str | Path = 'saved',
        save_period: int = 1,
        resume: str | Path | None = None,
    ):
        super().__init__(
            method, fusion_optimizer, train_loader,
            epochs=M,
            save_dir=save_dir,
            save_period=save_period,
            resume=resume,
        )
        self.seg_net = seg_net.to(method.device)
        self.seg_optimizer = seg_optimizer
        self.p = p
        self.q = q
        self.gamma = gamma
        self.joint_loss = SeAFusionLoss()
        self.semantic_loss_fn = SemanticLoss()


    def _train_epoch(self, m: int) -> dict:
        """One outer iteration m of Algorithm 1."""
        beta = self.gamma * (m - 1)  # Eq. 14: β = γ*(m-1)
        f_loss = self._phase_fusion(beta)
        s_loss = self._phase_seg()
        return {'fusion_loss': f_loss, 'seg_loss': s_loss, 'beta': beta}


    def _phase_fusion(self, beta: float) -> float:
        """p gradient steps on the fusion network; seg net frozen."""
        self.method.model.train()
        self.seg_net.eval()
        for param in self.seg_net.parameters():
            param.requires_grad_(False)

        total = 0.0
        for batch in _take(self.p, self.train_loader):
            ir = batch['ir'].to(self.method.device)
            vi_y, vi_cbcr = rgb_to_ycbcr(batch['vi'].to(self.method.device))
            labels = batch['seg_label'].to(self.method.device)

            self.optimizer.zero_grad()
            fused_y = self.method.model(ir, vi_y)
            logits_main, logits_aux = self.seg_net(ycbcr_to_rgb(fused_y, vi_cbcr))
            loss, _ = self.joint_loss(
                fused_y, ir, vi_y, logits_main, logits_aux, labels, beta=beta
            )
            loss.backward()
            self.optimizer.step()
            total += loss.item()

        for param in self.seg_net.parameters():
            param.requires_grad_(True)
        return total / self.p

    def _phase_seg(self) -> float:
        """q gradient steps on the seg network; fusion net frozen."""
        self.method.model.eval()
        self.seg_net.train()

        total = 0.0
        for batch in _take(self.q, self.train_loader):
            ir = batch['ir'].to(self.method.device)
            vi_y, vi_cbcr = rgb_to_ycbcr(batch['vi'].to(self.method.device))
            labels = batch['seg_label'].to(self.method.device)

            with torch.no_grad():
                fused_rgb = ycbcr_to_rgb(self.method.model(ir, vi_y), vi_cbcr)

            self.seg_optimizer.zero_grad()
            logits_main, logits_aux = self.seg_net(fused_rgb)
            loss = self.semantic_loss_fn(logits_main, logits_aux, labels)
            loss.backward()
            self.seg_optimizer.step()
            total += loss.item()

        return total / self.q


    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        state = {
            'epoch': epoch,
            'arch': type(self.method.model).__name__,
            'state_dict': self.method.model.state_dict(),
            'seg_state_dict': self.seg_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'seg_optimizer': self.seg_optimizer.state_dict(),
        }
        path = self.save_dir / f'checkpoint-iter{epoch}.pth'
        torch.save(state, path)
        self.logger.info('Saving checkpoint: %s', path)


def _take(n: int, loader: DataLoader):
    """Yield exactly n batches from loader, restarting when exhausted."""
    count = 0
    while count < n:
        for batch in loader:
            if count >= n:
                return
            yield batch
            count += 1


# ------------------------------------------------------------------ #
# Trainer factory — registered so main.py can dispatch generically    #
# ------------------------------------------------------------------ #

@register_trainer('SeAFusion')
def _make_seafusion_trainer(
    method: SeAFusionMethod,
    train_loader: DataLoader,
    config: dict,
) -> SeAFusionTrainer:
    """
    Build a SeAFusionTrainer from a config dict.

    Required config keys:
        seg_net (str): path to a saved segmentation model
                       (saved with torch.save(model), not just state_dict)

    Optional config keys (with defaults):
        lr_fusion   (float) : 1e-4
        lr_seg      (float) : 1e-2
        M           (int)   : 4      — outer iterations
        p           (int)   : 2700   — fusion gradient steps per iteration
        q           (int)   : 20000  — seg gradient steps per iteration
        gamma       (float) : 1.0
        save_dir    (str)   : 'saved'
        save_period (int)   : 1
        resume      (str)   : None
    """
    seg_net_path = config.get('seg_net')
    if not seg_net_path:
        raise ValueError(
            "SeAFusion training requires 'seg_net' in config: "
            "path to a segmentation model saved with torch.save(model, path)"
        )
    seg_net: nn.Module = torch.load(
        seg_net_path, map_location=method.device, weights_only=False
    )

    fusion_opt = torch.optim.Adam(
        method.model.parameters(), lr=config.get('lr_fusion', 1e-4)
    )
    seg_opt = torch.optim.SGD(
        seg_net.parameters(),
        lr=config.get('lr_seg', 1e-2),
        momentum=0.9,
        weight_decay=5e-4,
    )

    return SeAFusionTrainer(
        method=method,
        seg_net=seg_net,
        fusion_optimizer=fusion_opt,
        seg_optimizer=seg_opt,
        train_loader=train_loader,
        M=config.get('M', 4),
        p=config.get('p', 2700),
        q=config.get('q', 20000),
        gamma=config.get('gamma', 1.0),
        save_dir=config.get('save_dir', 'saved'),
        save_period=config.get('save_period', 1),
        resume=config.get('resume'),
    )
