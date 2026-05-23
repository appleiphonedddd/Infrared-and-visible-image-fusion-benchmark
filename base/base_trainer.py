import logging
import torch
from abc import abstractmethod
from pathlib import Path
from torch.utils.data import DataLoader

from .base_method import BaseMethod


class BaseFusionTrainer:
    """
    Abstract base trainer for image fusion methods.

    Subclasses must implement _train_epoch().

    Checkpoint format matches BaseMethod.load_checkpoint() — the 'state_dict'
    key holds bare model weights, so a trained checkpoint can be loaded
    directly via method.load_checkpoint(path) without any trainer dependency.
    """

    def __init__(
        self,
        method: BaseMethod,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        *,
        epochs: int = 100,
        save_dir: str | Path = 'saved',
        save_period: int = 10,
        resume: str | Path | None = None,
    ):
        self.method = method
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.epochs = epochs
        self.save_period = save_period
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(type(self).__name__)
        self.start_epoch = 1

        if resume is not None:
            self._resume_checkpoint(resume)

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Return a dict of metric_name → value."""
        raise NotImplementedError

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch, **result}
            for key, value in log.items():
                self.logger.info('    %-20s %s', key, value)
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        state = {
            'epoch': epoch,
            'arch': type(self.method.model).__name__,
            'state_dict': self.method.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
        }
        path = self.save_dir / f'checkpoint-epoch{epoch}.pth'
        torch.save(state, path)
        self.logger.info('Saving checkpoint: %s', path)
        if save_best:
            best_path = self.save_dir / 'model_best.pth'
            torch.save(state, best_path)
            self.logger.info('Saving current best: model_best.pth')

    def _resume_checkpoint(self, resume_path: str | Path) -> None:
        resume_path = Path(resume_path)
        self.logger.info('Loading checkpoint: %s', resume_path)
        ckpt = torch.load(
            resume_path, map_location=self.method.device, weights_only=False
        )
        self.start_epoch = ckpt['epoch'] + 1
        self.mnt_best = ckpt.get('monitor_best', self.mnt_best)
        state = {k.removeprefix('module.'): v for k, v in ckpt['state_dict'].items()}
        self.method.model.load_state_dict(state)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.logger.info('Resuming training from epoch %d', self.start_epoch)
