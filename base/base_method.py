from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class BaseMethod(ABC):
    """
    Abstract base class for every fusion method in the benchmark.

    A *Method* wraps a trained *Model* (nn.Module) with the full
    evaluation pipeline:

        load_checkpoint → preprocess → _fuse → postprocess

    Subclasses MUST implement:
        _build_model()   return the nn.Module (weights not loaded yet)
        _fuse()          raw forward pass on preprocessed device tensors

    Subclasses MAY override:
        preprocess()     e.g. RGB visible → YCbCr, keep only Y channel
        postprocess()    e.g. merge fused Y back with Cb/Cr from visible
        name             human-readable tag used in result tables/folders

    Example — minimal subclass
    --------------------------
    @register_method('MyNet')
    class MyNetMethod(BaseMethod):
        def _build_model(self):
            return MyNet()

        def _fuse(self, ir, vi):
            return self.model(ir, vi)
    """

    def __init__(self, device: str | torch.device = 'cuda'):
        self.device = torch.device(device)
        self.model: nn.Module = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Subclass contract                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Instantiate and return the model (weights not loaded yet)."""

    @abstractmethod
    def _fuse(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        """
        Raw forward pass.
        Both tensors are already on self.device and have been preprocessed.
        Returns fused tensor before postprocessing.
        """

    # ------------------------------------------------------------------ #
    # Optional pipeline hooks                                              #
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        ir: torch.Tensor,
        vi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for _fuse.  Runs on CPU tensors, should return
        device tensors ready for the model.

        Override for modality-specific transforms, e.g.:
            - Convert RGB visible to YCbCr and retain only the Y channel
            - Convert 3-channel IR (M3FD) to grayscale
            - Normalize beyond the default [0, 1] range

        Default: move both tensors to self.device unchanged.
        """
        return ir.to(self.device), vi.to(self.device)

    def postprocess(
        self,
        fused: torch.Tensor,
        vi_original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert _fuse output to a saveable image tensor in [0, 1].

        vi_original is the *raw* CPU visible tensor (before preprocess),
        available so YCbCr-based methods can merge the fused Y channel
        back with the Cb/Cr channels of the original visible image.

        Default: clamp fused result to [0, 1] and move to CPU.
        """
        return fused.clamp(0.0, 1.0).cpu()

    # ------------------------------------------------------------------ #
    # Checkpoint loading                                                   #
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load model weights from a checkpoint file.

        Accepts both:
          - raw state-dicts  (torch.save(model.state_dict(), path))
          - wrapped ckpts    (dicts with a 'state_dict' key, as saved by
                              BaseTrainer._save_checkpoint)

        Also strips the 'module.' prefix introduced by DataParallel / DDP.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------ #
    # Public evaluation interface                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fuse(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        """
        Full evaluation pipeline: preprocess → _fuse → postprocess.

        Expects raw DataLoader tensors (CPU, float32, [0, 1]).
        Returns a CPU tensor in [0, 1] suitable for saving or metric computation.

        Gradient computation is disabled automatically.
        """
        vi_original = vi.clone()
        ir, vi = self.preprocess(ir, vi)
        fused = self._fuse(ir, vi)
        return self.postprocess(fused, vi_original)

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Override to provide a cleaner display name, e.g. 'BSPFusion'."""
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}(device={self.device})"


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

METHOD_REGISTRY: dict[str, type[BaseMethod]] = {}


def register_method(name: Optional[str] = None):
    """
    Class decorator that registers a BaseMethod subclass.

    Usage:
        @register_method()
        class MyMethod(BaseMethod): ...          # key = 'MyMethod'

        @register_method('bspfusion')
        class BSPFusionMethod(BaseMethod): ...   # key = 'bspfusion'
    """
    def decorator(cls: type[BaseMethod]) -> type[BaseMethod]:
        key = name if name is not None else cls.__name__
        if key in METHOD_REGISTRY:
            raise KeyError(f"Method '{key}' is already registered.")
        METHOD_REGISTRY[key] = cls
        return cls
    return decorator


def build_method(name: str, **kwargs) -> BaseMethod:
    """Instantiate a registered method by name. kwargs are forwarded to __init__."""
    if name not in METHOD_REGISTRY:
        raise KeyError(
            f"Unknown method '{name}'. Available: {sorted(METHOD_REGISTRY)}"
        )
    return METHOD_REGISTRY[name](**kwargs)
