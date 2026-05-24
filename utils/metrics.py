from __future__ import annotations
import math
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve
from scipy.stats import entropy as _scipy_entropy
from skimage.metrics import structural_similarity as _skimage_ssim
from typing import Callable

from base.base_metrics import (
    BaseMetric, METRIC_REGISTRY, register_metric, build_metric,
    _to_numpy_hwc, _to_gray,
)


def _hist_entropy(img: np.ndarray, bins: int) -> float:
    lo, hi = float(img.min()), float(img.max())
    if lo == hi:
        return 0.0
    hist, _ = np.histogram(img.ravel(), bins=bins, range=(lo, hi))
    return float(_scipy_entropy(hist, base=2))


def _joint_entropy(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    a_lo, a_hi = float(a.min()), float(a.max())
    b_lo, b_hi = float(b.min()), float(b.max())
    if a_lo == a_hi or b_lo == b_hi:
        return 0.0
    h2d, _, _ = np.histogram2d(
        a.ravel(), b.ravel(),
        bins=bins,
        range=[[a_lo, a_hi], [b_lo, b_hi]],
    )
    return float(_scipy_entropy(h2d.ravel(), base=2))


def _mutual_info(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    return _hist_entropy(a, bins) + _hist_entropy(b, bins) - _joint_entropy(a, b, bins)


def _sobel_gradient(img: np.ndarray):
    hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    hy = hx.T
    gx = convolve(img, hx, mode='reflect')
    gy = convolve(img, hy, mode='reflect')
    return (np.sqrt(gx**2 + gy**2).astype(np.float32),
            np.arctan2(gy, gx).astype(np.float32))


def _haar_detail(img: np.ndarray) -> np.ndarray:
    _, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    return np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])

class SSIM(BaseMetric):
    """
    Structural Similarity Index Measure (Wang et al., TIP 2004).
    Computed as (SSIM(F, IR) + SSIM(F, VI)) / 2 using a Gaussian window.
    """

    name = 'SSIM'
    higher_is_better = True

    def __init__(self, win_size: int = 11, sigma: float = 1.5):
        self._kwargs = dict(data_range=1.0, gaussian_weights=True, sigma=sigma)

    def compute(self, fused, ir, vi):
        return (_skimage_ssim(fused, ir, **self._kwargs)
                + _skimage_ssim(fused, vi, **self._kwargs)) / 2.0


class MutualInformation(BaseMetric):
    """
    Mutual Information: I(F; IR) + I(F; VI).
    Estimated from 2-D intensity histograms (256 bins by default).
    """

    name = 'MI'
    higher_is_better = True

    def __init__(self, bins: int = 256):
        self.bins = bins

    def compute(self, fused, ir, vi):
        return float(
            _mutual_info(fused, ir, self.bins)
            + _mutual_info(fused, vi, self.bins)
        )


class Qabf(BaseMetric):
    """
    Gradient-based fusion quality (Xydeas & Petrovic, Signal Process. 2000).
    Measures how well gradient magnitude and orientation are preserved from
    both source images.  Q_abf ∈ (0, 1], higher is better.
    """

    name = 'Q_abf'
    higher_is_better = True

    # Sigmoid constants from Xydeas & Petrovic (2000)
    _Tg, _kg, _Dg = 0.9994, -15.0, 0.5
    _Ta, _ka, _Da = 0.9879, -22.0, 0.8
    _L = 1.0  # gradient weight exponent

    def _edge_quality(self, mag_s, ang_s, mag_f, ang_f) -> np.ndarray:
        """Per-pixel gradient preservation quality Q^{S→F}."""
        eps = 1e-10
        g = np.where(mag_f <= mag_s,
                     mag_f / (mag_s + eps),
                     mag_s / (mag_f + eps))
        Qg = self._Tg / (1.0 + np.exp(self._kg * (g - self._Dg)))

        da = np.abs(ang_s - ang_f)
        da = np.where(da > np.pi, 2.0 * np.pi - da, da)
        da = np.minimum(da, np.pi - da)
        # Q_{A,rel} = 1 - |da| / (pi/2): perfect match → 1, worst → 0
        Qa = self._Ta / (1.0 + np.exp(self._ka * (1.0 - da / (np.pi / 2.0) - self._Da)))
        return Qg * Qa

    def compute(self, fused, ir, vi):
        mag_f, ang_f = _sobel_gradient(fused)
        mag_i, ang_i = _sobel_gradient(ir)
        mag_v, ang_v = _sobel_gradient(vi)

        wA    = mag_i ** self._L
        wB    = mag_v ** self._L
        denom = (wA + wB).sum()
        if denom < 1e-10:
            return 0.0
        Q_if = self._edge_quality(mag_i, ang_i, mag_f, ang_f)
        Q_vf = self._edge_quality(mag_v, ang_v, mag_f, ang_f)
        return float((Q_if * wA + Q_vf * wB).sum() / denom)


class Nabf(BaseMetric):
    """
    Modified Fusion Artifacts measure (Shreyamsha Kumar, 2013).
    Quality-weighted fraction of fused gradient that constitutes artifacts
    (exceeds both source gradients), weighted by source gradient energy.
    N_abf ∈ [0, 1], lower is better.
    """

    name = 'N_abf'
    higher_is_better = False

    # MATLAB parameters from analysis_nabf.m (images in [0,255] with /8 Sobel)
    _Td     = 2.0
    _wt_min = 0.001
    _Lg     = 1.5
    _Nrg = 0.9999; _kg = 19.0; _sigmag = 0.5
    _Nra = 0.9995; _ka = 22.0; _sigmaa = 0.5

    def _quality(self, mag_s, ang_s, mag_f, ang_f) -> np.ndarray:
        eps = 1e-10
        g = np.where(
            (mag_s < eps) | (mag_f < eps), 0.0,
            np.where(mag_s > mag_f, mag_f / (mag_s + eps), mag_s / (mag_f + eps))
        )
        da = np.abs(ang_s - ang_f)
        da = np.where(da > np.pi, 2.0 * np.pi - da, da)
        da = np.minimum(da, np.pi - da)
        a = 1.0 - da / (np.pi / 2.0)
        Qg = self._Nrg / (1.0 + np.exp(-self._kg * (g - self._sigmag)))
        Qa = self._Nra / (1.0 + np.exp(-self._ka * (a - self._sigmaa)))
        return np.sqrt(Qg * Qa)

    def compute(self, fused, ir, vi):
        # Scale to MATLAB [0,255] space so published parameters apply directly
        mag_f, ang_f = _sobel_gradient(fused * 255.0)
        mag_i, ang_i = _sobel_gradient(ir * 255.0)
        mag_v, ang_v = _sobel_gradient(vi * 255.0)

        QAF = self._quality(mag_i, ang_i, mag_f, ang_f)
        QBF = self._quality(mag_v, ang_v, mag_f, ang_f)
        wtA = np.where(mag_i >= self._Td, mag_i ** self._Lg, self._wt_min)
        wtB = np.where(mag_v >= self._Td, mag_v ** self._Lg, self._wt_min)
        wt_sum = float((wtA + wtB).sum())
        if wt_sum < 1e-10:
            return 0.0
        na = ((mag_f > mag_i) & (mag_f > mag_v)).astype(np.float32)
        return float((na * ((1.0 - QAF) * wtA + (1.0 - QBF) * wtB)).sum() / wt_sum)


class FMIw(BaseMetric):
    """
    Feature Mutual Information using Haar wavelet detail sub-bands (FMI_w).
    Computes MI on the concatenated LH + HL + HH coefficients and averages
    over both source pairings.
    FMI_w = (MI(feat_F, feat_IR) + MI(feat_F, feat_VI)) / 2
    """

    name = 'FMI_w'
    higher_is_better = True

    def __init__(self, bins: int = 256):
        self.bins = bins

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-10)

    def compute(self, fused, ir, vi):
        ff = self._norm(_haar_detail(fused))
        fi = self._norm(_haar_detail(ir))
        fv = self._norm(_haar_detail(vi))
        return float(
            (_mutual_info(ff, fi, self.bins)
             + _mutual_info(ff, fv, self.bins)) / 2.0
        )


class NCIE(BaseMetric):
    """
    Nonlinear Correlation Information Entropy (Liu et al., TPAMI 2012).
    Builds the 3×3 nonlinear correlation matrix of {IR, VI, Fused} using
    r_ij = sqrt(1 - exp(-2*MI(i,j))), then measures the normalised entropy
    of its eigenvalue distribution.  A well-fused image yields highly
    structured eigenvalues → lower entropy → NCIE closer to 1.
    NCIE ∈ [0, 1], higher is better.
    """

    name = 'NCIE'
    higher_is_better = True

    def compute(self, fused, ir, vi):
        bins = 256
        # Nonlinear correlation coefficient: r = sqrt(1 - exp(-2*MI))
        # per Liu et al. (TPAMI 2012), "Objective assessment of multiresolution
        # image fusion algorithms for context enhancement in night vision"
        mi_fi = max(0.0, _mutual_info(fused, ir, bins))
        mi_fv = max(0.0, _mutual_info(fused, vi, bins))
        mi_iv = max(0.0, _mutual_info(ir, vi, bins))
        r_fi = float(np.sqrt(1.0 - np.exp(-2.0 * mi_fi)))
        r_fv = float(np.sqrt(1.0 - np.exp(-2.0 * mi_fv)))
        r_iv = float(np.sqrt(1.0 - np.exp(-2.0 * mi_iv)))
        R = np.array([[1.0, r_iv, r_fi],
                      [r_iv, 1.0, r_fv],
                      [r_fi, r_fv, 1.0]], dtype=np.float64)
        eigvals = np.abs(np.linalg.eigvals(R).real)
        total = eigvals.sum()
        if total < 1e-10:
            return 0.0
        p = eigvals / total
        p = p[p > 1e-10]
        H = float(-np.sum(p * np.log2(p)))
        return float(np.clip(1.0 - H / math.log2(3), 0.0, 1.0))


# Register the six built-in metrics
for _cls in (SSIM, MutualInformation, Qabf, FMIw, Nabf, NCIE):
    METRIC_REGISTRY[_cls.name] = _cls


# ---------------------------------------------------------------------------
# MetricSuite
# ---------------------------------------------------------------------------

_DEFAULT_METRIC_CLASSES: tuple[type[BaseMetric], ...] = (
    SSIM, MutualInformation, Qabf, FMIw, Nabf, NCIE
)


class MetricSuite:
    """
    Evaluates multiple metrics together and accumulates scores across images
    for dataset-level averages.

    Single-image usage
    ------------------
        suite  = MetricSuite()
        scores = suite(fused, ir, vi)   # -> dict[str, float]

    Dataset-level accumulation
    --------------------------
        suite = MetricSuite()
        suite.reset()
        for batch in dataloader:
            fused = method.fuse(batch['ir'], batch['vi'])
            suite.update(fused, batch['ir'], batch['vi'])
        print(suite.summary())          # -> dict[str, float]

    Custom metric set
    -----------------
        suite = MetricSuite(metrics=[SSIM(), MyMetric()])

    Adding a new metric to the default suite
    -----------------------------------------
    Append the new class to _DEFAULT_METRIC_CLASSES at module level, or
    pass a custom list to MetricSuite(metrics=...).
    """

    def __init__(self, metrics: list[BaseMetric] | None = None):
        self.metrics: list[BaseMetric] = (
            [cls() for cls in _DEFAULT_METRIC_CLASSES]
            if metrics is None else list(metrics)
        )
        self._accum: dict[str, list[float]] = {}
        self.reset()

    def __call__(
        self,
        fused: np.ndarray | torch.Tensor,
        ir:    np.ndarray | torch.Tensor,
        vi:    np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        """Compute all metrics for one image triplet."""
        return {m.name: m(fused, ir, vi) for m in self.metrics}

    def update(
        self,
        fused: np.ndarray | torch.Tensor,
        ir:    np.ndarray | torch.Tensor,
        vi:    np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        """Compute, record, and return per-metric scores for one triplet."""
        scores = self(fused, ir, vi)
        for name, val in scores.items():
            self._accum.setdefault(name, []).append(val)
        return scores

    def reset(self) -> None:
        """Clear all accumulated scores."""
        self._accum = {m.name: [] for m in self.metrics}

    def summary(self) -> dict[str, float]:
        """Dataset-level mean per metric. Returns NaN for empty accumulators."""
        return {
            name: float(np.mean(vals)) if vals else float('nan')
            for name, vals in self._accum.items()
        }

    def __repr__(self) -> str:
        return f"MetricSuite(metrics={[m.name for m in self.metrics]})"


# ---------------------------------------------------------------------------
# Model complexity: Params (M) and FLOPs (G)   [Fig. 6, BSPFusion paper]
# ---------------------------------------------------------------------------

# --- Manual FLOPs handler registry -----------------------------------------

_FLOP_HANDLERS: dict[type, Callable] = {}


def _register_flop(*module_types: type):
    """Decorator to associate a FLOPs counter with one or more nn.Module types."""
    def decorator(fn: Callable) -> Callable:
        for t in module_types:
            _FLOP_HANDLERS[t] = fn
        return fn
    return decorator


@_register_flop(nn.Conv1d, nn.Conv2d, nn.Conv3d)
def _conv_flops(m: nn.Conv2d, inp, out) -> int:
    N = inp[0].shape[0]
    out_spatial = math.prod(out.shape[2:])
    k_spatial   = math.prod(m.kernel_size)
    mac = N * m.out_channels * out_spatial * (m.in_channels // m.groups) * k_spatial
    if m.bias is not None:
        mac += N * m.out_channels * out_spatial
    return 2 * mac


@_register_flop(nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
def _convtranspose_flops(m: nn.ConvTranspose2d, inp, out) -> int:
    x = inp[0]
    N           = x.shape[0]
    in_spatial  = math.prod(x.shape[2:])
    k_spatial   = math.prod(m.kernel_size)
    mac = N * m.in_channels * in_spatial * (m.out_channels // m.groups) * k_spatial
    if m.bias is not None:
        mac += N * m.out_channels * math.prod(out.shape[2:])
    return 2 * mac


@_register_flop(nn.Linear)
def _linear_flops(m: nn.Linear, inp, out) -> int:
    batch_ops = inp[0].numel() // inp[0].shape[-1]
    mac = batch_ops * m.in_features * m.out_features
    if m.bias is not None:
        mac += batch_ops * m.out_features
    return 2 * mac


@_register_flop(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
def _bn_flops(m, inp, out) -> int:
    # eval: 1 multiply + 1 add per element; train: adds mean/var computation
    return (6 if m.training else 2) * inp[0].numel()


@_register_flop(nn.LayerNorm)
def _layernorm_flops(m: nn.LayerNorm, inp, out) -> int:
    return 5 * inp[0].numel()   # mean + var + norm + scale + bias


@_register_flop(nn.MultiheadAttention)
def _mha_flops(m: nn.MultiheadAttention, inp, out) -> int:
    """
    Count all MHA FLOPs: in-projections (Q/K/V) + attention + out-projection.
    out_proj is an nn.Linear child but is excluded from separate counting
    by _handler_modules, so it must be counted explicitly here.
    """
    query = inp[0]
    key   = inp[1] if (len(inp) > 1 and inp[1] is not None) else inp[0]

    if getattr(m, 'batch_first', False):
        N, L, _ = query.shape
        S = key.shape[1]
    else:
        L, N, _ = query.shape
        S = key.shape[0]

    E = m.embed_dim

    # Q proj (L×E), K proj (S×E), V proj (S×E)  — via in_proj_weight / F.linear
    proj_flops = 2 * N * E * E * (L + 2 * S)

    # Attention: QKᵀ (L×E × E×S) + weighted-sum AV (L×S × S×E)
    attn_flops = 4 * N * L * S * E

    # out_proj (nn.Linear child; not counted by _linear_flops due to recursion stop)
    out_flops = 2 * N * L * E * E
    if m.out_proj.bias is not None:
        out_flops += N * L * E

    return proj_flops + attn_flops + out_flops


# --- FLOPs counting helpers -------------------------------------------------

def _handler_modules(root: nn.Module) -> list[nn.Module]:
    """
    DFS traversal returning exactly the modules that should carry FLOPs hooks.
    Recursion stops at each handler module so its descendants are not counted
    twice — the handler's counter must include all sub-operations.
    """
    result: list[nn.Module] = []

    def _visit(m: nn.Module) -> None:
        if type(m) in _FLOP_HANDLERS:
            result.append(m)          # stop here; handler covers sub-ops
        else:
            for child in m.children():
                _visit(child)

    _visit(root)
    return result


def _manual_flop_count(model: nn.Module, dummy_inputs: tuple) -> int:
    """FLOPs via forward hooks on registered module types (manual fallback)."""
    total: list[int] = [0]
    hooks = []

    for m in _handler_modules(model):
        handler = _FLOP_HANDLERS[type(m)]
        def _hook(mod, inp, out, _h=handler):
            total[0] += _h(mod, inp, out)
        hooks.append(m.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(*dummy_inputs)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return total[0]


def _count_flops(model: nn.Module, dummy_inputs: tuple) -> int:
    """
    FLOPs = 2 × MACs for one forward pass.
    Primary: torch.utils.flop_counter.FlopCounterMode (handles all torch ops).
    Fallback: manual hook-based counting for Conv/Linear/BN/MHA.

    Important: FlopCounterMode is run in train() mode even if the model was in
    eval() mode.  In PyTorch 2.x, eval-mode MultiheadAttention uses a fused
    scaled_dot_product_attention kernel that bypasses torch dispatch, making it
    invisible to FlopCounterMode.  The forward arithmetic is identical in both
    modes; only the implementation path differs.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        was_training = model.training
        model.train()  # avoid eval-mode fused SDPA path (see docstring)
        try:
            with FlopCounterMode(display=False) as fc:
                with torch.no_grad():
                    model(*dummy_inputs)
            return fc.get_total_flops()
        finally:
            model.train(was_training)   # restore original state
    except Exception:
        return _manual_flop_count(model, dummy_inputs)


# --- Public class -----------------------------------------------------------

class ModelComplexity:
    """
    Parameter count and FLOPs for any nn.Module — the Params(M) / FLOPs(G)
    axes in Fig. 6 of the BSPFusion paper (Inf. Fusion 135, 2026).

    FLOPs are counted as 2 × MACs (1 multiply + 1 accumulate = 2 FLOPs).
    Parameter count includes both trainable and frozen weights.

    Usage
    -----
        mc = ModelComplexity(
            model,
            torch.zeros(1, 1, 256, 256),   # IR dummy  (batch=1, grayscale)
            torch.zeros(1, 3, 256, 256),    # VI dummy  (batch=1, RGB)
        )
        print(f"Params: {mc.params_M:.2f} M,  FLOPs: {mc.flops_G:.3f} G")

    Notes
    -----
    * Use batch_size=1 dummy tensors to obtain per-image FLOPs.
    * Dummy tensor values are irrelevant; only shapes affect FLOPs.
    * The model is temporarily set to eval() during the dummy forward pass
      and restored to its original training state afterwards.
    * If the model lives on GPU, pass GPU dummy tensors accordingly.
    """

    def __init__(self, model: nn.Module, *dummy_inputs: torch.Tensor):
        params = list(model.parameters())
        self._total      = sum(p.numel() for p in params)
        self._trainable  = sum(p.numel() for p in params if p.requires_grad)
        self._flops      = _count_flops(model, dummy_inputs)

    @property
    def params_M(self) -> float:
        """Total parameter count (trainable + frozen) in millions."""
        return self._total / 1e6

    @property
    def params_trainable_M(self) -> float:
        """Trainable-only parameter count in millions."""
        return self._trainable / 1e6

    @property
    def flops_G(self) -> float:
        """FLOPs (2 × MACs) for one forward pass, in giga-FLOPs."""
        return self._flops / 1e9

    def __repr__(self) -> str:
        return (f"ModelComplexity("
                f"params={self.params_M:.2f}M, "
                f"flops={self.flops_G:.3f}G)")
