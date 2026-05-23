# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

The benchmark follows a three-layer design: **datasets → methods → metrics**.

### Data Layer (`base/`, `data_loader/`)

`BaseFusionDataset` (`base/base_data_loader.py`) is the abstract dataset base. All concrete datasets implement `_load_pairs()` returning `list[tuple[ir_path, vi_path, name]]`. `__getitem__` always yields `{'ir': Tensor, 'vi': Tensor, 'name': str}` with images as float32 `[0, 1]`.

`FusionDataLoader` (`base/base_fusion_loader.py`) wraps `DataLoader` with `shuffle=False` and `batch_size=1` by default — order must be identical across methods for fair comparison, and batch=1 avoids padding for variable-resolution datasets.

Concrete datasets live in `data_loader/datasets/data_loaders.py`:

| Dataset | IR channels | VI channels | Split logic |
|---------|-------------|-------------|-------------|
| `M3FDDataset` | 3 (RGB-stored IR) | 3 | All 300 pairs as test |
| `MSRSDataset` | 1 (grayscale) | 3 | `train`/`test` splits; optional `seg_label` |
| `RoadSceneDataset` | 1 | 3 | Pair list from `meta/pred.txt` |
| `TNODataset` | 1 | 1 (grayscale) | Pair list from `meta/pred.txt` |

M3FD stores IR as 3-channel RGB PNG even though it is infrared — method wrappers must convert to grayscale if the model expects 1-channel input.

### Method Layer (`base/base_method.py`, `method/`)

`BaseMethod` wraps an `nn.Module` with a fixed evaluation pipeline:

```
fuse(ir, vi)  →  preprocess  →  _fuse  →  postprocess  →  CPU tensor [0,1]
```

`vi_original` (the raw CPU tensor before `preprocess`) is forwarded to `postprocess` so YCbCr-based methods can merge the fused Y channel back with Cb/Cr from the original visible image.

Subclasses **must** implement `_build_model()` and `_fuse()`. Override `preprocess()`/`postprocess()` for colour-space transforms.

Methods are registered with `@register_method('Name')` and instantiated via `build_method('Name', device='cuda')`. The registry lives in `METHOD_REGISTRY` in `base/base_method.py`.

Each method in `method/<name>/` contains two files:
- `model.py` — `nn.Module` subclassing `BaseModel`
- `method.py` — `BaseMethod` subclass decorated with `@register_method`

Currently `SeAFusionNet` and `TarDALNet` in `model.py` are architecture stubs (`raise NotImplementedError`).

### Metrics Layer (`utils/metrics.py`)

Six built-in metrics, all operating on grayscale after auto-converting any RGB input:

| Metric | Direction | Description |
|--------|-----------|-------------|
| `SSIM` | ↑ | Average SSIM(F,IR) + SSIM(F,VI) |
| `MI` | ↑ | Mutual information sum I(F;IR) + I(F;VI) |
| `Q_abf` | ↑ | Gradient magnitude+orientation preservation (Xydeas & Petrovic 2000) |
| `N_abf` | ↓ | Fraction of fused gradient exceeding both sources (artifact indicator) |
| `FMI_w` | ↑ | Feature MI on Haar wavelet detail sub-bands |
| `NCIE` | ↑ | Nonlinear correlation information entropy |

`MetricSuite` evaluates all six together and accumulates per-image scores for dataset-level `summary()` (means). Custom metric sets: `MetricSuite(metrics=[SSIM(), MyMetric()])`.

`ModelComplexity` counts Params (M) and FLOPs (G) using `FlopCounterMode` with a manual hook fallback. Always use `batch_size=1` dummy tensors. The model is temporarily forced into `train()` mode to avoid PyTorch 2.x's fused SDPA kernel hiding MHA FLOPs from `FlopCounterMode`.

### Adding a New Method

1. Create `method/<name>/model.py` with an `nn.Module` subclassing `BaseModel`.
2. Create `method/<name>/method.py` with a `BaseMethod` subclass decorated `@register_method('Name')`.
3. Implement `_build_model()` returning the model, and `_fuse(ir, vi)` for the forward pass.
4. Override `preprocess`/`postprocess` if needed (e.g., RGB→YCbCr).
5. Add `__init__.py` importing the method class so the decorator runs on import.

### Adding a New Dataset

Subclass `BaseFusionDataset`, declare `ir_channels` / `vi_channels`, implement `_load_pairs()`. Register under `data_loader/datasets/data_loaders.py`.

## Dataset Layout

```
dataset/
└── M3FD/
    ├── Ir/   # infrared PNGs
    └── Vis/  # visible PNGs
```

MSRS expects `{root}/{split}/ir/`, `{root}/{split}/vi/`, and optionally `{root}/{split}/Segmentation_labels/`.
RoadScene and TNO expect `{root}/ir/`, `{root}/vi/`, `{root}/meta/pred.txt`.

## Known Issues

`utils/metrics.py` has `from __future__ import annotations` placed after other imports (line 12), which is a `SyntaxError` in Python < 3.10 and a lint error. It must be the first statement in the file.
