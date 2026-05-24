# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash

# Evaluate a method on a dataset
python main.py eval --method SeAFusion --checkpoint path/to/weights.pth \
    --dataset M3FD --data-root dataset/M3FD [--save-dir results/]

# Train a method
python main.py train --method SeAFusion --dataset MSRS --data-root dataset/MSRS \
    --config config.json [--resume path/to/checkpoint.pth]

# Report parameter count and FLOPs
python main.py complexity --method SeAFusion --resolution 256 256
```

Available datasets: `M3FD`, `MSRS`, `RoadScene`, `TNO`, `LLVIP`  
Available methods: populated from `method/` at import time — run `python main.py eval --help` for the current list.

## Architecture

The benchmark follows a four-layer design: **datasets → methods → trainers → metrics**.

### Data Layer (`base/base_data_loader.py`, `data_loader/`)

`BaseFusionDataset` is the abstract dataset base. Subclasses implement `_load_pairs()` returning `list[tuple[ir_path, vi_path, name]]`. `__getitem__` always yields `{'ir': Tensor, 'vi': Tensor, 'name': str}` with images as float32 `[0, 1]`.

`FusionDataLoader` (`base/base_fusion_loader.py`) wraps `DataLoader` with `shuffle=False` and `batch_size=1` by default — order must be identical across methods for fair comparison.

Concrete datasets live in `data_loader/data_loaders.py`:

| Dataset | IR channels | VI channels | Split logic |
|---------|-------------|-------------|-------------|
| `M3FDDataset` | 3 (RGB-stored IR) | 3 | All 300 pairs as test |
| `MSRSDataset` | 1 (grayscale) | 3 | `train`/`test` splits; optional `seg_label` |
| `RoadSceneDataset` | 1 | 3 | Pair list from `meta/pred.txt` |
| `TNODataset` | 1 | 1 (grayscale) | Pair list from `meta/pred.txt` |
| `LLVIPDataset` | 3 (RGB-stored IR) | 3 | `train`/`test` splits; 12025/3463 pairs |

M3FD and LLVIP store IR as 3-channel RGB JPG/PNG even though it is infrared — method wrappers must convert to grayscale if the model expects 1-channel input. TNO is the only dataset where VI is also grayscale.

`data_loader/__init__.py` auto-imports all `*.py` files in the package so `@register_dataset` decorators fire without changes to existing files.

### Method Layer (`base/base_method.py`, `method/`, `model/`)

`BaseMethod` wraps an `nn.Module` with a fixed evaluation pipeline:

```
fuse(ir, vi)  →  preprocess  →  _fuse  →  postprocess  →  CPU tensor [0,1]
```

`vi_original` (the raw CPU tensor before `preprocess`) is forwarded to `postprocess` so YCbCr-based methods can merge the fused Y channel back with Cb/Cr from the original visible image.

Subclasses **must** implement `_build_model()` and `_fuse()`. Override `preprocess()`/`postprocess()` for colour-space transforms.

Methods are registered with `@register_method('Name')` and instantiated via `build_method('Name', device='cuda')`. The registry lives in `METHOD_REGISTRY` in `base/base_method.py`.

`method/__init__.py` auto-imports every subdirectory with an `__init__.py`, so adding a new method directory requires no changes to existing files.

Each method in `method/<name>/` contains:
- `model.py` — re-exports the `nn.Module` from `model/<name>.py`
- `method.py` — `BaseMethod` subclass decorated with `@register_method`
- `trainer.py` — trainer factory decorated with `@register_trainer`
- `__init__.py` — imports the method and trainer classes to trigger registration

The `nn.Module` implementations live in `model/` (e.g., `model/seafusion.py`) and subclass `BaseModel` from `base/base_model.py`.

### Trainer Layer (`base/base_trainer.py`, `method/<name>/trainer.py`)

`BaseFusionTrainer` provides a checkpoint-aware training loop. Subclasses implement `_train_epoch(epoch) -> dict`. Checkpoints saved by `BaseFusionTrainer._save_checkpoint` use a `state_dict` key compatible with `BaseMethod.load_checkpoint`.

Trainers are registered with `@register_trainer('Name')` as factory functions with the signature `fn(method, train_loader, config) -> BaseFusionTrainer`. The `config` dict comes from a JSON file passed via `--config`.

`SeAFusionTrainer` implements Algorithm 1 from the paper: alternating between `p` gradient steps on the fusion network and `q` steps on a frozen segmentation network per outer iteration `M`. The segmentation network is loaded as a full `torch.save(model)` object (not just state_dict) from `config['seg_net']`.

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

`ModelComplexity` counts Params (M) and FLOPs (G) using `FlopCounterMode`. Always uses `batch_size=1` dummy tensors. The model is temporarily forced into `train()` mode to avoid PyTorch 2.x's fused SDPA kernel hiding MHA FLOPs.

### Loss Layer (`loss/`)

Method-specific losses live in `loss/<name>.py`. `loss/seafusion.py` contains `IntensityLoss`, `TextureLoss`, `SeAFusionLoss` (joint fusion loss, Eq. 10), and `SemanticLoss` (cross-entropy on main and auxiliary segmentation heads).

## Adding a New Method

1. Create `model/<name>.py` with the `nn.Module` subclassing `BaseModel`.
2. Create `method/<name>/model.py` re-exporting from `model/<name>.py`.
3. Create `method/<name>/method.py` with a `BaseMethod` subclass decorated `@register_method('Name')`.
4. Create `method/<name>/trainer.py` with a factory function decorated `@register_trainer('Name')`.
5. Create `method/<name>/__init__.py` importing the method and trainer classes.
6. If the method has custom losses, add `loss/<name>.py`.

## Adding a New Dataset

Subclass `BaseFusionDataset`, declare `ir_channels` / `vi_channels`, implement `_load_pairs()`, and decorate with `@register_dataset('Name')`. Place the file in `data_loader/` — it will be auto-imported.

## Dataset Layout

```
dataset/
└── M3FD/
    ├── Ir/   # infrared PNGs
    └── Vis/  # visible PNGs
```

MSRS expects `{root}/{split}/ir/`, `{root}/{split}/vi/`, and optionally `{root}/{split}/Segmentation_labels/`.  
RoadScene and TNO expect `{root}/ir/`, `{root}/vi/`, `{root}/meta/pred.txt`.  
LLVIP expects `{root}/infrared/{split}/`, `{root}/visible/{split}/` with `.jpg` files.
