<div align="center">

# Infrared and Visible Image Fusion Benchmark

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**A unified framework for evaluating infrared and visible image fusion methods across multiple datasets and metrics — reproducible, extensible, and research-ready.**

</div>

---

## Why This Benchmark

Infrared and visible image fusion combines the thermal sensitivity of IR sensors with the textural richness of RGB cameras. Dozens of methods exist, but they are evaluated on different datasets with different metrics, making fair comparison nearly impossible.

This framework provides:

- A **single pipeline** that runs any registered method on any dataset
- **Six standard metrics** computed identically across all methods
- **Model complexity reporting** (Params + FLOPs) alongside quality scores
- A clean registry pattern so new methods and datasets drop in with minimal boilerplate

---

## Architecture

The benchmark follows a four-layer design: **datasets → methods → trainers → metrics**.

```
BaseFusionDataset  →  FusionDataLoader  →  BaseMethod  →  MetricSuite
     (data)               (loader)         (inference)     (evaluation)
                                   ↕
                           BaseFusionTrainer
                              (training)
```

| Layer | Base Class | Location |
|---|---|---|
| Dataset | `BaseFusionDataset` | `base/base_data_loader.py` |
| Loader | `FusionDataLoader` | `base/base_fusion_loader.py` |
| Method | `BaseMethod` | `base/base_method.py` |
| Trainer | `BaseFusionTrainer` | `base/base_trainer.py` |
| Metrics | `MetricSuite` | `utils/metrics.py` |

---

## Prerequisites

| Requirement | Version |
|---|---|
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | NVIDIA GPU with CUDA 13.x |
| Python | 3.11 (Conda) / 3.12 (Docker) |

---

## Installation

### Option 1 — Conda

```bash
conda env create -f env.yaml
conda activate IF
```

### Option 2 — Docker

```bash
docker build -t fusion-bench .
docker run --gpus all -v $(pwd)/dataset:/workspace/dataset fusion-bench
```

---

## Supported Datasets

| Dataset | IR Channels | VI Channels | Size | Split |
|---|---|---|---|---|
| **M3FD** | 3 (RGB-stored IR) | 3 | 300 pairs | All as test |
| **MSRS** | 1 (grayscale) | 3 | — | `train` / `test` |
| **RoadScene** | 1 | 3 | — | Pair list from `meta/pred.txt` |
| **TNO** | 1 | 1 (grayscale) | — | Pair list from `meta/pred.txt` |

> **M3FD note:** IR images are stored as 3-channel RGB PNGs even though they are infrared. Method wrappers must convert to grayscale if the model expects 1-channel input.

### Dataset Layout

```
dataset/
├── M3FD/
│   ├── Ir/        # infrared PNGs
│   └── Vis/       # visible PNGs
├── MSRS/
│   ├── train/
│   │   ├── ir/
│   │   ├── vi/
│   │   └── Segmentation_labels/   # optional, needed for SeAFusion training
│   └── test/
│       ├── ir/
│       └── vi/
├── RoadScene/
│   ├── ir/
│   ├── vi/
│   └── meta/pred.txt
└── TNO/
    ├── ir/
    ├── vi/
    └── meta/pred.txt
```

---

## Supported Methods

| Method | Key Idea | Reference |
|---|---|---|
| **SeAFusion** | Semantic-aware fusion via alternating fusion/segmentation training | [Tang et al., 2022](https://doi.org/10.1016/j.inffus.2021.12.001) |

---

## Supported Metrics

| Metric | Direction | Description |
|---|---|---|
| `SSIM` | ↑ | Average SSIM(F, IR) + SSIM(F, VI) |
| `MI` | ↑ | Mutual information sum I(F; IR) + I(F; VI) |
| `Q_abf` | ↑ | Gradient magnitude + orientation preservation (Xydeas & Petrovic 2000) |
| `N_abf` | ↓ | Fraction of fused gradient exceeding both sources (artifact indicator) |
| `FMI_w` | ↑ | Feature MI on Haar wavelet detail sub-bands |
| `NCIE` | ↑ | Nonlinear correlation information entropy |

All metrics operate on grayscale after auto-converting any RGB input.

---

## Usage

### Evaluate

Fuse images and compute all six metrics:

```bash
python main.py eval \
  --method SeAFusion \
  --checkpoint path/to/weights.pth \
  --dataset M3FD \
  --data-root dataset/M3FD \
  --save-dir results/seafusion_m3fd   # optional: save fused images
```

### Train

```bash
python main.py train \
  --method SeAFusion \
  --dataset MSRS \
  --data-root dataset/MSRS \
  --split train \
  --config config.json
```

SeAFusion training requires a `config.json` with at least `seg_net` (path to a segmentation model saved with `torch.save(model, path)`). Optional keys: `lr_fusion`, `lr_seg`, `M`, `p`, `q`, `gamma`, `save_dir`, `save_period`, `resume`.

### Model Complexity

```bash
python main.py complexity \
  --method SeAFusion \
  --resolution 256 256
```

---

## Extending the Framework

The framework uses a **registry pattern** — new components register themselves with a decorator and are discovered automatically.

### Add a New Method

```python
# 1. model/mynet.py — the nn.Module
from base.base_model import BaseModel

class MyNet(BaseModel):
    def forward(self, ir, vi):
        ...

# 2. method/mymethod/method.py — evaluation wrapper
from base.base_method import BaseMethod, register_method
from model.mynet import MyNet

@register_method('MyMethod')
class MyMethod(BaseMethod):
    def _build_model(self) -> MyNet:
        return MyNet()

    def _fuse(self, ir, vi):
        return self.model(ir, vi)

# 3. method/mymethod/trainer.py — training factory
from base.base_trainer import BaseFusionTrainer, register_trainer

@register_trainer('MyMethod')
def _make_mymethod_trainer(method, train_loader, config):
    optimizer = torch.optim.Adam(method.model.parameters(), lr=config.get('lr', 1e-4))
    return MyTrainer(method, optimizer, train_loader, **config)

# 4. method/mymethod/__init__.py — trigger registration
from .method import MyMethod
from .trainer import _make_mymethod_trainer
```

No changes to existing files are needed — `method/__init__.py` auto-discovers all subdirectories.

### Add a New Dataset

```python
# data_loader/mydata.py
from base.base_data_loader import BaseFusionDataset, register_dataset

@register_dataset('MyData')
class MyDataset(BaseFusionDataset):
    ir_channels = 1
    vi_channels = 3

    def _load_pairs(self):
        # return list of (ir_path, vi_path, name)
        ...
```

No changes to existing files — `data_loader/__init__.py` auto-imports all `*.py` files in the package.
