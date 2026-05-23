<div align="center">

# Infrared and Visible Image Fusion Benchmark

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
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

The benchmark follows a three-layer design: **datasets → methods → metrics**.

```
BaseFusionDataset  →  FusionDataLoader  →  BaseMethod  →  MetricSuite
     (data)               (loader)         (inference)     (evaluation)
```

| Layer | Base Class | Location |
|---|---|---|
| Dataset | `BaseFusionDataset` | `base/base_data_loader.py` |
| Loader | `FusionDataLoader` | `base/base_fusion_loader.py` |
| Method | `BaseMethod` | `base/base_method.py` |
| Metrics | `MetricSuite` | `utils/metrics.py` |

---

## Prerequisites

| Requirement | Version |
|---|---|
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | NVIDIA GPU with CUDA 12.x |
| Python | 3.11 via Conda |
| Docker | Optional (for containerised runs) |

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
│   │   └── Segmentation_labels/   # optional
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
| **SeAFusion** | Semantic-aware fusion via attention | [Tang et al., 2022](https://doi.org/10.1016/j.inffus.2021.12.001) |
| **TarDAL** | Task-oriented adversarial learning | [Liu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html) |

> Architecture stubs are in place. Pretrained weight loading must be wired in `_build_model()` for each method.

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

## Quick Start

### Step 1 — Create Environment

```bash
conda env create -f env.yaml
conda activate IF
```

Or use the provided Docker image:

```bash
docker build -t fusion-bench .
docker run --gpus all -v $(pwd)/dataset:/workspace/dataset fusion-bench
```

### Step 2 — Run a Method

```bash
python main.py --method SeAFusion --dataset M3FD --data_root dataset/M3FD
```

```bash
python main.py --method TarDAL --dataset MSRS --data_root dataset/MSRS --split test
```

---

## Extending the Framework

The framework uses a **registry pattern** — new components register themselves with a decorator and are available immediately.

### Add a New Method

```python
# 1. Create method/mymethod/model.py
from base.base_model import BaseModel

class MyNet(BaseModel):
    def forward(self, ir, vi):
        ...

# 2. Create method/mymethod/method.py
from base.base_method import BaseMethod, register_method
from .model import MyNet

@register_method('MyMethod')
class MyMethod(BaseMethod):
    def _build_model(self) -> MyNet:
        return MyNet()

    def _fuse(self, ir, vi):
        return self.model(ir, vi)

# 3. method/mymethod/__init__.py — import so the decorator runs
from .method import MyMethod
```

### Add a New Dataset

```python
# data_loader/datasets/data_loaders.py
from base.base_data_loader import BaseFusionDataset

class MyDataset(BaseFusionDataset):
    ir_channels = 1
    vi_channels = 3

    def _load_pairs(self):
        # return list of (ir_path, vi_path, name)
        ...
```

