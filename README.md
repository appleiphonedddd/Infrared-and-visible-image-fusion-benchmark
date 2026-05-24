<div align="center">

# Infrared and Visible Image Fusion Benchmark

**A unified, reproducible evaluation framework for infrared–visible image fusion research**

<p>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick%20Start-→-brightgreen?style=flat-square" alt="Quick Start"></a>
  <a href="#results"><img src="https://img.shields.io/badge/Benchmark%20Results-→-blue?style=flat-square" alt="Results"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-13.x-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

</div>

---

## What is This?

Infrared and visible image fusion (IVIF) merges the **thermal sensitivity of IR sensors** with the **textural richness of RGB cameras**, producing images that benefit downstream tasks such as semantic segmentation, object detection, and scene understanding.

Dozens of fusion methods have been proposed in recent years, yet direct comparison remains difficult: results are reported on inconsistent datasets, metrics, and preprocessing pipelines.

**This benchmark provides a single, fair, end-to-end evaluation harness:**

- **One command** runs any registered method on any registered dataset
- **Six standard metrics** computed with identical pre/postprocessing across all methods
- **Complexity reporting** (Params + FLOPs) alongside image quality scores
- **Registry-based extension** — drop in a new method or dataset with zero changes to existing files

---

## News

- **2026-05** — LLVIP dataset added (15488 pairs, train/test split).
- **2026-05** — Initial public release. SeAFusion supported on M3FD, MSRS, RoadScene, and TNO.

---

## Results

> All scores are dataset-level means. ↑ = higher is better, ↓ = lower is better.
> Models are evaluated at native resolution with batch size 1 on a single GPU.

### M3FD

| Method | SSIM ↑ | MI ↑ | Q_abf ↑ | N_abf ↓ | FMI_w ↑ | NCIE ↑ | Params (M) | FLOPs (G) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SeAFusion | — | — | — | — | — | — | — | — |

*Populate this table by running `python main.py eval` on each method and dataset.*

---

## Framework Architecture

The benchmark is structured as four composable layers:

```
╔══════════════╗     ╔══════════════════╗     ╔═════════════╗     ╔══════════════╗
║  Dataset     ║────▶║  BaseMethod      ║────▶║ MetricSuite ║────▶║   Summary    ║
║  (IR + VI)   ║     ║  preprocess      ║     ║  SSIM / MI  ║     ║  CSV / JSON  ║
╚══════════════╝     ║  _fuse           ║     ║  Q_abf ...  ║     ╚══════════════╝
                     ║  postprocess     ║     ╚═════════════╝
                     ╚══════════════════╝
                              ▲
                     ╔════════╩═════════╗
                     ║ BaseFusionTrainer ║
                     ║  (optional train) ║
                     ╚══════════════════╝
```

| Layer | Base Class | Location |
|:--|:--|:--|
| Dataset | `BaseFusionDataset` | `base/base_data_loader.py` |
| Loader | `FusionDataLoader` | `base/base_fusion_loader.py` |
| Method | `BaseMethod` | `base/base_method.py` |
| Trainer | `BaseFusionTrainer` | `base/base_trainer.py` |
| Metrics | `MetricSuite` | `utils/metrics.py` |

---

## Quick Start

### Prerequisites

| Requirement | Version |
|:--|:--|
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | NVIDIA GPU with CUDA 13.x |
| Python | 3.11+ (Conda recommended) |

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Infrared-and-visible-image-fusion-benchmark

# Create and activate the Conda environment
conda env create -f env.yaml
conda activate IF
```

### Evaluate a Method

```bash
python main.py eval --method SeAFusion --checkpoint checkpoint/SeAFusion.pth --dataset RoadScene --data-root dataset/RoadScene --save-dir results/
```

### Train a Method

```bash
python main.py train \
  --method   SeAFusion \
  --dataset  MSRS \
  --data-root dataset/MSRS \
  --split    train \
  --config   config.json
```

> **SeAFusion training** requires a `config.json` specifying at minimum `seg_net` — a path to a segmentation model saved with `torch.save(model, path)`. Optional keys: `lr_fusion`, `lr_seg`, `M`, `p`, `q`, `gamma`, `save_dir`, `save_period`, `resume`.

### Complexity Report

```bash
python main.py complexity \
  --method     SeAFusion \
  --resolution 256 256
```

---

## Supported Datasets

| Dataset | IR Channels | VI Channels | Pairs | Split |
|:--|:-:|:-:|:-:|:--|
| **M3FD** | 3 (RGB-stored IR) | 3 | 300 | All as test |
| **MSRS** | 1 | 3 | — | `train` / `test` |
| **RoadScene** | 1 | 3 | — | From `meta/pred.txt` |
| **TNO** | 1 | 1 | — | From `meta/pred.txt` |
| **LLVIP** | 3 (RGB-stored IR) | 3 | 15488 | `train` (12025) / `test` (3463) |

> **Note on M3FD and LLVIP:** IR images are stored as 3-channel RGB even though they are infrared (grayscale content). Method wrappers should convert to grayscale when the model expects 1-channel input.

<details>
<summary><b>Expected Directory Layout</b></summary>

```
dataset/
├── M3FD/
│   ├── Ir/                        # infrared PNGs
│   └── Vis/                       # visible PNGs
├── MSRS/
│   ├── train/
│   │   ├── ir/
│   │   ├── vi/
│   │   └── Segmentation_labels/   # required for SeAFusion training
│   └── test/
│       ├── ir/
│       └── vi/
├── RoadScene/
│   ├── ir/
│   ├── vi/
│   └── meta/pred.txt
├── TNO/
│   ├── ir/
│   ├── vi/
│   └── meta/pred.txt
└── LLVIP/
    ├── infrared/
    │   ├── train/
    │   └── test/
    └── visible/
        ├── train/
        └── test/
```

</details>

---

## Supported Methods

| Method | Venue | Key Idea | Paper |
|:--|:--|:--|:--|
| **SeAFusion** | Inf. Fusion 2022 | Semantic-aware fusion via alternating fusion / segmentation training | [Tang et al.](https://doi.org/10.1016/j.inffus.2021.12.001) |

---

## Metrics

All metrics operate on **grayscale** after auto-converting any RGB input.

| Metric | Direction | Description |
|:--|:-:|:--|
| `SSIM` | ↑ | Average SSIM(F, IR) + SSIM(F, VI) |
| `MI` | ↑ | Mutual information sum I(F; IR) + I(F; VI) |
| `Q_abf` | ↑ | Gradient magnitude + orientation preservation (Xydeas & Petrovic, 2000) |
| `N_abf` | ↓ | Fraction of fused gradient exceeding both sources (artifact indicator) |
| `FMI_w` | ↑ | Feature MI on Haar wavelet detail sub-bands |
| `NCIE` | ↑ | Nonlinear correlation information entropy |

---

## Extending the Framework

The framework uses a **decorator-based registry** — new components self-register at import time with no changes to existing files.

<details>
<summary><b>Add a New Method</b></summary>

```python
# 1. model/mynet.py
from base.base_model import BaseModel

class MyNet(BaseModel):
    def forward(self, ir, vi):
        ...

# 2. method/mymethod/method.py
from base.base_method import BaseMethod, register_method
from model.mynet import MyNet

@register_method('MyMethod')
class MyMethod(BaseMethod):
    def _build_model(self) -> MyNet:
        return MyNet()

    def _fuse(self, ir, vi):
        return self.model(ir, vi)

# 3. method/mymethod/trainer.py
from base.base_trainer import BaseFusionTrainer, register_trainer

@register_trainer('MyMethod')
def _make_trainer(method, train_loader, config):
    optimizer = torch.optim.Adam(method.model.parameters(), lr=config.get('lr', 1e-4))
    return MyTrainer(method, optimizer, train_loader, **config)

# 4. method/mymethod/__init__.py
from .method import MyMethod
from .trainer import _make_trainer
```

`method/__init__.py` auto-discovers all subdirectories — nothing else to change.

</details>

<details>
<summary><b>Add a New Dataset</b></summary>

```python
# data_loader/mydata.py
from base.base_data_loader import BaseFusionDataset, register_dataset

@register_dataset('MyData')
class MyDataset(BaseFusionDataset):
    ir_channels = 1
    vi_channels = 3

    def _load_pairs(self):
        # Return list of (ir_path, vi_path, name)
        ...
```

`data_loader/__init__.py` auto-imports all `*.py` in the package — nothing else to change.

</details>

<details>
<summary><b>Add a Custom Metric</b></summary>

```python
from utils.metrics import MetricSuite, SSIM, MI

# Use a subset of built-in metrics
suite = MetricSuite(metrics=[SSIM(), MI()])

# Or pass any callable with signature metric(fused, ir, vi) -> float
suite = MetricSuite(metrics=[SSIM(), my_custom_metric])
```

</details>

---

## Citation

If this benchmark is useful for your research, please cite it as:

```bibtex
@misc{ivif_benchmark_2026,
  title   = {Infrared and Visible Image Fusion Benchmark},
  author  = {},
  year    = {2026},
  url     = {}
}
```

---

## Acknowledgements

We thank the authors of [SeAFusion](https://doi.org/10.1016/j.inffus.2021.12.001) and the maintainers of M3FD, MSRS, RoadScene, TNO, and LLVIP for providing public datasets and implementations.

---

<div align="center">
<sub>Built with PyTorch · Released under the MIT License</sub>
</div>
