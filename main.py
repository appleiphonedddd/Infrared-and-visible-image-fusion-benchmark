import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

# Importing from `method` triggers auto-discovery of all subpackages,
# which registers every @register_method and @register_trainer decorator.
from method import build_method, METHOD_REGISTRY, build_trainer, TRAINER_REGISTRY

# Importing from `data_loader` triggers auto-discovery of all dataset modules,
# which registers every @register_dataset decorator.
from data_loader import build_dataset, DATASET_REGISTRY, FusionDataLoader

from utils.metrics import MetricSuite, ModelComplexity


def _default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ── eval ──────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    device = args.device or _default_device()

    m = build_method(args.method, device=device)
    m.load_checkpoint(args.checkpoint)

    dataset = build_dataset(args.dataset, root=args.data_root, split=args.split)
    loader = FusionDataLoader(dataset, num_workers=args.workers)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    suite = MetricSuite()
    metric_map = {met.name: met for met in suite.metrics}
    n = len(dataset)

    for i, batch in enumerate(loader, 1):
        ir, vi, name = batch['ir'], batch['vi'], batch['name'][0]
        fused = m.fuse(ir, vi)
        suite.update(fused, ir, vi)
        if save_dir:
            save_image(fused, save_dir / f'{name}.png')
        print(f'\r[{i}/{n}] {name:<40}', end='', flush=True)

    print()

    summary = suite.summary()
    col = 12
    print(f'\n{"Metric":<{col}} {"Mean":>10}')
    print('-' * (col + 11))
    for metric_name, val in summary.items():
        arrow = '↑' if metric_map[metric_name].higher_is_better else '↓'
        print(f'{metric_name:<{col}} {val:>10.4f}  {arrow}')


# ── train ─────────────────────────────────────────────────────────────────────

def cmd_train(args):
    config: dict = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # CLI args take precedence over config file values
    if args.save_dir:
        config['save_dir'] = args.save_dir
    if args.resume:
        config['resume'] = args.resume

    device = args.device or _default_device()

    m = build_method(args.method, device=device)

    dataset = build_dataset(
        args.dataset,
        root=args.data_root,
        split=args.split,
        include_seg=config.get('include_seg', False),
    )
    loader = FusionDataLoader(
        dataset,
        batch_size=config.get('batch_size', 1),
        num_workers=args.workers,
    )

    trainer = build_trainer(args.method, m, loader, config)
    trainer.train()


# ── complexity ────────────────────────────────────────────────────────────────

def cmd_complexity(args):
    device = args.device

    m = build_method(args.method, device=device)
    if args.checkpoint:
        m.load_checkpoint(args.checkpoint)

    H, W = args.resolution
    ir_raw = torch.zeros(1, args.ir_channels, H, W)
    vi_raw = torch.zeros(1, args.vi_channels, H, W)

    with torch.no_grad():
        ir_pre, vi_pre = m.preprocess(ir_raw, vi_raw)

    mc = ModelComplexity(m.model, ir_pre, vi_pre)
    print(f'Method : {m.name}')
    print(f'Params : {mc.params_M:.2f} M')
    print(f'FLOPs  : {mc.flops_G:.3f} G  (input {H}×{W})')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    # Registries are populated by the imports at the top of this file.
    methods  = sorted(METHOD_REGISTRY)
    datasets = sorted(DATASET_REGISTRY)
    trainable = sorted(TRAINER_REGISTRY)

    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Infrared & Visible Image Fusion Benchmark',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── eval ──────────────────────────────────────────────────────────────────
    ep = sub.add_parser('eval', help='Fuse images and compute metrics on a dataset')
    ep.add_argument('--method',     required=True, choices=methods,  help='Fusion method')
    ep.add_argument('--checkpoint', required=True,                   help='Path to weights file')
    ep.add_argument('--dataset',    required=True, choices=datasets, help='Dataset name')
    ep.add_argument('--data-root',  required=True, dest='data_root', help='Dataset root directory')
    ep.add_argument('--split',      default='test', choices=['train', 'test'],
                    help='Split for datasets that support it, e.g. MSRS (default: test)')
    ep.add_argument('--save-dir',   dest='save_dir', metavar='DIR',
                    help='Directory to save fused images (optional)')
    ep.add_argument('--device',     metavar='DEVICE',
                    help='torch device, e.g. cuda or cpu (default: cuda if available)')
    ep.add_argument('--workers',    type=int, default=4, metavar='N',
                    help='DataLoader worker processes (default: 4)')

    # ── train ─────────────────────────────────────────────────────────────────
    tp = sub.add_parser('train', help='Train a method on a dataset')
    tp.add_argument('--method',     required=True, choices=trainable, help='Fusion method')
    tp.add_argument('--dataset',    required=True, choices=datasets,  help='Dataset name')
    tp.add_argument('--data-root',  required=True, dest='data_root',  help='Dataset root directory')
    tp.add_argument('--config',     metavar='PATH',
                    help='JSON config file with method-specific training parameters')
    tp.add_argument('--split',      default='train', choices=['train', 'test'],
                    help='Split for datasets that support it (default: train)')
    tp.add_argument('--save-dir',   dest='save_dir', metavar='DIR',
                    help='Directory to save checkpoints (overrides config)')
    tp.add_argument('--resume',     metavar='PATH',
                    help='Checkpoint to resume from (overrides config)')
    tp.add_argument('--device',     metavar='DEVICE',
                    help='torch device, e.g. cuda or cpu (default: cuda if available)')
    tp.add_argument('--workers',    type=int, default=4, metavar='N',
                    help='DataLoader worker processes (default: 4)')

    # ── complexity ────────────────────────────────────────────────────────────
    cp = sub.add_parser('complexity', help='Report parameter count and FLOPs')
    cp.add_argument('--method',      required=True, choices=methods, help='Fusion method')
    cp.add_argument('--checkpoint',  metavar='PATH',
                    help='Optional weights file (does not affect FLOPs count)')
    cp.add_argument('--device',      default='cpu', metavar='DEVICE',
                    help='torch device (default: cpu)')
    cp.add_argument('--resolution',  type=int, nargs=2, default=[256, 256], metavar=('H', 'W'),
                    help='Spatial resolution for dummy inputs (default: 256 256)')
    cp.add_argument('--ir-channels', type=int, default=1, dest='ir_channels', metavar='C',
                    help='IR input channels before preprocess (default: 1)')
    cp.add_argument('--vi-channels', type=int, default=3, dest='vi_channels', metavar='C',
                    help='VI input channels before preprocess (default: 3)')

    args = parser.parse_args()

    if args.cmd == 'eval':
        cmd_eval(args)
    elif args.cmd == 'train':
        cmd_train(args)
    else:
        cmd_complexity(args)


if __name__ == '__main__':
    main()
