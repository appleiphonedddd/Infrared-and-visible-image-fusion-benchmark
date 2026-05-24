import argparse
import json
from pathlib import Path
import torch
from torchvision.utils import save_image
from method import build_method, METHOD_REGISTRY, build_trainer, TRAINER_REGISTRY
from data_loader import build_dataset, DATASET_REGISTRY, FusionDataLoader
from utils.metrics import MetricSuite, ModelComplexity


def run(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.cmd == 'eval':
        m = build_method(args.method, device=device)
        m.load_checkpoint(args.checkpoint)

        dataset = build_dataset(args.dataset, root=args.data_root, split=args.split or 'test')
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

    elif args.cmd == 'train':
        config: dict = {}
        if args.config:
            with open(args.config) as f:
                config = json.load(f)

        if args.save_dir:
            config['save_dir'] = args.save_dir
        if args.resume:
            config['resume'] = args.resume

        m = build_method(args.method, device=device)
        dataset = build_dataset(
            args.dataset,
            root=args.data_root,
            split=args.split or 'train',
            include_seg=config.get('include_seg', False),
        )
        loader = FusionDataLoader(
            dataset,
            batch_size=config.get('batch_size', 1),
            num_workers=args.workers,
        )
        trainer = build_trainer(args.method, m, loader, config)
        trainer.train()

    elif args.cmd == 'complexity':
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


if __name__ == '__main__':
    methods   = sorted(METHOD_REGISTRY)
    datasets  = sorted(DATASET_REGISTRY)
    trainable = sorted(TRAINER_REGISTRY)

    parser = argparse.ArgumentParser(prog='main.py',
                                     description='Infrared & Visible Image Fusion Benchmark')
    parser.add_argument('cmd',            choices=['eval', 'train', 'complexity'])
    parser.add_argument('--method',       required=True, choices=methods)
    parser.add_argument('--checkpoint',   metavar='PATH')
    parser.add_argument('--dataset',      choices=datasets)
    parser.add_argument('--data-root',    dest='data_root',  metavar='DIR')
    parser.add_argument('--split',        choices=['train', 'test'])
    parser.add_argument('--save-dir',     dest='save_dir',   metavar='DIR')
    parser.add_argument('--config',       metavar='PATH')
    parser.add_argument('--resume',       metavar='PATH')
    parser.add_argument('--device',       metavar='DEVICE')
    parser.add_argument('--workers',      type=int, default=4,         metavar='N')
    parser.add_argument('--resolution',   type=int, nargs=2, default=[256, 256], metavar=('H', 'W'))
    parser.add_argument('--ir-channels',  type=int, default=1, dest='ir_channels', metavar='C')
    parser.add_argument('--vi-channels',  type=int, default=3, dest='vi_channels', metavar='C')

    args = parser.parse_args()
    run(args)
