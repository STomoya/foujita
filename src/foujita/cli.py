"""Command line entry: `foujita paint hertzmann ...` and `foujita eval ...`."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from foujita import metrics
from foujita.data import list_images, load_image, prepare_mnist, save_image, write_sample_list
from foujita.experiment import Run
from foujita.legacy import hertzmann
from foujita.strokes import Strokes, blank_canvas, paint_stroke, render

logger = logging.getLogger(__name__)

# method slug -> (params class, paint function). Add learned/legacy methods here.
METHODS = {'hertzmann': (hertzmann.HertzmannParams, hertzmann.paint)}
METRICS = ('mse', 'psnr', 'ssim', 'lpips', 'stroke_count', 'time_s')


def cmd_paint(args: argparse.Namespace) -> None:
    """Paint every image of `--input`; save strokes, canvases and (for `--step-images`) per-step canvases."""
    params_cls, paint = METHODS[args.method]
    params = params_cls(
        **({'radii': tuple(args.radii)} if args.radii else {}),
        **({'threshold': args.threshold} if args.threshold is not None else {}),
    )
    paths = list_images(args.input)[: args.limit]
    config = {
        'kind': 'paint',
        'method': args.method,
        'params': dataclasses.asdict(params),
        'input': str(args.input),
        'size': args.size,
        'step_images': args.step_images,
        'step_every': args.step_every,
        'checkpoint': None,  # learned methods: record run-id + last/best here
    }
    with Run(args.outputs, args.method, args.name, config, args.seed) as run:
        rng = np.random.default_rng(args.seed)
        for d in ('strokes', 'canvases'):
            (run.dir / d).mkdir()
        for i, path in enumerate(paths):
            image = load_image(path, args.size)
            t0 = time.perf_counter()
            strokes = paint(image, params, rng)
            elapsed = time.perf_counter() - t0
            strokes.save(run.dir / 'strokes' / f'{path.stem}.npz', time_s=elapsed)
            save_image(run.dir / 'canvases' / f'{path.stem}.png', render(strokes))
            if path.stem in args.step_images:
                _save_steps(run.dir / 'steps' / path.stem, strokes, args.step_every)
            run.tb.add_scalar('paint/stroke_count', len(strokes), i)
            run.tb.add_scalar('paint/time_s', elapsed, i)
            logger.info('%s: %d strokes, %.2fs', path.name, len(strokes), elapsed)
    logger.info('run: %s', run.dir)


def _save_steps(out: Path, strokes: Strokes, every: int) -> None:
    out.mkdir(parents=True)
    canvas = blank_canvas(strokes.canvas_size)
    for i, (pts, r, c) in enumerate(zip(strokes.points, strokes.radius, strokes.color, strict=True), 1):
        paint_stroke(canvas, pts, r, c)
        if i % every == 0 or i == len(strokes):
            save_image(out / f'{i:06d}.png', canvas)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a paint run against its source images. Writes a new run with per-image and summary metrics."""
    src = yaml.safe_load((args.run / 'config.yaml').read_text())
    method, size = src['method'], src['size']
    ids = sorted(p.stem for p in (args.run / 'strokes').glob('*.npz'))
    images = {p.stem: p for p in list_images(args.input)}
    config = {
        'kind': 'eval',
        'method': method,
        'source_run': args.run.name,
        'checkpoint': src.get('checkpoint'),
        'dataset': str(args.input),
        'size': size,
        'image_list': ids,
        'fid_reference': str(args.fid_ref) if args.fid_ref else None,
    }
    with Run(args.outputs, method, args.name, config, args.seed) as run:
        rows, canvases = [], []
        for i in ids:
            target = load_image(images[i], size)
            strokes, extra = Strokes.load(args.run / 'strokes' / f'{i}.npz')
            canvas = render(strokes)
            canvases.append(canvas)
            rows.append(
                {
                    'image_id': i,
                    'mse': metrics.mse(canvas, target),
                    'psnr': metrics.psnr(canvas, target),
                    'ssim': metrics.ssim(canvas, target),
                    'lpips': metrics.lpips_dist(canvas, target),
                    'stroke_count': len(strokes),
                    'time_s': extra['time_s'],
                },
            )
        with (run.dir / 'metrics_per_image.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, ['image_id', *METRICS])
            w.writeheader()
            w.writerows(rows)
        summary: dict[str, dict[str, float] | float] = {
            m: {'mean': float(np.mean([r[m] for r in rows])), 'std': float(np.std([r[m] for r in rows]))}
            for m in METRICS
        }
        if args.fid_ref:
            ref = [load_image(p, size) for p in list_images(args.fid_ref)]
            summary['fid'] = metrics.fid(canvases, ref)  # ponytail: n < 2048 gives a noisy, rank-deficient estimate
        (run.dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        flat = {m: v['mean'] if isinstance(v, dict) else v for m, v in summary.items()}
        for m, v in flat.items():
            run.tb.add_scalar(f'eval/{m}', v, 0)
        logger.info('%s', {m: round(v, 4) for m, v in flat.items()})
    logger.info('run: %s', run.dir)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    ap = argparse.ArgumentParser(prog='foujita')
    sub = ap.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--input', type=Path, required=True, help='image file or folder')
    common.add_argument('--outputs', type=Path, default=Path('outputs'))
    common.add_argument('--name', default='run')
    common.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('paint', parents=[common])
    p.add_argument('method', choices=METHODS)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--limit', type=int, default=None, help='only the first N images')
    p.add_argument('--radii', type=float, nargs='+')
    p.add_argument('--threshold', type=float)
    p.add_argument('--step-images', nargs='*', default=[], help='image ids that get per-step canvases')
    p.add_argument('--step-every', type=int, default=50, help='strokes between per-step canvases')
    p.set_defaults(fn=cmd_paint)

    e = sub.add_parser('eval', parents=[common])
    e.add_argument('--run', type=Path, required=True, help='paint run directory to evaluate')
    e.add_argument('--fid-ref', type=Path, help='reference images (folder or .txt list) for FID; skipped if absent')
    e.set_defaults(fn=cmd_eval)

    d = sub.add_parser('prepare', help='dataset preparation')
    dsub = d.add_subparsers(required=True)
    m = dsub.add_parser('mnist', help='download MNIST, write test digits as png')
    m.add_argument('--root', type=Path, default=Path('data/raw'))
    m.add_argument('--out', type=Path, default=Path('data/mnist'))
    m.add_argument('-n', type=int, default=100)
    m.set_defaults(fn=lambda a: prepare_mnist(a.root, a.out, a.n))
    s = dsub.add_parser('sample', help='seeded fixed-size sample of an image folder, written as a .txt list')
    s.add_argument('--src', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('-n', type=int, required=True)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--subdirs', nargs='*', help='only these subfolders of --src (e.g. WikiArt styles)')
    s.set_defaults(fn=lambda a: write_sample_list(a.src, a.out, a.n, a.seed, a.subdirs))

    args = ap.parse_args()
    args.fn(args)
