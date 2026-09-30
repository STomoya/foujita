"""Command line entry: `foujita paint hertzmann ...` and `foujita eval ...`."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from foujita import metrics
from foujita.data import list_images, load_image, prepare_mnist, save_image
from foujita.experiment import Run
from foujita.fidstats import load as load_fid_stats
from foujita.hf import SOURCES, fetch, prepare_wikiart_fid
from foujita.legacy import hertzmann, im2oil
from foujita.strokes import PAINTERS, Strokes, blank_canvas, render

logger = logging.getLogger(__name__)

# method slug -> (params class, paint function). Add learned/legacy methods here.
METHODS: dict[str, tuple[Any, Any]] = {
    'hertzmann': (hertzmann.HertzmannParams, hertzmann.paint),
    'im2oil': (im2oil.Im2OilParams, im2oil.paint),
}
METRICS = ('mse', 'psnr', 'ssim', 'lpips', 'stroke_count', 'time_s')


def cmd_paint(args: argparse.Namespace) -> None:
    """Paint every image of `--input`; save strokes, canvases and (for `--step-images`) per-step canvases."""
    params_cls, paint = METHODS[args.method]
    overrides: dict[str, Any] = {
        'radii': tuple(args.radii) if args.radii else None,
        'threshold': args.threshold,
        'density': args.density,
    }
    fields = {f.name for f in dataclasses.fields(params_cls)}
    params = params_cls(**{k: v for k, v in overrides.items() if v is not None and k in fields})
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
    draw = PAINTERS[strokes.format]
    for i, (pts, r, c) in enumerate(zip(strokes.points, strokes.radius, strokes.color, strict=True), 1):
        draw(canvas, pts, r, c)
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
        'lpips_size': args.lpips_size,
        'image_list': ids,
        'fid_reference': str(args.fid_ref / str(size)) if args.fid_ref else None,
        'fid_styles': args.fid_styles,
    }
    with Run(args.outputs, method, args.name, config, args.seed) as run:
        rows, canvases = [], []
        for i in ids:
            target = load_image(images[i], size)
            strokes, extra = Strokes.load(args.run / 'strokes' / f'{i}.npz')
            canvas = render(strokes)
            canvases.append(canvas)  # painting size, for FID
            rows.append(
                {
                    'image_id': i,
                    'mse': metrics.mse(canvas, target),
                    'psnr': metrics.psnr(canvas, target),
                    'ssim': metrics.ssim(canvas, target),
                    'lpips': metrics.lpips_dist(
                        render(strokes, size=args.lpips_size),
                        load_image(images[i], args.lpips_size),
                    ),
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
            ref_mean, ref_cov = load_fid_stats(args.fid_ref / str(size), args.fid_styles)
            summary['fid'] = metrics.fid(canvases, ref_mean, ref_cov)  # ponytail: few canvases -> noisy, rank-deficient
        (run.dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        flat = {m: v['mean'] if isinstance(v, dict) else v for m, v in summary.items()}
        for m, v in flat.items():
            run.tb.add_scalar(f'eval/{m}', v, 0)
        logger.info('%s', {m: round(v, 4) for m, v in flat.items()})
    logger.info('run: %s', run.dir)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)  # datasets/hub log every request at INFO
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
    p.add_argument('--radii', type=float, nargs='+', help='hertzmann')
    p.add_argument('--threshold', type=float, help='hertzmann')
    p.add_argument('--density', type=float, help='im2oil: strokes per pixel (fineness)')
    p.add_argument('--step-images', nargs='*', default=[], help='image ids that get per-step canvases')
    p.add_argument('--step-every', type=int, default=50, help='strokes between per-step canvases')
    p.set_defaults(fn=cmd_paint)

    e = sub.add_parser('eval', parents=[common])
    e.add_argument('--run', type=Path, required=True, help='paint run directory to evaluate')
    e.add_argument(
        '--fid-ref', type=Path, help='FID stats dir (prepare wikiart-fid), e.g. data/wikiart-fid; skipped if absent'
    )
    e.add_argument(
        '--lpips-size', type=int, default=224, help='LPIPS resolution (strokes re-rendered, target reloaded)'
    )
    e.add_argument('--fid-styles', nargs='*', help='only these styles as FID reference (default: all)')
    e.set_defaults(fn=cmd_eval)

    d = sub.add_parser('prepare', help='dataset preparation')
    dsub = d.add_subparsers(required=True)
    m = dsub.add_parser('mnist', help='download MNIST, write test digits as png')
    m.add_argument('--root', type=Path, default=Path('data/raw'))
    m.add_argument('--out', type=Path, default=Path('data/mnist'))
    m.add_argument('-n', type=int, default=100)
    m.set_defaults(fn=lambda a: prepare_mnist(a.root, a.out, a.n))
    for name, src in SOURCES.items():
        h = dsub.add_parser(name, help=f'stream {src.repo} and save a seeded random sample as png')
        h.add_argument('--split', default=src.split)
        h.add_argument('-n', type=int, default=src.n, help='number of images; 0 = all')
        h.add_argument('--seed', type=int, default=0)
        h.add_argument('--out', type=Path, default=None, help=f'default data/{src.out}')
        h.set_defaults(fn=lambda a, name=name: fetch(name, a.split, a.n or None, a.seed, a.out))
    w = dsub.add_parser(
        'wikiart-fid', help='stream all of WikiArt, keep only per-style FID statistics (no images saved)'
    )
    w.add_argument('--sizes', type=int, nargs='+', default=[128], help='working resolutions to compute statistics for')
    w.add_argument('--out', type=Path, default=Path('data/wikiart-fid'))
    w.add_argument('--limit', type=int, default=None, help='only the first N images (for testing)')
    w.set_defaults(fn=lambda a: prepare_wikiart_fid(a.sizes, a.out, a.limit))

    args = ap.parse_args()
    args.fn(args)
