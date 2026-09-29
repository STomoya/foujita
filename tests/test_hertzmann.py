"""End to end check: paint synthetic images, then evaluate the run."""

import json

import numpy as np
from PIL import Image

from foujita import metrics
from foujita.cli import main
from foujita.legacy.hertzmann import HertzmannParams, paint
from foujita.strokes import Strokes, render


def _image() -> np.ndarray:
    img = np.ones((64, 64, 3), np.float32) * 0.9
    img[10:40, 10:40] = (0.8, 0.1, 0.1)
    img[30:60, 35:60] = (0.1, 0.2, 0.8)
    return img


def test_paint_reduces_error_and_roundtrips(tmp_path):
    img = _image()
    strokes = paint(img, HertzmannParams(), np.random.default_rng(0))
    canvas = render(strokes)
    assert len(strokes) > 0
    assert metrics.psnr(canvas, img) > metrics.psnr(np.ones_like(img), img) + 5
    assert metrics.ssim(img, img) > 0.999
    strokes.save(tmp_path / 's.npz', time_s=1.5)
    back, extra = Strokes.load(tmp_path / 's.npz')
    assert extra == {'time_s': 1.5}
    assert np.allclose(render(back), canvas)


def test_cli_paint_then_eval(tmp_path, monkeypatch):
    (tmp_path / 'in').mkdir()
    Image.fromarray((_image() * 255).astype(np.uint8)).save(tmp_path / 'in' / 'a.png')
    out = str(tmp_path / 'out')
    run_main = lambda *a: (monkeypatch.setattr('sys.argv', ['foujita', *a]), main())  # noqa: E731
    run_main(
        'paint',
        'hertzmann',
        '--input',
        str(tmp_path / 'in'),
        '--outputs',
        out,
        '--size',
        '64',
        '--step-images',
        'a',
        '--step-every',
        '20',
    )
    (run,) = (tmp_path / 'out' / 'hertzmann').iterdir()
    assert (run / 'strokes' / 'a.npz').exists()
    assert (run / 'canvases' / 'a.png').exists()
    assert list((run / 'steps' / 'a').glob('*.png'))
    assert 'status: done' in (run / 'config.yaml').read_text()
    run_main('eval', '--input', str(tmp_path / 'in'), '--outputs', out, '--run', str(run), '--name', 'ev')
    ev = next(p for p in (tmp_path / 'out' / 'hertzmann').iterdir() if p.name.endswith('_ev'))
    summary = json.loads((ev / 'summary.json').read_text())
    assert summary['psnr']['mean'] > 15
