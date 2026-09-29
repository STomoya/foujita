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
    for k in range(4):
        Image.fromarray((np.roll(_image(), 5 * k, axis=1) * 255).astype(np.uint8)).save(tmp_path / 'in' / f'a{k}.png')
    out, inp = str(tmp_path / 'out'), str(tmp_path / 'in')

    def run_main(*a):
        monkeypatch.setattr('sys.argv', ['foujita', *a])
        main()

    run_main('paint', 'hertzmann', '--input', inp, '--outputs', out, '--size', '64', '--step-images', 'a0')
    (run,) = (tmp_path / 'out' / 'hertzmann').iterdir()
    assert (run / 'strokes' / 'a0.npz').exists()
    assert (run / 'canvases' / 'a3.png').exists()
    assert list((run / 'steps' / 'a0').glob('*.png'))
    assert 'status: done' in (run / 'config.yaml').read_text()
    run_main('eval', '--input', inp, '--outputs', out, '--run', str(run), '--name', 'ev', '--fid-ref', inp)
    ev = next(p for p in (tmp_path / 'out' / 'hertzmann').iterdir() if p.name.endswith('_ev'))
    summary = json.loads((ev / 'summary.json').read_text())
    assert summary['psnr']['mean'] > 15
    assert 0 <= summary['lpips']['mean'] < 1
    assert summary['fid'] >= 0
    assert len((ev / 'metrics_per_image.csv').read_text().splitlines()) == 5
