"""Im2Oil: stroke count is linear in density, painting beats a blank canvas, strokes roundtrip."""

import numpy as np
from test_hertzmann import _image

from foujita import metrics
from foujita.legacy.im2oil import Im2OilParams, paint
from foujita.strokes import Strokes, render


def test_paint_fineness_and_quality():
    img = _image()
    coarse = paint(img, Im2OilParams(density=0.05, textured=False), np.random.default_rng(0))
    fine = paint(img, Im2OilParams(density=0.2, textured=False), np.random.default_rng(0))
    assert abs(len(coarse) - 0.05 * 64 * 64) < 0.1 * 0.05 * 64 * 64  # empty cells are dropped
    assert len(fine) > 3 * len(coarse)
    blank = metrics.psnr(np.ones_like(img), img)
    assert metrics.psnr(render(coarse), img) > blank + 5
    assert metrics.psnr(render(fine), img) > metrics.psnr(render(coarse), img)


def test_textured_brush_renders_and_roundtrips(tmp_path):
    img = _image()
    strokes = paint(img, Im2OilParams(), np.random.default_rng(0))
    assert strokes.format == 'textured-brush-v1'
    canvas = render(strokes)
    assert metrics.psnr(canvas, img) > metrics.psnr(np.ones_like(img), img) + 5
    assert render(strokes, size=128).shape == (128, 128, 3)
    strokes.save(tmp_path / 's.npz', time_s=0.0)
    back, _ = Strokes.load(tmp_path / 's.npz')
    assert np.allclose(render(back), canvas)
