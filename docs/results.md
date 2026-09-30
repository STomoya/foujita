# Experiment results

Summaries of finished evaluation runs. Heavy outputs (strokes, canvases, TensorBoard logs, per-image CSVs) stay in
`outputs/` and are not tracked. Add one row per evaluation run, newest last, and keep the run ids so a row can be
traced back to `outputs/<method>/<run-id>/`.

Columns show mean over images (std in parentheses for the per-image metrics). PSNR in dB, time in seconds per image on
the recorded device. FID is over the whole set against the WikiArt statistics (all styles merged) at the run's size;
with fewer than 2048 images it is rough.

## Hertzmann (1998)

Default parameters unless noted: radii 8/4/2, threshold 0.1, blur factor 0.5, grid 1.0, curvature 1.0, stroke length 4-16.
Image size 128, seed 0, device `mps`.

MSE, PSNR, SSIM and FID are computed at the painting size (128). **LPIPS is computed at 224** (`--lpips-size`, the size
AlexNet is trained on): the vector strokes are re-rendered at 224 and the target is reloaded from the original at 224.
LPIPS depends strongly on image size, so only compare LPIPS values measured at the same size.

| eval run | painted run | data | n | MSE | PSNR | SSIM | LPIPS@224 | strokes | time | FID |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260930-1634_base-afhq-cat-eval224` | `20260930-1157_base-afhq-cat` | AFHQv2 cat, official test (`ryushinn/AFHQv2`) | 493 | 0.0081 (0.0035) | 21.31 (1.82) | 0.525 (0.084) | 0.404 (0.053) | 2689 (457) | 0.46 (0.06) | 265.2 |
| `20260930-1645_base-imagenet-1k-eval224` | `20260930-1207_base-imagenet-1k` | ImageNet val, seeded 1000 sample (`ILSVRC/imagenet-1k`, seed 0) | 1000 | 0.0121 (0.0070) | 19.89 (2.65) | 0.499 (0.131) | 0.381 (0.095) | 2674 (716) | 0.44 (0.09) | 162.1 |

Painted runs were made at commit `f1d8f45`. The eval rows ran on `4b74927` plus the LPIPS-size change committed as `8d0497a`.

Superseded: the first evaluation (`20260930-1204_base-afhq-cat-eval`, `20260930-1220_base-imagenet-1k-eval`) scored LPIPS at
128 and got 0.236 (AFHQ-cat) and 0.251 (ImageNet-1k). Everything else in those runs is identical.

Notes:
- Parameters are untuned; this is the first baseline.
- Strokes are round-capped polylines rather than B-splines.
- No per-step canvases were saved.

## Im2Oil (2022)

Default parameters: density 0.1 strokes/pixel (1638 strokes at 128), texture weight 4, Lloyd iterations 5, stroke length
2.0x and width 1.0x cell diameter, textured oil-brush strokes (`textured-brush-v1`). Image size 128, seed 0, device
`mps`, LPIPS at 224, FID against all WikiArt styles. Run at commit `8f6f414`.

| eval run | painted run | data | n | MSE | PSNR | SSIM | LPIPS@224 | strokes | time | FID |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260930-1819_base-afhq-cat-eval224` | `20260930-1818_base-afhq-cat` | AFHQv2 cat, official test | 493 | 0.0064 (0.0026) | 22.28 (1.70) | 0.576 (0.062) | 0.329 (0.048) | 1638 (0) | 0.13 (0.01) | 286.0 |
| `20260930-1824_base-imagenet-1k-eval224` | `20260930-1821_base-imagenet-1k` | ImageNet val, seeded 1000 sample | 1000 | 0.0099 (0.0057) | 20.72 (2.52) | 0.538 (0.103) | 0.355 (0.083) | 1638 (0) | 0.12 (0.00) | 166.3 |

Notes: parameters untuned; Hertzmann uses ~1.6x more strokes here, so the comparison is not stroke-matched. No per-step
canvases were saved. Stroke count is fixed by `density` (std 0).
