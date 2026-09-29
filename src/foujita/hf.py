"""Stream a Hugging Face image dataset and save only a seeded random sample as png (plus a manifest)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SHUFFLE_BUFFER = 1000  # rows held in memory; full-res images are large


@dataclass
class Source:
    """Where a dataset lives on the Hub and what we sample from it by default."""

    repo: str
    split: str
    n: int  # 0 = all
    out: str  # under data/
    gated: bool = False


SOURCES = {
    # AFHQv2 cats, 512px; `test` is the official held-out split (all its cats). huggan/AFHQv2 was not used because it
    # only has a `train` split, so it cannot give the official test set.
    'afhq-cat': Source('ryushinn/AFHQv2', 'test', 0, 'afhq-cat'),
    # Needs a Hugging Face token and accepting the terms on the dataset page.
    'imagenet': Source('ILSVRC/imagenet-1k', 'validation', 1000, 'imagenet-1k', gated=True),
    'wikiart': Source('huggan/wikiart', 'train', 500, 'wikiart'),
}


def fetch(  # noqa: PLR0917
    name: str,
    split: str,
    n: int | None,
    seed: int,
    out: Path | None,
    styles: list[str] | None,
) -> None:
    """Save `n` random images (all if None) of `SOURCES[name]` to `out` (default `data/<out>/<split>`).

    # ponytail: streaming can't index rows, so "random" = seeded shuffle of shards plus a 1k-row buffer, then take n.
    # It is deterministic for a given seed and library version, but not a uniform sample. Reproducibility is the
    # saved manifest.json + the images themselves.
    """
    from datasets import load_dataset  # noqa: PLC0415

    src = SOURCES[name]
    out = out or Path('data') / src.out / split
    try:
        ds = load_dataset(src.repo, split=split, streaming=True)
    except Exception as e:
        if src.gated:
            msg = (
                f'{src.repo} is gated. Create a read token at https://huggingface.co/settings/tokens, accept the terms '
                f'at https://huggingface.co/datasets/{src.repo}, then `export HF_TOKEN=...` (or `hf auth login`).'
            )
            raise SystemExit(msg) from e
        raise
    feats = ds.features
    if feats is None:
        msg = f'{src.repo} has no schema in streaming mode'
        raise SystemExit(msg)
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)  # before filtering: reads ~n/keep-fraction rows, not all
    filt = {}
    if name == 'afhq-cat':
        cat = feats['label'].str2int('cat')
        ds = ds.filter(lambda r: r['label'] == cat)
        filt = {'label': 'cat'}
    if name == 'wikiart' and styles:
        ids = {feats['style'].str2int(s) for s in styles}
        ds = ds.filter(lambda r: r['style'] in ids)
        filt = {'style': styles}
    if n:
        ds = ds.take(n)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, row in enumerate(ds):
        row['image'].convert('RGB').save(out / f'{i:05d}.png')
        count += 1
    manifest = {'repo': src.repo, 'split': split, 'n': count, 'seed': seed, 'buffer': SHUFFLE_BUFFER, 'filter': filt}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    logger.info('saved %d images to %s', count, out)
