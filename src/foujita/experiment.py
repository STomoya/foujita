"""Run directory, config, status and TensorBoard, following `experiment_rules.md`."""

from __future__ import annotations

import random
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True, check=False).stdout.strip()


def device_name() -> str:
    """Accelerator name, or `cpu`."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Run:
    """One experiment run at `<root>/<method>/<timestamp>_<name>/`. Use as a context manager.

    Status is `running` on entry, `done` on clean exit, `failed` on exception. The directory must not exist yet
    (finished runs are immutable). Resuming is not supported yet.
    """

    def __init__(self, root: Path, method: str, name: str, config: dict[str, Any], seed: int) -> None:
        run_id = f'{datetime.now(tz=UTC).astimezone():%Y%m%d-%H%M}_{name}'
        self.dir = root / method / run_id
        self.dir.mkdir(parents=True)
        self.config = {
            **config,
            'seed': seed,
            'git_commit': _git('rev-parse', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')),
            'device': device_name(),
            'status': 'running',
        }
        self._write_config()
        seed_everything(seed)
        self.tb = SummaryWriter(self.dir / 'tb')
        self.tb.add_text('config', f'```\n{yaml.safe_dump(self.config)}\n```')

    def _write_config(self) -> None:
        (self.dir / 'config.yaml').write_text(yaml.safe_dump(self.config, sort_keys=False))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.config['status'] = 'failed' if exc_type else 'done'
        self._write_config()
        self.tb.close()
