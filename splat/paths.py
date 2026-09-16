"""Where things live on disk.

Scenes and runs both sit under the repo's gitignored ``data/``, so a scene is
addressable by relative path from anywhere in the repo with no environment to
set. ``DATA_ROOT`` overrides the parent for data that outgrows this disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", repo_root() / "data"))


def scene_dir(name: str) -> Path:
    """A COLMAP scene written by run_sfm.py."""
    return data_root() / "scenes" / name


@dataclass(frozen=True)
class RunPaths:
    """Everything one training run writes.

    ``ckpts``/``stats``/``renders``/``videos``/``tb`` are gsplat's own; the rest
    are ours, so an instrumented run stays a superset of a vanilla one and the
    two remain directly comparable.
    """

    root: Path

    @property
    def ckpts(self) -> Path:
        return self.root / "ckpts"

    @property
    def stats(self) -> Path:
        return self.root / "stats"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def events(self) -> Path:
        return self.root / "events.parquet"

    @property
    def report(self) -> Path:
        return self.root / "report"

    def mkdirs(self) -> "RunPaths":
        for d in (self.root, self.snapshots, self.report):
            d.mkdir(parents=True, exist_ok=True)
        return self


def run_dir(name: str) -> RunPaths:
    return RunPaths(data_root() / "runs" / name)
