"""Light snapshots of the Gaussian population, for scrubbing a timeline.

A snapshot is *not* a checkpoint. It keeps only what the playback viewer needs
to draw the scene and colour it by lineage -- position, shape, orientation,
opacity, base colour, and id -- in fp16, and drops the optimizer state and the
higher-order spherical harmonics that dominate a real checkpoint's size.

Cost: ~29 bytes per Gaussian, so ~29 MB per million. A 30k run snapshotting
every 250 steps holds 120 of them, which is why the higher-order SH have to go:
keeping shN would multiply that by about eight.

Positions in fp16 carry ~3 decimal digits, which is ample against a scene scale
of order 1 and is display precision, not training precision. Nothing is ever
restored from a snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

#: what we keep, and what it is for
FIELDS = ("means", "scales", "quats", "opacities", "sh0")


def _to_numpy(t) -> np.ndarray:
    return t.detach().to("cpu", dtype=__import__("torch").float16).numpy()


class SnapshotWriter:
    """Writes ``step_<n>.npz`` on a cadence.

    The cadence is deliberately dense early: the interesting structural
    behaviour -- the first densification at 500, the opacity resets -- all
    happens before 15k, after which the population is frozen and consecutive
    snapshots differ only by refinement.
    """

    def __init__(
        self,
        out_dir: Path,
        every: int = 250,
        also: Sequence[int] = (),
        enabled: bool = True,
    ) -> None:
        self.dir = Path(out_dir)
        self.every = int(every)
        self.also = set(int(s) for s in also)
        self.enabled = enabled
        self.steps: List[int] = []
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def due(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step in self.also or (self.every > 0 and step % self.every == 0)

    def maybe_save(self, step: int, splats, state: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        if not self.due(step):
            return None
        return self.save(step, splats, state)

    def save(self, step: int, splats, state: Optional[Dict[str, Any]] = None) -> Path:
        arrays = {k: _to_numpy(splats[k]) for k in FIELDS if k in splats}
        ids = None if state is None else state.get("ids")
        # int32 not int16: a real run hands out millions of ids
        arrays["ids"] = (
            np.arange(len(splats["means"]), dtype=np.int32)
            if ids is None
            else ids.detach().cpu().numpy().astype(np.int32)
        )
        path = self.dir / f"step_{step:07d}.npz"
        np.savez_compressed(path, step=np.int32(step), **arrays)
        self.steps.append(step)
        return path


class SnapshotReader:
    """Reads a snapshot directory back, in step order."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = Path(out_dir)

    @property
    def paths(self) -> List[Path]:
        return sorted(self.dir.glob("step_*.npz"))

    @property
    def steps(self) -> List[int]:
        return [int(p.stem.split("_")[1]) for p in self.paths]

    def load(self, step: int) -> Dict[str, np.ndarray]:
        path = self.dir / f"step_{step:07d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"no snapshot at step {step} in {self.dir}")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        for step in self.steps:
            yield self.load(step)

    def counts(self) -> Dict[int, int]:
        """step -> number of Gaussians, without loading the heavy arrays."""
        out = {}
        for step, path in zip(self.steps, self.paths):
            with np.load(path) as z:
                out[step] = int(z["ids"].shape[0])
        return out
