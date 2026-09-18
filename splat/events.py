"""The event log: every structural change to the Gaussian population.

One row per event, one table for the whole run. A 30k run on a real scene
produces a few million rows (every clone, split and prune is an event), which is
why this is parquet and not JSON: columnar, typed, and read back with a filter
rather than parsed in full.

Schema
------
    step    int32   training iteration the event happened on
    kind    str     sfm | seed | clone | split | death | reset | stage
    gid     int64   the Gaussian this is about (-1 for reset, which is global)
    parent  int64   the Gaussian it came from (-1 for sfm, seed, death, reset)

``stage`` is the one row that is not about a Gaussian: it marks a curriculum
stage beginning, and its ``gid`` carries the **dataset item index of the image
added** rather than a Gaussian id. That is a deliberate pun on the column, kept
because it lets the viewer label a timeline marker with the photo that caused
it from the event log alone. The richer per-stage record -- renders, metrics,
seed counts -- does not fit four columns and lives in ``stages.json``.

Births carry their reason in ``kind``, so "where did this Gaussian come from"
and "when" are the same lookup. Deaths are recorded by id only; the id is never
reused, so a (birth, death) pair fully describes a lifetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

BIRTH_KINDS = ("sfm", "seed", "clone", "split")
#: events that are about the run rather than about a Gaussian
GLOBAL_KINDS = ("reset", "stage")
KINDS = BIRTH_KINDS + ("death",) + GLOBAL_KINDS

_NONE = -1


class EventLog:
    """Buffers events in memory and writes one parquet file.

    Appends are numpy-array-at-a-time rather than row-at-a-time: densification
    produces tens of thousands of events in a single call, and per-row Python
    would dominate the step time it is supposed to be observing.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._step: List[np.ndarray] = []
        self._kind: List[np.ndarray] = []
        self._gid: List[np.ndarray] = []
        self._parent: List[np.ndarray] = []
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(
        self,
        step: int,
        kind: str,
        gid: np.ndarray,
        parent: Optional[np.ndarray] = None,
    ) -> None:
        """Record ``len(gid)`` events of one kind at one step."""
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}, expected one of {KINDS}")
        gid = np.asarray(gid, dtype=np.int64).reshape(-1)
        if gid.size == 0:
            return
        if parent is None:
            parent = np.full(gid.size, _NONE, dtype=np.int64)
        else:
            parent = np.asarray(parent, dtype=np.int64).reshape(-1)
            if parent.size != gid.size:
                raise ValueError(f"gid/parent length mismatch: {gid.size} vs {parent.size}")
        self._step.append(np.full(gid.size, step, dtype=np.int32))
        self._kind.append(np.full(gid.size, kind, dtype=object))
        self._gid.append(gid)
        self._parent.append(parent)
        self._n += int(gid.size)

    def add_reset(self, step: int) -> None:
        """An opacity reset: global, so it has no gid or parent."""
        self.add(step, "reset", np.array([_NONE], dtype=np.int64))

    def add_stage(self, step: int, image: int = _NONE) -> None:
        """A curriculum stage beginning. ``image`` is a dataset item index."""
        self.add(step, "stage", np.array([int(image)], dtype=np.int64))

    def to_arrays(self) -> dict:
        if not self._step:
            return {
                "step": np.empty(0, np.int32),
                "kind": np.empty(0, object),
                "gid": np.empty(0, np.int64),
                "parent": np.empty(0, np.int64),
            }
        return {
            "step": np.concatenate(self._step),
            "kind": np.concatenate(self._kind),
            "gid": np.concatenate(self._gid),
            "parent": np.concatenate(self._parent),
        }

    def flush(self, path: Optional[Path] = None) -> Optional[Path]:
        """Write the parquet file. Returns the path, or None if nowhere to write."""
        out = Path(path) if path is not None else self.path
        if out is None:
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        a = self.to_arrays()
        table = pa.table(
            {
                "step": pa.array(a["step"], pa.int32()),
                # dictionary-encoded: six distinct values over millions of rows
                "kind": pa.array(a["kind"].astype(str), pa.string()).dictionary_encode(),
                "gid": pa.array(a["gid"], pa.int64()),
                "parent": pa.array(a["parent"], pa.int64()),
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out, compression="zstd")
        return out


def read_events(path: Path):
    """Read an event log back as a pyarrow Table."""
    import pyarrow.parquet as pq

    return pq.read_table(path)
