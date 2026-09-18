"""The run as a timeline: what to draw at step N, and where each Gaussian came from.

Everything the playback viewer needs that is not rendering. It is kept out of
``viewer.py`` because it is pure numpy over a run directory -- no GPU, no
browser, no viser -- so the lineage arithmetic that colour-by-origin and
"follow this Gaussian" depend on can be tested headlessly. Getting that
arithmetic subtly wrong produces a picture that looks entirely plausible and is
entirely false, which is the failure mode ``lineage.py`` was written to avoid
and the same one applies on the way back out.

Two inputs, both written by an instrumented run:

- ``snapshots/step_*.npz`` -- the population as it was at each cadence step.
  This is what gets drawn.
- ``events.parquet`` -- every birth and death. This is what explains it.

The snapshot says *which* Gaussians were alive; only the event log says *why*.
They meet on ``ids``, which is why snapshots carry the id array at all.

A curriculum run adds two more, and they meet the others on ids as well:

- ``stages.json`` -- which photo was added when, and what it did.
- ``deltas/stage_NNN.npz`` -- per-Gaussian change over each stage, which is what
  "the Gaussians this photo changed" means once it is a picture.

Memory
------
The lineage index is dense arrays indexed by gid: 13 bytes per id ever issued,
plus 8 more if the child index is built. A 3k run on ``room-1`` issues ~1.06M
ids (~14 MB); a full 30k run issues tens of millions, so a few hundred MB. That
is the deliberate trade -- an interactive scrub cannot afford a parquet filter
per frame, and ids are dense (``lineage.py`` hands them out contiguously), so
direct indexing needs no hash map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .events import BIRTH_KINDS
from .paths import RunPaths
from .snapshots import SnapshotReader

#: origin -> colour, shared with report.py so a Gaussian is the same colour in
#: the viewer as its cohort is in population.png
ORIGIN_COLORS: Dict[str, Tuple[float, float, float]] = {
    "sfm": (0.298, 0.431, 0.961),  # #4C6EF5
    "seed": (0.071, 0.725, 0.525),  # #12B886
    "clone": (0.961, 0.624, 0.000),  # #F59F00
    "split": (0.878, 0.192, 0.192),  # #E03131
}
#: gid with no birth event in the log; drawn grey rather than dropped
UNKNOWN_COLOR = (0.522, 0.557, 0.588)  # #868E96

_NONE = -1


def load_cfg(run: RunPaths) -> Dict:
    """Read ``cfg.yml`` without executing it.

    ``simple_trainer`` dumps the config with ``yaml.dump``, so the strategy
    arrives as a ``!!python/object:gsplat.strategy.default.DefaultStrategy``
    tag. ``safe_load`` refuses it and ``unsafe_load`` would execute whatever a
    run directory happens to contain -- and run directories are meant to
    ``rsync`` between machines. So: safe loader, unknown tags degraded to plain
    mappings, which is all the viewer wants from them anyway.
    """
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _as_plain(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    _Loader.add_multi_constructor("", _as_plain)
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/object:", _as_plain)

    path = run.root / "cfg.yml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.load(f, Loader=_Loader) or {}


class ImageRoles(NamedTuple):
    """What each training image is doing at one step."""

    active: List[int]  #: the sampler is drawing from these
    resting: List[int]  #: seen and seeded, but not training right now
    pending: List[int]  #: never seen
    added: Optional[int]  #: first seen at this stage, if any


def load_stages(run: RunPaths) -> Optional[Dict]:
    """``stages.json``, or None for a run that was not a curriculum run."""
    import json

    path = Path(run.root) / "stages.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A run killed mid-write leaves a truncated file. Losing the stage
        # markers is survivable; refusing to open the viewer is not.
        return None


@dataclass(frozen=True)
class StageInfo:
    """One curriculum stage, as ``stages.json`` recorded it."""

    index: int
    step: int
    end_step: Optional[int]
    added: Optional[int]  #: dataset item index of the photo added
    image: Optional[str]
    n_images: int  #: images seen so far
    active: Tuple[int, ...] = ()  #: items the sampler drew from during this stage
    n_active: int = 0
    steps_per_active_image: float = 0.0
    group: Optional[int] = None
    round: Optional[int] = None
    n_seeded: int = 0
    n_survived: int = 0
    n_born: int = 0
    n_died: int = 0
    psnr_before: Optional[float] = None
    psnr_after: Optional[float] = None
    deltas: Optional[str] = None

    @classmethod
    def from_record(cls, rec: Dict) -> "StageInfo":
        return cls(
            index=int(rec["stage"]),
            step=int(rec["step"]),
            end_step=rec.get("end_step"),
            added=rec.get("added"),
            image=rec.get("image"),
            n_images=int(rec.get("n_images", 0)),
            active=tuple(rec.get("active") or ()),
            n_active=int(rec.get("n_active", 0) or 0),
            steps_per_active_image=float(rec.get("steps_per_active_image", 0) or 0),
            group=rec.get("group"),
            round=rec.get("round"),
            n_seeded=int(rec.get("n_seeded", 0) or 0),
            n_survived=int(rec.get("n_survived", 0) or 0),
            n_born=int(rec.get("n_born", 0) or 0),
            n_died=int(rec.get("n_died", 0) or 0),
            psnr_before=rec.get("psnr_before"),
            psnr_after=rec.get("psnr_after"),
            deltas=rec.get("deltas"),
        )

    @property
    def gain(self) -> Optional[float]:
        if self.psnr_before is None or self.psnr_after is None:
            return None
        return self.psnr_after - self.psnr_before


@dataclass(frozen=True)
class Schedule:
    """The densification schedule, for narrating what phase a step is in.

    Defaults are gsplat's ``DefaultStrategy`` defaults, so a run whose
    ``cfg.yml`` is missing or predates this still narrates correctly as long as
    it was trained with them.
    """

    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    refine_every: int = 100
    reset_every: int = 3000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    max_steps: int = 30000

    @classmethod
    def from_cfg(cls, cfg: Dict) -> "Schedule":
        strategy = cfg.get("strategy") or {}
        if not isinstance(strategy, dict):
            strategy = {}
        merged = {**strategy, **{k: cfg[k] for k in ("sh_degree", "sh_degree_interval", "max_steps") if k in cfg}}
        known = {f: merged[f] for f in cls.__dataclass_fields__ if f in merged}
        return cls(**{k: int(v) for k, v in known.items()})

    def sh_degree_at(self, step: int) -> int:
        """How many SH bands are active -- gsplat raises one per interval."""
        if self.sh_degree_interval <= 0:
            return self.sh_degree
        return int(min(step // self.sh_degree_interval, self.sh_degree))

    def phase(self, step: int) -> str:
        if step < self.refine_start_iter:
            return "warm-up"
        if step < self.refine_stop_iter:
            return "densification"
        return "refinement"


class Lineage:
    """Where every Gaussian came from, indexed by gid.

    Built once from ``events.parquet`` into dense arrays. ``lineage.py`` issues
    ids contiguously from 0, so gid *is* the index and no hash map is needed --
    a property worth checking rather than assuming, so the constructor does.
    """

    def __init__(
        self,
        birth_step: np.ndarray,
        birth_kind: np.ndarray,
        parent: np.ndarray,
        death_step: np.ndarray,
        resets: np.ndarray,
        stage_steps: np.ndarray = None,  # type: ignore[assignment]
    ) -> None:
        self.birth_step = birth_step  # int32[n_ids], -1 if never born (a gap)
        self.birth_kind = birth_kind  # int8[n_ids], index into BIRTH_KINDS, -1 unknown
        self.parent = parent  # int32[n_ids], -1 for sfm/seed
        self.death_step = death_step  # int32[n_ids], -1 if still alive at the end
        self.resets = resets  # int32[n_resets], the opacity-reset steps
        #: int32[n_stages], the curriculum stage-boundary steps. Read from the
        #: event log rather than stages.json so the markers survive a run whose
        #: stages.json was never written (a crash, or a copy of just the log).
        self.stage_steps = (
            np.empty(0, dtype=np.int32) if stage_steps is None else stage_steps
        )
        self._child_parent: Optional[np.ndarray] = None
        self._child_gid: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_events(cls, path: Path) -> "Lineage":
        from .events import read_events

        t = read_events(Path(path))
        step = t.column("step").to_numpy(zero_copy_only=False).astype(np.int32)
        kind = np.asarray(t.column("kind").to_pylist())
        gid = t.column("gid").to_numpy(zero_copy_only=False)
        parent = t.column("parent").to_numpy(zero_copy_only=False)
        return cls.from_arrays(step, kind, gid, parent)

    @classmethod
    def from_arrays(
        cls,
        step: np.ndarray,
        kind: np.ndarray,
        gid: np.ndarray,
        parent: np.ndarray,
    ) -> "Lineage":
        kind = np.asarray(kind)
        is_birth = np.isin(kind, BIRTH_KINDS)
        is_death = kind == "death"
        resets = np.unique(step[kind == "reset"]).astype(np.int32)
        stages = np.unique(step[kind == "stage"]).astype(np.int32)

        n = int(gid[is_birth].max()) + 1 if is_birth.any() else 0
        birth_step = np.full(n, _NONE, dtype=np.int32)
        birth_kind = np.full(n, _NONE, dtype=np.int8)
        parents = np.full(n, _NONE, dtype=np.int32)
        death_step = np.full(n, _NONE, dtype=np.int32)

        bgid = gid[is_birth].astype(np.int64)
        birth_step[bgid] = step[is_birth]
        parents[bgid] = parent[is_birth]
        for code, name in enumerate(BIRTH_KINDS):
            sel = is_birth & (kind == name)
            birth_kind[gid[sel].astype(np.int64)] = code

        # A death of an id that was never born would mean the log is broken;
        # clip rather than crash, since a viewer that opens is worth more than
        # one that refuses, and the gap shows up as grey.
        dgid = gid[is_death].astype(np.int64)
        in_range = (dgid >= 0) & (dgid < n)
        death_step[dgid[in_range]] = step[is_death][in_range]

        return cls(birth_step, birth_kind, parents, death_step, resets, stages)

    @classmethod
    def load(cls, run: RunPaths) -> Optional["Lineage"]:
        """None when the run was not instrumented -- the viewer still plays."""
        return cls.from_events(run.events) if run.events.exists() else None

    # -- per-Gaussian lookups ----------------------------------------------

    def __len__(self) -> int:
        return int(self.birth_step.shape[0])

    def _valid(self, ids: np.ndarray) -> np.ndarray:
        return (ids >= 0) & (ids < len(self))

    def origin_codes(self, ids: np.ndarray) -> np.ndarray:
        """int8 code into ``BIRTH_KINDS`` per id, -1 for unknown."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.full(ids.shape, _NONE, dtype=np.int8)
        ok = self._valid(ids)
        out[ok] = self.birth_kind[ids[ok]]
        return out

    def ages(self, ids: np.ndarray, step: int) -> np.ndarray:
        """Steps since birth, as float32. -1 where birth is unknown."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.full(ids.shape, -1.0, dtype=np.float32)
        ok = self._valid(ids)
        born = self.birth_step[ids[ok]]
        out[ok] = np.where(born >= 0, step - born, -1.0)
        return out

    def origin_counts(self, ids: np.ndarray) -> Dict[str, int]:
        codes = self.origin_codes(ids)
        out = {name: int(np.sum(codes == c)) for c, name in enumerate(BIRTH_KINDS)}
        out["unknown"] = int(np.sum(codes == _NONE))
        return out

    # -- family ------------------------------------------------------------

    def _ensure_children(self) -> None:
        """Sort births by parent once, so children are a searchsorted range.

        Lazy: only the lineage panel needs it, and it is the largest of the
        indexes.
        """
        if self._child_parent is not None:
            return
        has_parent = self.parent >= 0
        parents = self.parent[has_parent]
        kids = np.nonzero(has_parent)[0].astype(np.int32)
        order = np.argsort(parents, kind="stable")
        self._child_parent = parents[order]
        self._child_gid = kids[order]

    def children_of(self, gid: int) -> np.ndarray:
        self._ensure_children()
        lo, hi = np.searchsorted(self._child_parent, [gid, gid + 1])
        return self._child_gid[lo:hi]

    def ancestors(self, gid: int) -> List[int]:
        """``[parent, grandparent, ...]`` up to the sfm or seed root."""
        out: List[int] = []
        seen = set()
        cur = int(gid)
        while self._valid(np.array(cur)) and self.parent[cur] >= 0:
            cur = int(self.parent[cur])
            if cur in seen:  # a cycle cannot happen, but do not hang if it does
                break
            seen.add(cur)
            out.append(cur)
        return out

    def root_of(self, gid: int) -> int:
        chain = self.ancestors(gid)
        return chain[-1] if chain else int(gid)

    def descendants(self, gid: int, limit: int = 1_000_000) -> np.ndarray:
        """Every id below ``gid``, breadth-first. Excludes ``gid`` itself."""
        self._ensure_children()
        out: List[np.ndarray] = []
        frontier = np.array([gid], dtype=np.int32)
        total = 0
        while frontier.size and total < limit:
            lo = np.searchsorted(self._child_parent, frontier)
            hi = np.searchsorted(self._child_parent, frontier + 1)
            nxt = np.concatenate(
                [self._child_gid[a:b] for a, b in zip(lo, hi)] or [np.empty(0, np.int32)]
            )
            if nxt.size == 0:
                break
            out.append(nxt)
            total += int(nxt.size)
            frontier = np.unique(nxt)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int32)

    def family(self, gid: int) -> np.ndarray:
        """``gid``, its ancestors and all of their descendants.

        The whole clan from one sfm point, which is what "follow one Gaussian's
        split lineage" means once that Gaussian has itself been split: the
        Gaussian you clicked is gone, and its children are the thing to look at.
        """
        root = self.root_of(gid)
        return np.unique(
            np.concatenate(
                [
                    np.array([gid, root], dtype=np.int32),
                    np.asarray(self.ancestors(gid), dtype=np.int32),
                    self.descendants(root),
                ]
            )
        )

    def describe(self, gid: int) -> str:
        """One-line provenance for the text panel."""
        if not self._valid(np.array(gid)):
            return f"gid {gid}: out of range (0..{len(self) - 1})"
        code = int(self.birth_kind[gid])
        kind = BIRTH_KINDS[code] if code >= 0 else "unknown"
        born = int(self.birth_step[gid])
        died = int(self.death_step[gid])
        parent = int(self.parent[gid])
        kids = self.children_of(gid)
        parts = [f"gid {gid}: {kind} at step {born if born >= 0 else '?'}"]
        if parent >= 0:
            parts.append(f"parent {parent}")
        parts.append(f"died at {died}" if died >= 0 else "alive at end of run")
        parts.append(f"{kids.size} children" + (f" {list(kids[:4])}" if kids.size else ""))
        parts.append(f"depth {len(self.ancestors(gid))} below sfm point {self.root_of(gid)}")
        return " | ".join(parts)


class Timeline:
    """Snapshot steps plus the events that explain the jumps between them."""

    def __init__(self, run: RunPaths) -> None:
        self.run = run
        self.snapshots = SnapshotReader(run.snapshots)
        self.lineage = Lineage.load(run)
        self.cfg = load_cfg(run)
        self.schedule = Schedule.from_cfg(self.cfg)
        self._per_step: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        self.plan = load_stages(run)
        self.stages: List[StageInfo] = (
            [StageInfo.from_record(r) for r in self.plan.get("stages", [])]
            if self.plan
            else []
        )
        self._deltas: Dict[int, Optional[Dict[str, np.ndarray]]] = {}

    # -- steps -------------------------------------------------------------

    @property
    def steps(self) -> List[int]:
        return self.snapshots.steps

    def __len__(self) -> int:
        return len(self.steps)

    def nearest(self, step: int) -> int:
        """The snapshot step closest to ``step``; the slider lands on these."""
        steps = self.steps
        if not steps:
            raise ValueError(f"no snapshots in {self.run.snapshots}")
        arr = np.asarray(steps)
        return int(arr[int(np.argmin(np.abs(arr - step)))])

    def frame(self, step: int) -> Dict[str, np.ndarray]:
        return self.snapshots.load(step)

    # -- event aggregates --------------------------------------------------

    def _aggregate(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """{kind: (steps, counts)}, one pass per kind.

        ``np.unique(..., return_counts=True)`` rather than a per-step mask: a
        30k run logs millions of events over hundreds of refinement steps, and
        the quadratic form is what makes a viewer feel broken on startup.
        """
        if self._per_step is not None:
            return self._per_step
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.run.events.exists():
            from .events import read_events

            t = read_events(self.run.events)
            step = t.column("step").to_numpy(zero_copy_only=False)
            kind = np.asarray(t.column("kind").to_pylist())
            for name in ("clone", "split", "death"):
                s, c = np.unique(step[kind == name], return_counts=True)
                out[name] = (s.astype(np.int32), c.astype(np.int64))
        self._per_step = out
        return out

    @property
    def resets(self) -> np.ndarray:
        return self.lineage.resets if self.lineage else np.empty(0, dtype=np.int32)

    @property
    def is_curriculum(self) -> bool:
        return bool(self.stages)

    @property
    def stage_steps(self) -> np.ndarray:
        """Stage-boundary steps, from stages.json or failing that the event log."""
        if self.stages:
            return np.asarray([s.step for s in self.stages[1:]], dtype=np.int32)
        return self.lineage.stage_steps if self.lineage else np.empty(0, dtype=np.int32)

    @property
    def order(self) -> List[int]:
        """The curriculum order, as dataset item indices."""
        return list(self.plan.get("order", [])) if self.plan else []

    def stage_at(self, step: int) -> Optional[StageInfo]:
        """The curriculum stage in force at ``step``, or None."""
        out = None
        for stage in self.stages:
            if stage.step <= step:
                out = stage
            else:
                break
        return out

    def active_images(self, step: int) -> "ImageRoles":
        """Which images are doing what at ``step``, as dataset item indices.

        In incremental mode the active set is a prefix of the curriculum order,
        so the image count alone would reconstruct it. In groups mode it is one
        group and everything else already seen is *resting* -- seen, seeded, and
        not currently receiving gradient. That distinction is the point of the
        mode, so the active set is read from the record rather than inferred.
        """
        order = self.order
        stage = self.stage_at(step)
        if not order or stage is None:
            return ImageRoles(list(order), [], [], None)
        seen = order[: stage.n_images]
        active = list(stage.active) if stage.active else seen
        resting = [i for i in seen if i not in set(active)]
        return ImageRoles(active, resting, order[stage.n_images :], stage.added)

    def deltas(self, stage_index: int) -> Optional[Dict[str, np.ndarray]]:
        """Per-Gaussian change over one stage, or None if it was not recorded."""
        if stage_index in self._deltas:
            return self._deltas[stage_index]
        out = None
        match = [s for s in self.stages if s.index == stage_index and s.deltas]
        if match:
            path = self.run.root / match[0].deltas
            if path.exists():
                with np.load(path) as z:
                    out = {k: z[k] for k in z.files}
        self._deltas[stage_index] = out
        return out

    @property
    def densify_steps(self) -> np.ndarray:
        agg = self._aggregate()
        if not agg:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate([s for s, _ in agg.values()])).astype(np.int32)

    def last_densify(self, step: int) -> Dict[str, int]:
        """Clone/split/prune counts at the most recent refinement at or before ``step``."""
        agg = self._aggregate()
        if not agg:
            return {}
        cand = self.densify_steps
        cand = cand[cand <= step]
        if cand.size == 0:
            return {}
        at = int(cand[-1])
        out = {"step": at}
        for name in ("clone", "split", "death"):
            s, c = agg.get(name, (np.empty(0), np.empty(0)))
            hit = np.nonzero(s == at)[0]
            out[name] = int(c[hit[0]]) if hit.size else 0
        return out

    # -- narration ---------------------------------------------------------

    def narrate(self, step: int, ids: Optional[np.ndarray] = None) -> str:
        """The text panel: what phase this is, and what just happened.

        The point of the viewer is that the schedule in ``how_it_works.md`` is
        legible while you scrub, so the panel says what the strategy is doing at
        this step, not just how many Gaussians there are.
        """
        s = self.schedule
        phase = s.phase(step)
        lines = [f"step {step} - {phase}"]
        if phase == "warm-up":
            lines.append(
                f"No densification until {s.refine_start_iter}. Shapes, colours and"
                " opacities are optimising over the fixed SfM population."
            )
        elif phase == "densification":
            lines.append(
                f"Clone/split/prune every {s.refine_every} steps;"
                f" opacity reset every {s.reset_every}; stops at {s.refine_stop_iter}."
            )
        else:
            lines.append(
                f"Population frozen at {s.refine_stop_iter}. Only shapes, colours"
                " and opacities still move."
            )
        lines.append(f"SH degree {s.sh_degree_at(step)} of {s.sh_degree}")

        stage = self.stage_at(step)
        if stage is not None:
            roles = self.active_images(step)
            head = f"stage {stage.index}: {stage.n_images} images seen"
            if stage.group is not None:
                head += f", training group {stage.group} (round {stage.round})"
            if stage.image:
                head += f", just added {stage.image}"
            lines.append(head)
            if stage.n_active and stage.n_active != stage.n_images:
                lines.append(
                    f"{stage.n_active} active now, {len(roles.resting)} resting"
                )
            if roles.pending:
                lines.append(f"{len(roles.pending)} photos still pending")
            if stage.steps_per_active_image:
                lines.append(
                    f"{stage.steps_per_active_image:.0f} steps per active image this stage"
                )
            if stage.n_seeded:
                lines.append(f"{stage.n_seeded:,} Gaussians seeded from newly triangulable points")
            if stage.gain is not None:
                lines.append(
                    f"that view: {stage.psnr_before:.2f} dB blind -> "
                    f"{stage.psnr_after:.2f} dB ({stage.gain:+.2f})"
                )

        if ids is not None:
            lines.append(f"{ids.size:,} Gaussians")
            if self.lineage is not None:
                counts = self.lineage.origin_counts(ids)
                lines.append(
                    "origin: "
                    + ", ".join(f"{k} {v:,}" for k, v in counts.items() if v)
                )
        last = self.last_densify(step)
        if last:
            lines.append(
                f"last refinement at {last['step']}: +{last.get('clone', 0):,} clone,"
                f" +{last.get('split', 0):,} split, -{last.get('death', 0):,} pruned"
            )
        resets = self.resets
        resets = resets[resets <= step]
        if resets.size:
            lines.append(f"last opacity reset at {int(resets[-1]):,}")
        return "\n".join(lines)


def origin_palette() -> np.ndarray:
    """``float32[len(BIRTH_KINDS) + 1, 3]``; the last row is unknown."""
    rows = [ORIGIN_COLORS[k] for k in BIRTH_KINDS] + [UNKNOWN_COLOR]
    return np.asarray(rows, dtype=np.float32)


def colors_by_origin(lineage: Lineage, ids: np.ndarray) -> np.ndarray:
    """float32[N, 3] flat colours: blue sfm, green seed, orange clone, red split."""
    palette = origin_palette()
    codes = lineage.origin_codes(ids).astype(np.int64)
    codes[codes < 0] = len(BIRTH_KINDS)  # unknown -> the grey row
    return palette[codes]


def colors_by_age(
    lineage: Lineage, ids: np.ndarray, step: int, colormap: str = "turbo"
) -> np.ndarray:
    """float32[N, 3] by steps-since-birth, newest at one end of the ramp.

    Normalised against the oldest age present in this frame rather than the run
    length: late in a run almost everything is young, and a fixed scale would
    collapse the whole frame onto one colour.
    """
    from matplotlib import colormaps

    ages = lineage.ages(ids, step)
    hi = float(ages.max()) if ages.size and ages.max() > 0 else 1.0
    t = np.clip(np.where(ages >= 0, ages, 0.0) / hi, 0.0, 1.0)
    return colormaps[colormap](t)[:, :3].astype(np.float32)


def highlight(ids: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    """Boolean mask over ``ids`` for the members of ``selected`` still alive."""
    return np.isin(np.asarray(ids), np.asarray(selected, dtype=np.int64))


def colors_by_delta(
    ids: np.ndarray,
    deltas: Dict[str, np.ndarray],
    field: str = "d_means",
    colormap: str = "inferno",
    unchanged: Tuple[float, float, float] = (0.15, 0.16, 0.18),
) -> np.ndarray:
    """float32[N, 3] by how much each Gaussian changed over a stage.

    Gaussians that were not alive at both ends of the stage are drawn near-black
    rather than at the bottom of the ramp: "born during this stage" and "did not
    move" are completely different statements, and a shared colour would merge
    them.

    Normalised to the 99th percentile, not the maximum. One Gaussian that flew
    across the room -- and there always is one -- would otherwise compress every
    real change into the first percent of the ramp.
    """
    from matplotlib import colormaps

    out = np.tile(np.asarray(unchanged, dtype=np.float32), (len(ids), 1))
    values = deltas.get(field)
    if values is None or len(deltas.get("ids", ())) == 0:
        return out
    order = np.argsort(deltas["ids"])
    keys = deltas["ids"][order]
    pos = np.searchsorted(keys, ids)
    pos = np.clip(pos, 0, keys.size - 1)
    hit = keys[pos] == ids
    if not hit.any():
        return out
    v = np.asarray(values, dtype=np.float32)[order][pos[hit]]
    hi = float(np.percentile(np.asarray(values, dtype=np.float32), 99))
    t = np.clip(v / max(hi, 1e-9), 0.0, 1.0)
    out[hit] = colormaps[colormap](t)[:, :3].astype(np.float32)
    return out
