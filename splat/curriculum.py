"""The image-by-image curriculum: which photo to add next, and when (Phase 5).

A normal run shows the model all 32 photos from step 0. This one starts with
three and adds one at a time, so the question "what did *this* photo teach it"
has an answer you can point at: the view rendered before the model had ever
seen it, the same view after its stage, and the Gaussians that changed in
between.

Three things make that honest rather than merely pretty:

- **One coordinate frame.** SfM runs once, over all the photos, before training.
  Stages share its poses, so nothing moves between stages and a Gaussian's
  position means the same thing at stage 3 as at stage 30. Re-running SfM per
  stage would make every comparison meaningless.
- **The test set never participates.** Every 8th image is held out by gsplat's
  own split, and it is excluded from the ordering, the seeding and the sampler.
  A test image that helped choose the curriculum is not a test image.
- **Two index spaces, kept apart.** The sampler, ``data["image_id"]`` and the
  trainer all speak *dataset item* indices (0..len(trainset)-1, train images
  only). Covisibility and ``point_indices`` speak *parser image* indices (0..31,
  train and test interleaved). This module works in item space throughout and
  converts at the one boundary, because mixing them silently trains on the test
  set.

Ordering
--------
Start with the three images sharing the most 3D points -- the best-triangulated
corner of the room, where the model has the most to work with. Then repeatedly
add whichever pending image shares the most points with the union of the active
ones, so the reconstruction grows outward from that corner instead of jumping
around the room and having to bridge a gap it cannot see.

Two modes, because they answer different questions
--------------------------------------------------
**incremental** -- the narrative mode. Images accumulate one at a time and the
sampler draws uniformly from everything active, which is what makes the
blind-guess render possible: a photo the model has genuinely never seen. It is
the mode the viewer's story is built on.

It is a poor *instrument*, though, and the reason is worth stating in full.
Under uniform sampling from a growing pool, the expected number of gradient
steps an image receives is ``sum over its stages of stage_length / n_active``.
Measured on ``room-1`` with 28 images, a 200-step warm-up and equal 152-step
stages: **385 steps for the first image, 5 for the last -- a 71x spread**, and
still 7x inside a photo's own measurement window. A per-photo improvement
measured that way is partly measuring stage length over pool size.

``steps_per_image`` fixes the measurement window: stage length becomes
proportional to the active count, so every active image gets exactly ``e`` of
its own steps per stage whatever the pool size. It cannot fix the *total*, and
no incremental design can -- an image added third is present for twenty more
stages than one added last. That asymmetry is what "incremental" means.

**groups** -- the measurement mode. The images are partitioned into ``G``
disjoint groups of ``group_size`` and visited round-robin for ``rounds`` passes.
Every image then receives exactly the same number of gradient steps by
construction, so a difference between groups is a difference between the images.
The cost is that only one group is training at a time, so the schedule needs
enough rounds to avoid ending fitted to whichever group went last.

Handing over before densification stops
---------------------------------------
gsplat grows the population (clone and split) only between ``refine_start_iter``
and ``refine_stop_iter`` -- 500 and 15,000 by default -- whatever else the run
is doing. Both modes used to stretch their schedule over all of ``max_steps``,
and at 30k steps that broke them. Measured on ``room-1`` at factor 2:

- **incremental** introduced its last 8 photos after step 15,000. The population
  went from 2,686,289 to 2,686,712 over those 14,500 steps: the photos arrived
  with nothing left to build what they showed. Held-out PSNR 14.26 against
  18.84 for a plain run.
- **groups** never trained on more than 4 images at once, so the model ended
  fitted to whichever group went last, swinging 14.4-15.2 dB between visits.
  Held-out 14.77.

So ``end_step`` compresses the schedule into ``[0, end_step)`` and a final
*consolidation* stage trains on every image together until ``max_steps``. The
trainer defaults it to 80% of ``refine_stop_iter``, which leaves the last photo
the final fifth of densification. Short runs (``end_step >= max_steps``) and
the ablation's fixed-N runs (no stages to schedule) are unchanged.

Two sets, kept apart
--------------------
``Stage.active`` is what the sampler may draw from *now*; ``Stage.seen`` is
every image visited at or before this stage. They are equal in incremental mode
and differ in groups mode, where training is restricted to one group but
triangulability -- and therefore seeding -- is cumulative. Seeding from the
active group alone would starve it: points that two images in different groups
share would never become seedable at all.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: above this many train images, the exact best-triple search is replaced by a
#: greedy one (best pair, then best third). 120 images is ~7k pairs, which is
#: still instant; the exact search is O(n^2) matvecs and stops being so.
EXACT_TRIPLE_LIMIT = 120


class Covisibility:
    """Which images see which SfM points, as a boolean matrix.

    Rows are **dataset item** indices, not parser image indices: this is the
    space the curriculum and the sampler work in, and the conversion happens
    here, once.
    """

    def __init__(self, visible: np.ndarray) -> None:
        self.visible = np.asarray(visible, dtype=bool)  # [n_items, n_points]
        self._numeric: Optional[np.ndarray] = None

    @property
    def numeric(self) -> np.ndarray:
        """``visible`` as float32, for counting matmuls.

        Not a micro-optimisation: ``bool @ bool`` in numpy is a *logical*
        matmul, so it answers "do these share any point" where every ranking
        here needs "how many". It returns a bool array and ``argmax`` then picks
        the first True, which is a plausible-looking wrong answer -- it cost a
        duplicated image in the curriculum order before the tests caught it.
        float32 rather than int32 because BLAS handles it, and counts stay exact
        below 2^24 points per image.

        Costs ``n_images x n_points x 4`` bytes, built on first use: 0.8 MB for
        room-1, ~0.5 GB for a 120-image scene with a million points.
        """
        if self._numeric is None:
            self._numeric = self.visible.astype(np.float32)
        return self._numeric

    @classmethod
    def from_parser(cls, parser, item_to_parser: Sequence[int]) -> "Covisibility":
        n_points = len(parser.points)
        visible = np.zeros((len(item_to_parser), n_points), dtype=bool)
        for item, pidx in enumerate(item_to_parser):
            name = parser.image_names[pidx]
            pts = parser.point_indices.get(name)
            if pts is not None and len(pts):
                visible[item, np.asarray(pts, dtype=np.int64)] = True
        return cls(visible)

    @property
    def n_items(self) -> int:
        return int(self.visible.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.visible.shape[1])

    def counts(self) -> np.ndarray:
        """Points seen per image."""
        return self.visible.sum(1)

    def pair_counts(self) -> np.ndarray:
        """``[n, n]`` of shared points. One matmul, not a Python double loop."""
        return (self.numeric @ self.numeric.T).astype(np.int64)

    def best_triple(self) -> Tuple[int, int, int]:
        """The three images sharing the most points, all three at once.

        Not the three best *pairs*: a chain of two good pairs can share almost
        nothing across all three, and the initial population is built from
        points that three images agree on.
        """
        n = self.n_items
        if n < 3:
            raise ValueError(f"need at least 3 train images, got {n}")
        if n <= EXACT_TRIPLE_LIMIT:
            best, best_n = (0, 1, 2), -1
            for i, j in itertools.combinations(range(n), 2):
                both = self.visible[i] & self.visible[j]
                if not both.any():
                    continue
                shared = self.numeric @ both.astype(np.float32)  # every k at once
                shared[[i, j]] = -1.0
                k = int(np.argmax(shared))
                if int(shared[k]) > best_n:
                    best_n, best = int(shared[k]), tuple(sorted((i, j, k)))
            return best  # type: ignore[return-value]
        pairs = self.pair_counts().copy()
        np.fill_diagonal(pairs, -1)
        i, j = np.unravel_index(int(np.argmax(pairs)), pairs.shape)
        both = self.visible[int(i)] & self.visible[int(j)]
        shared = self.numeric @ both.astype(np.float32)
        shared[[int(i), int(j)]] = -1.0
        return tuple(sorted((int(i), int(j), int(np.argmax(shared)))))  # type: ignore[return-value]

    def next_image(self, active: Sequence[int]) -> Optional[int]:
        """The pending image sharing the most points with the active union.

        Ties go to the lowest index, so the order is a function of the scene and
        not of numpy's argmax tie-breaking on a different build.
        """
        active = list(active)
        pending = [i for i in range(self.n_items) if i not in set(active)]
        if not pending:
            return None
        covered = self.visible[active].any(0).astype(np.float32)
        shared = self.numeric[pending] @ covered
        return int(pending[int(np.argmax(shared))])

    def order(self) -> List[int]:
        """The full curriculum order: the initial triple, then one at a time."""
        active = list(self.best_triple())
        while True:
            nxt = self.next_image(active)
            if nxt is None:
                return active
            active.append(nxt)

    def triangulable(self, active: Sequence[int], min_views: int = 2) -> np.ndarray:
        """Boolean mask over SfM points with ``min_views`` active images in track.

        Two is the floor for triangulation: a point seen by one camera is a ray,
        not a position. Points below it exist in the model only because some
        other stage will earn them.
        """
        if len(active) == 0:
            return np.zeros(self.n_points, dtype=bool)
        return self.visible[list(active)].sum(0) >= min_views

    def newly_triangulable(
        self, active: Sequence[int], previous: Sequence[int], min_views: int = 2
    ) -> np.ndarray:
        """Points this stage unlocked and the last one had not."""
        return self.triangulable(active, min_views) & ~self.triangulable(previous, min_views)


@dataclass(frozen=True)
class Stage:
    """One step of the curriculum."""

    index: int
    step: int  #: the training step this stage begins on
    added: Optional[int]  #: item index first seen here; None if nothing is new
    active: Tuple[int, ...]  #: what the sampler may draw from during this stage
    #: every item seen at or before this stage. Equal to ``active`` in
    #: incremental mode; a superset in groups mode, where training is restricted
    #: to one group but seeding stays cumulative.
    seen: Tuple[int, ...] = ()
    length: int = 0  #: steps in this stage, 0 for the last (it runs to max_steps)
    group: Optional[int] = None  #: which group, in groups mode
    round: Optional[int] = None  #: which round-robin pass, in groups mode
    #: the closing stage after ``end_step``: every image seen, all of them
    #: training together, nothing new introduced
    consolidate: bool = False

    def __post_init__(self) -> None:
        if not self.seen:
            object.__setattr__(self, "seen", self.active)

    @property
    def n_images(self) -> int:
        """Images seen so far -- the x axis of every per-image-count chart."""
        return len(self.seen)

    @property
    def n_active(self) -> int:
        """Images the sampler draws from during this stage."""
        return len(self.active)

    @property
    def steps_per_active_image(self) -> float:
        """Expected gradient steps each active image gets during this stage."""
        return self.length / self.n_active if self.length and self.n_active else 0.0


@dataclass
class CurriculumConfig:
    #: "incremental" accumulates images one at a time (the narrative mode);
    #: "groups" visits disjoint groups round-robin at equal exposure (the
    #: measurement mode). See the module docstring on why both exist.
    mode: str = "incremental"
    init_images: int = 3
    #: hard cap on how many images the run ever sees; 0 means all of them.
    #: With ``max_images == init_images`` there are no added stages at all,
    #: which is the ablation condition: a fixed image count for the whole run,
    #: so "more images" is separated from "more training time".
    max_images: int = 0
    #: steps on the initial images before the first addition. Defaults to
    #: gsplat's refine_start_iter, so the starting population gets the same
    #: densification-free warm-up it would get in a normal run.
    warmup_steps: int = 500
    #: steps per added image; 0 divides whatever is left of max_steps evenly
    stage_steps: int = 0
    #: minimum active images in a point's track before it may be seeded
    min_views: int = 2
    #: a seed must be at least this many multiples of the SfM cloud's typical
    #: point spacing away from every existing Gaussian
    min_spacing: float = 1.0
    #: cap per stage, so one wide-angle photo cannot inject a million Gaussians
    max_seeds_per_stage: int = 200_000

    # -- incremental pacing ------------------------------------------------
    #: gradient steps each active image gets per stage, so stage length scales
    #: with the active count. 0 derives it from max_steps so the whole budget
    #: is used. This is what keeps a photo's measurement window comparable
    #: across stages; see the module docstring.
    steps_per_image: int = 0

    # -- groups mode -------------------------------------------------------
    #: images per group. Every image lands in exactly one group.
    group_size: int = 4
    #: round-robin passes over the groups. One pass would leave the model
    #: fitted to whichever group went last.
    rounds: int = 4

    # -- both modes --------------------------------------------------------
    #: step by which the schedule has introduced every image; from here to
    #: ``max_steps`` one consolidation stage trains on all of them together.
    #: 0 (or anything >= max_steps) lets the schedule fill the whole run.
    #: See "Handing over before densification stops" in the module docstring.
    end_step: int = 0


class Curriculum:
    """The schedule, and what each stage unlocks.

    Pure bookkeeping over item indices and point masks -- no torch, no trainer.
    The trainer asks it two questions: "which images may I sample right now"
    and "did a stage just start".
    """

    def __init__(
        self,
        covis: Covisibility,
        max_steps: int,
        cfg: Optional[CurriculumConfig] = None,
    ) -> None:
        self.covis = covis
        self.cfg = cfg or CurriculumConfig()
        self.max_steps = int(max_steps)
        #: the steps the image schedule itself may use; the rest is consolidation
        end = int(self.cfg.end_step)
        self.budget = end if 0 < end < self.max_steps else self.max_steps
        #: the full ordering. ``max_images`` truncates the *schedule*, not this,
        #: so an ablation condition still uses the same ordering as the full run
        #: -- N=5 is the first five images a full run would have used, which is
        #: what makes the conditions nested rather than merely different.
        self.order = covis.order()
        self.stages = self._schedule()
        self._current = 0

    @property
    def images(self) -> List[int]:
        """The images this run will ever see, in curriculum order."""
        return list(self.stages[-1].seen)

    # -- schedule ----------------------------------------------------------

    def _schedule(self) -> List[Stage]:
        order = self.order
        cfg = self.cfg
        if cfg.max_images:
            if cfg.max_images < max(3, int(cfg.init_images)):
                raise ValueError(
                    f"max_images={cfg.max_images} is below init_images={cfg.init_images}"
                )
            order = order[: cfg.max_images]
        if cfg.mode == "groups":
            stages = self._schedule_groups(order)
        elif cfg.mode == "incremental":
            stages = self._schedule_incremental(order)
        else:
            raise ValueError(f"unknown curriculum mode {cfg.mode!r}")
        if self.budget < self.max_steps and len(stages) > 1:
            stages.append(self._consolidation(stages))
        return stages

    def _consolidation(self, stages: List[Stage]) -> Stage:
        """Every image seen, all training together, until ``max_steps``.

        Starts where the schedule actually ended rather than at ``budget``:
        integer stage lengths leave a remainder of a few steps either way.
        """
        last = stages[-1]
        step = last.step + last.length
        return Stage(
            index=len(stages),
            step=step,
            added=None,
            active=last.seen,
            seen=last.seen,
            length=max(0, self.max_steps - step),
            consolidate=True,
        )

    def _schedule_incremental(self, order: List[int]) -> List[Stage]:
        """Images accumulate one at a time.

        Stage length is ``steps_per_image * n_active`` rather than a constant,
        so an image gets the same number of its own gradient steps during its
        stage whether the pool is 4 images or 28. With a constant length the
        window shrinks as 1/n and a late photo's measured improvement is
        depressed for a reason that has nothing to do with the photo.
        """
        cfg = self.cfg
        n_init = max(3, int(cfg.init_images))
        initial = tuple(order[:n_init])
        added = order[n_init:]
        warmup = int(cfg.warmup_steps)
        if not added:
            return [
                Stage(index=0, step=0, added=None, active=initial,
                      length=max(0, self.max_steps))
            ]

        counts = [n_init + 1 + i for i in range(len(added))]
        if cfg.stage_steps > 0:
            # explicit constant length: the confounded pacing, kept for
            # reproducing earlier runs and for deliberate comparisons
            lengths = [int(cfg.stage_steps)] * len(added)
        else:
            per = cfg.steps_per_image or self._derive_steps_per_image(counts, warmup)
            lengths = [max(1, int(per * c)) for c in counts]

        stages = [Stage(index=0, step=0, added=None, active=initial, length=warmup)]
        active = list(initial)
        step = warmup
        for i, item in enumerate(added):
            active.append(item)
            stages.append(
                Stage(
                    index=i + 1,
                    step=step,
                    added=item,
                    active=tuple(active),
                    length=lengths[i],
                )
            )
            step += lengths[i]
        return stages

    def _derive_steps_per_image(self, counts: List[int], warmup: int) -> float:
        """Spread ``max_steps`` over proportional stages.

        ``warmup + e * sum(counts) == max_steps``, so the whole budget is used
        and no flag has to be tuned by hand. At least 1, so a debug run still
        visits every stage instead of silently dropping the tail.
        """
        budget = max(1, self.budget - warmup)
        return max(1.0, budget / max(1, sum(counts)))

    def _schedule_groups(self, order: List[int]) -> List[Stage]:
        """Disjoint groups, visited round-robin at equal exposure.

        Every image is in exactly one group and every group is visited the same
        number of times for the same number of steps, so each image receives
        ``rounds * steps_per_image`` gradient steps -- equal by construction
        rather than by accident. That is the whole reason this mode exists.
        """
        cfg = self.cfg
        size = max(1, int(cfg.group_size))
        rounds = max(1, int(cfg.rounds))
        groups = [tuple(order[i : i + size]) for i in range(0, len(order), size)]
        if len(groups) > 1 and len(groups[-1]) < size:
            # A short tail group would get the same steps spread over fewer
            # images, which is exactly the inequality this mode removes. Fold
            # it into its predecessor and say so.
            groups[-2] = groups[-2] + groups[-1]
            groups.pop()

        per_visit = [
            max(1, int(round(self._group_visit_steps(groups, rounds) * len(g) / size)))
            for g in groups
        ]
        stages: List[Stage] = []
        seen: List[int] = []
        step = 0
        index = 0
        for r in range(rounds):
            for g, members in enumerate(groups):
                first_visit = r == 0
                for item in members:
                    if item not in seen:
                        seen.append(item)
                stages.append(
                    Stage(
                        index=index,
                        step=step,
                        # only a first visit introduces images; a revisit adds
                        # nothing, so there is no blind guess to render
                        added=members[0] if first_visit else None,
                        active=members,
                        seen=tuple(seen),
                        length=per_visit[g],
                        group=g,
                        round=r,
                    )
                )
                step += per_visit[g]
                index += 1
        return stages

    def _group_visit_steps(self, groups: List[Tuple[int, ...]], rounds: int) -> float:
        """Steps per full-size group visit, so the rounds fill the budget."""
        cfg = self.cfg
        if cfg.steps_per_image:
            return float(cfg.steps_per_image * max(1, int(cfg.group_size)))
        total_members = sum(len(g) for g in groups)
        return max(1.0, self.budget / max(1, rounds * total_members)) * max(
            1, int(cfg.group_size)
        )

    @property
    def stage_steps(self) -> int:
        """A representative stage length -- the median, since with proportional
        pacing or unequal groups they are deliberately not all the same."""
        lengths = [s.length for s in self.stages[1:] if s.length]
        return int(np.median(lengths)) if lengths else 0

    @property
    def steps_per_image(self) -> float:
        """Gradient steps an active image gets per stage, at the median stage."""
        rates = [s.steps_per_active_image for s in self.stages[1:] if s.length]
        return float(np.median(rates)) if rates else 0.0

    @property
    def final_step(self) -> int:
        return self.stages[-1].step

    @property
    def boundaries(self) -> List[int]:
        """The steps at which a stage begins, stage 0 excluded."""
        return [s.step for s in self.stages[1:]]

    def stage_at(self, step: int) -> Stage:
        """The stage in force at ``step``."""
        out = self.stages[0]
        for stage in self.stages:
            if stage.step <= step:
                out = stage
            else:
                break
        return out

    def active_at(self, step: int) -> Tuple[int, ...]:
        return self.stage_at(step).active

    @property
    def active(self) -> Tuple[int, ...]:
        return self.stages[self._current].active

    def on_step(self, step: int) -> Optional[Stage]:
        """Advance if a stage begins at ``step``. Returns it, or None.

        Called once per training step, so it walks forward rather than
        rescanning: a 30k-step run asks this 30,000 times.
        """
        started = None
        while (
            self._current + 1 < len(self.stages)
            and self.stages[self._current + 1].step <= step
        ):
            self._current += 1
            started = self.stages[self._current]
        return started

    # -- what a stage unlocks ---------------------------------------------

    def seed_mask(self, stage: Stage) -> np.ndarray:
        """SfM points this stage made triangulable for the first time.

        Keyed on ``seen``, not ``active``. In groups mode only one group trains
        at a time, and a point shared by two images in different groups would
        never have two *active* views at once -- so seeding from the active set
        would starve exactly the points that tie the groups together.
        """
        previous = self.stages[stage.index - 1].seen if stage.index > 0 else ()
        return self.covis.newly_triangulable(stage.seen, previous, self.cfg.min_views)

    def initial_mask(self) -> np.ndarray:
        """Hook 2: the points the initial images can triangulate.

        This is what the run starts from, rather than the whole cloud. Starting
        from all 6,762 points would put Gaussians in parts of the room no active
        photo has seen, and they would sit there un-supervised, un-pruned and
        wrong until some later stage happened to look at them.
        """
        return self.covis.triangulable(self.stages[0].seen, self.cfg.min_views)

    def summary(self) -> List[Dict]:
        """One row per stage, for the run record and the report."""
        rows = []
        for stage in self.stages:
            mask = self.initial_mask() if stage.index == 0 else self.seed_mask(stage)
            rows.append(
                {
                    "stage": stage.index,
                    "step": stage.step,
                    "added": stage.added,
                    "n_images": stage.n_images,
                    "n_active": stage.n_active,
                    "length": stage.length,
                    "steps_per_active_image": round(stage.steps_per_active_image, 2),
                    "group": stage.group,
                    "round": stage.round,
                    "consolidate": stage.consolidate,
                    "n_points_unlocked": int(mask.sum()),
                }
            )
        return rows

    def exposure(self) -> Dict[int, float]:
        """Expected gradient steps per image, under uniform sampling.

        The number that says whether a per-image comparison is fair. Recorded
        in every run so the per-photo improvements can be read with the right
        amount of trust rather than taken at face value: on ``room-1`` with
        constant-length stages it spans 71x, which is most of the way to
        meaningless.
        """
        out: Dict[int, float] = {i: 0.0 for i in self.order}
        for k, stage in enumerate(self.stages):
            length = stage.length
            if not length:
                nxt = self.stages[k + 1].step if k + 1 < len(self.stages) else self.max_steps
                length = max(0, nxt - stage.step)
            if not stage.active:
                continue
            share = length / len(stage.active)
            for item in stage.active:
                out[item] = out.get(item, 0.0) + share
        return out

    def exposure_spread(self) -> float:
        """max/min expected exposure over the images this run sees. 1.0 is fair."""
        seen = set(self.images)
        vals = [v for k, v in self.exposure().items() if k in seen]
        lo = min(vals) if vals else 0.0
        return float(max(vals) / lo) if lo > 0 else float("inf")


def point_spacing(points: np.ndarray, k: int = 4, sample: int = 20_000) -> float:
    """Median distance to the 3rd nearest neighbour in the SfM cloud.

    The natural unit for "is there already a Gaussian here". Absolute distances
    are meaningless across scenes -- `normalize_world_space` rescales every one
    of them -- and scene_scale describes the camera rig, not the density of the
    reconstruction.
    """
    import torch

    from .append import _knn_scales  # noqa: F401  (shares the vendored knn import)
    import sys

    from .paths import repo_root

    vendor = repo_root() / "third_party" / "gsplat_examples"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from utils import knn  # type: ignore

    pts = torch.from_numpy(np.asarray(points)).float()
    if pts.shape[0] > sample:
        idx = torch.randperm(pts.shape[0])[:sample]
        pts = pts[idx]
    d = knn(pts, k)[:, 1:]
    return float(d.median().item())


def far_from_existing(
    points: np.ndarray,
    means,
    min_dist: float,
    budget: int = 1 << 26,
) -> np.ndarray:
    """Boolean mask over ``points``: True where no Gaussian is within ``min_dist``.

    Seeding a point the model already covers is worse than useless -- it adds a
    Gaussian duplicating one already being optimized, and the two then compete
    for the same gradient.

    Chunked on **both** sides. By the late stages ``means`` is millions of rows,
    and a full pairwise matrix against even a few thousand candidates is tens of
    gigabytes; chunking only the candidates still allocates
    ``len(candidates_chunk) x len(means)``. ``budget`` caps the elements in
    flight (2^26 floats = 256 MB), which is what keeps this exact instead of
    approximate -- a voxel prefilter would be cheaper and would also reject
    points that are merely in an adjacent cell.
    """
    import torch

    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if means is None or len(means) == 0:
        return np.ones(len(points), dtype=bool)

    m = torch.as_tensor(means)
    m = m.to(torch.float32)
    pts = torch.as_tensor(np.asarray(points, dtype=np.float32)).to(m.device)

    n_means = m.shape[0]
    m_chunk = max(1, min(n_means, budget // 256))
    p_chunk = max(1, budget // m_chunk)

    out = torch.empty(pts.shape[0], dtype=torch.bool, device=m.device)
    for lo in range(0, pts.shape[0], p_chunk):
        block = pts[lo : lo + p_chunk]
        nearest = torch.full((block.shape[0],), float("inf"), device=m.device)
        for mlo in range(0, n_means, m_chunk):
            d = torch.cdist(block, m[mlo : mlo + m_chunk]).amin(1)
            nearest = torch.minimum(nearest, d)
        out[lo : lo + p_chunk] = nearest >= min_dist
    return out.cpu().numpy()
