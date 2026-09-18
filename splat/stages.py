"""The trainer-facing side of the curriculum (Phase 5).

``curriculum.py`` decides *what* to add and *when*; this decides what happens at
the moment it is added. Split because that module is pure numpy over index
sets and this one needs torch, the optimizers, and the trainer's renderer --
and because the ordering logic is worth testing without any of them.

Two things live here:

- ``ActiveSampler`` -- hook 3. The train loader draws only from images the
  curriculum has activated.
- ``StageDriver`` -- everything that happens at a stage boundary, in one object,
  so the edit inside the vendored trainer stays a single call.

What changed, per Gaussian
--------------------------
A stage's effect is not only the new view getting sharper. The driver snapshots
the population at stage start and diffs it at stage end, matched **by id** and
not by row: densification reorders and prunes rows constantly, so position ``i``
at the start of a stage and position ``i`` at the end are unrelated Gaussians.
Matching on ids is the whole reason ``lineage.py`` hands them out.

The diff is written per stage to ``deltas/stage_NNN.npz`` -- one row per Gaussian
alive at both ends -- which is what lets the viewer answer "which Gaussians did
this photo change". Four scalars, each chosen to be readable on its own scale:
positional movement in world units, log-scale change (so it reads as a relative
size change), opacity change in probability space, and DC colour change.

The order at a boundary is not arbitrary
----------------------------------------
1. Render the new view. This is the **blind guess**: what the model predicts
   for a photo it has never been shown, using only what the earlier photos
   taught it. It has to happen before step 2.
2. Inject seeds for the SfM points this photo just made triangulable. These
   positions come from the SfM solve, which saw *all* the photos -- so a seed
   already carries information from the new image, and rendering after seeding
   would quietly turn the blind guess into a partially sighted one.
3. Log the stage, so the viewer and the report can mark it.

Prefetch, honestly
------------------
The loader runs 4 workers with the default prefetch of 2, so ~8 indices are
drawn from the sampler ahead of the step that consumes them. A stage boundary is
therefore blurred by up to ~8 steps *in one direction only*: those indices were
drawn from the previous active set, so the new image cannot be trained on before
its boundary. The blind guess stays blind. The new image simply joins a handful
of steps late, which against stages of 1,000+ steps is not worth removing the
prefetch for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .curriculum import Curriculum, Stage, far_from_existing
from .report import abs_error, error_map, psnr, strip


class ActiveSampler(torch.utils.data.Sampler):
    """Yields dataset item indices from the curriculum's active set, forever.

    Infinite on purpose. A finite sampler would make one epoch as short as the
    active set -- three images at the start -- and the trainer would rebuild the
    loader iterator every three steps. Because it never raises ``StopIteration``
    the trainer's re-iterate branch simply never fires, and the active set is
    re-read at the top of every shuffled pass, which is how a stage advance
    reaches the sampler at all.
    """

    def __init__(self, curriculum: Curriculum, generator: Optional[torch.Generator] = None) -> None:
        self.curriculum = curriculum
        self.generator = generator

    def __iter__(self):
        while True:
            active = list(self.curriculum.active)
            order = torch.randperm(len(active), generator=self.generator).tolist()
            for i in order:
                yield active[i]

    def __len__(self) -> int:
        # Only ever used for len(dataloader), which the trainer does not call.
        return len(self.curriculum.active)


@dataclass
class StageRecord:
    """What one stage did. Serialized to ``stages.json``."""

    stage: int
    step: int
    added: Optional[int]
    image: Optional[str]
    n_images: int  #: images seen so far
    n_active: int = 0  #: images the sampler drew from during this stage
    length: int = 0
    #: expected gradient steps each active image got here. The number that says
    #: whether this stage's before/after gain is comparable to another's.
    steps_per_active_image: float = 0.0
    group: Optional[int] = None
    round: Optional[int] = None
    group_images: Optional[List[str]] = None
    #: item indices the sampler drew from. Needed by the viewer: in groups mode
    #: the active set is one group, not a prefix of the curriculum order, so it
    #: cannot be reconstructed from the image count alone.
    active: Optional[List[int]] = None
    n_points_unlocked: int = 0
    n_seeded: int = 0
    n_rejected_near: int = 0
    n_gaussians_before: int = 0
    n_gaussians_after: int = 0
    psnr_before: Optional[float] = None
    psnr_after: Optional[float] = None
    end_step: Optional[int] = None
    panel: Optional[str] = None
    #: population churn over the stage, matched by id
    n_survived: int = 0
    n_born: int = 0
    n_died: int = 0
    #: mean |delta| over the survivors, per param group
    d_means_mean: Optional[float] = None
    d_means_p99: Optional[float] = None
    d_scale_mean: Optional[float] = None
    d_opacity_mean: Optional[float] = None
    d_color_mean: Optional[float] = None
    deltas: Optional[str] = None


class StageDriver:
    """Owns the stage boundary: blind guess, seeds, records.

    ``render_fn`` takes one trainset item dict and returns ``[1, H, W, 3]`` in
    0..1 -- the trainer supplies it, so none of its rasterization kwargs leak in
    here.
    """

    def __init__(
        self,
        plan: Curriculum,
        trainset,
        parser,
        out_dir: Path,
        render_fn: Optional[Callable[[Dict[str, Any]], torch.Tensor]] = None,
        strategy=None,
        events=None,
        device: str = "cuda",
        sh_degree: int = 3,
        min_spacing_units: float = 1.0,
        spacing: float = 0.0,
        max_seeds: int = 200_000,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
    ) -> None:
        self.plan = plan
        self.trainset = trainset
        self.parser = parser
        self.out_dir = Path(out_dir)
        self.render_fn = render_fn
        self.strategy = strategy
        self.events = events
        self.device = device
        self.sh_degree = sh_degree
        self.min_dist = float(min_spacing_units) * float(spacing)
        self.max_seeds = int(max_seeds)
        self.init_opacity = init_opacity
        self.init_scale = init_scale
        self.records: List[StageRecord] = []
        self._open: Optional[StageRecord] = None
        self._before: Optional[np.ndarray] = None
        self._opened_at: Optional[Dict[str, np.ndarray]] = None
        self._seeded = np.zeros(len(parser.points), dtype=bool)

    # -- wiring ------------------------------------------------------------

    def sampler(self) -> ActiveSampler:
        return ActiveSampler(self.plan)

    @property
    def eval_steps(self) -> List[int]:
        """Stage boundaries, for the trainer's own eval to land on.

        Reusing ``cfg.eval_steps`` rather than computing metrics here means the
        per-stage numbers are produced by the trainer's PSNR/SSIM/LPIPS on its
        own held-out split -- the same code path the baseline used, so a
        curriculum run's test curve is comparable to it.
        """
        return list(self.plan.boundaries)

    def image_name(self, item: Optional[int]) -> Optional[str]:
        if item is None:
            return None
        return self.parser.image_names[int(self.trainset.indices[int(item)])]

    # -- the boundary ------------------------------------------------------

    def on_step(self, step: int, params, optimizers, state, scene=None) -> Optional[Stage]:
        """Advance the curriculum if a stage begins here. Returns it, or None."""
        stage = self.plan.on_step(step)
        if stage is None:
            return None
        n_before = int(params["means"].shape[0])
        if self._open is None and not self.records:
            # Stage 0 was never opened. begin() should have done it before the
            # loop; falling back here keeps "3 images" a row in the report like
            # every other image count rather than a gap at the origin of the
            # test-PSNR curve -- but with no stage-start snapshot, so its delta
            # row is the one that will be missing.
            self._begin(self.plan.stages[0], 0, int(self.plan.initial_mask().sum()))
        self._close(step, n_before, params, state)
        self._begin(stage, step, n_before, params, state)
        if self.events is not None:
            self.events.add_stage(step, -1 if stage.added is None else stage.added)
        self._seed(stage, step, params, optimizers, state, scene)
        return stage

    def begin(self, step: int, params, state) -> None:
        """Open stage 0 before the training loop starts.

        Separate from ``on_step`` because no boundary fires at step 0, and
        without it the initial images have no stage-start snapshot to diff
        against -- so the warm-up would be the one stage whose effect on the
        population is unmeasured.
        """
        if self._open is not None or self.records:
            return
        # ids are deferred to the first post-backward, so at step 0 there is no
        # id vector to key a snapshot on yet. Assigning them here is the same
        # assignment the strategy would make on this very step -- idempotent,
        # and it logs the sfm births exactly once either way.
        ensure = getattr(self.strategy, "_ensure_ids", None)
        if ensure is not None:
            ensure(params, state, step)
        self._begin(self.plan.stages[0], step, int(params["means"].shape[0]), params, state)

    def _begin(
        self, stage: Stage, step: int, n_before: int, params=None, state=None
    ) -> None:
        mask = self.plan.seed_mask(stage) if stage.index else self.plan.initial_mask()
        self._open = StageRecord(
            stage=stage.index,
            step=step,
            added=stage.added,
            image=self.image_name(stage.added),
            n_images=stage.n_images,
            n_active=stage.n_active,
            length=stage.length,
            steps_per_active_image=round(stage.steps_per_active_image, 3),
            group=stage.group,
            round=stage.round,
            group_images=(
                None if stage.group is None
                else [self.image_name(i) or "?" for i in stage.active]
            ),
            active=[int(i) for i in stage.active],
            n_points_unlocked=int(mask.sum()),
            n_gaussians_before=n_before,
        )
        self._before = self._render(stage.added)
        # Before seeding, so this stage's seeds count as born during it -- they
        # are part of what the photo did, not part of the baseline.
        self._opened_at = self._snapshot(params, state)

    def _close(self, step: int, n_now: int, params=None, state=None) -> None:
        """Finish the stage that was open: after-render, panel, counts, deltas."""
        rec = self._open
        if rec is None:
            return
        rec.end_step = step
        rec.n_gaussians_after = n_now
        self._diff(rec, params, state)
        after = self._render(rec.added)
        gt = self._gt(rec.added)
        if gt is not None and self._before is not None and after is not None:
            rec.psnr_before = psnr(gt, self._before)
            rec.psnr_after = psnr(gt, after)
            rec.panel = self._write_panel(rec, gt, self._before, after)
        self.records.append(rec)
        self._open = None
        self._before = None
        self._opened_at = None

    def finish(self, step: int, params, state=None) -> Path:
        """Close the last stage and write ``stages.json``."""
        self._close(step, int(params["means"].shape[0]), params, state)
        return self.flush()

    def flush(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "stages.json"
        exposure = self.plan.exposure()
        path.write_text(
            json.dumps(
                {
                    "mode": self.plan.cfg.mode,
                    "order": list(self.plan.order),
                    "images": list(self.plan.images),
                    "warmup_steps": self.plan.cfg.warmup_steps,
                    "stage_steps": self.plan.stage_steps,
                    "steps_per_image": round(self.plan.steps_per_image, 3),
                    # Recorded in every run, not just the fair ones: a per-photo
                    # comparison is only as good as this spread, and a reader
                    # who cannot see it will over-trust the numbers above.
                    "exposure": {str(k): round(v, 2) for k, v in sorted(exposure.items())},
                    "exposure_spread": round(self.plan.exposure_spread(), 3),
                    "group_size": self.plan.cfg.group_size if self.plan.cfg.mode == "groups" else None,
                    "rounds": self.plan.cfg.rounds if self.plan.cfg.mode == "groups" else None,
                    "min_dist": self.min_dist,
                    "stages": [asdict(r) for r in self.records],
                },
                indent=2,
            )
        )
        return path

    # -- seeding -----------------------------------------------------------

    def _seed(self, stage: Stage, step: int, params, optimizers, state, scene) -> int:
        """Inject Gaussians for points this stage unlocked, minus the covered ones."""
        from .append import seed_params

        if self.strategy is None or stage.added is None:
            return 0
        mask = self.plan.seed_mask(stage) & ~self._seeded
        candidates = np.nonzero(mask)[0]
        if candidates.size == 0:
            return 0
        if self.min_dist > 0:
            keep = far_from_existing(
                self.parser.points[candidates], params["means"].detach(), self.min_dist
            )
            if self._open is not None:
                self._open.n_rejected_near = int((~keep).sum())
            candidates = candidates[keep]
        if candidates.size > self.max_seeds:
            # A single wide-angle photo should not be able to inject millions.
            candidates = candidates[: self.max_seeds]
        if candidates.size == 0:
            return 0

        rows = seed_params(
            self.parser.points,
            self.parser.points_rgb,
            select=candidates,
            sh_degree=self.sh_degree,
            init_opacity=self.init_opacity,
            init_scale=self.init_scale,
            device=self.device,
        )
        self.strategy.append_seeds(params, optimizers, state, rows, step, scene=scene)
        self._seeded[candidates] = True
        if self._open is not None:
            self._open.n_seeded = int(candidates.size)
        return int(candidates.size)

    # -- what the stage changed, per Gaussian ------------------------------

    def _snapshot(self, params, state) -> Optional[Dict[str, np.ndarray]]:
        """The population as it is now, keyed by id.

        ``means`` stays fp32: positional deltas over one stage are small, and
        fp16 near a scene scale of order 1 resolves only ~1e-3, which is the
        same order as the movement being measured. The rest are fp16, where the
        quantity of interest is a ratio or a probability.
        """
        ids = state.get("ids") if isinstance(state, dict) else None
        if ids is None or "means" not in params:
            return None
        out: Dict[str, np.ndarray] = {
            "ids": ids.detach().cpu().numpy().astype(np.int64),
            "means": params["means"].detach().cpu().numpy().astype(np.float32),
        }
        for key in ("scales", "opacities", "sh0"):
            if key in params:
                out[key] = params[key].detach().cpu().numpy().astype(np.float16)
        return out

    def _diff(self, rec: StageRecord, params, state) -> None:
        """Compare against the stage-start snapshot, matched by id.

        Not by row. Densification appends, reorders and prunes on every
        refinement, so row ``i`` at the start of a stage and row ``i`` at the
        end are unrelated Gaussians; diffing positionally produces a dense,
        plausible, entirely meaningless map of change.
        """
        before = self._opened_at
        now = self._snapshot(params, state)
        if before is None or now is None:
            return
        common, i_before, i_now = np.intersect1d(
            before["ids"], now["ids"], assume_unique=True, return_indices=True
        )
        rec.n_survived = int(common.size)
        rec.n_born = int(now["ids"].size - common.size)
        rec.n_died = int(before["ids"].size - common.size)
        if common.size == 0:
            return

        d_means = np.linalg.norm(
            now["means"][i_now] - before["means"][i_before], axis=1
        ).astype(np.float32)
        cols: Dict[str, np.ndarray] = {"d_means": d_means}
        if "scales" in before and "scales" in now:
            # log space, so a difference of 0.1 is a ~10% size change whatever
            # the absolute size
            cols["d_scale"] = (
                np.abs(
                    now["scales"][i_now].astype(np.float32)
                    - before["scales"][i_before].astype(np.float32)
                )
                .mean(-1)
                .astype(np.float32)
            )
        if "opacities" in before and "opacities" in now:
            # probability space: "went from 0.1 to 0.6" rather than a logit gap
            cols["d_opacity"] = np.abs(
                _sigmoid(now["opacities"][i_now].astype(np.float32))
                - _sigmoid(before["opacities"][i_before].astype(np.float32))
            ).astype(np.float32)
        if "sh0" in before and "sh0" in now:
            cols["d_color"] = np.linalg.norm(
                now["sh0"][i_now].astype(np.float32).reshape(common.size, -1)
                - before["sh0"][i_before].astype(np.float32).reshape(common.size, -1),
                axis=1,
            ).astype(np.float32)

        rec.d_means_mean = float(d_means.mean())
        rec.d_means_p99 = float(np.percentile(d_means, 99))
        if "d_scale" in cols:
            rec.d_scale_mean = float(cols["d_scale"].mean())
        if "d_opacity" in cols:
            rec.d_opacity_mean = float(cols["d_opacity"].mean())
        if "d_color" in cols:
            rec.d_color_mean = float(cols["d_color"].mean())
        rec.deltas = self._write_deltas(rec, common, cols)

    def _write_deltas(
        self, rec: StageRecord, ids: np.ndarray, cols: Dict[str, np.ndarray]
    ) -> str:
        """One row per surviving Gaussian: ``deltas/stage_NNN.npz``.

        fp16 for the deltas themselves, which are display quantities -- this is
        what the viewer colours by, not anything training reads back. ~10 bytes
        per survivor, so ~4 MB a stage at 400k and ~50 MB at Phase 6 scale.
        """
        out = self.out_dir / "deltas"
        out.mkdir(parents=True, exist_ok=True)
        name = f"stage_{rec.stage:03d}.npz"
        np.savez_compressed(
            out / name,
            stage=np.int32(rec.stage),
            step=np.int32(rec.step),
            end_step=np.int32(rec.end_step or 0),
            ids=ids.astype(np.int32),
            **{k: v.astype(np.float16) for k, v in cols.items()},
        )
        return f"deltas/{name}"

    # -- rendering ---------------------------------------------------------

    def _render(self, item: Optional[int]) -> Optional[np.ndarray]:
        if item is None or self.render_fn is None:
            return None
        with torch.no_grad():
            out = self.render_fn(self.trainset[int(item)])
        return out[0].clamp(0, 1).detach().cpu().numpy()

    def _gt(self, item: Optional[int]) -> Optional[np.ndarray]:
        if item is None:
            return None
        return self.trainset[int(item)]["image"].numpy().astype(np.uint8)

    def _write_panel(
        self, rec: StageRecord, gt: np.ndarray, before: np.ndarray, after: np.ndarray
    ) -> str:
        """``[GT | blind guess | after | error before | error after]``.

        The one image that answers "what did this photo teach it": the same view
        the model had to guess at, what it guessed, and what it learned.
        """
        import imageio.v2 as imageio

        out = self.out_dir / "stages"
        out.mkdir(parents=True, exist_ok=True)
        # one shared error scale across both renders; scaled independently, the
        # blind guess and the trained result both fill the ramp and the
        # improvement the pair exists to show is normalised away
        vmax = max(float(abs_error(gt, before).max()), float(abs_error(gt, after).max()))
        panel = strip(
            [
                gt.astype(np.float32) / 255.0,
                np.clip(before, 0, 1),
                np.clip(after, 0, 1),
                error_map(gt, before, vmax=vmax),
                error_map(gt, after, vmax=vmax),
            ]
        )
        name = f"stage_{rec.stage:03d}_img_{rec.added:03d}.png"
        imageio.imwrite(out / name, panel)
        return f"stages/{name}"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
