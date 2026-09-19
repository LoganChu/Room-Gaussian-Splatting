"""Plots from a training run.

Two reports, from two independent sources:

- ``population.png`` needs ``events.parquet``, so only an instrumented run has
  it. It is the headline: Gaussian count against iteration with clone, split and
  prune broken out, which is where densification becomes visible -- the flat
  warm-up to 500, the growth phase, the sawtooth of each opacity reset, and the
  hard freeze at refine_stop_iter.
- ``training.png`` reads the TensorBoard scalars that ``simple_trainer`` writes
  for **every** run, instrumented or not. So a vanilla baseline can be reported
  and compared against without re-running it with hooks.

A curriculum run adds a third, ``curriculum.png``, from ``stages.json`` plus the
per-stage ``stats/val_step*.json`` the trainer already writes. It also owns the
small image-composition helpers -- ``psnr``, ``error_map``,
``strip`` -- because the viewer's camera comparison and the curriculum's
per-stage panels both need them, and two implementations of PSNR in one repo is
two numbers that can disagree in the same report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .events import read_events
from .paths import RunPaths

BIRTH_KINDS = ("sfm", "seed", "clone", "split")
COLORS = {
    "sfm": "#4C6EF5",
    "seed": "#12B886",
    "clone": "#F59F00",
    "split": "#E03131",
    "death": "#868E96",
}


#: TensorBoard scalars simple_trainer writes; all optional
TB_LOSS = "train/loss"
TB_COUNT = "train/num_GS"
TB_PSNR = "val/psnr"


def load_scalars(tb_dir: Path) -> Dict[str, tuple]:
    """Read TensorBoard scalars as {tag: (steps, values)}.

    Returns {} rather than raising when there is nothing to read, so a report
    over a run that predates tensorboard still produces its other plots.
    """
    from tensorboard.backend.event_processing import event_accumulator

    tb_dir = Path(tb_dir)
    if not tb_dir.exists():
        return {}
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        pts = ea.Scalars(tag)
        out[tag] = (
            np.array([p.step for p in pts]),
            np.array([p.value for p in pts], dtype=np.float64),
        )
    return out


def load(events_path: Path) -> Dict[str, np.ndarray]:
    t = read_events(events_path)
    return {
        "step": t.column("step").to_numpy(zero_copy_only=False),
        "kind": np.asarray(t.column("kind").to_pylist()),
        "gid": t.column("gid").to_numpy(zero_copy_only=False),
        "parent": t.column("parent").to_numpy(zero_copy_only=False),
    }


def population(ev: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-step births, deaths and the running population size."""
    is_birth = np.isin(ev["kind"], BIRTH_KINDS)
    is_death = ev["kind"] == "death"
    steps = np.unique(ev["step"][is_birth | is_death])
    births = np.array([np.sum(is_birth & (ev["step"] == s)) for s in steps])
    deaths = np.array([np.sum(is_death & (ev["step"] == s)) for s in steps])
    return {
        "step": steps,
        "births": births,
        "deaths": deaths,
        "count": np.cumsum(births - deaths),
    }


def per_kind(ev: Dict[str, np.ndarray], kind: str):
    sel = ev["kind"] == kind
    steps = np.unique(ev["step"][sel])
    if steps.size == 0:
        return steps, steps
    return steps, np.array([np.sum(sel & (ev["step"] == s)) for s in steps])


def plot(events_path: Path, out_dir: Path, title: str = "") -> Path:
    """Write population.png. Returns its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = load(events_path)
    pop = population(ev)
    resets = np.unique(ev["step"][ev["kind"] == "reset"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1]
    )

    ax.plot(pop["step"], pop["count"], color="#1F2937", lw=1.8, label="population")
    for s in resets:
        ax.axvline(s, color="#ADB5BD", lw=1, ls="--", zorder=0)
    if resets.size:
        # label once, not once per line
        ax.axvline(resets[0], color="#ADB5BD", lw=1, ls="--", label="opacity reset")
    ax.set_ylabel("Gaussians")
    ax.set_title(title or events_path.parent.name)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    for kind in ("clone", "split", "death"):
        steps, counts = per_kind(ev, kind)
        if steps.size:
            bx.plot(steps, counts, lw=1.2, color=COLORS[kind], label=kind)
    bx.set_xlabel("iteration")
    bx.set_ylabel("events / refinement")
    bx.legend(loc="upper right", frameon=False)
    bx.grid(alpha=0.25)

    path = out_dir / "population.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_training(
    tb_dir: Path, out_dir: Path, title: str = "", resets: Optional[np.ndarray] = None
) -> Optional[Path]:
    """Write training.png: loss, population and eval PSNR against iteration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = load_scalars(tb_dir)
    if TB_LOSS not in sc:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    steps, loss = sc[TB_LOSS]
    # log scale: loss falls by more than an order of magnitude, and the late
    # refinement phase is invisible on a linear axis
    ax.semilogy(steps, loss, color="#1F2937", lw=1.4, label="train loss")
    ax.set_ylabel("loss (log)")
    ax.set_title(title or Path(tb_dir).parent.name)
    ax.grid(alpha=0.25, which="both")

    if TB_PSNR in sc:
        ps, pv = sc[TB_PSNR]
        px = ax.twinx()
        px.plot(ps, pv, "o--", color="#1971C2", lw=1.2, ms=5, label="val PSNR")
        px.set_ylabel("val PSNR (dB)", color="#1971C2")
        px.tick_params(axis="y", colors="#1971C2")
        for x, y in zip(ps, pv):
            px.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(-4, 7), color="#1971C2", fontsize=9, ha="right")
    ax.legend(loc="upper right", frameon=False)

    if TB_COUNT in sc:
        cs, cv = sc[TB_COUNT]
        bx.plot(cs, cv, color="#E03131", lw=1.6, label="Gaussians")
        bx.set_ylabel("Gaussians")
    if resets is not None:
        for s_ in resets:
            bx.axvline(s_, color="#ADB5BD", lw=1, ls="--", zorder=0)
    bx.set_xlabel("iteration")
    bx.legend(loc="upper left", frameon=False)
    bx.grid(alpha=0.25)

    path = out_dir / "training.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summarize(events_path: Path) -> Dict[str, int]:
    ev = load(events_path)
    out = {k: int(np.sum(ev["kind"] == k)) for k in ("sfm", "seed", "clone", "split", "death", "reset")}
    out["final"] = int(
        np.sum(np.isin(ev["kind"], BIRTH_KINDS)) - np.sum(ev["kind"] == "death")
    )
    return out


def load_stages(run: RunPaths) -> Optional[Dict]:
    """``stages.json``, or None for a run that was not a curriculum run."""
    path = run.root / "stages.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def load_val_metrics(run: RunPaths) -> Dict[int, Dict[str, float]]:
    """``{step: metrics}`` from every ``stats/val_step*.json`` the run wrote."""
    import json

    out: Dict[int, Dict[str, float]] = {}
    stats = run.root / "stats"
    if not stats.exists():
        return out
    for path in sorted(stats.glob("val_step*.json")):
        digits = "".join(c for c in path.stem if c.isdigit())
        if digits:
            out[int(digits)] = json.loads(path.read_text())
    return out


def join_stage_metrics(stages: Dict, val: Dict[int, Dict[str, float]]) -> list:
    """Attach each stage's held-out metrics to its record.

    Matched by "the latest eval at or before this stage ended" rather than by an
    exact step. The trainer evals at ``step == eval_step - 1``, and the last
    stage is closed at ``max_steps - 1`` rather than at a boundary, so the two
    conventions differ by one at exactly one stage. Taking the most recent eval
    is right under both and survives a change of cadence.
    """
    steps = sorted(val)
    rows = []
    for rec in stages.get("stages", []):
        row = dict(rec)
        end = rec.get("end_step")
        if end is not None and steps:
            eligible = [s for s in steps if s <= end]
            if eligible:
                row["eval_step"] = eligible[-1]
                row.update(
                    {f"test_{k}": v for k, v in val[eligible[-1]].items()}
                )
        rows.append(row)
    return rows


#: an eval this soon after an opacity reset is measuring the dip, not the model.
#: Two refinement intervals: the reset drops every opacity, and it takes a
#: couple of prune/densify passes for the population to recover.
RESET_SHADOW_STEPS = 200


def reset_steps(run: RunPaths) -> np.ndarray:
    """Opacity-reset steps from the event log, or empty if there is none."""
    if not run.events.exists():
        return np.empty(0, dtype=np.int64)
    ev = load(run.events)
    return np.unique(ev["step"][ev["kind"] == "reset"])


def in_reset_shadow(step: int, resets: np.ndarray, window: int = RESET_SHADOW_STEPS) -> bool:
    """Is ``step`` close enough after a reset that its PSNR is the dip?

    Worth flagging rather than smoothing away. Measured on `dev-groups`: an eval
    2 steps after the reset at 3000 read **5.74 dB** against 16.87 before and
    17.14 after -- an 11 dB hole that is entirely the documented reset dip, and
    would read as a catastrophic regression to anyone who did not know.
    """
    return bool(np.any((resets <= step) & (step < resets + window)))


def plot_curriculum(run: RunPaths, out_dir: Path, title: str = "") -> Optional[Path]:
    """Write curriculum.png: the charts Phase 5's check asks for."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = load_stages(run)
    if not stages:
        return None
    rows = join_stage_metrics(stages, load_val_metrics(run))
    if not rows:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = stages.get("mode", "incremental")
    fig, (ax, bx, cx, dx) = plt.subplots(
        4, 1, figsize=(12, 15), height_ratios=[2, 2, 1.3, 1.3]
    )

    # --- test PSNR. In groups mode every round revisits all the images, so the
    # image count stops being an x axis after round 0 and the step is.
    have = [r for r in rows if r.get("test_psnr") is not None]
    # The consolidation stage adds no image, so on an image-count axis it would
    # sit on top of the last stage and read as a vertical jump. It is drawn as
    # its own marker instead, and kept off the curve.
    closing = [r for r in have if r.get("consolidate")]
    have = [r for r in have if not r.get("consolidate")]
    if have:
        by_step = mode == "groups"
        xs = [r["step"] if by_step else r["n_images"] for r in have]
        resets = reset_steps(run)
        shadowed = [
            in_reset_shadow(r.get("eval_step", -1), resets) for r in have
        ]
        ax.plot(xs, [r["test_psnr"] for r in have],
                "o-", color="#1971C2", lw=1.8, ms=5, label="test PSNR")
        if any(shadowed):
            ax.scatter(
                [x for x, bad in zip(xs, shadowed) if bad],
                [r["test_psnr"] for r, bad in zip(have, shadowed) if bad],
                s=120, facecolors="none", edgecolors="#E03131", lw=1.8, zorder=5,
                label=f"within {RESET_SHADOW_STEPS} steps of an opacity reset",
            )
        for r in closing:
            x = r["end_step"] if by_step and r.get("end_step") is not None else (
                r["step"] if by_step else r["n_images"])
            ax.plot([x], [r["test_psnr"]], "*", color="#1971C2", ms=16, zorder=6,
                    label=f"after consolidation (all {r['n_images']} images from"
                          f" step {r['step']:,})")
            if by_step:
                ax.axvspan(r["step"], x, color="#1971C2", alpha=0.06, zorder=0)
        ax.set_ylabel("held-out PSNR (dB)", color="#1971C2")
        ax.tick_params(axis="y", colors="#1971C2")
        gx = ax.twinx()
        gx.plot(xs, [r.get("test_num_GS") or r["n_gaussians_after"] for r in have],
                "s--", color="#E03131", lw=1.2, ms=4, label="Gaussians")
        gx.set_ylabel("Gaussians", color="#E03131")
        gx.tick_params(axis="y", colors="#E03131")
        gx.set_yscale("log")
        ax.set_xlabel("iteration" if by_step else "training images")
        ax.set_title(title or f"{run.root.name} - curriculum ({mode})")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=False)
        if by_step:
            rounds = sorted({r["round"] for r in rows if r.get("round") is not None})
            for rnd in rounds[1:]:
                first = min(r["step"] for r in rows if r.get("round") == rnd)
                ax.axvline(first, color="#ADB5BD", lw=1, ls="--", zorder=0)
            if len(rounds) > 1:
                ax.text(0.01, 0.97, f"dashed: start of each of {len(rounds)} round-robin passes",
                        transform=ax.transAxes, va="top", fontsize=8, color="#868E96")
        elif len(xs) > 1:
            # The curve conflates "more images" with "more training time": every
            # point further right has also trained longer. Only the ablation,
            # which holds iterations fixed, separates the two.
            ax.text(0.01, 0.97,
                    "more images AND more steps: see the ablation to separate them",
                    transform=ax.transAxes, va="top", fontsize=8, color="#868E96")

    # --- what each photo taught it
    gains = [r for r in rows if r.get("psnr_before") is not None]
    if gains:
        idx = np.arange(len(gains))
        before = np.array([r["psnr_before"] for r in gains])
        after = np.array([r["psnr_after"] for r in gains])
        bx.bar(idx, before, color="#ADB5BD", label="blind guess (never seen)")
        bx.bar(idx, after - before, bottom=before, color="#12B886",
               label="gained during its own stage")
        bx.set_xticks(idx)
        bx.set_xticklabels(
            [str(r["image"] or "").replace("IMG_", "").replace(".jpeg", "") for r in gains],
            rotation=90, fontsize=7,
        )
        bx.set_ylabel("PSNR on the added view (dB)")
        bx.grid(alpha=0.25, axis="y")
        bx.legend(loc="upper left", frameon=False)
        mean_gain = float((after - before).mean())
        bx.set_title(f"per-photo improvement - mean +{mean_gain:.2f} dB over {len(gains)} stages")
        lengths = [
            (r["end_step"] - r["step"]) for r in gains
            if r.get("end_step") is not None
        ]
        # A stage that trained longer improved more for that reason alone -- but
        # even division leaves a remainder, so the default schedule differs by a
        # step or two. Warning on that would teach the reader to ignore the
        # warning; 25% is where a length difference can carry a bar.
        if lengths and max(lengths) > 1.25 * min(lengths):
            bx.text(0.99, 0.97,
                    f"unequal stage lengths ({min(lengths)}-{max(lengths)} steps):"
                    " bars are not comparable",
                    transform=bx.transAxes, va="top", ha="right",
                    fontsize=8, color="#E03131")

    # --- where the population came from
    seeded = np.array([r.get("n_seeded", 0) for r in rows])
    rejected = np.array([r.get("n_rejected_near", 0) for r in rows])
    unlocked = np.array([r.get("n_points_unlocked", 0) for r in rows])
    idx = np.arange(len(rows))
    cx.bar(idx, seeded, color=COLORS["seed"], label="seeded")
    cx.bar(idx, rejected, bottom=seeded, color="#868E96",
           label="rejected: already covered")
    cx.plot(idx, unlocked, "k.-", lw=0.8, ms=3, label="points unlocked")
    cx.set_xlabel("stage")
    cx.set_ylabel("SfM points")
    cx.grid(alpha=0.25, axis="y")
    cx.legend(loc="upper right", frameon=False, fontsize=8)

    # --- per-image exposure: how much to trust the panel above it
    exposure = {int(k): v for k, v in (stages.get("exposure") or {}).items()}
    seen = [int(i) for i in (stages.get("images") or sorted(exposure))]
    if exposure and seen:
        values = [exposure.get(i, 0.0) for i in seen]
        spread = stages.get("exposure_spread")
        fair = spread is not None and spread < 1.01
        dx.bar(range(len(seen)), values,
               color=COLORS["seed"] if fair else "#E03131")
        dx.set_xlabel("image, in curriculum order")
        dx.set_ylabel("expected steps")
        dx.grid(alpha=0.25, axis="y")
        note = (
            "equal by construction -- per-photo gains above are comparable"
            if fair else
            f"{spread:.0f}x spread -- per-photo gains above are NOT comparable"
            " across images; use --curriculum_mode groups"
        )
        dx.set_title(f"per-image exposure: {note}", fontsize=10,
                     color="#2B8A3E" if fair else "#C92A2A")

    path = out_dir / "curriculum.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def report(run: RunPaths, title: str = "") -> Dict[str, Path]:
    """Write every plot the run has the data for. Returns {name: path}."""
    out: Dict[str, Path] = {}
    resets = None
    if run.events.exists():
        ev = load(run.events)
        resets = np.unique(ev["step"][ev["kind"] == "reset"])
        out["population"] = plot(run.events, run.report, title)
    training = plot_training(run.root / "tb", run.report, title, resets=resets)
    if training is not None:
        out["training"] = training
    curriculum = plot_curriculum(run, run.report, title)
    if curriculum is not None:
        out["curriculum"] = curriculum
    return out


# --------------------------------------------------------------------------
# image panels and metrics, shared by the viewer and the curriculum
# --------------------------------------------------------------------------


def psnr(gt: np.ndarray, render: np.ndarray) -> float:
    """PSNR of a float render in 0..1 against a uint8 ground truth."""
    gt_f = gt.astype(np.float32) / 255.0
    mse = float(np.mean((gt_f - np.clip(render, 0.0, 1.0)) ** 2))
    return float("inf") if mse == 0 else -10.0 * float(np.log10(mse))


def abs_error(gt: np.ndarray, render: np.ndarray) -> np.ndarray:
    """Mean absolute per-pixel error, as a float map."""
    return np.abs(gt.astype(np.float32) / 255.0 - np.clip(render, 0.0, 1.0)).mean(-1)


def error_map(
    gt: np.ndarray,
    render: np.ndarray,
    vmax: Optional[float] = None,
    colormap: str = "turbo",
) -> np.ndarray:
    """Colour-mapped absolute error, normalised to ``vmax`` or to its own max.

    Pass an explicit ``vmax`` whenever two error maps will be looked at side by
    side -- a before and an after, say. Scaled independently, both fill the
    colour ramp and the improvement between them becomes invisible, which is the
    one thing the pair exists to show. Left to itself it self-scales, because a
    converged render's absolute error is small everywhere and against a fixed
    1.0 the map is a black rectangle.
    """
    from matplotlib import colormaps

    err = abs_error(gt, render)
    scale = float(vmax) if vmax is not None else float(err.max())
    return colormaps[colormap](err / max(scale, 1e-6))[..., :3]


def strip(images: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate float images in 0..1 side by side into one uint8 panel."""
    return (np.concatenate(list(images), axis=1) * 255).clip(0, 255).astype(np.uint8)
