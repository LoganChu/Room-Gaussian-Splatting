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
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
    return out
