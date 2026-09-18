"""Image-count ablation: the analysis, separate from the driver (Phase 5).

The curriculum's test-PSNR curve conflates two things -- every point further
along it has seen more images *and* trained longer. The ablation separates them
by training each condition on a fixed set of N images for the same number of
iterations as every other condition. ``scripts/ablate.py`` launches the runs;
this is the arithmetic over their results, which is where the conclusions come
from and therefore what needs tests.

How many repeats
----------------
Two identical vanilla runs on this scene differ by more than the effect being
looked for. Measured in Phase 3 at n = 12: PSNR 19.576 +/- 0.708 dB, because the
rasterizer's backward accumulates gradients with atomics in nondeterministic
order and a single flipped densification decision diverges the run permanently.
So a difference between two conditions at one run each is not a result, and the
number of repeats has to be chosen before the numbers are seen rather than after.

Two-sample t-test, alpha 0.05 two-sided, 80% power::

    n per condition = 15.7 * sigma^2 / delta^2
    resolvable delta at n = sigma * sqrt(15.7 / n)

On the Phase 3 spread that is **32 runs per condition to resolve 0.5 dB** -- the
number worth knowing before committing a weekend to it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .paths import RunPaths

#: the plan's conditions; "all" resolves against the scene's train-image count
DEFAULT_CONDITIONS = "3,5,10,20,40,all"

#: Phase 3's measured run-to-run PSNR spread on room-1 at 3k steps, n = 12.
#: A prior for planning only; a finished ablation recomputes it from its own runs.
PRIOR_SD_DB = 0.708

#: 2 * (z(1 - alpha/2) + z(1 - beta))^2 for alpha = 0.05 two-sided, power = 0.80
_POWER_K = 2 * (1.959964 + 0.841621) ** 2


def n_for_effect(sd: float, delta: float) -> int:
    """Runs per condition needed to resolve a ``delta`` dB difference."""
    if delta <= 0 or sd < 0:
        return 0
    return int(math.ceil(_POWER_K * sd * sd / (delta * delta)))


def effect_at_n(sd: float, n: int) -> float:
    """Smallest difference resolvable with ``n`` runs per condition."""
    return float("inf") if n <= 0 else sd * math.sqrt(_POWER_K / n)


def conditions_for(spec: str, n_train: int) -> List[int]:
    """Parse a condition spec, drop what the scene cannot run, dedupe.

    A 28-image scene cannot run the plan's N=40, and silently running it *as* 28
    would put the same condition in the table twice under different names. It is
    dropped with a note instead.
    """
    out: List[int] = []
    skipped: List[int] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        value = n_train if part == "all" else int(part)
        if value < 3:
            raise ValueError(f"condition {part!r}: a curriculum needs at least 3 images")
        if value > n_train:
            skipped.append(value)
            continue
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"no runnable conditions in {spec!r} for {n_train} train images")
    return sorted(out), sorted(skipped)  # type: ignore[return-value]


def read_psnr(result: Path) -> Optional[float]:
    """The last held-out eval a run wrote, or None if it never got there."""
    stats = sorted(Path(result).glob("stats/val_step*.json"))
    if not stats:
        return None
    try:
        return float(json.loads(stats[-1].read_text())["psnr"])
    except (json.JSONDecodeError, KeyError):
        return None


def collect(out_dir: Path) -> Dict[int, List[float]]:
    """Scrape every finished run under ``out_dir`` into ``{N: [psnr, ...]}``."""
    got: Dict[int, List[float]] = {}
    for result in sorted(Path(out_dir).glob("n*_r*")):
        try:
            n_images = int(result.name.split("_")[0][1:])
        except ValueError:
            continue
        psnr = read_psnr(result)
        if psnr is not None:
            got.setdefault(n_images, []).append(psnr)
    return got


def settings_of(out_dir: Path) -> Dict[str, int]:
    """``max_steps``/``data_factor`` as the runs were actually trained.

    From a run's own ``cfg.yml``, not from the invocation's flags: on a
    report-only pass those flags are whatever the defaults happen to be, and
    labelling a plot with them states a step count that never ran.
    """
    from .timeline import load_cfg

    for result in sorted(Path(out_dir).glob("n*_r*")):
        cfg = load_cfg(RunPaths(result))
        if cfg:
            return {
                "max_steps": int(cfg.get("max_steps", 0)),
                "data_factor": int(cfg.get("data_factor", 0)),
            }
    return {}


def summarize(got: Dict[int, List[float]]) -> Dict:
    """Per-condition stats plus the pooled within-condition sd."""
    rows = []
    for n_images in sorted(got):
        values = np.asarray(got[n_images], dtype=np.float64)
        rows.append(
            {
                "n_images": int(n_images),
                "runs": int(values.size),
                "psnr_mean": float(values.mean()),
                # ddof=1: a sample sd, and at n=2 the difference from ddof=0 is 40%
                "psnr_sd": float(values.std(ddof=1)) if values.size > 1 else None,
                "psnr": [float(v) for v in values],
            }
        )
    spreads = [(r["runs"] - 1, r["psnr_sd"] ** 2) for r in rows if r["psnr_sd"] is not None]
    pooled = (
        float(np.sqrt(sum(w * v for w, v in spreads) / sum(w for w, _ in spreads)))
        if spreads
        else None
    )
    return {"conditions": rows, "pooled_sd": pooled}


def power_note(sd: float, repeats: int, source: str) -> str:
    """The lines that say what a repeat count can and cannot show."""
    lines = [
        f"run-to-run sd: {sd:.3f} dB ({source})",
        f"at {repeats} run(s) per condition, the smallest resolvable difference"
        f" is {effect_at_n(sd, repeats):.2f} dB",
    ]
    lines += [
        f"  to resolve {delta:>4} dB needs {n_for_effect(sd, delta):>4} runs per condition"
        for delta in (0.25, 0.5, 1.0, 2.0)
    ]
    return "\n".join("  " + line for line in lines)


def format_table(summary: Dict) -> str:
    rows = summary["conditions"]
    out = [f"{'N':>5} {'runs':>5} {'PSNR':>8} {'sd':>7}   values"]
    for r in rows:
        sd = "  -  " if r["psnr_sd"] is None else f"{r['psnr_sd']:.3f}"
        out.append(
            f"{r['n_images']:>5} {r['runs']:>5} {r['psnr_mean']:>8.3f} {sd:>7}   "
            + " ".join(f"{v:.2f}" for v in r["psnr"])
        )
    return "\n".join(out)


def plot(summary: Dict, out: Path, title: str = "") -> Optional[Path]:
    """Write ablation.png: held-out PSNR against a fixed image count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["conditions"]
    if not rows:
        return None
    out = Path(out)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [r["n_images"] for r in rows]
    ys = [r["psnr_mean"] for r in rows]
    es = [r["psnr_sd"] or 0.0 for r in rows]

    # individual runs first, nudged right so the mean marker does not hide them:
    # the spread between repeats is what decides whether any difference in the
    # means is worth reporting, so it has to be visible
    span = (max(xs) - min(xs)) or 1
    for r in rows:
        ax.scatter([r["n_images"] + span * 0.012] * r["runs"], r["psnr"], s=16,
                   color="#868E96", alpha=0.8, zorder=3)
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#1971C2", lw=1.8, ms=7,
                capsize=4, zorder=2, label="mean +/- sd")
    ax.scatter([], [], s=16, color="#868E96", label="individual runs")
    ax.set_xlabel("training images (fixed for the whole run)")
    ax.set_ylabel("held-out PSNR (dB)")
    ax.set_title(title or "images vs quality at equal iteration counts")
    ax.grid(alpha=0.25)
    sd = summary.get("pooled_sd")
    if sd:
        n = min(r["runs"] for r in rows)
        ax.text(0.99, 0.02,
                f"pooled sd {sd:.3f} dB; resolvable at n={n}: {effect_at_n(sd, n):.2f} dB",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#868E96")
    ax.legend(loc="upper left", frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
