#!/usr/bin/env python3
"""Open the playback viewer on a finished training run (Phase 4).

Scrub the run's snapshots from the SfM point cloud to the final model, with the
event log wired into the colours so clones, splits and prunes are visible as
history rather than as a grey cloud.

Needs a run trained with ``--snapshot_every N``; the origin, age and lineage
modes additionally need ``--lineage``. Both are off by default, so a vanilla
baseline has neither and this will say so rather than guess.

Examples
--------
    # a run under data/runs/
    python scripts/view.py --run dev-room-4k

    # any directory, on another port
    python scripts/view.py --run /mnt/big/runs/room-30k --port 8081

    # list what the run has, and exit
    python scripts/view.py --run dev-room-4k --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splat.paths import RunPaths, run_dir  # noqa: E402


def resolve(name: str) -> RunPaths:
    """A run name under data/runs/, or a path to a run directory."""
    path = Path(name)
    if path.exists() and (path / "snapshots").exists():
        return RunPaths(path.resolve())
    run = run_dir(name)
    if run.root.exists():
        return run
    if path.exists():
        return RunPaths(path.resolve())  # exists but has no snapshots; report below
    raise SystemExit(f"no such run: {name} (looked in {run.root} and ./{name})")


def check(run: RunPaths) -> int:
    from splat.timeline import Timeline

    tline = Timeline(run)
    steps = tline.steps
    print(f"run:       {run.root}")
    print(f"snapshots: {len(steps)}" + (f"  steps {steps[0]}..{steps[-1]}" if steps else "  (none)"))
    print(f"events:    {'yes' if run.events.exists() else 'no'}" + (
        f"  {len(tline.lineage):,} ids, resets at {list(map(int, tline.resets))}"
        if tline.lineage else "  (no --lineage, so no origin/age/lineage modes)"
    ))
    print(f"config:    {'yes' if tline.cfg else 'no'}")
    print(f"schedule:  {tline.schedule}")
    if steps:
        print()
        print(tline.narrate(steps[-1], tline.frame(steps[-1])["ids"]))
    return 0 if steps else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name under data/runs/, or a path")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--scene",
        default=None,
        help="COLMAP scene for the frustums; defaults to the run's recorded data_dir",
    )
    p.add_argument("--frustum-scale", type=float, default=0.12)
    p.add_argument("--check", action="store_true", help="report what the run has, then exit")
    args = p.parse_args()

    run = resolve(args.run)
    if args.check:
        return check(run)

    from splat.viewer import serve

    serve(
        run,
        port=args.port,
        device=args.device,
        scene_dir=Path(args.scene) if args.scene else None,
        frustum_scale=args.frustum_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
