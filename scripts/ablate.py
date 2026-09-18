#!/usr/bin/env python3
"""Ablate the number of training images at a fixed iteration count (Phase 5).

Each condition trains on a fixed set of N images -- the first N of the same
curriculum ordering, so the conditions are nested -- for the same `--max-steps`
as every other condition. That is what separates "more images" from "more
training time", which the curriculum's own PSNR curve cannot do.

`--repeats` is required and has no default. Two identical runs on this scene
differ by more than the effect being looked for (Phase 3: +/- 0.708 dB at
n = 12), so the repeat count has to be chosen before the numbers are seen.
`splat/ablation.py` holds the arithmetic and `--plan` prints it.

Examples
--------
    # what would this cost, and what could it show?
    python scripts/ablate.py --repeats 5 --plan

    # run it
    python scripts/ablate.py --repeats 5 --max-steps 7000 --data-factor 2 \
        --out ablate-room-1

    # re-report over runs that already exist, training nothing
    python scripts/ablate.py --out ablate-room-1 --report-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splat.ablation import (  # noqa: E402
    DEFAULT_CONDITIONS,
    PRIOR_SD_DB,
    collect,
    conditions_for,
    format_table,
    plot,
    power_note,
    read_psnr,
    settings_of,
    summarize,
)
from splat.paths import data_root, repo_root  # noqa: E402

TRAINER = repo_root() / "third_party" / "gsplat_examples" / "simple_trainer.py"


def train_image_count(scene: Path, factor: int, test_every: int = 8) -> int:
    """How many train images the scene has, via gsplat's own split."""
    sys.path.insert(0, str(repo_root() / "third_party" / "gsplat_examples"))
    from datasets.colmap import Dataset, Parser  # type: ignore

    parser = Parser(data_dir=str(scene), factor=factor, normalize=True, test_every=test_every)
    return len(Dataset(parser, "train"))


def run_one(
    scene: Path,
    out_dir: Path,
    n_images: int,
    repeat: int,
    max_steps: int,
    data_factor: int,
    extra: List[str],
) -> Optional[float]:
    """Train one condition once. Returns its final held-out PSNR."""
    result = out_dir / f"n{n_images:03d}_r{repeat}"
    cmd = [
        sys.executable, str(TRAINER), "default",
        "--data_dir", str(scene),
        "--data_factor", str(data_factor),
        "--result_dir", str(result),
        "--max_steps", str(max_steps),
        "--eval_steps", str(max_steps),
        "--save_steps", str(max_steps),
        "--ply_steps", str(max_steps),
        "--lineage",
        "--curriculum",
        "--curriculum_init_images", str(n_images),
        "--curriculum_max_images", str(n_images),
        # a fixed-N run has no stage boundaries, so there is nothing extra to
        # eval, and snapshots across dozens of runs are not worth the disk
        "--no-curriculum_eval_stages",
        "--snapshot_every", "0",
        "--disable_viewer",
        "--disable_video",
        *extra,
    ]
    tic = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    took = time.time() - tic
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
        print(f"[ablate] N={n_images} r={repeat} FAILED in {took:.0f}s\n{tail}")
        return None
    psnr = read_psnr(result)
    shown = "no eval" if psnr is None else f"{psnr:.3f}"
    print(f"[ablate] N={n_images:>3} r={repeat}  PSNR {shown}  ({took:.0f}s)")
    return psnr


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", default="data/scenes/room-1")
    p.add_argument("--out", default="ablate-room-1", help="run name under data/runs/")
    p.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    p.add_argument("--repeats", type=int, required=True,
                   help="runs per condition; deliberately has no default")
    p.add_argument("--max-steps", type=int, default=7000)
    p.add_argument("--data-factor", type=int, default=2)
    p.add_argument("--plan", action="store_true",
                   help="print the plan and the power analysis, launch nothing")
    p.add_argument("--report-only", action="store_true",
                   help="summarize the runs that already exist, train nothing")
    p.add_argument("--trainer-arg", action="append", default=[],
                   help="extra argument passed straight to simple_trainer (repeatable)")
    args = p.parse_args()

    out_dir = data_root() / "runs" / args.out
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = repo_root() / scene

    if args.report_only:
        got = collect(out_dir)
        if not got:
            raise SystemExit(f"no finished runs under {out_dir}")
    else:
        if not scene.exists():
            raise SystemExit(f"no such scene: {scene}")
        n_train = train_image_count(scene, args.data_factor)
        todo, skipped = conditions_for(args.conditions, n_train)
        for value in skipped:
            print(f"[ablate] skipping N={value}: the scene has {n_train} train images")
        print(f"[ablate] scene {scene.name}: {n_train} train images")
        print(f"[ablate] conditions {todo} x {args.repeats} repeats ="
              f" {len(todo) * args.repeats} runs of {args.max_steps} steps"
              f" at factor {args.data_factor}")
        print(power_note(PRIOR_SD_DB, args.repeats, "Phase 3 prior, room-1 @ 3k steps"))
        if args.plan:
            print("[ablate] --plan: nothing launched")
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        for n_images in todo:
            for repeat in range(args.repeats):
                run_one(scene, out_dir, n_images, repeat, args.max_steps,
                        args.data_factor, args.trainer_arg)
        got = collect(out_dir)

    summary = summarize(got)
    summary.update({"max_steps": args.max_steps, "data_factor": args.data_factor})
    summary.update(settings_of(out_dir))  # the runs are the authority, not the flags
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2))

    print()
    print(format_table(summary))
    sd = summary["pooled_sd"]
    if sd:
        print()
        print(power_note(sd, min(r["runs"] for r in summary["conditions"]), "measured here"))
    else:
        print("\n  One run per condition: no within-condition spread was measured, so"
              "\n  no difference in the table above can be called real.")
    label = f"{args.out} - {summary['max_steps']} steps"
    if summary.get("data_factor"):
        label += f", factor {summary['data_factor']}"
    path = plot(summary, out_dir / "ablation.png", label)
    print(f"\n[ablate] wrote {out_dir / 'ablation.json'}" + (f" and {path}" if path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
