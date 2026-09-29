#!/usr/bin/env python3
"""Export a PLY from a trainer checkpoint, for runs trained without --save_ply.

Writes the same file the trainer would have: gsplat's own ``export_splats``
over the checkpoint's ``means/scales/quats/opacities/sh0/shN``, so it opens in
SuperSplat or ``simple_viewer.py --ply`` like any trainer-written PLY.

Usage
-----
    # every checkpoint of a run -> <run>/ply/point_cloud_<step>.ply
    python scripts/export_ply.py --run baseline-room1-30k

    # the ablation: every run under it
    python scripts/export_ply.py --run ablate-room-1/n003_r0 ablate-room-1/n028_r0

Runs trained with ``--app_opt`` bake appearance into the colours at export
time, which needs the appearance module; those are refused rather than
exported with the wrong colours.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splat.paths import data_root  # noqa: E402

KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def run_path(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path.resolve()
    return data_root() / "runs" / name


def export(ckpt: Path, out: Path) -> int:
    import torch
    from gsplat import export_splats

    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "app_module" in data:
        raise SystemExit(f"{ckpt}: trained with --app_opt; export it from the trainer instead")
    splats = data["splats"]
    missing = [k for k in KEYS if k not in splats]
    if missing:
        raise SystemExit(f"{ckpt}: checkpoint has no {missing}")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_splats(**{k: splats[k] for k in KEYS}, format="ply", save_to=str(out))
    return int(splats["means"].shape[0])


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", nargs="+", required=True,
                   help="run name(s) under data/runs/, or run directories")
    p.add_argument("--overwrite", action="store_true",
                   help="replace a PLY that already exists")
    args = p.parse_args()

    for name in args.run:
        run = run_path(name)
        ckpts = sorted((run / "ckpts").glob("ckpt_*_rank0.pt"))
        if not ckpts:
            print(f"[export] {name}: no checkpoints, skipped")
            continue
        for ckpt in ckpts:
            step = re.match(r"ckpt_(\d+)_rank0", ckpt.stem).group(1)
            out = run / "ply" / f"point_cloud_{step}.ply"
            if out.exists() and not args.overwrite:
                print(f"[export] {out} exists, skipped (--overwrite to replace)")
                continue
            n = export(ckpt, out)
            print(f"[export] {out}  {n:,} Gaussians, {out.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
