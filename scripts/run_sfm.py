#!/usr/bin/env python3
"""Run COLMAP SfM on a folder of room photos (Phase 2).

Produces the directory layout gsplat's COLMAP Parser expects, so training can
point straight at the output directory.

Examples
--------
    # validate paths and show the planned layout, without running COLMAP
    python scripts/run_sfm.py --images photos/room-1 --dry-run

    # the real run; writes data/scenes/room-1/
    python scripts/run_sfm.py --images photos/room-1

    # video frames, or any sequential capture
    python scripts/run_sfm.py --images ~/frames --out data/scenes/room-1 --matcher sequential
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splat.sfm import DEFAULT_DOWNSCALES, SfmConfig, format_summary, run_sfm  # noqa: E402


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_out_root() -> Path:
    """Scenes live in the repo's data/ directory, which .gitignore excludes.

    Keeping them next to the code means a scene is found by relative path from
    anywhere in the repo, with no environment to set. DATA_ROOT still overrides
    it, for a scene that outgrows this disk.
    """
    return Path(os.environ.get("DATA_ROOT", repo_root() / "data"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="COLMAP SfM for Room-Gaussian-Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", required=True, type=Path,
                   help="Directory of input photos (read-only).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output scene directory. Default: data/scenes/<images dir name>, "
                        "or $DATA_ROOT/scenes/<images dir name> if DATA_ROOT is set.")
    p.add_argument("--camera-model", default="OPENCV",
                   choices=["OPENCV", "SIMPLE_RADIAL", "RADIAL", "PINHOLE", "FULL_OPENCV"],
                   help="OPENCV suits a phone camera.")
    p.add_argument("--per-image-cameras", action="store_true",
                   help="Give each photo its own intrinsics. Default assumes one camera "
                        "for every photo, which is right for a single phone.")
    p.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"],
                   help="exhaustive suits <=100 unordered photos; sequential suits video frames.")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU for features and matching.")
    p.add_argument("--max-image-size", type=int, default=3200,
                   help="Downscale above this before feature extraction.")
    p.add_argument("--max-num-features", type=int, default=8192)
    p.add_argument("--num-threads", type=int, default=4,
                   help="Threads for feature extraction and matching. Without a CUDA "
                        "build of COLMAP these run on the CPU at roughly 2 GB of RAM "
                        "each, so raise this only if you have the memory to spare.")
    p.add_argument("--downscales", type=int, nargs="*", default=list(DEFAULT_DOWNSCALES),
                   help="Make images_<N>/ PNG copies. gsplat needs these for --factor N.")
    p.add_argument("--keep-exif-orientation", action="store_true",
                   help="Skip re-encoding rotated photos upright.")
    p.add_argument("--undistort", action="store_true",
                   help="Also write an undistorted copy. Not needed: gsplat undistorts itself.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and print the planned layout; runs no COLMAP.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out or (default_out_root() / "scenes" / Path(args.images).resolve().name)

    cfg = SfmConfig(
        image_dir=Path(args.images),
        out=Path(out),
        camera_model=args.camera_model,
        single_camera=not args.per_image_cameras,
        matcher=args.matcher,
        use_gpu=not args.no_gpu,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
        num_threads=args.num_threads,
        downscales=tuple(args.downscales),
        normalize_orientation=not args.keep_exif_orientation,
        undistort=args.undistort,
        dry_run=args.dry_run,
    )

    report = run_sfm(cfg)
    if report.get("problems"):
        return 1  # validation failed; a dry run must still signal this to callers
    if not args.dry_run:
        print(format_summary(report))
        print(f"\nTrain with:  --data_dir {cfg.out} --data_factor {cfg.downscales[0] if cfg.downscales else 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
