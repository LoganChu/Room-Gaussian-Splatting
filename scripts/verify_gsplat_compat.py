#!/usr/bin/env python3
"""Check that a scene from run_sfm.py is readable by gsplat's COLMAP Parser.

pycolmap 4.2 writes the newer COLMAP model format, with rigs.bin and frames.bin
alongside cameras/images/points3D. gsplat's ``examples/datasets/colmap.py``
Parser was not necessarily written against that format, and a mismatch would
only surface at the start of a long training run. This replays every attribute
access the Parser makes, against the real scene, in seconds.

The helper functions are lifted out of gsplat's own source by AST rather than
re-implemented, so the check tracks upstream instead of drifting from it. They
are exec'd individually because importing colmap.py pulls in cv2 and torch.

    python scripts/verify_gsplat_compat.py --gsplat ~/src/gsplat

Re-run after any gsplat version bump. Exit status is 0 only if every check
passes at every downscale factor.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import numpy as np

HELPERS = {"_as_dict", "_camera_model_name", "_camera_distortion", "_image_w2c", "_get_rel_paths"}


def load_upstream(gsplat_dir: Path) -> dict:
    """exec gsplat's Parser helpers and normalize.py into one namespace."""
    ns: dict = {"np": np, "os": os, "Any": object, "Dict": dict, "List": list}
    sources = [
        (gsplat_dir / "examples/datasets/colmap.py", HELPERS),
        (gsplat_dir / "examples/datasets/normalize.py", None),  # all of it
    ]
    found = set()
    for path, wanted in sources:
        if not path.exists():
            raise SystemExit(f"not a gsplat checkout: {path} missing")
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and (wanted is None or node.name in wanted):
                exec(compile(ast.Module([node], []), str(path), "exec"), ns)
                found.add(node.name)
    missing = HELPERS - found
    if missing:
        raise SystemExit(
            f"gsplat's colmap.py no longer defines {sorted(missing)}. The Parser has been "
            "restructured upstream; update this script against the new source."
        )
    return ns


class Checks:
    """Runs named checks, printing each, and remembering the failures."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, label, fn):
        try:
            print(f"  PASS  {label}: {fn()}")
        except Exception as exc:  # noqa: BLE001 - any failure is a real finding
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
            self.failures.append(label)


def verify(scene: Path, gsplat_dir: Path, factor: int, ns: dict, check: Checks) -> None:
    import pycolmap
    from PIL import Image

    as_dict, distortion = ns["_as_dict"], ns["_camera_distortion"]
    image_w2c, rel_paths = ns["_image_w2c"], ns["_get_rel_paths"]

    print(f"\n[factor {factor}] pycolmap {pycolmap.__version__}, scene {scene}")

    # The Parser's own sparse/0-then-sparse fallback.
    colmap_dir = scene / "sparse" / "0"
    if not colmap_dir.exists():
        colmap_dir = scene / "sparse"
    check(
        "Reconstruction(sparse/0) [rig/frame format]",
        lambda: f"{len(pycolmap.Reconstruction(str(colmap_dir)).images)} images",
    )
    rec = pycolmap.Reconstruction(str(colmap_dir))
    cameras, images = as_dict(rec.cameras), as_dict(rec.images)
    imdata = {int(i): images[int(i)] for i in rec.reg_image_ids()}
    cam = next(iter(cameras.values()))

    check("reg_image_ids()", lambda: len(imdata))
    check("im.cam_from_world.matrix() -> w2c", lambda: np.stack([image_w2c(i) for i in imdata.values()]).shape)
    check("camtoworlds = inv(w2c)", lambda: np.linalg.inv(np.stack([image_w2c(i) for i in imdata.values()])).shape)
    check("cam.calibration_matrix()", lambda: np.asarray(cam.calibration_matrix()).round(1).tolist())
    check("_camera_distortion(cam)", lambda: (distortion(cam)[0].tolist(), distortion(cam)[1]))
    check("imsize // factor", lambda: (cam.width // factor, cam.height // factor))

    points3D = as_dict(rec.points3D)
    ids = sorted(points3D)
    check("points3D .xyz/.error/.color", lambda: (
        np.array([points3D[i].xyz for i in ids], dtype=np.float32).shape,
        round(float(np.mean([points3D[i].error for i in ids])), 4),
    ))
    check("point.track.elements[].image_id", lambda: sum(len(points3D[i].track.elements) for i in ids))

    colmap_image_dir = scene / "images"
    image_dir = scene / (f"images_{factor}" if factor > 1 else "images")

    def mapping():
        colmap_files = sorted(rel_paths(str(colmap_image_dir)))
        image_files = sorted(rel_paths(str(image_dir)))
        if len(colmap_files) != len(image_files):
            raise AssertionError(f"{len(colmap_files)} images vs {len(image_files)} in {image_dir.name}")
        pairs = dict(zip(colmap_files, image_files))
        # sorted-zip only maps correctly if the two listings share stem order
        bad = [p for p in pairs.items() if Path(p[0]).stem != Path(p[1]).stem]
        if bad:
            raise AssertionError(f"filename stems diverge, e.g. {bad[:2]}")
        missing = [n for n in (imdata[k].name for k in imdata) if not (image_dir / pairs[n]).exists()]
        if missing:
            raise AssertionError(f"{len(missing)} resolved paths do not exist")
        ext = Path(image_files[0]).suffix.lower()
        # upstream silently re-resizes into images_N_png/ if it finds JPEGs here
        rescale = factor > 1 and ext == ".jpg"
        return f"{len(pairs)} paths, ext={ext}, upstream re-resize={rescale}"

    check("images/ -> images_N/ name mapping", mapping)

    def intrinsics():
        first = sorted(rel_paths(str(image_dir)))[0]
        actual = Image.open(image_dir / first).size
        expected = (cam.width // factor, cam.height // factor)
        if actual != expected:
            raise AssertionError(f"Parser scales K to {expected}, images are {actual}")
        return f"K/{factor} matches {actual[0]}x{actual[1]}"

    check("scaled intrinsics vs actual pixels", intrinsics)

    def normalize():
        c2w = np.linalg.inv(np.stack([image_w2c(i) for i in imdata.values()]))
        pts = np.array([points3D[i].xyz for i in ids], dtype=np.float32)
        t1 = ns["similarity_from_cameras"](c2w)
        c2w, pts = ns["transform_cameras"](t1, c2w), ns["transform_points"](t1, pts)
        t2 = ns["align_principal_axes"](pts)
        pts = ns["transform_points"](t2, pts)
        flip = bool(np.median(pts[:, 2]) > np.mean(pts[:, 2]))
        return f"extent={np.abs(pts).max():.2f}, upside-down flip applies={flip}"

    check("normalize=True pipeline", normalize)


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsplat", required=True, type=Path, help="Path to a gsplat checkout.")
    ap.add_argument("--scene", type=Path, default=repo / "data/scenes/room-1",
                    help="Scene directory written by run_sfm.py.")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="--data_factor values to check; each needs its images_<N>/ directory.")
    args = ap.parse_args(argv)

    if not args.scene.exists():
        raise SystemExit(f"scene not found: {args.scene}")

    ns = load_upstream(args.gsplat.expanduser())
    print(f"[extract] Parser helpers lifted from {args.gsplat}")

    check = Checks()
    for factor in args.factors:
        verify(args.scene, args.gsplat, factor, ns, check)

    if check.failures:
        print(f"\n[result] {len(check.failures)} FAILED: {check.failures}")
        return 1
    print("\n[result] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
