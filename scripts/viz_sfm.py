#!/usr/bin/env python3
"""Export a COLMAP sparse model to JSON for the SfM inspection viewer.

The viewer (an HTML page) needs the reconstruction in a browser-friendly shape:
the sparse point cloud, the camera poses that produced it, and the covisibility
graph that explains how well the photos actually tie together.

    python scripts/viz_sfm.py --scene data/scenes/room-1 --out /tmp/sfm.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def export(scene: Path) -> dict:
    import pycolmap

    rec = pycolmap.Reconstruction(str(scene / "sparse" / "0"))

    # --- cameras -------------------------------------------------------------
    # COLMAP solves cam_from_world; the viewer wants each camera's world-space
    # centre and axes, which is the inverse.
    images = []
    for image_id, im in sorted(rec.images.items()):
        world_from_cam = im.cam_from_world().inverse()
        images.append(
            {
                "id": image_id,
                "name": im.name,
                "center": [round(float(v), 4) for v in world_from_cam.translation],
                # row-major 3x3; columns are the camera's right/down/forward axes
                "R": [round(float(v), 5) for v in world_from_cam.rotation.matrix().flatten()],
                "n_points": len([p for p in im.points2D if p.has_point3D()]),
            }
        )

    cam = list(rec.cameras.values())[0]

    # --- points --------------------------------------------------------------
    # Trim the long tail: a handful of far-flung outliers otherwise dominate the
    # bounding box and squash the actual room down to nothing on screen.
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
    err = np.array([p.error for p in rec.points3D.values()], dtype=np.float64)
    track = np.array([p.track.length() for p in rec.points3D.values()], dtype=np.int32)

    centre = np.median(xyz, axis=0)
    radius = np.percentile(np.linalg.norm(xyz - centre, axis=1), 99.0)
    keep = np.linalg.norm(xyz - centre, axis=1) <= radius * 1.5

    points = {
        "xyz": [round(float(v), 4) for v in xyz[keep].flatten()],
        "rgb": [int(v) for v in rgb[keep].flatten()],
        "track": [int(v) for v in track[keep]],
        "err": [round(float(v), 3) for v in err[keep]],
        "n_total": int(len(xyz)),
        "n_shown": int(keep.sum()),
    }

    # --- covisibility --------------------------------------------------------
    # Two photos are connected when they observe the same 3D point. This is what
    # actually determines whether the reconstruction holds together.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for p in rec.points3D.values():
        ids = sorted({el.image_id for el in p.track.elements})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair_counts[(ids[i], ids[j])] += 1

    edges = [{"a": a, "b": b, "n": n} for (a, b), n in pair_counts.items() if n >= 1]
    edges.sort(key=lambda e: -e["n"])

    n_img = len(rec.images)
    stats = {
        "n_images": n_img,
        "n_registered": rec.num_reg_images(),
        "n_points": len(rec.points3D),
        "n_observations": rec.compute_num_observations(),
        "mean_reproj_px": round(float(rec.compute_mean_reprojection_error()), 4),
        "mean_track_length": round(float(rec.compute_mean_track_length()), 3),
        "median_track_length": int(np.median(track)),
        "camera_model": cam.model.name,
        "width": cam.width,
        "height": cam.height,
        "params": [round(float(v), 3) for v in cam.params],
        "n_pairs_possible": n_img * (n_img - 1) // 2,
        "n_pairs_connected": len(pair_counts),
    }

    return {"stats": stats, "images": images, "points": points, "edges": edges}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, type=Path, help="Scene dir containing sparse/0/")
    ap.add_argument("--out", required=True, type=Path, help="Destination .json")
    args = ap.parse_args()

    data = export(args.scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":")))

    s = data["stats"]
    print(f"{s['n_registered']}/{s['n_images']} images, {s['n_points']} points, "
          f"{s['n_pairs_connected']}/{s['n_pairs_possible']} pairs connected")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
