"""COLMAP structure-from-motion, producing the layout gsplat expects (Phase 2).

Output layout, matching gsplat's ``examples/datasets/colmap.py`` Parser:

    <out>/
      images/           normalized copies of the input photos; COLMAP reads these
      images_2/ ...     PNG downscales. The Parser REQUIRES images_<factor>/ to
                        exist for factor > 1, and re-resizes it into a separate
                        images_<factor>_png/ if it finds JPEGs, so we write PNG.
      sparse/0/         cameras.bin, images.bin, points3D.bin (largest model)
      database.db
      sfm_report.json   stats, timings and quality warnings

Written against pycolmap 4.2.0, whose API differs from 3.x: ``extract_features``
takes ``extraction_options``/``reader_options``, and ``undistort_images`` takes
the output path first.

pycolmap is imported lazily so that ``--dry-run`` and the unit tests work in an
environment that does not have it installed yet.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
EXIF_ORIENTATION_TAG = 0x0112

DEFAULT_DOWNSCALES = (2, 4, 8)
MIN_IMAGES = 3
RECOMMENDED_IMAGES = 20
MIN_REGISTERED_RATIO = 0.9
MAX_MEAN_REPROJ_PX = 1.0


@dataclass
class SfmPaths:
    """Every path the pipeline reads or writes, derived from the output root."""

    out: Path
    images: Path
    sparse: Path
    sparse0: Path
    database: Path
    report: Path
    undistorted: Path

    def downscale_dir(self, factor: int) -> Path:
        """Directory gsplat's Parser looks for when --factor <factor> is used."""
        return self.out / f"images_{factor}"


def plan_outputs(out: Path) -> SfmPaths:
    """Derive the output layout without creating anything."""
    out = Path(out)
    return SfmPaths(
        out=out,
        images=out / "images",
        sparse=out / "sparse",
        sparse0=out / "sparse" / "0",
        database=out / "database.db",
        report=out / "sfm_report.json",
        undistorted=out / "undistorted",
    )


def list_images(image_dir: Path) -> List[Path]:
    """Return image files under image_dir, sorted by relative path.

    Sorted order matters: gsplat pairs images/ with images_N/ by zipping the two
    sorted file lists, so both must enumerate in the same order.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return []
    files = [
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(files, key=lambda p: p.relative_to(image_dir).as_posix())


def validate_inputs(image_dir: Path, out: Path, min_images: int = MIN_IMAGES) -> List[str]:
    """Return a list of blocking problems. Empty means good to run."""
    problems: List[str] = []
    image_dir = Path(image_dir)
    if not image_dir.exists():
        problems.append(f"Input image directory does not exist: {image_dir}")
        return problems
    if not image_dir.is_dir():
        problems.append(f"Input image path is not a directory: {image_dir}")
        return problems

    images = list_images(image_dir)
    if len(images) < min_images:
        problems.append(
            f"Found {len(images)} images in {image_dir}; need at least {min_images}."
        )

    out = Path(out)
    if out.exists() and not out.is_dir():
        problems.append(f"Output path exists and is not a directory: {out}")

    try:
        if out.resolve() == image_dir.resolve():
            problems.append("Output directory must differ from the input image directory.")
    except OSError:
        pass

    return problems


def input_warnings(image_dir: Path) -> List[str]:
    """Non-blocking advice about the capture itself."""
    warnings: List[str] = []
    images = list_images(image_dir)
    if 0 < len(images) < RECOMMENDED_IMAGES:
        warnings.append(
            f"Only {len(images)} images. 30-100 overlapping photos give a much "
            "better reconstruction; few images often fail to register."
        )
    suffixes = {p.suffix.lower() for p in images}
    if len(suffixes) > 1:
        warnings.append(f"Mixed image formats {sorted(suffixes)}; this is usually fine.")
    return warnings


def _exif_orientation(path: Path) -> int:
    """EXIF orientation of an image, or 1 when absent or unreadable."""
    try:
        from PIL import Image
    except ImportError:
        return 1
    try:
        with Image.open(path) as im:
            return int(im.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    except Exception:
        return 1


def prepare_images(
    sources: Sequence[Path],
    images_dir: Path,
    src_root: Path,
    normalize_orientation: bool = True,
    jpeg_quality: int = 95,
) -> Tuple[int, int]:
    """Copy photos into images_dir, returning (copied, rotated).

    Photos carrying a non-trivial EXIF orientation are re-encoded upright, so the
    pixels COLMAP sees match the pixels the downscales are made from. Everything
    else is copied byte-for-byte, avoiding a needless JPEG re-encode.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = rotated = 0

    for src in sources:
        rel = Path(src).relative_to(src_root)
        dst = images_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue

        orientation = _exif_orientation(src) if normalize_orientation else 1
        if orientation != 1:
            from PIL import Image, ImageOps

            with Image.open(src) as im:
                upright = ImageOps.exif_transpose(im)
                if upright.mode not in ("RGB", "L"):
                    upright = upright.convert("RGB")
                # exif_transpose returns a NEW image: its .format is None and its
                # EXIF is not carried into save() automatically. Both matter.
                #   - subsampling="keep" requires a JPEG source, so it raises here.
                #     4:4:4 keeps the full chroma the "keep" was protecting.
                #   - COLMAP seeds intrinsics from EXIF focal length (falling back
                #     to a ~1.8x-off guess of 1.2 * max(w, h) without it), so the
                #     tags must be written through, upright now that pixels are.
                exif = upright.getexif()
                exif[EXIF_ORIENTATION_TAG] = 1
                save_kwargs: Dict[str, Any] = {"exif": exif.tobytes()}
                if dst.suffix.lower() in (".jpg", ".jpeg"):
                    save_kwargs.update(quality=jpeg_quality, subsampling=0)
                upright.save(dst, **save_kwargs)
            rotated += 1
        else:
            shutil.copy2(src, dst)
        copied += 1

    return copied, rotated


def downscale_images(images_dir: Path, dst_dir: Path, factor: int) -> int:
    """Write PNG downscales of every image, skipping ones already present."""
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    from PIL import Image

    images_dir, dst_dir = Path(images_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for src in list_images(images_dir):
        rel = src.relative_to(images_dir)
        dst = dst_dir / rel.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        with Image.open(src) as im:
            im = im.convert("RGB")
            size = (int(round(im.width / factor)), int(round(im.height / factor)))
            im.resize(size, Image.BICUBIC).save(dst)
        written += 1

    return written


def summarize_reconstruction(rec: Any, n_input_images: int) -> Dict[str, Any]:
    """Collect the numbers that say whether SfM succeeded."""
    n_reg = int(rec.num_reg_images())
    return {
        "input_images": int(n_input_images),
        "registered_images": n_reg,
        "registered_ratio": (n_reg / n_input_images) if n_input_images else 0.0,
        "cameras": int(rec.num_cameras()),
        "points3D": int(rec.num_points3D()),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "mean_observations_per_image": float(rec.compute_mean_observations_per_reg_image()),
    }


def evaluate_quality(
    stats: Dict[str, Any],
    min_registered_ratio: float = MIN_REGISTERED_RATIO,
    max_reproj_px: float = MAX_MEAN_REPROJ_PX,
) -> List[str]:
    """Phase 2 acceptance check: >=90% registered, mean reprojection error <~1px."""
    warnings: List[str] = []
    ratio = stats.get("registered_ratio", 0.0)
    if ratio < min_registered_ratio:
        warnings.append(
            f"Only {stats.get('registered_images')}/{stats.get('input_images')} images "
            f"registered ({ratio:.0%}, want >={min_registered_ratio:.0%}). More overlap "
            "or more texture in frame usually fixes this."
        )
    err = stats.get("mean_reprojection_error_px", float("inf"))
    if err > max_reproj_px:
        warnings.append(
            f"Mean reprojection error {err:.2f}px exceeds {max_reproj_px:.2f}px; "
            "poses may be too loose for sharp splatting."
        )
    if stats.get("points3D", 0) < 5000:
        warnings.append(
            f"Only {stats.get('points3D')} 3D points. The initial point cloud will be "
            "sparse, so densification has to invent more of the room."
        )
    return warnings


def pick_best_model(maps: Dict[int, Any]) -> Tuple[int, Any]:
    """Pick the model with the most registered images (COLMAP may return several)."""
    if not maps:
        raise RuntimeError(
            "COLMAP produced no reconstruction. Usually too little overlap between "
            "photos, or too little texture (blank walls)."
        )
    best_id = max(maps, key=lambda k: int(maps[k].num_reg_images()))
    return best_id, maps[best_id]


@dataclass
class SfmConfig:
    """Everything the pipeline needs, so a run is reproducible from the report."""

    image_dir: Path
    out: Path
    camera_model: str = "OPENCV"
    single_camera: bool = True
    matcher: str = "exhaustive"
    use_gpu: bool = True
    max_image_size: int = 3200
    max_num_features: int = 8192
    downscales: Tuple[int, ...] = DEFAULT_DOWNSCALES
    normalize_orientation: bool = True
    undistort: bool = False
    dry_run: bool = False
    verbose_level: int = 1

    def to_json(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["image_dir"] = str(self.image_dir)
        data["out"] = str(self.out)
        data["downscales"] = list(self.downscales)
        return data


def run_sfm(cfg: SfmConfig, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run the full pipeline. Returns the report dict (also written as JSON).

    With cfg.dry_run set, validates inputs and reports the planned layout without
    importing pycolmap or writing anything.
    """
    paths = plan_outputs(cfg.out)
    sources = list_images(cfg.image_dir)
    report: Dict[str, Any] = {
        "config": cfg.to_json(),
        "layout": {
            "images": str(paths.images),
            "sparse0": str(paths.sparse0),
            "database": str(paths.database),
            "downscale_dirs": [str(paths.downscale_dir(f)) for f in cfg.downscales],
        },
        "input_images_found": len(sources),
        "timings_sec": {},
        "warnings": [],
    }

    problems = validate_inputs(cfg.image_dir, cfg.out)
    if problems:
        report["problems"] = problems
        for p in problems:
            log(f"ERROR: {p}")
        if not cfg.dry_run:
            raise SystemExit(1)
        return report

    report["warnings"].extend(input_warnings(cfg.image_dir))
    for w in report["warnings"]:
        log(f"NOTE: {w}")

    if cfg.dry_run:
        log("\nDry run. Planned layout:")
        for key, value in report["layout"].items():
            log(f"  {key}: {value}")
        log(f"  would process {len(sources)} images")
        return report

    import pycolmap  # imported late so --dry-run and tests work without it

    report["pycolmap_version"] = pycolmap.__version__
    paths.out.mkdir(parents=True, exist_ok=True)
    device = pycolmap.Device.auto if cfg.use_gpu else pycolmap.Device.cpu

    t0 = time.time()
    copied, rotated = prepare_images(
        sources, paths.images, Path(cfg.image_dir), cfg.normalize_orientation
    )
    report["timings_sec"]["prepare_images"] = round(time.time() - t0, 1)
    log(f"Prepared {copied} images ({rotated} re-encoded upright from EXIF).")

    t0 = time.time()
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = cfg.camera_model
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = cfg.use_gpu
    extraction_options.max_image_size = cfg.max_image_size
    extraction_options.sift.max_num_features = cfg.max_num_features
    camera_mode = (
        pycolmap.CameraMode.SINGLE if cfg.single_camera else pycolmap.CameraMode.AUTO
    )
    pycolmap.extract_features(
        database_path=paths.database,
        image_path=paths.images,
        camera_mode=camera_mode,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )
    report["timings_sec"]["extract_features"] = round(time.time() - t0, 1)
    log(
        f"Extracted features ({cfg.camera_model}, "
        f"{'single shared camera' if cfg.single_camera else 'per-image cameras'})."
    )

    t0 = time.time()
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = cfg.use_gpu
    if cfg.matcher == "sequential":
        pycolmap.match_sequential(
            database_path=paths.database, matching_options=matching_options, device=device
        )
    else:
        pycolmap.match_exhaustive(
            database_path=paths.database, matching_options=matching_options, device=device
        )
    report["timings_sec"]["match"] = round(time.time() - t0, 1)
    log(f"Matched features ({cfg.matcher}).")

    t0 = time.time()
    paths.sparse.mkdir(parents=True, exist_ok=True)
    maps = pycolmap.incremental_mapping(
        database_path=paths.database,
        image_path=paths.images,
        output_path=paths.sparse,
        options=pycolmap.IncrementalPipelineOptions(),
    )
    report["timings_sec"]["mapping"] = round(time.time() - t0, 1)

    best_id, rec = pick_best_model(maps)
    report["n_models"] = len(maps)
    report["best_model_id"] = int(best_id)
    if len(maps) > 1:
        report["warnings"].append(
            f"COLMAP split the scene into {len(maps)} models; using the largest. "
            "The photos likely form disconnected groups."
        )
    paths.sparse0.mkdir(parents=True, exist_ok=True)
    rec.write_binary(str(paths.sparse0))

    stats = summarize_reconstruction(rec, len(sources))
    report["stats"] = stats
    report["warnings"].extend(evaluate_quality(stats))

    t0 = time.time()
    for factor in cfg.downscales:
        written = downscale_images(paths.images, paths.downscale_dir(factor), factor)
        log(f"Downscaled x{factor}: wrote {written} PNGs.")
    report["timings_sec"]["downscale"] = round(time.time() - t0, 1)

    if cfg.undistort:
        t0 = time.time()
        paths.undistorted.mkdir(parents=True, exist_ok=True)
        pycolmap.undistort_images(
            output_path=paths.undistorted,
            input_path=paths.sparse0,
            image_path=paths.images,
        )
        report["timings_sec"]["undistort"] = round(time.time() - t0, 1)
        log(f"Wrote undistorted copy to {paths.undistorted}.")

    paths.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_summary(report: Dict[str, Any]) -> str:
    """Human-readable end-of-run summary."""
    lines = ["", "=" * 60, "SfM summary"]
    stats = report.get("stats")
    if stats:
        lines += [
            f"  registered : {stats['registered_images']}/{stats['input_images']} "
            f"({stats['registered_ratio']:.0%})",
            f"  3D points  : {stats['points3D']:,}",
            f"  reproj err : {stats['mean_reprojection_error_px']:.3f} px",
            f"  track len  : {stats['mean_track_length']:.2f}",
        ]
    timings = report.get("timings_sec", {})
    if timings:
        total = sum(timings.values())
        lines.append(f"  time       : {total / 60:.1f} min " + str(timings))
    warnings = report.get("warnings", [])
    if warnings:
        lines.append("  warnings:")
        lines += [f"    - {w}" for w in warnings]
    else:
        lines.append("  no warnings")
    lines.append("=" * 60)
    return "\n".join(lines)
