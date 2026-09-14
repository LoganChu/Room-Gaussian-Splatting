"""Tests for the SfM pipeline's pure-Python parts.

These deliberately avoid pycolmap so they run anywhere, including before the
Linux toolchain exists. The COLMAP calls themselves are covered by the Phase 2
acceptance check on real photos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splat.sfm import (  # noqa: E402
    EXIF_ORIENTATION_TAG,
    SfmConfig,
    downscale_images,
    evaluate_quality,
    list_images,
    pick_best_model,
    plan_outputs,
    prepare_images,
    run_sfm,
    validate_inputs,
)

PIL = pytest.importorskip("PIL", reason="Pillow needed for image tests")
from PIL import Image  # noqa: E402


# EXIF tags COLMAP reads to seed camera intrinsics.
FOCAL_LENGTH_35MM_TAG = 0xA405
MAKE_TAG = 0x010F


def make_image(path: Path, size=(64, 48), color=(120, 40, 200), exif=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    im.save(path, **({"exif": exif.tobytes()} if exif is not None else {}))
    return path


def phone_exif(orientation: int) -> "Image.Exif":
    """EXIF as a phone writes it: orientation plus the tags COLMAP reads."""
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = orientation
    exif[FOCAL_LENGTH_35MM_TAG] = 24
    exif[MAKE_TAG] = "Apple"
    return exif


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    src = tmp_path / "photos"
    for name in ["b.jpg", "a.jpg", "c.png"]:
        make_image(src / name)
    (src / "notes.txt").write_text("not an image", encoding="utf-8")
    return src


class TestListImages:
    def test_filters_non_images_and_sorts(self, photos: Path):
        names = [p.name for p in list_images(photos)]
        assert names == ["a.jpg", "b.jpg", "c.png"]

    def test_recurses_into_subdirs(self, photos: Path):
        make_image(photos / "sub" / "d.jpg")
        rels = [p.relative_to(photos).as_posix() for p in list_images(photos)]
        assert "sub/d.jpg" in rels
        assert rels == sorted(rels), "order must match gsplat's sorted pairing"

    def test_missing_dir_is_empty(self, tmp_path: Path):
        assert list_images(tmp_path / "nope") == []


class TestValidateInputs:
    def test_accepts_good_input(self, photos: Path, tmp_path: Path):
        assert validate_inputs(photos, tmp_path / "out") == []

    def test_rejects_missing_dir(self, tmp_path: Path):
        problems = validate_inputs(tmp_path / "nope", tmp_path / "out")
        assert len(problems) == 1 and "does not exist" in problems[0]

    def test_rejects_too_few_images(self, tmp_path: Path):
        src = tmp_path / "few"
        make_image(src / "only.jpg")
        problems = validate_inputs(src, tmp_path / "out")
        assert any("at least" in p for p in problems)

    def test_rejects_output_equal_to_input(self, photos: Path):
        assert any("must differ" in p for p in validate_inputs(photos, photos))


class TestPlanOutputs:
    def test_layout_matches_gsplat_expectations(self, tmp_path: Path):
        paths = plan_outputs(tmp_path / "scene")
        assert paths.images.name == "images"
        assert paths.sparse0 == tmp_path / "scene" / "sparse" / "0"
        assert paths.database.name == "database.db"

    def test_downscale_dir_naming(self, tmp_path: Path):
        # gsplat looks for images_<factor>; the name must match exactly.
        assert plan_outputs(tmp_path).downscale_dir(4).name == "images_4"


class TestPrepareImages:
    def test_copies_all_images(self, photos: Path, tmp_path: Path):
        sources = list_images(photos)
        copied, rotated = prepare_images(sources, tmp_path / "images", photos)
        assert copied == 3 and rotated == 0
        assert len(list_images(tmp_path / "images")) == 3

    def test_is_idempotent(self, photos: Path, tmp_path: Path):
        sources = list_images(photos)
        prepare_images(sources, tmp_path / "images", photos)
        copied, _ = prepare_images(sources, tmp_path / "images", photos)
        assert copied == 0, "existing files should be skipped so runs can resume"

    def test_preserves_subdir_structure(self, photos: Path, tmp_path: Path):
        make_image(photos / "sub" / "d.jpg")
        prepare_images(list_images(photos), tmp_path / "images", photos)
        assert (tmp_path / "images" / "sub" / "d.jpg").exists()

    # The orientation != 1 branch. Phone photos shot in portrait land here, so it
    # runs for most real captures even though the fixtures above never reach it.
    @pytest.mark.parametrize("name", ["portrait.jpg", "portrait.png"])
    def test_rotated_photo_is_uprighted(self, tmp_path: Path, name: str):
        src_root = tmp_path / "src"
        src = make_image(src_root / name, size=(64, 48), exif=phone_exif(6))

        copied, rotated = prepare_images([src], tmp_path / "images", src_root)

        assert (copied, rotated) == (1, 1)
        with Image.open(tmp_path / "images" / name) as out:
            assert out.size == (48, 64), "orientation 6 must transpose the pixels"
            assert out.getexif().get(EXIF_ORIENTATION_TAG) in (None, 1), (
                "orientation must be cleared, or viewers rotate the pixels twice"
            )

    def test_rotated_photo_keeps_focal_length_exif(self, tmp_path: Path):
        """COLMAP seeds intrinsics from EXIF; without it the initial focal length
        falls back to 1.2 * max(w, h), which is ~1.8x off for a phone camera."""
        src_root = tmp_path / "src"
        src = make_image(src_root / "portrait.jpg", exif=phone_exif(6))

        prepare_images([src], tmp_path / "images", src_root)

        with Image.open(tmp_path / "images" / "portrait.jpg") as out:
            exif = out.getexif()
        assert exif.get(FOCAL_LENGTH_35MM_TAG) == 24
        assert exif.get(MAKE_TAG) == "Apple"

    def test_upright_photo_is_copied_byte_for_byte(self, tmp_path: Path):
        src_root = tmp_path / "src"
        src = make_image(src_root / "landscape.jpg", exif=phone_exif(1))

        copied, rotated = prepare_images([src], tmp_path / "images", src_root)

        assert (copied, rotated) == (1, 0), "no needless re-encode"
        assert (tmp_path / "images" / "landscape.jpg").read_bytes() == src.read_bytes()


class TestDownscaleImages:
    def test_writes_png_at_right_size(self, photos: Path, tmp_path: Path):
        images = tmp_path / "images"
        prepare_images(list_images(photos), images, photos)
        written = downscale_images(images, tmp_path / "images_2", factor=2)
        assert written == 3
        out = sorted((tmp_path / "images_2").glob("*.png"))
        assert len(out) == 3, "PNG avoids gsplat re-resizing into images_2_png/"
        with Image.open(out[0]) as im:
            assert im.size == (32, 24)

    def test_skips_existing(self, photos: Path, tmp_path: Path):
        images = tmp_path / "images"
        prepare_images(list_images(photos), images, photos)
        downscale_images(images, tmp_path / "images_4", factor=4)
        assert downscale_images(images, tmp_path / "images_4", factor=4) == 0

    def test_rejects_bad_factor(self, tmp_path: Path):
        with pytest.raises(ValueError):
            downscale_images(tmp_path, tmp_path / "out", factor=0)


class TestQualityGate:
    def good_stats(self, **over):
        stats = {
            "input_images": 50,
            "registered_images": 50,
            "registered_ratio": 1.0,
            "points3D": 50_000,
            "mean_reprojection_error_px": 0.6,
        }
        stats.update(over)
        return stats

    def test_clean_run_has_no_warnings(self):
        assert evaluate_quality(self.good_stats()) == []

    def test_flags_low_registration(self):
        stats = self.good_stats(registered_images=30, registered_ratio=0.6)
        assert any("registered" in w for w in evaluate_quality(stats))

    def test_flags_high_reprojection_error(self):
        assert any("reprojection" in w
                   for w in evaluate_quality(self.good_stats(mean_reprojection_error_px=2.5)))

    def test_flags_sparse_point_cloud(self):
        assert any("3D points" in w for w in evaluate_quality(self.good_stats(points3D=900)))


class TestPickBestModel:
    def test_picks_most_registered(self):
        class FakeRec:
            def __init__(self, n): self._n = n
            def num_reg_images(self): return self._n

        best_id, rec = pick_best_model({0: FakeRec(4), 1: FakeRec(31), 2: FakeRec(9)})
        assert best_id == 1 and rec.num_reg_images() == 31

    def test_raises_when_colmap_found_nothing(self):
        with pytest.raises(RuntimeError, match="no reconstruction"):
            pick_best_model({})


class TestDryRun:
    def test_runs_without_pycolmap_and_writes_nothing(self, photos: Path, tmp_path: Path, capsys):
        out = tmp_path / "scene"
        report = run_sfm(SfmConfig(image_dir=photos, out=out, dry_run=True))
        assert report["input_images_found"] == 3
        assert not out.exists(), "dry run must not create the output tree"
        assert "sparse" in capsys.readouterr().out

    def test_reports_problems_instead_of_raising(self, tmp_path: Path):
        report = run_sfm(SfmConfig(image_dir=tmp_path / "nope", out=tmp_path / "o", dry_run=True))
        assert report["problems"]


def _run_cli(args: list[str]):
    """Invoke scripts/run_sfm.py as a subprocess, the way a user runs it."""
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sfm.py"
    return subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True
    )


class TestCliExitCodes:
    def test_dry_run_succeeds(self, photos: Path, tmp_path: Path):
        result = _run_cli(["--images", str(photos), "--out", str(tmp_path / "o"), "--dry-run"])
        assert result.returncode == 0, result.stderr

    def test_missing_images_dir_exits_nonzero(self, tmp_path: Path):
        # A dry run that cannot proceed must fail loudly, not exit 0.
        result = _run_cli(["--images", str(tmp_path / "nope"), "--out", str(tmp_path / "o"), "--dry-run"])
        assert result.returncode == 1
        assert "does not exist" in result.stdout

    def test_too_few_images_exits_nonzero(self, tmp_path: Path):
        src = tmp_path / "few"
        make_image(src / "one.jpg")
        result = _run_cli(["--images", str(src), "--out", str(tmp_path / "o"), "--dry-run"])
        assert result.returncode == 1
