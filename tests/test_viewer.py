"""Playback-layer tests: everything in viewer.py that is not viser or CUDA.

The render callback itself needs a GPU and a browser, so what is pinned here is
the part that decides *what* gets handed to the rasterizer -- the raw/activated
distinction, the mode dispatch, the selection mask and the fly-to geometry.
Getting those wrong is silent; getting the rasterization call wrong is loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from splat.events import EventLog
from splat.paths import RunPaths
from splat.timeline import Timeline

torch = pytest.importorskip("torch")

from splat.snapshots import SnapshotWriter  # noqa: E402
from splat.report import psnr  # noqa: E402
from splat.viewer import (  # noqa: E402
    AGE_MODE,
    ORIGIN_MODE,
    ROLE_COLORS,
    Cameras,
    Playback,
    error_triptych,
)

CPU = torch.device("cpu")


def make_splats(n: int, scale: float = 1.0):
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) * scale),
            "scales": torch.nn.Parameter(torch.full((n, 3), -2.0)),  # log space
            "quats": torch.nn.Parameter(torch.tile(torch.tensor([1.0, 0, 0, 0]), (n, 1))),
            "opacities": torch.nn.Parameter(torch.zeros(n)),  # logit space: sigmoid -> 0.5
            "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
        }
    )


@pytest.fixture
def playback(tmp_path) -> Playback:
    paths = RunPaths(tmp_path).mkdirs()
    log = EventLog()
    log.add(0, "sfm", np.arange(3))
    log.add(100, "clone", np.array([3]), np.array([0]))
    log.add(100, "split", np.array([4, 5]), np.array([1, 1]))
    log.add(100, "death", np.array([1]))
    log.flush(paths.events)
    writer = SnapshotWriter(paths.snapshots, every=100)
    writer.save(0, make_splats(3), {"ids": torch.arange(3)})
    writer.save(100, make_splats(5), {"ids": torch.tensor([0, 2, 3, 4, 5])})
    return Playback(Timeline(paths), CPU, cache=2)


# -- frames ----------------------------------------------------------------


def test_frame_promotes_fp16_and_keeps_raw_activations(playback):
    """Snapshots hold log scales and logit opacities, exactly as the trainer
    does. Activating them here would double-apply exp/sigmoid at render time."""
    f = playback.frame(0)
    assert f["means"].dtype == torch.float32
    assert f["ids"].dtype == torch.int64
    np.testing.assert_allclose(f["scales"].numpy(), -2.0, atol=1e-2)
    np.testing.assert_allclose(f["opacities"].numpy(), 0.0, atol=1e-2)
    np.testing.assert_allclose(f["scales"].exp().numpy(), np.exp(-2.0), atol=1e-2)
    np.testing.assert_allclose(f["opacities"].sigmoid().numpy(), 0.5, atol=1e-2)


def test_ids_are_kept_on_the_host_for_lineage_lookups(playback):
    f = playback.frame(100)
    assert isinstance(f["ids_np"], np.ndarray)
    assert f["ids_np"].tolist() == [0, 2, 3, 4, 5]


def test_frames_are_cached_and_evicted_in_order(playback):
    a = playback.frame(0)
    assert playback.frame(0) is a, "a re-scrub to the same frame must not reload"
    playback.frame(100)
    playback.frame(0)  # touching 0 makes 100 the eviction candidate
    assert sorted(playback._cache) == [0, 100]
    assert len(playback._cache) == 2


# -- what the rasterizer is handed -----------------------------------------


def test_rgb_mode_passes_sh_dc_coefficients(playback):
    colors, sh_degree = playback.colors(100, "rgb")
    assert sh_degree == 0, "0 means 'these are SH DC, evaluate them'"
    assert tuple(colors.shape) == (5, 1, 3)


@pytest.mark.parametrize("mode", [ORIGIN_MODE, AGE_MODE])
def test_lineage_modes_pass_flat_rgb(playback, mode):
    colors, sh_degree = playback.colors(100, mode)
    assert sh_degree is None, "None means 'already RGB'"
    assert tuple(colors.shape) == (5, 3)
    assert colors.min() >= 0 and colors.max() <= 1


def test_origin_colours_follow_the_frame_not_the_row_order(playback):
    """ids at step 100 are [0, 2, 3, 4, 5]: two sfm, one clone, two split."""
    from splat.timeline import ORIGIN_COLORS

    colors, _ = playback.colors(100, ORIGIN_MODE)
    np.testing.assert_allclose(colors[0].numpy(), ORIGIN_COLORS["sfm"], atol=1e-6)
    np.testing.assert_allclose(colors[2].numpy(), ORIGIN_COLORS["clone"], atol=1e-6)
    np.testing.assert_allclose(colors[4].numpy(), ORIGIN_COLORS["split"], atol=1e-6)


def test_selection_masks_by_gid_not_by_position(playback):
    mask = playback.selection(100, np.array([3, 5]))
    assert mask.tolist() == [False, False, True, False, True]
    assert playback.selection(100, np.array([])) is None
    assert playback.selection(100, None) is None
    assert playback.selection(100, np.array([1])) is None, "1 is dead at step 100"


# -- fly-to geometry -------------------------------------------------------


def test_extent_bounds_the_whole_frame(playback):
    # means are arange(9).reshape(3, 3), so the box runs [0,1,2]..[6,7,8]
    centre, radius = playback.extent(0)
    np.testing.assert_allclose(centre, [3.0, 4.0, 5.0], atol=1e-2)
    assert radius == pytest.approx(np.linalg.norm([6.0, 6.0, 6.0]) / 2, rel=1e-2)


def test_extent_bounds_a_selection(playback):
    centre, radius = playback.extent(100, np.array([0, 3]))
    # gid 0 is row 0 (means 0,1,2) and gid 3 is row 2 (means 6,7,8)
    np.testing.assert_allclose(centre, [3.0, 4.0, 5.0], atol=1e-2)
    assert radius > 0
    whole, _ = playback.extent(100)
    assert not np.allclose(centre, whole), "a selection must not bound the frame"


def test_extent_is_none_when_nothing_is_alive(playback):
    assert playback.extent(100, np.array([1])) is None


# -- the camera comparison panel -------------------------------------------


def test_triptych_is_three_panels_wide():
    gt = np.full((4, 5, 3), 200, np.uint8)
    render = np.zeros((4, 5, 3), np.float32)
    out = error_triptych(gt, render)
    assert out.shape == (4, 15, 3) and out.dtype == np.uint8


def test_triptych_error_is_scaled_to_the_pair():
    """A converged run's absolute error is small everywhere; against a fixed
    1.0 scale the error panel is a black rectangle."""
    gt = np.zeros((2, 2, 3), np.uint8)
    render = np.zeros((2, 2, 3), np.float32)
    render[0, 0] = 0.01  # a tiny error, but the largest one present
    out = error_triptych(gt, render)
    err = out[:, 4:, :]
    assert err[0, 0].max() > 0, "the worst pixel must still light up"


def test_psnr_matches_the_definition():
    gt = np.full((8, 8, 3), 128, np.uint8)
    assert psnr(gt, gt.astype(np.float32) / 255.0) == float("inf")
    render = np.full((8, 8, 3), 128 / 255.0 + 0.1, np.float32)
    assert psnr(gt, render) == pytest.approx(-10 * np.log10(0.01), rel=1e-3)


# -- degrading without the scene -------------------------------------------


def test_cameras_are_optional(tmp_path):
    """A run directory is meant to rsync; its data_dir may not exist here.
    That costs the frustums, not the timeline."""
    assert Cameras.from_cfg({}) is None
    assert Cameras.from_cfg({"data_dir": str(tmp_path / "nope")}) is None


def test_every_role_has_a_frustum_colour():
    assert set(ROLE_COLORS) == {"train", "test", "pending", "added", "resting"}
    assert all(len(v) == 3 for v in ROLE_COLORS.values())


def test_resting_reads_as_a_dimmer_active_not_as_pending():
    """"seen and seeded" is much closer to active than to never-seen, so it
    shares active's hue rather than pending's grey."""
    resting = np.array(ROLE_COLORS["resting"], dtype=float)
    active = np.array(ROLE_COLORS["train"], dtype=float)
    pending = np.array(ROLE_COLORS["pending"], dtype=float)
    assert resting.sum() < active.sum(), "dimmer"
    assert np.linalg.norm(resting - active) < np.linalg.norm(resting - pending)
