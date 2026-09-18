"""Report tests: the joins and the shared image helpers.

The plots themselves are checked only for being written -- what is pinned here
is the data assembly behind them, especially the stage/eval join, which has an
off-by-one that differs at exactly one stage.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from splat.paths import RunPaths
from splat.report import (
    abs_error,
    error_map,
    join_stage_metrics,
    load_stages,
    load_val_metrics,
    plot_curriculum,
    psnr,
    strip,
)

STAGES = {
    "order": [0, 1, 2, 3],
    "stages": [
        # closed at a boundary, so its eval is at end_step - 1
        {"stage": 0, "step": 0, "end_step": 200, "added": None, "image": None,
         "n_images": 3, "n_points_unlocked": 40, "n_seeded": 0, "n_rejected_near": 0,
         "n_gaussians_before": 40, "n_gaussians_after": 40},
        {"stage": 1, "step": 200, "end_step": 400, "added": 3, "image": "IMG_0003.jpeg",
         "n_images": 4, "n_points_unlocked": 12, "n_seeded": 9, "n_rejected_near": 3,
         "n_gaussians_before": 40, "n_gaussians_after": 90,
         "psnr_before": 10.0, "psnr_after": 14.5},
        # the last stage is closed at max_steps - 1, where the eval also lands
        {"stage": 2, "step": 400, "end_step": 999, "added": 2, "image": "IMG_0002.jpeg",
         "n_images": 5, "n_points_unlocked": 5, "n_seeded": 4, "n_rejected_near": 1,
         "n_gaussians_before": 90, "n_gaussians_after": 150,
         "psnr_before": 12.0, "psnr_after": 13.0},
    ],
}


@pytest.fixture
def run(tmp_path) -> RunPaths:
    paths = RunPaths(tmp_path).mkdirs()
    (paths.root / "stages.json").write_text(json.dumps(STAGES))
    stats = paths.root / "stats"
    stats.mkdir(exist_ok=True)
    for step, ps, gs in [(199, 8.0, 40), (399, 11.0, 90), (999, 13.5, 150)]:
        (stats / f"val_step{step:04d}.json").write_text(
            json.dumps({"psnr": ps, "ssim": 0.5, "lpips": 0.4, "num_GS": gs})
        )
    return paths


# -- loading ---------------------------------------------------------------


def test_a_non_curriculum_run_has_no_stages(tmp_path):
    assert load_stages(RunPaths(tmp_path).mkdirs()) is None


def test_val_metrics_are_keyed_by_step(run):
    val = load_val_metrics(run)
    assert sorted(val) == [199, 399, 999]
    assert val[399]["psnr"] == 11.0


def test_val_metrics_of_a_run_with_no_stats(tmp_path):
    assert load_val_metrics(RunPaths(tmp_path)) == {}


# -- the join --------------------------------------------------------------


def test_each_stage_gets_the_last_eval_at_or_before_it_ended(run):
    """The trainer evals at `step == eval_step - 1`, and the last stage closes
    at max_steps - 1 rather than at a boundary, so the two conventions differ by
    one at exactly one stage. "Most recent eval" is right under both."""
    rows = join_stage_metrics(load_stages(run), load_val_metrics(run))
    assert [r["eval_step"] for r in rows] == [199, 399, 999]
    assert [r["test_psnr"] for r in rows] == [8.0, 11.0, 13.5]


def test_the_join_cross_checks_the_population(run):
    """stages.json and the trainer's own eval count Gaussians independently."""
    rows = join_stage_metrics(load_stages(run), load_val_metrics(run))
    for row in rows:
        assert row["test_num_GS"] == row["n_gaussians_after"]


def test_a_stage_with_no_eval_keeps_its_other_fields(run):
    rows = join_stage_metrics(load_stages(run), {})
    assert len(rows) == 3
    assert all("test_psnr" not in r for r in rows)
    assert rows[1]["n_seeded"] == 9


def test_the_join_does_not_mutate_the_records(run):
    stages = load_stages(run)
    join_stage_metrics(stages, load_val_metrics(run))
    assert "test_psnr" not in stages["stages"][0]


# -- plots -----------------------------------------------------------------


def test_curriculum_plot_is_written(run):
    pytest.importorskip("matplotlib")
    out = plot_curriculum(run, run.report)
    assert out is not None and out.name == "curriculum.png"
    assert out.stat().st_size > 0


def test_no_curriculum_plot_without_stages(tmp_path):
    assert plot_curriculum(RunPaths(tmp_path).mkdirs(), tmp_path) is None


# -- the shared image helpers ---------------------------------------------


def test_psnr_matches_the_definition():
    gt = np.full((8, 8, 3), 128, np.uint8)
    assert psnr(gt, gt.astype(np.float32) / 255.0) == float("inf")
    render = np.full((8, 8, 3), 128 / 255.0 + 0.1, np.float32)
    assert psnr(gt, render) == pytest.approx(-10 * np.log10(0.01), rel=1e-3)


def test_abs_error_is_the_mean_over_channels():
    gt = np.zeros((1, 1, 3), np.uint8)
    render = np.array([[[0.3, 0.0, 0.0]]], np.float32)
    assert abs_error(gt, render)[0, 0] == pytest.approx(0.1)


def test_an_explicit_vmax_makes_two_maps_comparable():
    """Scaled independently, a before and an after both fill the ramp and the
    improvement between them becomes invisible."""
    gt = np.zeros((1, 2, 3), np.uint8)
    worse = np.array([[[0.8] * 3, [0.0] * 3]], np.float32)
    better = np.array([[[0.2] * 3, [0.0] * 3]], np.float32)
    alone = (error_map(gt, worse)[0, 0], error_map(gt, better)[0, 0])
    np.testing.assert_allclose(*alone, atol=1e-6), "self-scaled: identical"

    shared = 0.8
    a = error_map(gt, worse, vmax=shared)[0, 0]
    b = error_map(gt, better, vmax=shared)[0, 0]
    assert not np.allclose(a, b), "shared scale: the difference shows"


def test_strip_concatenates_horizontally():
    out = strip([np.zeros((2, 3, 3), np.float32), np.ones((2, 4, 3), np.float32)])
    assert out.shape == (2, 7, 3) and out.dtype == np.uint8
    assert out[0, 0].max() == 0 and out[0, -1].min() == 255


# -- the opacity-reset shadow ---------------------------------------------


def test_an_eval_just_after_a_reset_is_flagged():
    """Measured on dev-groups: an eval 2 steps after the reset at 3000 read
    5.74 dB against 16.87 before and 17.14 after. An 11 dB hole that is
    entirely the documented reset dip, and would read as a regression."""
    from splat.report import RESET_SHADOW_STEPS, in_reset_shadow

    resets = np.array([3000, 6000])
    assert in_reset_shadow(3002, resets)
    assert in_reset_shadow(3000, resets)
    assert in_reset_shadow(3000 + RESET_SHADOW_STEPS - 1, resets)
    assert not in_reset_shadow(3000 + RESET_SHADOW_STEPS, resets)
    assert not in_reset_shadow(2999, resets), "before the reset is fine"
    assert in_reset_shadow(6050, resets), "every reset casts one"


def test_no_resets_means_no_shadow():
    from splat.report import in_reset_shadow

    assert not in_reset_shadow(3002, np.empty(0, dtype=np.int64))


def test_reset_steps_of_a_run_without_an_event_log(tmp_path):
    from splat.report import reset_steps

    assert reset_steps(RunPaths(tmp_path).mkdirs()).size == 0
