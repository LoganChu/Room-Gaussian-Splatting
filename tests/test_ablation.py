"""Ablation analysis tests.

The power arithmetic is the part that changes decisions -- it says whether a
weekend of runs can resolve the effect being looked for -- so it is pinned
against closed-form values rather than against itself.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from splat.ablation import (
    PRIOR_SD_DB,
    collect,
    conditions_for,
    effect_at_n,
    format_table,
    n_for_effect,
    plot,
    power_note,
    read_psnr,
    settings_of,
    summarize,
)


def write_run(root, n_images: int, repeat: int, psnr: float, steps: int = 7000, factor: int = 4):
    result = root / f"n{n_images:03d}_r{repeat}"
    (result / "stats").mkdir(parents=True, exist_ok=True)
    (result / "stats" / f"val_step{steps - 1:04d}.json").write_text(
        json.dumps({"psnr": psnr, "ssim": 0.5, "num_GS": 1000})
    )
    (result / "cfg.yml").write_text(
        f"max_steps: {steps}\ndata_factor: {factor}\n"
        "strategy: !!python/object:gsplat.strategy.default.DefaultStrategy\n"
        "  refine_every: 100\n"
    )
    return result


# -- the power arithmetic --------------------------------------------------


def test_n_for_effect_matches_the_closed_form():
    # n = 2(1.96 + 0.8416)^2 sigma^2 / delta^2, ceil'd
    assert n_for_effect(1.0, 1.0) == 16
    assert n_for_effect(2.0, 1.0) == 63, "four times the variance, four times the runs"
    assert n_for_effect(1.0, 2.0) == 4, "twice the effect, a quarter of the runs"


def test_the_phase_3_spread_needs_32_runs_for_half_a_decibel():
    """The number worth knowing before committing to the ablation."""
    assert n_for_effect(PRIOR_SD_DB, 0.5) == 32
    assert n_for_effect(PRIOR_SD_DB, 1.0) == 8


def test_effect_at_n_inverts_n_for_effect():
    for sd in (0.3, 0.708, 1.5):
        for delta in (0.5, 1.0, 2.0):
            n = n_for_effect(sd, delta)
            assert effect_at_n(sd, n) <= delta + 1e-9, (sd, delta)


def test_effect_at_n_falls_as_the_root_of_n():
    assert effect_at_n(1.0, 16) == pytest.approx(effect_at_n(1.0, 4) / 2, rel=1e-6)


def test_one_run_resolves_nothing_useful():
    assert effect_at_n(PRIOR_SD_DB, 1) > 2.5
    assert effect_at_n(1.0, 0) == float("inf")


def test_a_zero_effect_needs_no_answer():
    assert n_for_effect(1.0, 0.0) == 0


# -- conditions ------------------------------------------------------------


def test_conditions_are_sorted_and_deduped():
    todo, skipped = conditions_for("10,3,5,3", 28)
    assert todo == [3, 5, 10]
    assert skipped == []


def test_all_resolves_to_the_train_count():
    todo, _ = conditions_for("3,all", 28)
    assert todo == [3, 28]


def test_conditions_the_scene_cannot_run_are_dropped_not_clamped():
    """Clamping N=40 to 28 would put one condition in the table twice."""
    todo, skipped = conditions_for("3,5,10,20,40,all", 28)
    assert todo == [3, 5, 10, 20, 28]
    assert skipped == [40]
    assert todo.count(28) == 1


def test_a_condition_below_three_images_is_refused():
    with pytest.raises(ValueError, match="at least 3"):
        conditions_for("2", 28)


def test_no_runnable_condition_is_an_error():
    with pytest.raises(ValueError, match="no runnable"):
        conditions_for("40,50", 28)


# -- collection ------------------------------------------------------------


def test_read_psnr_takes_the_last_eval(tmp_path):
    result = write_run(tmp_path, 10, 0, 17.5, steps=1000)
    (result / "stats" / "val_step0499.json").write_text(json.dumps({"psnr": 9.9}))
    assert read_psnr(result) == pytest.approx(17.5), "step 999 sorts after step 499"


def test_read_psnr_survives_a_run_that_never_evaluated(tmp_path):
    (tmp_path / "n003_r0" / "stats").mkdir(parents=True)
    assert read_psnr(tmp_path / "n003_r0") is None


def test_collect_groups_repeats_by_condition(tmp_path):
    write_run(tmp_path, 3, 0, 8.0)
    write_run(tmp_path, 3, 1, 9.0)
    write_run(tmp_path, 10, 0, 16.0)
    got = collect(tmp_path)
    assert sorted(got) == [3, 10]
    assert sorted(got[3]) == [8.0, 9.0]


def test_collect_ignores_unfinished_and_unrelated_directories(tmp_path):
    write_run(tmp_path, 3, 0, 8.0)
    (tmp_path / "n010_r0").mkdir()  # launched, never evaluated
    (tmp_path / "notes").mkdir()
    assert sorted(collect(tmp_path)) == [3]


def test_settings_come_from_the_runs_not_the_flags(tmp_path):
    """On a report-only pass the flags are defaults that never ran."""
    write_run(tmp_path, 3, 0, 8.0, steps=1234, factor=8)
    assert settings_of(tmp_path) == {"max_steps": 1234, "data_factor": 8}


def test_settings_of_an_empty_directory_is_empty(tmp_path):
    assert settings_of(tmp_path) == {}


# -- summary ---------------------------------------------------------------


def test_summarize_uses_a_sample_standard_deviation():
    """ddof=1. At n=2 the difference from the population sd is 40%."""
    s = summarize({10: [16.0, 18.0]})
    row = s["conditions"][0]
    assert row["psnr_mean"] == pytest.approx(17.0)
    assert row["psnr_sd"] == pytest.approx(np.std([16.0, 18.0], ddof=1))
    assert row["psnr_sd"] == pytest.approx(1.4142, abs=1e-3)


def test_a_single_run_has_no_spread():
    s = summarize({10: [16.0]})
    assert s["conditions"][0]["psnr_sd"] is None
    assert s["pooled_sd"] is None, "and nothing to pool"


def test_pooled_sd_weights_by_degrees_of_freedom():
    s = summarize({3: [1.0, 3.0], 10: [10.0, 10.5, 11.0, 11.5]})
    var = ((2 - 1) * np.var([1.0, 3.0], ddof=1)
           + (4 - 1) * np.var([10.0, 10.5, 11.0, 11.5], ddof=1)) / (1 + 3)
    assert s["pooled_sd"] == pytest.approx(float(np.sqrt(var)))


def test_pooled_sd_ignores_conditions_with_one_run():
    s = summarize({3: [1.0, 3.0], 10: [16.0]})
    assert s["pooled_sd"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_conditions_come_out_sorted():
    s = summarize({28: [1.0], 3: [2.0], 10: [3.0]})
    assert [r["n_images"] for r in s["conditions"]] == [3, 10, 28]


# -- presentation ----------------------------------------------------------


def test_power_note_names_the_source_and_the_resolvable_effect():
    text = power_note(0.708, 5, "Phase 3 prior")
    assert "0.708 dB (Phase 3 prior)" in text
    assert "at 5 run(s) per condition" in text
    assert "32 runs per condition" in text, "the 0.5 dB row"


def test_format_table_marks_a_missing_spread():
    text = format_table(summarize({3: [8.0]}))
    assert "-" in text.splitlines()[1]


def test_plot_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    s = summarize({3: [8.0, 9.0], 10: [16.0, 15.5]})
    out = plot(s, tmp_path / "ablation.png", "test")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_plot_of_nothing_is_none(tmp_path):
    assert plot({"conditions": []}, tmp_path / "x.png") is None
