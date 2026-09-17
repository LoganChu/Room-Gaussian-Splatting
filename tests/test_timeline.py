"""Timeline and lineage-index tests. Pure numpy/pyarrow -- no GPU, no viser.

The viewer's whole claim is that the colours mean what the panel says they
mean. A lineage read back wrong produces a picture that is coherent, pretty and
false -- the same failure ``test_lineage.py`` guards on the way in -- so the way
out is pinned here, on a synthetic run whose answers are worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from splat.events import BIRTH_KINDS, EventLog
from splat.paths import RunPaths
from splat.timeline import (
    Lineage,
    ORIGIN_COLORS,
    Schedule,
    Timeline,
    UNKNOWN_COLOR,
    colors_by_age,
    colors_by_origin,
    highlight,
    load_cfg,
)

torch = pytest.importorskip("torch")

from splat.snapshots import SnapshotWriter  # noqa: E402


# A four-point SfM cloud put through three refinements. Worked out by hand:
#
#   step   0  sfm    0 1 2 3
#   step 100  clone  0 -> 4       split 1 -> 5, 6   (1 dies)
#   step 200  prune  2
#   step 300  split  5 -> 7, 8    (5 dies)           + an opacity reset
#
# births 9, deaths 3, so 6 alive: 0 3 4 6 7 8.
def synthetic_log() -> EventLog:
    log = EventLog()
    log.add(0, "sfm", np.arange(4))
    log.add(100, "clone", np.array([4]), np.array([0]))
    log.add(100, "split", np.array([5, 6]), np.array([1, 1]))
    log.add(100, "death", np.array([1]))
    log.add(200, "death", np.array([2]))
    log.add(300, "split", np.array([7, 8]), np.array([5, 5]))
    log.add(300, "death", np.array([5]))
    log.add_reset(300)
    return log


ALIVE = np.array([0, 3, 4, 6, 7, 8], dtype=np.int32)


@pytest.fixture
def lineage() -> Lineage:
    a = synthetic_log().to_arrays()
    return Lineage.from_arrays(a["step"], a["kind"].astype(str), a["gid"], a["parent"])


def make_splats(n: int):
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3)),
            "scales": torch.nn.Parameter(torch.randn(n, 3)),
            "quats": torch.nn.Parameter(torch.randn(n, 4)),
            "opacities": torch.nn.Parameter(torch.randn(n)),
            "sh0": torch.nn.Parameter(torch.randn(n, 1, 3)),
        }
    )


@pytest.fixture
def run(tmp_path) -> RunPaths:
    """A complete instrumented run: snapshots, event log and a config."""
    paths = RunPaths(tmp_path).mkdirs()
    synthetic_log().flush(paths.events)
    writer = SnapshotWriter(paths.snapshots, every=100)
    for step, ids in [
        (0, [0, 1, 2, 3]),
        (100, [0, 2, 3, 4, 5, 6]),
        (200, [0, 3, 4, 5, 6]),
        (300, list(ALIVE)),
    ]:
        writer.save(step, make_splats(len(ids)), {"ids": torch.tensor(ids)})
    (paths.root / "cfg.yml").write_text(
        "data_factor: 8\n"
        "max_steps: 400\n"
        "sh_degree: 3\n"
        "sh_degree_interval: 100\n"
        "strategy: !!python/object:gsplat.strategy.default.DefaultStrategy\n"
        "  refine_start_iter: 100\n"
        "  refine_stop_iter: 350\n"
        "  refine_every: 100\n"
        "  reset_every: 300\n"
    )
    return paths


# -- the lineage index -----------------------------------------------------


def test_origins_are_read_back_per_gaussian(lineage):
    codes = lineage.origin_codes(ALIVE)
    kinds = [BIRTH_KINDS[c] for c in codes]
    assert kinds == ["sfm", "sfm", "clone", "split", "split", "split"]


def test_origin_counts_cover_every_alive_gaussian(lineage):
    counts = lineage.origin_counts(ALIVE)
    assert counts == {"sfm": 2, "seed": 0, "clone": 1, "split": 3, "unknown": 0}
    assert sum(counts.values()) == ALIVE.size


def test_births_minus_deaths_is_the_population(lineage):
    """The invariant report.py checks on the log, checked on the index."""
    born = int((lineage.birth_step >= 0).sum())
    dead = int((lineage.death_step >= 0).sum())
    assert (born, dead) == (9, 3)
    assert born - dead == ALIVE.size


def test_ids_are_dense_so_gid_can_be_the_index(lineage):
    """lineage.py hands ids out contiguously; direct indexing depends on it."""
    assert len(lineage) == 9
    assert not (lineage.birth_step < 0).any()


def test_ancestor_chain_reaches_the_sfm_point(lineage):
    assert lineage.ancestors(7) == [5, 1]
    assert lineage.root_of(7) == 1
    assert lineage.ancestors(3) == [], "an sfm point has no parent"
    assert lineage.root_of(3) == 3


def test_children_are_found_by_parent(lineage):
    assert lineage.children_of(1).tolist() == [5, 6]
    assert lineage.children_of(0).tolist() == [4]
    assert lineage.children_of(5).tolist() == [7, 8]
    assert lineage.children_of(3).tolist() == []


def test_descendants_are_transitive(lineage):
    assert sorted(lineage.descendants(1)) == [5, 6, 7, 8]
    assert sorted(lineage.descendants(5)) == [7, 8]
    assert lineage.descendants(8).tolist() == []


def test_family_is_the_whole_clan_from_the_root(lineage):
    """What "follow one Gaussian's lineage" has to mean once it has been split:
    the Gaussian you picked is dead and its siblings are the story."""
    assert sorted(lineage.family(7)) == [1, 5, 6, 7, 8]
    assert sorted(lineage.family(1)) == [1, 5, 6, 7, 8], "any member gives the same clan"
    assert sorted(lineage.family(3)) == [3], "a childless sfm point is its own family"


def test_split_parents_are_recorded_dead(lineage):
    assert int(lineage.death_step[1]) == 100
    assert int(lineage.death_step[5]) == 300
    assert int(lineage.death_step[7]) == -1, "-1 means alive at the end of the run"


def test_ages_count_from_birth(lineage):
    ages = lineage.ages(np.array([0, 4, 7]), step=300)
    assert ages.tolist() == [300.0, 200.0, 0.0]


def test_resets_are_kept(lineage):
    assert lineage.resets.tolist() == [300]


def test_unknown_gids_degrade_rather_than_crash(lineage):
    """A viewer that opens beats one that refuses; gaps show up grey."""
    codes = lineage.origin_codes(np.array([0, 99, -1]))
    assert codes.tolist() == [0, -1, -1]
    assert lineage.ages(np.array([99]), 300).tolist() == [-1.0]
    assert "out of range" in lineage.describe(99)


def test_a_death_without_a_birth_is_ignored(lineage):
    log = synthetic_log()
    log.add(400, "death", np.array([4242]))  # never born: a broken log
    a = log.to_arrays()
    built = Lineage.from_arrays(a["step"], a["kind"].astype(str), a["gid"], a["parent"])
    assert len(built) == 9, "the phantom must not stretch the index"


def test_describe_reads_as_provenance(lineage):
    text = lineage.describe(7)
    assert "split at step 300" in text
    assert "parent 5" in text
    assert "depth 2 below sfm point 1" in text


# -- colours ---------------------------------------------------------------


def test_origin_colours_match_the_report_palette(lineage):
    rgb = colors_by_origin(lineage, ALIVE)
    assert rgb.shape == (6, 3) and rgb.dtype == np.float32
    np.testing.assert_allclose(rgb[0], ORIGIN_COLORS["sfm"], atol=1e-6)
    np.testing.assert_allclose(rgb[2], ORIGIN_COLORS["clone"], atol=1e-6)
    np.testing.assert_allclose(rgb[3], ORIGIN_COLORS["split"], atol=1e-6)


def test_unknown_origin_is_grey(lineage):
    np.testing.assert_allclose(
        colors_by_origin(lineage, np.array([12345]))[0], UNKNOWN_COLOR, atol=1e-6
    )


def test_age_colours_span_the_frame_not_the_run(lineage):
    """Late in a run almost everything is young; a fixed scale flattens it."""
    rgb = colors_by_age(lineage, ALIVE, step=300)
    assert rgb.shape == (6, 3) and rgb.dtype == np.float32
    oldest = colors_by_age(lineage, np.array([0]), step=300)
    youngest = colors_by_age(lineage, np.array([7]), step=300)
    assert not np.allclose(oldest, youngest)


def test_highlight_selects_the_living_members(lineage):
    mask = highlight(ALIVE, lineage.family(7))
    assert ALIVE[mask].tolist() == [6, 7, 8], "1 and 5 are dead by step 300"


# -- the schedule ----------------------------------------------------------


def test_cfg_is_read_without_executing_it(run):
    """cfg.yml carries !!python/object tags and run dirs are meant to rsync."""
    cfg = load_cfg(run)
    assert cfg["data_factor"] == 8
    assert isinstance(cfg["strategy"], dict), "the tag must degrade to a mapping"
    assert cfg["strategy"]["reset_every"] == 300


def test_schedule_comes_from_the_run(run):
    s = Schedule.from_cfg(load_cfg(run))
    assert (s.refine_start_iter, s.refine_stop_iter, s.reset_every) == (100, 350, 300)
    assert s.max_steps == 400


def test_schedule_falls_back_to_gsplat_defaults():
    s = Schedule.from_cfg({})
    assert (s.refine_start_iter, s.refine_stop_iter, s.reset_every) == (500, 15000, 3000)


def test_phases_follow_the_refinement_window():
    s = Schedule(refine_start_iter=500, refine_stop_iter=15000)
    assert s.phase(0) == "warm-up"
    assert s.phase(499) == "warm-up"
    assert s.phase(500) == "densification"
    assert s.phase(14999) == "densification"
    assert s.phase(15000) == "refinement"


def test_sh_degree_rises_once_per_interval():
    s = Schedule(sh_degree=3, sh_degree_interval=1000)
    assert [s.sh_degree_at(x) for x in (0, 999, 1000, 3000, 30000)] == [0, 0, 1, 3, 3]


# -- the timeline ----------------------------------------------------------


def test_timeline_indexes_the_snapshots(run):
    t = Timeline(run)
    assert t.steps == [0, 100, 200, 300]
    assert len(t) == 4
    assert t.frame(300)["ids"].tolist() == list(ALIVE)


def test_nearest_snaps_to_an_existing_frame(run):
    t = Timeline(run)
    assert t.nearest(0) == 0
    assert t.nearest(140) == 100
    assert t.nearest(9999) == 300


def test_densify_steps_and_counts(run):
    t = Timeline(run)
    assert t.densify_steps.tolist() == [100, 200, 300]
    assert t.last_densify(100) == {"step": 100, "clone": 1, "split": 2, "death": 1}
    assert t.last_densify(250) == {"step": 200, "clone": 0, "split": 0, "death": 1}
    assert t.last_densify(50) == {}, "nothing has been refined yet"


def test_resets_reach_the_timeline(run):
    assert Timeline(run).resets.tolist() == [300]


def test_narration_names_the_phase_and_the_population(run):
    t = Timeline(run)
    early = t.narrate(0, t.frame(0)["ids"])
    assert "warm-up" in early and "No densification until 100" in early
    assert "SH degree 0 of 3" in early

    late = t.narrate(300, t.frame(300)["ids"])
    assert "densification" in late
    assert "6 Gaussians" in late
    assert "sfm 2" in late and "split 3" in late
    assert "last refinement at 300" in late
    assert "last opacity reset at 300" in late


def test_a_vanilla_run_still_has_a_timeline(tmp_path):
    """No events.parquet: the origin modes go away, the scrub does not."""
    paths = RunPaths(tmp_path).mkdirs()
    SnapshotWriter(paths.snapshots, every=1).save(0, make_splats(3), None)
    t = Timeline(paths)
    assert t.lineage is None
    assert t.steps == [0]
    assert t.resets.size == 0
    assert t.last_densify(0) == {}
    assert "step 0" in t.narrate(0, t.frame(0)["ids"])


def test_an_empty_run_reports_rather_than_guesses(tmp_path):
    t = Timeline(RunPaths(tmp_path).mkdirs())
    assert t.steps == []
    with pytest.raises(ValueError, match="no snapshots"):
        t.nearest(0)
