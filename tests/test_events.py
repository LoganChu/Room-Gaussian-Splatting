"""Event log tests. Pure numpy/pyarrow -- no GPU, no gsplat."""

from __future__ import annotations

import numpy as np
import pytest

from splat.events import BIRTH_KINDS, EventLog, read_events


def test_empty_log_round_trips(tmp_path):
    log = EventLog()
    assert len(log) == 0
    a = log.to_arrays()
    assert all(v.size == 0 for v in a.values())
    t = read_events(log.flush(tmp_path / "e.parquet"))
    assert t.num_rows == 0


def test_columns_and_dtypes_survive_the_round_trip(tmp_path):
    log = EventLog()
    log.add(0, "sfm", np.arange(3))
    t = read_events(log.flush(tmp_path / "e.parquet"))
    assert t.column_names == ["step", "kind", "gid", "parent"]
    assert t.schema.field("step").type == "int32"
    assert t.schema.field("gid").type == "int64"
    # dictionary-encoded: six distinct kinds over millions of rows
    assert "dictionary" in str(t.schema.field("kind").type)


def test_parent_defaults_to_minus_one():
    log = EventLog()
    log.add(0, "sfm", np.arange(3))
    assert list(log.to_arrays()["parent"]) == [-1, -1, -1]


def test_parent_is_kept_when_given():
    log = EventLog()
    log.add(600, "clone", np.array([7, 8]), np.array([1, 2]))
    a = log.to_arrays()
    assert list(a["gid"]) == [7, 8]
    assert list(a["parent"]) == [1, 2]


def test_empty_batch_is_a_no_op():
    """_grow_gs calls with n_dupli == 0 are common; they must not write rows."""
    log = EventLog()
    log.add(600, "clone", np.array([], dtype=np.int64))
    assert len(log) == 0


def test_reset_has_no_gid_or_parent():
    log = EventLog()
    log.add_reset(3000)
    a = log.to_arrays()
    assert list(a["kind"]) == ["reset"]
    assert list(a["gid"]) == [-1] and list(a["parent"]) == [-1]


def test_unknown_kind_is_rejected():
    log = EventLog()
    with pytest.raises(ValueError, match="unknown event kind"):
        log.add(0, "spawn", np.array([1]))


def test_mismatched_parent_length_is_rejected():
    log = EventLog()
    with pytest.raises(ValueError, match="length mismatch"):
        log.add(0, "clone", np.array([1, 2, 3]), np.array([9]))


def test_flush_without_a_path_is_a_no_op(tmp_path):
    log = EventLog()
    log.add(0, "sfm", np.arange(2))
    assert log.flush() is None


def test_books_balance_across_a_synthetic_run(tmp_path):
    """births - deaths must equal the population. This is the invariant that
    catches a miscounted densification: it is checked against a number derived
    independently of the log."""
    log = EventLog(tmp_path / "e.parquet")
    population = 100
    log.add(0, "sfm", np.arange(population))
    next_id = population
    rng = np.random.default_rng(0)
    for step in range(600, 3000, 100):
        n_clone, n_split, n_prune = 7, 5, 4
        clones = np.arange(next_id, next_id + n_clone); next_id += n_clone
        log.add(step, "clone", clones, rng.integers(0, population, n_clone))
        kids = np.arange(next_id, next_id + 2 * n_split); next_id += 2 * n_split
        parents = rng.integers(0, population, n_split)
        log.add(step, "split", kids, np.concatenate([parents, parents]))
        log.add(step, "death", parents)  # split removes its parents
        log.add(step, "death", rng.integers(0, population, n_prune))
        population += n_clone + 2 * n_split - n_split - n_prune

    a = read_events(log.flush())
    kinds = np.asarray(a.column("kind").to_pylist())
    births = int(np.isin(kinds, BIRTH_KINDS).sum())
    deaths = int((kinds == "death").sum())
    assert births - deaths == population


def test_large_log_stays_compact(tmp_path):
    """A real run logs millions of rows; the on-disk cost must stay sane."""
    log = EventLog(tmp_path / "e.parquet")
    n = 200_000
    log.add(700, "split", np.arange(n), np.repeat(np.arange(n // 2), 2))
    path = log.flush()
    assert read_events(path).num_rows == n
    # Raw is 20 bytes/row (int32 + int64 + int64, plus the kind string). zstd
    # over dictionary-encoded kinds and near-sorted ids gets well under half
    # that: ~4.8 here, and 2.96 measured on a real 1.35M-event run.
    per_row = path.stat().st_size / n
    assert per_row < 8, f"{per_row:.2f} bytes/row is worse than expected"
