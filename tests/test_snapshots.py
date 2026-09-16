"""Snapshot round-trip tests. These are pure numpy/torch and need no GPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from splat.snapshots import FIELDS, SnapshotReader, SnapshotWriter  # noqa: E402


def make_splats(n: int):
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3)),
            "scales": torch.nn.Parameter(torch.randn(n, 3)),
            "quats": torch.nn.Parameter(torch.randn(n, 4)),
            "opacities": torch.nn.Parameter(torch.randn(n)),
            "sh0": torch.nn.Parameter(torch.randn(n, 1, 3)),
            "shN": torch.nn.Parameter(torch.randn(n, 15, 3)),  # must NOT be saved
        }
    )


def test_round_trip_preserves_shapes_and_values(tmp_path):
    splats = make_splats(32)
    state = {"ids": torch.arange(32)}
    w = SnapshotWriter(tmp_path, every=100)
    w.save(500, splats, state)

    got = SnapshotReader(tmp_path).load(500)
    for k in FIELDS:
        assert got[k].shape == tuple(splats[k].shape), k
        # fp16 is lossy but must still be close
        np.testing.assert_allclose(
            got[k], splats[k].detach().numpy(), rtol=1e-2, atol=1e-2, err_msg=k
        )
    assert got["ids"].tolist() == list(range(32))
    assert int(got["step"]) == 500


def test_high_order_sh_is_dropped(tmp_path):
    """shN is ~8x the payload and the viewer never needs it."""
    w = SnapshotWriter(tmp_path, every=100)
    w.save(0, make_splats(16), None)
    got = SnapshotReader(tmp_path).load(0)
    assert "shN" not in got
    assert "sh0" in got


def test_ids_come_from_state_not_position(tmp_path):
    """After densification, ids are not arange -- the snapshot must carry the
    real ones or lineage colouring in the viewer is meaningless."""
    splats = make_splats(4)
    state = {"ids": torch.tensor([9, 3, 7, 1])}
    SnapshotWriter(tmp_path, every=1).save(700, splats, state)
    assert SnapshotReader(tmp_path).load(700)["ids"].tolist() == [9, 3, 7, 1]


def test_falls_back_to_arange_without_state(tmp_path):
    SnapshotWriter(tmp_path, every=1).save(0, make_splats(5), None)
    assert SnapshotReader(tmp_path).load(0)["ids"].tolist() == [0, 1, 2, 3, 4]


def test_cadence(tmp_path):
    w = SnapshotWriter(tmp_path, every=250, also=(1, 500))
    assert w.due(0) and w.due(250) and w.due(500)
    assert w.due(1), "explicit steps fire even off-cadence"
    assert not w.due(251)


def test_disabled_writer_writes_nothing(tmp_path):
    w = SnapshotWriter(tmp_path, every=1, enabled=False)
    assert w.maybe_save(0, make_splats(4), None) is None
    assert SnapshotReader(tmp_path).paths == []


def test_reader_orders_by_step_and_counts_cheaply(tmp_path):
    w = SnapshotWriter(tmp_path, every=1)
    for step, n in [(1000, 8), (0, 4), (250, 6)]:
        w.save(step, make_splats(n), {"ids": torch.arange(n)})
    r = SnapshotReader(tmp_path)
    assert r.steps == [0, 250, 1000], "must sort numerically, not lexically"
    assert r.counts() == {0: 4, 250: 6, 1000: 8}
    assert [len(s["ids"]) for s in r] == [4, 6, 8]
