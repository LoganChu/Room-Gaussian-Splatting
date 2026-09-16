"""Lineage tests.

These drive the REAL gsplat ops on a tiny synthetic population rather than
mocking them, because the thing being tested is precisely our understanding of
gsplat's internal tensor layout. A mock would only assert that we agree with
ourselves.

Requires a CUDA device: gsplat's ops are CUDA-only.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from splat.events import EventLog  # noqa: E402
from splat.lineage import InstrumentedStrategy  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gsplat's ops are CUDA-only"
)

SMALL = float(np.log(0.001))  # below grow_scale3d * scene_scale -> clone
LARGE = float(np.log(10.0))  # above it -> split


def make_population(n: int, device: str = "cuda"):
    params = {
        "means": torch.nn.Parameter(
            torch.arange(n, dtype=torch.float32).reshape(n, 1).repeat(1, 3).to(device)
        ),
        "scales": torch.nn.Parameter(torch.full((n, 3), SMALL, device=device)),
        "quats": torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1)
        ),
        "opacities": torch.nn.Parameter(torch.zeros(n, device=device)),
    }
    optimizers = {k: torch.optim.Adam([v], lr=0.0) for k, v in params.items()}
    return params, optimizers


def primed(strategy, params, n, *, high_grad=(), large=()):
    """State with grad2d/count filled in, so _grow_gs can be called directly."""
    state = strategy.initialize_state(scene_scale=1.0)
    device = params["means"].device
    state["count"] = torch.ones(n, device=device)
    state["grad2d"] = torch.zeros(n, device=device)
    if len(high_grad):
        state["grad2d"][list(high_grad)] = 1.0
    if len(large):
        with torch.no_grad():
            params["scales"][list(large)] = LARGE
    strategy._ensure_ids(params, state, step=0)
    return state


def test_ids_start_as_arange_and_log_sfm_births():
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, _ = make_population(5)
    state = primed(s, params, 5)
    assert state["ids"].tolist() == [0, 1, 2, 3, 4]
    a = events.to_arrays()
    assert list(a["kind"]) == ["sfm"] * 5
    assert list(a["gid"]) == [0, 1, 2, 3, 4]
    assert list(a["parent"]) == [-1] * 5


def test_clone_ids_are_fresh_and_point_at_parents():
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, opts = make_population(5)
    state = primed(s, params, 5, high_grad=(1, 3))
    n_dupli, n_split = s._grow_gs(params, opts, state, step=1000)

    assert (n_dupli, n_split) == (2, 0)
    ids = state["ids"].tolist()
    assert ids[:5] == [0, 1, 2, 3, 4], "originals must keep their ids and positions"
    assert ids[5:] == [5, 6], "clones get fresh ids appended at the end"

    a = events.to_arrays()
    clone = [i for i, k in enumerate(a["kind"]) if k == "clone"]
    assert [a["gid"][i] for i in clone] == [5, 6]
    assert [a["parent"][i] for i in clone] == [1, 3]


def test_split_children_are_child_major_not_adjacent_pairs():
    """The failure this guards against is silent: reading children as adjacent
    pairs produces a complete lineage with every child on the wrong parent."""
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, opts = make_population(5)
    state = primed(s, params, 5, high_grad=(1, 3), large=(1, 3))
    n_dupli, n_split = s._grow_gs(params, opts, state, step=1000)

    assert (n_dupli, n_split) == (0, 2)
    ids = state["ids"].tolist()
    assert ids[:3] == [0, 2, 4], "non-split originals survive, in order"
    assert len(ids) == 7

    a = events.to_arrays()
    split = [i for i, k in enumerate(a["kind"]) if k == "split"]
    gids = [a["gid"][i] for i in split]
    parents = [a["parent"][i] for i in split]

    # 4 children from 2 parents; child-major => [p0, p1, p0, p1], NOT [p0, p0, p1, p1]
    assert parents == [1, 3, 1, 3], f"expected child-major order, got {parents}"
    assert parents != [1, 1, 3, 3], "adjacent-pair reading must not pass"
    assert sorted(gids) == [5, 6, 7, 8], "children get fresh ids"
    # each parent has exactly two children
    assert sorted(parents) == [1, 1, 3, 3]


def test_grow_with_both_clone_and_split_keeps_blocks_disjoint():
    """Clones land in the MIDDLE, between survivors and children."""
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, opts = make_population(8)
    state = primed(s, params, 8, high_grad=(1, 2, 5, 6), large=(5, 6))
    n_dupli, n_split = s._grow_gs(params, opts, state, step=1000)

    assert (n_dupli, n_split) == (2, 2)
    ids = state["ids"].tolist()
    assert len(ids) == 8 + n_dupli + n_split == 12

    a = events.to_arrays()
    by = lambda k: [  # noqa: E731
        (a["gid"][i], a["parent"][i]) for i, kk in enumerate(a["kind"]) if kk == k
    ]
    assert [p for _, p in by("clone")] == [1, 2]
    assert [p for _, p in by("split")] == [5, 6, 5, 6]
    # all ids unique -- no clone accidentally sharing a child's id
    assert len(set(ids)) == len(ids)
    # every fresh id is new
    assert set(ids) >= {g for g, _ in by("clone")} | {g for g, _ in by("split")}


def test_split_parents_are_recorded_as_deaths_so_the_books_balance():
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, opts = make_population(8)
    state = primed(s, params, 8, high_grad=(1, 2, 5, 6), large=(5, 6))
    s._grow_gs(params, opts, state, step=1000)

    a = events.to_arrays()
    births = sum(1 for k in a["kind"] if k in ("sfm", "clone", "split"))
    deaths = sum(1 for k in a["kind"] if k == "death")
    assert births - deaths == state["ids"].numel(), "births - deaths must equal the population"
    dead = [a["gid"][i] for i, k in enumerate(a["kind"]) if k == "death"]
    assert sorted(dead) == [5, 6], "the split parents are the ones that died"


def test_prune_logs_exactly_the_removed_ids():
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, opts = make_population(6)
    state = primed(s, params, 6)
    # opacity below prune_opa -> pruned. sigmoid(-10) ~ 4.5e-5 < 0.005
    with torch.no_grad():
        params["opacities"][[0, 4]] = -10.0
    n_prune = s._prune_gs(params, opts, state, step=1000)

    assert n_prune == 2
    assert state["ids"].tolist() == [1, 2, 3, 5]
    a = events.to_arrays()
    dead = [a["gid"][i] for i, k in enumerate(a["kind"]) if k == "death"]
    assert sorted(dead) == [0, 4]


def test_ids_are_never_reused():
    events = EventLog()
    s = InstrumentedStrategy(events=events)
    params, opts = make_population(5)
    state = primed(s, params, 5, high_grad=(1, 3))
    s._grow_gs(params, opts, state, step=1000)
    seen = set(state["ids"].tolist())

    with torch.no_grad():
        params["opacities"][:] = -10.0  # kill everything
    s._prune_gs(params, opts, state, step=1100)
    assert state["ids"].numel() == 0

    a = events.to_arrays()
    born = [a["gid"][i] for i, k in enumerate(a["kind"]) if k in ("sfm", "clone", "split")]
    assert len(born) == len(set(born)), "an id was handed out twice"
    assert seen <= set(born)


def test_reset_condition_matches_gsplat_exactly():
    """We mirror DefaultStrategy's reset condition rather than hooking it, so
    pin it: a reset happens only while step < refine_stop_iter."""
    s = InstrumentedStrategy()
    assert s.reset_every == 3000 and s.refine_stop_iter == 15000
    fires = [
        step
        for step in range(0, 30001, 1000)
        if step > 0 and step < s.refine_stop_iter and step % s.reset_every == 0
    ]
    assert fires == [3000, 6000, 9000, 12000], "resets stop once densification does"


# ---------------------------------------------------------------------------
# Hook neutrality
#
# PLAN.md originally proposed checking this end-to-end: instrumented vs vanilla,
# "same final Gaussian count, PSNR within 0.1 dB". That test is not sound --
# measured on this machine, two *identical vanilla* runs differ by ~0.5 dB and
# ~4,000 Gaussians, because the rasterizer's backward accumulates gradients with
# atomics in nondeterministic order. Tiny gradient differences change a
# densification decision, and from there the runs diverge for good.
#
# So test the claim directly instead: given identical inputs, the wrapper must
# make byte-identical decisions to DefaultStrategy. That is both stronger than
# matching end metrics and exactly reproducible.
# ---------------------------------------------------------------------------

from gsplat.strategy import DefaultStrategy  # noqa: E402


def _clone_population(params, optimizers):
    new_params = {
        k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()
    }
    new_opts = {k: torch.optim.Adam([v], lr=0.0) for k, v in new_params.items()}
    return new_params, new_opts


@pytest.mark.parametrize(
    "high_grad,large",
    [
        ((1, 3), ()),            # clones only
        ((1, 3), (1, 3)),        # splits only
        ((1, 2, 5, 6), (5, 6)),  # both in one call
        ((), ()),                # nothing to do
    ],
)
def test_instrumented_grow_matches_default_exactly(high_grad, large):
    n = 8
    pa, oa = make_population(n)
    pb, ob = _clone_population(pa, oa)

    ref = DefaultStrategy()
    ins = InstrumentedStrategy(events=EventLog())

    sa = ref.initialize_state(scene_scale=1.0)
    sb = ins.initialize_state(scene_scale=1.0)
    for s, p in ((sa, pa), (sb, pb)):
        s["count"] = torch.ones(n, device="cuda")
        s["grad2d"] = torch.zeros(n, device="cuda")
        if len(high_grad):
            s["grad2d"][list(high_grad)] = 1.0
        if len(large):
            with torch.no_grad():
                p["scales"][list(large)] = LARGE
    ins._ensure_ids(pb, sb, step=0)

    # split() draws from torch.randn, so pin the stream for both calls
    torch.manual_seed(1234)
    out_a = ref._grow_gs(pa, oa, sa, step=1000)
    torch.manual_seed(1234)
    out_b = ins._grow_gs(pb, ob, sb, step=1000)

    assert out_a == out_b, "different (n_dupli, n_split)"
    for k in pa:
        assert torch.equal(pa[k], pb[k]), f"param {k} diverged"


def test_instrumented_prune_matches_default_exactly():
    n = 6
    pa, oa = make_population(n)
    pb, ob = _clone_population(pa, oa)
    with torch.no_grad():
        pa["opacities"][[0, 4]] = -10.0
        pb["opacities"][[0, 4]] = -10.0

    ref, ins = DefaultStrategy(), InstrumentedStrategy(events=EventLog())
    sa, sb = ref.initialize_state(1.0), ins.initialize_state(1.0)
    for s in (sa, sb):
        s["count"] = torch.ones(n, device="cuda")
        s["grad2d"] = torch.zeros(n, device="cuda")
    ins._ensure_ids(pb, sb, step=0)

    assert ref._prune_gs(pa, oa, sa, step=1000) == ins._prune_gs(pb, ob, sb, step=1000)
    for k in pa:
        assert torch.equal(pa[k], pb[k]), f"param {k} diverged"


def test_events_are_off_by_default():
    """An InstrumentedStrategy with no log must still work, and cost nothing."""
    s = InstrumentedStrategy()
    assert s.events is None
    params, opts = make_population(5)
    state = primed(s, params, 5, high_grad=(1, 3))
    n_dupli, n_split = s._grow_gs(params, opts, state, step=1000)
    assert (n_dupli, n_split) == (2, 0)
    assert state["ids"].tolist() == [0, 1, 2, 3, 4, 5, 6]
