"""Append tests: four things have to stay the same length, forever.

Driven against the real ``gsplat.strategy.ops`` helper, the real ``Adam``, the
real ``GaussianScene`` and the real ``InstrumentedStrategy`` -- never a mock.
What is being tested is our reading of gsplat's internal bookkeeping, so a mock
would only confirm that we agree with ourselves. CPU throughout: ``knn`` goes
through sklearn and the ops are plain tensor concatenation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from gsplat.scene import GaussianScene  # noqa: E402

from splat.append import SEED_FIELDS, append_gaussians, seed_params  # noqa: E402
from splat.events import EventLog  # noqa: E402
from splat.lineage import InstrumentedStrategy  # noqa: E402

DEVICE = "cpu"


def cloud(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, 3)).astype(np.float32),
        rng.random((n, 3)).astype(np.float32),
    )


def make_model(n: int = 40):
    """A trainable model in the shape the trainer builds: scene, params, Adam."""
    points, rgbs = cloud(n + 20)
    rows = seed_params(points, rgbs, select=np.arange(n), device=DEVICE)
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in rows.items()})
    scene = GaussianScene.from_splats(splats, id="scene")
    splats = scene.splats
    opts = {
        k: torch.optim.Adam([{"params": splats[k], "lr": 1e-3, "name": k}], eps=1e-15)
        for k in splats
    }
    return points, rgbs, splats, scene, opts


def step_optimizers(splats, opts, scale: float = 1e-3):
    for k in splats:
        splats[k].grad = torch.randn_like(splats[k]) * scale
    for o in opts.values():
        o.step()


def moments(opts, name: str):
    return list(opts[name].state.values())[0]


def live_state(n: int):
    """The strategy state as it looks after the first real step."""
    return {
        "grad2d": torch.zeros(n),
        "count": torch.zeros(n),
        "scene_scale": 1.0,
        "ids": torch.arange(n, dtype=torch.int64),
    }


# -- seed rows -------------------------------------------------------------


def test_seed_params_builds_every_field():
    points, rgbs = cloud(30)
    rows = seed_params(points, rgbs, select=np.arange(5), device=DEVICE)
    assert set(rows) == set(SEED_FIELDS)
    assert rows["means"].shape == (5, 3)
    assert rows["scales"].shape == (5, 3)
    assert rows["quats"].shape == (5, 4)
    assert rows["opacities"].shape == (5,)
    assert rows["sh0"].shape == (5, 1, 3)
    assert rows["shN"].shape == (5, 15, 3), "sh_degree 3 -> 16 bands, 1 DC + 15"


def test_seed_opacity_and_colour_are_in_raw_space():
    """The trainer holds logit opacities and SH coefficients, not probabilities
    and RGB. A seed injected in activated space is instantly opaque and wrong."""
    points, rgbs = cloud(10)
    rows = seed_params(points, rgbs, init_opacity=0.1, device=DEVICE)
    np.testing.assert_allclose(
        rows["opacities"].sigmoid().numpy(), 0.1, atol=1e-5
    )
    C0 = 0.28209479177387814
    np.testing.assert_allclose(
        rows["sh0"][:, 0, :].numpy(), (rgbs - 0.5) / C0, rtol=1e-5
    )
    assert rows["shN"].abs().max() == 0, "higher bands start at zero, as at init"


def test_seed_scales_come_from_the_whole_cloud_not_the_selection():
    """A seed's size should say how dense the reconstruction is around it, not
    how many of its neighbours happen to be triangulable this stage."""
    points, rgbs = cloud(400)
    sel = np.arange(6)
    whole = seed_params(points, rgbs, select=sel, device=DEVICE)["scales"]
    alone = seed_params(points[sel], rgbs[sel], device=DEVICE)["scales"]
    assert not torch.allclose(whole, alone)
    assert (whole < alone).all(), "six points on their own look far apart"


def test_seed_scales_survive_a_cloud_smaller_than_k():
    """A stage boundary is the wrong place for sklearn to raise."""
    points, rgbs = cloud(3)
    assert seed_params(points, rgbs, device=DEVICE)["scales"].shape == (3, 3)
    with pytest.raises(ValueError, match="at least 2 points"):
        seed_params(points[:1], rgbs[:1], device=DEVICE)


def test_seed_params_accepts_0_255_colours():
    points, rgbs = cloud(10)
    a = seed_params(points, rgbs, device=DEVICE)["sh0"]
    b = seed_params(points, (rgbs * 255).astype(np.float32), device=DEVICE)["sh0"]
    torch.testing.assert_close(a, b)


# -- the append ------------------------------------------------------------


def test_every_param_adam_moment_and_state_tensor_grows_together():
    points, rgbs, splats, scene, opts = make_model(40)
    step_optimizers(splats, opts)  # so the Adam moments exist
    state = live_state(40)

    new = seed_params(points, rgbs, select=np.arange(40, 55), device=DEVICE)
    m = append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(40, 55))

    assert m == 15
    for k in splats:
        assert splats[k].shape[0] == 55, k
        assert moments(opts, k)["exp_avg"].shape[0] == 55, k
        assert moments(opts, k)["exp_avg_sq"].shape[0] == 55, k
    for k in ("grad2d", "count", "ids"):
        assert state[k].shape[0] == 55, k
    assert scene.component_index.shape == (55,)
    scene.validate()


def test_an_optimizer_step_after_append_succeeds():
    """The check the plan asks for: the moments must line up with the grads."""
    points, rgbs, splats, scene, opts = make_model(40)
    step_optimizers(splats, opts)
    state = live_state(40)
    new = seed_params(points, rgbs, select=np.arange(40, 50), device=DEVICE)
    append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(40, 50))
    before = splats["means"].detach().clone()
    step_optimizers(splats, opts)
    assert not torch.allclose(before, splats["means"].detach())


def test_new_rows_start_with_zero_adam_moments():
    points, rgbs, splats, scene, opts = make_model(20)
    step_optimizers(splats, opts, scale=1.0)  # large moments on the originals
    state = live_state(20)
    new = seed_params(points, rgbs, select=np.arange(20, 25), device=DEVICE)
    append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(20, 25))
    for k in splats:
        assert moments(opts, k)["exp_avg"][20:].abs().max() == 0, k
        assert moments(opts, k)["exp_avg"][:20].abs().max() > 0, f"{k}: originals kept"


def test_running_state_is_zeroed_not_copied():
    """The one place this deliberately differs from ops.duplicate. A seed has
    never been rendered; inheriting someone's grad2d would make it eligible for
    splitting at the next refinement on borrowed evidence."""
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    state["grad2d"] += 7.0
    state["count"] += 3.0
    new = seed_params(points, rgbs, select=np.arange(20, 24), device=DEVICE)
    append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(20, 24))
    assert state["grad2d"][20:].abs().max() == 0
    assert state["count"][20:].abs().max() == 0
    assert (state["grad2d"][:20] == 7.0).all(), "and the originals keep theirs"


def test_ids_are_identity_and_are_carried_not_zeroed():
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    new = seed_params(points, rgbs, select=np.arange(20, 23), device=DEVICE)
    append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.tensor([99, 100, 101]))
    assert state["ids"][-3:].tolist() == [99, 100, 101]
    assert len(set(state["ids"].tolist())) == 23


def test_the_appended_values_are_the_ones_supplied():
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    new = seed_params(points, rgbs, select=np.arange(20, 26), device=DEVICE)
    append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(20, 26))
    torch.testing.assert_close(splats["means"][20:].detach(), new["means"])
    np.testing.assert_allclose(splats["means"][20:].detach().numpy(), points[20:26], atol=1e-5)


def test_state_entries_that_are_not_tensors_are_left_alone():
    """grad2d/count are None until the first step, and scene_scale is a float."""
    points, rgbs, splats, scene, opts = make_model(20)
    state = {"grad2d": None, "count": None, "scene_scale": 2.5, "ids": torch.arange(20)}
    new = seed_params(points, rgbs, select=np.arange(20, 22), device=DEVICE)
    append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(20, 22))
    assert state["grad2d"] is None and state["count"] is None
    assert state["scene_scale"] == 2.5
    assert state["ids"].shape[0] == 22


def test_appending_nothing_is_a_no_op():
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    empty = {k: v[:0] for k, v in seed_params(points, rgbs, select=np.arange(1), device=DEVICE).items()}
    assert append_gaussians(splats, opts, state, empty, scene=scene, new_ids=torch.arange(0)) == 0
    assert splats["means"].shape[0] == 20
    assert scene.component_index.shape == (20,)


# -- refusals --------------------------------------------------------------


def test_a_missing_param_is_refused():
    """Appending zeros for shN would train, converge, and be wrong."""
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    new = seed_params(points, rgbs, select=np.arange(20, 22), device=DEVICE)
    del new["shN"]
    with pytest.raises(ValueError, match="no rows supplied for \\['shN'\\]"):
        append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(2))


def test_a_wrong_trailing_shape_is_refused():
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    new = seed_params(points, rgbs, select=np.arange(20, 22), device=DEVICE)
    new["sh0"] = torch.zeros(2, 4, 3)
    with pytest.raises(ValueError, match="trailing shape"):
        append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(2))


def test_ids_in_state_without_new_ids_is_refused():
    points, rgbs, splats, scene, opts = make_model(20)
    state = live_state(20)
    new = seed_params(points, rgbs, select=np.arange(20, 22), device=DEVICE)
    with pytest.raises(ValueError, match="new_ids"):
        append_gaussians(splats, opts, state, new, scene=scene)


def test_a_multi_component_scene_is_refused():
    """Row 0's component is the right answer only while there is one."""
    points, rgbs, splats, scene, opts = make_model(20)
    extra = seed_params(points, rgbs, select=np.arange(5), device=DEVICE)
    scene.put("other", torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in extra.items()}))
    state = live_state(25)
    new = seed_params(points, rgbs, select=np.arange(20, 22), device=DEVICE)
    with pytest.raises(NotImplementedError, match="2 components"):
        append_gaussians(splats, opts, state, new, scene=scene, new_ids=torch.arange(2))


# -- with the real strategy ------------------------------------------------


def test_append_seeds_allocates_from_the_strategy_id_counter():
    """Two Gaussians sharing one id merges two lineages into a plausible one."""
    points, rgbs, splats, scene, opts = make_model(30)
    log = EventLog()
    strat = InstrumentedStrategy(verbose=False, events=log)
    state = strat.initialize_state(scene_scale=1.0)

    new = seed_params(points, rgbs, select=np.arange(30, 34), device=DEVICE)
    ids = strat.append_seeds(splats, opts, state, new, step=700, scene=scene)

    assert ids.tolist() == [30, 31, 32, 33], "fresh, after the 30 sfm ids"
    again = strat.append_seeds(
        splats, opts, state, seed_params(points, rgbs, select=np.arange(34, 36), device=DEVICE),
        step=800, scene=scene,
    )
    assert again.tolist() == [34, 35], "the counter does not restart"
    assert len(set(state["ids"].tolist())) == 36


def test_append_seeds_logs_seed_births():
    points, rgbs, splats, scene, opts = make_model(30)
    log = EventLog()
    strat = InstrumentedStrategy(verbose=False, events=log)
    state = strat.initialize_state(scene_scale=1.0)
    strat.append_seeds(
        splats, opts, state, seed_params(points, rgbs, select=np.arange(30, 33), device=DEVICE),
        step=700, scene=scene,
    )
    a = log.to_arrays()
    seeds = a["kind"] == "seed"
    assert seeds.sum() == 3
    assert a["step"][seeds].tolist() == [700] * 3
    assert a["gid"][seeds].tolist() == [30, 31, 32]
    assert a["parent"][seeds].tolist() == [-1] * 3, "a seed comes from SfM, not a parent"
    assert (a["kind"] == "sfm").sum() == 30, "ids were initialised on the way in"


def test_append_then_densify_keeps_everything_aligned():
    """The integration that matters: ops re-index every state tensor after an
    append, so a length mismatch surfaces here rather than at append time."""
    points, rgbs, splats, scene, opts = make_model(60)
    log = EventLog()
    strat = InstrumentedStrategy(verbose=False, events=log)
    state = strat.initialize_state(scene_scale=1.0)
    strat._ensure_ids(splats, state, 0)
    state["grad2d"] = torch.zeros(60)
    state["count"] = torch.ones(60)
    step_optimizers(splats, opts)

    strat.append_seeds(
        splats, opts, state,
        seed_params(points, rgbs, select=np.arange(60, 75), device=DEVICE),
        step=700, scene=scene,
    )
    assert splats["means"].shape[0] == 75

    # now make half of them look worth growing, and run the real op
    state["grad2d"] = torch.full((75,), 1e9)
    state["count"] = torch.ones(75)
    n_dupli, n_split = strat._grow_gs(splats, opts, state, step=800, scene=scene)
    assert n_dupli + n_split > 0, "the real strategy has to actually do something"

    n = splats["means"].shape[0]
    for k in splats:
        assert splats[k].shape[0] == n, k
        assert moments(opts, k)["exp_avg"].shape[0] == n, k
    assert state["ids"].shape[0] == n
    assert state["grad2d"].shape[0] == n
    assert len(set(state["ids"].tolist())) == n, "ids stay unique through growth"
    step_optimizers(splats, opts)
