"""Stage-driver tests, plus a structural check on the vendored trainer patch.

Two different things, deliberately together: what the driver records, and
whether the four marked edits are still where the driver expects them.

The structural half exists because of a real mistake. Inserting a ``def`` into
the middle of ``Runner.__init__`` is valid Python -- the rest of ``__init__``
simply becomes unreachable code in the new method's body -- and the file still
imports, still type-checks, and fails a thousand steps later with an
``AttributeError`` about something unrelated. An AST check is the cheap guard,
and it doubles as the tripwire for an upstream rename.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from splat.curriculum import Covisibility, Curriculum, CurriculumConfig
from splat.paths import repo_root

torch = pytest.importorskip("torch")

from splat.stages import ActiveSampler, StageDriver  # noqa: E402

TRAINER = repo_root() / "third_party" / "gsplat_examples" / "simple_trainer.py"

VIS = np.array(
    [
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ],
    dtype=bool,
)


@pytest.fixture
def plan() -> Curriculum:
    return Curriculum(
        Covisibility(VIS), max_steps=100, cfg=CurriculumConfig(warmup_steps=10, stage_steps=10)
    )


class FakeParser:
    def __init__(self, n_points: int = 8, n_images: int = 5) -> None:
        rng = np.random.default_rng(0)
        self.points = rng.normal(size=(n_points, 3)).astype(np.float32)
        self.points_rgb = (rng.random((n_points, 3)) * 255).astype(np.uint8)
        self.image_names = [f"IMG_{i:04d}.jpeg" for i in range(n_images)]


class FakeTrainset:
    """Just enough of gsplat's Dataset for the driver: indices and items."""

    def __init__(self, n: int = 5) -> None:
        self.indices = np.arange(n)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return {
            "image": torch.full((4, 6, 3), 128.0),
            "K": torch.eye(3),
            "camtoworld": torch.eye(4),
            "camera_idx": 0,
        }


def make_driver(tmp_path, plan, **kw) -> StageDriver:
    return StageDriver(
        plan=plan,
        trainset=FakeTrainset(),
        parser=FakeParser(),
        out_dir=tmp_path,
        render_fn=lambda data: torch.zeros(1, 4, 6, 3),
        device="cpu",
        **kw,
    )


# -- the sampler (hook 3) --------------------------------------------------


def test_sampler_draws_only_from_active_images(plan):
    sampler = ActiveSampler(plan)
    it = iter(sampler)
    drawn = {next(it) for _ in range(60)}
    assert drawn == {0, 1, 2}, "only the initial three are active"


def test_sampler_picks_up_a_stage_advance(plan):
    sampler = ActiveSampler(plan)
    it = iter(sampler)
    [next(it) for _ in range(30)]
    plan.on_step(10)  # stage 1 activates image 3
    drawn = {next(it) for _ in range(60)}
    assert 3 in drawn


def test_sampler_never_stops(plan):
    """A finite sampler would make one epoch three steps long at the start."""
    it = iter(ActiveSampler(plan))
    assert len({next(it) for _ in range(500)}) <= 3


def test_sampler_covers_the_active_set_evenly(plan):
    """Shuffled passes, not sampling with replacement: over 3n draws every
    active image appears exactly n times."""
    from collections import Counter

    it = iter(ActiveSampler(plan))
    counts = Counter(next(it) for _ in range(300))
    assert set(counts) == {0, 1, 2}
    assert set(counts.values()) == {100}


# -- the records -----------------------------------------------------------


def test_eval_steps_are_the_stage_boundaries(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    # 5 images: a 3-image start plus 2 added, so 2 boundaries
    assert driver.eval_steps == [10, 20] == plan.boundaries


def test_a_stage_boundary_opens_a_record_and_closes_the_previous(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    params = {"means": torch.zeros(100, 3)}
    assert driver.on_step(5, params, {}, {}) is None, "no stage begins at 5"
    assert driver.records == []

    stage = driver.on_step(10, params, {}, {})
    assert stage is not None and stage.index == 1
    assert [r.stage for r in driver.records] == [0], "stage 0 closed on the way past"

    driver.on_step(20, params, {}, {})
    assert [r.stage for r in driver.records] == [0, 1]


def test_stage_zero_is_recorded_with_the_initial_population(tmp_path, plan):
    """Otherwise '3 images' is a gap at the origin of the test-PSNR curve."""
    driver = make_driver(tmp_path, plan)
    driver.on_step(10, {"means": torch.zeros(50, 3)}, {}, {})
    zero = driver.records[0]
    assert zero.stage == 0 and zero.step == 0 and zero.added is None
    assert zero.n_images == 3
    assert zero.n_gaussians_before == int(plan.initial_mask().sum())
    assert zero.n_points_unlocked == int(plan.initial_mask().sum())


def test_records_carry_the_photo_name_and_population_change(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    driver.on_step(10, {"means": torch.zeros(50, 3)}, {}, {})
    driver.on_step(20, {"means": torch.zeros(90, 3)}, {}, {})
    rec = driver.records[1]
    assert rec.stage == 1
    assert rec.added == 3
    assert rec.image == "IMG_0003.jpeg"
    assert rec.n_images == 4
    assert (rec.n_gaussians_before, rec.n_gaussians_after) == (50, 90)
    assert rec.end_step == 20


def test_before_and_after_psnr_are_both_measured(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    driver.on_step(10, {"means": torch.zeros(50, 3)}, {}, {})
    driver.on_step(20, {"means": torch.zeros(50, 3)}, {}, {})
    rec = driver.records[1]
    assert rec.psnr_before is not None and rec.psnr_after is not None
    assert rec.panel == "stages/stage_001_img_003.png"
    assert (Path(tmp_path) / rec.panel).exists()


def test_the_panel_is_five_views_wide(tmp_path, plan):
    import imageio.v2 as imageio

    driver = make_driver(tmp_path, plan)
    driver.on_step(10, {"means": torch.zeros(50, 3)}, {}, {})
    driver.on_step(20, {"means": torch.zeros(50, 3)}, {}, {})
    panel = imageio.imread(Path(tmp_path) / driver.records[1].panel)
    assert panel.shape == (4, 6 * 5, 3), "GT | before | after | err before | err after"


def test_finish_closes_the_last_stage_and_writes_json(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    for step in (10, 20):
        driver.on_step(step, {"means": torch.zeros(50, 3)}, {}, {})
    path = driver.finish(99, {"means": torch.zeros(77, 3)})
    assert path.name == "stages.json"

    d = json.loads(path.read_text())
    assert [r["stage"] for r in d["stages"]] == [0, 1, 2], "every stage, last included"
    assert d["stages"][-1]["n_gaussians_after"] == 77
    assert d["stages"][-1]["end_step"] == 99
    assert d["order"] == list(plan.order)
    assert d["stage_steps"] == 10


def test_a_stage_logs_a_stage_event(tmp_path, plan):
    from splat.events import EventLog

    log = EventLog()
    driver = make_driver(tmp_path, plan, events=log)
    driver.on_step(10, {"means": torch.zeros(50, 3)}, {}, {})
    a = log.to_arrays()
    assert (a["kind"] == "stage").sum() == 1
    assert a["step"][a["kind"] == "stage"].tolist() == [10]
    assert a["gid"][a["kind"] == "stage"].tolist() == [3], "the image added, not a Gaussian"


def test_no_renderer_still_records_counts(tmp_path, plan):
    """A run that cannot render still gets its stage bookkeeping."""
    driver = StageDriver(
        plan=plan, trainset=FakeTrainset(), parser=FakeParser(),
        out_dir=tmp_path, render_fn=None, device="cpu",
    )
    driver.on_step(10, {"means": torch.zeros(50, 3)}, {}, {})
    driver.finish(20, {"means": torch.zeros(60, 3)})
    d = json.loads((Path(tmp_path) / "stages.json").read_text())
    assert d["stages"][1]["psnr_before"] is None
    assert d["stages"][1]["panel"] is None
    assert d["stages"][1]["n_gaussians_after"] == 60


# -- the vendored trainer patch -------------------------------------------


def trainer_tree() -> ast.Module:
    return ast.parse(TRAINER.read_text())


def runner_class() -> ast.ClassDef:
    return next(
        n for n in trainer_tree().body if isinstance(n, ast.ClassDef) and n.name == "Runner"
    )


def test_all_four_hooks_are_still_marked():
    text = TRAINER.read_text()
    for n in (1, 2, 3, 4):
        assert f"[splat] hook {n}" in text, f"hook {n} marker is gone"


def test_the_render_helper_is_a_method_and_not_nested_in_init():
    """The mistake this file exists for: a def inside __init__ silently makes
    the rest of __init__ dead code in its body."""
    runner = runner_class()
    methods = {n.name: n for n in runner.body if isinstance(n, ast.FunctionDef)}
    assert "_render_train_view" in methods
    init, helper = methods["__init__"], methods["_render_train_view"]
    assert helper.lineno > init.end_lineno, "must be a sibling of __init__"
    assert len(helper.body) <= 5, f"absorbed {len(helper.body)} statements from __init__"


def test_init_still_sets_up_everything_after_the_hooks():
    """Attributes assigned late in __init__, which a stray def would orphan."""
    init = next(
        n for n in runner_class().body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    def attrs(target):
        # self.splats, self.optimizers = ... is a Tuple target, not an Attribute
        if isinstance(target, ast.Attribute):
            return [target.attr]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [e.attr for e in target.elts if isinstance(e, ast.Attribute)]
        return []

    assigned = {
        name
        for node in ast.walk(init)
        for t in getattr(node, "targets", [])
        for name in attrs(t)
    }
    for name in ("curriculum", "stage_driver", "splats", "optimizers", "strategy_state",
                 "_gaussians_frozen", "pose_optimizers"):
        assert name in assigned, f"self.{name} is no longer assigned in __init__"


def test_config_exposes_the_curriculum_flags():
    cfg = next(
        n for n in trainer_tree().body if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    fields = {n.target.id for n in cfg.body if isinstance(n, ast.AnnAssign)}
    for name in ("curriculum", "curriculum_init_images", "curriculum_warmup",
                 "curriculum_stage_steps", "curriculum_min_spacing",
                 "curriculum_max_seeds", "curriculum_eval_stages", "lineage",
                 "snapshot_every"):
        assert name in fields, f"Config.{name} is gone"


def test_the_init_hook_reaches_the_splat_builder():
    """Hook 2 is only a hook if create_splats_with_optimizers accepts it."""
    fn = next(
        n for n in trainer_tree().body
        if isinstance(n, ast.FunctionDef) and n.name == "create_splats_with_optimizers"
    )
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "point_mask" in args
    assert "point_mask=point_mask" in TRAINER.read_text(), "and is actually passed"


# -- what the stage changed, per Gaussian ---------------------------------


def model(n: int, offset: float = 0.0, ids: Optional[list] = None):
    """A params/state pair in the shape the driver diffs."""
    means = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + offset
    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.full((n, 3), -2.0 + offset)),
        "opacities": torch.nn.Parameter(torch.zeros(n) + offset),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3) + offset),
    }
    state = {"ids": torch.tensor(ids if ids is not None else list(range(n)))}
    return params, state


def test_begin_opens_stage_zero_with_a_snapshot(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    params, state = model(10)
    driver.begin(0, params, state)
    assert driver._open is not None and driver._open.stage == 0
    assert driver._opened_at is not None
    assert driver._opened_at["ids"].tolist() == list(range(10))


def test_begin_is_idempotent(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    params, state = model(10)
    driver.begin(0, params, state)
    first = driver._opened_at
    driver.begin(0, *model(99))
    assert driver._opened_at is first, "a second call must not reopen the stage"


def test_deltas_are_matched_by_id_and_not_by_row(tmp_path, plan):
    """Densification reorders and prunes constantly, so row i at the start of a
    stage and row i at the end are unrelated Gaussians."""
    driver = make_driver(tmp_path, plan)
    driver.begin(0, *model(4, ids=[10, 11, 12, 13]))
    # id 11 died, ids 90/91 were born, and the survivors are in a new order
    params, state = model(4, offset=0.0, ids=[13, 90, 10, 91])
    params["means"].data = torch.tensor(
        [[9.0, 10.0, 11.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]
    )
    driver.on_step(10, params, {}, state)
    rec = driver.records[0]
    assert (rec.n_survived, rec.n_born, rec.n_died) == (2, 2, 2)
    # ids 10 and 13 kept their exact start positions, so nothing moved
    assert rec.d_means_mean == pytest.approx(0.0, abs=1e-5)


def test_churn_counts_balance(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    driver.begin(0, *model(6, ids=[0, 1, 2, 3, 4, 5]))
    params, state = model(7, ids=[0, 2, 4, 100, 101, 102, 103])
    driver.on_step(10, params, {}, state)
    rec = driver.records[0]
    assert rec.n_survived + rec.n_died == rec.n_gaussians_before
    assert rec.n_survived + rec.n_born == rec.n_gaussians_after


def test_movement_is_measured_in_world_units(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    driver.begin(0, *model(3))
    params, state = model(3)
    params["means"].data = params["means"].data + torch.tensor([3.0, 4.0, 0.0])
    driver.on_step(10, params, {}, state)
    assert driver.records[0].d_means_mean == pytest.approx(5.0, rel=1e-3), "3-4-5"


def test_opacity_change_is_in_probability_space(tmp_path, plan):
    """"went from 0.5 to 0.73" rather than a logit gap of 1.0."""
    driver = make_driver(tmp_path, plan)
    driver.begin(0, *model(3))  # opacities 0.0 in logit space -> 0.5
    params, state = model(3)
    params["opacities"].data = torch.ones(3)  # sigmoid(1) = 0.7311
    driver.on_step(10, params, {}, state)
    assert driver.records[0].d_opacity_mean == pytest.approx(0.2311, abs=1e-3)


def test_the_delta_file_has_one_row_per_survivor(tmp_path, plan):
    driver = make_driver(tmp_path, plan)
    driver.begin(0, *model(5, ids=[0, 1, 2, 3, 4]))
    params, state = model(3, ids=[1, 3, 4])
    driver.on_step(10, params, {}, state)
    rec = driver.records[0]
    assert rec.deltas == "deltas/stage_000.npz"
    with np.load(Path(tmp_path) / rec.deltas) as z:
        assert z["ids"].tolist() == [1, 3, 4]
        for key in ("d_means", "d_scale", "d_opacity", "d_color"):
            assert z[key].shape == (3,), key
            assert z[key].dtype == np.float16, key
        assert int(z["stage"]) == 0


def test_a_stage_with_no_survivors_records_no_deltas(tmp_path, plan):
    """Everything alive at the start was pruned; nothing to diff."""
    driver = make_driver(tmp_path, plan)
    driver.begin(0, *model(3, ids=[0, 1, 2]))
    params, state = model(2, ids=[50, 51])
    driver.on_step(10, params, {}, state)
    rec = driver.records[0]
    assert (rec.n_survived, rec.n_born, rec.n_died) == (0, 2, 3)
    assert rec.deltas is None and rec.d_means_mean is None


def test_no_ids_means_no_deltas_but_still_a_record(tmp_path, plan):
    """A run without --lineage has no id vector to match on."""
    driver = make_driver(tmp_path, plan)
    params, _ = model(4)
    driver.begin(0, params, {})
    driver.on_step(10, params, {}, {})
    rec = driver.records[0]
    assert rec.deltas is None
    assert rec.n_survived == 0
    assert rec.n_gaussians_before == 4


# -- exposure, recorded either way ----------------------------------------


def test_stages_json_records_the_exposure_spread(tmp_path, plan):
    """Recorded in every run, not just the fair ones: a reader who cannot see
    the spread will over-trust the per-photo gains."""
    driver = make_driver(tmp_path, plan)
    params, state = model(4)
    driver.begin(0, params, state)
    for step in (10, 20):
        driver.on_step(step, params, {}, state)
    d = json.loads(driver.finish(99, params, state).read_text())
    assert d["mode"] == "incremental"
    assert set(map(int, d["exposure"])) == set(plan.order)
    # stored rounded to 3 decimals, so compare at that precision
    assert d["exposure_spread"] == pytest.approx(plan.exposure_spread(), abs=5e-4)
    assert d["exposure_spread"] > 1.0, "incremental is never fair"
    assert d["images"] == list(plan.images)


def test_groups_mode_records_its_shape(tmp_path):
    from splat.curriculum import Covisibility

    plan = Curriculum(
        Covisibility(VIS), max_steps=600,
        cfg=CurriculumConfig(mode="groups", group_size=2, rounds=2),
    )
    driver = make_driver(tmp_path, plan)
    params, state = model(4)
    driver.begin(0, params, state)
    driver.on_step(plan.stages[1].step, params, {}, state)
    d = json.loads(driver.finish(599, params, state).read_text())
    assert d["mode"] == "groups"
    assert (d["group_size"], d["rounds"]) == (2, 2)
    assert d["exposure_spread"] == pytest.approx(1.0)
    first = d["stages"][0]
    assert first["group"] == 0 and first["round"] == 0
    assert first["n_active"] == 2 and first["n_images"] == 2
    assert first["active"] == list(plan.stages[0].active)
    assert first["group_images"] is not None


def test_records_carry_the_measurement_window(tmp_path, plan):
    """Two stages' gains are only comparable if this number matches."""
    driver = make_driver(tmp_path, plan)
    params, state = model(4)
    driver.begin(0, params, state)
    driver.on_step(10, params, {}, state)
    rec = driver.records[0]
    assert rec.length == plan.stages[0].length
    assert rec.steps_per_active_image == pytest.approx(
        plan.stages[0].steps_per_active_image, rel=1e-3
    )
