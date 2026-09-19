"""Curriculum tests: the ordering, the schedule, and what each stage unlocks.

A hand-built covisibility matrix, so every expected answer is countable by eye.
The ordering is the part worth pinning hardest: it decides what the model ever
sees, it is easy to get subtly wrong (numpy's ``bool @ bool`` is a *logical*
matmul, which ranks by "shares any point" and reads as plausible), and nothing
downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from splat.curriculum import (
    Covisibility,
    Curriculum,
    CurriculumConfig,
    Stage,
    far_from_existing,
)

# Six images over ten points. Images 0, 1 and 2 overlap heavily on points 0-3;
# 3 and 4 hang off the far end; 5 shares a single point with 0.
#            point: 0  1  2  3  4  5  6  7  8  9
VIS = np.array(
    [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 0
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 1
        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],  # 3
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # 4
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 5
    ],
    dtype=bool,
)


@pytest.fixture
def covis() -> Covisibility:
    return Covisibility(VIS)


# -- covisibility ----------------------------------------------------------


def test_counts_are_points_per_image(covis):
    assert covis.counts().tolist() == [4, 5, 5, 4, 3, 2]
    assert (covis.n_items, covis.n_points) == (6, 10)


def test_pair_counts_count_rather_than_test_for_overlap(covis):
    """`bool @ bool` would give True/False here and rank everything equal."""
    pairs = covis.pair_counts()
    assert pairs[0, 1] == 4, "images 0 and 1 share points 0,1,2,3"
    assert pairs[0, 2] == 3, "0 and 2 share points 0,1,2"
    assert pairs[0, 5] == 1, "0 and 5 share only point 0"
    assert pairs[0, 4] == 0
    assert pairs.dtype == np.int64


def test_best_triple_is_the_triple_intersection(covis):
    """Not the three best pairs: 0-1 and 3-4 are both decent pairs and share
    nothing across all three."""
    assert covis.best_triple() == (0, 1, 2)
    # points 0,1,2 are in all three
    assert int((VIS[0] & VIS[1] & VIS[2]).sum()) == 3


def test_best_triple_matches_brute_force_on_random_matrices():
    import itertools

    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.random((9, 40)) < 0.3
        c = Covisibility(v)
        got = c.best_triple()
        best = max(
            itertools.combinations(range(9), 3),
            key=lambda t: int((v[t[0]] & v[t[1]] & v[t[2]]).sum()),
        )
        assert int((v[got[0]] & v[got[1]] & v[got[2]]).sum()) == int(
            (v[best[0]] & v[best[1]] & v[best[2]]).sum()
        )


def test_next_image_grows_from_the_active_union(covis):
    # active {0,1,2} covers points 0-5; image 3 shares 4,5 and image 4 shares none
    assert covis.next_image([0, 1, 2]) == 3
    assert covis.next_image([0, 1, 2, 3]) == 4, "4 shares 6,7 with 3"
    assert covis.next_image(list(range(6))) is None


def test_order_visits_every_image_exactly_once(covis):
    order = covis.order()
    assert sorted(order) == list(range(6))
    assert order[:3] == [0, 1, 2]


def test_order_is_deterministic(covis):
    assert covis.order() == covis.order()


def test_ties_go_to_the_lowest_index():
    """So the order is a function of the scene, not of numpy's argmax."""
    v = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1]], dtype=bool)
    c = Covisibility(v)
    assert c.next_image([3]) == 0, "0, 1 and 2 all share point 1 with image 3"


def test_triangulable_needs_two_views(covis):
    """A point seen by one camera is a ray, not a position."""
    assert covis.triangulable([0]).sum() == 0
    assert covis.triangulable([0, 1]).tolist() == [
        True, True, True, True, False, False, False, False, False, False
    ]
    assert covis.triangulable([]).sum() == 0


def test_newly_triangulable_is_the_difference(covis):
    new = covis.newly_triangulable([0, 1, 2], [0, 1])
    assert new.tolist() == [False] * 4 + [True, False, False, False, False, False]
    assert not (new & covis.triangulable([0, 1])).any(), "never re-unlock a point"


# -- the schedule ----------------------------------------------------------


def test_stage_zero_is_the_initial_images(covis):
    cur = Curriculum(covis, max_steps=1000)
    zero = cur.stages[0]
    assert (zero.index, zero.step, zero.added) == (0, 0, None)
    assert zero.active == (0, 1, 2)
    assert zero.seen == zero.active, "incremental: active and seen are the same set"


def test_one_stage_per_added_image(covis):
    cur = Curriculum(covis, max_steps=1000)
    assert len(cur.stages) == 4, "3 initial + 3 added"
    assert [s.added for s in cur.stages] == [None, 3, 4, 5]
    assert [s.n_images for s in cur.stages] == [3, 4, 5, 6]


def test_stages_are_paced_proportionally_by_default(covis):
    """Stage length scales with the active count, so every active image gets the
    same number of its own steps whatever the pool size. 6 images, 3 initial:
    active counts are 4, 5, 6, so e = (1000-100)/15 = 60 and the lengths are
    240, 300, 360."""
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    assert [s.length for s in cur.stages] == [100, 240, 300, 360]
    assert [s.step for s in cur.stages] == [0, 100, 340, 640]
    assert sum(s.length for s in cur.stages) == 1000, "the whole budget is used"


def test_proportional_pacing_equalises_the_measurement_window(covis):
    """The thing it exists for: an image gets the same number of its own
    gradient steps during its stage whether the pool is 4 images or 28."""
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    rates = [s.steps_per_active_image for s in cur.stages[1:]]
    assert rates == [pytest.approx(60.0)] * 3


def test_constant_length_stages_are_still_available_and_still_confounded(covis):
    """Kept for reproducing earlier runs. The window shrinks as 1/n."""
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100, stage_steps=300))
    assert [s.step for s in cur.stages] == [0, 100, 400, 700]
    rates = [s.steps_per_active_image for s in cur.stages[1:]]
    assert rates[0] > rates[-1], "the first added image gets a longer window"
    assert rates[0] / rates[-1] == pytest.approx(6 / 4)


def test_explicit_stage_steps_win(covis):
    cur = Curriculum(covis, max_steps=99999, cfg=CurriculumConfig(warmup_steps=50, stage_steps=10))
    assert [s.step for s in cur.stages] == [0, 50, 60, 70]
    assert cur.stage_steps == 10


def test_a_short_run_still_visits_every_stage(covis):
    """Rather than silently dropping the tail of the curriculum."""
    cur = Curriculum(covis, max_steps=10, cfg=CurriculumConfig(warmup_steps=100))
    assert len(cur.stages) == 4
    assert cur.stage_steps >= 1


def test_stage_at_is_the_stage_in_force(covis):
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    # boundaries at 0, 100, 340, 640 under proportional pacing
    assert cur.stage_at(0).index == 0
    assert cur.stage_at(99).index == 0
    assert cur.stage_at(100).index == 1
    assert cur.stage_at(339).index == 1
    assert cur.stage_at(340).index == 2
    assert cur.stage_at(10**9).index == 3
    assert cur.active_at(0) == (0, 1, 2)
    assert cur.active_at(100) == (0, 1, 2, 3)


def test_on_step_fires_each_stage_once_in_order(covis):
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    fired = [(step, s.index) for step in range(1000) if (s := cur.on_step(step))]
    assert fired == [(100, 1), (340, 2), (640, 3)]


def test_on_step_does_not_skip_a_stage_when_steps_are_dense(covis):
    """Two stages landing on the same step must both be applied, or an image
    joins the active set with no seeds ever injected for it."""
    cur = Curriculum(covis, max_steps=4, cfg=CurriculumConfig(warmup_steps=0, stage_steps=1))
    assert [s.step for s in cur.stages] == [0, 0, 1, 2]
    assert cur.on_step(0).index == 1, "walks past stage 0 to the last one due"
    assert cur.active == (0, 1, 2, 3)


def test_active_tracks_on_step(covis):
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    assert cur.active == (0, 1, 2)
    cur.on_step(400)
    assert cur.active == (0, 1, 2, 3, 4), "both stages due by 400"


# -- what a stage unlocks --------------------------------------------------


def test_initial_mask_is_what_three_images_can_triangulate(covis):
    cur = Curriculum(covis, max_steps=1000)
    assert cur.initial_mask().tolist() == [
        True, True, True, True, True, False, False, False, False, False
    ]


def test_seed_masks_partition_the_unlocked_points(covis):
    """Every point is unlocked by exactly one stage, or never."""
    cur = Curriculum(covis, max_steps=1000)
    masks = np.stack([cur.initial_mask()] + [cur.seed_mask(s) for s in cur.stages[1:]])
    assert (masks.sum(0) <= 1).all(), "no point unlocked twice"
    assert masks.any(0).tolist() == covis.triangulable(range(6)).tolist()


def test_a_point_seen_by_one_image_is_never_seeded(covis):
    cur = Curriculum(covis, max_steps=1000)
    masks = np.stack([cur.initial_mask()] + [cur.seed_mask(s) for s in cur.stages[1:]])
    assert not masks.any(0)[9], "point 9 is only in image 5"


def test_summary_has_a_row_per_stage(covis):
    rows = Curriculum(covis, max_steps=1000).summary()
    assert [r["stage"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["n_points_unlocked"] == 5
    assert rows[0]["added"] is None
    assert sum(r["n_points_unlocked"] for r in rows) == int(
        Covisibility(VIS).triangulable(range(6)).sum()
    )


def test_too_few_images_is_an_error():
    with pytest.raises(ValueError, match="at least 3"):
        Covisibility(VIS[:2]).best_triple()


# -- the distance filter ---------------------------------------------------


def test_far_from_existing_is_exact():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(60, 3)).astype(np.float32)
    means = torch.as_tensor(rng.normal(size=(300, 3)).astype(np.float32))
    for d in (0.1, 0.5, 2.0):
        brute = (torch.cdist(torch.as_tensor(pts), means).amin(1) >= d).numpy()
        assert (far_from_existing(pts, means, d) == brute).all(), d


def test_far_from_existing_chunks_without_changing_the_answer():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(40, 3)).astype(np.float32)
    means = torch.as_tensor(rng.normal(size=(500, 3)).astype(np.float32))
    whole = far_from_existing(pts, means, 0.4, budget=1 << 26)
    tiny = far_from_existing(pts, means, 0.4, budget=512)
    assert (whole == tiny).all()


def test_everything_is_far_from_an_empty_model():
    pts = np.zeros((5, 3), dtype=np.float32)
    assert far_from_existing(pts, None, 1.0).all()
    assert far_from_existing(pts, np.zeros((0, 3), np.float32), 1.0).all()
    assert far_from_existing(np.zeros((0, 3), np.float32), None, 1.0).shape == (0,)


# -- the ablation cap ------------------------------------------------------


def test_max_images_equal_to_init_gives_a_single_stage(covis):
    """The ablation condition: a fixed image count for a whole run, so "more
    images" is separated from "more training time"."""
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(init_images=3, max_images=3))
    assert len(cur.stages) == 1
    assert cur.boundaries == []
    assert cur.images == [0, 1, 2]


def test_max_images_truncates_the_schedule_not_the_ordering(covis):
    """N=4 must be the first four images a full run would have used, or the
    conditions are merely different rather than nested."""
    full = Curriculum(covis, max_steps=1000)
    capped = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(max_images=4))
    assert capped.order == full.order, "the ordering itself is untouched"
    assert capped.images == full.images[:4]
    assert [s.added for s in capped.stages] == [None, 3]


def test_an_uncapped_curriculum_sees_everything(covis):
    cur = Curriculum(covis, max_steps=1000)
    assert sorted(cur.images) == list(range(covis.n_items))


def test_a_cap_below_the_initial_count_is_refused(covis):
    with pytest.raises(ValueError, match="below init_images"):
        Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(init_images=5, max_images=4))


def test_a_capped_run_only_seeds_what_its_images_unlock(covis):
    """Points only two excluded photos can triangulate must stay unseeded."""
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(max_images=3))
    assert cur.initial_mask().tolist() == covis.triangulable([0, 1, 2]).tolist()
    masks = np.stack([cur.initial_mask()] + [cur.seed_mask(s) for s in cur.stages[1:]])
    assert not masks.any(0)[6], "point 6 needs images 3 and 4, which this run never sees"


# -- groups mode: equal exposure by construction ---------------------------


def groups(covis, **kw):
    cfg = CurriculumConfig(mode="groups", **kw)
    return Curriculum(covis, max_steps=kw.pop("max_steps", 1200), cfg=cfg)


def test_groups_partition_the_images(covis):
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=3, rounds=2))
    first_round = [s for s in cur.stages if s.round == 0]
    members = [i for s in first_round for i in s.active]
    assert sorted(members) == list(range(6)), "every image in exactly one group"
    assert [s.active for s in first_round] == [(0, 1, 2), (3, 4, 5)]


def test_groups_are_visited_round_robin(covis):
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=3, rounds=3))
    assert [(s.round, s.group) for s in cur.stages] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
    ]


def test_every_image_gets_identical_exposure(covis):
    """The whole reason this mode exists."""
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=3, rounds=4))
    exposure = cur.exposure()
    values = [exposure[i] for i in cur.images]
    assert len(set(round(v, 6) for v in values)) == 1, values
    assert cur.exposure_spread() == pytest.approx(1.0)
    assert sum(values) == pytest.approx(sum(s.length for s in cur.stages))


def test_exposure_is_the_budget_split_evenly(covis):
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=2, rounds=5))
    assert cur.exposure()[0] == pytest.approx(1200 / 6, rel=1e-6), "6 images, 1200 steps"


def test_group_mode_uses_the_whole_budget(covis):
    for size, rounds in [(2, 3), (3, 4), (6, 2)]:
        cur = Curriculum(covis, max_steps=1200,
                         cfg=CurriculumConfig(mode="groups", group_size=size, rounds=rounds))
        assert sum(s.length for s in cur.stages) == pytest.approx(1200, rel=0.02), (size, rounds)


def test_only_a_first_visit_introduces_images(covis):
    """A revisit has no blind guess to render, so it must not claim one."""
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=3, rounds=3))
    added = [(s.round, s.added) for s in cur.stages]
    assert added == [(0, 0), (0, 3), (1, None), (1, None), (2, None), (2, None)]


def test_seen_is_cumulative_while_active_is_one_group(covis):
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=3, rounds=2))
    first, second = cur.stages[0], cur.stages[1]
    assert first.active == (0, 1, 2) and first.seen == (0, 1, 2)
    assert second.active == (3, 4, 5), "training is restricted to this group"
    assert second.seen == (0, 1, 2, 3, 4, 5), "but everything seen counts for seeding"
    assert second.n_active == 3 and second.n_images == 6


def test_seeding_follows_seen_not_active(covis):
    """A point shared by two images in different groups would never have two
    *active* views at once, so keying seeds on `active` would starve exactly
    the points that tie the groups together."""
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=3, rounds=2))
    # point 5 is in images 2 and 3: exactly one view in each group, so neither
    # group can triangulate it alone
    assert not covis.triangulable([0, 1, 2])[5], "group 0 alone: one view, a ray"
    assert not covis.triangulable([3, 4, 5])[5], "group 1 alone: also one view"
    assert cur.seed_mask(cur.stages[1])[5], "it unlocks once both groups are seen"


def test_a_short_tail_group_is_folded_in(covis):
    """A group of 1 would get a full visit's steps spread over one image, which
    is exactly the inequality this mode removes."""
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=4, rounds=2))
    sizes = [s.n_active for s in cur.stages if s.round == 0]
    assert sizes == [6], "4 + a tail of 2 folds into one group of 6"
    assert cur.exposure_spread() == pytest.approx(1.0)


def test_groups_mode_still_respects_max_images(covis):
    cur = Curriculum(covis, max_steps=1200,
                     cfg=CurriculumConfig(mode="groups", group_size=2, rounds=2, max_images=4))
    assert sorted(cur.images) == sorted(cur.order[:4])
    assert cur.exposure_spread() == pytest.approx(1.0)


def test_an_unknown_mode_is_refused(covis):
    with pytest.raises(ValueError, match="unknown curriculum mode"):
        Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(mode="spiral"))


# -- exposure, recorded either way ----------------------------------------


def test_exposure_sums_to_the_scheduled_steps(covis):
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    assert sum(cur.exposure().values()) == pytest.approx(1000)


def test_incremental_exposure_is_unequal_and_says_so(covis):
    """Proportional pacing fixes the measurement window, not the total: an image
    added third is present for more stages than one added last. That asymmetry
    is what "incremental" means, and the number is recorded rather than hidden."""
    cur = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    exposure = cur.exposure()
    assert exposure[cur.order[0]] > exposure[cur.order[-1]]
    assert cur.exposure_spread() > 2.0


def test_exposure_covers_every_image_the_run_sees(covis):
    cur = Curriculum(covis, max_steps=1000)
    assert set(cur.exposure()) >= set(cur.images)
    assert all(v > 0 for k, v in cur.exposure().items() if k in set(cur.images))


# -- end_step: hand over to all-images training ------------------------------


def test_end_step_compresses_the_incremental_schedule(covis):
    """Same proportional pacing, fitted into 600 steps instead of 1000:
    e = (600-100)/15, so the image stages are 133, 166, 200 and the schedule
    ends at 599 -- the integer remainder is the consolidation stage's."""
    cur = Curriculum(covis, max_steps=1000,
                     cfg=CurriculumConfig(warmup_steps=100, end_step=600))
    assert [s.added for s in cur.stages] == [None, 3, 4, 5, None]
    assert [s.length for s in cur.stages[:4]] == [100, 133, 166, 200]
    closing = cur.stages[-1]
    assert closing.consolidate and not any(s.consolidate for s in cur.stages[:-1])
    assert closing.step == 599 and closing.step + closing.length == 1000


def test_consolidation_trains_every_image_seen(covis):
    cur = Curriculum(covis, max_steps=1000,
                     cfg=CurriculumConfig(warmup_steps=100, end_step=600))
    closing = cur.stages[-1]
    assert sorted(closing.active) == sorted(cur.images) == list(range(6))
    assert closing.seen == cur.stages[-2].seen
    assert closing.added is None, "it introduces nothing, so there is no blind guess"
    assert not cur.seed_mask(closing).any(), "and unlocks no points"


def test_end_step_compresses_the_groups_schedule(covis):
    cur = Curriculum(covis, max_steps=1200, cfg=CurriculumConfig(
        mode="groups", group_size=3, rounds=2, end_step=600))
    rounds = [s for s in cur.stages if not s.consolidate]
    assert sum(s.length for s in rounds) == 600
    closing = cur.stages[-1]
    assert closing.consolidate and closing.group is None and closing.round is None
    assert sorted(closing.active) == list(range(6))
    assert closing.step == 600 and closing.length == 600


def test_consolidation_keeps_groups_exposure_equal(covis):
    """Every image gets an equal share of the consolidation stage too, so the
    mode's one guarantee survives it."""
    cur = Curriculum(covis, max_steps=1200, cfg=CurriculumConfig(
        mode="groups", group_size=3, rounds=2, end_step=600))
    assert cur.exposure_spread() == pytest.approx(1.0)
    assert sum(cur.exposure().values()) == pytest.approx(1200)


def test_consolidation_shrinks_the_incremental_exposure_spread(covis):
    base = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    handed = Curriculum(covis, max_steps=1000,
                        cfg=CurriculumConfig(warmup_steps=100, end_step=600))
    assert handed.exposure_spread() < base.exposure_spread()


@pytest.mark.parametrize("end", [0, 1000, 5000])
def test_an_end_step_at_or_past_max_steps_changes_nothing(covis, end):
    """Short dev runs (end_step 12000, max_steps 4000) keep their old schedule."""
    base = Curriculum(covis, max_steps=1000, cfg=CurriculumConfig(warmup_steps=100))
    cur = Curriculum(covis, max_steps=1000,
                     cfg=CurriculumConfig(warmup_steps=100, end_step=end))
    assert cur.stages == base.stages


def test_a_fixed_image_run_gets_no_consolidation_stage(covis):
    """The ablation's condition: no images are added, so there is no schedule
    to hand over from and the one stage still runs to max_steps."""
    cur = Curriculum(covis, max_steps=1000,
                     cfg=CurriculumConfig(init_images=3, max_images=3, end_step=600))
    assert len(cur.stages) == 1
    assert cur.stages[0].length == 1000 and not cur.stages[0].consolidate


def test_summary_marks_the_consolidation_stage(covis):
    cur = Curriculum(covis, max_steps=1000,
                     cfg=CurriculumConfig(warmup_steps=100, end_step=600))
    assert [r["consolidate"] for r in cur.summary()] == [False] * 4 + [True]
