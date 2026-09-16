# How 3DGS densification works, as gsplat implements it

A reading guide to `gsplat/strategy/default.py`, pinned at commit `28e794ca` (version 1.6.0).
Every claim here was checked against that source, and the layout facts in
[Where new Gaussians land](#where-new-gaussians-land) were verified by running the real ops on a
synthetic param set on this machine — not inferred from reading.

Read it with the file open. It is ~450 lines of plain Python and is the entire subject of Phases 3-5.

---

## The shape of a training step

`DefaultStrategy` is not a model. It owns no parameters and computes no loss. It is a pair of callbacks
the trainer invokes around `loss.backward()`, and everything it does is **structural**: adding, removing
and resetting Gaussians. Gradient descent on the Gaussians themselves is ordinary Adam, untouched.

```
  render  ->  strategy.step_pre_backward()   # retain_grad() on the 2D means
  loss.backward()
              strategy.step_post_backward()  # accumulate stats; every 100 steps, densify
  optimizer.step()
```

`step_pre_backward` does exactly one thing (`default.py:168`): calls `.retain_grad()` on the projected
2D means. That gradient is normally freed as a non-leaf tensor, and it is the signal the whole densification
scheme runs on, so it has to be asked for in advance.

## The signal: screen-space gradient

`_update_state` (`default.py:225`) accumulates two running tallies per Gaussian:

- `state["grad2d"]` — the summed norm of the **2D screen-space** gradient of the Gaussian's centre
- `state["count"]` — how many times it was visible

and `_grow_gs` divides them (`default.py:296`) to get a per-Gaussian **average** gradient.

The intuition is the whole method in one sentence: *a large screen-space gradient on a Gaussian's centre
means the renderer keeps being told to move it, which means one Gaussian is being asked to explain a
region that needs more than one.* The fix is not to move it — it is to add capacity there.

Two details that matter and are easy to miss:

- The gradient is scaled by `width/2 * n_cameras` (`default.py:248`), putting it in pixel units and making
  the threshold resolution-independent. **This is why `grow_grad2d=2e-4` is comparable across
  `--data_factor` settings.**
- Only Gaussians with `radii > 0` — actually visible in that view — contribute. An off-screen Gaussian
  accumulates nothing and so is never a densification candidate.

## The decision: clone or split

`_grow_gs` (`default.py:286`) splits high-gradient Gaussians into two cases by **size**:

| Condition | Action | Meaning |
|---|---|---|
| high grad **and** small (`≤ grow_scale3d × scene_scale`) | **duplicate** | under-reconstruction: detail exists that nothing covers. Add a second Gaussian at the same place. |
| high grad **and** large | **split** | over-reconstruction: one blob is smearing across detail. Replace it with two smaller ones. |

`scene_scale` is the normalizer that makes `grow_scale3d=0.01` mean "1% of the scene", not an absolute
distance — for `room-1` that scale came out at **2.81**.

`split` (`ops.py:175`) does three things to the children:
- **positions** are sampled from the parent's own Gaussian — `rotmats @ (scales * randn)`, so children
  scatter along the parent's principal axes, anisotropically. A long thin parent spawns children along
  its length.
- **scales** are divided by **1.6** (verified: the factor is exactly `0.625`).
- everything else — colour, rotation, opacity — is copied.

`duplicate` (`ops.py:141`) copies the parent wholesale, position included. The two coincident Gaussians
diverge only because gradient descent pushes them apart on subsequent steps.

## The decision: prune

`_prune_gs` (`default.py:343`) removes a Gaussian if:
- `sigmoid(opacity) < prune_opa` (0.005) — it has become invisible, **or**
- after the first opacity reset only, it is larger than `prune_scale3d × scene_scale` — it has ballooned.

Note the `step > self.reset_every` guard: size-based pruning is disabled for the first 3,000 steps, so
large initial Gaussians get a chance to shrink before being judged.

## Opacity reset

Every `reset_every` (3,000) steps, `reset_opa` clamps every opacity down to `prune_opa * 2`
(`default.py:218`). This is deliberate, periodic damage: it forces every Gaussian to re-earn its opacity
through gradient descent, and the ones that cannot are pruned on the next refinement pass.

**Expect a visible PSNR dip and a prune spike at 3k, 6k, 9k, 12k, 15k.** That is the system working. It is
also why `pause_refine_after_reset` exists — refinement is suppressed immediately after a reset
(`default.py:191`), because the statistics are meaningless while everything is recovering.

## The schedule

```
     0 ────────── 500 ─────────────────────── 15,000 ──────────── 30,000
       warm-up      densify every 100 steps     refinement only
                    opacity reset every 3,000
```

`step_post_backward` returns immediately once `step >= refine_stop_iter` (`default.py:183`), so the
Gaussian count is **frozen for the whole second half of training**. The last 15,000 steps only polish
parameters. Any lineage event you log therefore lands in the first half.

---

## Where new Gaussians land

This is the part Phase 3 depends on, and the part most likely to be got wrong. Verified by running the
real `duplicate` and `split` on a tagged 5-Gaussian set:

**`duplicate`** appends clones at the end; parents stay put.

```
ids before:  [0, 1, 2, 3, 4]     mask = {1, 3}
ids after:   [0, 1, 2, 3, 4, 1, 3]
              └── unchanged ──┘  └clones┘
```

**`split`** removes the parents and appends children:

```
ids before:  [0, 1, 2, 3, 4]     mask = {1, 3}
ids after:   [0, 2, 4, 1, 3, 1, 3]
              └ kept ┘ └ children ┘
```

**The trap:** the four children are *not* two adjacent pairs. With `rest = 3` kept and `n_split = 2`
parents, the two children of parent `i` sit at `rest + i` **and** `rest + n_split + i` — positions
`(3, 5)` and `(4, 6)`. The children are **child-major**: all first-children, then all second-children.

Reading them as adjacent pairs — `(3,4)` and `(5,6)` — is the natural assumption and it is wrong. It
would attribute every child to the wrong parent while producing a lineage that looks entirely plausible.
`test_lineage` must cover this case explicitly.

**Combined order within one `_grow_gs` call.** Duplicate runs first, then split, with the split mask
extended by `n_dupli` zeros (`default.py:325`) so **clones are never split in the same pass**. The final
layout is:

```
[ originals not split | clones | split children ]
```

**Every state tensor is carried along.** Both ops apply the identical indexing to every tensor in the
`state` dict (`ops.py:165`, `ops.py:228`), not just to params. That is the hook that makes
`InstrumentedStrategy` cheap: put `state["ids"] = arange(N)` in `initialize_state()` and the ids are
re-indexed, concatenated and pruned automatically, for free, forever. Read the ids after each call and the
lineage falls out — no gsplat code needs copying.

---

## Where the plan's hooks attach

| Hook | Site | Note |
|---|---|---|
| lineage | wrap `_grow_gs` / `_prune_gs` | call `super()`, diff `state["ids"]` before and after |
| counts | return values | `_grow_gs -> (n_dupli, n_split)`, `_prune_gs -> n_prune` |
| resets | `step % reset_every == 0` | log the step; expect the PSNR dip |
| seeding | `ops._update_param_with_optimizer` | the only reusable way to append Gaussians from given values — `sample_add` only resamples existing ones |

## Defaults, in one place

| Parameter | Default | Meaning |
|---|---|---|
| `refine_start_iter` | 500 | warm-up before any structural change |
| `refine_stop_iter` | 15,000 | after this the count is frozen |
| `refine_every` | 100 | densification cadence |
| `reset_every` | 3,000 | opacity reset cadence |
| `grow_grad2d` | 2e-4 | screen-space gradient threshold |
| `grow_scale3d` | 0.01 | clone-vs-split size boundary, × `scene_scale` |
| `prune_opa` | 0.005 | opacity below which a Gaussian is removed |
