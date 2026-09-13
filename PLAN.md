# Room Gaussian Splatting — Project Plan

## Context

The repo is empty apart from a README. The goal is to **see, step by step, how 3D Gaussian Splatting
(3DGS) works** while reconstructing your room from **30–100 phone photos**:

1. start from the COLMAP SfM point cloud,
2. watch Gaussians clone, split and prune during optimization,
3. watch each added image improve the existing model,
4. end with an interactive 3D room in the browser.

Decisions made:
- **Use the existing library end-to-end.** gsplat's reference trainer (`examples/simple_trainer.py`) and
  its `DefaultStrategy` densification are the same code research uses.
- **Your part is a visibility layer on top.** It adds Gaussian lineage, an event log, snapshots, an
  image-by-image curriculum, a playback viewer and a report. Deep implementation understanding is not a
  goal. Understanding comes from watching every iteration and reading gsplat's densification code, which is
  plain Python.
- **COLMAP (pycolmap) for camera poses.**
- **Platform: native Linux, Ubuntu 24.04 LTS.**
  - **Primary machine:** the **RTX 5090 desktop (32 GB)**, dual-booted.
  - **Laptop:** the RTX 5070 Laptop (8 GB) is only an early phase. It runs the same Ubuntu setup script in
    its existing WSL2 Ubuntu (GPU passthrough verified; miniconda present; no CUDA toolkit yet) or natively.
  - **Same GPU architecture:** both GPUs are sm_120.
  - **Repo and data move:** the repo moves out of OneDrive to a git remote, cloned into `~/code/`. Data
    lives in `DATA_ROOT` (default `~/splat_data`), never in git.

Verified facts about gsplat (current `main`):
- **Densification defaults** in `DefaultStrategy`:
  - `refine_start_iter=500` and `refine_stop_iter=15_000`, refining every 100 iterations;
  - `grow_grad2d=2e-4`, `grow_scale3d=0.01`, `prune_opa=0.005`;
  - `reset_every=3000`.
- **Counts are returned.** `_grow_gs()` returns `(n_dupli, n_split)` and `_prune_gs()` returns `n_prune`.
- **Extra per-Gaussian tensors stay aligned.** `gsplat.strategy.ops` (`duplicate`, `split`, `remove`)
  applies the same indexing or concatenation to **every tensor in the strategy `state` dict** as to the
  params:
  - clones are appended at the end;
  - split children (2 per parent) are appended at the end and their parents removed.
- **No op adds Gaussians from given values.** `sample_add` only samples existing ones, but
  `_update_param_with_optimizer` is the reusable helper for writing one.
- **Example dependencies:** `examples/requirements.txt` uses official `pycolmap>=3.10` and `fused-ssim`
  from git, which is a second CUDA extension to build.
- **Builds:** there are no prebuilt gsplat wheels for sm_120, so gsplat builds from source. CUDA 13 support
  landed on `main` (v1.6.0) after the last PyPI release (1.5.3).

---

## Architecture

```
photos ─► run_sfm.py (pycolmap) ─► COLMAP model ─► gsplat simple_trainer (vendored, pinned)
                                                     │  + splat hooks: lineage · events · snapshots · curriculum
                                                     ├─► live viser viewer (built into simple_trainer)
                                                     └─► run dir ─► playback viewer (timeline) · report · PLY
```

### Layout

```
third_party/gsplat_examples/   # gsplat examples/ copied at the SAME pinned commit as the gsplat package;
                               # our edits marked `# [splat]` and exported to patches/gsplat_examples.patch
splat/
  paths.py        # DATA_ROOT + run-dir layout
  lineage.py      # InstrumentedStrategy(DefaultStrategy) — see below
  append.py       # append_gaussians(): seed injection built on ops._update_param_with_optimizer
  curriculum.py   # image order, active-image sampler, stage schedule, before/after renders
  snapshots.py    # light fp16 snapshots (means, scales, quats, opacity, SH-DC, ids) + reader
  events.py       # densify / reset / stage events → parquet
  viewer.py       # playback viewer, built from gsplat examples/simple_viewer.py
  report.py       # plots
configs/  train/{default,incremental}.yaml · hardware/{desktop_5090_32gb,laptop_5070_8gb}.yaml
scripts/  setup_env.sh · run_sfm.py · train.py · view.py · report.py
docs/     how_it_works.md   # annotated reading guide to gsplat/strategy/default.py, tied to viewer screens
tests/    test_lineage · test_append · test_curriculum · test_snapshots
```

### The four hooks into the vendored trainer (small, marked edits)

1. **Strategy:** swap `DefaultStrategy` for `InstrumentedStrategy`.
2. **Init:** in incremental mode, initialize from the subset of SfM points the initial images can triangulate.
3. **Data:** the train sampler draws only from active images, and `curriculum.on_step()` advances stages.
4. **End of step:** `snapshots.maybe_save(step, splats, strategy_state)`.

### `InstrumentedStrategy` (lineage without copying gsplat code)

- **Initial IDs:** `initialize_state()` adds `state["ids"] = arange(N)` and gives each Gaussian the birth
  reason `sfm`.
- **Clones and splits:** wrap `_grow_gs`. Call `super()`, which returns `(n_dupli, n_split)`. The tensor
  layout is then `[kept originals | n_dupli clones | 2·n_split split children]`. The appended rows still
  carry their parent's id, so they get fresh ids, and events `(child, parent, step, clone|split)` are logged.
- **Prunes:** wrap `_prune_gs`. Ids present before but missing after are logged as deaths.
- **Opacity resets:** each reset step is logged as an event.

### Incremental image curriculum (image-by-image)

- **Poses:** SfM runs once on all images, so every stage shares one coordinate frame.
- **Order:** start with the 3 images sharing the most 3D points. Then repeatedly add the image sharing the
  most points with the active set.
- **Seed points:** adding an image calls `append_gaussians()` for SfM points that just became triangulable
  (at least 2 active images in their track) and are far from existing Gaussians. Their birth reason is
  `seed`.
- **Per-stage records:**
  - the new view rendered **before** training on it (the model's blind guess) and **after** its stage;
  - error maps for both renders;
  - PSNR/SSIM/LPIPS on a fixed held-out test set (every 8th image, never trained on);
  - Gaussians born, split and pruned;
  - per-Gaussian |Δparams| across the stage.
- **Ablation mode:** independent runs with N ∈ {3, 5, 10, 20, 40, all} images at equal iteration counts.
  This separates "more images" from "more training time".

### What you'll be able to see (the step-by-step story)

| When | What happens | Where you see it |
|---|---|---|
| Before training | SfM point cloud + camera frustums | playback viewer, step 0 |
| Iter 0 | one Gaussian per SfM point; blobby render | timeline start |
| 0–500 | shapes/colors optimize, no densification | render vs GT error shrinks |
| 500–15k, every 100 | clone (small + high gradient), split (large + high gradient), prune (transparent) | color-by-origin mode; count plot with markers |
| Every 3k | opacity reset → prune spike, brief PSNR dip | report + event markers |
| Every 1k | SH degree +1 → view-dependent shine | orbit a glossy surface |
| 15k–30k | refinement only | PSNR plateau |
| Incremental mode | each new image: blind guess → learned | before/after panel, orange frustum |

**Playback viewer (viser, `localhost:8080`):**
- **Scene:** frustums colored by role: active (green), pending (gray), test (blue), just added (orange).
- **Timeline slider** over snapshots, with densify, reset and stage markers.
- **Render modes:** RGB, depth, color by origin (sfm/seed/clone/split), ellipsoids (shrunken, opaque),
  and age.
- **Snap to camera:** GT | render | error for any training camera.
- **Text panel:** explains the current phase and the last densify counts.
- **Live watching:** during training, use simple_trainer's built-in viewer unchanged.

**Hardware profiles:**
- **`max_gaussians`:** 5090 ≈ 5M (via gsplat's cap/MCMC option if needed), laptop ≈ 1M.
- **`data_factor`:** 5090 uses 1–2, laptop uses 4.
- **Snapshot cadence:** also set per profile.
- **Selection:** `--hw auto` picks a profile from GPU memory.

**Self-contained runs:** each run dir stores the resolved config, git hash and the pinned gsplat commit,
so runs `rsync` between machines.

---

## Phases

**Phase 0a — Machine setup (per machine, not portable)**
- **On the laptop's WSL2:** nothing to do. The GPU is already visible there and the Windows driver serves
  it. Never install a Linux NVIDIA driver inside WSL.
- **On the 5090 desktop:** back up the BitLocker recovery key, disable Windows Fast Startup, install
  Ubuntu 24.04, and enroll the MOK if Secure Boot stays on. Install **`nvidia-open`**, r580 or newer;
  Blackwell requires the open kernel modules and CUDA 13 needs r580+.

**Phase 0b — Toolchain (same steps on WSL2 and native Ubuntu)**
- **CUDA toolkit:** install CUDA 13.x from NVIDIA's apt repo. Only the repo variant differs: `wsl-ubuntu`
  under WSL2, `ubuntu2404` on the desktop.
- **`scripts/setup_env.sh`** detects WSL2 (`grep -qi microsoft /proc/version`) to pick that repo and skip
  the driver step. It runs once per machine, because compiled extensions don't transfer, but the commands
  and the `TORCH_CUDA_ARCH_LIST=12.0` target are identical. It:
  - creates the conda env `splat` (Python 3.11, torch 2.14.0+cu130);
  - builds gsplat from a **pinned commit** (v1.6.0 tag if released, else a `main` commit with CUDA 13
    support) with `TORCH_CUDA_ARCH_LIST=12.0`;
  - installs `examples/requirements.txt`, including building fused-ssim;
  - vendors `examples/` from the same commit;
  - writes `environment.yml`.
- **Check:** `nvidia-smi` works, `torch.cuda.get_device_capability() == (12, 0)`, and a 7k-iteration
  `simple_trainer` run on Mip-NeRF 360 "room" opens the live viewer.

**Phase 1 — Baseline and read-through**
- **Baseline run:** train Mip-NeRF 360 "room" to 30k iterations with vanilla simple_trainer, and record
  PSNR/SSIM/LPIPS and the Gaussian count. This is the reference for "hooks don't change results".
- **Reading guide:** write `docs/how_it_works.md` alongside reading `gsplat/strategy/default.py` and the
  training loop.

**Phase 2 — Your data**
- **Capture protocol:**
  - lock exposure, focus and white balance; no zoom; avoid motion blur;
  - shoot from 3–4 positions at 2 heights with about 70% overlap;
  - keep texture in every frame; don't move objects.
- **`run_sfm.py`:** pycolmap SIFT → exhaustive matching → incremental mapping → undistort, written in the
  layout `simple_trainer` expects.
- **Check:** at least 90% of images registered, mean reprojection error under about 1 px, and the point
  cloud plus frustums look right in viser.

**Phase 3 — Instrumentation**
- **What to add:** `lineage.py`, `events.py`, `snapshots.py`, hooks 1 and 4, and `report.py` (Gaussian count
  vs. iteration with clone/split/prune/reset markers, plus loss and PSNR curves).
- **Check:** the instrumented run with the same seed matches the Phase 1 baseline, with the same final
  Gaussian count and PSNR within about 0.1 dB.

**Phase 4 — Playback viewer**
- **What to add:** `viewer.py` with the timeline, render modes, camera snapping and text panel.
- **Check:** scrub from the SfM cloud to the final model and follow one Gaussian's split lineage.

**Phase 5 — Image-by-image curriculum**
- **What to add:** `curriculum.py`, `append.py`, hooks 2 and 3, before/after renders, Δ maps, and ablation
  mode.
- **Check:** the report shows test PSNR vs. number of images plus a per-image improvement chart. The viewer
  highlights each new frustum and the Gaussians it changed.

**Phase 6 — Your room at full quality on the 5090**
- **Train:** run the full-resolution incremental run followed by a standard 30k run.
- **Output:** export the PLY (viewable in SuperSplat) and do the browser walkthrough.

### What runs where

WSL2 Ubuntu and native Ubuntu run the same userland, so **everything except Phase 0a and Phase 6 can be
built today in the laptop's WSL2** and carries over unchanged.

| Phase | Start now in WSL2? | Notes |
|---|---|---|
| 0a machine setup | no | WSL2 needs nothing; the desktop needs it once |
| 0b toolchain | yes | same script; rerun once on the desktop |
| 1 baseline + reading guide | yes | use `--hw laptop_5070_8gb` (`data_factor 4`, fewer iterations) |
| 2 capture + SfM | yes | **do this early** — it needs your physical room, and the COLMAP model is reused as-is on the desktop |
| 3 instrumentation | yes | pure Python; tests are tiny and synthetic |
| 4 playback viewer | yes | viser serves `localhost:8080`; open it in the Windows browser |
| 5 curriculum | yes | develop at low resolution and few images |
| 6 full-quality room run | no | the only phase gated on the 5090 |

**Immediate plan (before the Linux switch):** Phase 0b is **deferred** — the toolchain gets built once, on
the native Linux desktop, rather than twice. Two things are written on Windows now and carried over by git:

1. **`PLAN.md` at the repo root** — this plan, committed so it can be pulled onto the Linux machine.
2. **`scripts/run_sfm.py`** (Phase 2) — the photo directory is a runtime argument, so this needs neither the
   photos nor a GPU to be written. It cannot be *run* until the Linux env exists, so it ships with a
   `--dry-run` mode that validates paths, config and the output layout, plus unit tests for the pure-Python
   parts that import without pycolmap. Its first real run is on Linux, on your photos.

Then on the desktop: Phase 0a → `setup_env.sh` (0b) → Phase 2 for real → Phases 1, 3, 4, 5 → 6.

**To keep the work transferable:**
- **Repo:** in WSL's own filesystem (`~/code/...`), never `/mnt/c`, and synced through a git remote.
- **Photo originals:** keep a copy outside WSL, since they are the one irreplaceable input.
- **Data:** `DATA_ROOT=~/splat_data`, moved to the desktop with `rsync`.
- **Environment:** reproduced from pinned `environment.yml`, not copied.
- **Paths and budgets:** only through `paths.py` and `--hw auto`, with nothing machine-specific hardcoded.
- **WSL memory:** raise the default cap (50% of RAM, about 15.6 GB here) in `.wslconfig` for COLMAP matching.

---

## Risks

| Risk | Mitigation |
|---|---|
| Blackwell driver problems on Linux | `nvidia-open` r580+; MOK enrollment; Windows stays as fallback boot |
| gsplat / fused-ssim build fails | gcc 13 (24.04 default); CUDA toolkit major = torch's cu130; `TORCH_CUDA_ARCH_LIST=12.0`; if fused-ssim fails, swap to torchmetrics SSIM in one `# [splat]` edit |
| Hooks rely on gsplat internals (`_grow_gs`, `_prune_gs`, ops helper) | pinned commit; tests fail loudly on upgrade; vendored examples patch kept small |
| Textureless walls → weak SfM | capture protocol; hloc SuperPoint+LightGlue as a fallback matcher |
| Exposure drift / mirrors / screens | lock exposure; view-dependent artifacts accepted |
| Snapshot disk usage | fp16 light snapshots (~28 MB per 1M Gaussians); `DATA_ROOT` on local disk |

---

## Verification

- **Unit tests:** `pytest tests/` in the `splat` env.
  - **Lineage:** run real gsplat grow/prune ops on a tiny synthetic param set, then check that ids are
    unique, clone and split children point to the right parents, logged counts equal the returned counts,
    and pruned ids are recorded.
  - **Append:** params, Adam state and every `state` tensor keep the same length, and an optimizer step
    after append succeeds.
  - **Curriculum:** image ordering and seed-point filtering.
  - **Snapshots:** save/load round-trip.
- **Hook neutrality:** the instrumented run matches the vanilla run on Mip-NeRF 360 room (Phase 3 check).
- **End to end:**
  1. `run_sfm.py` → `train.py --config incremental --hw auto`
  2. `view.py --run <dir>` (scrub the timeline)
  3. `report.py`
- **Portability:** `setup_env.sh` on the 5090 reproduces the env, and a laptop run dir `rsync`'d over
  resumes and retrains with `--hw desktop_5090_32gb`.
