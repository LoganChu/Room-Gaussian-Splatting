# Room Gaussian Splatting — Project Plan

## Context

**Status: Phase 2 (capture + SfM) is done. Phase 0b (toolchain) is next and is the only thing standing
between here and a first training run.** See "Phase status" below.

The goal is to **see, step by step, how 3D Gaussian Splatting (3DGS) works** while reconstructing your room
from **30–100 phone photos**:

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
- **Platform: native Linux, Ubuntu 24.04.5 LTS — this is now the working machine.**
  - **Machine:** the **RTX 5090 desktop (32 GB)**, booted into Ubuntu. Driver **595.84**, CUDA **13.2**,
    compute capability **sm_120** (`TORCH_CUDA_ARCH_LIST=12.0`).
  - **Laptop (RTX 5070, 8 GB):** no longer on the critical path. The WSL2 phase is over — everything is
    built natively here. The `laptop_5070_8gb` hardware profile stays, because a low-VRAM profile is still
    the fastest way to iterate and a useful ablation, but it is not a required target.
  - **Repo:** `~/Projects/Room-Gaussian-Splatting`, on a native ext4 filesystem, out of OneDrive.
  - **Data:** in the repo at **`data/scenes/<scene>/`**, excluded by `.gitignore`. A scene is then found by
    relative path from anywhere in the repo, with no environment variable to set. `DATA_ROOT` still
    overrides the parent directory for a scene that outgrows this disk.
  - **Python environment: `uv`**, not conda. The repo already carries `pyproject.toml` + `uv.lock` and a
    `.venv/`; `uv` is installed and conda is not. The lockfile replaces `environment.yml` as the
    reproducibility artifact.

Verified facts about gsplat — **re-checked 2026-09-14 against `main` @ `28e794ca` (version 1.6.0)**, all
still true:
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
- **Example dependencies:** `examples/requirements.txt` uses the **official** `pycolmap>=3.10.0`
  (`import pycolmap`), confirmed — not rmbrualla's same-named `SceneManager` fork, so there is **no package
  name collision** with the `pycolmap 4.2.0` that `run_sfm.py` uses. One environment serves both. It also
  pins **`torch==2.9.1` / `torchvision==0.24.1`** (not 2.14), and pulls three more git/extension deps
  besides `fused-ssim`: `fused-bilagrid`, `ppisp` (`nv-tlabs/ppisp@v1.2.1`) and `nvidia-ncore>=19.0.0`.
- **Builds:** there are no prebuilt gsplat wheels for sm_120, so gsplat builds from source. CUDA 13 support
  is on `main` (version string 1.6.0); the newest **tag and PyPI release is still 1.5.3**, so the pin must
  be a `main` commit, not a tag.
- **`torch==2.9.1+cu130` wheels exist** for cp312 linux_x86_64 on `download.pytorch.org/whl/cu130`
  (cu128 and cu129 also have 2.9.1; plain PyPI does not).

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
data/                          # gitignored; NOT a package
  scenes/<scene>/              # run_sfm.py output: images/ images_2|4|8/ sparse/0/ database.db
  runs/<run>/                  # training outputs: snapshots, events, config, PLY
splat/
  sfm.py          # DONE — COLMAP SfM (Phase 2)
  paths.py        # repo data/ root (DATA_ROOT override) + run-dir layout
  lineage.py      # InstrumentedStrategy(DefaultStrategy) — see below
  append.py       # append_gaussians(): seed injection built on ops._update_param_with_optimizer
  curriculum.py   # image order, active-image sampler, stage schedule, before/after renders
  snapshots.py    # light fp16 snapshots (means, scales, quats, opacity, SH-DC, ids) + reader
  events.py       # densify / reset / stage events → parquet
  viewer.py       # playback viewer, built from gsplat examples/simple_viewer.py
  report.py       # plots
configs/  train/{default,incremental}.yaml · hardware/{desktop_5090_32gb,laptop_5070_8gb}.yaml
scripts/  setup_env.sh (DONE, unrun) · run_sfm.py (DONE) · viz_sfm.py (DONE)
          verify_gsplat_compat.py (DONE) · train.py · view.py · report.py
requirements-train.lock        # frozen training stack; uv.lock covers only the SfM deps
docs/     how_it_works.md   # annotated reading guide to gsplat/strategy/default.py, tied to viewer screens
tests/    test_sfm (DONE, 30 tests) · test_lineage · test_append · test_curriculum · test_snapshots
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

**Phase 0a — Machine setup — ✅ DONE**
- Ubuntu 24.04.5 LTS is installed and booted on the 5090 desktop.
- Driver **595.84** is loaded and `nvidia-smi` reports the RTX 5090 with 32 GB and CUDA 13.2. This is
  r580+, as Blackwell requires, so the driver step is finished. Nothing further to do here.

**Phase 0b — Toolchain — ⬅️ NEXT, and the only blocker for training**

Nothing of the training stack exists on this machine yet: **no `gcc`/`g++`, no `nvcc`, no CUDA toolkit, no
torch, no gsplat.** The only Python environment is the uv `.venv`, holding pillow + pycolmap + pytest, which
is what `run_sfm.py` needs and nothing more.

- **System packages (apt):** `build-essential` (gcc/g++ 13, the 24.04 default and a supported host compiler
  for CUDA 13) and `ninja-build`. Without a compiler nothing below can build.
- **CUDA toolkit:** CUDA 13.x from NVIDIA's apt repo, `ubuntu2404` variant. **Do not install a driver from
  it** — 595.84 is already newer than whatever the toolkit meta-package would pull; install the toolkit
  component only (`cuda-toolkit-13-x`, not `cuda`). Then `export CUDA_HOME=/usr/local/cuda`.
- **`scripts/setup_env.sh`** — native Ubuntu only now; the WSL2 detection branch is dropped. It:
  - creates the env with **uv** (`uv venv --python 3.12`), not conda;
  - installs **`torch==2.9.1` + `torchvision==0.24.1` from the cu130 index**
    (`--index-url https://download.pytorch.org/whl/cu130`) — the version gsplat's
    `examples/requirements.txt` pins, and the one CUDA 13 wheel that exists for cp312/linux;
  - builds gsplat from a **pinned `main` commit** (1.6.0 is unreleased, so a commit hash, not a tag) with
    `TORCH_CUDA_ARCH_LIST=12.0`;
  - installs `examples/requirements.txt`, which also builds **fused-ssim**, **fused-bilagrid** and
    **ppisp** — four CUDA extensions total, so budget real time for this step and set
    `MAX_JOBS` to keep the compile from exhausting RAM;
  - vendors `examples/` from the same commit into `third_party/gsplat_examples/`;
  - freezes the result to `requirements-train.lock`. Note the split: `uv.lock` covers only the SfM deps
    declared in `pyproject.toml`, because the training stack is installed imperatively (a git commit, a
    custom index, `--no-build-isolation`) and cannot be expressed there. The pins at the top of
    `setup_env.sh` plus that freeze are the reproducibility artifact.
- **Idempotent and resumable.** Four CUDA extensions is a long build that can fail partway, so every step
  is separately runnable (`--step gsplat`) and re-running a finished step is a no-op. `--check` verifies an
  existing install without changing anything.
- **Check:** the script's last step asserts `torch.cuda.get_device_capability() == (12, 0)`, imports
  gsplat, and **rasterizes one Gaussian on the GPU** — exercising the compiled kernel, not just the import,
  which is the difference between "it installed" and "it works". Then a 7k-iteration `simple_trainer` run
  on Mip-NeRF 360 "room" should open the live viewer.
- **Note:** the two CUDA-version numbers differ on purpose. The driver reports 13.2 (what it can run);
  torch is built against cu130 (what it was compiled with). A driver newer than the toolkit is the correct
  direction, so this is fine.

**Phase 1 — Baseline and read-through**
- **Baseline run:** train Mip-NeRF 360 "room" to 30k iterations with vanilla simple_trainer, and record
  PSNR/SSIM/LPIPS and the Gaussian count. This is the reference for "hooks don't change results".
- **Reading guide:** write `docs/how_it_works.md` alongside reading `gsplat/strategy/default.py` and the
  training loop.

**Phase 2 — Your data — ✅ DONE (`data/scenes/room-1/`)**
- **Capture protocol** (as followed): lock exposure, focus and white balance; no zoom; avoid motion blur;
  shoot from 3–4 positions at 2 heights with about 70% overlap; keep texture in every frame; don't move
  objects.
- **`run_sfm.py`:** pycolmap 4.2.0 SIFT → exhaustive matching → incremental mapping, written in the layout
  `simple_trainer` expects. Undistortion is skipped on purpose: gsplat's Parser undistorts OPENCV cameras
  itself, so a second copy of the images would be dead weight.
- **Result — every gate met:**

  | Check | Gate | Actual |
  |---|---|---|
  | Registered images | ≥ 90% | **32 / 32 (100%)** |
  | Mean reprojection error | ≲ 1 px | **0.93 px** |
  | Models reconstructed | 1 | **1** (no fragmentation) |
  | Camera | single | **1 × OPENCV, 3024×4032** |

  6,762 points3D · 24,222 observations · mean track length 3.58 · 757 observations per image.
- **Two soft spots, neither blocking.** 6.7k points with a mean track length of 3.58 is a **thin seed** for
  a room, so densification will be doing nearly all the work and Phase 5's seed-point injection has less to
  draw on. And 32 photos is the bottom of the 30–100 target. If quality disappoints, **more photos is the
  first lever**, before any hyperparameter.
- **SfM ran on CPU.** The installed pycolmap 4.2.0 wheel is built without CUDA, so feature extraction and
  matching used 8 CPU threads (recorded as a warning in `sfm_report.json`). It cost ~6 s of mapping at this
  scale, so it is not worth chasing a CUDA COLMAP build unless the photo count grows a lot.

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

### Phase status

Everything now runs on one machine: the native Ubuntu 5090 desktop. The WSL2 / Windows split is history.

| Phase | Status | Notes |
|---|---|---|
| 0a machine setup | ✅ done | Ubuntu 24.04.5, driver 595.84, CUDA 13.2, sm_120 |
| 0b toolchain | ⬅️ **next — blocks everything** | `setup_env.sh` written and unrun; needs sudo for apt + CUDA toolkit |
| 1 baseline + reading guide | todo | needs 0b; Mip-NeRF 360 "room" at 30k as the reference |
| 2 capture + SfM | ✅ done | `data/scenes/room-1/`, 32/32 registered, 0.93 px |
| 3 instrumentation | todo | pure Python once gsplat imports; tests tiny and synthetic |
| 4 playback viewer | todo | viser on `localhost:8080`, opened locally |
| 5 curriculum | todo | develop at `--data_factor 8`, few images, then scale up |
| 6 full-quality room run | todo | `--data_factor 1` or 2; the payoff |

**Immediate plan:**

1. **Phase 0b** — run `./scripts/setup_env.sh` (written; steps 1-2 need sudo). This is the whole blocker.
2. **Phase 1** — vanilla `simple_trainer` on Mip-NeRF 360 "room" for the baseline, *and* a first vanilla run
   on `data/scenes/room-1` at `--data_factor 4` just to see the room appear. That first room render is the
   cheapest possible check that Phase 2's output is genuinely trainable end to end.
3. **Phases 3 → 4 → 5 → 6** in order.

**Reproducibility (one machine, but still worth keeping):**
- **Repo:** `~/Projects/Room-Gaussian-Splatting` on ext4, synced through a git remote.
- **Photo originals:** `photos/` is gitignored — keep a copy on separate media. They are the one
  irreplaceable input; the SfM model can always be rebuilt from them, but nothing rebuilds them.
- **Data:** `data/scenes/` and `data/runs/`, gitignored, alongside the code.
- **Environment:** reproduced from `uv.lock` + `setup_env.sh`, not copied. Compiled CUDA extensions never
  transfer between machines; the script is what makes them reproducible.
- **Paths and budgets:** only through `paths.py` and `--hw auto`, with nothing machine-specific hardcoded.

---

## Risks

| Risk | Mitigation |
|---|---|
| ~~Blackwell driver problems on Linux~~ | **resolved** — 595.84 loaded, 5090 visible, CUDA 13.2 |
| ~~gsplat's COLMAP parser can't read a pycolmap 4.2 model~~ | **resolved** — verified 2026-09-14, see Verification below |
| CUDA extension builds fail (gsplat, fused-ssim, fused-bilagrid, ppisp) | gcc 13 from `build-essential`; CUDA toolkit major = torch's cu130; `TORCH_CUDA_ARCH_LIST=12.0`; cap `MAX_JOBS` so nvcc doesn't exhaust RAM; if fused-ssim fails, swap to torchmetrics SSIM in one `# [splat]` edit |
| Installing the CUDA toolkit drags in an older driver and breaks the working one | install the toolkit component only (`cuda-toolkit-13-x`), never the `cuda` meta-package |
| Hooks rely on gsplat internals (`_grow_gs`, `_prune_gs`, ops helper) | pinned commit; re-verified at `28e794ca`; tests fail loudly on upgrade; vendored examples patch kept small |
| **Thin SfM seed (6,762 points, track length 3.58)** | densification carries it; if the result is poor, shoot more photos before touching hyperparameters — 32 is the low end of the target |
| Textureless walls → weak SfM | capture protocol; hloc SuperPoint+LightGlue as a fallback matcher |
| Exposure drift / mirrors / screens | lock exposure; view-dependent artifacts accepted |
| Snapshot disk usage | fp16 light snapshots (~28 MB per 1M Gaussians); `data/` is on the local ext4 disk; `DATA_ROOT` moves it if that fills |
| Data now lives inside the repo | `.gitignore` excludes `data/`, `photos/`, `*.ply`, `*.bin`, `*.parquet`, `*.db` — check `git status` stays clean after a training run |

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
- **SfM → gsplat compatibility — ✅ verified 2026-09-14.** `data/scenes/room-1` was replayed through
  gsplat's real `examples/datasets/colmap.py` Parser logic (helpers extracted from upstream `28e794ca` by
  AST, so the check tracks upstream rather than a paraphrase), at `--data_factor` 1, 2, 4 and 8. All
  checks passed at every factor:
  - `pycolmap.Reconstruction()` reads the **new rig/frame-format model** written by pycolmap 4.2 — the
    extra `rigs.bin` / `frames.bin` are simply ignored, and `cameras.bin` / `images.bin` / `points3D.bin`
    parse normally (32 images, 6,762 points);
  - every attribute the Parser touches resolves: `reg_image_ids()`, `im.cam_from_world.matrix()`,
    `cam.calibration_matrix()`, `cam.params`, `point.xyz/.error/.color`, `point.track.elements`;
  - the OPENCV camera maps to `('perspective', k1 k2 p1 p2)`, so gsplat undistorts it itself;
  - `images/` → `images_N/` filename mapping resolves all 32 paths despite the `.jpeg` → `.png` extension
    change, and the PNG downscales avoid upstream's "re-resize JPEGs into `images_N_png/`" path;
  - scaled intrinsics `K/N` match the actual downscaled pixel dimensions exactly at every factor;
  - the `normalize=True` pipeline runs; note it will apply its **upside-down flip** heuristic to this scene.
  - Re-run with `scripts/verify_gsplat_compat.py --gsplat <checkout>` after any gsplat version bump: it is
    the cheapest guard against an upstream parser change silently invalidating the SfM output, and it
    fails loudly if the helpers it lifts from upstream have been renamed.
- **Hook neutrality:** the instrumented run matches the vanilla run on Mip-NeRF 360 room (Phase 3 check).
- **End to end:**
  1. `run_sfm.py --images photos/room-1` → `data/scenes/room-1/`
  2. `train.py --data_dir data/scenes/room-1 --config incremental --hw auto`
  3. `view.py --run <dir>` (scrub the timeline)
  4. `report.py`
- **Portability:** `setup_env.sh` reproduces the env from scratch on a clean Ubuntu 24.04 + r580 driver
  machine, and a run dir copied in resumes and retrains under a different `--hw` profile.
