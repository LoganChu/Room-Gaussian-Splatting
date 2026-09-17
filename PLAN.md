# Room Gaussian Splatting — Project Plan

## Context

**Status: Phases 0a, 0b, 1 and 2 are done.** The stack builds, is verified numerically correct against a
published benchmark, and has trained `room-1` end to end. **Phase 3 (instrumentation) is next.**
See "Phase status" below.

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
  params. Layout **verified empirically** on this machine (tagged 5-Gaussian set, real ops):
  - clones are appended at the end, parents untouched: `[0,1,2,3,4] + mask{1,3} -> [0,1,2,3,4,1,3]`;
  - split removes the parents and appends children: `[0,1,2,3,4] + mask{1,3} -> [0,2,4,|1,3,1,3]`;
  - **⚠️ split children are CHILD-MAJOR, not adjacent pairs.** With `rest` kept and `n_split` parents, the
    two children of parent `i` sit at `rest+i` **and** `rest+n_split+i` — for the example above, positions
    `(3,5)` and `(4,6)`, *not* `(3,4)` and `(5,6)`. Reading them as adjacent pairs is the natural
    assumption, is wrong, and **fails silently**: the lineage graph comes out complete and plausible with
    every child attributed to the wrong parent. `test_lineage` must cover this explicitly.
  - within one `_grow_gs`, duplicate runs first and the split mask is extended with `n_dupli` zeros, so
    **clones are never split in the same pass**. Final layout:
    `[originals not split | clones | split children]`.
  - split scales are divided by exactly 1.6 (measured factor `0.625`); child positions are sampled
    anisotropically along the parent's own principal axes.
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
  timeline.py     # DONE — snapshots + events as one index: lineage, ages, narration
  viewer.py       # DONE — playback viewer, built from gsplat examples/simple_viewer.py
  report.py       # plots
configs/  train/{default,incremental}.yaml · hardware/{desktop_5090_32gb,laptop_5070_8gb}.yaml
scripts/  setup_env.sh (DONE, unrun) · run_sfm.py (DONE) · viz_sfm.py (DONE)
          verify_gsplat_compat.py (DONE) · view.py (DONE) · train.py · report.py
requirements-train.lock        # frozen training stack; uv.lock covers only the SfM deps
docs/     how_it_works.md   # annotated reading guide to gsplat/strategy/default.py, tied to viewer screens
tests/    test_sfm (DONE, 30) · test_lineage (DONE, 14) · test_events (DONE, 11)
          test_snapshots (DONE, 7) · test_timeline (DONE, 30) · test_viewer (DONE, 16)
          test_append · test_curriculum
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
| Every 1k | SH degree +1 → view-dependent shine | orbit a glossy surface — **checkpoint only**, see Phase 4 |
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

**Phase 0b — Toolchain — ✅ DONE (`scripts/setup_env.sh`, run 2026-09-14)**

| Component | Installed |
|---|---|
| gcc / nvcc | 13 / **13.2** (`/usr/local/cuda-13.2`) |
| torch | **2.9.1+cu130**, device capability (12, 0) |
| gsplat | **1.6.0**, built from source for sm_120 |
| CUDA extensions | gsplat, fused-ssim, fused-bilagrid, ppisp — all verified to *execute*, not just import |
| Frozen | 107 packages → `requirements-train.lock` |

The gsplat compile took **~50 minutes**, nearly all of it single-threaded in `ptxas` on the templated
rasterization kernels — `MAX_JOBS` stops helping once the build reaches those few large translation units.

**Three things the first run exposed, all now fixed in the script:**
1. **`UV_HTTP_TIMEOUT=300`.** uv's 30 s default aborts partway through a 400 MB CUDA wheel.
2. **`python3-dev`.** `build-essential` does not provide `Python.h`, the venv sits on the system
   interpreter, and every torch CUDA extension needs it — failing ~5 minutes into the first compile with
   the cause buried at line 4567 of a 6,659-line log.
3. **`has_python_headers` is part of step 1's skip condition and of `cuda_env()`.** Without the former,
   re-running `--step apt` reports "already present" and never installs the missing package. Without the
   latter, the failure stays slow and illegible.

**Smoke test — the stack trained the real scene.** 300 steps on `data/scenes/room-1` at `--data_factor 8`:
1.04 s, 354 it/s, **49 MB** of VRAM, PSNR 14.04 / SSIM 0.651. Blobby as expected (densification starts at
500, SH still degree 0), but the piano, mirrors, frames and patterned cloth all land in the right place at
the right scale and colour — which is the real result, because it means **the COLMAP poses are
geometrically sound**. A pose or convention error looks like noise, not a soft version of the room. The
empty regions are the thin 6,762-point seed showing through.

Incidental find: the trainer reads **EXIF exposure** from all 32 images (mean 2.190 EV) and compensates,
which partly covers the "exposure drift" risk below.

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

**Phase 1 — Baseline and read-through — ✅ DONE (2026-09-14)**

**The build is numerically correct.** Vanilla `simple_trainer` on Mip-NeRF 360 "room", 30k steps,
`--data_factor 2`, held-out test set — against gsplat's published table:

| | PSNR | SSIM | LPIPS | Num GS |
|---|---|---|---|---|
| gsplat-30k (published) | 31.36 | — | — | 1.59M |
| inria-30k (published) | 31.31 | — | — | 1.55M |
| **this build, 30k** | **31.296** | **0.9196** | **0.1633** | **1,573,075** |
| **this build, 7k** | **29.046** | 0.8953 | 0.2100 | 1,122,119 |
| gsplat-7k (published) | 29.21 | — | — | 1.11M |

Within **0.06 dB** and **1.1%** on Gaussian count at 30k, and 0.16 dB / 1% at 7k. Matching on *both*
metrics is the point: PSNR alone could coincide by luck, but the same converged Gaussian count means
densification took the same decisions. The CUDA 13 / sm_120 build is sound.

**Cost on this machine:** 30,000 steps in **5 min 49 s** (~88 it/s), peak **2.38 GB** VRAM. Roughly 13× the
headroom on a 32 GB card, so `--data_factor 1` and much larger Gaussian budgets are comfortable.

**Run dir:** `data/runs/baseline-room-30k/` (861 MB, gitignored). **Keep it** — Phase 3's hook-neutrality
check compares against it directly.

**Scale note for Phase 6:** "room" initialises from **112,627** SfM points; `room-1` from **6,762** — about
17× fewer. The clearest single argument for shooting more photos.

- **Baseline run (as executed):** vanilla simple_trainer at
  `--data_factor 2` (what gsplat's own `benchmarks/basic.sh` uses for the indoor scenes), and record
  PSNR/SSIM/LPIPS and the Gaussian count. It serves **two** purposes:
  - **Correctness of this build.** "room" is a published benchmark, so the number is checkable against
    gsplat's and the 3DGS paper's. Landing near it proves the sm_120 kernels are numerically *right*, not
    merely non-crashing — something `room-1` can never show, having no reference value.
  - **The control for Phase 3.** The instrumented run must match it (same seed, same final Gaussian count,
    PSNR within ~0.1 dB), and that comparison needs a vanilla number to exist first.
- **Data:** `datasets/download_dataset.py --dataset mipnerf360` (one ~12 GB zip, all nine scenes; there is
  no per-scene download). Lives outside the repo or under gitignored `data/`.
- **Reading guide — ✅ DONE:** `docs/how_it_works.md`, pinned to `28e794ca`, with line references into
  `gsplat/strategy/default.py`. Covers the callback shape of a step, the screen-space gradient signal
  (and why `grow_grad2d` is resolution-independent), clone-vs-split, pruning, opacity reset, the schedule,
  the verified tensor layout above, and where the four hooks attach.

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

**Phase 3 — Instrumentation — ✅ DONE**
- **Added:** `paths.py`, `events.py` (parquet: `step, kind, gid, parent`), `lineage.py`
  (`InstrumentedStrategy`), `snapshots.py` (fp16 + reader), `report.py`, and hooks 1 and 4 in the vendored
  trainer behind `--lineage` / `--snapshot-every N`, both **off by default**. 32 new tests (62 total).
- **Two reports, from two independent sources.** `population.png` comes from `events.parquet`, so only an
  instrumented run has it. `training.png` reads the TensorBoard scalars `simple_trainer` writes for
  **every** run — so the Phase 1 vanilla baseline can be reported and compared against without re-running
  it with hooks. `report(RunPaths)` writes whichever the run has data for.
- **The training plot independently confirms `docs/how_it_works.md`.** On the 30k baseline the Gaussian
  count freezes at **exactly 15,000** (`refine_stop_iter`) and shows prune sawteeth at **exactly 3k, 6k,
  9k and 12k** — the opacity resets — with none afterwards. The documented schedule is observable in the
  data, not just in the source.
- **Event log size:** 2.96 bytes/row measured on a real 1.35M-event run, against 20 bytes/row raw.
  Dictionary-encoded kinds plus zstd; a full 30k run's log stays in the low tens of MB.
- **The event log balances exactly.** A 3k run on `room-1`: 6,762 sfm + 494,460 clone + 558,384 split −
  293,999 death = **765,607**, equal to the trainer's own reported `num_GS` to the unit. Books that balance
  against an independently computed number are the strongest cheap check available.

⚠️ **The original check in this plan was not a valid test, and has been replaced.**

It asked for "the same final Gaussian count and PSNR within about 0.1 dB". Measured here, **two identical
vanilla runs** differ far more than that:

| config | n | PSNR | Num GS |
|---|---|---|---|
| vanilla | 12 | 19.576 ± 0.708 | 762,587 ± 10,157 |
| instrumented | 12 | 19.259 ± 0.787 | 760,987 ± 14,333 |

`set_random_seed(42)` is called, but the rasterizer's backward accumulates gradients with **atomics in
nondeterministic order**; one flipped densification decision and the runs diverge permanently. No run can
meet a 0.1 dB criterion, instrumented or not — the criterion was measuring the harness, not the hooks.

**Replaced by two checks that are actually sound:**
1. **Exact, in `tests/test_lineage.py`:** given identical inputs, `InstrumentedStrategy` must produce
   **byte-identical params** to `DefaultStrategy` — asserted for `_grow_gs` (clone-only, split-only, both
   in one call, no-op) and `_prune_gs`, with the RNG stream pinned around each call. This is exactly
   reproducible and strictly stronger than matching end metrics.
2. **Statistical, end-to-end:** PSNR difference −0.317 dB (t = −1.04), Gaussian count −1,600 (t = −0.32) —
   neither distinguishable from zero at n = 12.

**Method note worth keeping:** at n = 4 the difference looked like 1.69 sd and the instrumented spread
looked wider. Both vanished by n = 12 — vanilla's own sd rose from 0.295 to 0.708 as samples accumulated.
Variance estimates from a handful of chaotic runs are close to worthless; 3,000 steps sits mid-densification,
the most chaotic point in training.

- **Test coverage:** `test_lineage` (14) drives the real gsplat ops; `test_snapshots` (7) round-trips and
  checks that `shN` is dropped and ids come from `state`, not position; `test_events` (11) covers schema
  and dtypes through parquet, the empty-batch no-op that every zero-clone refinement hits, rejection of
  unknown kinds and mismatched parent arrays, the births − deaths invariant over a synthetic run, and the
  size bound.

**Phase 4 — Playback viewer — ✅ DONE**
- **Added:** `timeline.py` + `viewer.py` + `scripts/view.py`, and 46 tests (108 total).
- **Why two modules and not the one the plan listed.** `viewer.py` is viser callbacks and CUDA; the
  lineage arithmetic behind colour-by-origin and "follow this Gaussian" is neither, and a wrong read of
  it produces a picture that is coherent, pretty and false — the exact failure `test_lineage.py` guards
  on the way *in*. So the index lives in `timeline.py`, pure numpy over the run dir, and 30 of the 46
  new tests run headlessly against a hand-worked synthetic run.
- **The timeline.** Slider over the snapshot steps, with ticks at densification start and each opacity
  reset; prev/next, and play at 1–20 fps. Frames are LRU-cached (4): ~100 ms to decompress a 900k-Gaussian
  snapshot and ~100 ms to upload it, which is fine for one scrub and not for dragging.
- **Render modes.** gsplat's own (rgb, depth, alpha) plus **colour by origin** (sfm/seed/clone/split, the
  same palette `population.png` uses, so a Gaussian is the same colour in both) and **age**, normalised
  to the frame rather than the run — late on, nearly everything is young and a fixed scale is one flat
  colour. Plus an **ellipsoids** toggle that shrinks and de-fades every splat, since at full size a
  cloud is a blended sum and individual Gaussians are not what you are looking at.
- **Lineage.** Enter a gid or sample one, and get its chain to the SfM point, its whole clan, isolation,
  and a fly-to. **The fly-to is not a convenience.** Measured on `room-1` at step 3900: a 7,992-Gaussian
  family descended from one SfM point spans ~3×2×8 world units in a 20×25×25 scene and is visible from
  5 of the 32 training cameras. Isolate one while pointed elsewhere and you get an empty frame with no
  hint why.
- **Cameras.** Frustums coloured by role (train green, test blue; grey/orange reserved for Phase 5), and
  snap-to-camera with a GT | render | error triptych. Poses come from `Parser(normalize=...)` built with
  the run's own `cfg.yml`, not re-derived from the COLMAP model: `normalize=True` applies a similarity
  transform, a principal-axis alignment and sometimes a 180° flip before training starts, so the
  snapshots are in the *normalized* frame and a re-derived frustum lands plausibly, subtly wrong. GT
  comes from upstream's own `Dataset`, so the undistortion is theirs and cannot drift from it.
- **`cfg.yml` is read without executing it.** The trainer dumps it with `yaml.dump`, so the strategy
  arrives as a `!!python/object:` tag; run dirs are meant to `rsync`. Safe loader, unknown tags degraded
  to plain mappings, which is all the viewer wants from them.
- **Found and fixed an upstream bug.** gsplat 1.6.0's `rasterization` drops `colors` for depth-only
  modes but still validates `sh_degree` against it, so `sh_degree=0` + `render_mode="D"` raises
  `sh_degree must be None when colors is None`. `examples/simple_viewer.py` has the same defect on its
  `--ckpt` path. Guarded by asking gsplat's own `render_mode_has_color`, so it tracks upstream.
- **Degrades instead of refusing.** No `events.parquet` → no origin/age/lineage, timeline still scrubs.
  Unreachable `data_dir` → no frustums, timeline still scrubs. A gid with no birth event draws grey.
- **⚠️ It cannot show view-dependent shine.** Snapshots drop `shN` (`snapshots.py`), so playback renders
  view-independent colour; the SH degree the panel reports is the degree *training* had reached, not what
  is being drawn. The "orbit a glossy surface" row of the story table is a checkpoint-only view — use
  `simple_viewer.py --ckpt`. Trading it away is what keeps a 120-snapshot timeline on disk at all.
- **Check — ✅ passed** on `data/runs/dev-room-4k` (4k steps, `--data_factor 8`, 40 snapshots, 1.40M ids).
  Scrubbed 0 → 3900: the blobby SfM render at 0, still blobby at 400 (warm-up), sharpening from 600 as
  densification starts, recognisable room by 3900. Followed gid 1262390 — a clone at step 3500, depth 27
  below SfM point 3368 — up its chain and out to its 13,523-id clan, 7,992 alive, isolated and rendered.
  Snap-to-camera on image 001 gives 23.76 dB against GT at 3900 (train view, no `shN`) versus 7.67 dB at
  step 0.
- **Snapshot disk cost is the open worry.** 40 snapshots of this run are **399 MB** — ~10 MB each at
  ~900k Gaussians, matching `snapshots.py`'s ~29 B/Gaussian. Phase 6 at `--data_factor` 1–2 with ~5M
  Gaussians and 120 snapshots extrapolates to **~17 GB for one run**. Decide the Phase 6 cadence (or a
  step-range window) before launching it, not after filling the disk.

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
| 0b toolchain | ✅ done | torch 2.9.1+cu130, gsplat 1.6.0 sm_120, 4 CUDA extensions verified |
| 1 baseline + reading guide | ✅ done | PSNR 31.296 vs 31.36 published; `docs/how_it_works.md` written |
| 2 capture + SfM | ✅ done | `data/scenes/room-1/`, 32/32 registered, 0.93 px |
| 3 instrumentation | ✅ done | lineage/events/snapshots/report + hooks 1 & 4; neutrality established |
| 4 playback viewer | ✅ done | `timeline.py`/`viewer.py`/`view.py`; verified on `dev-room-4k`, 108 tests |
| 5 curriculum | ⬅️ **next** | develop at `--data_factor 8`, few images, then scale up |
| 6 full-quality room run | todo | `--data_factor 1` or 2; the payoff |

**Immediate plan:**

1. **Phase 5** — `curriculum.py`, `append.py`, hooks 2 and 3. The viewer's frustum roles already have
   `pending` (grey) and `added` (orange) wired in with nowhere to come from yet; the curriculum is what
   fills them, and `ORIGIN_COLORS` already reserves green for the `seed` births `append.py` will log.
2. **Phase 6** after it — but settle the snapshot cadence first (see Phase 4's disk note).

**Carry into Phase 5.** The ablation ("N ∈ {3, 5, 10, 20, 40, all} images at equal iteration counts") is
one run per condition, in the same chaotic mid-densification regime measured above. A 0.5 dB effect there
is **inside run-to-run noise**. Decide the repeat count before running it, not after seeing the numbers.

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

- **Unit tests:** `.venv/bin/python -m pytest tests/` (108 passing).
  - **Lineage:** runs the **real** gsplat grow/prune ops on a tiny synthetic param set — never a mock,
    since what is being tested is our reading of gsplat's internal layout. Covers unique ids, correct
    parents for clones and splits, the child-major ordering, split parents recorded as deaths so
    births − deaths equals the population, exact prune ids, ids never reused, and the byte-identical
    equivalence with `DefaultStrategy` described in Phase 3.
  - **Append:** params, Adam state and every `state` tensor keep the same length, and an optimizer step
    after append succeeds.
  - **Curriculum:** image ordering and seed-point filtering.
  - **Snapshots:** save/load round-trip, `shN` excluded, ids taken from `state` rather than row position
    (after densification they are not `arange`, and colouring by lineage depends on the real ones).
  - **Events:** schema and dtypes survive parquet, empty batches write nothing, bad input is rejected,
    births − deaths equals the population, and the log stays compact.
  - **Timeline:** a four-point cloud through three refinements, with every answer worked out by hand —
    origins per Gaussian, the ancestor chain to the SfM point, transitive descendants, the clan from any
    member, split parents recorded dead, ages, reset steps, per-refinement clone/split/prune counts, and
    the narration naming the right phase. Plus the degradations: an unknown gid draws grey rather than
    raising, a death with no birth does not stretch the index, a run with no `events.parquet` still
    scrubs, and `cfg.yml`'s `!!python/object:` tag is read without being executed.
  - **Viewer:** the part that is neither viser nor CUDA — that log scales and logit opacities stay raw
    (activating them at load would double-apply `exp`/`sigmoid` at render time), that the lineage modes
    hand the rasterizer flat RGB with `sh_degree=None` while rgb hands it SH DC with `0`, that selection
    masks by gid rather than row position, LRU eviction order, the fly-to bounding sphere, and that a
    missing scene costs the frustums and not the timeline.
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
  3. `view.py --run <dir>` (scrub the timeline); `view.py --run <dir> --check` reports what a run has
     — snapshot count and range, whether it was instrumented, its schedule — without opening a browser
  4. `report.py`
- **Portability:** `setup_env.sh` reproduces the env from scratch on a clean Ubuntu 24.04 + r580 driver
  machine, and a run dir copied in resumes and retrains under a different `--hw` profile.
