"""Playback viewer: scrub a finished run and see why the population changed.

``simple_trainer``'s built-in viewer shows the model as it is *now*, live, and
that stays the right tool for watching a run. This one is the other half: a run
that has already finished, replayed from its snapshots, with the event log
wired into the colours so densification is legible rather than merely visible.

Built on the same stack as gsplat's ``examples/simple_viewer.py`` -- viser,
``nerfview.Viewer``, and ``GsplatViewer`` from the vendored examples -- so the
familiar rendering controls come for free and this file only adds what is ours:

- a **timeline** over the snapshot steps, with the opacity resets marked;
- **colour by origin** (sfm / seed / clone / split) and **by age**, from
  ``events.parquet`` -- the modes that turn a grey cloud into a history;
- **lineage isolation**: pick a Gaussian, see its whole clan and nothing else;
- **ellipsoids**: shrink and de-fade every splat so individual Gaussians,
  rather than their blended sum, are what you are looking at;
- **camera snap**: jump to a training view with its GT, render and error side
  by side.

What it cannot show
-------------------
Snapshots drop the higher-order spherical harmonics (see ``snapshots.py``), so
playback renders view-independent colour: the SH degree the panel reports is
the degree *training* had reached, not what is being drawn. View-dependent
shine is a checkpoint-only phenomenon -- use ``simple_viewer.py --ckpt`` for
that. Trading it away is what keeps a 120-snapshot timeline on disk at all.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import timeline as tl
from .paths import RunPaths, repo_root

#: the vendored gsplat examples, pinned to the same commit as the gsplat
#: package; importing the real Parser/Dataset rather than paraphrasing them is
#: the same principle scripts/verify_gsplat_compat.py enforces
_VENDOR = repo_root() / "third_party" / "gsplat_examples"

#: frustum colours by role. active/pending/added are Phase 5's curriculum
#: states; a Phase 4 run only ever has train and test.
ROLE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "train": (64, 192, 87),  # green
    "test": (51, 154, 240),  # blue
    "pending": (134, 142, 150),  # grey
    "added": (253, 126, 20),  # orange
}

#: our modes, appended to the ones GsplatViewer already offers
ORIGIN_MODE = "color by origin"
AGE_MODE = "age"
EXTRA_MODES = (ORIGIN_MODE, AGE_MODE)


def _import_vendored():
    """Import the vendored Parser/Dataset, adding the examples dir to the path."""
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from datasets.colmap import Dataset, Parser  # type: ignore

    return Parser, Dataset


# --------------------------------------------------------------------------
# the population, frame by frame
# --------------------------------------------------------------------------


class Playback:
    """Turns a snapshot step into rasterization inputs.

    Holds a small LRU of decoded frames: a snapshot of ~900k Gaussians costs
    ~100 ms to decompress and ~100 ms to upload, which is fine for a scrub but
    not for dragging back and forth across the same two frames.
    """

    def __init__(self, timeline: tl.Timeline, device: torch.device, cache: int = 4) -> None:
        self.timeline = timeline
        self.device = device
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._order: List[int] = []
        self._cache_size = max(1, cache)

    def frame(self, step: int) -> Dict[str, torch.Tensor]:
        if step in self._cache:
            self._order.remove(step)
            self._order.append(step)
            return self._cache[step]
        raw = self.timeline.frame(step)
        # fp16 on disk is a storage decision; everything downstream wants fp32
        out = {
            "means": torch.from_numpy(raw["means"]).to(self.device, torch.float32),
            "quats": torch.from_numpy(raw["quats"]).to(self.device, torch.float32),
            # stored raw, exactly as the trainer held them: log scales, logit opacities
            "scales": torch.from_numpy(raw["scales"]).to(self.device, torch.float32),
            "opacities": torch.from_numpy(raw["opacities"]).to(self.device, torch.float32),
            "sh0": torch.from_numpy(raw["sh0"]).to(self.device, torch.float32),
            "ids": torch.from_numpy(raw["ids"]).to(self.device, torch.int64),
        }
        out["ids_np"] = raw["ids"]  # kept on the host for the numpy lineage lookups
        self._cache[step] = out
        self._order.append(step)
        while len(self._order) > self._cache_size:
            del self._cache[self._order.pop(0)]
        return out

    # -- per-frame derived tensors -----------------------------------------

    def colors(self, step: int, mode: str) -> Tuple[torch.Tensor, Optional[int]]:
        """``(colors, sh_degree)`` for the rasterizer.

        ``sh_degree=0`` means "these are SH DC coefficients, evaluate them";
        ``None`` means "these are already RGB", which is what the lineage modes
        produce. Both paths are gsplat's, not ours.
        """
        f = self.frame(step)
        lineage = self.timeline.lineage
        if mode == ORIGIN_MODE and lineage is not None:
            rgb = tl.colors_by_origin(lineage, f["ids_np"])
            return torch.from_numpy(rgb).to(self.device), None
        if mode == AGE_MODE and lineage is not None:
            rgb = tl.colors_by_age(lineage, f["ids_np"], step)
            return torch.from_numpy(rgb).to(self.device), None
        return f["sh0"], 0

    def extent(
        self, step: int, gids: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """``(centre, radius)`` of a gid set at this step, in world units.

        The lineage modes are useless without this. A clan descended from one
        SfM point is a few metres across in a scene tens of metres wide, so
        isolating one while the camera points elsewhere renders an empty frame
        and gives no hint why -- measured on ``room-1``: a 7,992-Gaussian family
        seen by 5 of 32 training cameras.
        """
        f = self.frame(step)
        means = f["means"]
        if gids is not None and len(gids):
            mask = self.selection(step, gids)
            if mask is None:
                return None
            means = means[mask]
        if means.shape[0] == 0:
            return None
        lo = means.amin(0)
        hi = means.amax(0)
        centre = ((lo + hi) / 2).cpu().numpy()
        radius = float(torch.linalg.norm(hi - lo).item()) / 2
        return centre, max(radius, 1e-3)

    def selection(self, step: int, gids: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        """Boolean mask over this frame's Gaussians for a set of gids."""
        if gids is None or len(gids) == 0:
            return None
        f = self.frame(step)
        mask = tl.highlight(f["ids_np"], gids)
        if not mask.any():
            return None
        return torch.from_numpy(mask).to(self.device)


# --------------------------------------------------------------------------
# the cameras that produced the run
# --------------------------------------------------------------------------


@dataclass
class Camera:
    index: int
    name: str
    role: str
    camtoworld: np.ndarray  # (4, 4), in the same normalized frame as the splats
    K: np.ndarray  # (3, 3), undistorted
    width: int
    height: int


class Cameras:
    """The training cameras, in the splats' coordinate frame.

    The frame matters more than it looks. ``Parser(normalize=True)`` applies a
    similarity transform, a principal-axis alignment and sometimes a 180-degree
    flip before training ever starts, so the snapshots live in the *normalized*
    frame. Re-deriving poses from the COLMAP model instead of asking the Parser
    for them puts every frustum in the wrong place -- plausibly, subtly wrong.
    So the Parser is constructed with the run's own settings and asked.
    """

    def __init__(self, scene_dir: Path, factor: int, normalize: bool, test_every: int = 8) -> None:
        Parser, Dataset = _import_vendored()
        self.parser = Parser(
            data_dir=str(scene_dir), factor=factor, normalize=normalize, test_every=test_every
        )
        self.test_every = test_every
        # Upstream's Dataset owns the undistortion; reuse it rather than
        # reimplement those six lines and let them drift.
        self._splits = {"train": Dataset(self.parser, "train"), "test": Dataset(self.parser, "test")}
        self.cameras: List[Camera] = []
        for i, name in enumerate(self.parser.image_names):
            cid = self.parser.camera_ids[i]
            w, h = self.parser.imsize_dict[cid]
            self.cameras.append(
                Camera(
                    index=i,
                    name=name,
                    role="test" if i % test_every == 0 else "train",
                    camtoworld=np.asarray(self.parser.camtoworlds[i], dtype=np.float64),
                    K=np.asarray(self.parser.Ks_dict[cid], dtype=np.float64),
                    width=int(w),
                    height=int(h),
                )
            )

    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def names(self) -> List[str]:
        return [f"{c.index:03d} {c.name} [{c.role}]" for c in self.cameras]

    def by_label(self, label: str) -> Camera:
        return self.cameras[int(label.split()[0])]

    def gt(self, cam: Camera) -> np.ndarray:
        """The undistorted ground-truth image, exactly as training saw it."""
        ds = self._splits[cam.role]
        item = int(np.nonzero(ds.indices == cam.index)[0][0])
        return ds[item]["image"].numpy().astype(np.uint8)

    @classmethod
    def from_cfg(cls, cfg: Dict, scene_dir: Optional[Path] = None) -> Optional["Cameras"]:
        """Build from a run's ``cfg.yml``, or None if the scene is not reachable.

        A run directory is meant to rsync between machines, so its recorded
        ``data_dir`` may simply not exist here. That costs the frustums and the
        camera snap; it must not cost the timeline.
        """
        if scene_dir is None:
            raw = cfg.get("data_dir")
            if not raw:
                return None
            scene_dir = Path(raw)
            if not scene_dir.is_absolute():
                scene_dir = repo_root() / scene_dir
        if not Path(scene_dir).exists():
            return None
        try:
            return cls(
                Path(scene_dir),
                factor=int(cfg.get("data_factor", 1)),
                normalize=bool(cfg.get("normalize_world_space", True)),
                test_every=int(cfg.get("test_every", 8)),
            )
        except Exception as exc:  # pragma: no cover - depends on the scene on disk
            print(f"[viewer] cameras unavailable ({exc!r}); timeline still works")
            return None


def error_triptych(gt: np.ndarray, render: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """``[GT | render | |error|]`` as one uint8 image, for the camera panel."""
    from matplotlib import colormaps

    gt_f = gt.astype(np.float32) / 255.0
    err = np.abs(gt_f - render).mean(-1)
    # scaled to this pair, not to 1.0: late in a run the absolute error is small
    # everywhere and a fixed scale shows a black rectangle
    err = err / max(float(err.max()), 1e-6)
    err_rgb = colormaps[colormap](err)[..., :3]
    panels = [gt_f, np.clip(render, 0, 1), err_rgb]
    return (np.concatenate(panels, axis=1) * 255).astype(np.uint8)


def psnr(gt: np.ndarray, render: np.ndarray) -> float:
    gt_f = gt.astype(np.float32) / 255.0
    mse = float(np.mean((gt_f - np.clip(render, 0, 1)) ** 2))
    return float("inf") if mse == 0 else -10.0 * float(np.log10(mse))


# --------------------------------------------------------------------------
# the viser GUI
# --------------------------------------------------------------------------


def _render_tab_state_cls():
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from gsplat_viewer import GsplatRenderTabState  # type: ignore

    class PlaybackRenderTabState(GsplatRenderTabState):
        """``GsplatRenderTabState`` plus where we are on the timeline.

        Declared as a plain subclass with class-level defaults, matching how
        ``GsplatRenderTabState`` extends nerfview's dataclass: the parent's
        generated ``__init__`` still applies and these behave as defaults.
        """

        step: int = 0
        ellipsoids: bool = False
        ellipsoid_scale: float = 0.35
        ellipsoid_opacity: float = 0.95
        isolate: bool = False
        selected: Optional[np.ndarray] = None

    return PlaybackRenderTabState


def make_render_fn(playback: Playback) -> Callable:
    """The nerfview render callback: draw one snapshot from one camera."""
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from gsplat.rendering import rasterization, render_mode_has_color
    from nerfview import apply_float_colormap  # type: ignore

    device = playback.device
    #: our modes are RGB renders that differ only in what goes into `colors`
    MODE_MAP = {
        "rgb": "RGB",
        "depth(accumulated)": "D",
        "depth(expected)": "ED",
        "alpha": "RGB",
        ORIGIN_MODE: "RGB",
        AGE_MODE: "RGB",
    }

    def render_fn(camera_state, render_tab_state):
        s = render_tab_state
        width, height = (
            (s.render_width, s.render_height)
            if s.preview_render
            else (s.viewer_width, s.viewer_height)
        )
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)

        step = int(s.step)
        mode = MODE_MAP.get(s.render_mode, "RGB")
        f = playback.frame(step)
        colors, sh_degree = playback.colors(step, s.render_mode)
        if not render_mode_has_color(mode):
            # gsplat drops `colors` for depth-only modes (rendering.py: `colors if
            # has_color ...`) but still validates sh_degree against it, so a
            # non-None degree here is rejected by the CUDA op. Asking gsplat
            # which modes carry colour keeps this tracking upstream.
            sh_degree = None
        means, quats = f["means"], f["quats"]
        scales = f["scales"].exp()  # stored in log space, as the trainer holds it
        opacities = f["opacities"].sigmoid()

        if s.ellipsoids:
            # Shrink and de-fade: a splat cloud is a sum of thousands of faint
            # overlapping kernels, and at full size individual Gaussians are
            # simply not what you are looking at.
            scales = scales * float(s.ellipsoid_scale)
            opacities = torch.full_like(opacities, float(s.ellipsoid_opacity))

        mask = playback.selection(step, s.selected) if s.isolate else None
        if mask is not None:
            means, quats, scales, opacities = (
                means[mask], quats[mask], scales[mask], opacities[mask],
            )
            colors = colors[mask]

        with torch.inference_mode():
            render_colors, render_alphas, info = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats=c2w.inverse().contiguous()[None],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=None if sh_degree is None else min(s.max_sh_degree, sh_degree),
                near_plane=s.near_plane,
                far_plane=s.far_plane,
                radius_clip=s.radius_clip,
                eps2d=s.eps2d,
                backgrounds=torch.tensor([s.backgrounds], device=device) / 255.0,
                render_mode=mode,
                rasterize_mode=s.rasterize_mode,
                camera_model=s.camera_model,
                packed=False,
            )

        s.total_gs_count = int(means.shape[0])
        radii = info.get("radii")
        s.rendered_gs_count = int((radii > 0).all(-1).sum().item()) if radii is not None else 0

        if s.render_mode in ("depth(accumulated)", "depth(expected)"):
            depth = render_colors[0, ..., 0:1]
            near, far = (
                (s.near_plane, s.far_plane)
                if s.normalize_nearfar
                else (depth.min(), depth.max())
            )
            d = torch.clip((depth - near) / (far - near + 1e-10), 0, 1)
            if s.inverse:
                d = 1 - d
            return apply_float_colormap(d, s.colormap).cpu().numpy()
        if s.render_mode == "alpha":
            return apply_float_colormap(render_alphas[0, ..., 0:1], s.colormap).cpu().numpy()
        return render_colors[0, ..., 0:3].clamp(0, 1).cpu().numpy()

    return render_fn


def _viewer_cls():
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from gsplat_viewer import GsplatViewer  # type: ignore

    class PlaybackViewer(GsplatViewer):
        """``GsplatViewer`` plus the timeline, lineage and camera panels."""

        def __init__(
            self,
            server,
            render_fn: Callable,
            output_dir: Path,
            playback: Playback,
            cameras: Optional[Cameras] = None,
            frustum_scale: float = 0.12,
        ) -> None:
            # set before super(): Viewer.__init__ populates the tabs, and the
            # tab builders below read these
            self.playback = playback
            self.timeline = playback.timeline
            self.cameras = cameras
            self.frustum_scale = frustum_scale
            self._frustums: List = []
            self._playing = False
            super().__init__(
                server=server,
                render_fn=render_fn,
                output_dir=output_dir,
                mode="rendering",
                render_modes=(
                    "rgb", "depth(accumulated)", "depth(expected)", "alpha", *EXTRA_MODES
                ),
            )
            server.gui.set_panel_label("run playback")
            if self.cameras is not None:
                self._add_frustums()

        # -- state ---------------------------------------------------------

        def _init_rendering_tab(self):
            self.render_tab_state = _render_tab_state_cls()()
            steps = self.timeline.steps
            self.render_tab_state.step = steps[0] if steps else 0
            self._rendering_tab_handles = {}
            self._rendering_folder = self.server.gui.add_folder("Rendering")

        def _populate_rendering_tab(self):
            super()._populate_rendering_tab()
            self._populate_timeline_tab()
            self._populate_lineage_tab()
            self._populate_camera_tab()

        # -- timeline ------------------------------------------------------

        def _marks(self) -> List[Tuple[float, str]]:
            """Reset and densification-start ticks, in slider (index) space."""
            steps = np.asarray(self.timeline.steps)
            marks: List[Tuple[float, str]] = []
            interesting = [(self.timeline.schedule.refine_start_iter, "densify")]
            interesting += [(int(s), "reset") for s in self.timeline.resets]
            for at, label in interesting:
                if steps.size and steps.min() <= at <= steps.max():
                    marks.append((float(int(np.argmin(np.abs(steps - at)))), label))
            return marks

        def _populate_timeline_tab(self):
            server = self.server
            steps = self.timeline.steps
            with server.gui.add_folder("Timeline"):
                if not steps:
                    server.gui.add_markdown(
                        "**No snapshots.** Retrain with `--snapshot_every N` to get a timeline."
                    )
                    return
                slider = server.gui.add_slider(
                    "Frame",
                    min=0,
                    max=len(steps) - 1,
                    step=1,
                    initial_value=0,
                    marks=self._marks() or None,
                    hint="Snapshot index. Ticks are densification start and opacity resets.",
                )
                step_label = server.gui.add_number(
                    "Step", initial_value=steps[0], disabled=True
                )
                with server.gui.add_folder("Playback"):
                    back = server.gui.add_button("< prev")
                    fwd = server.gui.add_button("next >")
                    play = server.gui.add_checkbox("Play", initial_value=False)
                    fps = server.gui.add_slider(
                        "fps", min=1, max=20, step=1, initial_value=6
                    )
                panel = server.gui.add_markdown(self._narration(steps[0]))

                def goto(index: int) -> None:
                    index = int(np.clip(index, 0, len(steps) - 1))
                    slider.value = index  # fires the update handler below

                @slider.on_update
                def _(_) -> None:
                    step = steps[int(slider.value)]
                    self.render_tab_state.step = step
                    step_label.value = step
                    panel.content = self._narration(step)
                    self.rerender(_)

                @back.on_click
                def _(_) -> None:
                    goto(slider.value - 1)

                @fwd.on_click
                def _(_) -> None:
                    goto(slider.value + 1)

                @play.on_update
                def _(_) -> None:
                    self._playing = play.value

                self._rendering_tab_handles.update(
                    {"frame_slider": slider, "fps_slider": fps, "play_checkbox": play}
                )

        def _narration(self, step: int) -> str:
            text = self.timeline.narrate(step, self._ids_now(step))
            return "\n\n".join(text.split("\n"))

        def advance(self) -> None:
            """One frame forward, wrapping. Driven by ``serve``'s play loop."""
            slider = self._rendering_tab_handles.get("frame_slider")
            if slider is None:
                return
            slider.value = (int(slider.value) + 1) % (int(slider.max) + 1)

        @property
        def playing(self) -> bool:
            return self._playing

        @property
        def fps(self) -> float:
            handle = self._rendering_tab_handles.get("fps_slider")
            return float(handle.value) if handle is not None else 6.0

        # -- lineage -------------------------------------------------------

        def _populate_lineage_tab(self):
            server = self.server
            lineage = self.timeline.lineage
            with server.gui.add_folder("Lineage"):
                if lineage is None:
                    server.gui.add_markdown(
                        "**No event log.** Retrain with `--lineage` for origin,"
                        " age and lineage isolation."
                    )
                    return
                gid_input = server.gui.add_number(
                    "gid", initial_value=0, min=0, max=len(lineage) - 1, step=1
                )
                sample = server.gui.add_button("Sample a deep lineage")
                isolate = server.gui.add_checkbox("Isolate family", initial_value=False)
                look = server.gui.add_button("Look at family")
                clear = server.gui.add_button("Clear selection")
                info = server.gui.add_markdown("Pick a gid, or sample one.")

                def select(gid: int) -> None:
                    gid = int(gid)
                    family = lineage.family(gid)
                    self.render_tab_state.selected = family
                    alive = int(tl.highlight(self._ids_now(), family).sum())
                    chain = [gid] + lineage.ancestors(gid)
                    lines = [
                        lineage.describe(gid).replace(" | ", "  \n"),
                        "chain to the SfM point: " + " -> ".join(str(g) for g in chain),
                        f"family: **{family.size}** ids ever, **{alive}** alive at this step",
                    ]
                    got = self.playback.extent(int(self.render_tab_state.step), family)
                    if got is not None:
                        lines.append(f"radius {got[1]:.2f} world units - use *Look at family*")
                    info.content = "\n\n".join(lines)
                    self.rerender(None)

                @gid_input.on_update
                def _(_) -> None:
                    select(gid_input.value)

                @sample.on_click
                def _(_) -> None:
                    gid_input.value = self._deep_gid()  # fires select via on_update

                @isolate.on_update
                def _(_) -> None:
                    self.render_tab_state.isolate = isolate.value
                    self.rerender(_)

                @look.on_click
                def _(event) -> None:
                    got = self.playback.extent(
                        int(self.render_tab_state.step), self.render_tab_state.selected
                    )
                    if got is None:
                        info.content = "Nothing from that family is alive at this step."
                        return
                    self._look_at(event.client, *got)

                @clear.on_click
                def _(_) -> None:
                    self.render_tab_state.selected = None
                    self.render_tab_state.isolate = False
                    isolate.value = False
                    info.content = "Pick a gid, or sample one."
                    self.rerender(_)

        def _ids_now(self, step: Optional[int] = None) -> np.ndarray:
            """This frame's ids, through the playback cache.

            Going to ``Timeline.frame`` here instead would decompress the
            snapshot a second time on every slider move -- ~100 ms at 900k
            Gaussians, paid twice for one scrub.
            """
            if not len(self.timeline):
                return np.empty(0, dtype=np.int32)
            if step is None:
                step = int(self.render_tab_state.step)
            return self.playback.frame(int(step))["ids_np"]

        def _deep_gid(self, sample: int = 4096) -> int:
            """A Gaussian with a long ancestor chain -- the interesting case.

            Picking at random almost always lands on a first-generation clone,
            which has nothing to follow. This samples the live population and
            keeps the deepest, so "follow one Gaussian's split lineage" starts
            somewhere with a lineage to follow.
            """
            lineage = self.timeline.lineage
            ids = self._ids_now()
            if lineage is None or ids.size == 0:
                return 0
            rng = np.random.default_rng()
            pick = rng.choice(ids, size=min(sample, ids.size), replace=False)
            depths = [len(lineage.ancestors(int(g))) for g in pick]
            return int(pick[int(np.argmax(depths))])

        # -- cameras -------------------------------------------------------

        def _add_frustums(self) -> None:
            import viser.transforms as vtf

            assert self.cameras is not None
            for cam in self.cameras.cameras:
                c2w = cam.camtoworld
                fov = 2 * float(np.arctan(cam.height / (2 * cam.K[1, 1])))
                self._frustums.append(
                    self.server.scene.add_camera_frustum(
                        f"/cameras/{cam.index:03d}",
                        fov=fov,
                        aspect=cam.width / cam.height,
                        scale=self.frustum_scale,
                        color=ROLE_COLORS[cam.role],
                        wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                        position=c2w[:3, 3],
                    )
                )

        def _populate_camera_tab(self):
            server = self.server
            with server.gui.add_folder("Cameras"):
                if self.cameras is None:
                    server.gui.add_markdown(
                        "**Scene not found.** Frustums and camera snap need the"
                        " COLMAP scene this run was trained on."
                    )
                    return
                show = server.gui.add_checkbox("Show frustums", initial_value=True)
                which = server.gui.add_dropdown(
                    "Camera", tuple(self.cameras.names), initial_value=self.cameras.names[0]
                )
                snap = server.gui.add_button("Snap + compare")
                caption = server.gui.add_markdown("GT | render | error")
                image = server.gui.add_image(
                    np.zeros((2, 6, 3), np.uint8), label="", format="jpeg"
                )

                @show.on_update
                def _(_) -> None:
                    for handle in self._frustums:
                        handle.visible = show.value

                @snap.on_click
                def _(event) -> None:
                    cam = self.cameras.by_label(which.value)
                    self._snap(event.client, cam)
                    triptych, score = self._compare(cam)
                    image.image = triptych
                    caption.content = (
                        f"**{cam.name}** ({cam.role}) at step"
                        f" {int(self.render_tab_state.step)} - GT | render | error"
                        f"  \nPSNR **{score:.2f} dB** (view-independent colour only)"
                    )

        def _snap(self, client, cam: Camera) -> None:
            """Put the client's camera exactly where this training photo was."""
            import viser.transforms as vtf

            c2w = cam.camtoworld
            R = c2w[:3, :3]
            with client.atomic():
                client.camera.wxyz = vtf.SO3.from_matrix(R).wxyz
                client.camera.position = c2w[:3, 3]
                # viser's fov is vertical
                client.camera.fov = 2 * float(np.arctan(cam.height / (2 * cam.K[1, 1])))

        def _look_at(self, client, centre: np.ndarray, radius: float) -> None:
            """Frame a bounding sphere, keeping the direction the client faces.

            Backing off along the current view axis rather than choosing one
            keeps the jump legible: the family appears where you were already
            looking, instead of the scene spinning to an arbitrary pose.
            """
            eye = np.asarray(client.camera.position, dtype=np.float64)
            axis = eye - centre
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
            fov = float(client.camera.fov) or 1.0
            distance = radius / max(np.tan(fov / 2), 1e-3) * 1.4
            with client.atomic():
                client.camera.position = centre + axis * distance
                client.camera.look_at = centre

        def _compare(self, cam: Camera, max_width: int = 640) -> Tuple[np.ndarray, float]:
            """Render this frame from ``cam`` and stack it against the GT."""
            from nerfview import CameraState  # type: ignore

            gt = self.cameras.gt(cam)
            h, w = gt.shape[:2]
            scale = min(1.0, max_width / w)
            out_w, out_h = int(w * scale), int(h * scale)

            state = self.render_tab_state
            saved = (state.viewer_width, state.viewer_height, state.render_mode)
            state.viewer_width, state.viewer_height = out_w, out_h
            # the comparison is only meaningful against colour, not a colormap
            if state.render_mode not in ("rgb",):
                state.render_mode = "rgb"
            try:
                render = self.render_fn(
                    CameraState(
                        fov=2 * float(np.arctan(h / (2 * cam.K[1, 1]))),
                        aspect=w / h,
                        c2w=cam.camtoworld.astype(np.float32),
                    ),
                    state,
                )
            finally:
                state.viewer_width, state.viewer_height, state.render_mode = saved

            if (out_h, out_w) != (h, w):
                from PIL import Image

                gt = np.asarray(Image.fromarray(gt).resize((out_w, out_h), Image.BILINEAR))
            return error_triptych(gt, np.asarray(render)), psnr(gt, np.asarray(render))

    return PlaybackViewer


def serve(
    run: RunPaths,
    port: int = 8080,
    device: str = "cuda",
    scene_dir: Optional[Path] = None,
    frustum_scale: float = 0.12,
    block: bool = True,
):
    """Open the playback viewer on ``run``. Returns ``(server, viewer)``."""
    import viser

    timeline = tl.Timeline(run)
    if not len(timeline):
        raise SystemExit(
            f"no snapshots in {run.snapshots}\n"
            "Train with --snapshot_every N (and --lineage for the origin modes)."
        )
    playback = Playback(timeline, torch.device(device))
    cameras = Cameras.from_cfg(timeline.cfg, scene_dir)

    server = viser.ViserServer(port=port, verbose=False)
    viewer = _viewer_cls()(
        server=server,
        render_fn=make_render_fn(playback),
        output_dir=run.report,
        playback=playback,
        cameras=cameras,
        frustum_scale=frustum_scale,
    )
    steps = timeline.steps
    # flushed: this is the line telling you where to point a browser, and it is
    # routinely read through a pipe (nohup, tmux) where stdout is block-buffered
    print(
        f"[viewer] {run.root.name}: {len(steps)} snapshots, steps {steps[0]}..{steps[-1]}"
        + (f", {len(timeline.lineage):,} ids in the event log" if timeline.lineage else ", no event log")
        + (f", {len(cameras)} cameras" if cameras else ", no cameras"),
        flush=True,
    )
    print(f"[viewer] http://localhost:{port}  (Ctrl+C to exit)", flush=True)
    if block:
        _play_loop(viewer)
    return server, viewer


def _play_loop(viewer) -> None:
    """Drive the Play checkbox from the main thread."""
    try:
        while True:
            if viewer.playing:
                viewer.advance()
                time.sleep(1.0 / max(viewer.fps, 1e-3))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[viewer] bye")
