"""Injecting new Gaussians mid-run, without copying gsplat's ops (Phase 5).

The curriculum adds one photo at a time. Each new photo makes some SfM points
triangulable for the first time, and those points need to become Gaussians in a
model that is already training. That is not something ``DefaultStrategy`` does:
it only grows by cloning and splitting what is already there, so a region no
photo has covered yet has nothing to clone.

Growing the population by hand means four things have to stay the same length,
and forgetting any one of them fails later and elsewhere:

1. **the params** -- ``means``, ``scales``, ``quats``, ``opacities``, ``sh0``,
   ``shN``;
2. **the Adam state** -- ``exp_avg`` and ``exp_avg_sq`` per param, or the next
   optimizer step indexes a moment buffer shorter than its gradient;
3. **the strategy's running state** -- ``grad2d``, ``count``, ``radii``, and our
   ``ids``, all of which ``ops`` re-indexes on every densification;
4. **gsplat's ``GaussianScene``** -- ``component_index`` and ``signal``, which
   ``validate()`` requires to match ``num_gaussians``.

1 and 2 are done by gsplat's own ``_update_param_with_optimizer``, the same
helper ``duplicate`` uses, so the optimizer surgery is upstream's and not ours.
4 goes through the scene's own ``on_duplicate`` callback. Only 3 needs a
decision, and it is the interesting one -- see ``append_gaussians``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor

#: params whose rows a seed Gaussian needs; ``shN`` is included even though the
#: viewer drops it, because training optimizes it
SEED_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def _knn_scales(points: Tensor, k: int = 4, init_scale: float = 1.0) -> Tensor:
    """Log-scales from the mean distance to the k-1 nearest neighbours.

    Exactly what ``create_splats_with_optimizers`` does at init. Seeds arrive
    hundreds of steps later but come from the same SfM cloud, so they should
    arrive the same size they would have had on step 0 -- otherwise an
    incremental run and a full run are not comparable at the one place the
    comparison is supposed to be clean.
    """
    from .paths import repo_root

    import sys

    vendor = repo_root() / "third_party" / "gsplat_examples"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from utils import knn  # type: ignore

    n = int(points.shape[0])
    if n < 2:
        # No spacing exists with one point, and sklearn raises rather than say
        # so. A stage boundary is the wrong place to discover that.
        raise ValueError(f"_knn_scales needs at least 2 points, got {n}")
    k = min(int(k), n)  # sklearn wants n_neighbors <= n_samples
    dist2_avg = (knn(points, k)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    return torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)


def seed_params(
    points: np.ndarray,
    rgbs: np.ndarray,
    select: Optional[np.ndarray] = None,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    device: Union[str, torch.device] = "cuda",
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    """Build param rows for SfM points, the way init would have built them.

    ``points``/``rgbs`` are the **whole** SfM cloud and ``select`` picks the
    rows to return. The neighbourhood that sets a Gaussian's initial size is
    therefore the full cloud, not the handful of points being seeded: a seed's
    size should say how dense the reconstruction is around it, not how many of
    its neighbours happen to be triangulable this stage.
    """
    from .paths import repo_root

    import sys

    vendor = repo_root() / "third_party" / "gsplat_examples"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from utils import rgb_to_sh  # type: ignore

    all_points = torch.from_numpy(np.asarray(points)).float()
    scales = _knn_scales(all_points, init_scale=init_scale)

    sel = slice(None) if select is None else np.asarray(select)
    pts = all_points[sel]
    n = pts.shape[0]
    cols = torch.from_numpy(np.asarray(rgbs)).float()[sel]
    if cols.max() > 1.0:
        cols = cols / 255.0

    quats = torch.rand((n, 4), generator=generator)
    colors = torch.zeros((n, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(cols)
    out = {
        "means": pts,
        "scales": scales[sel],
        "quats": quats,
        "opacities": torch.logit(torch.full((n,), float(init_opacity))),
        "sh0": colors[:, :1, :],
        "shN": colors[:, 1:, :],
    }
    return {k: v.to(device) for k, v in out.items()}


@torch.no_grad()
def append_gaussians(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Any],
    new: Dict[str, Tensor],
    scene=None,
    new_ids: Optional[Tensor] = None,
) -> int:
    """Append ``new`` rows to every param, in place. Returns how many.

    The running state is zeroed rather than copied, which is the one place this
    deliberately differs from ``ops.duplicate``. ``duplicate`` copies its
    parent's ``grad2d`` and ``count`` because a clone genuinely inherits that
    history. A seed has no history: it has never been rendered, let alone
    accumulated a gradient. Copying someone else's accumulators in would make a
    brand-new Gaussian immediately eligible for cloning or splitting on borrowed
    evidence, at the next refinement, before it has been seen once.

    ``ids`` is the exception -- it is identity, not history -- so it takes
    ``new_ids`` when the run is instrumented.
    """
    from gsplat.strategy.ops import _update_param_with_optimizer

    missing = [k for k in params if k not in new]
    if missing:
        # Silently appending zeros here would train, converge, and be wrong.
        raise ValueError(f"append_gaussians: no rows supplied for {missing}")

    template = params["means"]
    device = template.device
    m = int(new["means"].shape[0])
    if m == 0:
        return 0
    for k, v in new.items():
        if k in params and v.shape[1:] != params[k].shape[1:]:
            raise ValueError(
                f"append_gaussians: {k} has trailing shape {tuple(v.shape[1:])},"
                f" expected {tuple(params[k].shape[1:])}"
            )

    def param_fn(name: str, p: Tensor) -> Tensor:
        rows = new[name].to(p.device, p.dtype)
        return torch.nn.Parameter(torch.cat([p, rows]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # fresh moments, exactly as duplicate/split give their new rows
        return torch.cat([v, torch.zeros((m, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if not isinstance(v, Tensor):
            continue  # None until the first step, or a plain float like scene_scale
        if k == "ids":
            if new_ids is None:
                raise ValueError("state carries 'ids' but no new_ids were supplied")
            state[k] = torch.cat([v, new_ids.to(v.device, v.dtype)])
        else:
            state[k] = torch.cat(
                [v, torch.zeros((m, *v.shape[1:]), device=v.device, dtype=v.dtype)]
            )

    if scene is not None:
        _extend_scene(scene, m, device)
    return m


def _extend_scene(scene, m: int, device) -> None:
    """Grow the scene's per-Gaussian side arrays by ``m`` rows.

    ``GaussianScene`` keeps ``component_index`` and ``signal`` alongside the
    params and ``validate()`` insists all three match. It offers no "append
    fresh rows" callback -- every hook it has (``on_duplicate``,
    ``on_sample_add``) copies from existing rows -- and ``put()`` is documented
    init-only because it rebuilds the Parameters and would orphan the ones the
    optimizers hold.

    So: ``on_duplicate`` with an index vector pointing at row 0. New seeds
    belong to the same component as everything else, so copying row 0's
    component id is not an approximation, it is the answer. That holds only
    while the scene has one component, which is asserted rather than assumed.
    """
    names = getattr(scene, "component_names", [])
    if len(names) > 1:
        raise NotImplementedError(
            f"append_gaussians: scene has {len(names)} components {names}; which one"
            " a seed belongs to is then a real question, not row 0's answer"
        )
    scene.on_duplicate(torch.zeros(m, dtype=torch.long, device=device))
