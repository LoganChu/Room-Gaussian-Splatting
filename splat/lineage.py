"""Gaussian lineage, without copying any gsplat code.

``InstrumentedStrategy`` subclasses ``DefaultStrategy`` and wraps the three
places the Gaussian population changes shape. It adds no behaviour: every
decision is still made by ``super()``, so an instrumented run and a vanilla run
with the same seed must produce identical results. That equivalence is the whole
point, and ``tests/test_lineage.py`` checks it.

How it works
------------
``gsplat.strategy.ops`` applies the same indexing and concatenation to **every
tensor in the strategy ``state`` dict** as it does to the params. So putting an
``ids`` tensor in ``state`` means it is re-indexed, concatenated and pruned for
free, forever. New rows arrive still carrying their *parent's* id, which is
exactly the information needed: read it, then overwrite with a fresh id.

Layout after ``_grow_gs`` (verified empirically against gsplat ``28e794ca``,
and asserted in the tests)::

    [ survivors | clones | split children ]
      n_before-n_split   n_dupli    2*n_split

- clones sit in the MIDDLE, not at the end: ``duplicate`` runs first and
  appends, then ``split`` removes its parents and appends children after.
- children are **child-major**: parent ``i``'s two children are at
  ``rest+n_dupli+i`` and ``rest+n_dupli+n_split+i``, NOT adjacent. Reading them
  as adjacent pairs yields a complete, plausible, entirely wrong lineage.
- clones are never split in the same pass (``default.py`` extends the split mask
  with ``n_dupli`` zeros), which is what keeps the three blocks disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
from gsplat.strategy import DefaultStrategy

from .events import EventLog


@dataclass
class InstrumentedStrategy(DefaultStrategy):
    """``DefaultStrategy`` plus a per-Gaussian id and an event log."""

    events: Optional[EventLog] = None
    #: assigned lazily on the first step, because N is not known until then
    _next_id: int = field(default=0, init=False, repr=False)

    # -- ids ---------------------------------------------------------------

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale)
        # Deferred like grow2d/count: the population size is unknown here, and a
        # None is skipped by the ops' isinstance(v, Tensor) check until it is real.
        state["ids"] = None
        return state

    def _ensure_ids(self, params, state: Dict[str, Any], step: int) -> None:
        if state.get("ids") is not None:
            return
        n = len(params["means"])
        device = params["means"].device
        state["ids"] = torch.arange(n, device=device, dtype=torch.int64)
        self._next_id = n
        if self.events is not None:
            # every Gaussian alive at step 0 came from the SfM point cloud
            self.events.add(step, "sfm", state["ids"].cpu().numpy())

    def _fresh(self, n: int, like: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(
            self._next_id, self._next_id + n, device=like.device, dtype=like.dtype
        )
        self._next_id += n
        return ids

    # -- hooks -------------------------------------------------------------

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        scene=None,
    ):
        self._ensure_ids(params, state, step)
        super().step_post_backward(
            params, optimizers, state, step, info, packed=packed, scene=scene
        )
        # Mirrors the reset condition in DefaultStrategy.step_post_backward: the
        # early `step >= refine_stop_iter` return means no reset happens after
        # densification stops. test_lineage pins this against the real strategy.
        if (
            self.events is not None
            and step > 0
            and step < self.refine_stop_iter
            and step % self.reset_every == 0
        ):
            self.events.add_reset(step)

    @torch.no_grad()
    def _grow_gs(
        self, params, optimizers, state: Dict[str, Any], step: int, scene=None
    ) -> Tuple[int, int]:
        n_before = state["ids"].numel()
        n_dupli, n_split = super()._grow_gs(
            params, optimizers, state, step, scene=scene
        )
        if n_dupli == 0 and n_split == 0:
            return n_dupli, n_split

        ids = state["ids"]  # the ops replaced the tensor, so re-read it
        rest = n_before - n_split
        clones = slice(rest, rest + n_dupli)
        children = slice(rest + n_dupli, rest + n_dupli + 2 * n_split)

        # New rows still hold their parent's id -- capture before overwriting.
        clone_parents = ids[clones].clone()
        child_parents = ids[children].clone()

        fresh = self._fresh(n_dupli + 2 * n_split, ids)
        ids[clones] = fresh[:n_dupli]
        ids[children] = fresh[n_dupli:]

        if self.events is not None:
            if n_dupli:
                self.events.add(
                    step, "clone", fresh[:n_dupli].cpu().numpy(), clone_parents.cpu().numpy()
                )
            if n_split:
                self.events.add(
                    step, "split", fresh[n_dupli:].cpu().numpy(), child_parents.cpu().numpy()
                )
                # The parents are gone: split keeps `rest` and appends children.
                # Logging this keeps births - deaths == the change in population.
                self.events.add(step, "death", child_parents[:n_split].cpu().numpy())
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self, params, optimizers, state: Dict[str, Any], step: int, scene=None
    ) -> int:
        before = state["ids"].clone()
        n_prune = super()._prune_gs(params, optimizers, state, step, scene=scene)
        if n_prune and self.events is not None:
            # ids are unique and never reused, so set difference is exact
            dead = before[~torch.isin(before, state["ids"])]
            self.events.add(step, "death", dead.cpu().numpy())
        return n_prune
