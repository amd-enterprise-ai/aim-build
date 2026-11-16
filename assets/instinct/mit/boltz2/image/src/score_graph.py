# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CUDA graph of the diffusion score model for one sampler run.

The score model is called once per diffusion step (200 times in the affinity
pass) with the same conditioning tensors and new noisy coordinates. For small
complexes the host cannot launch its kernels as fast as the GPU runs them. The
graph replays all of them with one launch and runs the same kernels as the
eager model, so the result does not change.

Per sampler run: the first call runs eager (warm-up), the second call captures
the graph, later calls copy the new inputs into the static buffers and replay.
With ``compile_count``, a call captures only if the eager call before it did not
compile: a Dynamo compile during the capture would fail it. A run whose first
call compiles thus captures at its third call.
The graph belongs to one run, because its conditioning tensors are new in each
run; it is freed when the run ends. A capture that fails turns the graph off
for the rest of the process and the model runs eager.

Do not compile the score model itself, only modules inside it: a captured graph
cannot hold a Dynamo trace of this wrapper or Inductor's own CUDA graphs.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Most eager calls of one run that may compile before the run stays eager.
MAX_COMPILING_CALLS = 3


class _Run:
    """State of the graph for one sampler run."""

    def __init__(self, settled: bool = True) -> None:
        self.settled = settled
        self.compiling_calls = 0
        self.key: Any = None
        self.graph: Any = None
        self.r_noisy: Any = None
        self.times: Any = None
        self.out: Any = None


class ScoreGraph:
    """Replay a captured ``DiffusionModule.forward`` between the start and the end of a sampler run."""

    def __init__(self, forward: Callable[..., Any], compile_count: Callable[[], int] | None = None) -> None:
        self._forward = forward
        self._compile_count = compile_count
        self._local = threading.local()
        self.enabled = True
        self.replays = 0

    @property
    def active(self) -> bool:
        return getattr(self._local, "run", None) is not None

    def start_run(self) -> None:
        self._local.run = _Run()

    def end_run(self) -> None:
        self._local.run = None

    def __call__(self, r_noisy: Any, times: Any, **conditioning: Any) -> Any:
        run = getattr(self._local, "run", None)
        if run is None or not self.enabled or not r_noisy.is_cuda:
            return self._forward(r_noisy=r_noisy, times=times, **conditioning)
        key = (
            tuple(r_noisy.shape),
            tuple(times.shape),
            tuple(id(value) for value in conditioning.values()),
        )
        if key != run.key:
            self._local.run = run = _Run(settled=self._compile_count is None)
            run.key = key
            return self._eager(run, r_noisy, times, conditioning)
        if run.graph is None and not run.settled:
            if run.compiling_calls >= MAX_COMPILING_CALLS:
                return self._forward(r_noisy=r_noisy, times=times, **conditioning)
            return self._eager(run, r_noisy, times, conditioning)
        if run.graph is None and not self._capture(run, r_noisy, times, conditioning):
            return self._forward(r_noisy=r_noisy, times=times, **conditioning)
        run.r_noisy.copy_(r_noisy)
        run.times.copy_(times)
        run.graph.replay()
        self.replays += 1
        return run.out.clone()

    def _eager(self, run: _Run, r_noisy: Any, times: Any, conditioning: dict[str, Any]) -> Any:
        if self._compile_count is None:
            return self._forward(r_noisy=r_noisy, times=times, **conditioning)
        before = self._compile_count()
        out = self._forward(r_noisy=r_noisy, times=times, **conditioning)
        run.settled = self._compile_count() == before
        if not run.settled:
            run.compiling_calls += 1
            if run.compiling_calls == MAX_COMPILING_CALLS:
                logger.warning(
                    "The score model still compiles after %d eager calls; this run stays eager", MAX_COMPILING_CALLS
                )
        return out

    def _capture(self, run: _Run, r_noisy: Any, times: Any, conditioning: dict[str, Any]) -> bool:
        import torch

        try:
            graph = torch.cuda.CUDAGraph()
            run.r_noisy = r_noisy.clone()
            run.times = times.clone()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                run.out = self._forward(r_noisy=run.r_noisy, times=run.times, **conditioning)
        except Exception:  # noqa: BLE001
            logger.warning("CUDA graph capture of the score model failed; running it eager", exc_info=True)
            self.enabled = False
            run.graph = run.out = run.r_noisy = run.times = None
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                logger.warning("CUDA synchronize after the failed capture failed", exc_info=True)
            return False
        run.graph = graph
        return True


def install_score_graph(model: Any, compile_count: Callable[[], int] | None = None) -> ScoreGraph:
    """Run ``model.structure_module.score_model`` through a CUDA graph inside ``structure_module.sample``.

    ``compile_count`` returns the number of Dynamo compiles so far in the whole model, not only in the score
    model: a compile anywhere in an eager call delays the capture to a later call.
    """
    structure_module = model.structure_module
    score_model = structure_module.score_model
    graph = ScoreGraph(score_model.forward, compile_count)
    sample = structure_module.sample

    def sample_with_graph(*args: Any, **kwargs: Any) -> Any:
        graph.start_run()
        try:
            return sample(*args, **kwargs)
        finally:
            graph.end_run()

    score_model.forward = graph
    structure_module.sample = sample_with_graph
    return graph
