# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Benchmark recipes for the vLLM-Omni diffusion harness.

These map ``recipe`` names to argv fragments passed to
``benchmarks/diffusion/diffusion_benchmark_serving.py``.  The default set
mirrors the Dataset A / B / C tables in
``vllm-omni/benchmarks/diffusion/performance_dashboard/wan_2_2_serving_performance.md``
plus a fast ``smoke`` configuration suitable for ``validate``.

Each recipe is a dict with:

- ``num_prompts``      : ``--num-prompts``
- ``max_concurrency``  : ``--max-concurrency``
- ``random_request_config`` : a list of dicts forwarded as the JSON-encoded
  ``--random-request-config`` argument.  Each dict can carry ``width``,
  ``height``, ``num_inference_steps``, ``num_frames``, ``fps`` and a sampling
  ``weight``.

Add new recipes here when you want to expose more presets via
``aim-runtime benchmark --config <yaml>``::

    # bench.yaml
    recipe: dataset_b_720p

The harness reads ``HarnessConfig.get("recipe", DEFAULT_RECIPE)`` so values
from ``--config`` take precedence over the profile and over the default.
"""

from __future__ import annotations

from typing import Any

DEFAULT_RECIPE = "dataset_a_480p"

RECIPES: dict[str, dict[str, Any]] = {
    # Tiny configuration for smoke / wiring checks.  Two prompts, very few
    # inference steps; still hits the same /v1/videos code path the full
    # benchmark does.
    "smoke": {
        "num_prompts": 2,
        "max_concurrency": 1,
        "random_request_config": [
            {
                "width": 854,
                "height": 480,
                "num_inference_steps": 3,
                "num_frames": 33,
                "fps": 16,
                "weight": 1,
            }
        ],
    },
    # Dataset A (480p) from the wan2.2 perf dashboard.  ``num_inference_steps``
    # uses the dashboard's "Example Benchmark Command" value (18) which gives
    # a meaningful per-step cost while keeping wall-clock reasonable.
    "dataset_a_480p": {
        "num_prompts": 10,
        "max_concurrency": 1,
        "random_request_config": [
            {
                "width": 854,
                "height": 480,
                "num_inference_steps": 18,
                "num_frames": 33,
                "fps": 16,
                "weight": 1,
            }
        ],
    },
    # Dataset B (720p).  Heavier; expect roughly 5x the per-request latency
    # of Dataset A on the same hardware.
    "dataset_b_720p": {
        "num_prompts": 10,
        "max_concurrency": 1,
        "random_request_config": [
            {
                "width": 1280,
                "height": 720,
                "num_inference_steps": 6,
                "num_frames": 80,
                "fps": 16,
                "weight": 1,
            }
        ],
    },
    # Dataset C (mixed resolution).  Stresses the scheduler with a realistic
    # weight distribution.
    "dataset_c_mix": {
        "num_prompts": 20,
        "max_concurrency": 1,
        "random_request_config": [
            {
                "width": 854,
                "height": 480,
                "num_inference_steps": 3,
                "num_frames": 80,
                "fps": 16,
                "weight": 0.15,
            },
            {
                "width": 854,
                "height": 480,
                "num_inference_steps": 4,
                "num_frames": 120,
                "fps": 24,
                "weight": 0.25,
            },
            {
                "width": 1280,
                "height": 720,
                "num_inference_steps": 6,
                "num_frames": 80,
                "fps": 16,
                "weight": 0.6,
            },
        ],
    },
}


def get_recipe(name: str | None) -> dict[str, Any]:
    """Return a recipe dict by name; falls back to :data:`DEFAULT_RECIPE`.

    Raises :class:`KeyError` if *name* is provided but not registered, so
    typos surface loudly instead of silently running the default.
    """
    if name is None:
        return RECIPES[DEFAULT_RECIPE]
    if name not in RECIPES:
        raise KeyError(f"Unknown recipe '{name}'.  Available: {sorted(RECIPES)}")
    return RECIPES[name]
