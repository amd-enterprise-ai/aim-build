# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Harness discovery — locate and instantiate the ModelHarness for this image.

Discovery order:
  1. If ``/workspace/model/src/harness.py`` exists, dynamically import it
     and find a :class:`ModelHarness` subclass.
  2. Otherwise, raise :class:`RuntimeError` (no default harness is shipped
     yet; callers should guard with :func:`has_custom_harness` first).

When a *profile* dict is supplied, the engine name from the profile is used
to select among multiple harness classes in the same file (if any class has
an ``ENGINE`` class attribute matching the profile's engine, it wins).
This enables per-profile dispatch in multi-engine images.

``/workspace/model/src/`` is added to ``sys.path`` via a ``.pth`` file
installed by the specialized Dockerfile, so harness modules can freely
import sibling modules (e.g. ``from service import BoltzService``).
As a fallback, :func:`discover_harness` also inserts the directory at
runtime in case the ``.pth`` file is absent.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from aim_runtime.harness import ModelHarness

logger = logging.getLogger(__name__)

HARNESS_PATH = Path("/workspace/model/src/harness.py")
MODEL_DIR = str(HARNESS_PATH.parent)


def has_custom_harness() -> bool:
    """Return True if a custom harness file exists on disk."""
    return HARNESS_PATH.exists()


def discover_harness(profile: dict[str, Any] | None = None) -> ModelHarness:
    """Discover and instantiate the ModelHarness for this image.

    Args:
        profile: Optional resolved profile dict. When provided, the engine
            name (``profile["engine"]``) is used to select among multiple
            harness classes if the harness file exports more than one.
            A harness class can declare ``ENGINE = "bentoml"`` as a class
            attribute; if the profile's engine matches, that class is
            preferred.

    Returns:
        An instance of the discovered :class:`ModelHarness` subclass.

    Raises:
        RuntimeError: If a custom harness file exists but contains no
            ``ModelHarness`` subclass.
    """
    if not HARNESS_PATH.exists():
        raise RuntimeError(
            f"No custom harness found at {HARNESS_PATH}. "
            "Use has_custom_harness() to check before calling discover_harness()."
        )

    logger.warning("Loading custom harness from %s", HARNESS_PATH)

    # Ensure /workspace/model/src is importable so harness modules can do
    # sibling imports like ``from my_utils import helper``.
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
        logger.warning("Added %s to sys.path for sibling imports", MODEL_DIR)

    spec = importlib.util.spec_from_file_location("_model_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load harness module spec from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates: list[type[ModelHarness]] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ModelHarness) and obj is not ModelHarness:
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(
            f"No ModelHarness subclass found in {HARNESS_PATH}. "
            f"The file must contain a class that subclasses aim_runtime.harness.ModelHarness."
        )

    engine = (profile or {}).get("engine")

    if engine and len(candidates) > 1:
        for cls in candidates:
            cls_engine = getattr(cls, "ENGINE", None)
            if cls_engine and cls_engine == engine:
                logger.warning("Selected harness %s for engine '%s'", cls.__name__, engine)
                return cls()
        logger.warning("No harness with ENGINE='%s'; using first candidate", engine)

    selected = candidates[0]
    logger.warning("Discovered harness class: %s", selected.__name__)
    return selected()
