# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Harness discovery — locate and instantiate the ModelHarness for this image.

Discovery order:
  1. If ``/workspace/model/src/harness.py`` exists, dynamically import it
     and find a :class:`ModelHarness` subclass.
  2. Otherwise, fall back to :class:`VLLMHarness`, the batteries-included
     harness shipped with aim-runtime for standard vLLM-based images. The
     fallback only applies to engines VLLMHarness can drive; for any other
     engine a missing harness file is an error, not a default.

When a *profile* dict is supplied, the engine name from the profile is used
to select among multiple harness classes in the same file (if any class names
the profile's engine in ``ENGINE`` or ``SUPPORTED_ENGINES``, it wins).
This enables per-profile dispatch in multi-engine images. A multi-harness
file where nothing declares the resolved engine is an error rather than an
arbitrary pick.

``/workspace/model/src/`` is added to ``sys.path`` via a ``.pth`` file
installed by the specialized Dockerfile, so harness modules can freely
import sibling modules (e.g. ``from service import BoltzService``).
As a fallback, :func:`discover_harness` also inserts the directory at
runtime in case the ``.pth`` file is absent.

A shared base class may live in a sibling module and be imported into
``harness.py``; only the most-derived class is selected, so a concrete base
like ``MonaiHarness`` never outranks the per-model leaf that subclasses it.
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


def _names_own_engines(cls: type[ModelHarness]) -> bool:
    """Return True if ``cls`` itself names its engines, rather than inheriting them."""
    return "ENGINE" in cls.__dict__ or "SUPPORTED_ENGINES" in cls.__dict__


def _declared_engines(cls: type[ModelHarness]) -> frozenset[str]:
    """Engines ``cls`` answers to, from ``SUPPORTED_ENGINES`` or ``ENGINE``.

    Both spellings are honoured so a subclass of :class:`VLLMHarness` (which
    declares ``SUPPORTED_ENGINES``) is selectable in a multi-harness file.
    Naming engines on the class replaces what it inherits instead of adding to
    it — a subclass declaring ``ENGINE = "bentoml"`` drives bentoml only, even
    though it inherits its base's vLLM engines.
    """
    if _names_own_engines(cls):
        single, supported = cls.__dict__.get("ENGINE"), cls.__dict__.get("SUPPORTED_ENGINES")
    else:
        single, supported = getattr(cls, "ENGINE", None), getattr(cls, "SUPPORTED_ENGINES", None)
    engines = set(supported or ())
    if single is not None:
        engines.add(single)
    return frozenset(engines)


def _is_base_of_any(cls: type[ModelHarness], classes: list[type[ModelHarness]]) -> bool:
    """Return True if any class in ``classes`` inherits from ``cls``."""
    return any(other is not cls and issubclass(other, cls) for other in classes)


def _most_derived(classes: list[type[ModelHarness]]) -> list[type[ModelHarness]]:
    """Drop classes that another class in ``classes`` inherits from."""
    return [cls for cls in classes if not _is_base_of_any(cls, classes)]


def _prefer_local(classes: list[type[ModelHarness]], module_name: str) -> type[ModelHarness]:
    """Pick one class, favouring those written in the harness file itself."""
    local = [cls for cls in classes if cls.__module__ == module_name]
    return (local or classes)[0]


def discover_harness(profile: dict[str, Any] | None = None) -> ModelHarness:
    """Discover and instantiate the ModelHarness for this image.

    Args:
        profile: Optional resolved profile dict. When provided, the engine
            name (``profile["engine"]``) is used to select among multiple
            harness classes if the harness file exports more than one.
            A harness class can declare ``ENGINE = "bentoml"`` or
            ``SUPPORTED_ENGINES = frozenset({...})`` as a class attribute; if
            the profile's engine matches either, that class is preferred.

    Note:
        Base classes are ignored when a candidate inherits from them, so a
        per-model harness may import its shared base from a sibling module.
        A base that declares its own engines is exempt and stays selectable.
        Harnesses shipped by aim-runtime (e.g. :class:`VLLMHarness`) never
        compete with a class from the harness file.

    Returns:
        An instance of the discovered :class:`ModelHarness` subclass, or a
        :class:`VLLMHarness` when no custom harness file is present.

    Raises:
        RuntimeError: If a custom harness file exists but contains no
            ``ModelHarness`` subclass, if it exports several and none declares
            the profile's engine, or if no harness file exists and the
            profile's engine is one :class:`VLLMHarness` cannot drive.
    """
    engine = (profile or {}).get("engine")

    if not has_custom_harness():
        from aim_runtime.harness.vllm_harness import VLLMHarness

        # An engine VLLMHarness can't drive means this image was supposed to
        # ship a harness and didn't. Fail here rather than let the vLLM checks
        # probe endpoints the service doesn't have and report it as a dead
        # service. The engine is unknown when profile resolution failed, in
        # which case falling back is still the best guess.
        if engine and engine not in VLLMHarness.SUPPORTED_ENGINES:
            raise RuntimeError(
                f"No harness at {HARNESS_PATH}, but the resolved profile uses engine "
                f"'{engine}', which VLLMHarness cannot drive. This image is expected "
                f"to ship image/src/harness.py — check that the asset's harness was "
                f"copied into the image."
            )

        logger.info("No custom harness at %s — using VLLMHarness", HARNESS_PATH)
        return VLLMHarness()

    logger.info("Loading custom harness from %s", HARNESS_PATH)

    # Ensure /workspace/model/src is importable so harness modules can do
    # sibling imports like ``from my_utils import helper``.
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
        logger.info("Added %s to sys.path for sibling imports", MODEL_DIR)

    spec = importlib.util.spec_from_file_location("_model_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load harness module spec from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Harnesses shipped with aim-runtime are never candidates: an image that
    # subclasses or merely imports VLLMHarness would otherwise offer it up too,
    # and since getmembers sorts alphabetically the import could outrank the
    # image's own harness.
    candidates: list[type[ModelHarness]] = [
        obj
        for _name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ModelHarness) and obj is not ModelHarness and not obj.__module__.startswith("aim_runtime.")
    ]

    # A candidate that another candidate inherits from is shared scaffolding, so
    # a per-model leaf can import its base from a sibling module without the base
    # shadowing it. A class naming its own engines is kept: in a multi-engine
    # image one harness may subclass another and both must stay selectable.
    candidates = [cls for cls in candidates if _names_own_engines(cls) or not _is_base_of_any(cls, candidates)]

    if not candidates:
        raise RuntimeError(
            f"No ModelHarness subclass found in {HARNESS_PATH}. "
            f"The file must contain (or import) a class that subclasses aim_runtime.harness.ModelHarness."
        )

    if engine and len(candidates) > 1:
        matches = [cls for cls in candidates if engine in _declared_engines(cls)]
        if matches:
            # Several matches means a subclass inherited its base's engines
            # without narrowing them; the subclass is the specialization.
            selected = _prefer_local(_most_derived(matches), module.__name__)
            logger.info("Selected harness %s for engine '%s'", selected.__name__, engine)
            return selected()
        declared = ", ".join(f"{cls.__name__}(engines={sorted(_declared_engines(cls))})" for cls in candidates)
        raise RuntimeError(
            f"{HARNESS_PATH} exports {len(candidates)} harness classes but none declares "
            f"engine '{engine}' (found: {declared}). Add an ENGINE (or SUPPORTED_ENGINES) "
            f"class attribute covering the profile's engine to the harness that should drive "
            f"it. Harness classes imported into the file count as candidates; import the "
            f"module instead of the class if one of these was not meant to be selectable."
        )

    selected = _prefer_local(candidates, module.__name__)
    if len(candidates) > 1:
        logger.warning(
            "%s exports %d harnesses and none was selected by engine; using %s",
            HARNESS_PATH,
            len(candidates),
            selected.__name__,
        )
    logger.info("Discovered harness class: %s", selected.__name__)
    return selected()
