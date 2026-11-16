#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Enumeration of *specialized* base images for AIM model builds.

A "specialized" base is an extra image layer (Layer 1) built from an ``image/``
directory that sits between the upstream vendor image and the standard AIM base.
The ``image/`` directory can live at two levels:

* **Engine-level** ``assets/<acc>/engines/<engine>/image/`` — a shared base
  ``aim-<acc>-<engine>-base`` for every model whose profiles use that engine.
* **Model-level** ``assets/<acc>/<org>/<model>/image/`` — a model-dedicated base
  ``aim-<acc>-<org>-<model>-base``.

CI discovers these targets (see :func:`enumerate_specialized_base_targets`) and
builds the layer stack — there is no manual/local resolution path::

    Layer 0: Upstream vendor image (e.g. python:3.12-slim, rocm/pytorch)
        v  FROM
    Layer 1: Specialized base  (aim-<acc>-specialized-<key>)
        v  FROM (build-arg)
    Layer 2: AIM base          (aim-<acc>-<target_id>-base)
        v  FROM (build-arg)
    Layer 3: Model image       (aim-<acc>-model-<org>-<model>)

For a model-dedicated base the target_id equals the model's own canonical name,
so the Layer 3 image drops the duplicated target segment and is simply
``aim-<acc>-model-<org>-<model>`` (an engine-shared base instead yields the
target-qualified ``aim-<acc>-target-<engine>-model-<org>-<model>``).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aim_common.object_model import AcceleratorFamily, Engine, JobType
from aim_utils.asset_utils import ENGINE_LEAF_DIRNAME, MODEL_LEAF_DIRNAME, PROFILES_DIRNAME
from aim_utils.image_naming import (
    LEGACY_VLLM_BASE_TARGET_ID,
    ImageName,
    get_base_image_name,
    get_model_image_name,
)
from aim_utils.yaml_utils import read_yaml

logger = logging.getLogger(__name__)

# What reading a YAML file can raise: I/O, a missing file (read_yaml raises ValueError), bad YAML.
_YAML_READ_ERRORS = (OSError, ValueError, yaml.YAMLError)

# Convention constants -------------------------------------------------------

#: Sub-directory (relative to a model or engine asset dir) holding the
#: specialized base build context.
IMAGE_SUBDIR = "image"
#: Dockerfile name inside :data:`IMAGE_SUBDIR`.
DOCKERFILE_NAME = "Dockerfile"
#: Directory under ``assets/<acc>/`` that groups engine-level assets.
ENGINES_DIRNAME = "engines"
#: Directory under ``assets/<acc>/`` that groups base image assets (config + profiles).
BASE_DIRNAME = "base"
#: Sub-directory of a base asset dir holding ``engines.yaml`` and other engine config.
CONFIG_SUBDIR = "config"
#: Filename (under :data:`CONFIG_SUBDIR`) declaring the engines a base ships.
ENGINES_FILENAME = "engines.yaml"


def _sanitize_name(name: str) -> str:
    """Return a Docker-safe slug (mirrors ``ci/model.py::_sanitize_name``)."""
    sanitized = name.lower()
    sanitized = re.sub(r"[^a-z0-9._-]", "-", sanitized)
    sanitized = re.sub(r"[-_.]+", "-", sanitized)
    return sanitized.strip("-_.")


def model_target_id(org: str, model: str) -> str:
    """Target id for a model-level ``image/`` dir: sanitized ``<org>-<model>``.

    Used both for a model-dedicated base and, for a job AIM, as the naming target
    that makes the image ``aim-<acc>-model-<org>-<model>``.
    """
    return f"{_sanitize_name(org)}-{_sanitize_name(model)}"


def _layer1_image_name(accelerator_family: str, key: str) -> ImageName:
    """Return the Layer 1 specialized base name for a sanitized ``key``.

    Routes through :func:`get_model_image_name` so the canonical/public alias
    rules stay consistent with every other AIM image (e.g. instinct drops the
    accelerator segment from the public name). The legacy target keeps the
    ``aim-<acc>-specialized-<key>`` form (no ``target-`` discriminator).
    """
    return get_model_image_name(accelerator_family, f"specialized-{key}", LEGACY_VLLM_BASE_TARGET_ID)


def is_job_aim(model_asset_dir: str | Path) -> bool:
    """Return True when every profile of a model asset is a job profile.

    A job AIM's ``image/Dockerfile`` builds the complete image (no aim-runtime, no
    HTTP server), so it is not a specialized base: CI builds it directly as the
    model image and runs it as a Kubernetes Job. A profile counts as a job profile
    when ``metadata.engine`` is ``job`` or ``metadata.job_type`` is
    ``long_running``. A model with no readable profiles is not a job AIM.
    """
    profiles_dir = Path(model_asset_dir) / PROFILES_DIRNAME
    if not profiles_dir.is_dir():
        return False
    profile_paths = sorted(profiles_dir.rglob("*.yaml"))
    if not profile_paths:
        return False
    for profile_path in profile_paths:
        try:
            profile = read_yaml(profile_path) or {}
        except _YAML_READ_ERRORS as exc:
            logger.warning(f"Cannot read {profile_path}, so {model_asset_dir} is not treated as a job AIM: {exc}")
            return False
        metadata = profile.get("metadata") if isinstance(profile, dict) else None
        if not isinstance(metadata, dict):
            return False
        engine = str(metadata.get("engine", "")).lower()
        if engine != Engine.JOB.value and metadata.get("job_type") != JobType.LONG_RUNNING.value:
            return False
    return True


def base_leaf_declares_dockerfile(base_leaf_dir: str | Path) -> bool:
    """Return True for a base leaf: one whose ``config.yaml`` names its own ``dockerfile:``.

    Every file in such a leaf is a build input of that one base image.
    """
    config_path = Path(base_leaf_dir) / "config.yaml"
    if not config_path.is_file():
        return False
    try:
        config = read_yaml(config_path) or {}
    except _YAML_READ_ERRORS as exc:
        logger.warning(f"Cannot read {config_path}, so its leaf is not treated as declaring a Dockerfile: {exc}")
        return False
    return isinstance(config, dict) and bool(config.get("dockerfile"))


def model_leaf_dockerfile_config(file_path: str) -> str | None:
    """Return the ``config.yaml`` of the model leaf whose own Dockerfile ``file_path`` is.

    Matches ``assets/<acc>/model/<org>/<model>/Dockerfile`` only.
    """
    parts = file_path.replace("\\", "/").strip().split("/")
    if len(parts) != 6 or parts[0] != "assets" or parts[2] != MODEL_LEAF_DIRNAME or parts[5] != DOCKERFILE_NAME:
        return None
    return "/".join([*parts[:5], "config.yaml"])


def parse_specialized_target_from_path(
    file_path: str,
    assets_root: str = "assets",
) -> tuple[AcceleratorFamily, str] | None:
    """Return ``(accelerator_family, target_id)`` for specialized ``image/`` paths.

    Accepted layouts:

    * ``assets/<acc>/engines/<engine>/image/**`` -> ``(<AcceleratorFamily>, sanitize(<engine>))``
    * ``assets/<acc>/<org>/<model>/image/**`` -> ``(<AcceleratorFamily>, sanitize(<org>)-sanitize(<model>))``

    Returns ``None`` when the path does not represent a real specialized build
    context (missing ``image/Dockerfile``) or does not match the supported
    conventions.
    """
    normalized = file_path.replace("\\", "/").strip()
    root = assets_root.strip("/")
    if not normalized.startswith(f"{root}/"):
        return None

    image_dir, sep, _ = normalized.partition(f"/{IMAGE_SUBDIR}/")
    if not sep:
        return None

    dockerfile_path = Path(image_dir) / IMAGE_SUBDIR / DOCKERFILE_NAME
    if not os.path.isfile(dockerfile_path):
        return None

    parts = normalized.split("/")
    root_parts = root.split("/")
    if len(parts) < len(root_parts) + 3 or parts[: len(root_parts)] != root_parts:
        return None

    acc_index = len(root_parts)
    accelerator = AcceleratorFamily.try_parse(parts[acc_index])
    if accelerator is None:
        return None

    # Engine-level: assets/<acc>/engines/<engine>/image/**
    if len(parts) >= acc_index + 4 and parts[acc_index + 1] == ENGINES_DIRNAME:
        target_id = _sanitize_name(parts[acc_index + 2])
        return (accelerator, target_id) if target_id else None

    # Model-level: assets/<acc>/<org>/<model>/image/**
    if len(parts) < acc_index + 4:
        return None
    if parts[acc_index + 1] in {BASE_DIRNAME, ENGINES_DIRNAME, ENGINE_LEAF_DIRNAME, MODEL_LEAF_DIRNAME}:
        return None
    if is_job_aim(Path(image_dir)):
        return None
    if not _sanitize_name(parts[acc_index + 1]) or not _sanitize_name(parts[acc_index + 2]):
        return None
    return (accelerator, model_target_id(parts[acc_index + 1], parts[acc_index + 2]))


@dataclass(frozen=True)
class SpecializedBaseTarget:
    """A specialized base image build target for CI enumeration.

    Describes the two extra layers CI must produce so the base is obtainable
    from the registry (no manual pushes): the Layer 1 specialized image (built
    from the ``image/`` directory) and the consumable Layer 2 AIM base built on
    top of it.
    """

    #: Base target discriminator used in the Layer 2 base name (engine id or model key).
    target_id: str
    #: Where the ``image/`` directory was found: ``"model"`` or ``"engine"``.
    scope: str
    #: Build context (the ``image/`` directory) as a POSIX path.
    context_path: str
    #: Path to the Layer 1 Dockerfile inside the build context.
    dockerfile: str
    #: Name pair for the Layer 1 specialized image (``aim-<acc>-specialized-<key>``).
    layer1_image_name: ImageName
    #: Name pair for the consumable Layer 2 AIM base (``aim-<acc>-<target_id>-base``).
    base_image_name: ImageName

    @property
    def layer1_repository(self) -> str:
        return self.layer1_image_name.canonical

    @property
    def base_repository(self) -> str:
        return self.base_image_name.canonical


def enumerate_specialized_base_targets(
    accelerator_family: str,
    assets_root: str = "assets",
) -> list[SpecializedBaseTarget]:
    """Enumerate every specialized base build target for an accelerator family.

    Scans both convention levels and returns one target per discovered
    ``image/`` directory:

    * **Engine-level** ``assets/<acc>/engines/<engine>/image/`` → shared base
      ``aim-<acc>-<engine>-base`` (target_id = engine).
    * **Model-level** ``assets/<acc>/<org>/<model>/image/`` → model-dedicated
      base ``aim-<acc>-<org>-<model>-base`` (target_id = sanitized ``<org>-<model>``).
      Because the target_id equals the model's own canonical name, the consuming
      model image drops the duplicated target segment and is simply
      ``aim-<acc>-model-<org>-<model>`` (e.g. ``aim-instinct-model-openfold-openfold3``).

    Job AIMs (:func:`is_job_aim`) are skipped: their ``image/`` builds the final
    image, not a base.

    Results are sorted by ``target_id`` for deterministic CI matrices.
    """
    acc = accelerator_family.lower()
    acc_root = Path(assets_root) / acc
    targets: list[SpecializedBaseTarget] = []
    seen_target_ids: set[str] = set()

    if not acc_root.is_dir():
        return targets

    # Engine-level: one shared base per engine directory.
    engines_root = acc_root / ENGINES_DIRNAME
    if engines_root.is_dir():
        for engine_dir in sorted(engines_root.iterdir()):
            if not engine_dir.is_dir():
                continue
            image_dir = engine_dir / IMAGE_SUBDIR
            dockerfile = image_dir / DOCKERFILE_NAME
            if not dockerfile.is_file():
                continue
            target_id = _sanitize_name(engine_dir.name)
            if target_id in seen_target_ids:
                continue
            seen_target_ids.add(target_id)
            targets.append(
                SpecializedBaseTarget(
                    target_id=target_id,
                    scope="engine",
                    context_path=image_dir.as_posix(),
                    dockerfile=dockerfile.as_posix(),
                    layer1_image_name=_layer1_image_name(acc, target_id),
                    base_image_name=get_base_image_name(acc, target_id),
                )
            )

    # Model-level: model-dedicated base per <org>/<model>/image/ directory.
    for org_dir in sorted(acc_root.iterdir()):
        if not org_dir.is_dir() or org_dir.name in {
            BASE_DIRNAME,
            ENGINES_DIRNAME,
            ENGINE_LEAF_DIRNAME,
            MODEL_LEAF_DIRNAME,
        }:
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            image_dir = model_dir / IMAGE_SUBDIR
            dockerfile = image_dir / DOCKERFILE_NAME
            if not dockerfile.is_file() or is_job_aim(model_dir):
                continue
            key = model_target_id(org_dir.name, model_dir.name)
            if key in seen_target_ids:
                continue
            seen_target_ids.add(key)
            targets.append(
                SpecializedBaseTarget(
                    target_id=key,
                    scope="model",
                    context_path=image_dir.as_posix(),
                    dockerfile=dockerfile.as_posix(),
                    layer1_image_name=_layer1_image_name(acc, key),
                    base_image_name=get_base_image_name(acc, key),
                )
            )

    targets.sort(key=lambda t: t.target_id)
    return targets


def _infer_model_engine(model_asset_dir: Path) -> str | None:
    """Infer the engine a model-dedicated base serves from the model's profiles.

    Profiles are named ``{engine}-{accelerator}-{precision}-tp{n}-{metric}.yaml``;
    the engine prefix is authoritative and kept in sync with ``metadata.engine`` by
    ``aim_utils.profile_utils check-metadata``. Reading the filename avoids parsing
    every profile and means no new ``engine`` field is needed in the specialized
    image's ``config.yaml``.

    Returns the single engine used by the profiles, or ``None`` when no profile is
    found or the profiles disagree (in which case callers fall back to the
    accelerator base root).
    """
    profiles_dir = model_asset_dir / PROFILES_DIRNAME
    if not profiles_dir.is_dir():
        return None
    engines = {p.stem.split("-", 1)[0] for p in profiles_dir.rglob("*.yaml") if p.stem.split("-", 1)[0]}
    if len(engines) == 1:
        return next(iter(engines))
    return None


def _base_dir_for_engine(base_root: Path, engine: str) -> Path | None:
    """Return the named base dir under ``base_root`` whose engines.yaml declares ``engine``.

    Matching on the engine key (rather than the directory name) is robust to the
    dir-vs-engine-id spelling differences — e.g. the ``vllm-omni`` base dir declares
    the ``vllm_omni`` engine.
    """
    if not base_root.is_dir():
        return None
    for child in sorted(base_root.iterdir()):
        engines_file = child / CONFIG_SUBDIR / ENGINES_FILENAME
        if not engines_file.is_file():
            continue
        try:
            declared = read_yaml(engines_file) or {}
        except Exception:
            continue
        if engine in declared:
            return child
    return None


def resolve_base_assets_dir(
    accelerator_family: str,
    base_target_id: str,
    assets_root: str = "assets",
) -> str:
    """Resolve the assets dir a base build copies its engine config + general profiles from.

    The returned dir is expected to contain ``config/engines.yaml`` and
    ``profiles/general/``; the base Dockerfile copies these into the image. Resolution
    order:

    1. **Named base dir** ``assets/<acc>/base/<target_id>/`` with its own ``config.yaml``
       — covers engine-level named bases such as ``bentoml`` / ``vllm-omni``.
    2. **Model-dedicated specialized base** (``target_id`` like ``<org>-<model>``): infer
       the engine from the model's profiles and use the named base dir that declares it,
       so a BentoML model base ships BentoML's engines.yaml and general profiles instead
       of the generic vLLM ones.
    3. **Fallback**: the accelerator base root ``assets/<acc>/base`` (standard
       ``legacy_vllm`` behaviour).
    """
    acc = accelerator_family.lower()
    acc_root = Path(assets_root) / acc
    base_root = acc_root / BASE_DIRNAME

    is_named_candidate = bool(base_target_id) and base_target_id != LEGACY_VLLM_BASE_TARGET_ID

    # 1. Named base target shipping its own config.yaml.
    named_dir = base_root / base_target_id
    if is_named_candidate and (named_dir / "config.yaml").is_file():
        return named_dir.as_posix()

    # 2. Model-dedicated specialized base — match the engine its profiles use.
    if is_named_candidate:
        for target in enumerate_specialized_base_targets(acc, assets_root):
            if target.target_id != base_target_id or target.scope != "model":
                continue
            engine = _infer_model_engine(Path(target.context_path).parent)
            if engine:
                engine_dir = _base_dir_for_engine(base_root, engine)
                if engine_dir is not None:
                    return engine_dir.as_posix()
            break

    # 3. Standard accelerator base root.
    return base_root.as_posix()


def resolve_asset_test_dir(
    accelerator_family: str,
    base_target_id: str,
    assets_root: str = "assets",
) -> str | None:
    """Return the ``image/tests`` dir of the model asset that owns ``base_target_id``.

    Only a model-dedicated base (target_id ``<org>-<model>``) has such an asset. Returns
    ``None`` for any other base, or when the asset has no ``image/tests`` directory.
    """
    for target in enumerate_specialized_base_targets(accelerator_family, assets_root):
        if target.target_id != base_target_id or target.scope != "model":
            continue
        tests_dir = Path(target.context_path) / "tests"
        return tests_dir.as_posix() if tests_dir.is_dir() else None
    return None
