#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Centralizes Docker image naming for AIM builds.

Provides canonical, private, and public repository names.
Canonical: aim-{accelerator}-{org}-{model}
Private: aim-{accelerator}-{org}-{model}
Public: aim-{org}-{model} (for instinct)
"""

import re
import warnings
from dataclasses import dataclass
from typing import Optional

from aim_common.object_model import AcceleratorFamily

_SAFE_IMAGE_REF_COMPONENT_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


@dataclass(frozen=True)
class ParsedImageName:
    """Result of parsing an image repository name."""

    accelerator: str
    canonical_name_sanitized: Optional[str]
    is_base: bool
    # Always set. ``legacy_vllm`` for legacy-compatible base/model names;
    # the explicit target_id for named targets. Unrecognized names raise ValueError
    # before a ParsedImageName is constructed, so this field is never empty.
    base_target_id: str


# The one accelerator that omits its name from public image names for backward compatibility
_LEGACY_ACCELERATOR = AcceleratorFamily.INSTINCT

# The legacy vLLM base target whose assets sit at base/config.yaml root.
# Named targets (base/<target_id>/config.yaml) include their target_id in the image name.
LEGACY_VLLM_BASE_TARGET_ID = "legacy_vllm"
_TARGET_QUALIFIED_MODEL_PREFIX = "target-"
_TARGET_QUALIFIED_MODEL_SEPARATOR = "-model-"

# Built once at import time; AcceleratorFamily is a fixed enum.
_KNOWN_BASE_IMAGES: dict[str, AcceleratorFamily] = {
    **{f"aim-{f.value}-base": f for f in AcceleratorFamily},
    "aim-base": _LEGACY_ACCELERATOR,
}

# Sorted longest-value-first so that if one accelerator name is a prefix of another,
# the more-specific match wins in prefix-scan parser loops.
_FAMILIES_BY_SPECIFICITY: list[AcceleratorFamily] = sorted(AcceleratorFamily, key=lambda f: len(f.value), reverse=True)


@dataclass
class ImageName:
    """Canonical and public repository name pair for an AIM image."""

    canonical: str  # Canonical unprefixed name, e.g. 'aim-instinct-base'
    public: str  # Backward-compatible name, e.g. 'aim-base'

    @property
    def has_alias(self) -> bool:
        """True if the public name differs from canonical (needs dual-tagging)."""
        return self.canonical != self.public

    @property
    def private(self) -> str:
        """Repository name used for CI/developer pushes (same as canonical)."""
        return self.canonical


def get_image_name(
    accelerator: str,
    canonical_name_sanitized: Optional[str] = None,
    is_base: bool = False,
    base_target_id: Optional[str] = None,
) -> ImageName:
    """Return image names for any AIM image (base or model-specific).

    Compatibility wrapper around :func:`get_base_image_name` and
    :func:`get_model_image_name`.

    For base images, pass ``base_target_id`` to get the target-qualified name.
    Omitting it (or passing ``legacy_vllm``) preserves the backward-compatible
    per-accelerator naming (e.g. ``aim-instinct-base`` / ``aim-base``).
    Named base targets produce ``aim-{accelerator}-{target_id}-base`` with no public alias.
    Named model targets produce
    ``aim-{accelerator}-target-{target_id}-model-{canonical_name_sanitized}``
    with no public alias.
    """
    warnings.warn(
        "get_image_name() is deprecated and will be removed in a future version. "
        "Use get_base_image_name() or get_model_image_name() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    target_id = base_target_id or LEGACY_VLLM_BASE_TARGET_ID

    if is_base:
        return get_base_image_name(accelerator, target_id)

    if canonical_name_sanitized is None:
        raise ValueError("canonical_name_sanitized is required for model images")
    return get_model_image_name(accelerator, canonical_name_sanitized, target_id)


def get_base_image_name(accelerator: str, base_target_id: str) -> ImageName:
    """Return internal/public repository names for a base AIM image."""
    family = AcceleratorFamily.try_parse(accelerator)
    if family is None:
        valid = ", ".join(a.value for a in AcceleratorFamily)
        raise ValueError(f"Unknown accelerator: '{accelerator}'. Must be one of: {valid}")
    if base_target_id.startswith("target-"):
        raise ValueError(
            f"Invalid base_target_id: '{base_target_id}'. "
            "Must not start with 'target-' (reserved prefix for model image names)."
        )
    if not _is_safe_image_ref_component(base_target_id):
        raise ValueError(
            f"Invalid base_target_id: '{base_target_id}'. Must contain only alphanumerics, '.', '_', or '-'."
        )
    return _get_base_image_name(family, base_target_id)


def get_model_image_name(accelerator: str, canonical_name_sanitized: str, target_id: str) -> ImageName:
    """Return internal/public repository names for a model-specific AIM image."""
    family = AcceleratorFamily.try_parse(accelerator)
    if family is None:
        valid = ", ".join(a.value for a in AcceleratorFamily)
        raise ValueError(f"Unknown accelerator: '{accelerator}'. Must be one of: {valid}")
    if not _is_safe_image_ref_component(canonical_name_sanitized):
        raise ValueError(
            f"Invalid canonical_name_sanitized: '{canonical_name_sanitized}'. "
            "Must contain only alphanumerics, '.', '_', or '-'."
        )
    if not _is_safe_image_ref_component(target_id):
        raise ValueError(f"Invalid target_id: '{target_id}'. Must contain only alphanumerics, '.', '_', or '-'.")
    return _get_model_image_name(family, canonical_name_sanitized, target_id)


def parse_image_name(repository: str) -> ParsedImageName:
    """Parse an image repository name into accelerator, canonical_name_sanitized, and is_base.

    Accepts public or canonical forms, e.g. 'aim-instinct-base' or 'aim-base'.
    """

    # Order matters: more-specific formats must be attempted before legacy catch-alls.
    parsers = [
        _parse_known_base_image,
        _parse_target_qualified_model_image,
        _parse_target_qualified_base_image,
        _parse_legacy_or_standard_model_image,
    ]

    for parser in parsers:
        parsed = parser(repository)
        if parsed is not None:
            return parsed

    raise ValueError(f"Unrecognized image repository format: '{repository}'")


def _parse_known_base_image(repository: str) -> Optional[ParsedImageName]:
    """Parse legacy-compatible known base repository names."""
    matched_family = _KNOWN_BASE_IMAGES.get(repository)
    if matched_family is None:
        return None

    return ParsedImageName(
        accelerator=matched_family.value,
        canonical_name_sanitized=None,
        is_base=True,
        base_target_id=LEGACY_VLLM_BASE_TARGET_ID,
    )


def _parse_target_qualified_base_image(repository: str) -> Optional[ParsedImageName]:
    """Parse target-qualified base repository names: aim-{acc}-{target_id}-base."""
    for family in _FAMILIES_BY_SPECIFICITY:
        prefix = f"aim-{family.value}-"
        if repository.startswith(prefix) and repository.endswith("-base"):
            middle = repository[len(prefix) : -len("-base")]
            if (
                middle
                and middle != LEGACY_VLLM_BASE_TARGET_ID
                and not middle.startswith("target-")
                and _is_safe_image_ref_component(middle)
            ):
                return ParsedImageName(
                    accelerator=family.value,
                    canonical_name_sanitized=None,
                    is_base=True,
                    base_target_id=middle,
                )
    return None


def _parse_target_qualified_model_image(repository: str) -> Optional[ParsedImageName]:
    """Parse target-qualified model repository names.

    Expected format: aim-{acc}-target-{target_id}-model-{canonical_name_sanitized}
    """
    for family in _FAMILIES_BY_SPECIFICITY:
        prefix = f"aim-{family.value}-{_TARGET_QUALIFIED_MODEL_PREFIX}"
        if repository.startswith(prefix):
            remainder = repository[len(prefix) :]
            target_id, separator, canonical_name_sanitized = remainder.partition(_TARGET_QUALIFIED_MODEL_SEPARATOR)
            if (
                separator
                and target_id
                and target_id != LEGACY_VLLM_BASE_TARGET_ID
                and canonical_name_sanitized
                and _is_safe_image_ref_component(target_id)
                and _is_safe_image_ref_component(canonical_name_sanitized)
            ):
                return ParsedImageName(
                    accelerator=family.value,
                    canonical_name_sanitized=canonical_name_sanitized,
                    is_base=False,
                    base_target_id=target_id,
                )
    return None


def _parse_legacy_or_standard_model_image(repository: str) -> Optional[ParsedImageName]:
    """Parse legacy model repository names without explicit target discriminator."""
    # Check non-legacy accelerators first (their prefixes are more specific)
    ordered: list[AcceleratorFamily] = sorted(AcceleratorFamily, key=lambda f: f == _LEGACY_ACCELERATOR)

    for family in ordered:
        if family == _LEGACY_ACCELERATOR:
            for prefix in [f"aim-{family.value}-", "aim-"]:
                if repository.startswith(prefix):
                    remainder = repository[len(prefix) :]
                    if remainder and _is_safe_image_ref_component(remainder):
                        return ParsedImageName(
                            accelerator=family.value,
                            canonical_name_sanitized=remainder,
                            is_base=False,
                            base_target_id=LEGACY_VLLM_BASE_TARGET_ID,
                        )
                    break  # prefix matched but no model suffix — don't try the shorter prefix
        else:
            prefix = f"aim-{family.value}-"
            if repository.startswith(prefix):
                remainder = repository[len(prefix) :]
                if remainder and _is_safe_image_ref_component(remainder):
                    return ParsedImageName(
                        accelerator=family.value,
                        canonical_name_sanitized=remainder,
                        is_base=False,
                        base_target_id=LEGACY_VLLM_BASE_TARGET_ID,
                    )
    return None


def _get_base_image_name(family: AcceleratorFamily, base_target_id: str) -> ImageName:
    """Return canonical/public repository names for a base AIM image.

    When ``base_target_id`` is ``legacy_vllm``, the existing per-accelerator
    name is preserved (e.g. ``aim-instinct-base`` / ``aim-base``).
    Any other target ID is embedded in the name: ``aim-{acc}-{target_id}-base``
    with no public alias.
    """
    is_legacy = base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    if is_legacy:
        canonical = f"aim-{family.value}-base"
        if family == _LEGACY_ACCELERATOR:
            public = "aim-base"
        else:
            public = canonical
        return ImageName(canonical=canonical, public=public)

    # Named target: include target_id in the image name; no public alias
    canonical = f"aim-{family.value}-{base_target_id}-base"
    return ImageName(canonical=canonical, public=canonical)


def _get_model_image_name(family: AcceleratorFamily, canonical_name_sanitized: str, target_id: str) -> ImageName:
    """Return canonical/public repository names for a model-specific AIM image."""
    is_legacy = target_id == LEGACY_VLLM_BASE_TARGET_ID

    if is_legacy:
        canonical = f"aim-{family.value}-{canonical_name_sanitized}"

        if family == _LEGACY_ACCELERATOR:
            public = f"aim-{canonical_name_sanitized}"
        else:
            public = canonical

        return ImageName(canonical=canonical, public=public)

    canonical = (
        f"aim-{family.value}-{_TARGET_QUALIFIED_MODEL_PREFIX}{target_id}"
        f"{_TARGET_QUALIFIED_MODEL_SEPARATOR}{canonical_name_sanitized}"
    )
    return ImageName(canonical=canonical, public=canonical)


def parse_image_ref(image_ref: str) -> tuple[str, str, str, str]:
    """Parse a full Docker image reference into its components.

    Args:
        image_ref: Complete reference like ghcr.io/silogen/aim-instinct-base:0.3.0

    Returns:
        Tuple of (registry_host, namespace, repository, tag).

    Raises:
        ValueError: If the image reference format is invalid, uses digest syntax,
            or is missing a tag.
    """
    parts = image_ref.split("/", 2)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid image reference format: {image_ref}. " "Expected format: registry/namespace/repository:tag"
        )

    registry_host = parts[0]
    namespace = parts[1]
    repo_and_tag = parts[2]

    if "@" in repo_and_tag:
        raise ValueError(
            f"Digest references are not supported: {image_ref}. " "Expected format: registry/namespace/repository:tag"
        )

    if ":" not in repo_and_tag:
        raise ValueError(
            f"Image reference missing tag: {image_ref}. " "Expected format: registry/namespace/repository:tag"
        )

    repository, tag = repo_and_tag.rsplit(":", 1)
    if ":" in repository:
        raise ValueError(
            f"Invalid image reference format: {image_ref}. "
            "Repository must not contain ':'; expected format: registry/namespace/repository:tag"
        )

    if not all(_is_safe_image_ref_component(c) for c in (namespace, repository, tag)):
        raise ValueError(
            f"Invalid image reference format: {image_ref}. "
            "namespace, repository, and tag may contain only alphanumerics, '.', '_', or '-'."
        )

    return registry_host, namespace, repository, tag


def _is_safe_image_ref_component(value: str) -> bool:
    """Return True when a single image-ref component uses a conservative safe charset.

    This helper intentionally allows only alphanumerics plus ``.``, ``_``, and ``-``
    for use in internal validation of registry components.
    """
    return bool(value) and _SAFE_IMAGE_REF_COMPONENT_RE.fullmatch(value) is not None
