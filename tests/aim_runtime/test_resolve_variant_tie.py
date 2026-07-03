# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the _resolve_variant_tie helper in profile_selector."""

import sys
from pathlib import Path
from typing import Optional

import pytest

src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from aim_common.object_model import (  # noqa: E402
    AcceleratorModel,
    Engine,
    Metric,
    Precision,
    ProfileMetadata,
    ProfileType,
)
from aim_runtime.object_model import Profile, ProfileHandling  # noqa: E402
from aim_runtime.profile_selector import ProfileNotFound, _resolve_variant_tie  # noqa: E402


def _make_profile(variant: Optional[str] = None) -> Profile:
    """Build a minimal Profile for testing _resolve_variant_tie."""
    metadata = ProfileMetadata(
        engine=Engine.VLLM,
        accelerator_model=AcceleratorModel.MI300X,
        precision=Precision.FP16,
        accelerator_count=1,
        metric=Metric.LATENCY,
        manual_selection_only=False,
        type=ProfileType.OPTIMIZED,
        variant=variant,
    )
    filename = f"{metadata.accelerator_label}.yaml"
    handling = ProfileHandling(
        path=f"/fake/profiles/{filename}",
        filename=filename,
        priority=1,
    )
    return Profile(
        profile_handling=handling,
        metadata=metadata,
        aim_id="test/Model",
        model_id="test/Model",
        engine_args={},
        env_vars={},
    )


class TestResolveVariantTie:
    """Tests for _resolve_variant_tie."""

    def test_single_candidate_returned_directly(self):
        """When there is only one candidate, return it directly."""
        p = _make_profile()
        assert _resolve_variant_tie([p]) is p

    def test_no_tie_returns_first(self):
        """When candidates differ in accelerator_count (not a variant tie), return the first."""
        p1 = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.OPTIMIZED,
        )
        p2 = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=2,  # different — no tie
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.OPTIMIZED,
        )

        # Build profiles with different base tuples
        def make_with_meta(meta: ProfileMetadata) -> Profile:
            handling = ProfileHandling(
                path=f"/fake/{meta.accelerator_label}.yaml",
                filename=f"{meta.accelerator_label}.yaml",
                priority=1,
            )
            return Profile(
                profile_handling=handling,
                metadata=meta,
                aim_id="test/Model",
                model_id="test/Model",
                engine_args={},
                env_vars={},
            )

        prof1 = make_with_meta(p1)
        prof2 = make_with_meta(p2)
        result = _resolve_variant_tie([prof1, prof2])
        assert result is prof1

    def test_variant_tie_raises_directing_to_aim_profile_id(self):
        """When two profiles share a base tuple but differ by variant, the choice is the
        operator's: raise and tell them to set AIM_PROFILE_ID rather than picking arbitrarily."""
        p1 = _make_profile(variant="alpha")
        p2 = _make_profile(variant="beta")
        with pytest.raises(ProfileNotFound, match="AIM_PROFILE_ID"):
            _resolve_variant_tie([p1, p2])

    def test_same_base_tuple_no_variant_returns_first_without_error(self):
        """Two profiles with same base tuple but NO variant → no tie-break, return first."""
        # This preserves backward-compatible behaviour for legacy profiles that happened
        # to share the same hardware slot without using variant.
        p1 = _make_profile(variant=None)
        p2 = _make_profile(variant=None)
        result = _resolve_variant_tie([p1, p2])
        assert result is p1

    def test_empty_candidates_raises(self):
        """Empty list should raise ProfileNotFound."""
        with pytest.raises(ProfileNotFound):
            _resolve_variant_tie([])

    def test_no_variant_tie_single_no_variant_profile(self):
        """A single non-variant profile is returned fine."""
        p = _make_profile(variant=None)
        assert _resolve_variant_tie([p]) is p

    def test_variant_tie_error_lists_candidate_profile_ids(self):
        """The ambiguity error names every tied candidate so the operator can pick one."""
        p1 = _make_profile(variant="alpha")
        p2 = _make_profile(variant="beta")
        with pytest.raises(ProfileNotFound) as exc_info:
            _resolve_variant_tie([p1, p2])
        msg = str(exc_info.value)
        assert p1.profile_id in msg
        assert p2.profile_id in msg
