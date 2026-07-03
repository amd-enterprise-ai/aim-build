# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for ProfileMetadata.features (LoRA adapter tokens, ADR-0004)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aim_common import AdapterToken, Engine, Metric, Precision, ProfileMetadata, ProfileType


def _metadata(**overrides) -> ProfileMetadata:
    base = dict(
        engine=Engine.VLLM,
        precision=Precision.BF16,
        accelerator_count=1,
        metric=Metric.LATENCY,
        manual_selection_only=False,
        type=ProfileType.OPTIMIZED,
    )
    base.update(overrides)
    return ProfileMetadata(**base)


class TestFeaturesParsing:
    def test_default_is_empty(self):
        m = _metadata()
        assert m.features == []
        assert m.supports_adapters is False
        assert m.adapter_token is None

    def test_adapters_token(self):
        m = _metadata(features=["adapters"])
        assert m.adapter_token is AdapterToken.ADAPTERS
        assert m.supports_adapters is True

    def test_scale_only_token(self):
        m = _metadata(features=["adapters-scale-only"])
        assert m.adapter_token is AdapterToken.ADAPTERS_SCALE_ONLY
        assert m.supports_adapters is True

    def test_mutually_exclusive_tokens_rejected(self):
        with pytest.raises(ValidationError):
            _metadata(features=["adapters", "adapters-scale-only"])

    def test_unknown_token_rejected(self):
        with pytest.raises(ValidationError):
            _metadata(features=["not-a-token"])


class TestFeaturesRoundTrip:
    def test_empty_features_omitted_from_to_dict(self):
        # Legacy profiles must not grow a spurious `features: []`.
        assert "features" not in _metadata().to_dict()

    def test_features_present_in_to_dict(self):
        d = _metadata(features=["adapters"]).to_dict()
        assert d["features"] == ["adapters"]

    def test_round_trip_preserves_features(self):
        original = _metadata(features=["adapters"])
        restored = ProfileMetadata.from_dict(original.to_dict())
        assert restored.features == [AdapterToken.ADAPTERS]
        assert restored == original

    def test_round_trip_empty(self):
        original = _metadata()
        restored = ProfileMetadata.from_dict(original.to_dict())
        assert restored == original


class TestFeaturesIdentity:
    def test_hash_differs_by_features(self):
        assert hash(_metadata(features=["adapters"])) != hash(_metadata())

    def test_equal_when_features_equal(self):
        assert _metadata(features=["adapters"]) == _metadata(features=["adapters"])

    def test_accelerator_label_unaffected_by_features(self):
        # accelerator_label intentionally does not encode features (documented decision).
        assert _metadata(features=["adapters"]).accelerator_label == _metadata().accelerator_label
