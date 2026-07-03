# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for AIM_ADAPTER_* configuration (LoRA contract, ADR-0004)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aim_runtime.config import (
    DEFAULT_ADAPTER_MAX_COUNT,
    DEFAULT_ADAPTER_MAX_CPU_COUNT,
    DEFAULT_ADAPTER_MAX_RANK,
    DEFAULT_ADAPTER_REFRESH_INTERVAL,
    DEFAULT_ADAPTER_SOURCE,
    AdapterMode,
    AIMConfig,
)


def _config(**env) -> AIMConfig:
    # clear=True wipes the environment for the duration of the patch (and
    # restores it on exit), so any AIM_ADAPTER_* vars set on the host can't leak
    # into these tests and we never mutate the real os.environ.
    merged = {"AIM_MODEL_ID": "test/model", **env}
    with patch.dict(os.environ, merged, clear=True):
        return AIMConfig.from_environment()


class TestAdapterDefaults:
    def test_defaults(self):
        c = _config()
        assert c.adapter_source == DEFAULT_ADAPTER_SOURCE
        assert c.adapter_mode is AdapterMode.STATIC
        assert c.adapter_refresh_interval == DEFAULT_ADAPTER_REFRESH_INTERVAL
        assert c.adapter_max_count == DEFAULT_ADAPTER_MAX_COUNT
        assert c.adapter_max_cpu_count == DEFAULT_ADAPTER_MAX_CPU_COUNT
        assert c.adapter_max_rank == DEFAULT_ADAPTER_MAX_RANK

    def test_in_to_dict(self):
        d = _config().to_dict()
        for key in (
            "adapter_source",
            "adapter_mode",
            "adapter_refresh_interval",
            "adapter_max_count",
            "adapter_max_cpu_count",
            "adapter_max_rank",
        ):
            assert key in d


class TestAdapterMode:
    def test_dynamic(self):
        assert _config(AIM_ADAPTER_MODE="dynamic").adapter_mode is AdapterMode.DYNAMIC

    def test_case_insensitive(self):
        assert _config(AIM_ADAPTER_MODE="DYNAMIC").adapter_mode is AdapterMode.DYNAMIC

    def test_invalid_defaults_static(self):
        assert _config(AIM_ADAPTER_MODE="bogus").adapter_mode is AdapterMode.STATIC


class TestAdapterRefreshInterval:
    def test_valid(self):
        assert _config(AIM_ADAPTER_REFRESH_INTERVAL="60").adapter_refresh_interval == 60

    def test_below_minimum_clamped(self):
        assert _config(AIM_ADAPTER_REFRESH_INTERVAL="1").adapter_refresh_interval == 5

    def test_non_integer_defaults(self):
        assert _config(AIM_ADAPTER_REFRESH_INTERVAL="x").adapter_refresh_interval == DEFAULT_ADAPTER_REFRESH_INTERVAL


class TestAdapterCaps:
    def test_max_count_valid(self):
        assert _config(AIM_ADAPTER_MAX_COUNT="4").adapter_max_count == 4

    def test_max_count_below_one_clamped(self):
        assert _config(AIM_ADAPTER_MAX_COUNT="0").adapter_max_count == 1

    def test_cpu_count_raised_to_match_gpu(self):
        c = _config(AIM_ADAPTER_MAX_COUNT="8", AIM_ADAPTER_MAX_CPU_COUNT="4")
        assert c.adapter_max_cpu_count == 8

    def test_cpu_count_kept_when_higher(self):
        c = _config(AIM_ADAPTER_MAX_COUNT="4", AIM_ADAPTER_MAX_CPU_COUNT="32")
        assert c.adapter_max_cpu_count == 32


class TestAdapterMaxRank:
    @pytest.mark.parametrize("rank", [1, 8, 16, 32, 64, 128, 256, 320, 512])
    def test_allowed_vllm_ranks(self, rank):
        assert _config(AIM_ADAPTER_MAX_RANK=str(rank)).adapter_max_rank == rank

    def test_disallowed_rank_defaults(self):
        # 48 is a power-of-twoish value vLLM does not allow.
        assert _config(AIM_ADAPTER_MAX_RANK="48").adapter_max_rank == DEFAULT_ADAPTER_MAX_RANK

    def test_non_integer_defaults(self):
        assert _config(AIM_ADAPTER_MAX_RANK="big").adapter_max_rank == DEFAULT_ADAPTER_MAX_RANK
