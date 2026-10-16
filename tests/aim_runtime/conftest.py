# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest

from aim_runtime.engines import VllmEngineArgsModel, VllmOmniEngineArgsModel

_NATIVE_VLLM_TEST_NODEIDS = (
    "TestVllmOmniEngineArgsModelNative",
    "TestVllmParserIntegration",
    "TestValidateVllmEnvVars::test_known_vllm_vars_pass",
    "TestValidateVllmEnvVars::test_unrecognized_vllm_vars_raise",
    "TestValidateVllmEnvVars::test_error_message_includes_source_and_var_names",
)


@pytest.fixture(autouse=True)
def use_deterministic_vllm_arg_validation(request, monkeypatch):
    if any(native_nodeid in request.node.nodeid for native_nodeid in _NATIVE_VLLM_TEST_NODEIDS):
        return

    monkeypatch.setattr(VllmEngineArgsModel, "_vllm_parser", staticmethod(lambda: None))
    monkeypatch.setattr(VllmOmniEngineArgsModel, "_vllm_parser", staticmethod(lambda: None))
