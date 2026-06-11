# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_common.engine_args_models — Pydantic engine argument models.
"""

from typing import Literal, get_args, get_origin
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aim_common.engine_args_models import (
    ENGINE_ARGS_MODELS,
    BentomlEngineArgsModel,
    EngineArgsModel,
    VllmEngineArgsModel,
    VllmOmniEngineArgsModel,
    validate_vllm_env_vars,
)


class TestEngineArgsModelBase:
    """Tests for the EngineArgsModel base class."""

    def test_base_class_accepts_extra_fields(self):
        """Extra fields are allowed (extra='allow')."""

        class MinimalModel(EngineArgsModel):
            pass

        m = MinimalModel.model_validate({"unknown-arg": "value", "another": 42})
        assert m.model_extra == {"unknown-arg": "value", "another": 42}

    def test_base_class_is_subclassable(self):
        """EngineArgsModel can be subclassed."""
        assert issubclass(VllmEngineArgsModel, EngineArgsModel)
        assert issubclass(VllmOmniEngineArgsModel, EngineArgsModel)
        assert issubclass(VllmOmniEngineArgsModel, VllmEngineArgsModel)


class TestVllmOmniEngineArgsModel:
    """Tests for VllmOmniEngineArgsModel."""

    def test_registered_under_vllm_omni(self):
        assert ENGINE_ARGS_MODELS["vllm_omni"] is VllmOmniEngineArgsModel

    def test_empty_args_accepted(self):
        m = VllmOmniEngineArgsModel.model_validate({})
        assert m.usp is None
        assert m.vae_patch_parallel_size is None

    def test_profile_like_kebab_keys_accepted_without_native_parser(self, monkeypatch):
        monkeypatch.setattr(VllmOmniEngineArgsModel, "_vllm_parser", staticmethod(lambda: None))
        m = VllmOmniEngineArgsModel.model_validate(
            {
                "gpu-memory-utilization": 0.95,
                "usp": 1,
                "vae-patch-parallel-size": 1,
                "vae-use-tiling": True,
            }
        )
        assert m.gpu_memory_utilization == 0.95
        assert m.usp == 1
        assert m.vae_patch_parallel_size == 1
        assert m.vae_use_tiling is True

    def test_native_parser_receives_serve_omni_model_prefix(self, monkeypatch):
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = None
        monkeypatch.setattr(VllmOmniEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        VllmOmniEngineArgsModel.model_validate({"gpu-memory-utilization": 0.9})
        call_args = mock_parser.parse_args.call_args[0][0]
        assert call_args[:4] == [
            "serve",
            "--omni",
            "--model",
            "aim-engine-args-validation-placeholder",
        ]
        assert "--gpu-memory-utilization" in call_args
        assert "0.9" in call_args

    def test_native_parser_error_message_uses_omni_label(self, monkeypatch):
        mock_parser = MagicMock()

        def _fail(argv):
            raise ValueError("bad flag")

        mock_parser.parse_args.side_effect = _fail
        monkeypatch.setattr(VllmOmniEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        with pytest.raises(ValueError, match="vLLM-Omni engine_args validation failed"):
            VllmOmniEngineArgsModel.model_validate({"tensor-parallel-size": 2})

    def test_bypasses_literal_when_native_parser_accepts(self, monkeypatch):
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = None
        monkeypatch.setattr(VllmOmniEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        m = VllmOmniEngineArgsModel.model_validate({"dtype": "future-dtype"})
        assert m.dtype == "future-dtype"


_VLLM_OMNI_AVAILABLE = bool(VllmOmniEngineArgsModel._vllm_parser())


@pytest.mark.skipif(not _VLLM_OMNI_AVAILABLE, reason="vLLM-Omni not installed")
class TestVllmOmniEngineArgsModelNative:
    """Integration checks when vllm_omni imports succeed."""

    def test_profile_like_args_accepted_by_omni_parser(self):
        VllmOmniEngineArgsModel.model_validate(
            {
                "gpu-memory-utilization": 0.95,
                "usp": 1,
                "vae-patch-parallel-size": 1,
                "vae-use-tiling": True,
            }
        )


class TestVllmEngineArgsModel:
    """Tests for VllmEngineArgsModel."""

    # ------------------------------------------------------------------
    # Valid inputs
    # ------------------------------------------------------------------

    def test_empty_args_accepted(self):
        """An empty dict is valid (all fields are optional)."""
        m = VllmEngineArgsModel.model_validate({})
        assert m.model is None
        assert m.dtype is None

    def test_kebab_case_keys_accepted(self):
        """Kebab-case keys (as they appear in YAML) are accepted via aliases."""
        m = VllmEngineArgsModel.model_validate({"max-model-len": 4096, "dtype": "float16"})
        assert m.max_model_len == 4096
        assert m.dtype == "float16"

    def test_snake_case_keys_accepted(self):
        """Snake_case keys (Python field names) are also accepted."""
        m = VllmEngineArgsModel.model_validate({"max_model_len": 4096, "dtype": "float16"})
        assert m.max_model_len == 4096

    def test_valid_dtype_values(self):
        """All documented dtype literals are accepted."""
        for value in ("auto", "bfloat16", "float", "float16", "float32", "half"):
            m = VllmEngineArgsModel.model_validate({"dtype": value})
            assert m.dtype == value

    def test_valid_kv_cache_dtype(self):
        """Valid kv-cache-dtype values are accepted."""
        for value in ("auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"):
            m = VllmEngineArgsModel.model_validate({"kv-cache-dtype": value})
            assert m.kv_cache_dtype == value

    def test_valid_tensor_parallel_size(self):
        """Integer tensor-parallel-size is accepted."""
        m = VllmEngineArgsModel.model_validate({"tensor-parallel-size": 8})
        assert m.tensor_parallel_size == 8

    def test_valid_gpu_memory_utilization(self):
        """Float gpu-memory-utilization is accepted."""
        m = VllmEngineArgsModel.model_validate({"gpu-memory-utilization": 0.9})
        assert m.gpu_memory_utilization == 0.9

    def test_valid_rope_scaling_dict(self):
        """Dict-type rope-scaling is accepted."""
        m = VllmEngineArgsModel.model_validate({"rope-scaling": {"type": "linear", "factor": 2.0}})
        assert m.rope_scaling == {"type": "linear", "factor": 2.0}

    def test_valid_served_model_name_string(self):
        """served-model-name accepts a plain string."""
        m = VllmEngineArgsModel.model_validate({"served-model-name": "my-model"})
        assert m.served_model_name == "my-model"

    def test_valid_served_model_name_list(self):
        """served-model-name accepts a list of strings."""
        m = VllmEngineArgsModel.model_validate({"served-model-name": ["model-a", "model-b"]})
        assert m.served_model_name == ["model-a", "model-b"]

    def test_no_flag_accepts_none(self):
        """CLI negation flags (no-*) accept None."""
        m = VllmEngineArgsModel.model_validate({"no-trust-remote-code": None})
        assert m.no_trust_remote_code is None

    def test_extra_unknown_args_passed_through(self):
        """Unknown args are silently stored (extra='allow')."""
        m = VllmEngineArgsModel.model_validate({"future-arg": "value"})
        assert m.model_extra.get("future-arg") == "value"

    def test_vllm_validation_bypasses_literal_field_constraint(self, monkeypatch):
        """A Literal field value accepted by vLLM is not rejected by Pydantic.

        When vLLM validates successfully the model is built via model_construct(),
        bypassing Pydantic's field validation.  This ensures that a dtype (or other
        Literal field) added in a newer vLLM release is not rejected by our field
        definitions even before we update them.
        """
        # Confirm "future-dtype" is genuinely absent from the field's Literal so
        # the bypass is meaningful and not a vacuous assertion.
        dtype_annotation = VllmEngineArgsModel.model_fields["dtype"].annotation
        literal_type = next(t for t in get_args(dtype_annotation) if get_origin(t) is Literal)
        assert "future-dtype" not in get_args(literal_type)

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = None
        monkeypatch.setattr(VllmEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        m = VllmEngineArgsModel.model_validate({"dtype": "future-dtype"})
        assert m.dtype == "future-dtype"

    def test_model_construct_field_accessible_after_vllm_validation(self, monkeypatch):
        """Fields with kebab-case aliases are accessible as snake_case attributes after
        vLLM validation — pydantic v2 model_construct resolves aliases, so this works."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = None
        monkeypatch.setattr(VllmEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        m = VllmEngineArgsModel.model_validate({"max-model-len": 4096, "dtype": "auto"})
        assert m.max_model_len == 4096
        assert m.dtype == "auto"

    def test_fields_set_uses_field_names_not_aliases(self, monkeypatch):
        """__pydantic_fields_set__ contains snake_case field names after vLLM validation,
        so model_dump(exclude_unset=True) includes all explicitly provided fields."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = None
        monkeypatch.setattr(VllmEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        m = VllmEngineArgsModel.model_validate({"max-model-len": 4096, "dtype": "auto"})
        dumped = m.model_dump(exclude_unset=True)
        assert (
            "max_model_len" in dumped
        ), f"max_model_len missing from exclude_unset dump; __pydantic_fields_set__={m.__pydantic_fields_set__}"
        assert dumped["max_model_len"] == 4096

    def test_fields_set_uses_field_names_not_aliases_by_alias(self, monkeypatch):
        """model_dump(exclude_unset=True, by_alias=True) must include explicitly provided fields
        using their kebab-case aliases as output keys."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = None
        monkeypatch.setattr(VllmEngineArgsModel, "_vllm_parser", staticmethod(lambda: mock_parser))

        m = VllmEngineArgsModel.model_validate({"max-model-len": 4096, "dtype": "auto"})
        dumped = m.model_dump(exclude_unset=True, by_alias=True)
        assert (
            "max-model-len" in dumped
        ), f"max-model-len missing from exclude_unset+by_alias dump; __pydantic_fields_set__={m.__pydantic_fields_set__}"
        assert dumped["max-model-len"] == 4096

    # ------------------------------------------------------------------
    # Invalid inputs
    # ------------------------------------------------------------------

    def test_invalid_dtype_raises(self):
        """An unrecognised dtype value raises ValidationError."""
        with pytest.raises(ValidationError):
            VllmEngineArgsModel.model_validate({"dtype": "invalid-dtype"})

    def test_invalid_max_model_len_type_raises(self):
        """A non-integer max-model-len raises ValidationError."""
        with pytest.raises(ValidationError):
            VllmEngineArgsModel.model_validate({"max-model-len": "not-a-number"})

    def test_invalid_gpu_memory_utilization_type_raises(self):
        """A non-numeric gpu-memory-utilization raises ValidationError."""
        with pytest.raises(ValidationError):
            VllmEngineArgsModel.model_validate({"gpu-memory-utilization": "high"})

    def test_invalid_kv_cache_dtype_raises(self):
        """An unrecognised kv-cache-dtype value raises ValidationError."""
        with pytest.raises(ValidationError):
            VllmEngineArgsModel.model_validate({"kv-cache-dtype": "fp4"})

    def test_invalid_block_size_raises(self):
        """A block-size value not in the allowed set raises ValidationError."""
        with pytest.raises(ValidationError):
            VllmEngineArgsModel.model_validate({"block-size": 7})

    def test_no_flag_rejects_non_none_value(self):
        """CLI negation flags must be None; any other value is invalid."""
        with pytest.raises(ValidationError):
            VllmEngineArgsModel.model_validate({"no-trust-remote-code": True})


class TestBentomlEngineArgsModel:
    """Tests for BentomlEngineArgsModel (Pydantic-only, no native parser)."""

    # ------------------------------------------------------------------
    # Valid inputs
    # ------------------------------------------------------------------

    def test_empty_args_accepted(self):
        """An empty dict is valid (all fields are optional)."""
        m = BentomlEngineArgsModel.model_validate({})
        assert m.port is None
        assert m.host is None

    def test_is_subclass_of_engine_args_model(self):
        """BentomlEngineArgsModel inherits from EngineArgsModel."""
        assert issubclass(BentomlEngineArgsModel, EngineArgsModel)

    def test_kebab_case_keys_accepted(self):
        """Kebab-case keys (as they appear in YAML) are accepted via aliases."""
        m = BentomlEngineArgsModel.model_validate(
            {
                "api-workers": 4,
                "timeout-keep-alive": 30,
                "working-dir": "/opt/svc",
            }
        )
        assert m.api_workers == 4
        assert m.timeout_keep_alive == 30
        assert m.working_dir == "/opt/svc"

    def test_snake_case_keys_accepted(self):
        """Snake_case keys (Python field names) are also accepted."""
        m = BentomlEngineArgsModel.model_validate(
            {
                "api_workers": 2,
                "timeout_keep_alive": 60,
            }
        )
        assert m.api_workers == 2
        assert m.timeout_keep_alive == 60

    def test_valid_serve_args(self):
        """A representative set of bentoml serve arguments is accepted."""
        m = BentomlEngineArgsModel.model_validate(
            {
                "port": 3000,
                "host": "0.0.0.0",
                "api-workers": 4,
                "timeout": 120,
                "backlog": 2048,
                "reload": True,
                "development": False,
                "working-dir": "/app",
            }
        )
        assert m.port == 3000
        assert m.host == "0.0.0.0"
        assert m.api_workers == 4
        assert m.timeout == 120
        assert m.backlog == 2048
        assert m.reload is True
        assert m.development is False
        assert m.working_dir == "/app"

    def test_ssl_args_accepted(self):
        """SSL-related arguments are accepted."""
        m = BentomlEngineArgsModel.model_validate(
            {
                "ssl-certfile": "/certs/cert.pem",
                "ssl-keyfile": "/certs/key.pem",
                "ssl-ca-certs": "/certs/ca.pem",
            }
        )
        assert m.ssl_certfile == "/certs/cert.pem"
        assert m.ssl_keyfile == "/certs/key.pem"
        assert m.ssl_ca_certs == "/certs/ca.pem"

    def test_extra_unknown_args_passed_through(self):
        """Unknown args are silently stored (extra='allow')."""
        m = BentomlEngineArgsModel.model_validate({"future-bentoml-flag": "value"})
        assert m.model_extra.get("future-bentoml-flag") == "value"

    # ------------------------------------------------------------------
    # Invalid inputs
    # ------------------------------------------------------------------

    def test_invalid_port_type_raises(self):
        """A non-integer port raises ValidationError."""
        with pytest.raises(ValidationError):
            BentomlEngineArgsModel.model_validate({"port": "not-a-number"})

    def test_invalid_api_workers_type_raises(self):
        """A non-integer api-workers raises ValidationError."""
        with pytest.raises(ValidationError):
            BentomlEngineArgsModel.model_validate({"api-workers": "many"})

    def test_invalid_timeout_type_raises(self):
        """A non-integer timeout raises ValidationError."""
        with pytest.raises(ValidationError):
            BentomlEngineArgsModel.model_validate({"timeout": []})

    def test_invalid_ssl_version_type_raises(self):
        """A non-integer ssl-version raises ValidationError."""
        with pytest.raises(ValidationError):
            BentomlEngineArgsModel.model_validate({"ssl-version": "TLSv1.2"})


_VLLM_AVAILABLE = bool(VllmEngineArgsModel._vllm_parser())


class TestValidateVllmEnvVars:
    """Test validate_vllm_env_vars — VLLM_* env var validation against vLLM registry."""

    def test_non_vllm_vars_ignored(self):
        """Non-VLLM_* env vars should never produce errors."""
        validate_vllm_env_vars(
            {
                "HIP_FORCE_DEV_KERNARG": "1",
                "NCCL_MIN_NCHANNELS": "112",
                "PYTORCH_TUNABLEOP_ENABLED": "1",
            }
        )

    def test_empty_env_vars(self):
        """Empty env_vars dict should produce no errors."""
        validate_vllm_env_vars({})

    @pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vLLM not installed")
    def test_known_vllm_vars_pass(self):
        """Known VLLM_* env vars should not produce errors."""
        validate_vllm_env_vars({"VLLM_DO_NOT_TRACK": "1"})

    @pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vLLM not installed")
    def test_unrecognized_vllm_vars_raise(self):
        """Unrecognized VLLM_* env vars should raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized vLLM environment variable"):
            validate_vllm_env_vars(
                {
                    "VLLM_DO_NOT_TRACK": "1",
                    "VLLM_TPYO_VAR": "1",
                    "HIP_FORCE_DEV_KERNARG": "1",
                },
                source="test_profile.yaml",
            )

    @pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vLLM not installed")
    def test_error_message_includes_source_and_var_names(self):
        """Error message should include the source and unrecognized var names."""
        with pytest.raises(ValueError, match="VLLM_FAKE_VAR") as exc_info:
            validate_vllm_env_vars({"VLLM_FAKE_VAR": "1"}, source="my_profile.yaml")
        assert "my_profile.yaml" in str(exc_info.value)

    def test_no_op_when_vllm_unavailable(self, monkeypatch):
        """When vLLM is not available, function should not raise even with bad vars."""
        monkeypatch.setattr(VllmEngineArgsModel, "_vllm_parser", staticmethod(lambda: None))
        validate_vllm_env_vars({"VLLM_FAKE_VAR": "1"})
