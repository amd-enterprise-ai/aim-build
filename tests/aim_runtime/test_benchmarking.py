# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for the ``vllm bench serve`` command aim_runtime.benchmarking builds.

The external benchmark backend is the reference invocation, so these cover the
flags that keep the harness measuring the same thing: warmup requests, the
tokenizer mode the served engine was started with, and trust-remote-code.
"""

from unittest.mock import patch

import pytest
import yaml

from aim_runtime.benchmarking import AIMBenchmark

BENCH_CONFIG = {
    "name": "isl256_osl256_conc8_np80",
    "input_seq_len": 256,
    "output_seq_len": 256,
    "concurrency": 8,
    "num_prompts": 80,
}


@pytest.fixture(autouse=True)
def _no_ambient_extra_args(monkeypatch):
    """A stale value in the ambient environment would leak into every command."""
    monkeypatch.delenv("VLLM_BENCH_EXTRA_ARGS", raising=False)


def build_command(settings=None, engine_args=None, extra_args=None):
    """Return the argv AIMBenchmark would run for a single configuration."""
    benchmark = AIMBenchmark.__new__(AIMBenchmark)
    benchmark.config = {"settings": {"service_host": "localhost", "service_port": 8000, **(settings or {})}}
    benchmark._engine_args = engine_args or {}
    benchmark._engine_args_resolved = True

    with (
        patch.dict("os.environ", {"VLLM_BENCH_EXTRA_ARGS": extra_args} if extra_args else {}),
        patch("aim_runtime.benchmarking.subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        benchmark.run_vllm_benchmark("test-model", BENCH_CONFIG)
        return mock_run.call_args[0][0]


def flag_value(cmd, flag):
    """Return the argument following ``flag``, or None when the flag is absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class TestNumWarmups:
    """settings.num_warmups drives --num-warmups (external passes 5)."""

    def test_default_matches_external(self):
        """A settings block without the key still warms up.

        ``--config-file`` replaces the packaged settings rather than layering over
        them, so a custom config that says nothing about warmups would otherwise
        measure the first configuration cold — the skew this flag exists to remove.
        """
        assert flag_value(build_command(), "--num-warmups") == "5"

    def test_setting_is_honored(self):
        assert flag_value(build_command({"num_warmups": 12}), "--num-warmups") == "12"

    def test_zero_omits_the_flag(self):
        assert "--num-warmups" not in build_command({"num_warmups": 0})

    def test_null_omits_the_flag(self):
        assert "--num-warmups" not in build_command({"num_warmups": None})

    @pytest.mark.parametrize("setting", ["five", "5", 2.7, -1, True, []])
    def test_a_malformed_count_is_rejected(self, setting):
        """``int()`` would turn 2.7 into 2 and drop a negative count silently."""
        with pytest.raises(ValueError, match="must be a non-negative integer or unset"):
            build_command({"num_warmups": setting})

    def test_the_rejection_names_the_offending_value(self):
        with pytest.raises(ValueError, match="'five'"):
            build_command({"num_warmups": "five"})


class TestTokenizerMode:
    """The profile that started the engine decides the benchmark tokenizer mode."""

    def test_mistral_profile_sets_tokenizer_mode(self):
        cmd = build_command(engine_args={"tokenizer-mode": "mistral"})
        assert flag_value(cmd, "--tokenizer-mode") == "mistral"

    def test_underscored_engine_arg_is_recognized(self):
        cmd = build_command(engine_args={"tokenizer_mode": "mistral"})
        assert flag_value(cmd, "--tokenizer-mode") == "mistral"

    def test_absent_from_profile_omits_the_flag(self):
        assert "--tokenizer-mode" not in build_command(engine_args={"tensor-parallel-size": 4})

    def test_unknown_mode_omits_the_flag(self):
        assert "--tokenizer-mode" not in build_command(engine_args={"tokenizer-mode": "nonsense"})

    def test_the_mode_is_read_once_per_run(self):
        """A profile the engine would reject warns once, not once per sweep point."""
        benchmark = AIMBenchmark.__new__(AIMBenchmark)
        benchmark._engine_args = {"tokenizer-mode": "nonsense"}
        benchmark._engine_args_resolved = True

        with patch("aim_runtime.benchmarking.tokenizer_mode_from_engine_args", return_value=None) as mock_read:
            assert benchmark.tokenizer_mode is None
            assert benchmark.tokenizer_mode is None

        mock_read.assert_called_once()


class TestTrustRemoteCode:
    """settings.trust_remote_code overrides the profile, unset follows it."""

    def test_unset_follows_profile_when_enabled(self):
        assert "--trust-remote-code" in build_command(engine_args={"trust-remote-code": None})

    def test_unset_follows_profile_when_absent(self):
        assert "--trust-remote-code" not in build_command(engine_args={"no-trust-remote-code": None})

    def test_setting_true_forces_the_flag(self):
        cmd = build_command({"trust_remote_code": True}, engine_args={"no-trust-remote-code": None})
        assert "--trust-remote-code" in cmd

    def test_setting_false_suppresses_the_flag(self):
        cmd = build_command({"trust_remote_code": False}, engine_args={"trust-remote-code": True})
        assert "--trust-remote-code" not in cmd

    def test_not_duplicated_when_also_in_extra_args(self):
        cmd = build_command(engine_args={"trust-remote-code": None}, extra_args="--trust-remote-code")
        assert cmd.count("--trust-remote-code") == 1

    @pytest.mark.parametrize("setting", ["false", "no", "true", 0, 1, []])
    def test_a_non_boolean_setting_is_rejected(self, setting):
        """``bool("false")`` is True, and guessing either way is silent in the results."""
        with pytest.raises(ValueError, match="must be a boolean or unset"):
            build_command({"trust_remote_code": setting}, engine_args={"no-trust-remote-code": None})

    def test_the_rejection_names_the_offending_value(self):
        with pytest.raises(ValueError, match="'true'"):
            build_command({"trust_remote_code": "true"})

    def test_an_explicit_null_still_follows_the_profile(self):
        """``trust_remote_code: null`` is how the packaged config spells "unset"."""
        cmd = build_command({"trust_remote_code": None}, engine_args={"trust-remote-code": None})
        assert "--trust-remote-code" in cmd


class TestExtraArgs:
    """VLLM_BENCH_EXTRA_ARGS still carries genuinely ad-hoc flags."""

    def test_extra_args_are_appended(self):
        cmd = build_command(extra_args="--seed 1234")
        assert cmd[-2:] == ["--seed", "1234"]

    def test_extra_args_outrank_the_settings(self):
        cmd = build_command({"num_warmups": 5}, extra_args="--num-warmups 9")
        assert cmd[-2:] == ["--num-warmups", "9"]

    def test_extra_args_can_still_ask_for_trust_remote_code(self):
        """The escape hatch stays usable for a profile that does not request it."""
        cmd = build_command(
            {"trust_remote_code": False},
            engine_args={"no-trust-remote-code": None},
            extra_args="--trust-remote-code",
        )
        assert cmd[-1] == "--trust-remote-code"


class TestSettingsValidatedAtConstruction:
    """A malformed setting fails before the service is probed, not mid-sweep."""

    @staticmethod
    def _config_file(tmp_path, settings):
        config_path = tmp_path / "benchmark-config.yaml"
        config_path.write_text(
            yaml.safe_dump({"active_config": "suite", "config_suites": {"suite": [[256, 256, 8, 80]]}, **settings}),
            encoding="utf-8",
        )
        return str(config_path)

    def _construct(self, tmp_path, settings):
        return AIMBenchmark(
            service_url="http://localhost:8000",
            config_file=self._config_file(tmp_path, {"settings": settings}),
        )

    def test_a_malformed_trust_remote_code_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="trust_remote_code must be a boolean or unset"):
            self._construct(tmp_path, {"trust_remote_code": "true"})

    def test_a_malformed_num_warmups_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="num_warmups must be a non-negative integer or unset"):
            self._construct(tmp_path, {"num_warmups": -1})

    def test_the_packaged_spelling_of_unset_constructs(self, tmp_path):
        """``trust_remote_code: null`` with no num_warmups is what a bare config looks like."""
        benchmark = self._construct(tmp_path, {"trust_remote_code": None})

        assert benchmark._num_warmups(benchmark.config["settings"]) == 5


class TestEngineArgsResolution:
    """Callers without a resolved profile fall back to selecting it themselves."""

    def test_supplied_engine_args_skip_profile_resolution(self):
        benchmark = AIMBenchmark.__new__(AIMBenchmark)
        benchmark._engine_args = {"tokenizer-mode": "mistral"}
        benchmark._engine_args_resolved = True

        with patch.object(AIMBenchmark, "_resolve_profile_engine_args") as mock_resolve:
            assert benchmark.engine_args == {"tokenizer-mode": "mistral"}
        mock_resolve.assert_not_called()

    def test_missing_engine_args_are_resolved_once(self):
        benchmark = AIMBenchmark.__new__(AIMBenchmark)
        benchmark._engine_args = None
        benchmark._engine_args_resolved = False

        with patch.object(
            AIMBenchmark, "_resolve_profile_engine_args", return_value={"trust-remote-code": None}
        ) as mock_resolve:
            assert benchmark.engine_args == {"trust-remote-code": None}
            assert benchmark.engine_args == {"trust-remote-code": None}
        mock_resolve.assert_called_once()

    def test_unresolvable_profile_yields_no_flags(self):
        with patch("aim_runtime.config.AIMConfig.from_environment", side_effect=RuntimeError("no profile")):
            assert AIMBenchmark._resolve_profile_engine_args() == {}
