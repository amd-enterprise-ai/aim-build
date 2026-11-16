# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for the ``vllm bench serve`` command aim_runtime.benchmarking builds.

The external benchmark backend is the reference invocation, so these cover the
flags that keep the harness measuring the same thing: warmup requests, the
tokenizer mode the served engine was started with, and trust-remote-code.

They also cover which suite a run lands on, and the packaged suites themselves,
since a mis-selected suite reports plausible numbers for the wrong workload.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from aim_runtime import benchmarking
from aim_runtime.benchmarking import AIMBenchmark
from aim_runtime.utils import read_yaml

PACKAGED_CONFIG_PATH = Path(benchmarking.__file__).parent / "benchmark-config.yaml"

BENCH_CONFIG = {
    "name": "isl256_osl256_conc8_np80",
    "input_seq_len": 256,
    "output_seq_len": 256,
    "concurrency": 8,
    "num_prompts": 80,
}


# A real table from vllm 0.22.1+cpu, taken verbatim from an EPYC_9965 run of the
# isl256_osl256_conc4_np40 point with --metric-percentiles 75,90,99.
SERVING_BENCHMARK_RESULT = """\
============ Serving Benchmark Result ============
Successful requests:                     40
Benchmark duration (s):                  63.78
Total input tokens:                      10240
Total generated tokens:                  10240
Request throughput (req/s):              0.63
Output token throughput (tok/s):         160.55
Peak output token throughput (tok/s):    192.00
Peak concurrent requests:                4.00
Total token throughput (tok/s):          321.14
---------------Time to First Token----------------
Mean TTFT (ms):                          115.83
Median TTFT (ms):                        117.23
P75 TTFT (ms):                           119.79
P90 TTFT (ms):                           138.66
P99 TTFT (ms):                           189.30
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.54
Median TPOT (ms):                        24.56
P75 TPOT (ms):                           24.63
P90 TPOT (ms):                           24.69
P99 TPOT (ms):                           24.75
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.54
Median ITL (ms):                         24.34
P75 ITL (ms):                            24.58
P90 ITL (ms):                            24.77
P99 ITL (ms):                            25.27
----------------End-to-end Latency----------------
Mean E2EL (ms):                          6374.33
Median E2EL (ms):                        6378.09
P75 E2EL (ms):                           6394.74
P90 E2EL (ms):                           6399.11
P99 E2EL (ms):                           6445.32
==================================================
"""


@pytest.fixture(autouse=True)
def _no_ambient_benchmark_env(monkeypatch):
    """A stale value in the ambient environment would leak into every command.

    ``ACTIVE_SUITE`` and the accelerator variables matter as much as the extra
    args: they decide which suite a run lands on, so a developer who exported one
    to reproduce a sweep would otherwise see the selection tests pass or fail on
    their shell rather than on the code.
    """
    for name in ("VLLM_BENCH_EXTRA_ARGS", "ACTIVE_SUITE", "AIM_ACCELERATOR_TYPE", "AIM_PROFILE_ID", "PROFILE_ID"):
        monkeypatch.delenv(name, raising=False)


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


class TestSuiteSelection:
    """Which suite a run lands on: ACTIVE_SUITE > first matching rule > active_config."""

    # Each suite is identified by the ISL of its single configuration, so a test can
    # name the suite it expects instead of asserting on a generated config name.
    SUITE_MARKERS = {"fallback": 1, "by_type": 2, "by_count": 3, "explicit": 4}

    @classmethod
    def _config_file(cls, tmp_path):
        config_path = tmp_path / "benchmark-config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "active_config": "fallback",
                    "suite_selection": [
                        {"suite": "by_type", "match": {"accelerator_type": "cpu"}},
                        {
                            "suite": "by_count",
                            "match": {"accelerator_type": "gpu", "accelerator_count": [1, 2]},
                        },
                    ],
                    "config_suites": {name: [[isl, 256, 8, 80]] for name, isl in cls.SUITE_MARKERS.items()},
                }
            ),
            encoding="utf-8",
        )
        return str(config_path)

    @classmethod
    def _selected_suite(cls, tmp_path, **env):
        """The name of the suite AIMBenchmark picks under ``env``."""
        with patch.dict("os.environ", env):
            benchmark = AIMBenchmark(service_url="http://localhost:8000", config_file=cls._config_file(tmp_path))

        isl = benchmark.config["benchmark_configs"][0]["input_seq_len"]
        return {marker: name for name, marker in cls.SUITE_MARKERS.items()}[isl]

    def test_a_type_rule_selects_a_suite(self, tmp_path):
        assert self._selected_suite(tmp_path, AIM_ACCELERATOR_TYPE="cpu") == "by_type"

    def test_the_cpu_rule_wins_over_the_gpu_count_rule(self, tmp_path):
        """An EPYC image reports one accelerator, so it must not select the GPU TP1 suite."""
        selected = self._selected_suite(tmp_path, AIM_ACCELERATOR_TYPE="cpu", AIM_PROFILE_ID="epyc-zen5-tp1-bf16")

        assert selected == "by_type"

    def test_active_suite_outranks_the_rules(self, tmp_path):
        selected = self._selected_suite(tmp_path, ACTIVE_SUITE="explicit", AIM_ACCELERATOR_TYPE="cpu")

        assert selected == "explicit"

    def test_the_type_is_matched_case_insensitively(self, tmp_path):
        """``AIM_ACCELERATOR_TYPE`` is lowercased everywhere else it is read."""
        assert self._selected_suite(tmp_path, AIM_ACCELERATOR_TYPE="CPU") == "by_type"

    def test_a_gpu_count_rule_selects_a_suite(self, tmp_path):
        selected = self._selected_suite(tmp_path, AIM_ACCELERATOR_TYPE="gpu", AIM_PROFILE_ID="mi300x-tp1-fp8")

        assert selected == "by_count"

    def test_a_gpu_count_list_selects_a_suite(self, tmp_path):
        selected = self._selected_suite(tmp_path, AIM_ACCELERATOR_TYPE="gpu", AIM_PROFILE_ID="mi300x-tp2-fp8")

        assert selected == "by_count"

    def test_an_unset_type_uses_the_gpu_count_rule(self, tmp_path):
        """The default GPU path keeps selecting by accelerator count."""
        assert self._selected_suite(tmp_path, AIM_PROFILE_ID="mi300x-tp1-fp8") == "by_count"

    def test_no_match_anywhere_falls_back_to_active_config(self, tmp_path):
        assert self._selected_suite(tmp_path, AIM_ACCELERATOR_TYPE="gpu") == "fallback"


class TestPackagedSuites:
    """The shipped benchmark-config.yaml, whose suites are hand-written tuples."""

    @pytest.fixture(scope="class")
    def packaged_config(self):
        return read_yaml(PACKAGED_CONFIG_PATH)

    @pytest.fixture(scope="class")
    def epyc_suites(self, packaged_config):
        return {name: suite for name, suite in packaged_config["config_suites"].items() if name.startswith("epyc_")}

    def test_the_expected_epyc_suites_ship(self, epyc_suites):
        assert sorted(epyc_suites) == ["epyc_full_sweep", "epyc_quick"]

    def test_every_epyc_entry_is_a_positive_four_tuple(self, epyc_suites):
        malformed = [
            (name, entry)
            for name, suite in epyc_suites.items()
            for entry in suite
            if len(entry) != 4 or not all(isinstance(v, int) and v > 0 for v in entry)
        ]
        assert malformed == []

    def test_every_epyc_entry_runs_ten_prompts_per_concurrent_request(self, epyc_suites):
        """A sweep point that does not scale its prompt count measures a different window."""
        mismatched = [
            (name, entry) for name, suite in epyc_suites.items() for entry in suite if entry[3] != 10 * entry[2]
        ]
        assert mismatched == []

    def test_cpu_rule_selects_an_epyc_suite(self, packaged_config):
        assert any(
            rule["suite"].startswith("epyc_") and rule["match"] == {"accelerator_type": "cpu"}
            for rule in packaged_config["suite_selection"]
        )

    def test_epyc_quick_is_a_subset_of_full_sweep(self, epyc_suites):
        """epyc_quick is sampled from epyc_full_sweep; drifting off that basis breaks the docs."""
        quick = {tuple(entry) for entry in epyc_suites["epyc_quick"]}
        full_sweep = {tuple(entry) for entry in epyc_suites["epyc_full_sweep"]}
        assert quick <= full_sweep

    def test_every_rule_references_an_existing_suite(self, packaged_config):
        suites = packaged_config["config_suites"]

        assert [rule["suite"] for rule in packaged_config["suite_selection"] if rule["suite"] not in suites] == []

    def test_the_1k_8k_32k_suite_matches_the_trigger_grid(self, packaged_config):
        """The trigger benchmark page is an irregular grid, so a dropped concurrency is a different workload."""
        assert packaged_config["config_suites"]["1k-8k-32k"] == [
            [1024, 1024, 1, 10],
            [1024, 1024, 4, 40],
            [1024, 1024, 8, 80],
            [1024, 1024, 16, 160],
            [1024, 1024, 32, 320],
            [1024, 1024, 64, 640],
            [1024, 1024, 128, 1280],
            [1024, 1024, 256, 2560],
            [1024, 1024, 512, 5120],
            [8192, 1024, 1, 10],
            [8192, 1024, 4, 40],
            [8192, 1024, 8, 80],
            [8192, 1024, 16, 160],
            [8192, 1024, 32, 320],
            [8192, 1024, 64, 640],
            [8192, 1024, 128, 1280],
            [32768, 1024, 1, 10],
            [32768, 1024, 4, 40],
            [32768, 1024, 8, 80],
            [32768, 1024, 16, 160],
            [32768, 1024, 32, 320],
        ]


class TestLegacyCountSuiteMap:
    """``accelerator_count_suite_map``, the schema that preceded ``suite_selection``.

    A config file written against the released schema is a supported input to
    ``benchmark --config``, and upgrading the runtime must not quietly swap the
    workload such a file asks for.
    """

    SUITE_MARKERS = {"fallback": 1, "mapped_tp2": 2, "by_rule": 3, "explicit": 4}

    @classmethod
    def _config_file(cls, tmp_path, *, with_rules=False):
        config = {
            "active_config": "fallback",
            "accelerator_count_suite_map": {2: "mapped_tp2"},
            "config_suites": {name: [[isl, 256, 8, 80]] for name, isl in cls.SUITE_MARKERS.items()},
        }
        if with_rules:
            config["suite_selection"] = [{"suite": "by_rule", "match": {"accelerator_count": 2}}]

        config_path = tmp_path / "legacy-config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    @classmethod
    def _selected_suite(cls, tmp_path, *, with_rules=False, **env):
        with patch.dict("os.environ", env):
            benchmark = AIMBenchmark(
                service_url="http://localhost:8000",
                config_file=cls._config_file(tmp_path, with_rules=with_rules),
            )

        isl = benchmark.config["benchmark_configs"][0]["input_seq_len"]
        return {marker: name for name, marker in cls.SUITE_MARKERS.items()}[isl]

    def test_the_mapped_suite_is_still_selected(self, tmp_path):
        """Without this, a two-GPU run silently measures active_config instead."""
        selected = self._selected_suite(tmp_path, AIM_PROFILE_ID="vllm-mi300x-fp8-tp2-throughput")

        assert selected == "mapped_tp2"

    def test_the_new_rules_outrank_the_legacy_map(self, tmp_path):
        selected = self._selected_suite(tmp_path, with_rules=True, AIM_PROFILE_ID="vllm-mi300x-fp8-tp2-throughput")

        assert selected == "by_rule"

    def test_active_suite_outranks_the_legacy_map(self, tmp_path):
        selected = self._selected_suite(
            tmp_path, ACTIVE_SUITE="explicit", AIM_PROFILE_ID="vllm-mi300x-fp8-tp2-throughput"
        )

        assert selected == "explicit"

    def test_an_unmapped_count_falls_back_to_active_config(self, tmp_path):
        selected = self._selected_suite(tmp_path, AIM_PROFILE_ID="vllm-mi300x-fp8-tp8-throughput")

        assert selected == "fallback"

    def test_a_run_without_a_profile_falls_back_to_active_config(self, tmp_path):
        """No profile id means no accelerator count to look the suite up by."""
        assert self._selected_suite(tmp_path) == "fallback"

    def test_the_legacy_key_is_reported_as_deprecated(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="aim_runtime.benchmarking"):
            self._selected_suite(tmp_path, AIM_PROFILE_ID="vllm-mi300x-fp8-tp2-throughput")

        assert "accelerator_count_suite_map' is deprecated" in caplog.text

    def test_a_config_using_the_new_rules_is_not_warned_about(self, tmp_path, caplog):
        config = {
            "active_config": "fallback",
            "suite_selection": [{"suite": "by_rule", "match": {"accelerator_count": 2}}],
            "config_suites": {name: [[isl, 256, 8, 80]] for name, isl in self.SUITE_MARKERS.items()},
        }
        config_path = tmp_path / "modern-config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        with caplog.at_level("WARNING", logger="aim_runtime.benchmarking"):
            with patch.dict("os.environ", {"AIM_PROFILE_ID": "vllm-mi300x-fp8-tp2-throughput"}):
                AIMBenchmark(service_url="http://localhost:8000", config_file=str(config_path))

        assert "deprecated" not in caplog.text


class TestMetricParsing:
    """The regexes that lift metrics out of the ``vllm bench serve`` result table.

    The table is the only place these numbers exist: the subprocess runs with
    ``capture_output``, so a pattern that does not match silently writes an empty
    column rather than failing the run.
    """

    def test_the_throughput_metrics_parse(self):
        """``Total token throughput`` is lower-case in the table it is read from."""
        parsed = AIMBenchmark._parse_benchmark_output(AIMBenchmark.__new__(AIMBenchmark), SERVING_BENCHMARK_RESULT)

        assert parsed["req_throughput"] == "0.63"
        assert parsed["output_tok_throughput"] == "160.55"
        assert parsed["total_tok_throughput"] == "321.14"

    def test_every_metric_parses(self):
        """A pattern that matches nothing leaves its column blank in the CSV."""
        parsed = AIMBenchmark._parse_benchmark_output(AIMBenchmark.__new__(AIMBenchmark), SERVING_BENCHMARK_RESULT)

        assert [key for key, value in parsed.items() if value is None] == []
