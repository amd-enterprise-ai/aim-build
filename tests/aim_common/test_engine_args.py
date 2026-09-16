# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for src/aim_common/engine_args.py

This module is the one place that says how a profile's ``engine_args`` are spelled
and when a flag in them is on, for the serving engine's own argument serializer,
the harness, the accuracy evaluation and the async benchmark alike. These tests
pin the rule itself, the profile-file reader built on it, the agreement between
what the serializer writes and what a reader concludes, and the import weight the
CI pools that run from a bare checkout depend on.
"""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aim_common.engine_args import (
    ENGINE_ARG_TOKENIZER_MODE,
    ENGINE_ARG_TRUST_REMOTE_CODE,
    VALID_TOKENIZER_MODES,
    engine_arg_value,
    engine_flag_enabled,
    engine_flag_is_set,
    normalize_engine_arg_key,
    normalize_engine_args,
    profile_engine_args,
    profile_flag_enabled,
    tokenizer_mode_from_engine_args,
)
from aim_runtime.engines import EngineArgsFormat, engine_args_to_cli_list

FLAG = ENGINE_ARG_TRUST_REMOTE_CODE


def _write_profile(tmp_path: Path, engine_args) -> Path:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump({"engine_args": engine_args}), encoding="utf-8")
    return profile_path


class TestNormalizeEngineArgKey:
    """One spelling for a key, whoever wrote it."""

    @pytest.mark.parametrize("key", ["trust-remote-code", "trust_remote_code"])
    def test_both_separators_reach_the_cli_spelling(self, key):
        assert normalize_engine_arg_key(key) == "trust-remote-code"

    def test_a_non_string_key_does_not_raise(self):
        assert normalize_engine_arg_key(8) == "8"


class TestEngineFlagIsSet:
    """When a boolean engine arg reaches the engine."""

    def test_valueless_means_passed(self):
        """A profile writes a bare CLI flag as ``key:``, which YAML reads as None."""
        assert engine_flag_is_set(None) is True

    def test_explicit_true_means_passed(self):
        assert engine_flag_is_set(True) is True

    def test_only_an_explicit_false_withholds_it(self):
        assert engine_flag_is_set(False) is False


class TestNormalizeEngineArgs:
    """One entry per flag, whatever the document spelled."""

    def test_both_separators_collapse_into_one_entry(self):
        collapsed = normalize_engine_args({"trust-remote-code": True, "trust_remote_code": False})

        assert collapsed == {FLAG: False}

    def test_the_last_spelling_wins_as_a_command_line_does(self):
        collapsed = normalize_engine_args({"trust_remote_code": False, "trust-remote-code": True})

        assert collapsed == {FLAG: True}

    def test_a_duplicate_is_reported(self, caplog):
        with caplog.at_level(logging.WARNING):
            normalize_engine_args({"trust-remote-code": True, "trust_remote_code": False})

        assert "spell one flag twice" in caplog.text
        assert "trust_remote_code=False" in caplog.text

    def test_a_duplicate_that_differs_only_in_case_is_still_a_duplicate(self, caplog):
        with caplog.at_level(logging.WARNING):
            collapsed = normalize_engine_args({"Trust-Remote-Code": True, "trust-remote-code": False})

        assert collapsed == {FLAG: False}
        assert "spell one flag twice" in caplog.text

    def test_one_spelling_per_flag_says_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            collapsed = normalize_engine_args({"trust_remote_code": None, "tensor-parallel-size": 4})

        assert collapsed == {FLAG: None, "tensor-parallel-size": 4}
        assert caplog.text == ""

    def test_the_case_of_a_key_reaches_the_engine_unchanged(self):
        """The key is a CLI flag of the engine, so this function does not lower it."""
        assert normalize_engine_args({"Served-Model_Name": "org/model"}) == {"Served-Model-Name": "org/model"}

    @pytest.mark.parametrize("engine_args", [None, [], "trust-remote-code", 4])
    def test_args_that_are_not_a_mapping_give_nothing(self, engine_args):
        assert normalize_engine_args(engine_args) == {}


class TestEngineArgValue:
    """Reading an engine arg that carries a value rather than being a flag."""

    def test_the_value_is_returned(self):
        assert engine_arg_value({"tensor-parallel-size": 4}, "tensor-parallel-size") == 4

    @pytest.mark.parametrize("key", ["tensor_parallel_size", "Tensor-Parallel-Size"])
    def test_the_key_is_read_in_any_spelling(self, key):
        assert engine_arg_value({key: 4}, "tensor-parallel-size") == 4

    def test_the_arg_may_be_asked_for_in_either_spelling(self):
        assert engine_arg_value({"tensor-parallel-size": 4}, "tensor_parallel_size") == 4

    def test_an_absent_arg_gives_the_default(self):
        assert engine_arg_value({"tensor-parallel-size": 4}, ENGINE_ARG_TOKENIZER_MODE, "auto") == "auto"

    def test_a_valueless_arg_is_distinct_from_an_absent_one(self):
        """``key:`` is how a profile spells an enabled flag, so None is a real value."""
        assert engine_arg_value({FLAG: None}, FLAG, "default") is None

    @pytest.mark.parametrize("engine_args", [None, [], "tensor-parallel-size", 4])
    def test_engine_args_that_are_not_a_mapping_give_the_default(self, engine_args):
        assert engine_arg_value(engine_args, "tensor-parallel-size", 4) == 4


class TestTokenizerModeFromEngineArgs:
    """The tokenizer mode a benchmark client has to match the engine on."""

    @pytest.mark.parametrize("mode", VALID_TOKENIZER_MODES)
    def test_every_mode_the_engine_accepts_is_returned(self, mode):
        assert tokenizer_mode_from_engine_args({ENGINE_ARG_TOKENIZER_MODE: mode}) == mode

    def test_absent_reads_as_none(self):
        assert tokenizer_mode_from_engine_args({"tensor-parallel-size": 4}) is None

    @pytest.mark.parametrize("key", ["tokenizer_mode", "Tokenizer-Mode"])
    def test_the_key_is_read_in_any_spelling(self, key):
        assert tokenizer_mode_from_engine_args({key: "mistral"}) == "mistral"

    @pytest.mark.parametrize("value", ["  MISTRAL  ", "Mistral"])
    def test_case_and_surrounding_space_do_not_hide_a_mode(self, value):
        assert tokenizer_mode_from_engine_args({ENGINE_ARG_TOKENIZER_MODE: value}) == "mistral"

    def test_a_mode_the_engine_would_reject_reads_as_none(self, caplog):
        """Better no flag than a benchmark command the engine refuses to start."""
        with caplog.at_level(logging.WARNING):
            assert tokenizer_mode_from_engine_args({ENGINE_ARG_TOKENIZER_MODE: "nonsense"}) is None
        assert "nonsense" in caplog.text

    @pytest.mark.parametrize("value", [None, True, 4, ["mistral"]])
    def test_a_value_that_is_not_a_string_reads_as_none(self, value):
        assert tokenizer_mode_from_engine_args({ENGINE_ARG_TOKENIZER_MODE: value}) is None

    @pytest.mark.parametrize("engine_args", [None, [], ENGINE_ARG_TOKENIZER_MODE, 4])
    def test_engine_args_that_are_not_a_mapping_read_as_none(self, engine_args):
        assert tokenizer_mode_from_engine_args(engine_args) is None


class TestEngineFlagEnabled:
    """Reading a flag out of engine args held in memory."""

    def test_a_valueless_key_enables_it(self):
        assert engine_flag_enabled({FLAG: None, "tensor-parallel-size": 4}, FLAG) is True

    def test_an_explicit_true_enables_it(self):
        assert engine_flag_enabled({FLAG: True}, FLAG) is True

    def test_an_explicit_false_disables_it(self):
        assert engine_flag_enabled({FLAG: False}, FLAG) is False

    def test_absent_by_default(self):
        assert engine_flag_enabled({"tensor-parallel-size": 4}, FLAG) is False

    def test_the_engines_negative_flag_does_not_enable_it(self):
        """vLLM spells "off" as a different flag, which most profiles carry."""
        assert engine_flag_enabled({"no-trust-remote-code": None}, FLAG) is False

    @pytest.mark.parametrize("key", ["trust_remote_code", "Trust-Remote-Code"])
    def test_the_key_is_read_in_any_spelling(self, key):
        assert engine_flag_enabled({key: None}, FLAG) is True

    def test_the_flag_may_be_asked_for_in_either_spelling(self):
        assert engine_flag_enabled({FLAG: None}, "trust_remote_code") is True

    @pytest.mark.parametrize("engine_args", [None, [], "trust-remote-code", 4])
    def test_engine_args_that_are_not_a_mapping_cost_the_flag(self, engine_args):
        assert engine_flag_enabled(engine_args, FLAG) is False


class TestSerializerAgreement:
    """What the engine is launched with is what a reader concludes.

    The point of the shared rule: a profile whose flag reaches the engine's argv
    must read as enabled, and one whose flag does not must not. Both sides derive
    from :func:`engine_flag_is_set`, so this holds by construction — and this test
    is what says so out loud.
    """

    @pytest.mark.parametrize("value", [None, True, False])
    def test_a_flag_reads_as_enabled_exactly_when_it_reaches_argv(self, value):
        engine_args = {FLAG: value}

        in_argv = f"--{FLAG}" in engine_args_to_cli_list(engine_args)

        assert engine_flag_enabled(engine_args, FLAG) is in_argv

    def test_the_negative_flag_is_its_own_argument(self):
        engine_args = {"no-trust-remote-code": None}

        assert engine_args_to_cli_list(engine_args) == ["--no-trust-remote-code"]
        assert engine_flag_enabled(engine_args, FLAG) is False

    @pytest.mark.parametrize(
        "engine_args",
        [
            {"trust-remote-code": True, "trust_remote_code": False},
            {"trust_remote_code": False, "trust-remote-code": True},
        ],
    )
    def test_a_flag_spelled_twice_still_agrees(self, engine_args):
        """The engine takes one flag, so both sides must read the same one."""
        in_argv = f"--{FLAG}" in engine_args_to_cli_list(engine_args)

        assert engine_flag_enabled(engine_args, FLAG) is in_argv

    def test_a_flag_spelled_twice_reaches_the_engine_once(self):
        engine_args = {"trust_remote_code": False, "trust-remote-code": True}

        assert engine_args_to_cli_list(engine_args) == [f"--{FLAG}"]

    def test_forwarded_args_keep_the_spelling_of_the_document(self):
        """``--arg key=value`` names a parameter of the engine's API, not a CLI flag."""
        engine_args = {"model_tag": "org/model", "trust_remote_code": True}

        assert engine_args_to_cli_list(engine_args, EngineArgsFormat.FORWARDED) == [
            "--arg",
            "model_tag=org/model",
            "--arg",
            "trust_remote_code=True",
        ]


class TestProfileEngineArgs:
    """Reading engine args off a profile file, the only source CI has."""

    def test_engine_args_are_returned_as_written(self, tmp_path):
        profile_path = _write_profile(tmp_path, {FLAG: None, "tensor-parallel-size": 4})

        assert profile_engine_args(profile_path) == {FLAG: None, "tensor-parallel-size": 4}

    def test_a_string_path_is_accepted(self, tmp_path):
        profile_path = _write_profile(tmp_path, {"tensor-parallel-size": 4})

        assert profile_engine_args(str(profile_path)) == {"tensor-parallel-size": 4}

    @pytest.mark.parametrize("profile_path", [None, "", "   "])
    def test_no_path_is_not_worth_a_warning(self, profile_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert profile_engine_args(profile_path) is None

        assert caplog.records == []

    def test_a_missing_file_is_reported(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert profile_engine_args(tmp_path / "does-not-exist.yaml") is None

        assert "not found" in caplog.text

    def test_malformed_yaml_is_reported(self, tmp_path, caplog):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("engine_args: [unclosed\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert profile_engine_args(profile_path) is None

        assert "Could not read profile" in caplog.text

    @pytest.mark.parametrize("document", ["- a list", "just a string", ""])
    def test_a_document_that_is_not_a_profile(self, tmp_path, document):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text(document, encoding="utf-8")

        assert profile_engine_args(profile_path) is None

    def test_engine_args_that_are_not_a_mapping(self, tmp_path):
        assert profile_engine_args(_write_profile(tmp_path, ["--trust-remote-code"])) is None

    def test_a_profile_without_engine_args(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text(yaml.safe_dump({"metadata": {"engine": "vllm"}}), encoding="utf-8")

        assert profile_engine_args(profile_path) is None


class TestProfileFlagEnabled:
    """The whole read, which is what both CI callers use."""

    def test_a_valueless_key_enables_it(self, tmp_path):
        assert profile_flag_enabled(_write_profile(tmp_path, {FLAG: None}), FLAG) is True

    def test_an_explicit_false_disables_it(self, tmp_path):
        assert profile_flag_enabled(_write_profile(tmp_path, {"trust_remote_code": False}), FLAG) is False

    def test_the_engines_negative_flag_does_not_enable_it(self, tmp_path):
        assert profile_flag_enabled(_write_profile(tmp_path, {"no-trust-remote-code": None}), FLAG) is False

    def test_an_unreadable_profile_costs_the_flag_not_the_caller(self, tmp_path):
        assert profile_flag_enabled(tmp_path / "does-not-exist.yaml", FLAG) is False

    def test_enabling_the_flag_is_worth_saying(self, tmp_path, caplog):
        """Letting a tokenizer run code from a model repository is worth a log line."""
        profile_path = _write_profile(tmp_path, {FLAG: None})

        with caplog.at_level(logging.INFO):
            assert profile_flag_enabled(profile_path, FLAG) is True

        assert f"--{FLAG}" in caplog.text


class TestImportWeight:
    """What the CI pools running from a bare checkout depend on.

    The pool that submits async Workloads runs ``ci/async`` scripts with no
    installed package and nothing beyond PyYAML, so this module has to be
    importable without pulling the object model — and pydantic — in with it.
    """

    def test_importing_it_does_not_import_pydantic(self):
        src = Path(__file__).resolve().parents[2] / "src"
        script = "import sys; import aim_common.engine_args; print('pydantic' in sys.modules)"
        # The developer's own environment, with src ahead of anything installed: what
        # is under test is what the module imports, so replacing the environment only
        # makes the result depend on how the interpreter was set up.
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        }

        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)

        assert result.returncode == 0, f"could not import the module:\n{result.stderr}"
        assert result.stdout.strip() == "False", f"importing it pulled in pydantic:\n{result.stdout}{result.stderr}"

    def test_the_object_model_is_still_re_exported(self):
        from aim_common import ProfileMetadata

        assert ProfileMetadata.__name__ == "ProfileMetadata"

    def test_an_unknown_name_still_raises(self):
        # The module object itself, since the lazy re-export is a module __getattr__.
        aim_common = importlib.import_module("aim_common")

        with pytest.raises(AttributeError, match="NotAThing"):
            aim_common.NotAThing
