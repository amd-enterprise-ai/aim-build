# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from aim_utils.asset_lint import cli, find_unpinned_pyproject_dependencies, find_unpinned_requirements


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def asset_tree(tmp_path: Path) -> Path:
    """A minimal assets/ tree with one clean profile and one specialized image."""
    model = tmp_path / "instinct" / "acme" / "widget"
    (model / "profiles").mkdir(parents=True)
    (model / "profiles" / "vllm-mi300x-fp8-1-throughput.yaml").write_text(
        "model_id: acme/widget\nmetadata:\n  engine: vllm\n"
    )
    image = model / "image"
    image.mkdir()
    (image / "requirements.txt").write_text("bentoml==1.4.8\npydantic==2.9.2\n")
    return tmp_path


def invoke(runner: CliRunner, command: str, assets_root: Path, *extra: str):
    return runner.invoke(cli, [command, "--assets_root", str(assets_root), *extra])


def test_clean_tree_passes_all_checks(runner, asset_tree):
    metadata = asset_tree / "instinct" / "acme" / "widget" / "metadata.yaml"
    metadata.write_text("com:\n  amd:\n    aim:\n      model:\n        canonicalName: acme/widget\n")
    for command in (
        "check-canonical-name",
        "check-placeholders",
        "check-internal-comments",
        "check-requirements-pinned",
    ):
        result = invoke(runner, command, asset_tree)
        assert result.exit_code == 0, f"{command} failed on a clean tree"


@pytest.mark.parametrize(
    "content",
    [
        "model_id: __MODEL_ID__\n",
        "model_id: acme/widget  # [AGENT: confirm this]\n",
        "notes: <TODO>\n",
    ],
)
def test_placeholders_are_rejected(runner, asset_tree, content):
    (asset_tree / "instinct" / "acme" / "widget" / "profiles" / "bad.yaml").write_text(content)
    result = invoke(runner, "check-placeholders", asset_tree)
    assert result.exit_code == 1


def test_canonical_name_must_match_asset_path(runner, asset_tree):
    metadata = asset_tree / "instinct" / "acme" / "widget" / "metadata.yaml"
    metadata.write_text("com:\n  amd:\n    aim:\n      model:\n        canonicalName: other/widget\n")
    assert invoke(runner, "check-canonical-name", asset_tree).exit_code == 1


def test_canonical_name_check_accepts_explicit_precommit_path(runner, asset_tree):
    metadata = asset_tree / "instinct" / "acme" / "widget" / "metadata.yaml"
    metadata.write_text("com:\n  amd:\n    aim:\n      model:\n        canonicalName: acme/widget\n")
    assert runner.invoke(cli, ["check-canonical-name", "--assets_root", str(asset_tree), str(metadata)]).exit_code == 0


def test_placeholder_check_skips_unlisted_extensions(runner, asset_tree):
    (asset_tree / "instinct" / "acme" / "widget" / "notes.rst").write_text("__AIM_ID__\n")
    assert invoke(runner, "check-placeholders", asset_tree).exit_code == 0


@pytest.mark.parametrize(
    "comment",
    [
        "# DECISION: kept tp=8 because the smaller size OOMs",
        "# Controller override: skip validation",
        "# AGENT NOTE: revisit after benchmarks",
        "# blocked on PROJECT-1046",
    ],
)
def test_internal_comments_are_rejected(runner, asset_tree, comment):
    profile = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "vllm-mi300x-fp8-1-throughput.yaml"
    profile.write_text(f"{comment}\nmodel_id: acme/widget\n")
    result = invoke(runner, "check-internal-comments", asset_tree)
    assert result.exit_code == 1


def test_internal_comment_words_in_values_are_ignored(runner, asset_tree):
    """Only comments are scanned — a value that happens to contain a keyword is fine."""
    profile = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "vllm-mi300x-fp8-1-throughput.yaml"
    profile.write_text('description: "A model for clinical DECISION support"\n')
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 0


@pytest.mark.parametrize(
    "content",
    [
        'description: "# DECISION support"\n',
        'url: "https://example.invalid/#PROJECT-1046"\n',
        "description: '# AGENT NOTE: literal'\n",
    ],
)
def test_hashes_inside_yaml_strings_are_not_comments(runner, asset_tree, content):
    profile = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "profile.yaml"
    profile.write_text(content)
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 0


def test_arbitrary_jira_project_key_is_rejected(runner, asset_tree):
    profile = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "profile.yaml"
    profile.write_text("# blocked on OTHERPROJECT-9876\n")
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 1


@pytest.mark.parametrize(
    "comment", ["# Substitute eval data: 5 OASIS-1 T1w cases", "# GPT-4 baseline", "# LLAMA-2 comparison"]
)
def test_single_digit_model_dataset_names_are_not_rejected(runner, asset_tree, comment):
    """A single-digit suffix (OASIS-1, GPT-4, LLAMA-2) reads as a model/dataset name, not a ticket ref."""
    profile = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "profile.yaml"
    profile.write_text(f"{comment}\n")
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 0


def test_internal_comments_also_scan_python_source(runner, asset_tree):
    """check-internal-comments covers .py files too, as of the check-internal-comments-py hook."""
    (asset_tree / "instinct" / "acme" / "widget" / "image" / "harness.py").write_text("# DECISION: use fp8\n")
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 1


def test_python_same_line_trailing_comment_is_caught(runner, asset_tree):
    """A trailing comment with no preceding whitespace is still a real comment."""
    harness = asset_tree / "instinct" / "acme" / "widget" / "image" / "harness.py"
    harness.write_text("x = 1# blocked on PROJECT-1046\n")
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 1


def test_python_hash_inside_triple_quoted_string_is_not_a_comment(runner, asset_tree):
    """A '#' inside a docstring is text, not a comment, and must not be flagged."""
    harness = asset_tree / "instinct" / "acme" / "widget" / "image" / "harness.py"
    harness.write_text('"""\nSee ticket # PROJECT-1046 in the upstream repo for context.\n"""\n')
    assert invoke(runner, "check-internal-comments", asset_tree).exit_code == 0


@pytest.mark.parametrize(
    "line,unpinned",
    [
        ("bentoml==1.4.8", False),
        ("monai[all]==1.5.2", False),
        ('importlib-metadata==8.7.0; python_version < "3.10"', False),
        ("git+https://github.com/acme/pkg.git@0123456789012345678901234567890123456789#egg=pkg", False),
        ("-e git+https://github.com/acme/pkg.git@0123456789012345678901234567890123456789#egg=pkg", False),
        ("bentoml>=1.4.8", True),
        ("bentoml~=1.4", True),
        ("bentoml==1.4.*", True),
        ("torchvision", True),
        ("git+https://github.com/acme/pkg.git@main#egg=pkg", True),
        ("-e git+https://github.com/acme/pkg.git@v1.2.3#egg=pkg", True),
        ("# just a comment", False),
        ("--extra-index-url https://example.invalid", False),
        ("-r base.txt", False),
        ("--constraint constraints.txt", False),
        ("", False),
        ("scipy  # transitively required", True),
    ],
)
def test_requirement_pinning_classification(tmp_path, line, unpinned):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\n")
    assert bool(find_unpinned_requirements(path)) is unpinned


def test_pyproject_dependency_pinning_classification(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "widget"\nversion = "1.0.0"\n' 'dependencies = ["bentoml==1.4.8", "pydantic>=2.0"]\n'
    )
    assert find_unpinned_pyproject_dependencies(path) == [f"{path}: project.dependencies: pydantic>=2.0"]


def test_unpinned_pyproject_dependencies_fail_under_strict(runner, asset_tree):
    image = asset_tree / "instinct" / "acme" / "widget" / "image"
    (image / "pyproject.toml").write_text(
        '[project]\nname = "widget"\nversion = "1.0.0"\ndependencies = ["pydantic>=2.0"]\n'
    )
    assert invoke(runner, "check-requirements-pinned", asset_tree, "--strict").exit_code == 1


def test_unpinned_requirements_warn_but_do_not_fail_by_default(runner, asset_tree):
    (asset_tree / "instinct" / "acme" / "widget" / "image" / "requirements.txt").write_text("bentoml>=1.4.8\n")
    result = invoke(runner, "check-requirements-pinned", asset_tree)
    assert result.exit_code == 0


def test_unpinned_requirements_fail_under_strict(runner, asset_tree):
    (asset_tree / "instinct" / "acme" / "widget" / "image" / "requirements.txt").write_text("bentoml>=1.4.8\n")
    result = invoke(runner, "check-requirements-pinned", asset_tree, "--strict")
    assert result.exit_code == 1


def test_requirements_outside_image_dir_are_ignored(runner, asset_tree):
    (asset_tree / "instinct" / "acme" / "widget" / "requirements.txt").write_text("bentoml>=1.4.8\n")
    result = invoke(runner, "check-requirements-pinned", asset_tree, "--strict")
    assert result.exit_code == 0


def test_explicit_file_arguments_override_the_scan(runner, asset_tree):
    """pre-commit passes changed files; the scan root must not be consulted then."""
    bad = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "bad.yaml"
    bad.write_text("model_id: __MODEL_ID__\n")
    clean = asset_tree / "instinct" / "acme" / "widget" / "profiles" / "vllm-mi300x-fp8-1-throughput.yaml"

    assert runner.invoke(cli, ["check-placeholders", str(clean)]).exit_code == 0
    assert runner.invoke(cli, ["check-placeholders", str(bad)]).exit_code == 1


def test_findings_cite_the_conventions_doc(runner, asset_tree, caplog):
    (asset_tree / "instinct" / "acme" / "widget" / "profiles" / "bad.yaml").write_text("model_id: __MODEL_ID__\n")
    with caplog.at_level(logging.ERROR, logger="aim_utils.asset_lint"):
        invoke(runner, "check-placeholders", asset_tree)
    assert "docs/aim_conventions.md" in caplog.text
    assert "bad.yaml:1" in caplog.text
