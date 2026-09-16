# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Contract between the base images and the evaluation backend they ship.

The image installs lm-eval into a virtualenv and exports its path; the runtime
reads that path from the environment. Nothing at runtime can detect a mismatch
between the two — a wrong path only surfaces as "backend not available" on a GPU
runner — so the agreement is asserted here instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aim_runtime.evaluation.config import BACKEND_COMMAND_ENV, IN_IMAGE_BACKEND_COMMAND

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Base Dockerfiles expected to ship the evaluation backend.
DOCKERFILES = ("docker/Dockerfile.aim-instinct-base",)

#: Directory the Dockerfiles create the virtualenv in.
EVAL_VENV_DIR = str(Path(IN_IMAGE_BACKEND_COMMAND).parent.parent)


@pytest.fixture(params=DOCKERFILES)
def dockerfile(request) -> str:
    return (REPO_ROOT / request.param).read_text()


def test_the_image_exports_the_backend_path_the_runtime_reads(dockerfile: str) -> None:
    assert f'ENV {BACKEND_COMMAND_ENV}="{IN_IMAGE_BACKEND_COMMAND}"' in dockerfile


def test_the_exported_path_is_the_virtualenv_the_build_creates(dockerfile: str) -> None:
    assert f"python3 -m venv {EVAL_VENV_DIR}" in dockerfile
    assert f"{EVAL_VENV_DIR}/bin/pip install" in dockerfile


def test_the_install_is_opt_in_so_prod_images_can_omit_it(dockerfile: str) -> None:
    assert "ARG INSTALL_EVALUATION_DEPS=false" in dockerfile


def test_the_installs_cache_nothing_into_the_layer(dockerfile: str) -> None:
    """Why the build needs no cache-cleanup step: pip writes no cache to remove."""
    installs = [line for line in dockerfile.splitlines() if f"{EVAL_VENV_DIR}/bin/pip install" in line]

    assert installs
    assert all("--no-cache-dir" in line for line in installs)


def test_the_build_proves_the_virtualenv_runs(dockerfile: str) -> None:
    """A venv that cannot start lm-eval must fail the build, not the GPU run."""
    assert f"{IN_IMAGE_BACKEND_COMMAND} --help" in dockerfile


def test_the_pinned_requirements_are_the_ones_the_build_installs(dockerfile: str) -> None:
    """The venv installs the compiled pin file, not a loose `pip install lm-eval`."""
    assert "requirements/evaluation-requirements.txt" in dockerfile
    assert (REPO_ROOT / "requirements/evaluation-requirements.txt").exists()
