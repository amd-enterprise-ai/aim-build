# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Static guards on the REINVENT4 image build recipe."""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")


def test_engine_is_bentoml() -> None:
    assert re.search(r"^ENV AIM_ENGINE=bentoml\s*$", DOCKERFILE, re.M)


def test_reinvent4_ref_is_a_commit_sha() -> None:
    m = re.search(r"^ARG REINVENT4_REF=(\S+)$", DOCKERFILE, re.M)
    assert m and re.fullmatch(r"[0-9a-f]{40}", m.group(1))


def test_pip_install_uses_constraints_with_exact_pins() -> None:
    assert re.search(r"pip install[^\n]*-c /tmp/requirements\.txt[^\n]*\.\[chemprop2\]", DOCKERFILE)
    assert "COPY requirements.txt /tmp/requirements.txt" in DOCKERFILE
    pins = [
        ln for ln in (IMAGE_DIR / "requirements.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]
    assert pins and all(re.match(r"^[A-Za-z0-9_.\-\[\]]+==[^=<>~!\s]+", p) for p in pins)
    assert not any(re.match(r"(torch|torchvision|triton|nvidia-)", p, re.I) for p in pins)


def test_dockerfile_does_not_fetch_the_prior() -> None:
    assert "zenodo" not in DOCKERFILE.lower()
    assert "reinvent.prior" not in DOCKERFILE.replace("# ", "") or "priors/" in DOCKERFILE
    assert not re.search(r"^(RUN|COPY|ADD)[^\n]*\.prior", DOCKERFILE, re.M)
    assert "wget -q" not in DOCKERFILE


# Bounds REINVENT4 declares at the pinned commit (pyproject.toml), as printed by the pip resolver
# in the verified build ("reinvent 4.8.24 depends on pandas<4 and >=3", numpy<3,>=2).
UPSTREAM_BOUNDS = {"pandas": ">=3,<4", "numpy": ">=2,<3"}


def _pins() -> dict[str, str]:
    out = {}
    for ln in (IMAGE_DIR / "requirements.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            name, ver = ln.split(";")[0].strip().split("==")
            out[re.sub(r"[-_.]+", "-", name).lower()] = ver
    return out


def test_pins_satisfy_known_upstream_bounds() -> None:
    from packaging.specifiers import SpecifierSet

    pins = _pins()
    for name, spec in UPSTREAM_BOUNDS.items():
        assert SpecifierSet(spec).contains(pins[name]), f"{name}=={pins[name]} violates upstream {spec}"
    assert pins["pandas"] != "2.3.3"


def test_pins_satisfy_upstream_pyproject_when_clone_available() -> None:
    """With REINVENT4_SRC=<clone at 80a8d21a> every pin is checked against pyproject's bounds."""
    import os

    import pytest
    from packaging.requirements import Requirement

    src = os.environ.get("REINVENT4_SRC")
    if not src or not (Path(src) / "pyproject.toml").is_file():
        pytest.skip("REINVENT4_SRC not set (no offline upstream clone)")
    import tomllib

    data = tomllib.loads((Path(src) / "pyproject.toml").read_text())
    deps = list(data["project"].get("dependencies", []))
    for extra in ("chemprop2",):
        deps += data["project"].get("optional-dependencies", {}).get(extra, [])
    pins = _pins()
    for d in deps:
        req = Requirement(d)
        key = re.sub(r"[-_.]+", "-", req.name).lower()
        if key in pins and req.specifier and not key.startswith(("torch", "nvidia")):
            assert req.specifier.contains(pins[key]), f"{key}=={pins[key]} violates {d}"


def test_requirements_header_states_provenance() -> None:
    head = (IMAGE_DIR / "requirements.txt").read_text().split("absl-py")[0]
    assert "pip" in head and "NOT the upstream uv.lock" in head
    assert "not from upstream's uv.lock" in DOCKERFILE


def test_no_openeye_in_image() -> None:
    """Licence-driven V1: the proprietary OpenEye toolkit must not be installed or indexed."""
    assert "OpenEye" not in DOCKERFILE.replace("OpenEye toolkit", "")
    assert "[all" not in DOCKERFILE
    assert "openeye" not in (IMAGE_DIR / "requirements.txt").read_text().lower().split("absl-py")[1]
