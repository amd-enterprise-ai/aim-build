# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""GPU/HTTP smoke tier — exercises the live BentoML service in the built image.

Everything here needs a real GPU, real weights and a running service, so the
whole module is marked ``gpu`` and skipped by the CPU pre-gate.

Run inside the image on a GPU node, against an already-serving instance::

    test_on_gpu.sh <aim-dir> --cmd "pytest tests/smoke -m gpu -x --tb=short"

with ``AIM_SERVICE_URL`` (or ``AIM_PORT``) pointing at the service.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.gpu

# Trp-cage miniprotein (20 residues) — the smallest input that still exercises
# the full featurize → diffuse → write path.
SMOKE_SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"

# Deliberately tiny: this tier checks wiring, not prediction quality.
FAST_PARAMS = {"sampling_steps": 10, "diffusion_samples": 1, "recycling_steps": 1}

# Generous by design. Upstream publishes no per-input latency for a 20-residue
# single-sequence run, so this is a "catch the model landing on CPU" budget, not
# a performance SLO — those belong to the Tuner.
SMOKE_BUDGET_S = 600.0


def _post(url: str, payload: dict, timeout: float = SMOKE_BUDGET_S) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _body(sequence: str = SMOKE_SEQUENCE, **overrides) -> dict:
    data = {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": sequence}}],
        "name": "http_smoke",
        "use_msa_server": False,
        "output_format": "pdb",
        **FAST_PARAMS,
        **overrides,
    }
    return {"data": data}


def _structure_text(body: str) -> str:
    """The structure document inside a ``/v1/inference`` answer."""
    payload = json.loads(body)
    assert payload["structures"], "no structures returned"
    return payload["structures"][0]["content"]


@pytest.mark.parametrize("path", ["/healthz", "/health"])
def test_health_endpoints_return_200(service_url, path):
    with urllib.request.urlopen(f"{service_url}{path}", timeout=30) as resp:
        assert resp.status == 200


def test_models_lists_the_served_model(service_url):
    with urllib.request.urlopen(f"{service_url}/v1/models", timeout=30) as resp:
        assert resp.status == 200
        listing = json.loads(resp.read().decode())
    assert listing["object"] == "list"
    assert listing["data"][0]["id"] == "mit/boltz2"


def test_the_pre_adr_routes_are_gone(service_url):
    for path in ("/predict", "/predict_full"):
        status, _ = _post(f"{service_url}{path}", _body(), timeout=60)
        assert status == 404, path


def test_inference_returns_a_pdb_document(service_url):
    """The contract src/harness.py depends on: /v1/inference answers JSON whose
    first structure holds the PDB document."""
    started = time.monotonic()
    status, raw = _post(f"{service_url}/v1/inference", _body())
    elapsed = time.monotonic() - started

    assert status == 200, raw[:500]
    body = _structure_text(raw)

    atom_lines = [line for line in body.splitlines() if line.startswith("ATOM")]
    assert atom_lines, f"no ATOM records in the response ({len(body)} bytes)"
    assert elapsed < SMOKE_BUDGET_S, f"inference took {elapsed:.1f}s, budget {SMOKE_BUDGET_S:.0f}s"


def test_predicted_coordinates_are_physically_plausible(service_url):
    """Catches a model that ran but produced garbage (NaNs, exploded coords)."""
    status, raw = _post(f"{service_url}/v1/inference", _body())
    assert status == 200, raw[:500]
    body = _structure_text(raw)

    coords = []
    for line in body.splitlines():
        if not line.startswith("ATOM"):
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    assert coords, "no parsable ATOM coordinates"

    for x, y, z in coords:
        for value in (x, y, z):
            assert value == value, "coordinate is NaN"  # noqa: PLR0124 — NaN check
            assert abs(value) < 1000, f"coordinate {value} is outside any plausible box"

    # A folded 20-residue protein is a few tens of angstroms across; a collapsed
    # or exploded prediction falls outside this range.
    span = max(max(c) for c in coords) - min(min(c) for c in coords)
    assert 5.0 < span < 200.0, f"structure spans {span:.1f} A, which is not a folded miniprotein"


def test_inference_returns_confidence_metrics(service_url):
    status, body = _post(f"{service_url}/v1/inference", _body(output_format="mmcif"))
    assert status == 200, body[:500]

    payload = json.loads(body)
    assert payload["structures"], "no structures returned"
    assert payload["structures"][0]["format"] == "mmcif"

    confidence = payload["confidence"]
    assert confidence, "no confidence summary returned"
    summary = next(iter(confidence.values()))
    assert 0.0 <= summary["complex_plddt"] <= 1.0
    assert 0.0 <= summary["confidence_score"] <= 1.0
    assert payload["timing"]["runtime_s"] > 0


def test_ligand_complex_is_accepted(service_url):
    """Boltz-2's reason for existing is biomolecular *complexes*, so the smoke
    covers a protein+ligand input and not only a bare chain."""
    status, raw = _post(
        f"{service_url}/v1/inference",
        _body(
            sequences=[
                {"protein": {"id": "A", "sequence": SMOKE_SEQUENCE}},
                {"ligand": {"id": "L", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
            ]
        ),
    )
    assert status == 200, raw[:500]
    body = _structure_text(raw)
    hetatm = [line for line in body.splitlines() if line.startswith("HETATM")]
    assert hetatm, "ligand atoms missing from the predicted complex"


def test_malformed_payload_is_rejected_before_compute(service_url):
    """A bad request must cost a 400, not a GPU-minute."""
    started = time.monotonic()
    status, _ = _post(f"{service_url}/v1/inference", {"data": {"sequences": []}}, timeout=60)
    assert status == 400
    assert time.monotonic() - started < 30.0


def test_a_file_path_in_msa_is_rejected(service_url):
    sequences = [{"protein": {"id": "A", "sequence": SMOKE_SEQUENCE, "msa": "/etc/passwd"}}]
    status, _ = _post(f"{service_url}/v1/inference", _body(sequences=sequences), timeout=60)
    assert status == 400


def test_unknown_output_format_is_rejected(service_url):
    status, _ = _post(f"{service_url}/v1/inference", _body(output_format="cif"), timeout=60)
    assert status == 400


def test_model_runs_on_the_gpu():
    """Guards the failure this whole tier exists to catch: a silent CPU fallback
    that passes every functional assertion and is 50x slower."""
    import torch

    assert torch.cuda.is_available(), "no accelerator visible inside the container"
    assert torch.version.hip, f"torch is not a ROCm build: {torch.__version__}"


def test_cuequivariance_is_absent_so_the_cueq_kernel_path_stays_off():
    """The ROCm constraint, asserted against the image rather than the docs.

    Scope: this constrains the *cuEquivariance* path only. boltz's
    ``use_kernels`` seam is reachable here through the ROCm Triton port
    (``src/rocm_triangle``). cuEquivariance is always off;
    ``BOLTZ2_OPTIMIZATIONS`` turns the ported kernels on or off.
    """
    import importlib.util

    assert importlib.util.find_spec("cuequivariance_torch") is None, (
        "cuequivariance_torch is present; re-evaluate the always-off cuEquivariance path before "
        "assuming the cuEquivariance path must stay off"
    )


def test_the_rocm_triton_kernels_are_importable_and_runnable():
    """The counterpart: the path that REPLACES cuEquivariance must be present.

    Without this, an image that silently lost Triton would fall back to the
    reference path and quietly give up the measured speedup while every other
    assertion still passed.
    """
    from rocm_triangle import triton_available

    assert triton_available(), (
        "the vendored ROCm Triton triangle kernels are not usable in this image "
        "(triton not importable, or evoformer.TritonEvoformer is None)"
    )


def test_the_serving_process_reports_the_optimizations_it_was_asked_for(service_url):
    """Ask the SERVER which optimizations answered, via ``runtime`` in a real ``/v1/inference`` response."""
    import os
    import urllib.request

    expected = os.environ.get("BOLTZ2_OPTIMIZATIONS", "on")

    payload = {
        "data": {
            "version": 1,
            "sequences": [{"protein": {"id": "A", "sequence": SMOKE_SEQUENCE}}],
            "name": "kernel_mode_probe",
            "use_msa_server": False,
            "output_format": "pdb",
            "sampling_steps": 10,
            "recycling_steps": 1,
            "diffusion_samples": 1,
        }
    }
    req = urllib.request.Request(
        f"{service_url}/v1/inference",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())

    runtime = body.get("runtime")
    assert runtime is not None, (
        "/v1/inference carried no `runtime` block, so the served kernel mode is "
        "unobservable — the exact condition this test exists to prevent"
    )
    assert runtime["optimizations"] == expected, (
        f"the environment asked for BOLTZ2_OPTIMIZATIONS={expected!r} but the "
        f"serving process resolved {runtime['optimizations']!r}"
    )
    assert runtime["triton"]["attention"] is (expected == "on")
    assert runtime["triton"]["mul_impl"] == ("l1" if expected == "on" else None)


def test_boltz_version_matches_the_parity_reference(boltz_cli_defaults):
    """A silent boltz upgrade would invalidate the recipe-parity reference."""
    import importlib.metadata as md

    assert md.version("boltz") == boltz_cli_defaults["boltz_version"]


def test_gemmi_version_matches_the_boltz_pin():
    """The accuracy metric is written against the gemmi the IMAGE has, not the
    one a host venv happens to install.

    Boltz-2 pins gemmi==0.6.5, which lacks the one-shot string parser that newer
    releases added. A CPU unit tier running against a newer gemmi cannot see that
    skew — it passed here while the in-image evaluate failed with
    `module 'gemmi' has no attribute 'read_structure_string'` on all six cases.
    This asserts the version the parse path was written against.
    """
    import importlib.metadata as md

    import gemmi

    assert md.version("gemmi") == "0.6.5", (
        f"image ships gemmi {md.version('gemmi')}; the harness's structure parsing "
        "was written against the 0.6.5 API that Boltz-2 pins"
    )
    # The APIs _ca_rmsd actually uses must be present.
    assert hasattr(gemmi.cif, "read_string")
    assert hasattr(gemmi, "make_structure_from_block")
    assert hasattr(gemmi, "calculate_superposition")
    assert hasattr(gemmi.SupSelect, "CaP")
