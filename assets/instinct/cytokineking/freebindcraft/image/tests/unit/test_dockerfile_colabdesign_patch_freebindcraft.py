# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Text guards on the Dockerfile's ColabDesign / JAX 0.8 compatibility fix.

ColabDesign HEAD still calls ``jax.lib.xla_bridge.get_backend()`` (removed in
JAX 0.8), which made every design request fail. The Dockerfile pins ColabDesign
and patches that call at build time. The service is never imported here.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"
PINNED_COMMIT = "e31a56fe1d9b4de25c8697f3a28b75892941cc72"


def _text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_colabdesign_is_installed_only_at_the_pinned_commit() -> None:
    text = _text()
    assert f"ARG COLABDESIGN_COMMIT={PINNED_COMMIT}" in text
    installs = re.findall(r"ColabDesign\.git@(\S+?)\"", text)
    assert installs == ["${COLABDESIGN_COMMIT}"], "one install, at the pin; never a moving ref"
    assert text.index("ARG COLABDESIGN_COMMIT") < text.index("ColabDesign.git@")


def test_patch_step_rewrites_removed_api_and_fails_closed() -> None:
    text = _text()
    assert "jax.lib.xla_bridge.get_backend()" in text
    assert "jax.extend.backend.get_backend()" in text
    assert "import jax.extend" in text
    assert "sys.exit(" in text  # pattern-not-found fails the build
    assert re.search(r"grep -c 'jax\.lib\.xla_bridge'.*= 0", text)  # still-present check


def test_patch_and_canary_come_after_all_installs() -> None:
    text = _text()
    patch = text.index("# ── ColabDesign JAX 0.8 compatibility patch")
    for marker in (
        "pip install --no-cache-dir --no-deps",
        "pip install --no-cache-dir freesasa",
        "checkout -q FETCH_HEAD",
    ):
        assert text.index(marker) < patch
    assert "colabdesign.shared.utils" in text[patch:]
    assert "clear_mem()" in text[patch:]


def test_comments_do_not_claim_colabdesign_tracks_jax() -> None:
    assert "tracks JAX API changes" not in _text()


def test_image_is_a_job_not_a_service() -> None:
    text = _text()
    assert re.search(
        r'^ENTRYPOINT \["/miniforge3/envs/BindCraft/bin/python", "/opt/aim/src/entrypoint.py"\]$', text, re.M
    )
    assert not re.search(r"^(CMD|EXPOSE)\b", text, re.M)
    assert "bentoml" not in text.lower().replace("no bentoml", "")


def test_freebindcraft_is_pinned_to_a_commit() -> None:
    assert re.search(r"^ARG FREEBINDCRAFT_COMMIT=[0-9a-f]{40}$", _text(), re.M)


def test_import_gate_runs_after_the_entrypoint_is_copied() -> None:
    text = _text()
    assert text.index("COPY src/ /opt/aim/src/") < text.index("bindcraft.py --help") < text.index("ENTRYPOINT")


def test_matplotlib_keeps_get_cmap_for_colabdesign_animations() -> None:
    text = _text()
    pinned = re.search(r'"matplotlib==(\d+)\.(\d+)\.\d+"', text)
    assert pinned and (int(pinned[1]), int(pinned[2])) < (3, 9)  # get_cmap was removed in 3.9
    assert "hasattr(c, 'get_cmap')" in text  # build-time guard


def test_numeric_stack_is_pinned_to_the_validated_image() -> None:
    """numpy, scipy, ml_dtypes and the conda package list are exact pins (PR #1435 review, items 5 and 6)."""
    text = _text()
    for pin in ('"numpy==2.5.4"', '"scipy==1.18.1"', '"ml_dtypes==0.6.0"', '"opt_einsum==3.4.0"', "freesasa==2.2.1"):
        assert pin in text
    assert '"numpy<2.0.0"' not in text  # the pip step replaces conda's numpy; the old bound was misleading
    conda_block = text[text.index("conda install -y -c conda-forge") : text.index("conda clean -afy")]
    for line in conda_block.splitlines():
        line = line.strip().rstrip("\\").strip()
        if line.startswith('"'):
            assert "==" in line, f"conda package not pinned exactly: {line}"


def test_ffmpeg_is_not_installed_and_the_build_guards_its_absence() -> None:
    text = _text()
    assert not re.search(r"^\s+ffmpeg \\$", text, re.M)  # not in the conda package list
    assert "ffmpeg -version" not in text  # no build step calls it
    assert "ffmpeg-variant" not in text  # the variant record is gone with the package
    assert "! command -v ffmpeg" in text  # build-time guard: it must stay absent
    assert """grep -q '"name": "ffmpeg"'""" in text  # ... including as a conda package


def test_licence_removals_are_applied_after_the_checkout() -> None:
    text = _text()
    rm = re.search(r"rm -f /app/functions/DAlphaBall\.gcc /app/functions/dssp", text)
    assert rm and text.index("checkout -q FETCH_HEAD") < rm.start()
    assert re.search(r"chmod \+x /app/functions/sc /app/functions/FASPR\n", text)  # no dssp in the chmod list
    assert "/app/functions/dssp /app/functions/sc" not in text
    # FASPR data, sc-rs and the other kept binaries are not removed
    assert "dun2010bbdep" not in text.split("rm -f", 1)[1].split("\n", 1)[0]


def test_conda_dssp_replaces_the_bundled_binary_with_a_build_smoke_test() -> None:
    text = _text()
    assert re.search(r'^\s+"dssp==[0-9.]+" \\$', text, re.M)  # conda-forge package list
    assert "Bio.PDB import PDBParser, DSSP" in text and "mkdssp" in text
    assert text.index("rm -f /app/functions/DAlphaBall.gcc") < text.index("from Bio.PDB import PDBParser, DSSP")


def test_mkdssp_is_a_wrapper_and_the_smoke_test_covers_a_remark_containing_pdb() -> None:
    text = _text()
    assert "COPY src/mkdssp_shim.sh" in text
    assert "mkdssp.real" in text and "install -m 0755 /tmp/mkdssp_shim.sh /miniforge3/envs/BindCraft/bin/mkdssp" in text
    # smoke test: PDL1 with an injected free-text REMARK must parse and match the plain file
    assert "REMARK   Copyright" in text and "PDL1_remark.pdb" in text
    assert "different result" in text


def test_af2_parameters_are_not_baked() -> None:
    text = _text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "alphafold_params" not in code and "/app/params" not in code and "tar -xf" not in code
    assert "AIM_CACHE_PATH" in text


def test_jax_canary_imports_the_rocm_plugin_and_can_fail() -> None:
    text = _text()
    start = text.index("# ── JAX GPU canary")
    block = text[start : text.index("# ── ColabDesign (JAX-based")]
    assert "import jax_plugins.xla_rocm7" in block
    # a failed import must exit non-zero, and the block must not end in an unconditional exit(0)
    assert "except Exception" in block
    assert "sys.exit('JAX ROCm canary failed" in block
    assert "sys.exit(0)" not in block
