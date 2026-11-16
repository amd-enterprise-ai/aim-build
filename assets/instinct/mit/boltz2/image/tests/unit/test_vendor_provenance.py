# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Turn the "re-lift, do not hand-edit" convention into an enforcement.

``rocm_triangle`` carries two vendored files from a pinned OpenFold3 ref:
``primitives.py`` is byte-identical, ``evoformer.py`` is the upstream file plus one
recorded AMD patch. Nothing else detects a hand-edit or a silent re-lift: the
pre-commit exclusion keeps formatters away, but it does not notice a change.

These tests close that. They are offline — they compare on-disk digests against
the values recorded in ``rocm_triangle/__init__.py``, so they run in the CPU
tier with no network and no GPU. They cannot tell you that upstream still
matches (that needs a fetch); they tell you that OUR copy is the one we
documented, which is the failure mode that actually bites during a rebuild.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

VENDOR_DIR = Path(__file__).resolve().parents[2] / "src" / "rocm_triangle"

# This asset's own MIT code in the same directory. Every other .py file must be recorded as vendored.
OWN_FILES = {"__init__.py", "adapter.py", "trimul_l1.py"}

# (filename, size_bytes, sha256) — both transcribed from the provenance
# block in rocm_triangle/__init__.py, which is where the narrative lives.
VENDORED = [
    ("evoformer.py", 41086, "d63b1a00325f25397e21da3461dc18cdc0419db26b03e29f19cf9bcb51186abe"),
    ("primitives.py", 18277, "4fa70b4c5cea52ec2be21fb92f7b17f450365252757bebc731c68c27902cbd4f"),
    ("LICENSE-APACHE-2.0", 11353, "5b16c6df57578e44e330e9094173d6cb7d31d26b7248c496c8ec5a6aa818472b"),
]


def _package_docstring() -> str:
    """The docstring of ``rocm_triangle/__init__.py``, read without importing the package."""
    return ast.get_docstring(ast.parse((VENDOR_DIR / "__init__.py").read_text())) or ""


@pytest.mark.parametrize(("name", "size", "digest"), VENDORED)
def test_vendored_file_matches_its_recorded_digest(name, size, digest):
    path = VENDOR_DIR / name
    if not path.is_file():
        pytest.fail(f"{name} is missing from {VENDOR_DIR} and nothing explains its absence")
    blob = path.read_bytes()
    assert len(blob) == size, (
        f"{name} is {len(blob)} bytes, recorded as {size}. If this is a deliberate "
        "re-lift, update the provenance block in rocm_triangle/__init__.py and this "
        "table together; if it is a hand-edit, revert it — the pre-commit exclusion "
        "exists precisely so this file is never reformatted in place."
    )
    actual = hashlib.sha256(blob).hexdigest()
    assert actual == digest, f"{name} sha256 is {actual}, recorded as {digest}"


def test_the_provenance_block_names_every_vendored_file():
    """A third vendored file added without a provenance entry must fail here."""
    module_doc = _package_docstring()

    on_disk = {p.name for p in VENDOR_DIR.glob("*.py") if p.name not in OWN_FILES}
    recorded = {name for name, _, _ in VENDORED if name.endswith(".py")}
    assert on_disk == recorded, f"vendored file(s) on disk with no provenance entry: {sorted(on_disk - recorded)}"
    for name, _, _ in VENDORED:
        assert name in (module_doc or ""), f"{name} has no provenance entry in rocm_triangle/__init__.py"


def test_the_apache_license_text_ships_in_the_package_directory_the_image_copies():
    text = (VENDOR_DIR / "LICENSE-APACHE-2.0").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text


UPSTREAM_EVOFORMER = (40787, "7bea98829015a94c7c16f3c45cec5e11521a0d493c2df09048c145b76fd8ef08")

# The AMD patch to evoformer.py as (upstream text, shipped text) hunks, applied in this order.
EVOFORMER_PATCH = [
    (
        "# limitations under the License.\n",
        "# limitations under the License.\n"
        "#\n"
        "# Modified by Advanced Micro Devices, Inc. (changes to the file as published in\n"
        "# aqlaboratory/openfold-3 0.4.1):\n"
        "#   - SEQ_LEN (in _attn_fwd and _attn_fwd_inner) and N_SEQ (in _attn_fwd) are runtime\n"
        "#     arguments instead of tl.constexpr, so a new sequence length does not compile a\n"
        "#     new kernel. The backward kernels are unchanged.\n",
    ),
    ("        SEQ_LEN: tl.constexpr,\n    ):", "        SEQ_LEN,\n    ):"),
    (
        "        N_SEQ: tl.constexpr,\n        SEQ_LEN: tl.constexpr,\n        DIM: tl.constexpr,",
        "        N_SEQ,\n        SEQ_LEN,\n        DIM: tl.constexpr,",
    ),
]


def test_evoformer_is_upstream_plus_exactly_the_recorded_patch():
    """Reversing the recorded hunks on the shipped file must give the upstream bytes."""
    text = (VENDOR_DIR / "evoformer.py").read_text()
    for upstream, shipped in reversed(EVOFORMER_PATCH):
        assert text.count(shipped) == 1, f"patch hunk not found exactly once: {shipped[:60]!r}"
        text = text.replace(shipped, upstream)
    blob = text.encode()
    assert len(blob) == UPSTREAM_EVOFORMER[0]
    assert (
        hashlib.sha256(blob).hexdigest() == UPSTREAM_EVOFORMER[1]
    ), "evoformer.py differs from upstream by more than the recorded patch"


def test_the_provenance_docstring_records_every_digest_in_the_table():
    module_doc = _package_docstring()

    for name, size, digest in [*VENDORED, ("upstream evoformer.py", *UPSTREAM_EVOFORMER)]:
        assert digest in (module_doc or ""), f"{name}: digest missing from the rocm_triangle provenance docstring"
