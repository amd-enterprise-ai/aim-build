# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""ROCm Triton triangle kernels for Boltz-2.

``adapter.py`` and ``trimul_l1.py`` are this asset's own code (MIT).
``adapter.py`` holds the boltz-shaped call adaptation plus ``install()``;
``trimul_l1.py`` is the triangle multiply written for TorchInductor. Everything
else here is vendored source under Apache-2.0, excluded from pre-commit so formatters
cannot break the hashes that make the vendoring verifiable. ``primitives.py`` is byte-identical to upstream and
is re-lifted, not hand-edited. ``evoformer.py`` is upstream plus one recorded AMD patch.

Provenance is recorded here rather than left to a commit message, because the
file headers alone do not say where the files came from.
``tests/unit/test_vendor_provenance.py`` asserts these digests against what is
on disk, so a hand-edit or a silent re-lift fails a test instead of going
unnoticed:

``LICENSE-APACHE-2.0``
    The Apache License 2.0 text covering ``evoformer.py`` and ``primitives.py``.
    Verbatim copy of ``LICENSE`` at aqlaboratory/openfold-3 ref ``0.4.1`` (upstream has
    no NOTICE file), 11353 bytes, sha256
    ``5b16c6df57578e44e330e9094173d6cb7d31d26b7248c496c8ec5a6aa818472b``.

``evoformer.py``
    aqlaboratory/openfold-3, ref ``0.4.1``, path
    ``openfold3/core/kernels/triton/evoformer.py``. Upstream file: 40787 bytes, sha256
    ``7bea98829015a94c7c16f3c45cec5e11521a0d493c2df09048c145b76fd8ef08``, verified
    byte-identical against
    ``https://raw.githubusercontent.com/aqlaboratory/openfold-3/0.4.1/openfold3/core/kernels/triton/evoformer.py``.
    Shipped file: upstream plus one AMD patch (change notice in the file header, as
    Apache-2.0 section 4(b) requires): ``SEQ_LEN`` (``_attn_fwd`` and ``_attn_fwd_inner``)
    and ``N_SEQ`` (``_attn_fwd``) are runtime arguments instead of ``tl.constexpr``.
    41086 bytes, sha256
    ``d63b1a00325f25397e21da3461dc18cdc0419db26b03e29f19cf9bcb51186abe``.
    ``tests/unit/test_vendor_provenance.py`` checks the shipped hash and that reversing
    the recorded patch gives the upstream hash.

``primitives.py``
    Same repo and ref, path
    ``openfold3/core/model/layers/triangular_multiplicative_update.py``
    (sha256 ``a5b0252df3cd2ffcfe261cc97468307a6543d00ee527cce7ae42d66f0417454f``,
    51395 bytes), lines 37-608 verbatim plus a four-line ``import warnings`` /
    ``import torch`` preamble replacing the module code that was dropped.
    18277 bytes, sha256
    ``4fa70b4c5cea52ec2be21fb92f7b17f450365252757bebc731c68c27902cbd4f``.
"""

from .adapter import (
    MODE_ATTN,
    MODE_BOTH,
    MODE_MUL,
    MODE_OFF,
    MUL_IMPL_L1,
    MUL_IMPL_NONE,
    VALID_MODES,
    install,
    mode_flags,
    mul_impl,
    prepare_model,
    seam_status,
    triton_available,
    triton_triangular_attn,
    warm_up_large,
)

__all__ = [
    "MODE_ATTN",
    "MODE_BOTH",
    "MODE_MUL",
    "MODE_OFF",
    "MUL_IMPL_L1",
    "MUL_IMPL_NONE",
    "VALID_MODES",
    "seam_status",
    "install",
    "mode_flags",
    "mul_impl",
    "triton_available",
    "triton_triangular_attn",
    "prepare_model",
    "warm_up_large",
]
