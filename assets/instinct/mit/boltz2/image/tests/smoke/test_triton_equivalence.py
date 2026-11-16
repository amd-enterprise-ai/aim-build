# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""GPU equivalence + microbenchmark for the ROCm Triton triangle kernels.

This is the gate that decides whether the port may ship. It runs boltz's OWN
triangle layers — real modules, real weights, real shapes — down both paths in
one process and asks whether substituting the kernels changes the answer by more
than the precision the AIM already accepts.

Why not a tolerance pulled out of the air
------------------------------------------
bf16 carries an 8-bit mantissa (eps = 2^-8 = 3.906e-3), and these layers reduce
over N = 186..624 terms, so *any* reassociation of the reduction moves the
result. An absolute atol would therefore be a number chosen to make the test
pass. Instead fp32 is treated as ground truth and two relative errors are
compared:

    err_ref = ||reference_bf16 - reference_fp32|| / ||reference_fp32||
    err_tri = ||triton_bf16    - reference_fp32|| / ||reference_fp32||

``err_ref`` is the error the served AIM *already* incurs today by running under
``precision="bf16-mixed"``. The substitution passes when ``err_tri`` is within
``ERROR_RATIO_LIMIT`` of it — i.e. the kernels are no worse a bf16
implementation than the one already in production. That makes the threshold a
property of the existing build rather than of the author's optimism.

A negative control runs alongside: a deliberately wrong path (heads transposed,
the exact hazard the adapter warns about) must FAIL the same check, otherwise
the check has no power to detect the bug it exists for.

Timings are recorded in the same pass and written to
``$BOLTZ2_TRITON_REPORT`` (default ``/tmp/triton_equivalence.json``) so the
build report can cite measured numbers rather than estimates.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

# Boltz-2 pairformer geometry (boltz.main.PairformerArgsV2 + MSAModuleArgs):
# token_z = 128, pairwise_head_width = 32, pairwise_num_heads = 4.
TOKEN_Z = 128
HEAD_WIDTH = 32
NUM_HEADS = 4

# Token counts spanning the six evaluation targets: 8eil 186, 8c4d 330, 8gpp 624.
# 384 is boltz's own const.chunk_size_threshold, where the reference path
# switches chunk_size_tri_attn from 512 to 128 — worth straddling.
SHAPES = (186, 384, 624)

# How much larger than the AIM's existing bf16 error the ported kernels may be.
# 3.0 allows a different but comparable reduction order; it does not allow a
# different operation. The negative control below is what proves that.
ERROR_RATIO_LIMIT = 3.0

# Microbenchmark shape.
WARMUP_ITERS = 5
TIMED_ITERS = 20

_REPORT: dict = {"equivalence": [], "timing": [], "negative_control": []}


def _report_path() -> Path:
    return Path(os.environ.get("BOLTZ2_TRITON_REPORT", "/tmp/triton_equivalence.json"))


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    _report_path().write_text(json.dumps(_REPORT, indent=2))


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("no GPU visible")
    return torch.device("cuda")


@pytest.fixture(scope="module", autouse=True)
def _install_kernels():
    """Install both seams once; per-test dispatch is via the use_kernels argument."""
    from rocm_triangle import MODE_BOTH, install, triton_available

    if not triton_available():
        pytest.skip("vendored Triton kernels unavailable in this image")
    install(MODE_BOTH)


def _relative_error(actual: torch.Tensor, truth: torch.Tensor) -> float:
    diff = (actual.float() - truth.float()).norm().item()
    scale = truth.float().norm().item()
    return diff / scale if scale > 0 else float("inf")


def _randomize(module: torch.nn.Module, seed: int = 1234) -> torch.nn.Module:
    """Give every parameter a non-degenerate value.

    Boltz follows AlphaFold and initialises the *final* projection of both
    triangle blocks to zero — ``init.final_init_(self.p_out.weight)`` in
    ``triangular_mult.py``, and ``init="final"`` on ``linear_o`` in the attention
    primitive. A freshly constructed layer therefore returns EXACTLY ZERO, which
    makes any relative-error denominator zero and every comparison vacuous.
    (The first run of this test reported ``err_reference = Infinity`` for all
    nine cases for precisely that reason.)

    Weights are drawn at ``1/sqrt(fan_in)`` so activations stay in a sane range
    rather than saturating the sigmoids or overflowing bf16. LayerNorm gains are
    left near one and its biases near zero, matching a trained model's regime.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim >= 2:
                fan_in = param.shape[-1]
                sample = torch.randn(param.shape, generator=generator) / (fan_in**0.5)
            elif "norm" in name and "weight" in name:
                sample = torch.ones(param.shape) + 0.02 * torch.randn(param.shape, generator=generator)
            else:
                sample = 0.02 * torch.randn(param.shape, generator=generator)
            param.copy_(sample.to(param.device, param.dtype))
    return module


def _assert_not_degenerate(tensor: torch.Tensor, what: str) -> None:
    """Guard the denominator, so a zero reference can never read as a pass."""
    norm = tensor.float().norm().item()
    assert norm > 0.0 and norm == norm, f"{what} is degenerate (norm={norm}); the comparison would be vacuous"


def _record_matmul_dtypes(fn) -> set:
    """Return the set of dtypes every matmul-class op ran in during ``fn()``.

    THE POINT OF THIS. Comparing a Triton arm that runs its linears in fp32
    against a reference arm whose ``nn.Linear`` calls autocast cast to bf16
    gives an error ratio that is evidence about the *precision*, not the
    *implementation*. autocast intercepts ATen ops; a Triton kernel launch is not
    one, so nothing in the framework would catch that.

    A TorchDispatch mode sees every ATen op regardless of who issued it, so this
    records what actually executed. The equivalence tests use it to REQUIRE that
    both arms ran at the same precision before their error ratio is allowed to
    mean anything.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    seen: set = set()
    interesting = {"mm", "bmm", "addmm", "matmul", "einsum", "linear", "baddbmm", "_scaled_mm"}

    class _Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
            name = func.overloadpacket.__name__ if hasattr(func, "overloadpacket") else str(func)
            out = func(*args, **(kwargs or {}))
            if name in interesting:
                for t in args or ():
                    if isinstance(t, torch.Tensor) and t.is_floating_point():
                        seen.add(t.dtype)
            return out

    with _Recorder():
        fn()
    return seen


def _sync_time(fn, warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS) -> float:
    """Median wall time per call in ms, with the device synchronised each side."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


# --------------------------------------------------------------------------- #
# Triangle multiplicative update
# --------------------------------------------------------------------------- #


def _make_mul(outgoing: bool, device: torch.device):
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    torch.manual_seed(0)
    return _randomize(cls(dim=TOKEN_Z)).to(device).eval()


def _mul_inputs(n: int, device: torch.device):
    torch.manual_seed(n)
    x = torch.randn(1, n, n, TOKEN_Z, device=device)
    mask = torch.ones(1, n, n, device=device)
    # A ragged tail, so the mask is actually exercised rather than being all-ones.
    mask[:, :, int(n * 0.9) :] = 0.0
    return x, mask


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("n", SHAPES)
def test_triangle_mult_matches_reference_within_existing_bf16_error(outgoing, n, device):
    layer = _make_mul(outgoing, device)
    x, mask = _mul_inputs(n, device)

    with torch.no_grad():
        truth = layer(x, mask, use_kernels=False).float()
    _assert_not_degenerate(truth, f"tri-mul fp32 reference (N={n})")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = layer(x, mask, use_kernels=False)
            triton_out = layer(x, mask, use_kernels=True)

    # PRECISION PARITY GATE — must pass before the error ratio means anything.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_dtypes = _record_matmul_dtypes(lambda: layer(x, mask, use_kernels=False))
        tri_dtypes = _record_matmul_dtypes(lambda: layer(x, mask, use_kernels=True))
    assert ref_dtypes == tri_dtypes, (
        f"tri-mul {'out' if outgoing else 'in'} N={n}: the two arms did NOT run at the same "
        f"precision (reference {sorted(str(d) for d in ref_dtypes)}, triton "
        f"{sorted(str(d) for d in tri_dtypes)}). The error ratio below would be measuring "
        "precision, not implementation. This is exactly the confound that made an earlier "
        "revision of this suite report 0.443 and call it a fp32-accumulation benefit."
    )

    err_ref = _relative_error(reference, truth)
    err_tri = _relative_error(triton_out, truth)
    ratio = err_tri / err_ref if err_ref > 0 else float("inf")

    _REPORT["equivalence"].append(
        {
            "seam": "mul",
            "direction": "outgoing" if outgoing else "incoming",
            "n": n,
            "matmul_dtypes_reference": sorted(str(d) for d in ref_dtypes),
            "matmul_dtypes_triton": sorted(str(d) for d in tri_dtypes),
            "err_reference_bf16": err_ref,
            "err_triton_bf16": err_tri,
            "ratio": ratio,
            "limit": ERROR_RATIO_LIMIT,
            "verdict": "PASS" if ratio <= ERROR_RATIO_LIMIT else "FAIL",
        }
    )

    assert ratio <= ERROR_RATIO_LIMIT, (
        f"tri-mul {'out' if outgoing else 'in'} N={n}: triton relative error "
        f"{err_tri:.3e} vs reference bf16 error {err_ref:.3e}, ratio {ratio:.2f} "
        f"> limit {ERROR_RATIO_LIMIT}"
    )


@pytest.mark.parametrize("n", SHAPES)
def test_triangle_mult_timing(n, device):
    layer = _make_mul(True, device)
    x, mask = _mul_inputs(n, device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_ms = _sync_time(lambda: layer(x, mask, use_kernels=False))
        tri_ms = _sync_time(lambda: layer(x, mask, use_kernels=True))

    _REPORT["timing"].append(
        {
            "seam": "mul",
            "n": n,
            "reference_ms": ref_ms,
            "triton_ms": tri_ms,
            "speedup": ref_ms / tri_ms if tri_ms > 0 else 0.0,
        }
    )


# --------------------------------------------------------------------------- #
# Triangle attention
# --------------------------------------------------------------------------- #


def _make_attn(device: torch.device):
    from boltz.model.layers.triangular_attention.attention import (
        TriangleAttentionStartingNode,
    )

    torch.manual_seed(0)
    layer = TriangleAttentionStartingNode(TOKEN_Z, HEAD_WIDTH, NUM_HEADS, inf=1e9)
    return _randomize(layer).to(device).eval()


def _attn_inputs(n: int, device: torch.device):
    torch.manual_seed(n)
    z = torch.randn(1, n, n, TOKEN_Z, device=device)
    mask = torch.ones(1, n, n, device=device)
    mask[:, :, int(n * 0.9) :] = 0.0
    return z, mask


@pytest.mark.parametrize("n", SHAPES)
def test_triangle_attention_matches_reference_within_existing_bf16_error(n, device):
    layer = _make_attn(device)
    z, mask = _attn_inputs(n, device)

    with torch.no_grad():
        truth = layer(z, mask=mask, chunk_size=None, use_kernels=False).float()
    _assert_not_degenerate(truth, f"tri-attn fp32 reference (N={n})")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = layer(z, mask=mask, chunk_size=None, use_kernels=False)
            triton_out = layer(z, mask=mask, chunk_size=None, use_kernels=True)

    # PRECISION PARITY GATE — see the mul case. attn was already fair, because
    # _prep_qkv's Linears are autocast-intercepted; this asserts it stays fair.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_dtypes = _record_matmul_dtypes(lambda: layer(z, mask=mask, chunk_size=None, use_kernels=False))
        tri_dtypes = _record_matmul_dtypes(lambda: layer(z, mask=mask, chunk_size=None, use_kernels=True))
    assert ref_dtypes == tri_dtypes, (
        f"tri-attn N={n}: arms ran at different precisions (reference "
        f"{sorted(str(d) for d in ref_dtypes)}, triton {sorted(str(d) for d in tri_dtypes)}); "
        "the error ratio would be measuring precision, not implementation."
    )

    err_ref = _relative_error(reference, truth)
    err_tri = _relative_error(triton_out, truth)
    ratio = err_tri / err_ref if err_ref > 0 else float("inf")

    _REPORT["equivalence"].append(
        {
            "seam": "attn",
            "direction": "starting",
            "n": n,
            "matmul_dtypes_reference": sorted(str(d) for d in ref_dtypes),
            "matmul_dtypes_triton": sorted(str(d) for d in tri_dtypes),
            "err_reference_bf16": err_ref,
            "err_triton_bf16": err_tri,
            "ratio": ratio,
            "limit": ERROR_RATIO_LIMIT,
            "verdict": "PASS" if ratio <= ERROR_RATIO_LIMIT else "FAIL",
        }
    )

    assert ratio <= ERROR_RATIO_LIMIT, (
        f"tri-attn N={n}: triton relative error {err_tri:.3e} vs reference bf16 "
        f"error {err_ref:.3e}, ratio {ratio:.2f} > limit {ERROR_RATIO_LIMIT}"
    )


@pytest.mark.parametrize("n", SHAPES)
def test_triangle_attention_timing(n, device):
    layer = _make_attn(device)
    z, mask = _attn_inputs(n, device)
    # Reference chunking is what the served model actually uses at this size.
    chunk = 128 if n > 384 else 512

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_ms = _sync_time(lambda: layer(z, mask=mask, chunk_size=chunk, use_kernels=False))
        tri_ms = _sync_time(lambda: layer(z, mask=mask, chunk_size=chunk, use_kernels=True))

    _REPORT["timing"].append(
        {
            "seam": "attn",
            "n": n,
            "chunk_size": chunk,
            "reference_ms": ref_ms,
            "triton_ms": tri_ms,
            "speedup": ref_ms / tri_ms if tri_ms > 0 else 0.0,
        }
    )


# --------------------------------------------------------------------------- #
# Negative control — the check must be able to fail
# --------------------------------------------------------------------------- #


def test_head_transpose_bug_is_detected(device):
    """Reintroduce the hazard the adapter warns about and confirm the gate catches it.

    ``EvoformerAttention`` already returns ``[*, Q, H, C]``. Boltz's own
    cuEquivariance branch transposes because cuEquivariance returns
    ``[*, H, Q, C]``, so a port that keeps that transpose scrambles heads against
    residues. With ``NUM_HEADS=4`` and ``N>4`` the shapes disagree, so this
    control uses a square case (N == NUM_HEADS is impractical) and instead
    transposes the kernel output before ``_wrap_up`` at a shape where it is
    legal, proving the error metric moves far outside the limit.
    """
    from rocm_triangle import adapter

    n = NUM_HEADS  # square in (H, Q) so the wrong transpose is shape-legal
    layer = _make_attn(device)
    z, mask = _attn_inputs(n, device)

    with torch.no_grad():
        truth = layer(z, mask=mask, chunk_size=None, use_kernels=False).float()
    _assert_not_degenerate(truth, "negative-control fp32 reference")

    original = adapter.triton_triangular_attn

    def buggy_impl(module, q_x, kv_x, tri_bias, mask_bias):
        from rocm_triangle.evoformer import TritonEvoformer

        q, k, v = module._prep_qkv(q_x, kv_x, apply_scale=False)  # noqa: SLF001
        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))
        out = TritonEvoformer(q, k, v, mask_bias.to(q.dtype), tri_bias.to(q.dtype), True)
        # THE BUG: boltz's cuEquivariance-shaped transpose, wrongly kept.
        return module._wrap_up(out.transpose(-2, -3), q_x)  # noqa: SLF001

    adapter.triton_triangular_attn = buggy_impl
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = layer(z, mask=mask, chunk_size=None, use_kernels=False)
            wrong = layer(z, mask=mask, chunk_size=None, use_kernels=True)
    finally:
        adapter.triton_triangular_attn = original

    err_ref = _relative_error(reference, truth)
    err_bug = _relative_error(wrong, truth)
    ratio = err_bug / err_ref if err_ref > 0 else float("inf")

    _REPORT["negative_control"].append(
        {
            "case": "attention output transposed (cuEquivariance-shaped)",
            "n": n,
            "err_reference_bf16": err_ref,
            "err_buggy": err_bug,
            "ratio": ratio,
            "limit": ERROR_RATIO_LIMIT,
            "detected": ratio > ERROR_RATIO_LIMIT,
        }
    )

    assert ratio > ERROR_RATIO_LIMIT, (
        "the equivalence gate did NOT detect a transposed attention output "
        f"(ratio {ratio:.2f} <= limit {ERROR_RATIO_LIMIT}); the gate has no power"
    )
