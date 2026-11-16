# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Install OpenFold3's ROCm Triton triangle kernels into Boltz-2 at runtime.

Boltz-2's accelerated triangle path is cuEquivariance, which has no ROCm
build, so the AIM serves with ``use_kernels=False``. That argument is about
cuEquivariance specifically, not about kernels in general: the sibling
``assets/instinct/openfold/openfold3`` AIM runs Triton triangle kernels on ROCm
today because AMD's implementations are merged into *its* upstream. This module
carries those same kernels (vendored in ``evoformer.py`` / ``primitives.py``) and
adapts them to Boltz-2's call shapes.

Why a runtime shim rather than a source patch
---------------------------------------------
Iron rule 2 keeps upstream model source off-limits as a target of edits, and
``boltz`` arrives here as an installed wheel, so patching it would be an edit to
``site-packages``. The same reasoning already governs ``runner.bounded_msa_server``.
``install()`` therefore wraps three bound ``forward`` methods and leaves every
upstream file untouched.

Why ``forward`` and not the two ``kernel_*`` functions
------------------------------------------------------
Boltz names its two seams ``kernel_triangular_mult`` and ``kernel_triangular_attn``,
but replacing those cannot give independent control of the two paths. Boltz gates
both off one model-level ``use_kernels``; ``PairformerLayer.forward`` does carry
separate ``use_cuequiv_mul`` / ``use_cuequiv_attn`` booleans, but ``Boltz2`` never
threads them, so they are unreachable from the checkpoint-loading path this AIM
uses. Wrapping ``forward`` instead means a *disabled* seam can delegate to the
captured original with ``use_kernels=False`` — i.e. run boltz's own reference
bytes — while an enabled seam takes the Triton path. That matters because the two
seams have different mechanisms and must be measured apart (see MECHANISM below).

MECHANISM — what each seam actually buys, and why they differ
--------------------------------------------------------------
``attn`` replaces the whole attention with ``TritonEvoformer``, a fused
flash-attention-style kernel that never materialises the ``[*, H, N, N]`` logits.
Boltz's reference path builds them and chunks over rows to survive the memory
cost (``chunk_size_tri_attn`` is 128 above 384 tokens, else 512). This is a real
algorithmic change in memory traffic. For one unpadded sample the attention runs through
torch SDPA instead (``sdpa_triangular_attn``): the pair bias is the attention mask, with a
leading axis of 1 so it broadcasts over the rows.

``mul`` is ``trimul_l1.py``: boltz's triangle multiply written for TorchInductor. It keeps
the tensors channel-major around the contraction (no permute copies) and lets Inductor
fuse the norm, gate, mask and cast chains.

Numerical fidelity
------------------
``trimul_l1`` follows boltz's own composition and weights. The two contractions were
checked to be the same operation by index algebra, outgoing
``out[i,j,d] = sum_k a[i,k,d] b[j,k,d]`` and incoming
``out[i,j,d] = sum_k a[k,i,d] b[k,j,d]``. The wrapper returns the update (boltz's
caller adds the residual) and the dtype policy is boltz's under bf16 autocast.

And one hazard that is *not* a difference but a trap: ``TritonEvoformer`` already
transposes its output back to ``[*, Q, H, C]``, whereas boltz applies its own
``o.transpose(-2, -3)`` after the cuEquivariance call. The ported attention path
must therefore NOT transpose. Getting this wrong yields a service that returns
plausible-looking structures with silently scrambled heads, which is why
``tests/unit/test_triton_adapter.py`` asserts it against a CPU reference.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

import torch

from .trimul_l1 import MAX_TOKENS, trimul_l1

logger = logging.getLogger(__name__)

# Accepted install modes.
#
# The two axes are independent and every combination is reachable, which is the
# point: a mode must be able to turn exactly one thing on, or a measurement
# cannot attribute a change to it.
#
#   attention seam   OFF      ON
#   mul seam
#     off            off      attn
#     l1             mul      both
MODE_OFF = "off"
MODE_ATTN = "attn"
MODE_MUL = "mul"
MODE_BOTH = "both"
VALID_MODES = (MODE_OFF, MODE_ATTN, MODE_MUL, MODE_BOTH)

# Which triangle-multiplication implementation a mode selects.
MUL_IMPL_NONE = None
MUL_IMPL_L1 = "l1"

# Set by install(); read by the wrappers.
_MUL_ENABLED = False
_ATTN_ENABLED = False
_MUL_IMPL = MUL_IMPL_NONE
_INSTALLED = False

# Captured originals, so a disabled seam runs boltz's own code and so tests can
# compare against it in-process.
_ORIG: dict[str, Callable[..., Any]] = {}

# Per direction (True = outgoing): the first module and call state seen by the wrapper.
_SEEN: dict[bool, tuple[Any, torch.dtype, torch.dtype, bool, bool]] = {}

_WARNED: set[str] = set()

# Triangle attention on inputs of at least this many tokens runs through torch SDPA with the
# pair bias as the attention mask, when no token is padded. Other inputs use the Triton kernel.
SDPA_MIN_TOKENS = 186

# The memory-efficient SDPA kernel gives wrong bf16 output when the key length is not a
# multiple of this value. The keys are padded to it, with -inf pair bias on the padded keys.
SDPA_KEY_ALIGN = 32

# Token count of the start-up warm-up of the long-sequence path (above UNCHUNKED_MAX_TOKENS).
WARMUP_LARGE_TOKENS = 2100


def mode_flags(mode: str) -> tuple[bool, bool]:
    """Return ``(mul_enabled, attn_enabled)`` for a mode string.

    Pure, so the CPU unit tier can exercise every branch with no GPU and no
    ``boltz`` import.
    """
    normalized = _normalize(mode)
    return (
        normalized in (MODE_MUL, MODE_BOTH),
        normalized in (MODE_ATTN, MODE_BOTH),
    )


def _normalize(mode: str) -> str:
    normalized = (mode or MODE_OFF).strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError(f"triangle kernel mode must be one of {VALID_MODES}, got {mode!r}")
    return normalized


def seam_status() -> dict[str, Any]:
    """The seams in effect now: ``attention`` on/off and the ``mul_impl`` (None when off)."""
    return {"attention": _ATTN_ENABLED, "mul_impl": _MUL_IMPL if _MUL_ENABLED else MUL_IMPL_NONE}


def mul_impl(mode: str) -> Optional[str]:
    """Which triangle-multiplication implementation ``mode`` selects, or None.

    Kept separate from :func:`mode_flags` rather than widened into it, so the
    existing two-flag contract every caller and test already depends on is
    unchanged, and so the mul *implementation* is a third independent axis
    rather than a special case of the on/off one.
    """
    normalized = _normalize(mode)
    if normalized in (MODE_MUL, MODE_BOTH):
        return MUL_IMPL_L1
    return MUL_IMPL_NONE


def triton_available() -> bool:
    """Whether the vendored kernels can actually run here."""
    try:
        from .evoformer import TritonEvoformer
        from .primitives import TRITON_AVAILABLE
    except ImportError:
        return False
    return bool(TRITON_AVAILABLE) and TritonEvoformer is not None


# --------------------------------------------------------------------------- #
# Triangle multiplicative update
# --------------------------------------------------------------------------- #


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log ``message`` once per process for ``key``."""
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message, *args)


def _record_call(module: Any, x: torch.Tensor, mask: torch.Tensor, outgoing: bool) -> None:
    """Remember one module and the call state per direction, for the start-up warm-up."""
    if outgoing not in _SEEN:
        _SEEN[outgoing] = (
            module,
            x.dtype,
            mask.dtype,
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
        )


def prepare_model(model: Any) -> int:
    """Pack the ``trimul_l1`` weights of every triangle multiply layer of ``model``.

    Returns the number of layers packed. Call it once after the weights are on
    their device, so no request packs weights.
    """
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    from .trimul_l1 import params_for

    _SEEN.clear()
    count = 0
    for module in model.modules():
        if isinstance(module, (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)):
            params_for(module)
            count += 1
    return count


def warm_up_large(tokens: int = WARMUP_LARGE_TOKENS) -> dict[str, Any]:
    """Compile the ``trimul_l1`` graphs for lengths above ``UNCHUNKED_MAX_TOKENS``.

    Runs both directions once on a synthetic ``tokens`` x ``tokens`` pair tensor, with
    the dtypes and the grad state of the first real calls, so the compile guards match
    real requests. Needs one real call per direction first (the start-up warm-up
    prediction provides it).
    """
    from .trimul_l1 import UNCHUNKED_MAX_TOKENS, trimul_l1

    if tokens <= UNCHUNKED_MAX_TOKENS:
        raise ValueError(f"tokens must be above {UNCHUNKED_MAX_TOKENS}, got {tokens}")
    missing = [name for name, outgoing in (("outgoing", True), ("incoming", False)) if outgoing not in _SEEN]
    if missing:
        raise RuntimeError(f"no real triangle multiply call seen yet for: {', '.join(missing)}")
    for outgoing in (True, False):
        module, x_dtype, mask_dtype, grad, inference = _SEEN[outgoing]
        device = next(module.parameters()).device
        with torch.inference_mode(inference), torch.set_grad_enabled(grad):
            x = torch.zeros(1, tokens, tokens, module.norm_in.weight.shape[0], dtype=x_dtype, device=device)
            mask = torch.ones(1, tokens, tokens, dtype=mask_dtype, device=device)
            trimul_l1(module, x, mask, outgoing=outgoing)
            del x, mask
    return {"tokens": tokens}


# --------------------------------------------------------------------------- #
# Triangle attention
# --------------------------------------------------------------------------- #


def triton_triangular_attn(
    module: Any,
    q_x: torch.Tensor,
    kv_x: torch.Tensor,
    tri_bias: torch.Tensor,
    mask_bias: torch.Tensor,
) -> torch.Tensor:
    """Boltz's triangle attention through OpenFold3's fused Triton Evoformer kernel.

    Shape contract, which lines up with no adaptation because both trees inherit
    the OpenFold ``Attention`` primitive and both target the DeepSpeed-Evoformer
    layout:

    ==================  ==========================  ==============================
    kernel expects      boltz supplies              built by
    ==================  ==========================  ==============================
    Q/K/V  [B,S,R,H,D]  [B,I,H,J,C] -> transposed   ``Attention._prep_qkv``
    res_mask [B,S,1,1,R]  ``mask_bias`` [B,I,1,1,J] ``inf * (mask - 1)``
    pair_bias [B,1,H,R,R] ``tri_bias`` [B,1,H,I,J]  ``permute_final_dims(...).unsqueeze(-4)``
    ==================  ==========================  ==============================

    ``res_mask`` is additive (``QK_block = tl.dot(Q, K) + res_mask_block``), so
    boltz's ``-1e9`` mask bias is the right thing to pass. Scaling matches too:
    the kernel multiplies Q by ``DIM ** -0.5`` internally, and boltz asks
    ``_prep_qkv`` not to pre-scale when kernels are on.

    AUTOCAST — this seam needs no manual cast.
    ``q``/``k``/``v`` are produced by ``module._prep_qkv``, whose ``nn.Linear``
    calls autocast intercepts normally, so they already arrive in the autocast
    dtype; ``tri_bias`` likewise comes from an intercepted ``Linear``. The kernel
    is then fed exactly what boltz's reference ``_attention`` would have
    multiplied. Only ``mask_bias`` is built arithmetically (``inf * (mask - 1)``)
    and so is cast explicitly below. ``tests/smoke/test_triton_equivalence.py``
    asserts this dtype parity rather than trusting it.

    Returns ``[*, Q, C_q]`` — already through ``_wrap_up``.
    """
    from .evoformer import TritonEvoformer

    # evoformer.py binds this to None when triton is unimportable, and install()
    # refuses to patch in that case, so reaching here means it is bound. The
    # assert is what carries that invariant across the module boundary.
    assert TritonEvoformer is not None

    # Kernel scales internally, so ask for unscaled q/k/v.
    q, k, v = module._prep_qkv(q_x, kv_x, apply_scale=False)  # noqa: SLF001

    # [*, H, Q, C] -> [*, Q, H, C], the layout the kernel documents.
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # The kernel is fp16/bf16 only.
    original_dtype = q.dtype
    if original_dtype not in (torch.bfloat16, torch.float16):
        out = TritonEvoformer(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            mask_bias.to(torch.bfloat16),
            tri_bias.to(torch.bfloat16),
            True,
        ).to(original_dtype)
    else:
        out = TritonEvoformer(q, k, v, mask_bias.to(q.dtype), tri_bias.to(q.dtype), True)

    # NO transpose here. TritonEvoformer already returns [*, Q, H, C]; boltz's
    # cuEquivariance branch transposes because cuEquivariance returns
    # [*, H, Q, C]. Adding one back scrambles heads silently.
    return module._wrap_up(out, q_x)  # noqa: SLF001


def sdpa_triangular_attn(module: Any, q_x: torch.Tensor, kv_x: torch.Tensor, tri_bias: torch.Tensor) -> torch.Tensor:
    """Triangle attention through ``scaled_dot_product_attention`` for one unpadded sample.

    ``q_x`` is ``[1, I, J, C]`` and ``tri_bias`` is ``[1, 1, H, I, J]``. The pair bias is
    the attention mask, ``[1, H, J, J]``, and broadcasts over the ``I`` rows: no
    ``[I, H, J, J]`` tensor exists. The keys are padded to ``SDPA_KEY_ALIGN``. The padding
    mask is not used, so the caller must check that no token is padded.

    Returns ``[1, I, J, C_q]``, through ``_wrap_up``.
    """
    q, k, v = module._prep_qkv(q_x, kv_x, apply_scale=False)  # noqa: SLF001
    _, rows, heads, length, dim = q.shape
    q, k, v = (t.reshape(rows, heads, length, dim) for t in (q, k, v))
    bias = tri_bias.reshape(heads, length, -1).to(q.dtype).unsqueeze(0)
    pad = -length % SDPA_KEY_ALIGN
    if pad:
        k, v = (torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (k, v))
        bias = torch.nn.functional.pad(bias, (0, pad), value=float("-inf"))
    bias = bias.contiguous()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale_for(module))
    # [I, H, J, C] -> [1, I, J, H, C], the layout of the kernel path.
    out = out.unsqueeze(0).transpose(-2, -3)
    return module._wrap_up(out, q_x)  # noqa: SLF001


def _unpadded(mask: torch.Tensor) -> bool:
    """True when every token is valid. Synchronises with the device."""
    return bool(mask.all().item())


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


def install(mode: str) -> tuple[bool, bool]:
    """Patch boltz's triangle layers according to ``mode``; return the flags applied.

    Idempotent: the originals are captured once, and re-calling only re-points the
    two enable flags. ``mode='off'`` leaves boltz entirely alone on the first call.
    """
    global _MUL_ENABLED, _ATTN_ENABLED, _MUL_IMPL, _INSTALLED

    mul_on, attn_on = mode_flags(mode)
    impl = mul_impl(mode)
    if not (mul_on or attn_on) and not _INSTALLED:
        return (False, False)

    if (mul_on or attn_on) and not triton_available():
        raise RuntimeError(
            f"triangle kernel mode {mode!r} but the vendored Triton kernels are "
            "unavailable (triton not importable, or evoformer.TritonEvoformer is None)."
        )

    _MUL_ENABLED, _ATTN_ENABLED, _MUL_IMPL = mul_on, attn_on, impl
    if _INSTALLED:
        return (mul_on, attn_on)

    from boltz.model.layers.triangular_attention.primitives import Attention
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    _ORIG["mul_out"] = TriangleMultiplicationOutgoing.forward
    _ORIG["mul_in"] = TriangleMultiplicationIncoming.forward
    _ORIG["attn"] = Attention.forward

    def _make_mul_forward(original: Callable[..., Any], outgoing: bool) -> Callable[..., Any]:
        def forward(self, x, mask, use_kernels: bool = False):  # noqa: ANN001, ANN202
            if use_kernels and _MUL_ENABLED:
                if x.shape[1] <= MAX_TOKENS:
                    _record_call(self, x, mask, outgoing)
                    return trimul_l1(self, x, mask, outgoing=outgoing)
                _warn_once(
                    "long",
                    "trimul_l1 supports at most %d tokens, got %d: using the upstream path",
                    MAX_TOKENS,
                    x.shape[1],
                )
            # Force the reference branch: with the seam disabled we run boltz's
            # own code, never a reimplementation of it.
            return original(self, x, mask, use_kernels=False)

        return forward

    def _attn_forward(self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels: bool = False):  # noqa: ANN001, ANN202
        if use_kernels and _ATTN_ENABLED:
            if q_x.shape[0] == 1 and q_x.shape[-2] >= SDPA_MIN_TOKENS:
                if _unpadded(mask):
                    return sdpa_triangular_attn(self, q_x, kv_x, tri_bias)
                _warn_once("padded", "triangle attention input has padded tokens: using the Triton kernel")
            return triton_triangular_attn(self, q_x, kv_x, tri_bias, mask_bias)
        return _ORIG["attn"](self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False)

    TriangleMultiplicationOutgoing.forward = _make_mul_forward(_ORIG["mul_out"], True)
    TriangleMultiplicationIncoming.forward = _make_mul_forward(_ORIG["mul_in"], False)
    Attention.forward = _attn_forward
    _INSTALLED = True

    logger.info("ROCm Triton triangle kernels installed (mul=%s, attn=%s)", mul_on, attn_on)
    return (mul_on, attn_on)


def reference_forward(name: str) -> Optional[Callable[..., Any]]:
    """The captured pre-patch ``forward`` for ``name`` in ``{mul_out, mul_in, attn}``.

    Exposed so an equivalence test can drive boltz's own implementation and the
    Triton one from the same process against the same weights.
    """
    return _ORIG.get(name)


def scale_for(module: Any) -> float:
    """Boltz's attention scale, for tests that build a reference by hand."""
    return 1.0 / math.sqrt(module.c_hidden)
