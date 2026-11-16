# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Boltz-2 triangle multiplication written for Inductor fusion (level 1).

Same math and weights as ``TriangleMultiplicationOutgoing`` and
``TriangleMultiplicationIncoming`` in ``boltz/model/layers/triangular_mult.py``.
The code is written from that math. It uses no vendored kernel.

Layout plan
-----------
The contraction is ``c[d,i,j] = sum_k a[d,i,k] * b[d,j,k]`` (outgoing) or
``sum_k a[d,k,i] * b[d,k,j]`` (incoming). It is one batched matrix multiply
over the channel axis ``d``. The pair tensor has ``d`` as its last axis, so a
channel-last layout needs a permute copy before and after the contraction.
This code keeps the tensors channel-major between the two projections instead:

1. ``w @ xb^T`` GEMMs give the channel-major projections of ``a``, ``b`` and
   their gates directly (the transpose is a GEMM flag, not a copy).
2. One pointwise kernel makes ``a`` and ``b`` (``[D, N, N]``, contiguous):
   sigmoid gate, mask and cast fused.
3. ``bmm`` does the contraction. Incoming only transposes a view of ``a``.
4. The output LayerNorm reduces over the channel axis (outer reduction, the
   spatial axis stays contiguous), and writes channel-major ``cn``.
5. ``cn^T @ w_out^T`` gives channel-last ``[N*N, D]`` directly.
6. The output gate is a pointwise multiply with ``sigmoid(xb @ w_gate^T)``.

Dtype policy follows the eager path under bf16 autocast: input LayerNorm in
fp32, projections and contraction in bf16 with fp32 accumulation, output
LayerNorm in fp32, output returned in ``OUT_DTYPE``.

All layers share one compiled function. Weights are arguments, not constants,
so a new layer or a new length does not recompile. Compile it with
``dynamic=True`` (``compiled_core``). Lengths above ``UNCHUNKED_MAX_TOKENS``
use a second graph (one recompile per direction, on the first such call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

COMPUTE_DTYPE = torch.bfloat16
OUT_DTYPE = torch.float32

# Above UNCHUNKED_MAX_TOKENS the projection GEMM runs in PROJ_CHUNKS column chunks. The longer
# path is a second compiled graph, built on the first call above this size.
UNCHUNKED_MAX_TOKENS = 2048
PROJ_CHUNKS = 4

# Size limit: D * N * N must stay below 2**31 elements (one tensor index range).
MAX_TOKENS = 4000


@dataclass
class TriMulL1Params:
    """Weights of one triangle multiply layer, packed for ``trimul_l1``."""

    norm_in_weight: torch.Tensor  # [D] fp32
    norm_in_bias: torch.Tensor  # [D] fp32
    w_in: torch.Tensor  # [4D, D] bf16: rows p_in (2D) then g_in (2D)
    w_gate_out: torch.Tensor  # [D, D] bf16
    norm_out_weight: torch.Tensor  # [D] fp32
    norm_out_bias: torch.Tensor  # [D] fp32
    w_out: torch.Tensor  # [D, D] bf16
    eps_in: float
    eps_out: float

    @classmethod
    def from_module(cls, module: Any) -> "TriMulL1Params":
        """Pack the weights of a Boltz ``TriangleMultiplication*`` module."""
        with torch.no_grad():
            return cls(
                norm_in_weight=module.norm_in.weight.detach().float().contiguous(),
                norm_in_bias=module.norm_in.bias.detach().float().contiguous(),
                w_in=torch.cat([module.p_in.weight, module.g_in.weight]).detach().to(COMPUTE_DTYPE).contiguous(),
                w_gate_out=module.g_out.weight.detach().to(COMPUTE_DTYPE).contiguous(),
                norm_out_weight=module.norm_out.weight.detach().float().contiguous(),
                norm_out_bias=module.norm_out.bias.detach().float().contiguous(),
                w_out=module.p_out.weight.detach().to(COMPUTE_DTYPE).contiguous(),
                eps_in=float(module.norm_in.eps),
                eps_out=float(module.norm_out.eps),
            )


def _project(xb: torch.Tensor, mask: torch.Tensor, w_in: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated, masked projections ``a`` and ``b``, channel-major ``[B*D, N, N]``, from ``xb`` ``[B, N*N, D]``."""
    batch, pairs, dim = xb.shape
    xt = xb.transpose(-1, -2)
    m = mask.reshape(batch, 1, pairs).float()
    if n <= UNCHUNKED_MAX_TOKENS:
        # One GEMM per D weight rows: p_a, p_b, g_a, g_b.
        pa, pb, ga, gb = (torch.matmul(w, xt) for w in w_in.chunk(4))
        a = (pa.float() * torch.sigmoid(ga.float()) * m).to(COMPUTE_DTYPE)
        b = (pb.float() * torch.sigmoid(gb.float()) * m).to(COMPUTE_DTYPE)
    else:
        # hipBLASLt returns wrong values for some elements when a GEMM has more than about 4M
        # columns, so split the pair axis.
        step = ((pairs + PROJ_CHUNKS - 1) // PROJ_CHUNKS + 7) // 8 * 8
        a_parts, b_parts = [], []
        for k in range(PROJ_CHUNKS):
            lo, hi = k * step, (pairs if k == PROJ_CHUNKS - 1 else (k + 1) * step)
            pg = torch.matmul(w_in, xt[:, :, lo:hi])
            mk = m[:, :, lo:hi]
            a_parts.append(
                (pg[:, :dim].float() * torch.sigmoid(pg[:, 2 * dim : 3 * dim].float()) * mk).to(COMPUTE_DTYPE)
            )
            b_parts.append(
                (pg[:, dim : 2 * dim].float() * torch.sigmoid(pg[:, 3 * dim :].float()) * mk).to(COMPUTE_DTYPE)
            )
        a = torch.cat(a_parts, dim=2)
        b = torch.cat(b_parts, dim=2)
    return a.reshape(batch * dim, n, n), b.reshape(batch * dim, n, n)


def _core(
    x: torch.Tensor,
    mask: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    w_in: torch.Tensor,
    w_gate_out: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    w_out: torch.Tensor,
    outgoing: bool,
    eps_in: float = 1e-5,
    eps_out: float = 1e-5,
) -> torch.Tensor:
    """Triangle multiply update for ``x`` ``[B, N, N, D]`` and ``mask`` ``[B, N, N]``."""
    batch, n, _, dim = x.shape
    pairs = n * n

    xb = F.layer_norm(x.float(), (dim,), norm_in_weight, norm_in_bias, eps_in).to(COMPUTE_DTYPE)
    xb = xb.reshape(batch, pairs, dim)

    a, b = _project(xb, mask, w_in, n)

    if outgoing:
        c = torch.bmm(a, b.transpose(1, 2))
    else:
        c = torch.bmm(a.transpose(1, 2), b)

    # Output LayerNorm over the channel axis, channel-major in and out.
    c = c.reshape(batch, dim, pairs).float()
    centered = c - c.mean(dim=1, keepdim=True)
    var = centered.square().mean(dim=1, keepdim=True)
    cn = centered * torch.rsqrt(var + eps_out) * norm_out_weight[:, None] + norm_out_bias[:, None]
    cn = cn.to(COMPUTE_DTYPE)

    # Channel-last output projection and gate.
    o = torch.matmul(cn.transpose(1, 2), w_out.t())
    gate = torch.matmul(xb, w_gate_out.t())
    out = o.float() * torch.sigmoid(gate.float())
    return out.to(OUT_DTYPE).reshape(batch, n, n, dim)


_COMPILED: Optional[Callable[..., torch.Tensor]] = None


def compiled_core() -> Callable[..., torch.Tensor]:
    """The shared ``torch.compile(dynamic=True)`` wrapper of ``_core`` (built once)."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(_core, dynamic=True, fullgraph=True)
    return _COMPILED


def params_for(module: Any) -> TriMulL1Params:
    """Packed weights for ``module``, cached on the module."""
    params = getattr(module, "_trimul_l1_params", None)
    if params is None:
        params = TriMulL1Params.from_module(module)
        module._trimul_l1_params = params  # noqa: SLF001
    return params


def trimul_l1(
    module: Any,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    outgoing: bool,
    compiled: bool = True,
) -> torch.Tensor:
    """Boltz's triangle multiplicative update for ``module``; returns the update (no residual)."""
    if x.shape[1] > MAX_TOKENS:
        raise ValueError(f"trimul_l1 supports at most {MAX_TOKENS} tokens, got {x.shape[1]}")
    p = params_for(module)
    fn = compiled_core() if compiled else _core
    # The graph has explicit dtypes. Autocast off keeps one cache entry for any caller state.
    with torch.autocast(device_type=x.device.type, enabled=False):
        return fn(
            x,
            mask,
            p.norm_in_weight,
            p.norm_in_bias,
            p.w_in,
            p.w_gate_out,
            p.norm_out_weight,
            p.norm_out_bias,
            p.w_out,
            outgoing,
            p.eps_in,
            p.eps_out,
        )
