# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton primitives for the triangle multiplicative update.

VENDORED VERBATIM from openfold3/core/model/layers/triangular_multiplicative_update.py
at aqlaboratory/openfold-3 ref 0.4.1 (lines 37-608), the same ref the sibling
openfold3 AIM pins in its Dockerfile. Only the surrounding module code was
dropped: OF3's cuequivariance probe, its nn.Module class hierarchy, and its
imports of openfold3 internals. The kernels and their python wrappers are
unmodified, so this file tracks upstream and must be re-lifted, not hand-edited,
if the pinned ref moves.
"""

import warnings

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

warnings.filterwarnings("once")

if TRITON_AVAILABLE:

    @triton.jit
    def sigmoid_mul_kernel(
        x_ptr,
        gate_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel: output = x * sigmoid(gate)
        """
        pid = tl.program_id(0)

        block_start = pid * BLOCK_SIZE
        if block_start >= N:
            return

        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = (offs >= 0) & (offs < N)

        safe_offs = offs.to(tl.int64)
        x_vals = tl.load(x_ptr + safe_offs, mask=mask, other=0.0)
        gate_vals = tl.load(gate_ptr + safe_offs, mask=mask, other=0.0)

        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_vals))
        result = x_vals * sigmoid_gate

        tl.store(output_ptr + safe_offs, result, mask=mask)

    @triton.jit
    def layernorm_kernel(
        x_ptr,
        output_ptr,
        weight_ptr,
        bias_ptr,
        M,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        Vectorized LayerNorm kernel with multi-row processing and safe bounds checking.
        Each program processes ROWS_PER_PROGRAM rows to amortize launch overhead.
        Uses vectorized tl.sum reductions instead of scalar Welford loop.
        Normalizes over the last dimension (N).
        """
        row_start = tl.program_id(0) * ROWS_PER_PROGRAM

        # Early exit if out of bounds
        if row_start >= M:
            return

        for row in range(ROWS_PER_PROGRAM):
            row_idx = row_start + row
            if row_idx < M:
                row_offset = row_idx.to(tl.int64) * N
                x_row_ptr = x_ptr + row_offset
                out_row_ptr = output_ptr + row_offset

                # Pass 1: vectorized mean and variance computation
                sum_val = 0.0
                sum_sq = 0.0
                for block_start in range(0, N, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs >= 0) & (offs < N)
                    safe_offs = offs.to(tl.int64)
                    x_block = tl.load(x_row_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    sum_val += tl.sum(x_block, axis=0)
                    sum_sq += tl.sum(x_block * x_block, axis=0)

                mean = sum_val / N
                var = sum_sq / N - mean * mean
                rstd = 1.0 / tl.sqrt(tl.maximum(var, eps))

                # Pass 2: normalize and apply affine transform
                for block_start in range(0, N, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs >= 0) & (offs < N)

                    safe_offs = offs.to(tl.int64)
                    x_vals = tl.load(x_row_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    w_vals = tl.load(weight_ptr + safe_offs, mask=mask, other=1.0).to(
                        tl.float32
                    )
                    b_vals = tl.load(bias_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )

                    out = (x_vals - mean) * rstd * w_vals + b_vals
                    tl.store(out_row_ptr + safe_offs, out, mask=mask)

    @triton.jit
    def linear_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        output_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Linear layer: output = x @ weight.T + bias
        x: [M, K], weight: [N, K] (PyTorch layout), bias: [N], output: [M, N]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = rm < M
        n_mask = rn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            k_mask = rk < K

            x_offsets = (
                rm[:, None].to(tl.int64) * stride_xm
                + rk[None, :].to(tl.int64) * stride_xk
            )

            w_offsets = (
                rn[None, :].to(tl.int64) * stride_wn
                + rk[:, None].to(tl.int64) * stride_wk
            )

            x = tl.load(
                x_ptr + x_offsets,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w = tl.load(
                w_ptr + w_offsets,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            acc = tl.dot(x, w, acc)

        if HAS_BIAS:
            bias_offsets = rn.to(tl.int64)
            bias = tl.load(bias_ptr + bias_offsets, mask=n_mask, other=0.0)
            acc += bias[None, :]

        out_offsets = (
            rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
        )

        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            output_ptr + out_offsets,
            acc,
            mask=out_mask,
        )

    @triton.jit
    def linear_fused_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        other_ptr,
        mask_ptr,
        add_tensor_ptr,
        output_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        stride_other_m,
        stride_other_n,
        stride_mask_m,
        stride_mask_n,
        stride_add_m,
        stride_add_n,
        HAS_BIAS: tl.constexpr,
        APPLY_SIGMOID: tl.constexpr,
        APPLY_MUL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_ADD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused linear layer with optional sigmoid, elementwise multiply, mask, and add
        Supports combinations: linear [+ sigmoid] [* other] [* mask] [+ add_tensor]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = rm < M
        n_mask = rn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            k_mask = rk < K

            x_offsets = (
                rm[:, None].to(tl.int64) * stride_xm
                + rk[None, :].to(tl.int64) * stride_xk
            )

            w_offsets = (
                rn[None, :].to(tl.int64) * stride_wn
                + rk[:, None].to(tl.int64) * stride_wk
            )

            x = tl.load(
                x_ptr + x_offsets,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w = tl.load(
                w_ptr + w_offsets,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            acc = tl.dot(x, w, acc)

        if HAS_BIAS:
            bias_offsets = rn.to(tl.int64)
            bias = tl.load(bias_ptr + bias_offsets, mask=n_mask, other=0.0)
            acc += bias[None, :]

        if APPLY_SIGMOID:
            # Clamp to avoid exp overflow
            acc = tl.where(acc > 20.0, 20.0, acc)
            acc = tl.where(acc < -20.0, -20.0, acc)
            acc = 1.0 / (1.0 + tl.exp(-acc))

        if APPLY_MUL:
            other_offsets = (
                rm[:, None].to(tl.int64) * stride_other_m
                + rn[None, :].to(tl.int64) * stride_other_n
            )
            other_vals = tl.load(
                other_ptr + other_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc * other_vals

        if HAS_MASK:
            mask_offsets = (
                rm[:, None].to(tl.int64) * stride_mask_m
                + rn[None, :].to(tl.int64) * stride_mask_n
            )
            mask_vals = tl.load(
                mask_ptr + mask_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc * mask_vals

        if HAS_ADD:
            add_offsets = (
                rm[:, None].to(tl.int64) * stride_add_m
                + rn[None, :].to(tl.int64) * stride_add_n
            )
            add_vals = tl.load(
                add_tensor_ptr + add_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc + add_vals

        out_offsets = (
            rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
        )

        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            output_ptr + out_offsets,
            acc,
            mask=out_mask,
        )


def triton_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Fused sigmoid + elementwise multiply: x * sigmoid(gate)
    Supports broadcasting like PyTorch's native operations.

    Args:
        x: Input tensor
        gate: Gate tensor (broadcastable with x)

    Returns:
        x * sigmoid(gate)
    """
    broadcasted_shape = torch.broadcast_shapes(x.shape, gate.shape)
    x_expanded = x.expand(broadcasted_shape)
    gate_expanded = gate.expand(broadcasted_shape)

    x_flat = x_expanded.contiguous().reshape(-1)
    gate_flat = gate_expanded.contiguous().reshape(-1)
    N = x_flat.numel()

    output_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    sigmoid_mul_kernel[grid](x_flat, gate_flat, output_flat, N, BLOCK_SIZE=BLOCK_SIZE)

    return output_flat.reshape(broadcasted_shape)


def triton_layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Triton LayerNorm with vectorized reductions and multi-row processing.

    Args:
        x: Input tensor [..., N]
        weight: Scale parameters [N]
        bias: Shift parameters [N]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as x
    """
    input_shape = x.shape
    N = input_shape[-1]
    M = x.numel() // N

    x_2d = x.reshape(M, N).contiguous()
    output_2d = torch.empty_like(x_2d)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    ROWS_PER_PROGRAM = 8
    grid = (triton.cdiv(M, ROWS_PER_PROGRAM),)

    layernorm_kernel[grid](
        x_2d,
        output_2d,
        weight,
        bias,
        M,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
    )

    return output_2d.reshape(input_shape)


def triton_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Triton Linear: output = x @ weight.T + bias

    Args:
        x: Input tensor [..., K]
        weight: Weight tensor [N, K] (same layout as nn.Linear.weight)
        bias: Optional bias tensor [N]

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    K = input_shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bias_ptr = bias if bias is not None else weight  # guarded by HAS_BIAS constexpr

    linear_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    output_shape = input_shape[:-1] + (N,)
    return output_2d.reshape(output_shape)


def triton_linear_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    add_tensor: torch.Tensor | None = None,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """
    Fused linear layer with optional sigmoid, elementwise multiply, mask, and add.

    Computes: linear(x) [* sigmoid] [* other] [* mask] [+ add_tensor]

    Args:
        x: Input tensor [..., K]
        weight: Weight tensor [N, K] (same layout as nn.Linear.weight)
        bias: Optional bias tensor [N]
        other: Optional tensor to multiply with (same shape as output)
        mask: Optional mask tensor (broadcastable with output)
        add_tensor: Optional tensor to add to result (same shape as output)
        apply_sigmoid: Whether to apply sigmoid to linear result

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    K = input_shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    if other is not None:
        # other should match the output shape [..., N], so reshape accordingly
        other_M = other.numel() // N
        other_2d = other.reshape(other_M, N).contiguous()
        if other_M != M:
            raise ValueError(
                f"Shape mismatch: tensor expects {other_M} rows but output has {M}"
            )
        other_ptr = other_2d
        other_stride_m, other_stride_n = other_2d.stride(0), other_2d.stride(1)
    else:
        other_ptr = x_2d  # guarded by APPLY_MUL constexpr
        other_stride_m, other_stride_n = 0, 0

    if mask is not None:
        # mask should broadcast to output shape [..., N]
        mask_M = mask.numel() // (mask.shape[-1] if mask.numel() > 0 else 1)
        mask_features = mask.shape[-1] if mask.ndim > 0 else 1
        if mask_features == 1:
            # Broadcast mask to match output
            mask_2d = mask.reshape(mask_M, 1).expand(M, N).contiguous()
        else:
            mask_2d = mask.reshape(mask_M, mask_features).contiguous()
            if mask_M != M or mask_features != N:
                raise ValueError(
                    f"Mask shape mismatch: got {mask_M}x{mask_features}, "
                    f"expected {M}x{N}"
                )
        mask_ptr = mask_2d
        mask_stride_m, mask_stride_n = mask_2d.stride(0), mask_2d.stride(1)
    else:
        mask_ptr = x_2d  # guarded by HAS_MASK constexpr
        mask_stride_m, mask_stride_n = 0, 0

    if add_tensor is not None:
        # add_tensor should match the output shape [..., N]
        add_M = add_tensor.numel() // N
        add_tensor_2d = add_tensor.reshape(add_M, N).contiguous()
        if add_M != M:
            raise ValueError(
                f"Shape mismatch: add_tensor expects {add_M} rows but output has {M}"
            )
        add_tensor_ptr = add_tensor_2d
        add_stride_m, add_stride_n = add_tensor_2d.stride(0), add_tensor_2d.stride(1)
    else:
        add_tensor_ptr = x_2d  # guarded by HAS_ADD constexpr
        add_stride_m, add_stride_n = 0, 0

    bias_ptr = bias if bias is not None else x_2d  # guarded by HAS_BIAS constexpr

    linear_fused_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        other_ptr,
        mask_ptr,
        add_tensor_ptr,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        other_stride_m,
        other_stride_n,
        mask_stride_m,
        mask_stride_n,
        add_stride_m,
        add_stride_n,
        HAS_BIAS=(bias is not None),
        APPLY_SIGMOID=apply_sigmoid,
        APPLY_MUL=(other is not None),
        HAS_MASK=(mask is not None),
        HAS_ADD=(add_tensor is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    output_shape = input_shape[:-1] + (N,)
    return output_2d.reshape(output_shape)
