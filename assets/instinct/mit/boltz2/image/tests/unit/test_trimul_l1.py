# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU tests of ``rocm_triangle.trimul_l1``: the math, both paths, both directions.

The compiled GPU path is checked on the GPU (module accuracy, rule fp32-reference-v1).
Here the uncompiled function is compared with boltz's own module in fp32, with a
tolerance for its bf16 matrix multiplies.
"""

from __future__ import annotations

import pytest
import torch
from boltz.model.layers.triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
from rocm_triangle import trimul_l1 as l1

CLASSES = {True: TriangleMultiplicationOutgoing, False: TriangleMultiplicationIncoming}


def _case(outgoing: bool, n: int, batch: int = 1):
    torch.manual_seed(0)
    module = CLASSES[outgoing](128)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(batch, n, n, 128)
    valid = torch.ones(batch, n)
    valid[:, n - 3 :] = 0.0
    mask = valid[:, :, None] * valid[:, None, :]
    return module, x, mask


def _relative_error(y: torch.Tensor, ref: torch.Tensor) -> float:
    return ((y.float() - ref).abs().mean() / ref.abs().mean()).item()


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
def test_matches_boltz_in_fp32(outgoing, batch):
    module, x, mask = _case(outgoing, 24, batch)
    ref = module(x, mask)
    out = l1.trimul_l1(module, x, mask, outgoing=outgoing, compiled=False)
    assert out.shape == ref.shape and out.dtype == torch.float32
    assert _relative_error(out, ref) < 0.02


@pytest.mark.parametrize("outgoing", [True, False])
def test_the_chunked_projection_path_gives_the_same_result(monkeypatch, outgoing):
    module, x, mask = _case(outgoing, 24)
    plain = l1.trimul_l1(module, x, mask, outgoing=outgoing, compiled=False)
    monkeypatch.setattr(l1, "UNCHUNKED_MAX_TOKENS", 8)
    chunked = l1.trimul_l1(module, x, mask, outgoing=outgoing, compiled=False)
    assert _relative_error(chunked, plain) < 1e-3


def test_the_two_directions_differ():
    module, x, mask = _case(True, 16)
    incoming_module = CLASSES[False](128)
    incoming_module.load_state_dict(module.state_dict())
    out = l1.trimul_l1(module, x, mask, outgoing=True, compiled=False)
    inc = l1.trimul_l1(incoming_module, x, mask, outgoing=False, compiled=False)
    assert not torch.allclose(out, inc)


def test_returns_the_update_without_a_residual():
    module, x, mask = _case(True, 16)
    out = l1.trimul_l1(module, x, mask, outgoing=True, compiled=False)
    assert _relative_error(out, module(x, mask)) < 0.02
    assert _relative_error(out, module(x, mask) + x) > 0.1


def test_masked_pairs_do_not_feed_the_contraction():
    module, x, mask = _case(True, 16)
    changed = x.clone()
    changed[:, -3:, :, :] += 5.0
    changed[:, :, -3:, :] += 5.0
    a = l1.trimul_l1(module, x, mask, outgoing=True, compiled=False)
    b = l1.trimul_l1(module, changed, mask, outgoing=True, compiled=False)
    assert torch.equal(a[:, :-3, :-3], b[:, :-3, :-3])


def test_rejects_more_than_max_tokens():
    module, _, _ = _case(True, 8)
    x = torch.zeros(1, l1.MAX_TOKENS + 1, 1, 128)
    with pytest.raises(ValueError, match="at most"):
        l1.trimul_l1(module, x, torch.ones(1, l1.MAX_TOKENS + 1, 1), outgoing=True, compiled=False)


def test_params_are_packed_once_and_in_bf16():
    module, _, _ = _case(True, 8)
    first = l1.params_for(module)
    assert l1.params_for(module) is first
    assert first.w_in.shape == (512, 128) and first.w_in.dtype == torch.bfloat16
    assert torch.equal(first.w_in[:256].float(), module.p_in.weight.to(torch.bfloat16).float())
    assert torch.equal(first.w_in[256:].float(), module.g_in.weight.to(torch.bfloat16).float())
