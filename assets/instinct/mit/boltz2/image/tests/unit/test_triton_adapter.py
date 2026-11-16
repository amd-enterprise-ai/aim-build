# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tier for the ROCm Triton triangle-kernel adapter.

Triton itself needs a GPU, so this tier cannot execute the kernels. What it CAN
do — and what actually catches the dangerous bugs — is pin the three
adaptation facts that are pure shape/algebra and that produce a *running but
wrong* service if they are got backwards:

1. the two contractions boltz uses are the ones the adapter emits,
2. the attention output layout, which differs between cuEquivariance (boltz's
   own kernel branch) and TritonEvoformer by one transpose,
3. the mode -> (mul, attn) mapping and its rejection of junk values.

Numerical equivalence of the kernels themselves is a GPU claim and is made by
``tests/smoke/test_triton_equivalence.py``, not here.
"""

from __future__ import annotations

import pytest
import torch
from rocm_triangle import (
    MODE_ATTN,
    MODE_BOTH,
    MODE_MUL,
    MODE_OFF,
    MUL_IMPL_L1,
    MUL_IMPL_NONE,
    mode_flags,
    mul_impl,
)

# --------------------------------------------------------------------------- #
# Mode parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (MODE_OFF, (False, False)),
        (MODE_MUL, (True, False)),
        (MODE_ATTN, (False, True)),
        (MODE_BOTH, (True, True)),
        ("  BOTH  ", (True, True)),
    ],
)
def test_mode_flags_maps_every_declared_mode(mode, expected):
    assert mode_flags(mode) == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (MODE_OFF, MUL_IMPL_NONE),
        (MODE_ATTN, MUL_IMPL_NONE),
        (MODE_MUL, MUL_IMPL_L1),
        (MODE_BOTH, MUL_IMPL_L1),
    ],
)
def test_mul_impl_selects_the_contraction_implementation(mode, expected):
    """The mul seam selects one implementation, so a mode must name it unambiguously."""
    assert mul_impl(mode) == expected


def test_every_declared_mode_is_reachable_and_the_axes_are_independent():
    """Four modes, and each (attn, mul-impl) pair is reached by exactly one."""
    from rocm_triangle import VALID_MODES

    seen = {}
    for mode in VALID_MODES:
        mul_on, attn_on = mode_flags(mode)
        key = (attn_on, mul_impl(mode) if mul_on else MUL_IMPL_NONE)
        assert key not in seen, f"{mode!r} and {seen[key]!r} select the same behaviour"
        seen[key] = mode
    assert len(seen) == len(VALID_MODES) == 4


@pytest.mark.parametrize("removed", ["mulf", "bothf", "MULF", "BothF"])
def test_removed_fused_modes_are_rejected_not_remapped(removed):
    """The old fused-path modes must fail clearly, never silently resolve to another mode."""
    with pytest.raises(ValueError, match="triangle kernel mode"):
        mode_flags(removed)


def test_mode_flags_defaults_to_off_when_empty():
    """An unset env var must mean 'leave boltz alone', never 'guess'."""
    assert mode_flags("") == (False, False)
    assert mode_flags(None) == (False, False)


@pytest.mark.parametrize("bad", ["on", "true", "triton", "attn,mul", "1"])
def test_mode_flags_rejects_undeclared_values(bad):
    with pytest.raises(ValueError, match="triangle kernel mode"):
        mode_flags(bad)


def test_triton_and_no_kernels_settings_are_not_read(monkeypatch):
    """The seams are chosen only by BOLTZ2_OPTIMIZATIONS; the old settings do nothing."""
    from service import ServiceConfig

    monkeypatch.setenv("BOLTZ2_TRITON_KERNELS", "both")
    monkeypatch.setenv("BOLTZ2_NO_KERNELS", "false")
    config = ServiceConfig()
    assert config.optimizations == "on"
    assert not {"triton_kernels", "no_kernels"} & set(type(config).model_fields)
    monkeypatch.setenv("BOLTZ2_OPTIMIZATIONS", "off")
    assert ServiceConfig().optimized is False


@pytest.mark.parametrize(
    ("mode", "attention", "impl"),
    [(MODE_OFF, False, None), (MODE_ATTN, True, None), (MODE_MUL, False, "l1"), (MODE_BOTH, True, "l1")],
)
def test_seam_status_reports_the_flags_in_effect(monkeypatch, mode, attention, impl):
    import rocm_triangle.adapter as adapter

    mul_on, attn_on = adapter.mode_flags(mode)
    monkeypatch.setattr(adapter, "_MUL_ENABLED", mul_on)
    monkeypatch.setattr(adapter, "_ATTN_ENABLED", attn_on)
    monkeypatch.setattr(adapter, "_MUL_IMPL", adapter.mul_impl(mode))
    assert adapter.seam_status() == {"attention": attention, "mul_impl": impl}


# --------------------------------------------------------------------------- #
# Hazard 1 — the contraction the adapter emits is boltz's
# --------------------------------------------------------------------------- #


def _boltz_reference_contraction(a: torch.Tensor, b: torch.Tensor, *, outgoing: bool) -> torch.Tensor:
    """Transcribed from boltz/model/layers/triangular_mult.py, both directions."""
    if outgoing:
        return torch.einsum("bikd,bjkd->bijd", a, b)
    return torch.einsum("bkid,bkjd->bijd", a, b)


@pytest.mark.parametrize("outgoing", [True, False])
def test_of3_permute_form_equals_boltz_einsum_form(outgoing):
    """OpenFold3 spells the same contraction with permutes; prove it is the same.

    This is the load-bearing semantic-compatibility claim of the whole port. If
    these two disagree, the vendored kernels are computing a different operation
    and no amount of numerical tolerance rescues it.
    """
    torch.manual_seed(0)
    a = torch.randn(1, 7, 7, 5, dtype=torch.float64)
    b = torch.randn(1, 7, 7, 5, dtype=torch.float64)

    boltz = _boltz_reference_contraction(a, b, outgoing=outgoing)

    # OF3: permute_final_dims to [*, C, I, J], einsum over the last two dims,
    # then permute back. (2,0,1)/(2,1,0) for outgoing, (2,1,0)/(2,0,1) for
    # incoming — see openfold3 FusedTriangleMultiplicativeUpdate.
    def permute_final_dims(t, order):
        first = list(range(len(t.shape[:-3])))
        return t.permute(first + [len(first) + i for i in order])

    if outgoing:
        a_p, b_p = permute_final_dims(a, (2, 0, 1)), permute_final_dims(b, (2, 1, 0))
    else:
        a_p, b_p = permute_final_dims(a, (2, 1, 0)), permute_final_dims(b, (2, 0, 1))
    of3 = permute_final_dims(torch.einsum("...ij,...jk->...ik", a_p, b_p), (1, 2, 0))

    assert torch.allclose(boltz, of3, atol=1e-12), (
        f"outgoing={outgoing}: boltz and OpenFold3 contractions differ; "
        f"max abs diff={(boltz - of3).abs().max().item():.3e}"
    )


def test_the_two_directions_are_actually_different():
    """Negative control: a test that passes for both directions proves nothing."""
    torch.manual_seed(0)
    a = torch.randn(1, 6, 6, 3, dtype=torch.float64)
    b = torch.randn(1, 6, 6, 3, dtype=torch.float64)
    out = _boltz_reference_contraction(a, b, outgoing=True)
    inc = _boltz_reference_contraction(a, b, outgoing=False)
    assert not torch.allclose(out, inc)


# --------------------------------------------------------------------------- #
# Hazard 2 — the attention output layout
# --------------------------------------------------------------------------- #


def test_attention_output_layout_differs_by_one_transpose():
    """cuEquivariance returns [*,H,Q,C]; TritonEvoformer returns [*,Q,H,C].

    boltz applies ``o.transpose(-2, -3)`` after its cuEquivariance call. The
    ported path must NOT, because ``EvoformerAttention.forward`` already does
    ``O = O.transpose(-2, -3)`` before returning. This asserts the two layouts
    are genuinely distinct, so dropping the transpose is a real requirement and
    not a cosmetic one — with H != Q the shapes disagree outright, and with
    H == Q it would silently transpose heads against residues.
    """
    batch, rows, heads, seq, dim = 1, 4, 3, 5, 2
    cueq_style = torch.randn(batch, rows, heads, seq, dim)
    triton_style = cueq_style.transpose(-2, -3)

    assert cueq_style.shape == (batch, rows, heads, seq, dim)
    assert triton_style.shape == (batch, rows, seq, heads, dim)
    # _wrap_up flattens the final two dims, so feeding it the wrong layout
    # produces a differently-ordered vector even when the shapes happen to match.
    square = torch.randn(batch, rows, heads, heads, dim)
    assert not torch.allclose(
        square.flatten(-2), square.transpose(-2, -3).flatten(-2)
    ), "square head/seq case must still be order-sensitive"


def test_boltz_mask_and_bias_shapes_match_the_kernel_contract():
    """boltz's mask_bias / triangle_bias already have the kernel's documented shapes.

    EvoformerAttention documents res_mask as [Batch, N_seq, 1, 1, N_res] and
    pair_bias as [Batch, 1, Head, N_res, N_res]. Rebuild both the way
    TriangleAttention.forward does and assert the ranks line up, so a future
    boltz version that changes them fails here rather than inside a Triton
    kernel.
    """
    batch, i, j, heads, inf = 2, 9, 9, 4, 1e9

    mask = torch.ones(batch, i, j)
    mask = mask[..., :, None, None, :]
    mask_bias = inf * (mask - 1)
    assert mask_bias.shape == (batch, i, 1, 1, j)

    linear_out = torch.randn(batch, i, j, heads)
    first = list(range(len(linear_out.shape[:-3])))
    triangle_bias = linear_out.permute(first + [len(first) + k for k in (2, 0, 1)])
    triangle_bias = triangle_bias.unsqueeze(-4)
    assert triangle_bias.shape == (batch, 1, heads, i, j)


def test_mask_bias_is_additive_not_multiplicative():
    """The kernel does QK + res_mask, so a valid position must contribute 0."""
    mask = torch.tensor([[[1.0, 0.0]]])
    mask_bias = 1e9 * (mask[..., :, None, None, :] - 1)
    assert mask_bias.flatten()[0].item() == 0.0
    assert mask_bias.flatten()[1].item() == -1e9


def test_adapter_does_not_transpose_the_attention_output():
    """The single most dangerous line that could be added back."""
    import inspect

    from rocm_triangle import adapter

    source = inspect.getsource(adapter.triton_triangular_attn)
    body = source.split('"""')[-1]
    assert "out.transpose" not in body, (
        "triton_triangular_attn must not transpose the kernel output; "
        "EvoformerAttention already returns [*, Q, H, C]"
    )


# --------------------------------------------------------------------------- #
# Install-time behaviour
# --------------------------------------------------------------------------- #


def test_install_off_is_a_no_op_without_touching_boltz(monkeypatch):
    """mode=off must not even import boltz, so the default path cannot regress."""
    import sys

    from rocm_triangle import adapter

    monkeypatch.setattr(adapter, "_INSTALLED", False, raising=False)
    before = "boltz.model.layers.triangular_mult" in sys.modules
    assert adapter.install(MODE_OFF) == (False, False)
    after = "boltz.model.layers.triangular_mult" in sys.modules
    assert after == before, "install('off') imported boltz; it must be inert"


@pytest.mark.parametrize("mode", [MODE_ATTN, MODE_MUL, MODE_BOTH])
def test_install_refuses_when_triton_is_unavailable(monkeypatch, mode):
    """Fail loudly at startup rather than at the first forward pass.

    This is the shape of the bug the KB records for cuEquivariance: a lazy
    import inside the kernel path means an import-only healthcheck passes on an
    image that cannot serve a request. install() is called from
    load_boltz2_model at startup and raises there instead.
    """
    from rocm_triangle import adapter

    monkeypatch.setattr(adapter, "triton_available", lambda: False)
    monkeypatch.setattr(adapter, "_INSTALLED", False, raising=False)
    with pytest.raises(RuntimeError, match="Triton kernels are|unavailable"):
        adapter.install(mode)


# --------------------------------------------------------------------------- #
# trimul_l1 hooks: weight packing and the long-sequence warm-up
# --------------------------------------------------------------------------- #


def test_prepare_model_packs_every_triangle_multiply_layer():
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )
    from rocm_triangle import prepare_model

    model = torch.nn.ModuleList(
        [TriangleMultiplicationOutgoing(128), TriangleMultiplicationIncoming(128), torch.nn.Linear(2, 2)]
    )
    assert prepare_model(model) == 2
    assert all(hasattr(m, "_trimul_l1_params") for m in list(model)[:2])


def test_warm_up_large_needs_a_real_call_per_direction_first(monkeypatch):
    import rocm_triangle.adapter as adapter

    monkeypatch.setattr(adapter, "_SEEN", {})
    with pytest.raises(RuntimeError, match="no real triangle multiply call"):
        adapter.warm_up_large()


def test_warm_up_large_must_be_above_the_unchunked_limit():
    import rocm_triangle.adapter as adapter
    from rocm_triangle.trimul_l1 import UNCHUNKED_MAX_TOKENS

    assert adapter.WARMUP_LARGE_TOKENS > UNCHUNKED_MAX_TOKENS
    with pytest.raises(ValueError, match="above"):
        adapter.warm_up_large(UNCHUNKED_MAX_TOKENS)


def test_a_warning_is_logged_once_per_key(monkeypatch, caplog):
    import rocm_triangle.adapter as adapter

    monkeypatch.setattr(adapter, "_WARNED", set())
    with caplog.at_level("WARNING"):
        adapter._warn_once("long", "too long %d", 5000)
        adapter._warn_once("long", "too long %d", 6000)
        adapter._warn_once("padded", "padded")
    assert [r.getMessage() for r in caplog.records] == ["too long 5000", "padded"]


# --------------------------------------------------------------------------- #
# Triangle attention through SDPA
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tokens", [20, 32, 45])
@pytest.mark.parametrize("starting", [True, False])
def test_sdpa_triangle_attention_matches_boltz_in_fp32(monkeypatch, starting, tokens):
    from boltz.model.layers.triangular_attention import attention as att
    from rocm_triangle import adapter

    torch.manual_seed(0)
    module = att.TriangleAttention(32, 16, 4, starting=starting)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = torch.randn(1, tokens, tokens, 32)
    mask = torch.ones(1, tokens, tokens)
    ref = module(x, mask)

    def sdpa_attention(self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False):
        return adapter.sdpa_triangular_attn(self, q_x, kv_x, tri_bias)

    monkeypatch.setattr(att.Attention, "forward", sdpa_attention)
    out = module(x, mask, use_kernels=True)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("tokens", [513, 2074, 2048])
def test_sdpa_keys_are_padded_to_the_alignment_whenever_the_length_is_not_a_multiple(monkeypatch, tokens):
    from rocm_triangle import adapter

    seen = {}

    def fake_sdpa(q, k, v, attn_mask=None, scale=None):
        seen.update(
            q=q.shape[-2], k=k.shape[-2], v=v.shape[-2], bias=tuple(attn_mask.shape), bias_pad=attn_mask[..., tokens:]
        )
        return torch.zeros_like(q)

    class Module:
        c_hidden = 4

        def _prep_qkv(self, q_x, kv_x, apply_scale):
            return (torch.zeros(1, 1, 2, tokens, 4),) * 3

        def _wrap_up(self, out, q_x):
            return out

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    adapter.sdpa_triangular_attn(Module(), None, None, torch.zeros(1, 1, 2, tokens, tokens))
    padded = -(-tokens // adapter.SDPA_KEY_ALIGN) * adapter.SDPA_KEY_ALIGN
    assert seen["q"] == tokens
    assert seen["k"] == seen["v"] == padded
    assert seen["bias"] == (1, 2, tokens, padded)
    assert torch.isneginf(seen["bias_pad"]).all()


def test_unpadded_detects_a_padded_token():
    from rocm_triangle import adapter

    mask = torch.ones(1, 6, 6)
    assert adapter._unpadded(mask)
    mask[0, 5, :] = 0
    assert not adapter._unpadded(mask)


def test_the_sdpa_limit_is_a_module_constant_below_the_smallest_benchmark_target():
    from rocm_triangle import adapter

    assert 0 < adapter.SDPA_MIN_TOKENS <= 186  # 186 is the smallest size measured and the smallest benchmark target


def test_attention_wrapper_picks_sdpa_only_for_one_unpadded_sample_above_the_limit(monkeypatch):
    from boltz.model.layers.triangular_attention import attention as att
    from boltz.model.layers.triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
    from rocm_triangle import adapter

    originals = (att.Attention.forward, TriangleMultiplicationOutgoing.forward, TriangleMultiplicationIncoming.forward)
    monkeypatch.setattr(adapter, "triton_available", lambda: True)
    monkeypatch.setattr(adapter, "_INSTALLED", False)
    monkeypatch.setattr(adapter, "_ORIG", {})
    for name in ("_MUL_ENABLED", "_ATTN_ENABLED", "_MUL_IMPL"):
        monkeypatch.setattr(adapter, name, getattr(adapter, name))
    calls = []
    monkeypatch.setattr(adapter, "triton_triangular_attn", lambda *a: calls.append("triton") or "t")
    monkeypatch.setattr(adapter, "sdpa_triangular_attn", lambda *a: calls.append("sdpa") or "s")
    monkeypatch.setattr(adapter, "SDPA_MIN_TOKENS", 8)
    try:
        adapter.install("attn")
        forward = att.Attention.forward

        def run(batch, tokens, padded):
            mask = torch.ones(batch, tokens, 1, 1, tokens)
            if padded:
                mask[..., -1] = 0
            calls.clear()
            forward(None, torch.zeros(batch, tokens, tokens, 2), None, None, None, mask, use_kernels=True)
            return calls[0]

        assert run(1, 8, padded=False) == "sdpa"
        assert run(1, 7, padded=False) == "triton"
        assert run(1, 8, padded=True) == "triton"
        assert run(2, 8, padded=False) == "triton"
    finally:
        att.Attention.forward, TriangleMultiplicationOutgoing.forward, TriangleMultiplicationIncoming.forward = (
            originals
        )
