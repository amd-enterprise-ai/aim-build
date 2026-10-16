# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Pad and crop rules for the OpenFold3 bucketing patch.

The module exists only after the image applies ``of3_bucketing.patch``, and it
imports torch. These tests extract that file from the patch and run it against
a stand-in tensor so they do not need a GPU, torch, or OpenFold3.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

OF3_IMAGE_DIR = Path(__file__).resolve().parents[1]
OF3_BUCKETING_PATCH = OF3_IMAGE_DIR / "patches/of3_bucketing.patch"
_MODULE_HEADER = "+++ b/openfold3/core/utils/of3_bucketing.py\n"


class _Tensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape = []
        for size, sl in zip(self.shape, idx):
            start = 0 if sl.start is None else sl.start
            stop = size if sl.stop is None else sl.stop
            shape.append(stop - start)
        return _Tensor(shape)


def _is_tensor(value: object) -> bool:
    return isinstance(value, _Tensor)


def _pad(tensor: _Tensor, pad_pairs: list[int], value: int = 0) -> _Tensor:
    del value
    shape = list(tensor.shape)
    for index, dim in enumerate(reversed(range(tensor.dim()))):
        shape[dim] += pad_pairs[2 * index] + pad_pairs[2 * index + 1]
    return _Tensor(shape)


def _install_fake_torch() -> None:
    functional = types.ModuleType("torch.nn.functional")
    setattr(functional, "pad", _pad)
    nn = types.ModuleType("torch.nn")
    setattr(nn, "functional", functional)
    torch = types.ModuleType("torch")
    setattr(torch, "Tensor", _Tensor)
    setattr(torch, "is_tensor", _is_tensor)
    setattr(torch, "nn", nn)
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional


def _bucketing_source() -> str:
    text = OF3_BUCKETING_PATCH.read_text(encoding="utf-8")
    start = text.index(_MODULE_HEADER) + len(_MODULE_HEADER)
    body = text[start:].split("\n", 1)[1]
    lines = []
    for line in body.splitlines():
        if not line.startswith("+"):
            break
        lines.append(line[1:])
    return "\n".join(lines) + "\n"


def _load_bucketing():
    saved = {name: sys.modules.get(name) for name in ("torch", "torch.nn", "torch.nn.functional")}
    _install_fake_torch()
    try:
        spec = importlib.util.spec_from_loader("_of3_bucketing_under_test", loader=None)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        exec(compile(_bucketing_source(), str(OF3_BUCKETING_PATCH), "exec"), module.__dict__)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


bkt = _load_bucketing()


def _T(shape):
    return _Tensor(shape)


def _batch(n_token, n_atom, **extra):
    data = {"token_mask": _T([n_token]), "atom_mask": _T([n_atom])}
    data.update(extra)
    return data


@pytest.fixture
def bucketing(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_BUCKET", "1")
    monkeypatch.delenv("OPENFOLD3_BUCKET_ATOMS", raising=False)
    return bkt


def test_exact_ladder_step_is_not_padded(bucketing):
    out, n_token, n_atom = bucketing.bucketize_batch(_batch(256, 2048))
    assert (n_token, n_atom) == (256, 2048)
    assert out["token_mask"].shape == (256,)
    assert out["atom_mask"].shape == (2048,)


def test_next_step_is_the_smallest_size_at_least_the_input(bucketing):
    out, _, _ = bucketing.bucketize_batch(_batch(257, 2049))
    assert out["token_mask"].shape == (384,)
    assert out["atom_mask"].shape == (3072,)


def test_token_and_atom_steps_are_independent(bucketing):
    out, _, _ = bucketing.bucketize_batch(_batch(40, 256, restype=_T([40, 32]), ref_pos=_T([256, 3])))
    assert out["token_mask"].shape == (256,)
    assert out["restype"].shape == (256, 32)
    assert out["atom_mask"].shape == (2048,)
    assert out["ref_pos"].shape == (2048, 3)


def test_oversized_input_stays_at_its_real_size(bucketing):
    out, n_token, n_atom = bucketing.bucketize_batch(_batch(2049, 16385))
    assert (n_token, n_atom) == (2049, 16385)
    assert out["token_mask"].shape == (2049,)
    assert out["atom_mask"].shape == (16385,)


def test_confidence_output_already_at_real_atom_length_is_kept():
    cropped = bkt.crop_output(
        {
            "plddt_logits": _T([1, 1, 1380, 50]),
            "experimentally_resolved_logits": _T([1, 1, 1380, 2]),
            "si_trunk": _T([1, 256, 384]),
        },
        256,
        186,
        2048,
        1380,
    )
    assert cropped["plddt_logits"].shape == (1, 1, 1380, 50)
    assert cropped["experimentally_resolved_logits"].shape == (1, 1, 1380, 2)
    assert cropped["si_trunk"].shape == (1, 186, 384)


def test_listed_axis_of_neither_padded_nor_real_length_raises(bucketing):
    with pytest.raises(RuntimeError, match="msa"):
        bucketing.bucketize_batch(_batch(32, 10, msa=_T([8, 32])))
    with pytest.raises(RuntimeError, match="999"):
        bkt.crop_output({"plddt_logits": _T([1, 1, 999, 50])}, 256, 186, 2048, 1380)
