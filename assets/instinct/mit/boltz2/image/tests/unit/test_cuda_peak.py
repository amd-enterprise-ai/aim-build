# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""How the runner measures ``runtime.cuda_peak_gb``."""

from __future__ import annotations

import torch
from runner import predict_tracking_cuda_peak


def _fake_cuda(monkeypatch, calls: list[tuple], peak_bytes: int) -> None:
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append(("reset", device)))
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda device=None: calls.append(("read", device)) or peak_bytes
    )


def test_the_peak_is_reset_before_the_prediction_and_read_after_it_on_the_device(monkeypatch):
    calls: list[tuple] = []
    _fake_cuda(monkeypatch, calls, peak_bytes=12_345_000_000)
    device = torch.device("cuda", 0)

    peak = predict_tracking_cuda_peak(lambda: calls.append(("predict", None)), device)

    assert calls == [("reset", device), ("predict", None), ("read", device)]
    assert peak == 12.35


def test_a_request_not_on_cuda_reports_none_and_never_touches_the_counters(monkeypatch):
    calls: list[tuple] = []
    _fake_cuda(monkeypatch, calls, peak_bytes=1)

    for device in (None, torch.device("cpu")):
        assert predict_tracking_cuda_peak(lambda: calls.append(("predict", None)), device) is None

    assert calls == [("predict", None), ("predict", None)]
