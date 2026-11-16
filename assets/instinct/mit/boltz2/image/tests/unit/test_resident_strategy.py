# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tier: the model stays on its device between requests."""

from __future__ import annotations

import pytest
import torch

pl = pytest.importorskip("pytorch_lightning")

from resident_strategy import module_on_device, resident_strategy, resolve_device  # noqa: E402


class _Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.cpu_calls = 0
        self.to_calls: list = []

    def predict_step(self, batch, batch_idx):
        return self.lin(batch[0])

    def cpu(self):
        self.cpu_calls += 1
        return super().cpu()

    def to(self, *args, **kwargs):
        self.to_calls.append(args)
        return super().to(*args, **kwargs)


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("gpu") == torch.device("cuda", 0)
    assert resolve_device("auto") == torch.device("cuda", 0)
    assert resolve_device("cpu") is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("gpu") is None


def test_module_on_device():
    module = torch.nn.Linear(2, 2)
    assert module_on_device(module, torch.device("cpu"))
    assert not module_on_device(module, torch.device("meta"))
    assert not module_on_device(torch.nn.Module(), torch.device("cpu"))


def test_model_to_device_does_nothing_when_the_model_is_already_there():
    strategy = resident_strategy(torch.device("cpu"))
    model = _Tiny()
    strategy.model = model
    strategy.model_to_device()
    assert model.to_calls == []


def test_model_to_device_moves_a_model_that_is_elsewhere():
    strategy = resident_strategy(torch.device("meta"))
    model = _Tiny()
    strategy.model = model
    strategy.model_to_device()
    assert model.to_calls == [(torch.device("meta"),)]


def test_teardown_keeps_the_model_and_releases_the_plugins():
    strategy = resident_strategy(torch.device("cpu"))
    model = _Tiny()
    strategy.model = model
    calls = []

    class _Part:
        def __init__(self, name):
            self.name = name

        def teardown(self):
            calls.append(self.name)

    strategy.precision_plugin = _Part("precision")
    strategy.accelerator = _Part("accelerator")
    strategy.checkpoint_io = _Part("checkpoint_io")
    strategy.teardown()
    assert model.cpu_calls == 0
    assert calls == ["precision", "accelerator", "checkpoint_io"]


def test_the_stock_strategy_does_move_the_model_to_cpu_at_teardown():
    """Why the subclass exists: pin the upstream behaviour it removes."""
    from pytorch_lightning.strategies import SingleDeviceStrategy

    strategy = SingleDeviceStrategy(device=torch.device("cpu"))
    model = _Tiny()
    strategy.connect(model)
    strategy.precision_plugin.teardown = lambda: None
    strategy.accelerator = type("A", (), {"teardown": lambda self: None})()
    strategy.teardown()
    assert model.cpu_calls == 1


def test_trainer_predict_runs_with_the_strategy_and_repeats_without_moving_the_model():
    model = _Tiny().eval()
    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(2, 2)), batch_size=2)
    for _ in range(2):
        trainer = pl.Trainer(
            strategy=resident_strategy(torch.device("cpu")),
            accelerator="cpu",
            devices=1,
            precision="bf16-mixed",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        model._trainer = None  # noqa: SLF001
        out = trainer.predict(model, dataloaders=batches)
        assert len(out) == 1
    assert model.cpu_calls == 0
    assert model.to_calls == []
