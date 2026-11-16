# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Lightning strategy that leaves the model on the GPU between requests.

Lightning's ``SingleDeviceStrategy`` moves the module to the device when a
``Trainer`` starts and back to the CPU in ``Strategy.teardown``. A served model
is built once and reused, so that round trip copies about 5100 tensors per
request (about 0.6 s). This strategy skips both moves when the module is already
on the target device. Teardown still releases the precision plugin, the
accelerator (CUDA cache) and the checkpoint IO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_STRATEGY_CLASS: type | None = None


def module_on_device(module: torch.nn.Module, device: torch.device) -> bool:
    """True when every parameter and buffer of ``module`` is on ``device``."""
    tensors = list(module.parameters()) + list(module.buffers())
    return bool(tensors) and all(t.device == device for t in tensors)


def resolve_device(accelerator: str) -> torch.device | None:
    """The device a ``gpu`` / ``cuda`` / ``auto`` accelerator request runs on, else ``None``."""
    import torch

    if accelerator in ("gpu", "cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda", 0)
    return None


def resident_strategy(device: torch.device) -> Any:
    """A single-device strategy for ``device`` that does not move the model off it."""
    global _STRATEGY_CLASS
    if _STRATEGY_CLASS is None:
        from pytorch_lightning.strategies import SingleDeviceStrategy

        class ResidentSingleDeviceStrategy(SingleDeviceStrategy):
            def model_to_device(self) -> None:
                assert self.model is not None, "self.model must be set before self.model.to()"
                if not module_on_device(self.model, self.root_device):
                    self.model.to(self.root_device)

            def teardown(self) -> None:
                assert self.accelerator is not None
                self.precision_plugin.teardown()
                self.accelerator.teardown()
                self.checkpoint_io.teardown()

        _STRATEGY_CLASS = ResidentSingleDeviceStrategy
    return _STRATEGY_CLASS(device=device)
