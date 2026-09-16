# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Per-worker GPU affinity for the OpenFold3 BentoML service.

An N-accelerator OpenFold3 profile runs N data-parallel BentoML workers, one
GPU each. This module maps a worker onto its GPU and restricts the process to
it, so a prediction never spreads across the machine.

Follows BentoML's documented multi-GPU pattern — ``worker_index - 1`` picks the
device — but narrows the visibility env vars instead of building a
``torch.device``, so the restriction also covers whatever the model does
internally:
https://docs.bentoml.com/en/latest/build-with-bentoml/gpu-inference.html

No torch, no bentoml — this must run before the ROCm runtime initialises. The
service passes in the worker index.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

logger = logging.getLogger(__name__)

# PyTorch-ROCm honours either variable; setting only one leaves the other free
# to disagree, which surfaces as a wrong-device failure far from its cause.
_VISIBILITY_ENV_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

DEV_DRI = Path("/dev/dri")


def current_visible_devices(environ: Mapping[str, str] | None = None) -> list[str] | None:
    """Return the device ids currently visible to this process, or None if unrestricted.

    ``HIP_VISIBLE_DEVICES`` wins over ``CUDA_VISIBLE_DEVICES`` when both are set,
    matching the ROCm runtime's own precedence. An empty value means "no devices"
    and is reported as an empty list, not as unrestricted.
    """
    env = os.environ if environ is None else environ
    for name in _VISIBILITY_ENV_VARS:
        if (raw := env.get(name)) is not None:
            return [part.strip() for part in raw.split(",") if part.strip()]
    return None


def _detected_gpu_count() -> int:
    """Accelerator count from aim-runtime's detector, or 0 when unavailable.

    aim_runtime lives on PYTHONPATH from the AIM base layer, so it is importable
    when the service is launched by ``aim-runtime serve`` but not when ``bentoml
    serve`` is run directly.
    """
    try:
        from aim_runtime.gpu_detector import GPUDetector
    except ImportError:
        return 0
    try:
        return GPUDetector().gpu_count
    except Exception:  # detection is best-effort; never block startup on it
        logger.debug("aim-runtime GPU detection failed", exc_info=True)
        return 0


def _render_node_count(dev_dri: Path) -> int:
    """Number of ``renderD*`` nodes under ``dev_dri``, or 0 if it cannot be read."""
    try:
        if not dev_dri.is_dir():
            return 0
        return sum(1 for entry in dev_dri.iterdir() if entry.name.startswith("renderD"))
    except OSError:
        return 0


def visible_device_count(dev_dri: Path = DEV_DRI) -> int | None:
    """Return the number of accelerators exposed to this container, or None if unknown.

    Prefers aim-runtime's detector, which validates candidates against sysfs and
    filters to the container's own devices. Its ``gpu_count`` cannot express
    "undeterminable" — it reports 0 both for a GPU-less host and for a container
    exposing only ``renderD*`` nodes — so a zero falls through to counting those
    nodes directly.

    Returns None rather than 0 when neither source finds anything. Callers use
    this to reject an over-sized worker count, so guessing low would fail a
    working deployment; staying silent only leaves the pre-existing behaviour.
    """
    return _detected_gpu_count() or _render_node_count(dev_dri) or None


def select_worker_device(
    visible: list[str] | None,
    worker_index: int,
    worker_count: int,
    device_count: int | None = None,
) -> str:
    """Return the device id worker ``worker_index`` should use.

    ``worker_index`` is BentoML's 1-based index. ``visible`` is the device list
    already imposed on the process (from :func:`current_visible_devices`); when
    it is None the workers address devices ``0..worker_count-1`` directly, and
    ``device_count`` (from :func:`visible_device_count`) bounds that range when
    it is known.

    Raises ValueError when the worker cannot be given a device of its own —
    failing at startup beats two workers silently sharing one GPU.
    """
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")
    if not 1 <= worker_index <= worker_count:
        raise ValueError(f"worker_index {worker_index} out of range for {worker_count} worker(s)")

    slot = worker_index - 1
    if visible is None:
        if device_count is not None and device_count < worker_count:
            raise ValueError(
                f"AIM_ACCELERATOR_COUNT={worker_count} requires {worker_count} accelerator(s), "
                f"but this container has {device_count}. Lower AIM_ACCELERATOR_COUNT (and use the "
                f"matching tp{device_count} profile), or give the container more accelerators."
            )
        return str(slot)
    if len(visible) < worker_count:
        raise ValueError(
            f"AIM_ACCELERATOR_COUNT={worker_count} requires {worker_count} accelerator(s), but "
            f"HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES expose {len(visible)}: {visible}"
        )
    if len(set(visible)) < worker_count:
        raise ValueError(
            f"HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES repeat device ids ({visible}), so "
            f"{worker_count} worker(s) cannot each get their own accelerator"
        )
    return visible[slot]


def pin_worker_device(
    worker_index: int,
    worker_count: int,
    environ: MutableMapping[str, str] | None = None,
    dev_dri: Path = DEV_DRI,
) -> str:
    """Restrict this process to the single device belonging to ``worker_index``.

    Must run before torch is imported — the ROCm runtime reads these variables
    at initialisation. Returns the pinned device id.
    """
    env = os.environ if environ is None else environ
    device = select_worker_device(
        current_visible_devices(env),
        worker_index,
        worker_count,
        device_count=visible_device_count(dev_dri),
    )
    for name in _VISIBILITY_ENV_VARS:
        env[name] = device
    return device
