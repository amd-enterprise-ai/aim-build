# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""In-pod LoRA adapter watcher for dynamic mode (ADR-0004).

Polls ``AIM_ADAPTER_SOURCE`` every ``AIM_ADAPTER_REFRESH_INTERVAL`` seconds,
diffs the directory against the last-known state, and reconciles the engine via
its OpenAI-compatible runtime endpoints:

    appear        -> POST /v1/load_lora_adapter
    content change-> POST /v1/load_lora_adapter (load_inplace=true)
    disappear     -> POST /v1/unload_lora_adapter

Disk is ground truth: an adapter unloaded out-of-band but still present on disk
is re-loaded on the next tick (ADR §Known hazards). Failures are logged and the
loop continues — a bad adapter must not crash the pod.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from aim_runtime.engines.adapters import ADAPTER_CONFIG_FILENAME, ADAPTER_WEIGHT_FILENAMES, enumerate_adapters

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30

# Files whose (size, mtime) make up an adapter's change signature. Weights are
# included so an in-place weight swap that leaves adapter_config.json untouched
# still triggers a reload.
_SIGNATURE_FILENAMES = (ADAPTER_CONFIG_FILENAME, *ADAPTER_WEIGHT_FILENAMES)

# (size, mtime) per signature file, in _SIGNATURE_FILENAMES order; (-1, 0.0) for
# a file that is absent or unreadable.
Signature = tuple[tuple[int, float], ...]


@dataclass
class _Seen:
    path: str
    signature: Signature  # (size, mtime) of config + weight files; change-detector


def _signature(adapter_path: str) -> Signature:
    """Return a change signature over the adapter's config and weight files.

    Cheap (stat only) but catches weight-only updates: a swapped
    adapter_model.safetensors changes its size/mtime even when
    adapter_config.json is untouched.
    """
    sig: list[tuple[int, float]] = []
    for fname in _SIGNATURE_FILENAMES:
        try:
            st = Path(adapter_path, fname).stat()
            sig.append((st.st_size, st.st_mtime))
        except OSError:
            sig.append((-1, 0.0))
    return tuple(sig)


class AdapterWatcher:
    """Reconciles on-disk adapters with a running engine over HTTP."""

    def __init__(self, source: str, max_rank: int, base_url: str, refresh_interval: int) -> None:
        self.source = source
        self.max_rank = max_rank
        self.base_url = base_url.rstrip("/")
        self.refresh_interval = refresh_interval
        self._seen: dict[str, _Seen] = {}

    def reconcile_once(self) -> None:
        """Run a single diff + reconcile pass against the engine."""
        current = {a.name: a for a in enumerate_adapters(self.source, self.max_rank)}

        # Adds and in-place updates.
        for name, adapter in current.items():
            sig = _signature(adapter.path)
            prev = self._seen.get(name)
            if prev is None:
                if self._load(name, adapter.path, load_inplace=False):
                    self._seen[name] = _Seen(path=adapter.path, signature=sig)
            elif sig != prev.signature:
                if self._load(name, adapter.path, load_inplace=True):
                    self._seen[name] = _Seen(path=adapter.path, signature=sig)

        # Unloads: present last tick, gone now.
        for name in [n for n in self._seen if n not in current]:
            if self._unload(name):
                del self._seen[name]

    def run_forever(self, stop) -> None:
        """Poll until ``stop()`` returns True."""
        logger.info(f"Adapter watcher polling {self.source} every {self.refresh_interval}s")
        while not stop():
            try:
                self.reconcile_once()
            except Exception as exc:  # never let the watcher kill the pod
                logger.warning(f"Adapter reconcile pass failed (continuing): {exc}")
            time.sleep(self.refresh_interval)

    def _load(self, name: str, path: str, load_inplace: bool) -> bool:
        """Load/reload an adapter. Returns True iff the result is terminal.

        A terminal result (True) means ``reconcile_once`` records the adapter as
        seen and stops acting on it until its signature changes:
          - 2xx: loaded successfully.
          - 4xx: a permanent client error (bad weights, rank over cap, …). Log
            once and mark seen so we don't hammer the endpoint every tick.
        A non-terminal result (False) leaves the adapter unseen so the next tick
        retries:
          - 5xx: transient server error (engine busy/restarting).
          - connection error / timeout.
        """
        payload = {"lora_name": name, "lora_path": path, "load_inplace": load_inplace}
        try:
            resp = requests.post(f"{self.base_url}/v1/load_lora_adapter", json=payload, timeout=_HTTP_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning(f"Transient error loading adapter '{name}' (will retry next tick): {exc}")
            return False
        if resp.status_code >= 500:
            logger.warning(
                f"Server error loading adapter '{name}' (will retry next tick): "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )
            return False
        if resp.status_code >= 400:
            logger.error(f"Failed to load adapter '{name}': HTTP {resp.status_code} {resp.text[:200]}")
            return True
        logger.info(f"{'Reloaded' if load_inplace else 'Loaded'} adapter '{name}'")
        return True

    def _unload(self, name: str) -> bool:
        """Unload an adapter. Returns True iff the adapter is confirmed gone.

        Only a confirmed removal (True) lets ``reconcile_once`` drop the adapter
        from ``_seen``; otherwise it stays tracked and the unload is retried next
        tick (so a deleted-on-disk adapter the engine failed to drop can't keep
        serving silently until pod restart):
          - 2xx: unloaded successfully.
          - 404: the engine has no such adapter — already gone.
        Non-terminal (False), retried next tick:
          - other 4xx (e.g. 409 conflict), 5xx, connection error / timeout.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=_HTTP_TIMEOUT
            )
        except requests.RequestException as exc:
            logger.warning(f"Transient error unloading adapter '{name}' (will retry next tick): {exc}")
            return False
        if resp.status_code == 404:
            logger.info(f"Adapter '{name}' already absent from engine; treating as unloaded.")
            return True
        if resp.status_code >= 400:
            logger.warning(
                f"Failed to unload adapter '{name}' (will retry next tick): "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )
            return False
        logger.info(f"Unloaded adapter '{name}'")
        return True
