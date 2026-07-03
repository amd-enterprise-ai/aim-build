# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the dynamic-mode AdapterWatcher (ADR-0004)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from aim_runtime.engines.adapter_watcher import AdapterWatcher


def _make_adapter(source, name: str, rank: int = 16) -> None:
    d = source / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": rank}))
    (d / "adapter_model.safetensors").write_text("weights")


def _ok():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "ok"
    return resp


def _watcher(tmp_path):
    return AdapterWatcher(source=str(tmp_path), max_rank=32, base_url="http://localhost:8000", refresh_interval=5)


def _calls(mock_post):
    """Return [(endpoint_suffix, payload), ...] for each POST."""
    out = []
    for call in mock_post.call_args_list:
        url = call.args[0] if call.args else call.kwargs["url"]
        out.append((url.rsplit("/", 1)[-1], call.kwargs["json"]))
    return out


class TestReconcile:
    def test_add_loads_adapter(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()) as post:
            w.reconcile_once()
        calls = _calls(post)
        assert len(calls) == 1
        endpoint, payload = calls[0]
        assert endpoint == "load_lora_adapter"
        assert payload["lora_name"] == "a1"
        assert payload["load_inplace"] is False

    def test_unchanged_adapter_not_reloaded(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()) as post:
            w.reconcile_once()
            w.reconcile_once()  # second pass: nothing changed
        assert len(_calls(post)) == 1

    def test_content_change_triggers_inplace_reload(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()) as post:
            w.reconcile_once()
            # Bump the config mtime to simulate a weight update.
            cfg = tmp_path / "a1" / "adapter_config.json"
            cfg.write_text(json.dumps({"r": 16, "note": "v2"}))
            import os
            import time

            future = time.time() + 10
            os.utime(cfg, (future, future))
            w.reconcile_once()
        calls = _calls(post)
        assert len(calls) == 2
        assert calls[1][0] == "load_lora_adapter"
        assert calls[1][1]["load_inplace"] is True

    def test_weight_only_update_triggers_inplace_reload(self, tmp_path):
        # Swapping the weights file without touching adapter_config.json must
        # still reload: the signature includes the weight file's size/mtime.
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()) as post:
            w.reconcile_once()
            weights = tmp_path / "a1" / "adapter_model.safetensors"
            weights.write_text("new weights v2")  # different size
            import os
            import time

            future = time.time() + 10
            os.utime(weights, (future, future))
            w.reconcile_once()
        calls = _calls(post)
        assert len(calls) == 2
        assert calls[1][0] == "load_lora_adapter"
        assert calls[1][1]["load_inplace"] is True

    def test_removal_triggers_unload(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()) as post:
            w.reconcile_once()
            import shutil

            shutil.rmtree(tmp_path / "a1")
            w.reconcile_once()
        calls = _calls(post)
        assert calls[1][0] == "unload_lora_adapter"
        assert calls[1][1]["lora_name"] == "a1"

    def test_ground_truth_reload_after_manual_unload(self, tmp_path):
        # Adapter still on disk but engine reports it gone → re-loaded next tick.
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()):
            w.reconcile_once()
        # Simulate out-of-band unload by clearing the watcher's seen state.
        w._seen.clear()
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()) as post:
            w.reconcile_once()
        assert _calls(post)[0][0] == "load_lora_adapter"


class TestFailureHandling:
    def test_transient_error_retries_next_tick(self, tmp_path):
        import requests

        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch(
            "aim_runtime.engines.adapter_watcher.requests.post",
            side_effect=requests.ConnectionError("boom"),
        ) as post:
            w.reconcile_once()  # fails, not marked seen
            w.reconcile_once()  # retried
        assert len(_calls(post)) == 2  # attempted both ticks

    def test_permanent_4xx_not_retried_forever(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        bad = MagicMock()
        bad.status_code = 400
        bad.text = "bad adapter"
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=bad) as post:
            w.reconcile_once()
            w.reconcile_once()
        assert len(_calls(post)) == 1  # marked seen after the 4xx; not hammered

    def test_server_error_on_load_retries_next_tick(self, tmp_path):
        # 5xx is transient (engine busy/restarting): not marked seen, retried.
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        err = MagicMock()
        err.status_code = 503
        err.text = "service unavailable"
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=err) as post:
            w.reconcile_once()
            w.reconcile_once()
        assert len(_calls(post)) == 2  # retried both ticks
        assert "a1" not in w._seen

    def test_failed_unload_retried_next_tick(self, tmp_path):
        # Engine fails the unload but adapter is gone from disk: keep retrying so
        # a removed adapter can't keep serving until pod restart.
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()):
            w.reconcile_once()  # load
        import shutil

        shutil.rmtree(tmp_path / "a1")
        err = MagicMock()
        err.status_code = 500
        err.text = "boom"
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=err) as post:
            w.reconcile_once()
            w.reconcile_once()
        calls = _calls(post)
        assert len(calls) == 2  # unload attempted both ticks
        assert all(c[0] == "unload_lora_adapter" for c in calls)
        assert "a1" in w._seen  # still tracked until confirmed gone

    def test_unload_404_treated_as_gone(self, tmp_path):
        # 404 means the engine has no such adapter: confirmed gone, stop retrying.
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=_ok()):
            w.reconcile_once()
        import shutil

        shutil.rmtree(tmp_path / "a1")
        gone = MagicMock()
        gone.status_code = 404
        gone.text = "not found"
        with patch("aim_runtime.engines.adapter_watcher.requests.post", return_value=gone) as post:
            w.reconcile_once()
            w.reconcile_once()
        assert len(_calls(post)) == 1  # dropped from _seen after 404; not retried
        assert "a1" not in w._seen

    def test_run_forever_guard_swallows_unexpected_errors(self, tmp_path):
        # An unexpected error in a reconcile pass must not crash the loop/pod.
        _make_adapter(tmp_path, "a1")
        w = _watcher(tmp_path)
        # Stop after the first iteration so the loop terminates.
        stop_states = iter([False, True])

        with (
            patch.object(w, "reconcile_once", side_effect=RuntimeError("unexpected")),
            patch("aim_runtime.engines.adapter_watcher.time.sleep"),
        ):
            # Should return cleanly despite reconcile_once raising.
            w.run_forever(lambda: next(stop_states))
