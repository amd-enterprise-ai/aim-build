# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the OpenFold3 per-worker GPU affinity helper.

These tests do **not** require torch, bentoml or OpenFold3 — the module under
test is pure stdlib and is loaded directly from the asset path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

OF3_GPU_AFFINITY_PATH = (
    Path(__file__).resolve().parents[2] / "assets/instinct/openfold/openfold3/image/src/gpu_affinity.py"
)


def _load_gpu_affinity():
    spec = importlib.util.spec_from_file_location("_of3_gpu_affinity_under_test", OF3_GPU_AFFINITY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_affinity = _load_gpu_affinity()


@pytest.fixture(autouse=True)
def _no_detected_gpus(monkeypatch):
    """Neutralise aim-runtime GPU detection.

    ``visible_device_count`` consults it first, so without this every test that
    reaches the unrestricted path would pass or fail depending on how many GPUs
    the test machine happens to have. Tests that care override it explicitly.
    """
    monkeypatch.setattr(gpu_affinity, "_detected_gpu_count", lambda: 0)


# --- current_visible_devices ------------------------------------------------


def test_visible_devices_unrestricted_when_unset():
    assert gpu_affinity.current_visible_devices({}) is None


def test_visible_devices_parses_comma_list():
    assert gpu_affinity.current_visible_devices({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}) == ["0", "1", "2", "3"]


def test_visible_devices_tolerates_whitespace():
    assert gpu_affinity.current_visible_devices({"CUDA_VISIBLE_DEVICES": "2, 3"}) == ["2", "3"]


def test_visible_devices_prefers_hip_over_cuda():
    env = {"HIP_VISIBLE_DEVICES": "4,5", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert gpu_affinity.current_visible_devices(env) == ["4", "5"]


def test_visible_devices_empty_value_means_no_devices():
    assert gpu_affinity.current_visible_devices({"HIP_VISIBLE_DEVICES": ""}) == []


# --- select_worker_device ---------------------------------------------------


@pytest.mark.parametrize("worker_count", [1, 2, 4, 8])
def test_each_worker_gets_a_distinct_device(worker_count):
    devices = [gpu_affinity.select_worker_device(None, index, worker_count) for index in range(1, worker_count + 1)]
    assert devices == [str(i) for i in range(worker_count)]
    assert len(set(devices)) == worker_count


def test_single_worker_uses_first_device():
    assert gpu_affinity.select_worker_device(None, 1, 1) == "0"


def test_worker_device_indexes_into_visible_list():
    visible = ["3", "5", "6", "7"]
    devices = [gpu_affinity.select_worker_device(visible, index, 4) for index in range(1, 5)]
    assert devices == visible


def test_extra_visible_devices_are_left_unused():
    assert gpu_affinity.select_worker_device(["4", "5", "6", "7"], 2, 2) == "5"


def test_fewer_visible_devices_than_workers_is_rejected():
    with pytest.raises(ValueError, match="expose 2"):
        gpu_affinity.select_worker_device(["0", "1"], 1, 4)


def test_no_visible_devices_is_rejected():
    with pytest.raises(ValueError, match="expose 0"):
        gpu_affinity.select_worker_device([], 1, 1)


def test_repeated_device_ids_are_rejected():
    """A repeated id passes the count check but would give two workers one GPU."""
    with pytest.raises(ValueError, match="repeat device ids"):
        gpu_affinity.select_worker_device(["0", "0", "1"], 1, 3)


def test_repeated_device_ids_are_allowed_when_enough_remain_distinct():
    assert gpu_affinity.select_worker_device(["4", "4", "5"], 1, 2) == "4"


def test_rejection_messages_name_the_setting_to_change():
    with pytest.raises(ValueError, match="AIM_ACCELERATOR_COUNT"):
        gpu_affinity.select_worker_device(["0"], 1, 2)
    with pytest.raises(ValueError, match="AIM_ACCELERATOR_COUNT"):
        gpu_affinity.select_worker_device(None, 1, 2, device_count=1)


# --- device_count guard on the unrestricted path ----------------------------


def test_unrestricted_path_rejects_more_workers_than_devices():
    with pytest.raises(ValueError, match="this container has 2"):
        gpu_affinity.select_worker_device(None, 1, 8, device_count=2)


def test_unrestricted_path_accepts_exactly_enough_devices():
    devices = [gpu_affinity.select_worker_device(None, i, 4, device_count=4) for i in range(1, 5)]
    assert devices == ["0", "1", "2", "3"]


def test_unrestricted_path_allows_spare_devices():
    assert gpu_affinity.select_worker_device(None, 2, 2, device_count=8) == "1"


def test_unknown_device_count_does_not_block_startup():
    """None means undetectable — it must not fail an otherwise valid deployment."""
    assert gpu_affinity.select_worker_device(None, 8, 8, device_count=None) == "7"


@pytest.mark.parametrize("worker_index", [0, -1, 3])
def test_worker_index_outside_the_worker_range_is_rejected(worker_index):
    with pytest.raises(ValueError, match="out of range"):
        gpu_affinity.select_worker_device(None, worker_index, 2)


def test_worker_count_below_one_is_rejected():
    with pytest.raises(ValueError, match="worker_count must be >= 1"):
        gpu_affinity.select_worker_device(None, 1, 0)


# --- pin_worker_device ------------------------------------------------------

# Tests that reach the unrestricted path must not consult the host's real
# /dev/dri, or they would pass or fail depending on the machine's GPU count.
NO_DEV_DRI = Path("/nonexistent-dev-dri-for-tests")


def test_pinning_narrows_both_visibility_vars_to_one_device():
    env: dict[str, str] = {}
    assert gpu_affinity.pin_worker_device(3, 4, env, dev_dri=NO_DEV_DRI) == "2"
    assert env == {"HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "2"}


def test_pinning_rejects_more_workers_than_the_container_has_devices(tmp_path):
    (tmp_path / "renderD128").touch()
    with pytest.raises(ValueError, match="this container has 1"):
        gpu_affinity.pin_worker_device(1, 4, {}, dev_dri=tmp_path)


def test_pinning_narrows_an_already_restricted_process():
    env = {"HIP_VISIBLE_DEVICES": "4,5,6,7", "CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    assert gpu_affinity.pin_worker_device(2, 4, env) == "5"
    assert env == {"HIP_VISIBLE_DEVICES": "5", "CUDA_VISIBLE_DEVICES": "5"}


def test_pinning_is_idempotent_for_a_single_worker():
    env = {"HIP_VISIBLE_DEVICES": "6", "CUDA_VISIBLE_DEVICES": "6"}
    assert gpu_affinity.pin_worker_device(1, 1, env) == "6"
    assert gpu_affinity.pin_worker_device(1, 1, env) == "6"
    assert env == {"HIP_VISIBLE_DEVICES": "6", "CUDA_VISIBLE_DEVICES": "6"}


def test_pinning_does_not_touch_the_real_process_environment():
    import os

    before = dict(os.environ)
    gpu_affinity.pin_worker_device(1, 2, {}, dev_dri=NO_DEV_DRI)
    assert dict(os.environ) == before


def test_pinning_overwrites_a_disagreeing_cuda_variable():
    """HIP wins the read; CUDA must be rewritten so the two cannot disagree."""
    env = {"HIP_VISIBLE_DEVICES": "4,5", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert gpu_affinity.pin_worker_device(2, 2, env) == "5"
    assert env == {"HIP_VISIBLE_DEVICES": "5", "CUDA_VISIBLE_DEVICES": "5"}


def test_pinning_rejects_an_empty_visibility_variable():
    with pytest.raises(ValueError, match="expose 0"):
        gpu_affinity.pin_worker_device(1, 1, {"HIP_VISIBLE_DEVICES": ""})


# --- visible_device_count ---------------------------------------------------


def test_device_count_counts_render_nodes(tmp_path):
    for name in ("renderD128", "renderD129", "card0", "card1", "by-path"):
        (tmp_path / name).touch()
    assert gpu_affinity.visible_device_count(tmp_path) == 2


def test_device_count_is_unknown_when_directory_is_absent(tmp_path):
    assert gpu_affinity.visible_device_count(tmp_path / "nope") is None


def test_device_count_is_unknown_rather_than_zero_when_no_nodes_present(tmp_path):
    """Zero would fail every deployment; unknown leaves behaviour unchanged."""
    (tmp_path / "card0").touch()
    assert gpu_affinity.visible_device_count(tmp_path) is None


def test_device_count_prefers_aim_runtime_detection(monkeypatch, tmp_path):
    """The repo's detector validates against sysfs, so it wins over raw node counting."""
    monkeypatch.setattr(gpu_affinity, "_detected_gpu_count", lambda: 4)
    for name in ("renderD128", "renderD129"):
        (tmp_path / name).touch()
    assert gpu_affinity.visible_device_count(tmp_path) == 4


def test_device_count_falls_back_when_detection_reports_zero(monkeypatch, tmp_path):
    """gpu_count == 0 is ambiguous — it also means 'sysfs said nothing'."""
    monkeypatch.setattr(gpu_affinity, "_detected_gpu_count", lambda: 0)
    (tmp_path / "renderD128").touch()
    assert gpu_affinity.visible_device_count(tmp_path) == 1


def test_detection_is_skipped_when_aim_runtime_is_absent(monkeypatch):
    """Running bentoml directly, without aim-runtime on PYTHONPATH, must not crash."""
    monkeypatch.setitem(sys.modules, "aim_runtime.gpu_detector", None)
    # Fresh load: the autouse fixture has stubbed the function on the shared module.
    assert _load_gpu_affinity()._detected_gpu_count() == 0
