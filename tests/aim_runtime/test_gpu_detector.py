# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys
from types import ModuleType

from aim_runtime.gpu_detector import GPUDetector


def test_model_mapping_and_normalization():
    d = GPUDetector()
    assert d._normalize_device_id("74a1") == "0x74a1"
    assert d._normalize_device_id("0x74a1") == "0x74a1"
    assert d.get_gpu_model("0x74a1") == "MI300X"
    assert d.get_gpu_model("0x7000") == "UNKNOWN"


def test_library_path_idle(monkeypatch):
    # Fake amdsmi module API
    fake_handles = [object(), object()]

    def amdsmi_init():
        pass

    def amdsmi_shut_down():
        pass

    def amdsmi_get_processor_handles():
        return fake_handles

    def amdsmi_get_gpu_asic_info(h):
        return {"device_id": "74a1" if h is fake_handles[0] else "0x74a2"}

    def amdsmi_get_gpu_vram_usage(h):
        return {
            "vram_total": 16384 if h is fake_handles[0] else 8192,
            "vram_used": 6144 if h is fake_handles[0] else 2048,
        }

    def amdsmi_get_gpu_activity(h):
        return {"gfx_activity": 0, "umc_activity": 0}

    fake_mod = ModuleType("amdsmi")
    fake_mod.amdsmi_init = amdsmi_init
    fake_mod.amdsmi_get_gpu_asic_info = amdsmi_get_gpu_asic_info
    fake_mod.amdsmi_get_gpu_vram_usage = amdsmi_get_gpu_vram_usage
    fake_mod.amdsmi_get_gpu_activity = amdsmi_get_gpu_activity
    fake_mod.amdsmi_get_processor_handles = amdsmi_get_processor_handles
    fake_mod.amdsmi_shut_down = amdsmi_shut_down
    monkeypatch.setitem(sys.modules, "amdsmi", fake_mod)

    d = GPUDetector()
    assert d.has_gpus is True
    assert d.gpu_count == 2
    assert d.device_ids == ["0x74a1", "0x74a2"]
    # (16384-6144) + (8192-2048) = 10240 + 6144 = 16384
    assert d.total_free_vram == 16384
    assert d.gpu_models == ["MI300X", "MI308X"]
    assert d.all_gpus_idle is True


def test_library_path_busy_sets_idle_false(monkeypatch):
    handles = [object()]

    def amdsmi_init():
        pass

    def amdsmi_shut_down():
        pass

    def amdsmi_get_processor_handles():
        return handles

    def amdsmi_get_gpu_asic_info(h):
        return {"device_id": "0x74a1"}

    def amdsmi_get_gpu_vram_usage(h):
        return {"vram_total": 1024, "vram_used": 24}

    def amdsmi_get_gpu_activity(h):
        return {"gfx_activity": 1, "umc_activity": 0}

    fake_mod = ModuleType("amdsmi")
    fake_mod.amdsmi_init = amdsmi_init
    fake_mod.amdsmi_get_gpu_asic_info = amdsmi_get_gpu_asic_info
    fake_mod.amdsmi_get_gpu_vram_usage = amdsmi_get_gpu_vram_usage
    fake_mod.amdsmi_get_gpu_activity = amdsmi_get_gpu_activity
    fake_mod.amdsmi_get_processor_handles = amdsmi_get_processor_handles
    fake_mod.amdsmi_shut_down = amdsmi_shut_down
    monkeypatch.setitem(sys.modules, "amdsmi", fake_mod)

    d = GPUDetector()
    assert d.has_gpus is True
    assert d.all_gpus_idle is False


def test_no_gpus_when_library_returns_none(monkeypatch):
    # Force library path to return None (no GPUs)
    monkeypatch.setattr(GPUDetector, "_get_gpu_info", lambda self: None)

    d = GPUDetector()
    assert d.has_gpus is False
    assert d.device_ids is None
    assert d.total_free_vram is None
    assert d.gpu_models is None
    assert d.get_gpu_info() is None
    assert d.all_gpus_idle is True


def test_library_empty_handles(monkeypatch):
    # Fake amdsmi returning empty handles
    fake_mod = ModuleType("amdsmi")
    fake_mod.amdsmi_init = lambda: None
    fake_mod.amdsmi_shut_down = lambda: None
    fake_mod.amdsmi_get_processor_handles = lambda: []
    fake_mod.amdsmi_get_gpu_asic_info = lambda h: {}
    fake_mod.amdsmi_get_gpu_vram_usage = lambda h: {}
    fake_mod.amdsmi_get_gpu_activity = lambda h: {}
    monkeypatch.setitem(sys.modules, "amdsmi", fake_mod)

    # Library returns empty handles; should report no GPUs

    d = GPUDetector()
    assert d.has_gpus is False


def test_library_importerror(monkeypatch):
    # Ensure amdsmi is not importable — detector should report no GPUs
    monkeypatch.delitem(sys.modules, "amdsmi", raising=False)

    d = GPUDetector()
    assert d.has_gpus is False


def test_library_generic_exception(monkeypatch):
    # Fake amdsmi where amdsmi_init raises
    class FakeErrMod(ModuleType):
        pass

    fake_mod = FakeErrMod("amdsmi")

    def boom():
        raise RuntimeError("init fail")

    fake_mod.amdsmi_init = boom
    fake_mod.amdsmi_shut_down = lambda: None
    fake_mod.amdsmi_get_processor_handles = lambda: [object()]
    fake_mod.amdsmi_get_gpu_asic_info = lambda h: {"device_id": "74a1"}
    fake_mod.amdsmi_get_gpu_vram_usage = lambda h: {"vram_total": 1, "vram_used": 0}
    fake_mod.amdsmi_get_gpu_activity = lambda h: {"gfx_activity": 0, "umc_activity": 0}
    monkeypatch.setitem(sys.modules, "amdsmi", fake_mod)

    d = GPUDetector()
    assert d.has_gpus is False
