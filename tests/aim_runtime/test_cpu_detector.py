# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for EpycDetector including override_cpu_env_vars.

Covers:
- EPYC model extraction from /proc/cpuinfo model name strings
- CPU model resolution (known, unknown, non-EPYC AMD, non-AMD)
- Zen5 / Zen4 fallback for chips not individually enumerated
- cgroup v2 / v1 quota parsing (whole cores, fractional, unlimited, missing)
- cpuset parsing and bind-string propagation
- override_cpu_env_vars validation and binding behaviour
"""
import sys

import pytest

from aim_common.object_model import CPUModel
from aim_runtime.cpu_detector import EpycDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPYC_9965_CPUINFO = "AMD EPYC 9965 192-Core Processor"
EPYC_9755_CPUINFO = "AMD EPYC 9755 128-Core Processor"
EPYC_9755P_CPUINFO = "AMD EPYC 9755P 128-Core Processor"
RYZEN_CPUINFO = "AMD Ryzen 9 9950X 16-Core Processor"
INTEL_CPUINFO = "Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz"


# ---------------------------------------------------------------------------
# EPYC model extraction
# ---------------------------------------------------------------------------


def test_extract_model_number_standard():
    assert EpycDetector._extract_model_number(EPYC_9965_CPUINFO) == "9965"


def test_extract_model_number_no_match_ryzen():
    assert EpycDetector._extract_model_number(RYZEN_CPUINFO) is None


def test_extract_model_number_no_match_intel():
    assert EpycDetector._extract_model_number(INTEL_CPUINFO) is None


# ---------------------------------------------------------------------------
# CPU model resolution
# ---------------------------------------------------------------------------


def test_resolve_recognized_epyc_9965():
    assert EpycDetector._resolve_cpu_model("9965", "AuthenticAMD") == CPUModel.EPYC_9965


def test_resolve_unrecognized_epyc_returns_unknown(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="aim_runtime.cpu_detector"):
        result = EpycDetector._resolve_cpu_model("9999", "AuthenticAMD")

    assert result is None
    assert "9999" in caplog.text


def test_resolve_non_epyc_amd_returns_none(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="aim_runtime.cpu_detector"):
        result = EpycDetector._resolve_cpu_model(None, "AuthenticAMD")

    assert result is None
    assert "EPYC" in caplog.text


def test_resolve_non_amd_returns_none():
    assert EpycDetector._resolve_cpu_model(None, "GenuineIntel") is None


# ---------------------------------------------------------------------------
# CPU model resolution — Zen5 fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chip_id",
    [
        "9845",
        "9825",
        "9755",  # TODO: Update this to EPYC_9755 when it is available
        "9745",
        "9655P",
        "9655",
        "9645",
        "9575F",
        "9565",
        "9555P",
        "9555",
        "9375F",
        "9365",
        "9355P",
        "9355",
        "9335",
        "9275F",
        "9255",
        "9175F",
        "9135",
        "9115",
        "9015",
    ],
)
def test_resolve_zen5_fallback_chips(chip_id):
    """Zen5 EPYC chips not individually enumerated fall back to EPYC_ZEN5."""
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN5


# ---------------------------------------------------------------------------
# CPU model resolution — Zen4 fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chip_id",
    [
        "9754",
        "9754S",
        "9734",
        "9654",
        "9654P",
        "9634",
        "9554",
        "9554P",
        "9534",
        "9454",
        "9454P",
        "9354",
        "9354P",
        "9334",
        "9254",
        "9224",
        "9124",
        # 3D V-Cache variants
        "9684X",
        "9384X",
        "9184X",
        # High-frequency variants
        "9474F",
        "9374F",
        "9274F",
        "9174F",
        # Custom OEM variants
        "9J14",
    ],
)
def test_resolve_zen4_fallback_chips(chip_id):
    """Zen4 EPYC 9004-series chips fall back to EPYC_ZEN4."""
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN4


# ---------------------------------------------------------------------------
# End-to-end extraction + resolution for fallback chips
# ---------------------------------------------------------------------------


def test_end_to_end_zen5_fallback():
    """Full pipeline: cpuinfo string -> extract -> resolve for a Zen5 fallback chip."""
    model_name = "AMD EPYC 9845 96-Core Processor"
    chip_id = EpycDetector._extract_model_number(model_name)
    assert chip_id == "9845"
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN5


def test_end_to_end_zen5_high_freq_fallback():
    """Full pipeline for a Zen5 high-frequency chip (alpha suffix in model number)."""
    model_name = "AMD EPYC 9575F 64-Core Processor"
    chip_id = EpycDetector._extract_model_number(model_name)
    assert chip_id == "9575F"
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN5


def test_end_to_end_zen4_fallback():
    """Full pipeline: cpuinfo string -> extract -> resolve for a Zen4 fallback chip."""
    model_name = "AMD EPYC 9654 96-Core Processor"
    chip_id = EpycDetector._extract_model_number(model_name)
    assert chip_id == "9654"
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN4


def test_end_to_end_zen4_vcache_fallback():
    """Full pipeline for a Zen4 3D V-Cache chip (alpha suffix in model number)."""
    model_name = "AMD EPYC 9684X 96-Core Processor"
    chip_id = EpycDetector._extract_model_number(model_name)
    assert chip_id == "9684X"
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN4


def test_end_to_end_zen4_high_freq_fallback():
    """Full pipeline for a Zen4 high-frequency chip (alpha suffix in model number)."""
    model_name = "AMD EPYC 9474F 48-Core Processor"
    chip_id = EpycDetector._extract_model_number(model_name)
    assert chip_id == "9474F"
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN4


def test_end_to_end_zen4_custom_oem_alphanumeric_sku():
    """Full pipeline for a custom OEM Zen4 SKU with an alphanumeric model number."""
    model_name = "AMD EPYC 9J14 96-Core Processor"
    chip_id = EpycDetector._extract_model_number(model_name)
    assert chip_id == "9J14"
    assert EpycDetector._resolve_cpu_model(chip_id, "AuthenticAMD") == CPUModel.EPYC_ZEN4


# ---------------------------------------------------------------------------
# Fallback chips are distinct from specifically-enumerated chips
# ---------------------------------------------------------------------------


def test_specific_chip_not_zen5_fallback():
    """EPYC 9965 has its own enum member, not the ZEN5 fallback."""
    assert EpycDetector._resolve_cpu_model("9965", "AuthenticAMD") == CPUModel.EPYC_9965
    assert EpycDetector._resolve_cpu_model("9965", "AuthenticAMD") != CPUModel.EPYC_ZEN5


# TODO: Add test for zen4 fallback

# ---------------------------------------------------------------------------
# cgroup v2 quota
# ---------------------------------------------------------------------------


def test_cgroup_v2_whole_cores(tmp_path, monkeypatch):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000")
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(cpu_max))

    assert EpycDetector._read_cgroup_v2_quota() == 2


def test_cgroup_v2_fractional_cores_warns(tmp_path, monkeypatch, caplog):
    import logging

    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("150000 100000")
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(cpu_max))

    with caplog.at_level(logging.WARNING, logger="aim_runtime.cpu_detector"):
        result = EpycDetector._read_cgroup_v2_quota()

    assert result == 1  # floor division, conservative
    assert "Fractional" in caplog.text


def test_cgroup_v2_unlimited(tmp_path, monkeypatch):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000")
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(cpu_max))

    assert EpycDetector._read_cgroup_v2_quota() is None


def test_cgroup_v2_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(tmp_path / "nonexistent"))

    assert EpycDetector._read_cgroup_v2_quota() is None


# ---------------------------------------------------------------------------
# cgroup v1 quota
# ---------------------------------------------------------------------------


def test_cgroup_v1_whole_cores(tmp_path, monkeypatch):
    quota = tmp_path / "quota"
    period = tmp_path / "period"
    quota.write_text("200000")
    period.write_text("100000")
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(quota))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_PERIOD", str(period))

    assert EpycDetector._read_cgroup_v1_quota() == 2


def test_cgroup_v1_fractional_cores_warns(tmp_path, monkeypatch, caplog):
    import logging

    quota = tmp_path / "quota"
    period = tmp_path / "period"
    quota.write_text("150000")
    period.write_text("100000")
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(quota))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_PERIOD", str(period))

    with caplog.at_level(logging.WARNING, logger="aim_runtime.cpu_detector"):
        result = EpycDetector._read_cgroup_v1_quota()

    assert result == 1  # floor division, conservative
    assert "Fractional" in caplog.text


def test_cgroup_v1_disabled(tmp_path, monkeypatch):
    """quota of -1 means no limit (cgroup v1 convention)."""
    quota = tmp_path / "quota"
    period = tmp_path / "period"
    quota.write_text("-1")
    period.write_text("100000")
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(quota))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_PERIOD", str(period))

    assert EpycDetector._read_cgroup_v1_quota() is None


def test_cgroup_v1_files_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(tmp_path / "nonexistent"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_PERIOD", str(tmp_path / "nonexistent"))

    assert EpycDetector._read_cgroup_v1_quota() is None


# ---------------------------------------------------------------------------
# cpuset parsing
# ---------------------------------------------------------------------------


def test_parse_cpuset_simple_range():
    assert len(EpycDetector._parse_cpuset_to_set("0-7")) == 8


def test_parse_cpuset_single_cpu():
    assert len(EpycDetector._parse_cpuset_to_set("0")) == 1


def test_parse_cpuset_comma_list():
    assert len(EpycDetector._parse_cpuset_to_set("0,2,4")) == 3


def test_parse_cpuset_complex_non_contiguous():
    """NUMA-style non-contiguous assignment e.g. two half-sockets."""
    assert len(EpycDetector._parse_cpuset_to_set("0-3,5,7-9")) == 8


def test_parse_cpuset_empty_string():
    assert len(EpycDetector._parse_cpuset_to_set("")) == 0


# ---------------------------------------------------------------------------
# cpuset_bind propagated from cpuset file
# ---------------------------------------------------------------------------


def test_cpuset_bind_returned_when_detected_from_cpuset_file(tmp_path, monkeypatch):
    """When cores come from a cpuset file, the raw string is returned for explicit binding."""
    cpuset_file = tmp_path / "cpuset.cpus.effective"
    cpuset_file.write_text("0-127")
    monkeypatch.setattr("aim_runtime.cpu_detector.CPUSET_V2_EFFECTIVE", str(cpuset_file))
    # Disable quota paths so cpuset is reached
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(tmp_path / "nope"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(tmp_path / "nope"))

    result = EpycDetector._read_cpuset()
    assert result is not None
    cores, bind_str = result
    assert cores == 128
    assert bind_str == "0-127"


def test_cpuset_bind_non_contiguous_preserved(tmp_path, monkeypatch):
    """Non-contiguous NUMA cpuset string is preserved verbatim for vLLM."""
    cpuset_file = tmp_path / "cpuset.cpus.effective"
    cpuset_file.write_text("16-47,64-95")
    monkeypatch.setattr("aim_runtime.cpu_detector.CPUSET_V2_EFFECTIVE", str(cpuset_file))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(tmp_path / "nope"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(tmp_path / "nope"))

    result = EpycDetector._read_cpuset()
    assert result is not None
    cores, bind_str = result
    assert cores == 64
    assert bind_str == "16-47,64-95"


def test_cpuset_bind_none_when_detected_from_cgroup_quota(tmp_path, monkeypatch):
    """When cores come from cgroup quota, CPU IDs are unknown — bind is None."""
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("12800000 100000")  # 128 whole cores
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(cpu_max))

    cores, bind_str = EpycDetector._detect_available_cores(physical_cores=192)
    assert cores == 128
    assert bind_str is None


@pytest.mark.skipif(sys.platform == "darwin", reason="sched_getaffinity not available on macOS")
def test_cpuset_bind_none_when_detected_from_affinity(tmp_path, monkeypatch):
    """sched_getaffinity gives a count but not specific IDs — bind is None."""
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V2_CPU_MAX", str(tmp_path / "nope"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CGROUP_V1_QUOTA", str(tmp_path / "nope"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CPUSET_V2_EFFECTIVE", str(tmp_path / "nope"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CPUSET_V2", str(tmp_path / "nope"))
    monkeypatch.setattr("aim_runtime.cpu_detector.CPUSET_V1", str(tmp_path / "nope"))
    monkeypatch.setattr("os.sched_getaffinity", lambda _: set(range(64)))

    cores, bind_str = EpycDetector._detect_available_cores(physical_cores=192)
    assert cores == 64
    assert bind_str is None


# ---------------------------------------------------------------------------
# override_cpu_env_vars — core count validation
# ---------------------------------------------------------------------------


def test_insufficient_cores_scales_down(caplog):
    """Profile requires 192 cores but only 128 detected — scales down and warns."""
    import logging

    env_vars = {"OMP_NUM_THREADS": "192"}

    with caplog.at_level(logging.WARNING, logger="aim_runtime.cpu_detector"):
        EpycDetector.override_cpu_env_vars(env_vars, detected_cores=128)

    assert env_vars["OMP_NUM_THREADS"] == "128"
    assert "Scaling down" in caplog.text


def test_sufficient_cores_exact_match_passes():
    env_vars = {"OMP_NUM_THREADS": "128"}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=128)  # no exception


def test_sufficient_cores_excess_passes():
    """More cores than required is fine."""
    env_vars = {"OMP_NUM_THREADS": "128"}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=192)  # no exception


def test_no_omp_threads_in_profile_skips_validation():
    """Profile without OMP_NUM_THREADS should not raise even with 1 core."""
    env_vars = {}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=1)  # no exception


def test_unparseable_omp_threads_warns_not_raises(caplog):
    import logging

    env_vars = {"OMP_NUM_THREADS": "auto"}
    with caplog.at_level(logging.WARNING, logger="aim_runtime.cpu_detector"):
        EpycDetector.override_cpu_env_vars(env_vars, detected_cores=128)

    assert "Could not parse" in caplog.text


# ---------------------------------------------------------------------------
# override_cpu_env_vars — binding behaviour
# ---------------------------------------------------------------------------


def test_no_bind_when_cpuset_unknown():
    """When CPU IDs are not known (cgroup quota path), use 'auto' binding."""
    env_vars = {"OMP_NUM_THREADS": "128"}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=128, cpuset_bind=None)

    assert env_vars["VLLM_CPU_OMP_THREADS_BIND"] == "auto"


def test_auto_bind_when_cpuset_known_but_profile_unset():
    """When profile has no VLLM_CPU_OMP_THREADS_BIND, default to 'auto' even if cpuset is known."""
    env_vars = {"OMP_NUM_THREADS": "128"}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=128, cpuset_bind="0-127")

    assert env_vars["VLLM_CPU_OMP_THREADS_BIND"] == "auto"


def test_auto_bind_when_non_contiguous_cpuset_but_profile_unset():
    """When profile has no VLLM_CPU_OMP_THREADS_BIND, default to 'auto' even with non-contiguous cpuset."""
    env_vars = {"OMP_NUM_THREADS": "64"}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=64, cpuset_bind="16-47,64-95")

    assert env_vars["VLLM_CPU_OMP_THREADS_BIND"] == "auto"


def test_reserved_cpu_not_injected():
    """VLLM_CPU_NUM_OF_RESERVED_CPU should not be injected by the override.

    vLLM interprets this as cores to *exclude* from inference, so setting it
    equal to detected_cores would leave zero cores for work.  Let the profile
    or vLLM defaults control this value.
    """
    env_vars = {"OMP_NUM_THREADS": "128"}
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=128, cpuset_bind=None)

    assert "VLLM_CPU_NUM_OF_RESERVED_CPU" not in env_vars


def test_omp_num_threads_overridden_to_detected():
    """OMP_NUM_THREADS from profile is always replaced with detected count."""
    env_vars = {"OMP_NUM_THREADS": "192"}  # profile value (we have more than needed)
    EpycDetector.override_cpu_env_vars(env_vars, detected_cores=192)

    assert env_vars["OMP_NUM_THREADS"] == "192"
