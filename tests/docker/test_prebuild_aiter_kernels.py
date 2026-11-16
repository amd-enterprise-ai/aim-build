# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "docker" / "prebuild_aiter_kernels.py"
SPEC = importlib.util.spec_from_file_location("prebuild_aiter_kernels", SCRIPT_PATH)
assert SPEC and SPEC.loader
prebuild = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prebuild
SPEC.loader.exec_module(prebuild)


def test_build_kernel_timeout_is_reported_as_failure(tmp_path, monkeypatch, capsys):
    def raise_timeout(*args, **kwargs):
        raise prebuild.subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(prebuild.subprocess, "run", raise_timeout)

    result = prebuild._build_kernel_in_subprocess(
        tmp_path,
        "kernel.so",
        "test",
        [],
        "cpu",
        1,
    )

    assert result is False
    assert "timed out after 1s" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["prebuild_aiter_kernels.py"], "--prebuild-device"),
        (["prebuild_aiter_kernels.py", "--prebuild-device", "cpu", "--kernel-timeout-seconds", "0"], "positive"),
    ],
)
def test_cli_validates_build_options(monkeypatch, argv, message, capsys):
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        prebuild.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cpu_runtime_shims_report_requested_architecture(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    original_path = list(sys.path)
    try:
        prebuild._install_cpu_runtime_shims(tmp_path)
        rocminfo = tmp_path / "bin" / "rocminfo"
        gpu_triton = tmp_path / "python" / "jax" / "_src" / "lib" / "gpu_triton.py"

        assert rocminfo.stat().st_mode & 0o111
        assert 'echo "  Name:                    ${arch}"' in rocminfo.read_text()
        assert "return f'{arch}:sramecc+:xnack-'" in gpu_triton.read_text()
    finally:
        sys.path[:] = original_path
