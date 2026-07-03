# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for AcceleratorDetector unified hardware detection."""

from unittest.mock import MagicMock, patch

from aim_common import AcceleratorFamily
from aim_runtime.accelerator_detector import AcceleratorDetectionResult, AcceleratorDetector
from aim_runtime.cpu_detector import CPUInfo
from aim_runtime.gpu_detector import GPUInfo
from aim_runtime.object_model import AcceleratorModel, AcceleratorType

# ---------------------------------------------------------------------------
# GPU detection tests
# ---------------------------------------------------------------------------


class TestGPUDetection:
    """Tests for GPU detection via AcceleratorDetector."""

    def test_gpu_auto_detection_with_gpus(self):
        """Test that GPUDetector results are correctly propagated."""
        mock_gpu = GPUInfo(
            device_id="0x74a1",
            model=AcceleratorModel.MI300X,
            vram_total=65536,
            vram_used=0,
            gfx_utilization=0,
            mem_utilization=0,
        )

        with patch("aim_runtime.accelerator_detector.GPUDetector") as mock_cls:
            instance = mock_cls.return_value
            instance.all_gpus_idle = True
            instance.has_gpus = True
            instance.gpu_models = [AcceleratorModel.MI300X]
            instance.gpu_count = 2
            instance.gpus = [mock_gpu, mock_gpu]

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.GPU)

        assert result.accelerator_type == AcceleratorType.GPU
        assert result.accelerator_model == AcceleratorModel.MI300X
        assert result.accelerator_count == 2
        assert len(result.gpu_info) == 2
        assert result.cpu_info is None
        assert result.cpu_cores == 0

    def test_gpu_auto_detection_no_gpus(self):
        """Test result when no GPUs are detected."""
        with patch("aim_runtime.accelerator_detector.GPUDetector") as mock_cls:
            instance = mock_cls.return_value
            instance.all_gpus_idle = True
            instance.has_gpus = False
            instance.gpu_models = []

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.GPU)

        assert result.accelerator_type == AcceleratorType.GPU
        assert result.accelerator_model is None
        assert result.accelerator_count == 0
        assert result.gpu_info == []

    def test_gpu_auto_detection_with_count_override(self):
        """Test that accelerator_count_override overrides auto-detected GPU count."""
        with patch("aim_runtime.accelerator_detector.GPUDetector") as mock_cls:
            instance = mock_cls.return_value
            instance.all_gpus_idle = True
            instance.has_gpus = True
            instance.gpu_models = [AcceleratorModel.MI300X]
            instance.gpu_count = 4
            instance.gpus = [MagicMock(spec=GPUInfo)] * 4

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.GPU, accelerator_count_override=2)

        assert result.accelerator_model == AcceleratorModel.MI300X
        assert result.accelerator_count == 2

    def test_gpu_model_override_with_explicit_count(self):
        """Test that model override skips GPUDetector entirely."""
        with patch("aim_runtime.accelerator_detector.GPUDetector") as mock_cls:
            detector = AcceleratorDetector()
            result = detector.detect(
                accelerator_type=AcceleratorType.GPU,
                accelerator_model_override=AcceleratorModel.MI325X,
                accelerator_count_override=4,
            )

        # GPUDetector should not be instantiated
        mock_cls.assert_not_called()
        assert result.accelerator_type == AcceleratorType.GPU
        assert result.accelerator_model == AcceleratorModel.MI325X
        assert result.accelerator_count == 4
        assert result.gpu_info == []

    def test_gpu_model_override_with_auto_count_defaults_to_1(self):
        """Test that model override with auto count defaults to 1."""
        with patch("aim_runtime.accelerator_detector.GPUDetector") as mock_cls:
            detector = AcceleratorDetector()
            result = detector.detect(
                accelerator_type=AcceleratorType.GPU,
                accelerator_model_override=AcceleratorModel.MI300A,
                accelerator_count_override="auto",
            )

        mock_cls.assert_not_called()
        assert result.accelerator_model == AcceleratorModel.MI300A
        assert result.accelerator_count == 1

    def test_gpu_detection_warns_on_busy_gpus(self):
        """Test that a warning is logged when GPUs are not idle."""
        with patch("aim_runtime.accelerator_detector.GPUDetector") as mock_cls:
            instance = mock_cls.return_value
            instance.all_gpus_idle = False
            instance.has_gpus = True
            instance.gpu_models = [AcceleratorModel.MI300X]
            instance.gpu_count = 1
            instance.gpus = [MagicMock(spec=GPUInfo)]

            detector = AcceleratorDetector()
            with patch("aim_runtime.accelerator_detector.logger") as mock_logger:
                result = detector.detect(accelerator_type=AcceleratorType.GPU)
                mock_logger.warning.assert_any_call("Some GPUs are not idle! Check GPU usage.")

        assert result.accelerator_model == AcceleratorModel.MI300X


# ---------------------------------------------------------------------------
# CPU detection tests
# ---------------------------------------------------------------------------


class TestCPUDetection:
    """Tests for CPU detection via AcceleratorDetector."""

    def test_cpu_detection_reports_node_capacity_with_cpuset(self):
        """accelerator_count reflects node capacity; cpu_cores stays cpuset-limited.

        This is the detect-hardware fix: under an 8-core cpuset on a 384-core node,
        labelling reports 384 while serve/dry-run use the 8 available cores.
        """
        mock_cpu_info = CPUInfo(
            vendor="AMD",
            model=AcceleratorModel.EPYC_9965,
            model_number="9965",
            physical_cores=384,
            available_cores=8,
            cpuset_bind="0-7",
            node_cores=384,
        )

        with patch("aim_runtime.accelerator_detector.EpycDetector") as mock_cls:
            instance = mock_cls.return_value
            instance.cpu_info = mock_cpu_info
            instance.available_cores = 8
            instance.node_cores = 384
            instance.cpu_model = AcceleratorModel.EPYC_9965
            instance.cpuset_bind = "0-7"

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.CPU, accelerator_family=AcceleratorFamily.EPYC)

        assert result.accelerator_type == AcceleratorType.CPU
        assert result.accelerator_model == AcceleratorModel.EPYC_9965
        # Node capacity is reported for labelling, not the cpuset-limited count.
        assert result.accelerator_count == 384
        assert result.cpu_info == mock_cpu_info
        assert result.cpu_cores == 8
        assert result.cpuset_bind == "0-7"
        assert result.gpu_info == []

    def test_cpu_detection_without_cpuset(self):
        """Test CPU detection when cpuset is not available."""
        mock_cpu_info = CPUInfo(
            vendor="AMD",
            model=AcceleratorModel.EPYC_ZEN5,
            model_number="9005",
            physical_cores=128,
            available_cores=128,
            cpuset_bind=None,
            node_cores=128,
        )

        with patch("aim_runtime.accelerator_detector.EpycDetector") as mock_cls:
            instance = mock_cls.return_value
            instance.cpu_info = mock_cpu_info
            instance.available_cores = 128
            instance.node_cores = 128
            instance.cpu_model = AcceleratorModel.EPYC_ZEN5
            instance.cpuset_bind = None

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.CPU, accelerator_family=AcceleratorFamily.EPYC)

        assert result.accelerator_count == 128
        assert result.cpu_cores == 128
        assert result.cpuset_bind is None


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestDetectionDispatch:
    """Tests for accelerator_type dispatch logic."""

    def test_gpu_type_dispatches_to_gpu_detection(self):
        """Test that accelerator_type='gpu' triggers GPU detection."""
        with (
            patch("aim_runtime.accelerator_detector.GPUDetector") as mock_gpu,
            patch("aim_runtime.accelerator_detector.EpycDetector") as mock_cpu,
        ):
            mock_gpu.return_value.all_gpus_idle = True
            mock_gpu.return_value.has_gpus = False
            mock_gpu.return_value.gpu_models = []

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.GPU)

        assert result.accelerator_type == AcceleratorType.GPU
        mock_gpu.assert_called_once()
        mock_cpu.assert_not_called()

    def test_cpu_type_dispatches_to_cpu_detection(self):
        """Test that accelerator_type='cpu' triggers CPU detection."""
        with (
            patch("aim_runtime.accelerator_detector.GPUDetector") as mock_gpu,
            patch("aim_runtime.accelerator_detector.EpycDetector") as mock_cpu,
        ):
            mock_cpu.return_value.cpu_info = None
            mock_cpu.return_value.available_cores = 8
            mock_cpu.return_value.node_cores = 8
            mock_cpu.return_value.cpu_model = None
            mock_cpu.return_value.cpuset_bind = None

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.CPU, accelerator_family=AcceleratorFamily.EPYC)

        assert result.accelerator_type == AcceleratorType.CPU
        assert result.accelerator_model == AcceleratorModel.CPU
        assert result.accelerator_count == 8
        mock_cpu.assert_called_once()
        mock_gpu.assert_not_called()

    def test_cpu_type_defaults_to_generic_cpu_detector_when_family_omitted(self):
        """CPU detection without family uses generic CpuDetector default path."""
        with (
            patch("aim_runtime.accelerator_detector.EpycDetector") as mock_epyc,
            patch("aim_runtime.accelerator_detector.CpuDetector") as mock_cpu,
        ):
            mock_cpu.return_value.cpu_info = None
            mock_cpu.return_value.available_cores = 16
            mock_cpu.return_value.node_cores = 16
            mock_cpu.return_value.cpu_model = None
            mock_cpu.return_value.cpuset_bind = None

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.CPU)

        assert result.accelerator_type == AcceleratorType.CPU
        assert result.accelerator_model == AcceleratorModel.CPU
        mock_cpu.assert_called_once()
        mock_epyc.assert_not_called()

    def test_cpu_type_with_cpu_family_uses_generic_cpu_detector(self):
        """Explicit CPU family continues to use generic CpuDetector."""
        with (
            patch("aim_runtime.accelerator_detector.EpycDetector") as mock_epyc,
            patch("aim_runtime.accelerator_detector.CpuDetector") as mock_cpu,
        ):
            mock_cpu.return_value.cpu_info = None
            mock_cpu.return_value.available_cores = 8
            mock_cpu.return_value.node_cores = 8
            mock_cpu.return_value.cpu_model = None
            mock_cpu.return_value.cpuset_bind = None

            detector = AcceleratorDetector()
            result = detector.detect(accelerator_type=AcceleratorType.CPU, accelerator_family=AcceleratorFamily.CPU)

        assert result.accelerator_type == AcceleratorType.CPU
        assert result.accelerator_model == AcceleratorModel.CPU
        mock_cpu.assert_called_once()
        mock_epyc.assert_not_called()

    def test_cpu_type_ignores_model_override(self):
        """Test that accelerator_type='cpu' ignores accelerator_model_override."""
        with patch("aim_runtime.accelerator_detector.EpycDetector") as mock_cls:
            mock_cls.return_value.cpu_info = None
            mock_cls.return_value.available_cores = 16
            mock_cls.return_value.node_cores = 16
            mock_cls.return_value.cpu_model = None
            mock_cls.return_value.cpuset_bind = None

            detector = AcceleratorDetector()
            result = detector.detect(
                accelerator_type=AcceleratorType.CPU,
                accelerator_model_override=AcceleratorModel.MI300X,
            )

        # CPU detection should run regardless of model override, falls back to CPU sentinel
        assert result.accelerator_type == AcceleratorType.CPU
        assert result.accelerator_model == AcceleratorModel.CPU


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestToLabelDicts:
    """Tests for AcceleratorDetectionResult.to_label_dicts()."""

    def test_gpu_result(self):
        result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=AcceleratorModel.MI300X,
            accelerator_count=8,
        )
        label_dicts = result.to_label_dicts()
        assert label_dicts == [{"accelerator_type": "GPU", "accelerator_model": "MI300X", "accelerator_count": 8}]

    def test_cpu_result(self):
        result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.CPU,
            accelerator_model=AcceleratorModel.EPYC_9965,
            accelerator_count=1,
        )
        label_dicts = result.to_label_dicts()
        assert label_dicts == [{"accelerator_type": "CPU", "accelerator_model": "EPYC_9965", "accelerator_count": 1}]

    def test_no_model_returns_empty_list(self):
        result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=None,
            accelerator_count=0,
        )
        assert result.to_label_dicts() == []


class TestToDetailDict:
    """Tests for AcceleratorDetectionResult.to_detail_dict()."""

    def test_gpu_includes_gpu_info(self):
        gpu = GPUInfo(
            device_id="0x74a1",
            model=AcceleratorModel.MI300X,
            vram_total=65536,
            vram_used=1024,
            gfx_utilization=0,
            mem_utilization=2,
        )
        result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=AcceleratorModel.MI300X,
            accelerator_count=1,
            gpu_info=[gpu],
        )
        detail = result.to_detail_dict()
        assert detail["accelerator_type"] == "gpu"
        assert detail["accelerator_model"] == "MI300X"
        assert detail["accelerator_count"] == 1
        assert len(detail["gpu_info"]) == 1
        assert detail["gpu_info"][0]["device_id"] == "0x74a1"
        assert "cpu_info" not in detail

    def test_cpu_includes_cpu_info(self):
        cpu = CPUInfo(
            vendor="AuthenticAMD",
            model=AcceleratorModel.EPYC_9965,
            model_number="9965",
            physical_cores=384,
            available_cores=8,
            cpuset_bind="0-7",
            node_cores=384,
        )
        result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.CPU,
            accelerator_model=AcceleratorModel.EPYC_9965,
            accelerator_count=384,
            cpu_info=cpu,
            cpu_cores=8,
            cpuset_bind="0-7",
        )
        detail = result.to_detail_dict()
        assert detail["accelerator_type"] == "cpu"
        assert detail["accelerator_model"] == "EPYC_9965"
        assert detail["accelerator_count"] == 384
        assert detail["cpu_info"]["vendor"] == "AuthenticAMD"
        assert detail["cpu_info"]["node_cores"] == 384
        assert detail["cpu_cores"] == 8
        assert detail["node_cores"] == 384
        assert detail["cpuset_bind"] == "0-7"
        assert "gpu_info" not in detail

    def test_no_model(self):
        result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=None,
            accelerator_count=0,
        )
        detail = result.to_detail_dict()
        assert detail["accelerator_model"] is None
