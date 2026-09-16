# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from aim_common import AcceleratorFamily
from aim_utils.asset_utils import (
    AssetDescriptor,
    Initializer,
    infer_accelerator_family_from_path,
    infer_accelerator_family_from_repo,
)


class TestInitializer:
    class TestImplementation(Initializer):
        def initialize(self, descriptor: AssetDescriptor) -> None:
            pass

        def get_reference_descriptors(self) -> List[AssetDescriptor]:
            return [
                AssetDescriptor(
                    is_base=False,
                    is_custom=False,
                    directory=Path(self.assets_path) / "meta-llama" / "Llama-3.1-8B-Instruct",
                    org="meta-llama",
                    model_name="Llama-3.1-8B-Instruct",
                )
            ]

    def test_initializer_creation(self, assets_path):
        initializer = TestInitializer.TestImplementation(assets_path=str(assets_path))
        assert initializer.assets_path == str(assets_path)
        assert initializer.file_name is None
        assert initializer.recreate is False

    def test_initializer_custom_params(self, assets_path):
        initializer = TestInitializer.TestImplementation(
            assets_path=str(assets_path),
            file_name="custom.yaml",
            recreate=True,
        )
        assert initializer.file_name == "custom.yaml"
        assert initializer.recreate is True

    def test_initialize_all_nonexistent_directory(self, tmp_path):
        initializer = TestInitializer.TestImplementation(assets_path=str(tmp_path / "nonexistent"))
        with patch("aim_utils.asset_utils.logger") as mock_logger:
            initializer.initialize_all()
            mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for infer_accelerator_family_from_path
# ---------------------------------------------------------------------------


class TestInferAcceleratorFamilyFromPath:

    def test_none_path_raises(self):
        with pytest.raises(Exception):
            infer_accelerator_family_from_path(None)  # type: ignore[arg-type]

    def test_path_without_assets_segment_returns_default(self):
        result = infer_accelerator_family_from_path(Path("/home/user/models/instinct"))
        assert result == AcceleratorFamily.INSTINCT

    def test_instinct_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/instinct"))
        assert result == AcceleratorFamily.INSTINCT

    def test_radeon_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/radeon"))
        assert result == AcceleratorFamily.RADEON

    def test_epyc_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/epyc"))
        assert result == AcceleratorFamily.EPYC

    def test_cpu_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/cpu"))
        assert result == AcceleratorFamily.CPU

    def test_nested_instinct_path(self):
        result = infer_accelerator_family_from_path(Path("assets/instinct/google/gemma-3-1b-it"))
        assert result == AcceleratorFamily.INSTINCT

    def test_nested_radeon_path(self):
        result = infer_accelerator_family_from_path(Path("assets/radeon/meta-llama/llama-3-8b"))
        assert result == AcceleratorFamily.RADEON

    def test_absolute_path_with_assets(self):
        result = infer_accelerator_family_from_path(Path("/home/user/aim-build/assets/epyc"))
        assert result == AcceleratorFamily.EPYC

    def test_assets_only_path_returns_default_instinct(self):
        # Path has "assets" as its only component: no subfolder follows
        result = infer_accelerator_family_from_path(Path("assets"))
        assert result == AcceleratorFamily.INSTINCT

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            infer_accelerator_family_from_path(Path("assets/unknown"))


# ---------------------------------------------------------------------------
# Tests for infer_accelerator_family_from_repo
# ---------------------------------------------------------------------------


class TestInferAcceleratorFamilyFromRepo:

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            infer_accelerator_family_from_repo("")

    def test_aim_base_returns_instinct(self):
        assert infer_accelerator_family_from_repo("aim-base") == AcceleratorFamily.INSTINCT

    def test_aim_epyc_base_returns_epyc(self):
        assert infer_accelerator_family_from_repo("aim-epyc-base") == AcceleratorFamily.EPYC

    def test_aim_radeon_base_returns_radeon(self):
        assert infer_accelerator_family_from_repo("aim-radeon-base") == AcceleratorFamily.RADEON

    def test_aim_cpu_base_returns_cpu(self):
        assert infer_accelerator_family_from_repo("aim-cpu-base") == AcceleratorFamily.CPU

    def test_aim_instinct_model_repo_returns_instinct(self):
        assert infer_accelerator_family_from_repo("aim-instinct-google-gemma-3-1b-it") == AcceleratorFamily.INSTINCT

    def test_aim_epyc_model_repo_returns_epyc(self):
        assert infer_accelerator_family_from_repo("aim-epyc-meta-llama-3-8b") == AcceleratorFamily.EPYC

    def test_aim_radeon_model_repo_returns_radeon(self):
        assert infer_accelerator_family_from_repo("aim-radeon-meta-llama-3-8b") == AcceleratorFamily.RADEON

    def test_unknown_family_raises(self):
        assert infer_accelerator_family_from_repo("aim-unknown-base") == AcceleratorFamily.INSTINCT
