# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import List
from unittest.mock import patch

from aim_utils.asset_utils import AssetDescriptor, Initializer


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
