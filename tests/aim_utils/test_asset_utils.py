# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
from pathlib import Path
from unittest.mock import patch

import pytest

from aim_utils.asset_utils import Initializer


@pytest.fixture
def temp_assets_dir(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    return assets_dir


class TestInitializer:
    class TestImplementation(Initializer):
        def initialize(self, model_dir: Path) -> None:
            pass

    def test_initializer_creation(self, temp_assets_dir):
        initializer = TestInitializer.TestImplementation(assets_path=str(temp_assets_dir))
        assert initializer.assets_path == str(temp_assets_dir)
        assert initializer.file_name is None
        assert initializer.recreate is False

    def test_initializer_custom_params(self, temp_assets_dir):
        initializer = TestInitializer.TestImplementation(
            assets_path=str(temp_assets_dir),
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
