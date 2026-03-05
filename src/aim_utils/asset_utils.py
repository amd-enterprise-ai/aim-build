# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .file_utils import get_leaf_dirs

logger = logging.getLogger(__name__)


class Initializer(ABC):

    def __init__(
        self,
        assets_path: str = "assets",
        reference_path: Optional[str] = None,
        file_name: Optional[str] = None,
        recreate: bool = False,
    ) -> None:
        self.assets_path = assets_path
        self.reference_path = reference_path
        if reference_path is None:
            self.reference_path = assets_path
        self.file_name = file_name
        self.recreate = recreate

    def initialize_all(self) -> None:
        assets_path = Path(self.assets_path)

        if not assets_path.exists():
            logger.error(f"Directory does not exist: {assets_path}")
            return

        model_dirs = get_leaf_dirs(Path(self.reference_path))  # type: ignore[arg-type]
        logger.info(f"Found {len(model_dirs)} config directories")
        for model_dir in model_dirs:
            self.initialize(model_dir)

    @abstractmethod
    def initialize(self, model_dir: Path) -> None:
        pass
