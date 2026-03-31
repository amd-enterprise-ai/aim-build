# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import click

from aim_common.object_model import CanonicalName

logger = logging.getLogger(__name__)


@dataclass
class AssetDescriptor:
    is_base: bool
    is_custom: bool
    directory: Path
    org: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class Asset:
    asset_metadata: AssetDescriptor
    file_path: Path


class Initializer(ABC):

    def __init__(
        self,
        assets_path: str = "assets/instinct",
        file_name: Optional[str] = None,
        recreate: bool = False,
    ) -> None:
        self.assets_path = assets_path
        self.file_name = file_name
        self.recreate = recreate

    def initialize_all(self) -> None:
        assets_path = Path(self.assets_path)

        if not assets_path.exists():
            logger.error(f"Directory does not exist: {assets_path}")
            return

        reference_descriptors = self.get_reference_descriptors()
        logger.debug(f"Found {len(reference_descriptors)} reference assets")
        for descriptor in reference_descriptors:
            self.initialize(descriptor)

    def get_reference_descriptors(self) -> List[AssetDescriptor]:
        return DefaultAssetManager().get_descriptors()

    @abstractmethod
    def initialize(self, descriptor: AssetDescriptor) -> None:
        pass


class AssetManager(ABC):

    def __init__(self, assets_path: str = "assets/instinct") -> None:
        self.assets_path = assets_path
        assets_path_object = Path(self.assets_path)

        self.built_in_model_specific = self.__get_model_specific(assets_path_object)
        self.built_in_general = [AssetDescriptor(directory=assets_path_object / "base", is_base=True, is_custom=False)]
        self.custom_general = [AssetDescriptor(directory=assets_path_object / "custom", is_base=True, is_custom=True)]
        self.custom_model_specific = self.__get_model_specific(assets_path_object / "custom")

    @staticmethod
    def __get_model_specific(assets_path: Path) -> List[AssetDescriptor]:
        if not assets_path.exists():
            logger.debug(f"Optional directory does not exist: {assets_path}")
            return []

        result = []
        for org_dir in assets_path.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith("."):
                continue

            if org_dir == assets_path / "base":
                continue

            if org_dir == assets_path / "custom":
                continue

            if org_dir == assets_path / "custom" / "profiles":
                continue

            for model_dir in org_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                result.append(
                    AssetDescriptor(
                        org=org_dir.name,
                        model_name=model_dir.name,
                        is_base=False,
                        is_custom="custom" in assets_path.parts,
                        directory=model_dir,
                    )
                )

        return result

    @abstractmethod
    def get_yamls(self, canonical_name: Optional[CanonicalName] = None) -> List[Path]:
        pass

    def get_descriptors(
        self,
        canonical_name: Optional[CanonicalName] = None,
        skip_base: bool = False,
        skip_custom: bool = True,
        skip_model_specific=False,
    ) -> List[AssetDescriptor]:
        result = []

        if canonical_name is not None:
            if not skip_model_specific:
                for descriptor in self.built_in_model_specific:
                    if descriptor.org == canonical_name.org and descriptor.model_name == canonical_name.model_name:
                        result.append(descriptor)
                        break

            if not skip_custom:
                for descriptor in self.custom_model_specific:
                    if descriptor.org == canonical_name.org and descriptor.model_name == canonical_name.model_name:
                        result.append(descriptor)
                        break

            return result

        if not skip_model_specific:
            result.extend(self.built_in_model_specific)

        if not skip_base:
            result.extend(self.built_in_general)

        if not skip_custom:
            if not skip_model_specific:
                result.extend(self.custom_model_specific)

            if not skip_base:
                result.extend(self.custom_general)

        return result

    def get_dirs(self) -> List[Path]:
        yamls = self.get_yamls()
        folders = set(yaml.parent for yaml in yamls)
        return list(folders)

    def delete_assets(self):
        yaml_paths = self.get_yamls()
        for yaml_path in yaml_paths:
            yaml_path.unlink(missing_ok=True)


class DefaultAssetManager(AssetManager):

    def get_yamls(self, canonical_name: Optional[CanonicalName] = None) -> List[Path]:
        raise NotImplementedError()


def assets_path_option(func: Callable[..., None]) -> Callable[..., None]:
    """Reusable decorator for assets_path option."""
    return click.option(
        "--assets_path",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        default="assets/instinct",
        help="Path to the root assets directory",
    )(func)
