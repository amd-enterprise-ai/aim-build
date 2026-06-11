# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import click

from aim_common import Engine
from aim_common.object_model import CanonicalName
from aim_utils.dict_utils import delete_key, get_value, set_value
from aim_utils.yaml_utils import read_yaml, save_yaml

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
        assets_path: str,
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
        return DefaultAssetManager(self.assets_path).get_descriptors()

    @abstractmethod
    def initialize(self, descriptor: AssetDescriptor) -> None:
        pass


class AssetManager(ABC):

    def __init__(self, assets_path: str, enforce_double_quotes: bool) -> None:
        self.assets_path = assets_path
        assets_path_object = Path(self.assets_path)

        self.built_in_model_specific = self.__get_model_specific(assets_path_object)
        self.built_in_general = [AssetDescriptor(directory=assets_path_object / "base", is_base=True, is_custom=False)]

        self.built_in_general_engine = []

        for engine in Engine:
            if engine == Engine.VLLM:
                continue

            path_addition = engine.value

            if engine == Engine.VLLM_OMNI:
                path_addition = path_addition.replace("_", "-")

            self.built_in_general_engine.append(
                AssetDescriptor(directory=assets_path_object / "base" / path_addition, is_base=True, is_custom=False)
            )

        self.custom_general = [AssetDescriptor(directory=assets_path_object / "custom", is_base=True, is_custom=True)]
        self.custom_model_specific = self.__get_model_specific(assets_path_object / "custom")
        self.enforce_double_quotes = enforce_double_quotes

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
            result.extend(self.built_in_general_engine)

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

    def copy_value(
        self,
        source_key: str,
        target_key: str,
        canonical_name: Optional[str] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        separator: str = "",
        add_if_missing: bool = False,
    ) -> None:
        def _copy_value(
            data: Dict[str, Any],
            source_key: str,
            target_key: str,
            prefix: Optional[str] = None,
            postfix: Optional[str] = None,
            separator: str = "",
            add_if_missing: bool = False,
        ) -> Dict[str, Any]:
            source_value = get_value(data, source_key)
            if isinstance(source_value, str):
                if prefix:
                    source_value = f"{prefix}{separator}{source_value}"
                if postfix:
                    source_value = f"{source_value}{separator}{postfix}"
            return set_value(data, target_key, source_value, add_if_missing)

        yaml_files = self.get_yamls(CanonicalName.from_string(canonical_name))
        logger.debug(f"Found {len(yaml_files)} metadata files to update")

        for file_path in yaml_files:
            data = read_yaml(file_path)
            try:
                data = _copy_value(
                    data,
                    source_key,
                    target_key,
                    prefix,
                    postfix,
                    separator,
                    add_if_missing,
                )
                save_yaml(data, path=file_path, enforce_double_quotes=self.enforce_double_quotes)
            except Exception as e:
                logger.error(f"Error copying value in '{file_path}'")
                raise e

    def delete_key(self, key: str, canonical_name: Optional[str] = None) -> None:
        """
        Remove a specific key from all metadata.yaml files
        :param key: key to remove in dot notation (e.g., "org.opencontainers.image.vendor")
        :param canonical_name: directory name to filter by (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        """
        files = self.get_yamls(CanonicalName.from_string(canonical_name))

        for file_path in files:
            data = read_yaml(file_path)

            logger.info(f"Deleting key from metadata file: '{file_path}'")
            try:
                data = delete_key(data, key)
                save_yaml(data, path=file_path, enforce_double_quotes=self.enforce_double_quotes)
            except Exception as e:
                logger.error(f"Error removing key from '{file_path}': {str(e)}")
                raise e

    def rename_key(self, source_key: str, target_key: str, canonical_name: Optional[str] = None):
        self.copy_value(source_key, target_key, canonical_name, add_if_missing=True)
        self.delete_key(source_key, canonical_name)

    def update_value(
        self,
        key: str,
        new_value: Optional[Any] = None,
        canonical_name: Optional[str] = None,
        add_if_missing: bool = False,
    ) -> None:
        """
        Update a specific field in YAML file(s)

        Args:
            key: Dot notation path to the key (e.g., "org.opencontainers.image.vendor")
            new_value: New value to set for the key
            canonical_name: If provided, only update files matching this canonical name
            add_if_missing: If True, add the key if it doesn't exist
        """
        files = self.get_yamls(CanonicalName.from_string(canonical_name))
        logger.debug(f"Found {len(files)} metadata files to update")

        for file_path in files:
            logger.info(f"File name: '{file_path}'")
            try:
                data = read_yaml(file_path)
                data = set_value(data, key, new_value, add_if_missing=add_if_missing)
                save_yaml(data, path=file_path, enforce_double_quotes=self.enforce_double_quotes)
            except Exception as e:
                logger.error(f"Error updating '{file_path}': {str(e)}")
                raise e


class DefaultAssetManager(AssetManager):

    def __init__(self, assets_path: str):
        super().__init__(assets_path, enforce_double_quotes=False)

    def get_yamls(self, canonical_name: Optional[CanonicalName] = None) -> List[Path]:
        raise NotImplementedError()


def assets_path_option(func: Callable[..., None]) -> Callable[..., None]:
    """Reusable decorator for assets_path option."""
    return click.option(
        "--assets_path",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        required=True,
        help="Path to the root assets directory",
    )(func)


def assets_root_option(func: Callable[..., None]) -> Callable[..., None]:
    """Reusable decorator for assets_root option."""
    return click.option(
        "--assets_root",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        default="assets",
        help="Root directory containing accelerator-specific asset directories",
    )(func)


def discover_assets_paths(assets_root: str) -> List[str]:
    """Discover all accelerator-specific asset directories under assets_root."""
    root = Path(assets_root)
    paths = sorted(str(d) for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not paths:
        logger.warning(f"No asset directories found under '{assets_root}'")
    return paths
