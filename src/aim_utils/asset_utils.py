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
from aim_common.object_model import AcceleratorFamily, AcceleratorType, CanonicalName
from aim_utils.dict_utils import delete_key, get_value, set_value
from aim_utils.image_naming import parse_image_name
from aim_utils.yaml_utils import read_yaml, save_yaml

logger = logging.getLogger(__name__)


def infer_accelerator_family_from_path(assets_path: Path) -> AcceleratorFamily:
    if not assets_path:
        raise ValueError("Assets path is not set")

    parts = assets_path.parts
    if parts:
        if "assets" in parts:
            assets_index = parts.index("assets")
            if len(parts) >= assets_index + 2:
                return AcceleratorFamily(parts[assets_index + 1])

    return AcceleratorFamily.INSTINCT


def infer_accelerator_family_from_repo(repository: str) -> AcceleratorFamily:
    parsed = parse_image_name(repository)
    if parsed.accelerator is None:
        raise ValueError(f"Could not determine accelerator family from repository name: {repository}")
    return AcceleratorFamily(parsed.accelerator)


def infer_accelerator_type_from_family(accelerator_family: AcceleratorFamily) -> AcceleratorType:
    mapping = {
        AcceleratorFamily.INSTINCT: AcceleratorType.GPU,
        AcceleratorFamily.RADEON: AcceleratorType.GPU,
        AcceleratorFamily.EPYC: AcceleratorType.CPU,
        AcceleratorFamily.CPU: AcceleratorType.CPU,
    }
    return mapping.get(accelerator_family, AcceleratorType.GPU)


@dataclass
class AssetDescriptor:
    is_base: bool
    is_custom: bool
    directory: Path
    org: Optional[str] = None
    model_name: Optional[str] = None
    engine: Engine = Engine.VLLM
    accelerator_family: AcceleratorFamily = AcceleratorFamily.INSTINCT

    @property
    def engine_folder_name(self) -> str:
        ENGINE_TO_BASE_FOLDER_MAPPING = {
            Engine.VLLM: ".",
            Engine.VLLM_OMNI: "vllm-omni",
            Engine.BENTOML: "bentoml",
        }
        return ENGINE_TO_BASE_FOLDER_MAPPING.get(self.engine, self.engine.value)

    @property
    def accelerator_family_folder_name(self) -> str:
        return self.accelerator_family.value


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
        self.assets_path_object = Path(self.assets_path)
        if not self.assets_path_object.exists():
            raise ValueError(f"Assets directory does not exist: {self.assets_path}")

        self.built_in_model_specific = self.__get_model_specific()
        self.built_in_general = self.__get_general()

        self.custom_general = self.__get_general(is_custom=True)
        self.custom_model_specific = self.__get_model_specific(is_custom=True)

        self.enforce_double_quotes = enforce_double_quotes

    def __get_general(self, is_custom: bool = False) -> List[AssetDescriptor]:

        def create_descriptor(assets_path: Path, engine: Engine, is_custom: bool = False) -> Optional[AssetDescriptor]:
            if is_custom:
                descriptor_directory = assets_path / "custom"
            else:
                descriptor_directory = assets_path / "base"

            descriptor = AssetDescriptor(
                directory=descriptor_directory,
                is_base=True,
                is_custom=False,
                engine=engine,
                accelerator_family=infer_accelerator_family_from_path(Path(self.assets_path)),
            )

            descriptor_directory = descriptor_directory / descriptor.engine_folder_name

            if descriptor_directory.exists():
                return descriptor

            return None

        result: list[AssetDescriptor] = []
        for engine in Engine:
            engine_base = create_descriptor(self.assets_path_object, engine, is_custom=is_custom)
            if engine_base:
                result.append(engine_base)

        return result

    def __get_model_specific(self, is_custom: bool = False) -> List[AssetDescriptor]:
        if not self.assets_path_object.exists():
            logger.debug(f"Optional directory does not exist: {self.assets_path_object}")
            return []

        result = []
        for org_dir in self.assets_path_object.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith("."):
                continue

            if org_dir == self.assets_path_object / "base":
                continue

            if org_dir == self.assets_path_object / "custom":
                continue

            if org_dir == self.assets_path_object / "custom" / "profiles":
                continue

            for model_dir in org_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                if not model_dir.exists():
                    continue
                result.append(
                    AssetDescriptor(
                        org=org_dir.name,
                        model_name=model_dir.name,
                        is_base=False,
                        is_custom=is_custom,
                        directory=model_dir,
                        accelerator_family=infer_accelerator_family_from_path(Path(self.assets_path)),
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
        skip_model_specific: bool = False,
        supported_engines: Optional[set[Engine]] = None,
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

            if not skip_base:
                result.extend(self.built_in_general)

            if not skip_custom:
                if not skip_base:
                    result.extend(self.custom_general)

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

        if supported_engines is None:
            supported_engines = set(Engine)

        result = [d for d in result if d.engine in supported_engines]
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
