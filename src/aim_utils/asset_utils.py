# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import click

from aim_common import Engine
from aim_common.object_model import AcceleratorFamily, AcceleratorType, CanonicalName
from aim_utils.dict_utils import delete_key, get_value, set_value
from aim_utils.image_naming import parse_image_name
from aim_utils.yaml_utils import read_yaml, save_yaml

logger = logging.getLogger(__name__)


PROFILES_DIRNAME = "profiles"


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


@functools.cache
def infer_engine_from_profiles(asset_directory: Path) -> Engine:
    """Infer and cache the asset engine from its profiles.

    Use the first primary profile in path order, or the first profile with an
    engine. Skip unreadable profiles. Default to vLLM if no engine is found or
    the selected engine is unknown.
    """
    return _read_engine_from_profiles(asset_directory)


def _read_engine_from_profiles(asset_directory: Path) -> Engine:
    profiles_directory = asset_directory / PROFILES_DIRNAME
    if not profiles_directory.is_dir():
        return Engine.VLLM

    engine_value: Optional[str] = None

    for profile_path in sorted(profiles_directory.rglob("*.yaml")):
        try:
            profile = read_yaml(profile_path)
        except Exception as e:
            logger.warning(f"Could not read profile '{profile_path}': {e}")
            continue

        engine = get_value(profile, "metadata.engine")
        if not engine:
            continue

        if engine_value is None:
            engine_value = engine

        if get_value(profile, "metadata.primary"):
            engine_value = engine
            break

    if engine_value is None:
        return Engine.VLLM

    try:
        return Engine(engine_value)
    except ValueError:
        logger.warning(f"Unknown engine '{engine_value}' in profiles of '{asset_directory}'; assuming vLLM")
        return Engine.VLLM


def infer_accelerator_type_from_family(accelerator_family: AcceleratorFamily) -> AcceleratorType:
    mapping = {
        AcceleratorFamily.INSTINCT: AcceleratorType.GPU,
        AcceleratorFamily.RADEON: AcceleratorType.GPU,
        AcceleratorFamily.EPYC: AcceleratorType.CPU,
        AcceleratorFamily.CPU: AcceleratorType.CPU,
    }
    return mapping.get(accelerator_family, AcceleratorType.GPU)


ENGINE_TO_BASE_FOLDER_MAPPING = {
    Engine.VLLM: ".",
    Engine.VLLM_OMNI: "vllm-omni",
    Engine.BENTOML: "bentoml",
}


def _engine_folder_name(engine: Engine) -> str:
    return ENGINE_TO_BASE_FOLDER_MAPPING.get(engine, engine.value)


@dataclass
class AssetDescriptor:
    """A base or model-specific asset and its engine, location, and accelerator family."""

    is_base: bool
    is_custom: bool
    directory: Path
    org: Optional[str] = None
    model_name: Optional[str] = None
    engine: Engine = Engine.VLLM
    accelerator_family: AcceleratorFamily = AcceleratorFamily.INSTINCT

    @property
    def engine_folder_name(self) -> str:
        return _engine_folder_name(self.engine)

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
        return DefaultAssetManager.for_path(self.assets_path).get_descriptors()

    @abstractmethod
    def initialize(self, descriptor: AssetDescriptor) -> None:
        pass


DEFAULT_ASSETS_ROOT = "assets"

_T = TypeVar("_T", bound="AssetManager")


class AssetManager(ABC):

    def __init__(self, enforce_double_quotes: bool = False) -> None:
        self.enforce_double_quotes = enforce_double_quotes

        self.built_in_model_specific: List[AssetDescriptor] = []
        self.built_in_general: List[AssetDescriptor] = []
        self.built_in_general_engine: List[AssetDescriptor] = []
        self.custom_general: List[AssetDescriptor] = []
        self.custom_general_engine: List[AssetDescriptor] = []
        self.custom_model_specific: List[AssetDescriptor] = []

    @classmethod
    def for_path(cls: Type[_T], assets_path: Union[str, Path], **kwargs) -> _T:
        """Create an instance scoped to a single accelerator directory (e.g. 'assets/instinct')."""
        path_object = Path(assets_path)
        if not path_object.exists():
            raise ValueError(f"Assets path does not exist: {path_object}")
        instance = cls(**kwargs)
        instance._extend_descriptors_from_path(path_object)
        return instance

    @classmethod
    def for_root(cls: Type[_T], assets_root: Union[str, Path] = DEFAULT_ASSETS_ROOT, **kwargs) -> _T:
        """Create an instance that aggregates all accelerator directories under assets_root."""
        root = Path(assets_root)
        if not root.exists():
            raise ValueError(f"Assets root does not exist: {root}")
        instance = cls(**kwargs)
        for path in discover_assets_paths(root):
            instance._extend_descriptors_from_path(path)
        return instance

    @classmethod
    def for_optional_path(cls: Type[_T], assets_path: Optional[Union[str, Path]], **kwargs) -> _T:
        """Create an instance scoped to assets_path, or aggregating assets_root if assets_path is falsy."""
        if assets_path:
            return cls.for_path(assets_path, **kwargs)
        else:
            return cls.for_root(**kwargs)

    def _extend_descriptors_from_path(self, assets_path: Path) -> None:
        af = infer_accelerator_family_from_path(assets_path)
        self.built_in_model_specific.extend(self.__get_model_specific(assets_path, af, is_custom=False))
        base_directory = assets_path / "base"
        self.built_in_general.append(
            AssetDescriptor(directory=base_directory, is_base=True, is_custom=False, accelerator_family=af)
        )
        self.built_in_general_engine.extend(self.__get_engine_descriptors(base_directory, af, is_custom=False))

        custom_directory = assets_path / "custom"
        if custom_directory.exists():
            self.custom_general.append(
                AssetDescriptor(directory=custom_directory, is_base=True, is_custom=True, accelerator_family=af)
            )
        self.custom_general_engine.extend(self.__get_engine_descriptors(custom_directory, af, is_custom=True))
        self.custom_model_specific.extend(self.__get_model_specific(custom_directory, af, is_custom=True))

    @staticmethod
    def __get_engine_descriptors(
        base_directory: Path,
        accelerator_family: AcceleratorFamily,
        is_custom: bool,
    ) -> List[AssetDescriptor]:
        result = []
        for engine in Engine:
            if engine == Engine.VLLM:
                continue

            engine_directory = base_directory / _engine_folder_name(engine)
            if not engine_directory.exists():
                logger.debug(f"Optional directory does not exist: {engine_directory}")
                continue

            result.append(
                AssetDescriptor(
                    directory=engine_directory,
                    is_base=True,
                    is_custom=is_custom,
                    engine=engine,
                    accelerator_family=accelerator_family,
                )
            )

        return result

    @staticmethod
    def __get_model_specific(
        assets_path: Path,
        accelerator_family: AcceleratorFamily = AcceleratorFamily.INSTINCT,
        is_custom: bool = False,
    ) -> List[AssetDescriptor]:
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
                        is_custom=is_custom,
                        directory=model_dir,
                        engine=infer_engine_from_profiles(model_dir),
                        accelerator_family=accelerator_family,
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

            if not skip_custom:
                for descriptor in self.custom_model_specific:
                    if descriptor.org == canonical_name.org and descriptor.model_name == canonical_name.model_name:
                        result.append(descriptor)

            if not skip_base:
                result.extend(self.built_in_general)
                result.extend(self.built_in_general_engine)

                if not skip_custom:
                    result.extend(self.custom_general)
                    result.extend(self.custom_general_engine)

            if supported_engines is None:
                supported_engines = set(Engine)

            return [d for d in result if d.engine in supported_engines]

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
                result.extend(self.custom_general_engine)

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

    def get_yamls(self, canonical_name: Optional[CanonicalName] = None) -> List[Path]:
        raise NotImplementedError()


def assets_path_option(func: Callable[..., None]) -> Callable[..., None]:
    """Reusable decorator for required assets_path option."""
    return click.option(
        "--assets_path",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        required=True,
        help="Path to a specific accelerator assets directory (e.g., 'assets/instinct')",
    )(func)


def assets_path_optional_option(func: Callable[..., None]) -> Callable[..., None]:
    """Reusable decorator for optional assets_path option. When omitted, all accelerator directories under the default root are used."""
    return click.option(
        "--assets_path",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
        required=False,
        default=None,
        help=f"Path to a specific accelerator assets directory. When omitted, all directories under '{DEFAULT_ASSETS_ROOT}' are processed.",
    )(func)


def confirm_root_wide_mutation(assets_path: Optional[str], action: str, yes: bool = False) -> None:
    """Prompt for confirmation before a mutating command runs across every accelerator directory."""
    if assets_path is None and not yes:
        click.confirm(
            f"No --assets_path given; this will {action} across every accelerator directory "
            f"under '{DEFAULT_ASSETS_ROOT}'. Continue?",
            abort=True,
        )


def yes_option(func: Callable[..., None]) -> Callable[..., None]:
    """Reusable decorator for skipping the root-wide mutation confirmation prompt (e.g. in scripts/CI)."""
    return click.option(
        "--yes",
        "-y",
        is_flag=True,
        default=False,
        help="Skip the confirmation prompt when running a mutating command across every accelerator directory.",
    )(func)


def discover_assets_paths(assets_root: Path) -> List[Path]:
    """Discover all accelerator-specific asset directories under assets_root."""
    paths = sorted(d for d in assets_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not paths:
        logger.warning(f"No asset directories found under '{assets_root}'")
    return paths
