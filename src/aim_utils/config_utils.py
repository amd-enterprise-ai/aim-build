# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, FrozenSet, Mapping, Optional

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field, field_validator

from aim_common.object_model import AcceleratorFamily
from aim_utils.image_naming import LEGACY_VLLM_BASE_TARGET_ID, ImageName, get_base_image_name

from .asset_utils import AssetDescriptor, Initializer, assets_path_option
from .dict_utils import get_value
from .file_utils import TomlFileReader
from .version_utils import validate_version_tag
from .yaml_utils import read_yaml, save_yaml

logger = logging.getLogger(__name__)


class BaseImageConfig(BaseModel):
    """Configuration for base Docker images used in AIM builds.
    This represents the 'base_image' section in config.yaml files.
    """

    # Allowed container registry hostnames. Extend this set if additional registries are adopted.
    ALLOWED_REGISTRY_HOSTS: ClassVar[FrozenSet[str]] = frozenset({"docker.io", "ghcr.io"})

    registry_host: str = Field(..., description="Registry hostname (e.g., docker.io, ghcr.io)")
    base_registry_namespace: str = Field(..., description="Registry namespace (e.g., rocm, amdih)")
    base_repository: str = Field(..., description="Repository name (e.g., vllm, zendnn_zentorch)")
    base_tag: str = Field(..., description="Image tag/version")

    @field_validator("registry_host")
    @classmethod
    def _validate_registry_host(cls, v: str) -> str:
        if v not in cls.ALLOWED_REGISTRY_HOSTS:
            raise ValueError(f"Invalid registry host: '{v}'. Allowed registries: {sorted(cls.ALLOWED_REGISTRY_HOSTS)}")
        return v

    @field_validator("base_tag")
    @classmethod
    def _validate_base_tag(cls, v: str, info) -> str:
        # Only validate as an AIM version when the repository is an AIM image.
        # Upstream base images (e.g. docker.io/rocm/vllm) use free-form tags.
        repo = info.data.get("base_repository", "")
        if repo.startswith("aim"):
            validate_version_tag(v, is_base=True)
        return v

    @property
    def image_ref(self) -> str:
        """Construct full image reference."""
        return f"{self.registry_host}/{self.base_registry_namespace}/{self.base_repository}:{self.base_tag}"

    @classmethod
    def from_yaml_file(cls, config_path: Path) -> "BaseImageConfig":
        """Load BaseImageConfig from a YAML config file."""
        config_dict = read_yaml(config_path)
        parsed_config = BaseImageConfigFile.model_validate(config_dict)
        return parsed_config.base_image

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()


class BaseImageConfigFile(BaseModel):
    """Top-level YAML schema containing one required ``base_image`` block."""

    # Extra top-level keys allowed
    model_config = ConfigDict(extra="allow")

    base_image: BaseImageConfig

    @classmethod
    def from_yaml_file(cls, config_path: Path) -> "BaseImageConfigFile":
        """Load and validate a base config file."""
        config_dict = read_yaml(config_path)
        return cls.model_validate(config_dict)


class BaseImageTargetConfig(BaseImageConfig):
    """Base image config extended with a target identifier."""

    target_id: str = Field(..., description="Target identifier (e.g. legacy_vllm, bentoml)")


class CiBaseImageTarget(BaseModel):
    """CI build metadata for one normalized base image target."""

    target_id: str = Field(..., description="Target identifier (e.g. legacy_vllm, bentoml)")
    image_name: ImageName = Field(
        ...,
        description="Private and public repository name pair (e.g., aim-instinct-base / aim-base)",
    )
    dockerfile: str = Field(
        ...,
        description="Path to Dockerfile (e.g., docker/Dockerfile.aim-instinct-base, docker/Dockerfile.aim-radeon-base)",
    )
    run_validation: bool = Field(..., description="Whether to run validation after build")
    upstream_image_ref: str = Field(..., description="Full upstream image reference")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def repository(self) -> str:
        return self.image_name.private

    @computed_field  # type: ignore[prop-decorator]
    @property
    def public_repository(self) -> str:
        return self.image_name.public

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_alias(self) -> bool:
        return self.image_name.has_alias


def _normalize_target_dict(raw_target: dict[str, Any], target_id: str) -> BaseImageTargetConfig:
    """Validate and normalize one target entry into a BaseImageTargetConfig."""
    return BaseImageTargetConfig(target_id=target_id, **raw_target)


def normalize_base_image_targets(
    config_dict: Mapping[str, Any],
) -> dict[str, BaseImageTargetConfig]:
    """Return a single-entry target mapping from a base config dict.

    Reads ``base_image`` and returns it keyed under ``legacy_vllm``.
    Returns an empty dict when ``base_image`` is absent.
    Raises if ``base_images`` is present.
    """
    target_configs = config_dict.get("base_images")

    if target_configs is not None:
        raise ValueError(
            "inline 'base_images' mapping is not supported; define each new target under "
            "assets/<accelerator>/base/<target_id>/config.yaml"
        )

    elif config_dict.get("base_image") is None:
        return {}

    else:
        parsed_config = BaseImageConfigFile.model_validate(config_dict)
        return {
            LEGACY_VLLM_BASE_TARGET_ID: _normalize_target_dict(
                parsed_config.base_image.model_dump(),
                LEGACY_VLLM_BASE_TARGET_ID,
            )
        }


def _load_base_target_config(config_path: Path, target_id: str) -> BaseImageTargetConfig:
    """Load one target config file and return normalized target config."""
    parsed_config = BaseImageConfigFile.from_yaml_file(config_path)
    return BaseImageTargetConfig(target_id=target_id, **parsed_config.base_image.model_dump())


def resolve_base_image_targets(accelerator_family: str) -> list[BaseImageTargetConfig]:
    """Resolve all base targets for one accelerator family from base root and subdirectories."""
    acc_lower = accelerator_family.lower()
    base_dir = Path("assets") / acc_lower / "base"

    targets: list[BaseImageTargetConfig] = []

    legacy_config_path = base_dir / "config.yaml"
    if legacy_config_path.exists():
        targets.append(_load_base_target_config(legacy_config_path, LEGACY_VLLM_BASE_TARGET_ID))

    if not base_dir.is_dir():
        raise click.UsageError(
            f"Base configuration not found for accelerator family '{accelerator_family}'. "
            f"Ensure the accelerator is valid and the required assets are available."
        )

    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue

        target_config_path = child / "config.yaml"
        if not target_config_path.exists():
            continue

        targets.append(_load_base_target_config(target_config_path, child.name))

    return targets


def resolve_ci_base_image_targets(accelerator_family: str) -> list[CiBaseImageTarget]:
    """Resolve CI build target metadata for all discovered base targets."""
    acc_lower = accelerator_family.lower()
    dockerfile = f"docker/Dockerfile.aim-{acc_lower}-base"
    default_run_validation = acc_lower == "instinct"

    ci_targets: list[CiBaseImageTarget] = []
    for target in resolve_base_image_targets(accelerator_family):
        image_name = get_base_image_name(acc_lower, target.target_id)
        run_validation = default_run_validation and target.target_id != "bentoml"
        ci_targets.append(
            CiBaseImageTarget(
                target_id=target.target_id,
                image_name=image_name,
                dockerfile=dockerfile,
                run_validation=run_validation,
                upstream_image_ref=target.image_ref,
            )
        )

    return ci_targets


class CiBaseImageConfig(BaseModel):
    """Complete build configuration for CI pipelines.

    Combines the base image config from YAML with CI-specific build metadata.
    The image_name field owns the canonical/public name pair; repository,
    canonical_repository, public_repository, and has_alias are derived from it
    so they stay in sync.
    """

    base_target_id: str = Field(..., description="Target identifier (e.g. legacy_vllm, bentoml)")
    image_name: ImageName = Field(
        ...,
        description="Canonical and public repository name pair (e.g., aim-instinct-base / aim-base)",
    )
    dockerfile: str = Field(
        ...,
        description="Path to Dockerfile (e.g., docker/Dockerfile.aim-instinct-base, docker/Dockerfile.aim-radeon-base)",
    )
    run_validation: bool = Field(..., description="Whether to run validation after build")
    upstream_image_ref: str = Field(..., description="Full upstream image reference")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def repository(self) -> str:
        """Private image repository name used for CI/developer pushes (e.g., aim-instinct-base)."""
        return self.image_name.private

    @computed_field  # type: ignore[prop-decorator]
    @property
    def public_repository(self) -> str:
        """Public image repository name (backward-compatible, e.g., aim-base for instinct)."""
        return self.image_name.public

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_alias(self) -> bool:
        """Whether the public name differs from canonical (true if dual-tagging is needed)."""
        return self.image_name.has_alias


class ConfigInitializer(Initializer):

    def __init__(
        self,
        registry_host: str,
        registry_namespace: str,
        assets_path: str = "assets/instinct",
        file_name: Optional[str] = None,
        recreate: bool = False,
    ) -> None:
        if file_name is None:
            file_name = "config.yaml"
        super().__init__(
            assets_path=assets_path,
            file_name=file_name,
            recreate=recreate,
        )
        self._registry_host = registry_host
        self._registry_namespace = registry_namespace

    def initialize(self, assets_descriptor: AssetDescriptor) -> None:
        output_path = assets_descriptor.directory / self.file_name  # type: ignore

        if output_path.exists() and output_path.stat().st_size > 0:
            if self.recreate:
                logger.warning(f"Config already exists and is not empty for '{output_path}', recreating...")
            else:
                logger.info(f"Config already exists and is not empty for '{output_path}', skipping...")
                return
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if assets_descriptor.is_base:
            base_registry_host = "docker.io"
            base_registry_namespace = "vllm"
            base_repository = "vllm-openai-rocm"
            base_tag = "v0.20.0"
        else:
            file_reader = TomlFileReader(Path("pyproject.toml"))
            accelerator = Path(self.assets_path).name
            # Use centralized naming utility
            image_name = get_base_image_name(accelerator, LEGACY_VLLM_BASE_TARGET_ID)
            base_repository = image_name.canonical  # Use canonical name for config generation
            base_registry_namespace = self._registry_namespace
            base_tag = file_reader.read_value("project.version")
            base_registry_host = self._registry_host

        config = {
            "base_image": {
                "base_registry_namespace": base_registry_namespace,
                "base_repository": base_repository,
                "base_tag": base_tag,
                "registry_host": base_registry_host,
            }
        }

        save_yaml(config, path=output_path, enforce_double_quotes=False)
        logger.debug(f"Generated config for {assets_descriptor.directory}")


def get_canonical_name(canonical_name_option: Optional[str] = None) -> str:
    if canonical_name_option is None:
        canonical_name = os.getenv("CANONICAL_NAME")
    else:
        return canonical_name_option

    if canonical_name is None:
        raise ValueError(
            "Canonical name must be provided either as an option or via the CANONICAL_NAME environment variable."
        )

    return canonical_name


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command(name="init")
@assets_path_option
@click.option("--recreate", is_flag=True, default=False, help="Whether to recreate existing configuration files.")
def init_config(assets_path: str = "assets/instinct", recreate: bool = False) -> None:
    registry_host = os.environ["AIM_REGISTRY_HOSTNAME"]
    registry_namespace = os.environ["AIM_REGISTRY_NAMESPACE"]
    ConfigInitializer(
        assets_path=assets_path,
        recreate=recreate,
        registry_host=registry_host,
        registry_namespace=registry_namespace,
    ).initialize_all()


@cli.command(name="get")
@click.argument("key", type=str)
@click.option("--canonical_name", type=str, default=None)
@assets_path_option
def get_config_value(key: str, canonical_name: Optional[str] = None, assets_path: str = "assets/instinct"):
    canonical_name = get_canonical_name(canonical_name)
    config = read_yaml(Path(assets_path) / canonical_name / "config.yaml")
    value = get_value(config, key)
    print(value)


@cli.command(name="get-base-image-ref")
@click.option("--canonical_name", type=str, default=None)
@assets_path_option
def get_base_image_ref(canonical_name: Optional[str] = None, assets_path: str = "assets/instinct"):
    canonical_name = get_canonical_name(canonical_name)
    config_path = Path(assets_path) / canonical_name / "config.yaml"

    try:
        base_config = BaseImageConfig.from_yaml_file(config_path)
    except ValidationError as exc:
        raise ValueError(f"Base image information is incomplete in the configuration: {exc}") from exc

    result = base_config.image_ref

    github_output = os.getenv("GITHUB_OUTPUT")

    if github_output:
        with open(github_output, "a") as f:
            f.write(f"base_image_ref={result}\n")
    else:
        print(f"base_image_ref={result}")


@cli.command(name="resolve-build-config")
@click.option(
    "--accelerator-family",
    type=click.Choice([af.value for af in AcceleratorFamily]),
    required=True,
    help="Accelerator family (e.g., instinct, epyc, radeon)",
)
@click.option(
    "--base-target-id",
    type=str,
    required=False,
    default=None,
    help="Base image target identifier (e.g. legacy_vllm, bentoml). Defaults to legacy_vllm.",
)
def resolve_build_config(accelerator_family: str, base_target_id: Optional[str] = None) -> None:
    """
    Resolve all build configuration details for a given accelerator family and target.

    Outputs (for GitHub Actions):
    - config: JSON object with base_target_id, repository, dockerfile, run_validation, upstream_image_ref
    """
    target_id = base_target_id or LEGACY_VLLM_BASE_TARGET_ID

    ci_targets = resolve_ci_base_image_targets(accelerator_family)
    ci_target = next((t for t in ci_targets if t.target_id == target_id), None)

    if ci_target is None:
        available = ", ".join(t.target_id for t in ci_targets)
        raise click.UsageError(
            f"No base target '{target_id}' found for accelerator family '{accelerator_family}'. "
            f"Available targets: {available}"
        )

    ci_config = CiBaseImageConfig(
        base_target_id=ci_target.target_id,
        image_name=ci_target.image_name,
        dockerfile=ci_target.dockerfile,
        run_validation=ci_target.run_validation,
        upstream_image_ref=ci_target.upstream_image_ref,
    )

    # Output as JSON
    config_json = ci_config.model_dump_json()

    # Write to GITHUB_OUTPUT or print
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"config={config_json}\n")
    else:
        # Print JSON for local testing
        print(config_json)


@cli.command(name="resolve-build-targets")
@click.option(
    "--accelerator-family",
    type=click.Choice([af.value for af in AcceleratorFamily]),
    required=True,
    help="Accelerator family (e.g., instinct, epyc, radeon)",
)
def resolve_build_targets(accelerator_family: str) -> None:
    """Resolve all discovered base build targets for one accelerator family."""
    targets = resolve_ci_base_image_targets(accelerator_family)
    targets_json = json.dumps([target.model_dump(mode="json") for target in targets])

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"targets={targets_json}\n")
    else:
        print(targets_json)


@cli.command(name="validate")
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--assets_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=None,
    help="Path to the root assets directory (required when no FILES are given)",
)
def validate_configs(files, assets_path: Optional[str] = None) -> None:
    """Validate config.yaml files against the AIM versioning and registry conventions.

    If FILES are given, validate only those files; otherwise validate all config.yaml files
    discovered under --assets_path.
    """
    from pathlib import Path as _Path

    errors: list[str] = []
    paths: list[_Path] = []

    if files:
        paths = [_Path(f) for f in files]
    elif assets_path:
        assets = _Path(assets_path)
        paths = sorted(assets.rglob("config.yaml"))
    else:
        raise click.UsageError("Either FILES or --assets_path must be provided.")

    for config_path in paths:
        try:
            BaseImageConfig.from_yaml_file(config_path)
        except Exception as exc:
            errors.append(f"{config_path}: {exc}")

    if errors:
        for err in errors:
            click.echo(f"ERROR: {err}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
