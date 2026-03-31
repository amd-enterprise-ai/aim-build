# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import ClassVar, FrozenSet, Optional

import click
from pydantic import BaseModel, Field, field_validator

from .asset_utils import AssetDescriptor, Initializer, assets_path_option
from .dict_utils import get_value
from .file_utils import KeyValueFileReader, TomlFileReader
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
        base_image_dict = config_dict.get("base_image", {})
        return cls(**base_image_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()


class CiBaseImageConfig(BaseModel):
    """Complete build configuration for CI pipelines.

    Combines the base image config from YAML with CI-specific build metadata.
    """

    repository: str = Field(..., description="Docker repository name to build (e.g., aim-base, aim-epyc-base)")
    dockerfile: str = Field(
        ...,
        description="Path to Dockerfile (e.g., docker/Dockerfile.aim-base or docker/Dockerfile.aim-<accelerator_type>-base)",
    )
    run_validation: bool = Field(..., description="Whether to run validation after build")
    base_image_ref: str = Field(..., description="Full base image reference")

    @classmethod
    def from_accelerator_type(cls, accelerator_type: str) -> "CiBaseImageConfig":
        """Create CiBaseImageConfig by resolving configuration for given accelerator type."""
        acc_lower = accelerator_type.lower()

        # Load and parse base image config
        config_path = Path("assets") / acc_lower / "base" / "config.yaml"
        base_config = BaseImageConfig.from_yaml_file(config_path)

        # Determine accelerator-specific CI values
        # Instinct uses "aim-" prefix for backward compatibility;
        # other accelerators use "aim-{accelerator}-" prefix.
        if acc_lower == "instinct":
            repository = "aim-base"
            dockerfile = "docker/Dockerfile.aim-base"
            run_validation = True
        else:
            repository = f"aim-{acc_lower}-base"
            dockerfile = f"docker/Dockerfile.aim-{acc_lower}-base"
            run_validation = False

        return cls(
            repository=repository,
            dockerfile=dockerfile,
            run_validation=run_validation,
            base_image_ref=base_config.image_ref,
        )


class ConfigInitializer(Initializer):

    def __init__(
        self,
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
            file_reader = KeyValueFileReader(Path("makefile-defaults.mk"))
            base_registry_namespace = file_reader.read_value("BASE_REGISTRY_NAMESPACE")
            base_repository = file_reader.read_value("BASE_REPOSITORY")
            base_tag = file_reader.read_value("BASE_TAG")
            base_registry_host = "docker.io"
        else:
            file_reader = TomlFileReader(Path("pyproject.toml"))
            base_repository = "aim-base"
            base_registry_namespace = "silogen"
            base_tag = file_reader.read_value("project.version")
            base_registry_host = "ghcr.io"

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
    ConfigInitializer(assets_path=assets_path, recreate=recreate).initialize_all()


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

    config = read_yaml(Path(assets_path) / canonical_name / "config.yaml")
    namespace = get_value(config, "base_image.base_registry_namespace")
    repository = get_value(config, "base_image.base_repository")
    tag = get_value(config, "base_image.base_tag")
    registry = get_value(config, "base_image.registry_host")

    if any(value is None for value in [namespace, repository, tag, registry]):
        raise ValueError(
            f"Base image information is incomplete in the configuration. Registry: {registry}, Namespace: '{namespace}', Repository: '{repository}', Tag: '{tag}'"
        )

    result = f"{registry}/{namespace}/{repository}:{tag}"

    github_output = os.getenv("GITHUB_OUTPUT")

    if github_output:
        with open(github_output, "a") as f:
            f.write(f"base_image_ref={result}\n")
    else:
        print(f"base_image_ref={result}")


@cli.command(name="resolve-build-config")
@click.option(
    "--accelerator-type",
    type=str,
    required=True,
    help="Accelerator type (e.g., instinct, epyc)",
)
def resolve_build_config(accelerator_type: str) -> None:
    """
    Resolve all build configuration details for a given accelerator type.

    Outputs (for GitHub Actions):
    - config: JSON object with repository, dockerfile, run_validation, base_image_ref
    """
    # Create CI configuration from accelerator type
    ci_config = CiBaseImageConfig.from_accelerator_type(accelerator_type)

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
