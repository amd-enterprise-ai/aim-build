# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import Optional

import click

from .asset_utils import Initializer
from .dict_utils import get_value
from .file_utils import KeyValueFileReader, TomlFileReader, extract_model_info
from .yaml_utils import read_yaml, save_yaml

logger = logging.getLogger(__name__)


class ConfigInitializer(Initializer):

    def __init__(
        self,
        assets_path: str = "assets",
        reference_path: Optional[str] = None,
        file_name: Optional[str] = None,
        recreate: bool = False,
    ) -> None:
        if file_name is None:
            file_name = "config.yaml"
        super().__init__(
            assets_path=assets_path,
            reference_path=reference_path,
            file_name=file_name,
            recreate=recreate,
        )

    def initialize(self, model_dir: Path) -> None:
        org, model, is_base = extract_model_info(model_dir)
        if is_base:
            output_path = Path(self.assets_path) / "base" / self.file_name  # type: ignore
        else:
            if org and model:
                output_path = Path(self.assets_path) / org / model / self.file_name  # type: ignore
            else:
                logger.warning(f"Could not extract organization and model from path: {model_dir}")
                return

        # Skip if file already exists and is not empty
        if output_path.exists() and output_path.stat().st_size > 0:
            if self.recreate:
                logger.warning(f"Config already exists and is not empty for '{output_path}', recreating...")
            else:
                logger.info(f"Config already exists and is not empty for '{output_path}', skipping...")
                return
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if is_base:
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
        logger.info(f"Generated config for {model_dir}")


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
@click.option("--assets_path", type=str, default="assets", help="Path to the assets directory.")
@click.option("--recreate", type=bool, default=False, help="Whether to recreate existing configuration files.")
def init_config(assets_path: str = "assets", recreate: bool = False) -> None:
    ConfigInitializer(assets_path=assets_path, recreate=recreate).initialize_all()


@cli.command(name="get")
@click.argument("key", type=str)
@click.option("--canonical_name", type=str, default=None)
@click.option("--assets_path", type=str, default="assets", help="Path to the assets directory.")
def get_config_value(key: str, canonical_name: Optional[str] = None, assets_path: str = "assets"):
    canonical_name = get_canonical_name(canonical_name)
    config = read_yaml(Path(assets_path) / canonical_name / "config.yaml")
    value = get_value(config, key)
    print(value)


@cli.command(name="get-base-image-ref")
@click.option("--canonical_name", type=str, default=None)
@click.option("--assets_path", type=str, default="assets", help="Path to the assets directory.")
def get_base_image_ref(canonical_name: Optional[str] = None, assets_path: str = "assets"):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
