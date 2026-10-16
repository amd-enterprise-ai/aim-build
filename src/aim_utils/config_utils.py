# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, FrozenSet, Mapping, Optional

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field, field_validator, model_validator

from aim_common.object_model import AcceleratorFamily
from aim_utils.image_naming import LEGACY_VLLM_BASE_TARGET_ID, ImageName, get_base_image_name, parse_image_name
from aim_utils.specialized_utils import enumerate_specialized_base_targets, resolve_base_assets_dir

from .asset_utils import AssetDescriptor, Initializer, assets_path_option
from .dict_utils import get_value
from .file_utils import TomlFileReader
from .version_utils import validate_version_tag
from .yaml_utils import read_yaml, save_yaml

logger = logging.getLogger(__name__)

# Repo root derived from this module's location (src/aim_utils/config_utils.py ->
# parents[2]), mirroring docs_utils._REPO_ROOT. Declared documentation-template
# paths are repo-root-relative, so they are anchored here rather than to the
# process CWD — the validator must give the same result from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# TEMPORARY: base targets EXCLUDED from the generic vLLM model-service smoke test.
# Most lack vLLM entirely; vllm-omni is vLLM-family but serves generative media, so it
# still can't run the harness's hardcoded Llama text smoke test. Excluded bases also
# skip AITER and are validated by their own model pipeline instead. Keyed by base target
# id (the base dir name). This is the single source of truth for the exclusion signal:
# it drives model_service_validation_supported here (which CiBaseImageConfig.run_validation
# mirrors) and aiter_supported in ci/discover_base_build_targets.py.
#
# REGISTER HERE: every new base that can't run the generic vLLM smoke test must add its
# target_id to this set. Otherwise it defaults to "vLLM" and will silently run the
# generic vLLM smoke test (and fail). Remove this set once an engine-agnostic,
# harness-based base smoke test exists — see docs/plans/engine-agnostic-base-validation.md.
NON_VLLM_BASE_TARGET_IDS: frozenset[str] = frozenset(
    {
        "bentoml",
        "example-echo-model",
        "facebookresearch-dinov2-vitl14-embedding",
        "mit-boltz2",
        "monai-pancreas-ct-dints-segmentation",
        "monai-spleen-deepedit-annotation",
        "monai-swinunetr",
        "monai-vista3d",
        "monai-wholebody-ct-segmentation",
        "monai-wholebrainseg-large-unest-segmentation",
        "openfold-openfold3",
        "vllm-omni",
        "wan-ai-wan2-2-t2v-a14b-diffusers",
    }
)


class EvaluationDepsMixin(BaseModel):
    """Declares the evaluation opt-in once, for every config that carries it.

    A base target opts in with ``install_evaluation_deps`` in its
    ``config.yaml``. The flag then travels unchanged: the file schema reads it,
    the target config keeps it, and the CI configs hand it to the build. One
    declaration keeps the four descriptions from drifting apart.
    """

    install_evaluation_deps: bool = Field(
        default=False,
        description=(
            "Whether the build installs requirements/evaluation-requirements.txt into the image's "
            "evaluation virtualenv, which is what makes `aim-runtime evaluate` runnable in-image"
        ),
    )


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

    @field_validator("base_repository")
    @classmethod
    def _validate_base_repository(cls, v: str) -> str:
        # AIM base images that have a public alias must be declared by that public name
        # (e.g. 'aim-base', not 'aim-instinct-base'). The canonical name is the private
        # CI push repository; only the public alias is published for consumers to pull.
        try:
            parsed = parse_image_name(v)
        except ValueError:
            # Upstream base images (pytorch, vllm-openai-rocm, ...) are not AIM-named.
            return v

        if not parsed.is_base:
            return v

        image_name = get_base_image_name(parsed.accelerator, parsed.base_target_id)
        if image_name.has_alias and v == image_name.canonical:
            raise ValueError(f"Invalid base repository: '{v}'. Use the public name '{image_name.public}'.")
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


class TemplatesConfig(BaseModel):
    """Declared, repo-root-relative documentation template paths for a leaf.

    The documentation template is a *declared per-leaf path* instead of a
    hardcoded constant. Every leaf ``config.yaml`` names the Jinja templates its
    docs render from; vanilla vLLM leaves point at the shared defaults under
    ``docs/templates/`` while a specialized base can point at its own. There is
    no fallback chain — a missing key fails validation.
    """

    model_config = ConfigDict(extra="forbid")

    readme: str = Field(..., description="Repo-root-relative path to the README Jinja template")
    deployment: str = Field(..., description="Repo-root-relative path to the DEPLOYMENT Jinja template")


class DocsConfig(BaseModel):
    """Declared, per-leaf documentation-rendering switches.

    ADR-0007 Appendix B: whether a leaf is rendered as a standalone AIM document
    is a *declared per-leaf* property, not a hardcoded engine list. A base image
    that ships no runnable service of its own (e.g. the bentoml base, which only
    does anything once a model AIM supplies ``/workspace/model``) declares
    ``standalone: false`` so README/DEPLOYMENT generation skips it; only the model
    leaves built on it are documented. Defaults to ``true`` so every existing leaf
    keeps rendering without needing a ``docs`` block.
    """

    # Reject unknown keys so a typo (e.g. ``standalon``) fails loudly.
    model_config = ConfigDict(extra="forbid")

    standalone: bool = Field(
        default=True,
        description="Whether this leaf renders its own standalone README/DEPLOYMENT AIM docs",
    )


class BaseImageConfigFile(EvaluationDepsMixin):
    """Top-level YAML schema for a leaf ``config.yaml``.

    Applies to both base and model leaves. ``base_image`` is required, and
    ``install_evaluation_deps`` is inherited from :class:`EvaluationDepsMixin`.

    ``templates`` is intentionally *optional here*, even though every leaf is
    required to declare it, because this same model validates the build critical
    path (``BaseImageConfig.from_yaml_file`` → ``get-base-image-ref`` and the
    base build matrix). Keeping the field optional stops an *absent*
    ``templates:`` block from failing an image *build*; the requirement is
    enforced where it is consumed — the ``validate`` command
    (``python -m aim_utils.config_utils validate``) and
    ``DocsGeneratorSingleAccelerator._leaf_template_path`` — so it stays loud
    (no fallback) there.

    Note this decouples only the *missing* case: a *present-but-malformed*
    ``templates:`` block is still validated by this model, so it will fail an
    image build too, not just docs generation. Keep declarations well-formed, or
    parse only ``base_image`` on the build path if full isolation is needed.

    An optional ``docs`` block carries per-leaf rendering switches
    (e.g. ``standalone``).
    """

    # Extra top-level keys allowed
    model_config = ConfigDict(extra="allow")

    base_image: BaseImageConfig
    templates: Optional[TemplatesConfig] = None
    docs: DocsConfig = Field(default_factory=DocsConfig)

    @model_validator(mode="before")
    @classmethod
    def _warn_about_a_nested_evaluation_opt_in(cls, data: Any) -> Any:
        """Name the correct place when the opt-in sits under ``base_image``.

        ``BaseImageConfig`` ignores a key it does not declare, so the nested flag
        installs nothing and says nothing. The image then serves normally and
        fails only the evaluation check, which is a long way from the typo.
        """
        if isinstance(data, Mapping):
            base_image = data.get("base_image")
            if isinstance(base_image, Mapping) and "install_evaluation_deps" in base_image:
                logger.warning(
                    "'install_evaluation_deps' under 'base_image:' has no effect. "
                    "Move it to the top level of the base config, beside 'base_image:'."
                )
        return data

    @classmethod
    def from_yaml_file(cls, config_path: Path) -> "BaseImageConfigFile":
        """Load and validate a base config file."""
        config_dict = read_yaml(config_path)
        return cls.model_validate(config_dict)


class BaseImageTargetConfig(EvaluationDepsMixin, BaseImageConfig):
    """Base image config extended with a target identifier."""

    target_id: str = Field(..., description="Target identifier (e.g. legacy_vllm, bentoml)")


class CiBaseImageTarget(EvaluationDepsMixin):
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
    model_service_validation_supported: bool = Field(
        default=True, description="Whether the target supports the generic vLLM model-service smoke test"
    )
    integration_validation_supported: bool = Field(
        default=True, description="Whether the target supports hardware integration validation"
    )
    # Empty for standard targets (a pull-able vendor upstream). For specialized
    # targets CI builds Layer 1 from the image/ dir and uses it as the upstream.
    upstream_image_ref: str = Field(..., description="Full upstream image reference")
    # Set only for specialized targets: a non-empty layer1_repository is the single
    # signal that CI must build a Layer 1 image before the base.
    layer1_repository: str = Field(default="", description="Layer 1 specialized image repository")
    layer1_context_path: str = Field(default="", description="Layer 1 build context (the image/ dir)")
    layer1_dockerfile: str = Field(default="", description="Layer 1 Dockerfile path")

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
                {
                    **parsed_config.base_image.model_dump(),
                    "install_evaluation_deps": parsed_config.install_evaluation_deps,
                },
                LEGACY_VLLM_BASE_TARGET_ID,
            )
        }


def _load_base_target_config(config_path: Path, target_id: str) -> BaseImageTargetConfig:
    """Load one target config file and return normalized target config."""
    parsed_config = BaseImageConfigFile.from_yaml_file(config_path)
    return BaseImageTargetConfig(
        target_id=target_id,
        install_evaluation_deps=parsed_config.install_evaluation_deps,
        **parsed_config.base_image.model_dump(),
    )


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
    # Generic CPU has no runner fleet/accelerator model to validate against, so hardware
    # integration validation is intentionally deferred for it.
    integration_validation_supported = acc_lower != "cpu"

    ci_targets: list[CiBaseImageTarget] = []
    seen_target_ids: set[str] = set()
    for target in resolve_base_image_targets(accelerator_family):
        image_name = get_base_image_name(acc_lower, target.target_id)
        model_service_validation_supported = (
            target.target_id not in NON_VLLM_BASE_TARGET_IDS and integration_validation_supported
        )
        ci_targets.append(
            CiBaseImageTarget(
                target_id=target.target_id,
                image_name=image_name,
                dockerfile=dockerfile,
                model_service_validation_supported=model_service_validation_supported,
                integration_validation_supported=integration_validation_supported,
                upstream_image_ref=target.image_ref,
                install_evaluation_deps=target.install_evaluation_deps,
            )
        )
        seen_target_ids.add(target.target_id)

    # Specialized targets (engine-level + model-level image/ directories).
    # These have no fixed upstream: CI builds their Layer 1 image from the
    # image/ Dockerfile and uses it as the parent for the Layer 2 base build.
    for specialized in enumerate_specialized_base_targets(acc_lower):
        if specialized.target_id in seen_target_ids:
            # A named base/ target already claims this id; skip the specialized
            # one so the build matrix never contains duplicate targets that would
            # push to the same image name.
            logger.warning(
                f"Skipping specialized base target '{specialized.target_id}' for '{acc_lower}': "
                "a base/ target already uses this id."
            )
            continue
        specialized_model_service_validation_supported = (
            integration_validation_supported and specialized.target_id not in NON_VLLM_BASE_TARGET_IDS
        )
        ci_targets.append(
            CiBaseImageTarget(
                target_id=specialized.target_id,
                image_name=specialized.base_image_name,
                dockerfile=dockerfile,
                model_service_validation_supported=specialized_model_service_validation_supported,
                integration_validation_supported=integration_validation_supported,
                upstream_image_ref="",
                layer1_repository=specialized.layer1_repository,
                layer1_context_path=specialized.context_path,
                layer1_dockerfile=specialized.dockerfile,
            )
        )
        seen_target_ids.add(specialized.target_id)

    return ci_targets


class CiBaseImageConfig(EvaluationDepsMixin):
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
    # A non-empty layer1_repository is the single signal that this is a specialized
    # target: CI builds Layer 1 from the image/ dir and uses it as the base upstream.
    layer1_repository: str = Field(default="", description="Layer 1 specialized image repository")
    layer1_context_path: str = Field(default="", description="Layer 1 build context (the image/ dir)")
    layer1_dockerfile: str = Field(default="", description="Layer 1 Dockerfile path")

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
            # Public name, not canonical: configs declare the image consumers pull,
            # and the canonical name is the private CI push repository.
            base_repository = image_name.public
            base_registry_namespace = self._registry_namespace
            base_tag = file_reader.read_value("project.version")
            base_registry_host = self._registry_host

        config = {
            "base_image": {
                "base_registry_namespace": base_registry_namespace,
                "base_repository": base_repository,
                "base_tag": base_tag,
                "registry_host": base_registry_host,
            },
            # Every leaf is required to declare ``templates`` — enforced by the
            # ``validate`` command and the validate-config pre-commit hook (the
            # field is optional on ``BaseImageConfigFile`` so a missing block
            # can't gate image builds). Scaffold the shared defaults here;
            # otherwise a freshly ``init``-ed leaf fails validation on its first
            # commit.
            "templates": {
                "deployment": "docs/templates/DEPLOYMENT.md",
                "readme": "docs/templates/README.md",
            },
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
        run_validation=ci_target.model_service_validation_supported,
        upstream_image_ref=ci_target.upstream_image_ref,
        layer1_repository=ci_target.layer1_repository,
        layer1_context_path=ci_target.layer1_context_path,
        layer1_dockerfile=ci_target.layer1_dockerfile,
        install_evaluation_deps=ci_target.install_evaluation_deps,
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


@cli.command(name="resolve-base-assets-dir")
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
    help="Base image target identifier (e.g. legacy_vllm, bentoml, mit-boltz2). Defaults to legacy_vllm.",
)
def resolve_base_assets_dir_command(accelerator_family: str, base_target_id: Optional[str] = None) -> None:
    """Print the assets dir a base build copies its engine config + general profiles from.

    For model-dedicated specialized bases the engine (and thus the source dir) is
    inferred from the model's profiles, so no extra field is needed in the
    specialized image's config.yaml. The printed path contains ``config/`` and
    ``profiles/general/`` for the base Dockerfile to COPY.
    """
    target_id = base_target_id or LEGACY_VLLM_BASE_TARGET_ID
    print(resolve_base_assets_dir(accelerator_family, target_id))


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


# Document template suffixes that can be declared as readme/deployment. Only
# files of these types are subject to the "must be referenced" rule; anything
# else in a leaf ``templates/`` dir (embedded images, data assets, …) is an
# auxiliary that templates pull in without declaring.
_DECLARABLE_TEMPLATE_SUFFIXES = frozenset({".md", ".markdown", ".j2", ".jinja"})
_MODEL_SPECIFIC_DOC_FILENAMES = frozenset({"API.md"})


def is_allowed_template_include(template_file: Path, leaf_templates_dir: Path) -> bool:
    """Check the template's relative path against the include filename allowlist."""
    return template_file.relative_to(leaf_templates_dir).as_posix() in _MODEL_SPECIFIC_DOC_FILENAMES


def _check_unreferenced_local_templates(config_path: Path, templates: TemplatesConfig) -> list[str]:
    """Fail a leaf that ships a local document template its ``templates:`` block ignores.

    ADR-0007 Appendix B forbids a fallback chain: a leaf that carries its own
    ``templates/`` directory must point ``templates.readme`` / ``templates.deployment``
    at every *document* in it. An unreferenced local document is almost always a leaf
    that was given its own README/DEPLOYMENT but forgot to declare it, so it is rejected
    rather than silently rendered from the shared default.

    Files in ``_MODEL_SPECIFIC_DOC_FILENAMES`` are exempt because they supply
    model-specific content to a declared template, not separate output pages.
    For example, ``templates/API.md`` supplies model API content to a declared
    deployment template.
    Two other kinds of files are exempt:
    ``_``-prefixed Jinja partials/includes — whether the ``_`` is on the filename or on
    any parent directory (e.g. ``templates/_partials/foo.md``), since they are pulled in
    by a declared template and never declared themselves — and non-document assets such as
    images (any suffix outside :data:`_DECLARABLE_TEMPLATE_SUFFIXES`).
    """
    leaf_templates_dir = config_path.parent / "templates"
    if not leaf_templates_dir.is_dir():
        return []

    # Declared template paths are repo-root-relative, while the on-disk files are
    # discovered via ``config_path.parent``. Anchor the declared paths to the repo
    # root (module-derived, not the process CWD) and resolve both sides to absolute
    # paths before comparing, so the check gives the same result regardless of the
    # working directory or whether the caller passed a relative or absolute config
    # path (e.g. ``validate --assets_path /abs/path`` from anywhere). ``os.path.normpath``
    # only collapses ``.``/``..``; it does not reconcile the two path bases.
    referenced = {
        (_REPO_ROOT / templates.readme).resolve(),
        (_REPO_ROOT / templates.deployment).resolve(),
    }
    errors: list[str] = []
    for template_file in sorted(leaf_templates_dir.rglob("*")):
        if not template_file.is_file():
            continue
        if is_allowed_template_include(template_file, leaf_templates_dir):
            continue
        # Jinja partials/includes are conventionally ``_``-prefixed and are pulled
        # in by a declared template, so they are never declared themselves. Exempt
        # a ``_``-prefixed *directory* too (e.g. ``templates/_partials/foo.md``),
        # not just the leaf filename, so a whole folder of includes can be grouped.
        if any(part.startswith("_") for part in template_file.relative_to(leaf_templates_dir).parts):
            continue
        # Only enforce the rule for declarable document templates; auxiliary assets
        # (e.g. embedded images) can't be a readme/deployment and are left alone.
        if template_file.suffix.lower() not in _DECLARABLE_TEMPLATE_SUFFIXES:
            continue
        if template_file.resolve() not in referenced:
            errors.append(
                f"{config_path}: local template '{template_file}' is not referenced by "
                f"templates: (readme/deployment) — point templates: at it or remove it."
            )
    return errors


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
            parsed = BaseImageConfigFile.from_yaml_file(config_path)
            # ``templates`` is optional on the schema so a *missing* block can't
            # break image builds (see BaseImageConfigFile), but every leaf is
            # required to declare it. Enforce that here — the docs/validate
            # boundary — so a missing block still fails loudly with no fallback.
            if parsed.templates is None:
                raise ValueError(
                    "missing required 'templates' block (declare 'readme' and 'deployment' "
                    "documentation template paths)"
                )
        except Exception as exc:
            errors.append(f"{config_path}: {exc}")
            continue

        errors.extend(_check_unreferenced_local_templates(config_path, parsed.templates))

    if errors:
        for err in errors:
            click.echo(f"ERROR: {err}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
