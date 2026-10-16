# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import logging
import shutil
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from aim_common import AcceleratorFamily
from aim_common.object_model import CanonicalName, Engine
from aim_utils.asset_utils import (
    AssetDescriptor,
    DefaultAssetManager,
    Initializer,
    infer_accelerator_family_from_path,
    infer_accelerator_family_from_repo,
    infer_engine_from_profiles,
)
from aim_utils.metadata_utils import MetadataManager


@pytest.fixture
def any_model_from_instinct(assets_instinct_path: Path) -> tuple[str, str]:
    """Get any available model from instinct assets (org, model_name) for flexible testing."""
    manager = MetadataManager.for_path(assets_instinct_path)
    descriptors = manager.get_descriptors(skip_base=True, skip_custom=True)
    if descriptors:
        d = descriptors[0]
        return (d.org, d.model_name)
    return ("test", "model")


@pytest.fixture
def accelerator_model_counts(assets_path: Path) -> dict[str, int]:
    """Get actual model counts per accelerator for flexible testing.

    Returns dict mapping accelerator name to count of model-specific descriptors.
    This allows tests to adapt to different test asset compositions without hardcoding counts.
    """
    counts = {}
    for accel_dir in (assets_path / "instinct").parent.iterdir():
        if accel_dir.is_dir() and not accel_dir.name.startswith("."):
            try:
                manager = MetadataManager.for_path(accel_dir)
                descriptors = manager.get_descriptors(skip_base=True, skip_custom=True)
                counts[accel_dir.name] = len(descriptors)
            except Exception:
                # Skip directories that aren't valid accelerator paths or don't have assets
                pass
    return counts


class TestAssetManagerFactoryMethods:
    """Test suite for AssetManager factory methods (for_path and for_root).

    Note: These tests depend on test assets in tests/assets/{instinct,radeon}.
    They test factory behavior (for_path vs for_root) which is independent of
    specific model content, so they're resilient to test asset changes.
    """

    def test_for_path_loads_single_accelerator(self, assets_instinct_path: Path):
        """for_path() should load descriptors only from specified accelerator."""
        manager = MetadataManager.for_path(assets_instinct_path)

        descriptors = manager.get_descriptors(skip_base=True, skip_custom=True)
        assert len(descriptors) > 0, "Test assets must contain at least one model"

    def test_for_path_single_accelerator_scope(self, assets_instinct_path: Path, assets_radeon_path: Path):
        """for_path() should only load from specified accelerator, not others."""
        instinct_manager = MetadataManager.for_path(assets_instinct_path)
        radeon_manager = MetadataManager.for_path(assets_radeon_path)

        instinct_descriptors = instinct_manager.get_descriptors(skip_base=True, skip_custom=True)
        radeon_descriptors = radeon_manager.get_descriptors(skip_base=True, skip_custom=True)

        assert len(instinct_descriptors) > 0
        assert len(radeon_descriptors) > 0

    def test_for_root_aggregates_multiple_accelerators(
        self, assets_instinct_path: Path, assets_radeon_path: Path, assets_path: Path
    ):
        """for_root() should aggregate descriptors from all accelerators."""
        instinct_manager = MetadataManager.for_path(assets_instinct_path)
        radeon_manager = MetadataManager.for_path(assets_radeon_path)
        root_manager = MetadataManager.for_root(assets_path)

        instinct_count = len(instinct_manager.get_descriptors(skip_base=True, skip_custom=True))
        radeon_count = len(radeon_manager.get_descriptors(skip_base=True, skip_custom=True))
        root_count = len(root_manager.get_descriptors(skip_base=True, skip_custom=True))

        assert root_count >= instinct_count + radeon_count, "for_root() should aggregate all accelerators"

    def test_for_root_default_path(self):
        """for_root() should work with default assets path."""
        manager = MetadataManager.for_root()
        descriptors = manager.get_descriptors()
        assert isinstance(descriptors, list)


class TestGetDescriptorsWithCanonicalName:
    """Test suite for get_descriptors() with canonical_name filtering.

    CRITICAL: These tests verify the early-break bug fix where `break` statements
    were removed from get_descriptors(). The key behavior: when using for_root()
    with canonical_name filtering, ALL matching descriptors across accelerators
    should be returned, not just the first match.
    """

    def test_get_descriptors_with_canonical_name_for_path(self, any_model_from_instinct: tuple[str, str]):
        """get_descriptors() with canonical_name should return descriptor when it matches."""
        from aim_utils.metadata_utils import MetadataManager

        org, model_name = any_model_from_instinct
        manager = MetadataManager.for_path("tests/assets/instinct")
        canonical_name = CanonicalName.from_string(f"{org}/{model_name}")

        descriptors = manager.get_descriptors(canonical_name=canonical_name, skip_base=True, skip_custom=True)

        assert len(descriptors) >= 1, "Should find at least one matching descriptor"
        assert all(d.org == org and d.model_name == model_name for d in descriptors)

    def test_get_descriptors_with_canonical_name_for_root_returns_all_matches(self, accelerator_model_counts: dict):
        """
        CRITICAL: get_descriptors() with canonical_name must return ALL matches.

        Regression test for early-break bug: ensures for_root() with canonical_name
        finds model across all accelerators, not just the first one.
        """
        if len(accelerator_model_counts) < 2:
            pytest.skip("Test requires at least 2 accelerators with models")

        from aim_utils.metadata_utils import MetadataManager

        # Get a model from the first accelerator
        first_accel = list(accelerator_model_counts.keys())[0]
        first_accel_path = Path(f"tests/assets/{first_accel}")
        first_manager = MetadataManager.for_path(first_accel_path)
        descriptors = first_manager.get_descriptors(skip_base=True, skip_custom=True)

        if not descriptors:
            pytest.skip(f"No models found in {first_accel}")

        org, model_name = descriptors[0].org, descriptors[0].model_name
        canonical_name = CanonicalName.from_string(f"{org}/{model_name}")

        # Now search with for_root() - should find in ALL accelerators that have it
        root_manager = MetadataManager.for_root("tests/assets")
        root_descriptors = root_manager.get_descriptors(canonical_name=canonical_name, skip_base=True, skip_custom=True)

        # Must find at least in first accelerator, potentially more
        assert len(root_descriptors) >= 1, "Should find at least in first accelerator"

    def test_get_descriptors_no_match_returns_empty_list(self):
        """get_descriptors() should return empty list when canonical_name doesn't match."""
        manager = MetadataManager.for_root("tests/assets")
        canonical_name = CanonicalName.from_string("nonexistent-org/nonexistent-model")

        descriptors = manager.get_descriptors(canonical_name=canonical_name, skip_base=True, skip_custom=True)

        assert descriptors == []

    def test_get_descriptors_with_none_canonical_name_returns_all(self, assets_instinct_path: Path):
        """get_descriptors() with canonical_name=None should return all descriptors."""
        manager = MetadataManager.for_path(assets_instinct_path)

        descriptors = manager.get_descriptors(canonical_name=None, skip_base=True, skip_custom=True)

        assert len(descriptors) > 0

    def test_get_descriptors_with_canonical_name_includes_base_by_default(
        self, any_model_from_instinct: tuple[str, str]
    ):
        """
        Regression test: canonical_name-scoped lookups must still include base/general
        descriptors when skip_base is left at its default (False), matching the
        behavior of the non-canonical_name path.
        """
        org, model_name = any_model_from_instinct
        manager = MetadataManager.for_path("tests/assets/instinct")
        canonical_name = CanonicalName.from_string(f"{org}/{model_name}")

        descriptors = manager.get_descriptors(canonical_name=canonical_name)

        assert any(not d.is_base for d in descriptors), "Model-specific descriptor should still be included"
        assert any(d.is_base for d in descriptors), "Base/general descriptors must be included by default"


class TestEngineSpecificBaseDescriptors:
    """Test suite for built_in_general_engine descriptors.

    Regression tests for the dropped `.exists()` guard: engine-specific base
    descriptors (e.g. bentoml, vllm-omni) must only be created for accelerators
    that actually have that engine's base directory.
    """

    def test_skips_engines_without_base_directory(self, assets_radeon_path: Path):
        """No engine-specific descriptor should be created when its base directory is missing."""
        manager = MetadataManager.for_path(assets_radeon_path)

        assert manager.built_in_general_engine == []

    def test_includes_engines_with_base_directory(self, assets_instinct_path: Path):
        """An engine-specific descriptor should be created when its base directory exists."""
        manager = MetadataManager.for_path(assets_instinct_path)

        engines = {d.engine.value for d in manager.built_in_general_engine}
        assert engines == {"bentoml", "vllm_omni"}
        assert all(d.directory.exists() for d in manager.built_in_general_engine)

    def test_built_in_is_custom_unaffected_by_ancestor_directory_named_custom(
        self, assets_instinct_path: Path, tmp_path: Path
    ):
        """
        Regression test: is_custom must be determined by which directory was
        iterated (built-in vs. custom), not by inspecting path components. A
        checkout located under an ancestor directory literally named "custom"
        must not mislabel built-in descriptors as custom.
        """
        checkout = tmp_path / "custom" / "aim-build-checkout"
        shutil.copytree(assets_instinct_path, checkout)

        manager = MetadataManager.for_path(checkout)

        assert manager.built_in_model_specific, "Test setup must produce built-in descriptors"
        assert all(not d.is_custom for d in manager.built_in_model_specific)


class TestCustomGeneralAndEngineDescriptors:
    """Test suite for custom_general and custom_general_engine descriptors.

    Regression tests for the missing `.exists()` guard on `custom_general` and
    the missing per-engine enumeration under the custom tree: both must mirror
    the built-in `built_in_general`/`built_in_general_engine` behavior.
    """

    def test_custom_general_skipped_when_directory_missing(self, tmp_path: Path):
        """No custom_general descriptor should be created when custom/ is absent."""
        manager = MetadataManager.for_path(tmp_path)

        assert manager.custom_general == []
        assert manager.custom_general_engine == []

    def test_custom_general_included_when_directory_exists(self, assets_instinct_path: Path):
        """A custom_general descriptor should be created when custom/ exists."""
        manager = MetadataManager.for_path(assets_instinct_path)

        assert len(manager.custom_general) == 1
        assert manager.custom_general[0].directory == assets_instinct_path / "custom"
        assert manager.custom_general[0].is_custom is True
        assert manager.custom_general_engine == [], "Fixture has no engine subdirectories under custom/"

    def test_custom_general_engine_includes_engines_with_custom_directory(
        self, assets_instinct_path: Path, tmp_path: Path
    ):
        """An engine-specific descriptor should be created for a custom engine override directory."""
        checkout = tmp_path / "instinct"
        shutil.copytree(assets_instinct_path, checkout)
        (checkout / "custom" / "bentoml").mkdir()

        manager = MetadataManager.for_path(checkout)

        engines = {d.engine.value for d in manager.custom_general_engine}
        assert engines == {"bentoml"}
        assert all(d.is_custom for d in manager.custom_general_engine)
        assert all(d.directory.exists() for d in manager.custom_general_engine)


class TestInitializer:
    class TestImplementation(Initializer):
        def initialize(self, descriptor: AssetDescriptor) -> None:
            pass

        def get_reference_descriptors(self) -> List[AssetDescriptor]:
            return [
                AssetDescriptor(
                    is_base=False,
                    is_custom=False,
                    directory=Path(self.assets_path) / "meta-llama" / "Llama-3.1-8B-Instruct",
                    org="meta-llama",
                    model_name="Llama-3.1-8B-Instruct",
                )
            ]

    def test_initializer_creation(self, assets_path):
        initializer = TestInitializer.TestImplementation(assets_path=str(assets_path))
        assert initializer.assets_path == str(assets_path)
        assert initializer.file_name is None
        assert initializer.recreate is False

    def test_initializer_custom_params(self, assets_path):
        initializer = TestInitializer.TestImplementation(
            assets_path=str(assets_path),
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


# ---------------------------------------------------------------------------
# Tests for infer_accelerator_family_from_path
# ---------------------------------------------------------------------------


class TestInferAcceleratorFamilyFromPath:

    def test_none_path_raises(self):
        with pytest.raises(Exception):
            infer_accelerator_family_from_path(None)  # type: ignore[arg-type]

    def test_path_without_assets_segment_returns_default(self):
        result = infer_accelerator_family_from_path(Path("/home/user/models/instinct"))
        assert result == AcceleratorFamily.INSTINCT

    def test_instinct_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/instinct"))
        assert result == AcceleratorFamily.INSTINCT

    def test_radeon_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/radeon"))
        assert result == AcceleratorFamily.RADEON

    def test_epyc_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/epyc"))
        assert result == AcceleratorFamily.EPYC

    def test_cpu_assets_path(self):
        result = infer_accelerator_family_from_path(Path("assets/cpu"))
        assert result == AcceleratorFamily.CPU

    def test_nested_instinct_path(self):
        result = infer_accelerator_family_from_path(Path("assets/instinct/google/gemma-3-1b-it"))
        assert result == AcceleratorFamily.INSTINCT

    def test_nested_radeon_path(self):
        result = infer_accelerator_family_from_path(Path("assets/radeon/meta-llama/llama-3-8b"))
        assert result == AcceleratorFamily.RADEON

    def test_absolute_path_with_assets(self):
        result = infer_accelerator_family_from_path(Path("/home/user/aim-build/assets/epyc"))
        assert result == AcceleratorFamily.EPYC

    def test_assets_only_path_returns_default_instinct(self):
        # Path has "assets" as its only component: no subfolder follows
        result = infer_accelerator_family_from_path(Path("assets"))
        assert result == AcceleratorFamily.INSTINCT

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            infer_accelerator_family_from_path(Path("assets/unknown"))


# ---------------------------------------------------------------------------
# Tests for infer_accelerator_family_from_repo
# ---------------------------------------------------------------------------


class TestInferAcceleratorFamilyFromRepo:

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            infer_accelerator_family_from_repo("")

    def test_aim_base_returns_instinct(self):
        assert infer_accelerator_family_from_repo("aim-base") == AcceleratorFamily.INSTINCT

    def test_aim_epyc_base_returns_epyc(self):
        assert infer_accelerator_family_from_repo("aim-epyc-base") == AcceleratorFamily.EPYC

    def test_aim_radeon_base_returns_radeon(self):
        assert infer_accelerator_family_from_repo("aim-radeon-base") == AcceleratorFamily.RADEON

    def test_aim_cpu_base_returns_cpu(self):
        assert infer_accelerator_family_from_repo("aim-cpu-base") == AcceleratorFamily.CPU

    def test_aim_instinct_model_repo_returns_instinct(self):
        assert infer_accelerator_family_from_repo("aim-instinct-google-gemma-3-1b-it") == AcceleratorFamily.INSTINCT

    def test_aim_epyc_model_repo_returns_epyc(self):
        assert infer_accelerator_family_from_repo("aim-epyc-meta-llama-3-8b") == AcceleratorFamily.EPYC

    def test_aim_radeon_model_repo_returns_radeon(self):
        assert infer_accelerator_family_from_repo("aim-radeon-meta-llama-3-8b") == AcceleratorFamily.RADEON

    def test_unknown_family_raises(self):
        assert infer_accelerator_family_from_repo("aim-unknown-base") == AcceleratorFamily.INSTINCT


class TestInferEngineFromProfiles:

    def _write_profile(self, asset_dir: Path, name: str, engine: str, primary: bool = False) -> None:
        profiles_dir = asset_dir / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "aim_id: acme/model",
            "model_id: acme/model",
            "metadata:",
            "  accelerator_count: 1",
            "  accelerator_model: MI300X",
            "  accelerator_type: gpu",
            f"  engine: {engine}",
            "  metric: latency",
            "  precision: fp16",
            "  type: optimized",
        ]
        if primary:
            lines.append("  primary: true")
        (profiles_dir / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_no_profiles_directory_returns_vllm(self, tmp_path):
        assert infer_engine_from_profiles(tmp_path / "model") == Engine.VLLM

    def test_empty_profiles_directory_returns_vllm(self, tmp_path):
        (tmp_path / "profiles").mkdir()
        assert infer_engine_from_profiles(tmp_path) == Engine.VLLM

    def test_bentoml_profiles_return_bentoml(self, tmp_path):
        self._write_profile(tmp_path, "bentoml-mi300x-fp16-tp1-latency", "bentoml")
        assert infer_engine_from_profiles(tmp_path) == Engine.BENTOML

    def test_vllm_profiles_return_vllm(self, tmp_path):
        self._write_profile(tmp_path, "mi300x-fp16-tp1-latency", "vllm")
        assert infer_engine_from_profiles(tmp_path) == Engine.VLLM

    def test_vllm_omni_profiles_return_vllm_omni(self, tmp_path):
        self._write_profile(tmp_path, "mi300x-fp16-tp1-latency", "vllm_omni")
        assert infer_engine_from_profiles(tmp_path) == Engine.VLLM_OMNI

    def test_primary_profile_decides(self, tmp_path):
        self._write_profile(tmp_path, "a-first-by-name", "vllm")
        self._write_profile(tmp_path, "z-last-by-name", "bentoml", primary=True)
        assert infer_engine_from_profiles(tmp_path) == Engine.BENTOML

    def test_first_by_name_decides_without_primary(self, tmp_path):
        self._write_profile(tmp_path, "a-first-by-name", "bentoml")
        self._write_profile(tmp_path, "z-last-by-name", "vllm")
        assert infer_engine_from_profiles(tmp_path) == Engine.BENTOML

    def test_unknown_engine_returns_vllm(self, tmp_path):
        self._write_profile(tmp_path, "mi300x-fp16-tp1-latency", "not-an-engine")
        assert infer_engine_from_profiles(tmp_path) == Engine.VLLM

    def test_unknown_engine_logs_warning_naming_the_value(self, tmp_path, caplog):
        self._write_profile(tmp_path, "mi300x-fp16-tp1-latency", "not-an-engine")
        with caplog.at_level(logging.WARNING):
            infer_engine_from_profiles(tmp_path)

        assert "not-an-engine" in caplog.text

    def test_primary_profile_with_no_engine_does_not_decide(self, tmp_path):
        """Skip profiles without an engine, including primary profiles."""
        self._write_profile(tmp_path, "a-first-by-name", "bentoml")
        profiles_dir = tmp_path / "profiles"
        (profiles_dir / "z-last-by-name.yaml").write_text(
            "aim_id: acme/model\n"
            "model_id: acme/model\n"
            "metadata:\n"
            "  accelerator_count: 1\n"
            "  accelerator_model: MI300X\n"
            "  accelerator_type: gpu\n"
            "  metric: latency\n"
            "  precision: fp16\n"
            "  primary: true\n"
            "  type: optimized\n",
            encoding="utf-8",
        )
        assert infer_engine_from_profiles(tmp_path) == Engine.BENTOML

    def test_result_is_cached_per_directory(self, tmp_path):
        """Return the cached engine even if the profile changes."""
        try:
            self._write_profile(tmp_path, "mi300x-fp16-tp1-latency", "bentoml")
            assert infer_engine_from_profiles(tmp_path) == Engine.BENTOML

            self._write_profile(tmp_path, "mi300x-fp16-tp1-latency", "vllm")
            assert infer_engine_from_profiles(tmp_path) == Engine.BENTOML
        finally:
            infer_engine_from_profiles.cache_clear()


class TestModelSpecificDescriptorEngine:
    """The engine of a model-specific descriptor comes from its own profiles."""

    def _make_assets_tree(self, root: Path, org: str, model: str, engine: str) -> Path:
        model_dir = root / "instinct" / org / model
        profiles_dir = model_dir / "profiles"
        profiles_dir.mkdir(parents=True)
        (profiles_dir / "profile.yaml").write_text(
            "aim_id: {org}/{model}\n"
            "model_id: {org}/{model}\n"
            "metadata:\n"
            "  accelerator_count: 1\n"
            "  accelerator_model: MI300X\n"
            "  accelerator_type: gpu\n"
            "  engine: {engine}\n"
            "  metric: latency\n"
            "  precision: fp16\n"
            "  type: optimized\n".format(org=org, model=model, engine=engine),
            encoding="utf-8",
        )
        return model_dir

    def test_bentoml_asset_descriptor_has_bentoml_engine(self, tmp_path):
        self._make_assets_tree(tmp_path, "openfold", "openfold3", "bentoml")

        descriptors = DefaultAssetManager.for_path(tmp_path / "instinct").get_descriptors(skip_base=True)

        assert [d.engine for d in descriptors] == [Engine.BENTOML]

    def test_asset_without_profiles_has_vllm_engine(self, tmp_path):
        (tmp_path / "instinct" / "google" / "gemma-3-1b-it").mkdir(parents=True)

        descriptors = DefaultAssetManager.for_path(tmp_path / "instinct").get_descriptors(skip_base=True)

        assert [d.engine for d in descriptors] == [Engine.VLLM]
