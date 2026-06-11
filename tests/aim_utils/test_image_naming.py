#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for image naming utility."""

import pytest

from aim_common.object_model import AcceleratorFamily
from aim_utils.image_naming import (
    LEGACY_VLLM_BASE_TARGET_ID,
    ImageName,
    get_base_image_name,
    get_image_name,
    get_model_image_name,
    parse_image_name,
    parse_image_ref,
)


class TestBaseImageNaming:
    """Test base image naming conventions."""

    def test_instinct_base_has_alias(self):
        """Instinct base image has different canonical and public names."""
        name = get_image_name(AcceleratorFamily.INSTINCT.value, is_base=True)
        assert name.canonical == "aim-instinct-base"
        assert name.public == "aim-base"
        assert name.has_alias is True

    def test_epyc_base_no_alias(self):
        """EPYC base image has same canonical and public names."""
        name = get_image_name(AcceleratorFamily.EPYC.value, is_base=True)
        assert name.canonical == "aim-epyc-base"
        assert name.public == "aim-epyc-base"
        assert name.has_alias is False

    def test_radeon_base_no_alias(self):
        """Radeon base image has same canonical and public names."""
        name = get_image_name(AcceleratorFamily.RADEON.value, is_base=True)
        assert name.canonical == "aim-radeon-base"
        assert name.public == "aim-radeon-base"
        assert name.has_alias is False

    def test_case_insensitive(self):
        """Accelerator names are case-insensitive (tested via string API)."""
        name = get_image_name("INSTINCT", is_base=True)
        assert name.canonical == "aim-instinct-base"
        assert name.public == "aim-base"


class TestModelImageNaming:
    """Test model image naming conventions."""

    def test_instinct_model_has_alias(self):
        """Instinct model image has different canonical and public names."""
        name = get_image_name(
            AcceleratorFamily.INSTINCT.value, canonical_name_sanitized="meta-llama-llama-3-1-8b-instruct"
        )
        assert name.canonical == "aim-instinct-meta-llama-llama-3-1-8b-instruct"
        assert name.public == "aim-meta-llama-llama-3-1-8b-instruct"
        assert name.has_alias is True

    def test_epyc_model_no_alias(self):
        """EPYC model image has same canonical and public names."""
        name = get_image_name(AcceleratorFamily.EPYC.value, canonical_name_sanitized="amd-amd-llama-135m")
        assert name.canonical == "aim-epyc-amd-amd-llama-135m"
        assert name.public == "aim-epyc-amd-amd-llama-135m"
        assert name.has_alias is False

    def test_radeon_model_no_alias(self):
        """Radeon model image has same canonical and public names."""
        name = get_image_name(
            AcceleratorFamily.RADEON.value, canonical_name_sanitized="microsoft-phi-3-mini-4k-instruct"
        )
        assert name.canonical == "aim-radeon-microsoft-phi-3-mini-4k-instruct"
        assert name.public == "aim-radeon-microsoft-phi-3-mini-4k-instruct"
        assert name.has_alias is False

    def test_named_target_model_uses_target_qualified_name_no_alias(self):
        """Named-target model image encodes target_id and has no public alias."""
        name = get_model_image_name(
            AcceleratorFamily.INSTINCT.value,
            canonical_name_sanitized="meta-llama-llama-3-1-8b-instruct",
            target_id="bentoml",
        )
        assert name.private == "aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct"
        assert name.public == "aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct"
        assert name.has_alias is False


class TestUnifiedGetImageName:
    """Test unified get_image_name function."""

    def test_deprecation_warning(self):
        """get_image_name warns callers to move to the explicit APIs."""
        with pytest.warns(DeprecationWarning, match="get_image_name"):
            get_image_name("instinct", is_base=True)

    def test_base_image(self):
        """get_image_name works for base images."""
        name = get_image_name("instinct", is_base=True)
        assert name.canonical == "aim-instinct-base"
        assert name.public == "aim-base"

    def test_model_image(self):
        """get_image_name works for model images."""
        name = get_image_name("epyc", canonical_name_sanitized="amd-amd-llama-135m", is_base=False)
        assert name.canonical == "aim-epyc-amd-amd-llama-135m"
        assert name.public == "aim-epyc-amd-amd-llama-135m"

    def test_model_image_requires_canonical_name(self):
        """get_image_name raises error if canonical_name_sanitized missing for model images."""
        with pytest.raises(ValueError, match="canonical_name_sanitized is required"):
            get_image_name("instinct", is_base=False)


class TestPrivateProperty:
    """Test that private returns canonical (no prefix)."""

    def test_private_base_instinct(self):
        name = get_image_name("instinct", is_base=True)
        assert name.private == "aim-instinct-base"

    def test_private_base_epyc(self):
        name = get_image_name("epyc", is_base=True)
        assert name.private == "aim-epyc-base"

    def test_private_model(self):
        name = get_image_name("instinct", canonical_name_sanitized="meta-llama-llama-3-1-8b-instruct")
        assert name.private == "aim-instinct-meta-llama-llama-3-1-8b-instruct"

    def test_private_equals_canonical(self):
        """private is always the same as canonical."""
        name = get_image_name("instinct", is_base=True)
        assert name.private == name.canonical


class TestParseImageName:
    """Test parsing image repository names."""

    def test_parse_instinct_base_full(self):
        """Parse canonical instinct base name."""
        parsed = parse_image_name("aim-instinct-base")
        assert parsed.accelerator == "instinct"
        assert parsed.canonical_name_sanitized is None
        assert parsed.is_base is True

    def test_parse_instinct_base_public(self):
        """Parse public (abbreviated) instinct base name."""
        parsed = parse_image_name("aim-base")
        assert parsed.accelerator == "instinct"
        assert parsed.canonical_name_sanitized is None
        assert parsed.is_base is True

    def test_parse_epyc_base(self):
        """Parse EPYC base name."""
        parsed = parse_image_name("aim-epyc-base")
        assert parsed.accelerator == "epyc"
        assert parsed.is_base is True

    def test_parse_radeon_base(self):
        """Parse Radeon base name."""
        parsed = parse_image_name("aim-radeon-base")
        assert parsed.accelerator == "radeon"
        assert parsed.is_base is True

    def test_parse_instinct_model_internal(self):
        """Parse internal instinct model name."""
        parsed = parse_image_name("aim-instinct-meta-llama-llama-3-1-8b-instruct")
        assert parsed.accelerator == "instinct"
        assert parsed.canonical_name_sanitized == "meta-llama-llama-3-1-8b-instruct"
        assert parsed.is_base is False
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_parse_instinct_model_public(self):
        """Parse public (abbreviated) instinct model name."""
        parsed = parse_image_name("aim-meta-llama-llama-3-1-8b-instruct")
        assert parsed.accelerator == "instinct"
        assert parsed.canonical_name_sanitized == "meta-llama-llama-3-1-8b-instruct"
        assert parsed.is_base is False
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_parse_epyc_model(self):
        """Parse EPYC model name."""
        parsed = parse_image_name("aim-epyc-amd-amd-llama-135m")
        assert parsed.accelerator == "epyc"
        assert parsed.canonical_name_sanitized == "amd-amd-llama-135m"
        assert parsed.is_base is False
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_parse_radeon_model(self):
        """Parse Radeon model name."""
        parsed = parse_image_name("aim-radeon-microsoft-phi-3-mini-4k-instruct")
        assert parsed.accelerator == "radeon"
        assert parsed.canonical_name_sanitized == "microsoft-phi-3-mini-4k-instruct"
        assert parsed.is_base is False
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_parse_named_target_model(self):
        """Parse target-qualified model name."""
        parsed = parse_image_name("aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct")
        assert parsed.accelerator == "instinct"
        assert parsed.canonical_name_sanitized == "meta-llama-llama-3-1-8b-instruct"
        assert parsed.is_base is False
        assert parsed.base_target_id == "bentoml"

    def test_parse_unknown_format(self):
        """Unknown format fails fast instead of returning a synthetic fallback object."""
        with pytest.raises(ValueError, match="Unrecognized image repository format"):
            parse_image_name("unknown-image-name")

    def test_parse_legacy_dev_prefixed_not_recognized(self):
        """Legacy dev- prefixed names are no longer parsed by the naming module."""
        with pytest.raises(ValueError, match="Unrecognized image repository format"):
            parse_image_name("dev-aim-instinct-base")

    def test_parse_model_with_illegal_characters_raises(self):
        """Model suffix with unsupported characters raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized image repository format"):
            parse_image_name("aim-epyc-bad model")


class TestImageNameDataclass:
    """Test ImageName dataclass properties."""

    def test_has_alias_true(self):
        """has_alias is True when canonical and public differ."""
        name = ImageName(canonical="aim-instinct-base", public="aim-base")
        assert name.has_alias is True

    def test_has_alias_false(self):
        """has_alias is False when canonical and public are same."""
        name = ImageName(canonical="aim-epyc-base", public="aim-epyc-base")
        assert name.has_alias is False


class TestParseImageRef:
    """Test parsing full Docker image references."""

    def test_standard_reference(self):
        result = parse_image_ref("ghcr.io/silogen/aim-instinct-base:0.3.0")
        assert result == ("ghcr.io", "silogen", "aim-instinct-base", "0.3.0")

    def test_rc_tag(self):
        result = parse_image_ref("ghcr.io/silogen/aim-base:0.7-rc28")
        assert result == ("ghcr.io", "silogen", "aim-base", "0.7-rc28")

    def test_model_image(self):
        result = parse_image_ref("ghcr.io/silogen/aim-epyc-amd-amd-llama-135m:1.0.0")
        assert result == ("ghcr.io", "silogen", "aim-epyc-amd-amd-llama-135m", "1.0.0")

    def test_colon_heavy_reference_raises(self):
        with pytest.raises(ValueError, match="must not contain ':'"):
            parse_image_ref("ghcr.io/silogen/aim-base:sha256:abc123")

    def test_digest_reference_raises(self):
        with pytest.raises(ValueError, match="Digest references are not supported"):
            parse_image_ref("ghcr.io/silogen/aim-base@sha256:abc123")

    def test_missing_registry_namespace_raises(self):
        with pytest.raises(ValueError, match="Invalid image reference format"):
            parse_image_ref("aim-instinct-base:0.3.0")

    def test_missing_tag_raises(self):
        with pytest.raises(ValueError, match="missing tag"):
            parse_image_ref("ghcr.io/silogen/aim-instinct-base")

    def test_illegal_characters_raise(self):
        with pytest.raises(ValueError, match="may contain only"):
            parse_image_ref("ghcr.io/silo gen/aim-instinct-base:0.3.0")


class TestTargetAwareBaseImageNaming:
    """Test target-aware base image naming for multi-target support."""

    def test_legacy_vllm_target_id_preserves_instinct_alias(self):
        """legacy_vllm target ID keeps existing naming (aim-instinct-base / aim-base)."""
        name = get_image_name("instinct", is_base=True, base_target_id=LEGACY_VLLM_BASE_TARGET_ID)
        assert name.private == "aim-instinct-base"
        assert name.public == "aim-base"
        assert name.has_alias is True

    def test_none_base_target_id_same_as_legacy(self):
        """None base_target_id is identical to legacy_vllm (backward compat)."""
        assert get_image_name("instinct", is_base=True, base_target_id=None) == get_image_name("instinct", is_base=True)

    def test_named_target_instinct_includes_target_id_no_alias(self):
        """Named target on instinct embeds target_id and has no public alias."""
        name = get_image_name("instinct", is_base=True, base_target_id="bentoml")
        assert name.private == "aim-instinct-bentoml-base"
        assert name.public == "aim-instinct-bentoml-base"
        assert name.has_alias is False

    def test_named_target_epyc_includes_target_id(self):
        """Named target on epyc embeds target_id."""
        name = get_image_name("epyc", is_base=True, base_target_id="sglang")
        assert name.private == "aim-epyc-sglang-base"
        assert name.public == "aim-epyc-sglang-base"
        assert name.has_alias is False

    def test_named_target_radeon_includes_target_id(self):
        """Named target on radeon embeds target_id."""
        name = get_image_name("radeon", is_base=True, base_target_id="llamacpp")
        assert name.private == "aim-radeon-llamacpp-base"
        assert name.public == "aim-radeon-llamacpp-base"
        assert name.has_alias is False

    def test_parse_named_target_base_image_instinct(self):
        """Parsing aim-instinct-bentoml-base yields instinct base with base_target_id."""
        parsed = parse_image_name("aim-instinct-bentoml-base")
        assert parsed.accelerator == "instinct"
        assert parsed.is_base is True
        assert parsed.canonical_name_sanitized is None
        assert parsed.base_target_id == "bentoml"

    def test_parse_named_target_base_image_epyc(self):
        """Parsing aim-epyc-sglang-base yields epyc base with base_target_id."""
        parsed = parse_image_name("aim-epyc-sglang-base")
        assert parsed.accelerator == "epyc"
        assert parsed.is_base is True
        assert parsed.canonical_name_sanitized is None
        assert parsed.base_target_id == "sglang"

    def test_parse_legacy_base_has_no_base_target_id(self):
        """Regression: aim-instinct-base parses with base_target_id=LEGACY_VLLM_BASE_TARGET_ID."""
        parsed = parse_image_name("aim-instinct-base")
        assert parsed.accelerator == "instinct"
        assert parsed.is_base is True
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_parse_legacy_public_base_has_no_base_target_id(self):
        """Regression: aim-base parses with base_target_id=LEGACY_VLLM_BASE_TARGET_ID."""
        parsed = parse_image_name("aim-base")
        assert parsed.accelerator == "instinct"
        assert parsed.is_base is True
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_parse_legacy_model_has_legacy_vllm_target_id(self):
        """Regression: legacy model images parse with base_target_id=LEGACY_VLLM_BASE_TARGET_ID."""
        parsed = parse_image_name("aim-instinct-meta-llama-llama-3-1-8b-instruct")
        assert parsed.is_base is False
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_named_target_model_via_compat_wrapper(self):
        """Compatibility wrapper uses target-aware model naming for non-legacy targets."""
        name = get_image_name(
            "instinct",
            canonical_name_sanitized="meta-llama-llama-3-1-8b-instruct",
            is_base=False,
            base_target_id="bentoml",
        )
        assert name.private == "aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct"
        assert name.public == "aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct"
        assert name.has_alias is False

    def test_target_qualified_base_rejects_legacy_vllm(self):
        """aim-instinct-legacy_vllm-base is NOT parsed as a named-target base image.

        The legacy_vllm target uses distinct legacy formats (aim-{acc}-base / aim-base),
        not the target-qualified format. The target-qualified base parser rejects it,
        and it falls through to the model parser (no code path generates this name).
        """
        parsed = parse_image_name("aim-instinct-legacy_vllm-base")
        # NOT parsed as a base image — the target-qualified parser rejects legacy_vllm
        assert parsed.is_base is False
        assert parsed.canonical_name_sanitized == "legacy_vllm-base"

    def test_target_qualified_model_rejects_legacy_vllm(self):
        """aim-instinct-target-legacy_vllm-model-... is NOT parsed as a named-target model.

        The legacy_vllm target uses the legacy naming (aim-{acc}-{model}), not the
        target-qualified format. The target-qualified model parser rejects it,
        and it falls through to the legacy model parser.
        """
        parsed = parse_image_name("aim-instinct-target-legacy_vllm-model-meta-llama-llama-3-1-8b-instruct")
        # NOT parsed as a target-qualified model — falls through to legacy model parser
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID
        assert parsed.canonical_name_sanitized == "target-legacy_vllm-model-meta-llama-llama-3-1-8b-instruct"


class TestGeneratorInputValidation:
    """Verify that name generators reject unsafe inputs."""

    def test_get_base_image_name_rejects_unsafe_target_id(self):
        with pytest.raises(ValueError, match="Invalid base_target_id"):
            get_base_image_name("instinct", "bad target")

    def test_get_base_image_name_rejects_empty_target_id(self):
        with pytest.raises(ValueError, match="Invalid base_target_id"):
            get_base_image_name("instinct", "")

    def test_get_base_image_name_allows_legacy_vllm(self):
        """legacy_vllm passes all validation checks without any special-casing."""
        name = get_base_image_name("instinct", LEGACY_VLLM_BASE_TARGET_ID)
        assert name.canonical == "aim-instinct-base"

    def test_get_base_image_name_allows_safe_target_id(self):
        name = get_base_image_name("instinct", "bentoml")
        assert name.canonical == "aim-instinct-bentoml-base"

    def test_get_model_image_name_rejects_unsafe_canonical_name(self):
        with pytest.raises(ValueError, match="Invalid canonical_name_sanitized"):
            get_model_image_name("instinct", "bad model", LEGACY_VLLM_BASE_TARGET_ID)

    def test_get_model_image_name_rejects_empty_canonical_name(self):
        with pytest.raises(ValueError, match="Invalid canonical_name_sanitized"):
            get_model_image_name("instinct", "", LEGACY_VLLM_BASE_TARGET_ID)

    def test_get_model_image_name_rejects_unsafe_target_id(self):
        with pytest.raises(ValueError, match="Invalid target_id"):
            get_model_image_name("instinct", "meta-llama-llama-3-1-8b-instruct", "bad/target")

    def test_get_model_image_name_allows_legacy_target_id(self):
        name = get_model_image_name("instinct", "meta-llama-llama-3-1-8b-instruct", LEGACY_VLLM_BASE_TARGET_ID)
        assert name.canonical == "aim-instinct-meta-llama-llama-3-1-8b-instruct"

    def test_get_model_image_name_allows_safe_named_target(self):
        name = get_model_image_name("instinct", "meta-llama-llama-3-1-8b-instruct", "bentoml")
        assert name.canonical == "aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct"

    def test_get_base_image_name_rejects_target_prefix(self):
        """target_id starting with 'target-' is reserved for the model image namespace."""
        with pytest.raises(ValueError, match="Must not start with 'target-'"):
            get_base_image_name("instinct", "target-foo")

    def test_get_base_image_name_rejects_target_prefix_compound(self):
        """Compound target-id that would produce a name parseable as a model image is rejected."""
        with pytest.raises(ValueError, match="Must not start with 'target-'"):
            get_base_image_name("instinct", "target-foo-model-bar")


class TestParserNamespaceDisjointness:
    """Verify that the base and model parser namespaces are disjoint by construction.

    The model format reserves the 'target-' prefix in the discriminator segment so that
    no base target_id can produce a name that the model parser would claim first.
    """

    def test_name_that_would_be_ambiguous_is_rejected_at_generation(self):
        """A target_id starting with 'target-' would produce an ambiguous base name; generation rejects it."""
        # aim-instinct-target-foo-model-bar-base would be grabbed by the model parser
        # as target_id=foo, canonical_name_sanitized=bar-base.  The guard prevents this.
        with pytest.raises(ValueError, match="Must not start with 'target-'"):
            get_base_image_name("instinct", "target-foo-model-bar")

    def test_parser_does_not_misparse_reserved_prefix_as_base(self):
        """aim-instinct-target-bentoml-base is NOT parsed as a base image.

        'target-bentoml' starts with 'target-', so _parse_target_qualified_base_image
        rejects the middle segment. The target-qualified model parser also rejects it
        (no '-model-' separator). The legacy model parser then matches 'aim-instinct-'
        and treats 'target-bentoml-base' as a canonical model name. The key property is
        that it is never misidentified as a base image.
        """
        parsed = parse_image_name("aim-instinct-target-bentoml-base")
        assert parsed.is_base is False
        assert parsed.accelerator == "instinct"
        assert parsed.canonical_name_sanitized == "target-bentoml-base"
        assert parsed.base_target_id == LEGACY_VLLM_BASE_TARGET_ID

    def test_genuine_target_qualified_base_is_unaffected(self):
        """A normal target_id like 'bentoml' (no 'target-' prefix) still parses correctly."""
        parsed = parse_image_name("aim-instinct-bentoml-base")
        assert parsed.is_base is True
        assert parsed.accelerator == "instinct"
        assert parsed.base_target_id == "bentoml"

    def test_genuine_target_qualified_model_is_unaffected(self):
        """The model format aim-instinct-target-bentoml-model-... still parses correctly."""
        parsed = parse_image_name("aim-instinct-target-bentoml-model-meta-llama-llama-3-1-8b-instruct")
        assert parsed.is_base is False
        assert parsed.accelerator == "instinct"
        assert parsed.base_target_id == "bentoml"
        assert parsed.canonical_name_sanitized == "meta-llama-llama-3-1-8b-instruct"

    def test_roundtrip_base_name_is_stable(self):
        """generate → parse roundtrip for a named base target is stable."""
        name = get_base_image_name("instinct", "bentoml")
        parsed = parse_image_name(name.canonical)
        assert parsed.is_base is True
        assert parsed.base_target_id == "bentoml"
        assert parsed.accelerator == "instinct"

    def test_roundtrip_model_name_is_stable(self):
        """generate → parse roundtrip for a named model target is stable."""
        name = get_model_image_name("instinct", "meta-llama-llama-3-1-8b-instruct", "bentoml")
        parsed = parse_image_name(name.canonical)
        assert parsed.is_base is False
        assert parsed.base_target_id == "bentoml"
        assert parsed.canonical_name_sanitized == "meta-llama-llama-3-1-8b-instruct"
