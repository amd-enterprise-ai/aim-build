# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the _validate_path_within path-containment helper.

These tests run on CPU with no GPU, no MONAI, and no BentoML dependencies —
the helper lives at module level in service.py and can be imported after
patching out heavy imports.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Lightweight import of service.py:
# Stub out all heavy deps (bentoml, torch, monai, …) before importing so
# the module-level helper _validate_path_within loads without a GPU.
# ---------------------------------------------------------------------------


def _stub_heavy_deps():
    """Insert lightweight stub modules for all imports service.py needs.

    Returns the sys.modules entries this displaced, so the caller can put them
    back. Every assets/**/image/tests directory is collected in one pytest
    session, so a stub left registered here is still there when the next AIM's
    tests are collected -- an importorskip("bentoml") elsewhere then succeeds
    against this stub instead of skipping.
    """
    displaced = {}

    def _make_module(name, **attrs):
        displaced.setdefault(name, sys.modules.get(name))
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    # bentoml — the @bentoml.service and @bentoml.api decorators are used at
    # class-definition time; stub them as pass-through. bentoml.exceptions has
    # to be a real submodule entry: service.py does `from bentoml.exceptions
    # import InvalidArgument`, which a bare ModuleType attribute cannot satisfy.
    bentoml_mod = _make_module("bentoml")
    bentoml_mod.service = lambda *a, **kw: (lambda cls: cls)
    bentoml_mod.api = lambda *a, **kw: (lambda fn: fn)
    exceptions_mod = _make_module("bentoml.exceptions")
    exceptions_mod.InvalidArgument = type("InvalidArgument", (Exception,), {})
    bentoml_mod.exceptions = exceptions_mod

    # torch — used for amp.autocast at module level inside __init__
    torch_mod = _make_module("torch")
    amp_mod = _make_module("torch.amp")
    amp_mod.autocast = lambda *a, **kw: None
    torch_mod.amp = amp_mod
    torch_mod.device = str  # good enough to pass type checks in stubs

    # config — the real file lives inside the service src dir; stub constants.
    _make_module(
        "config",
        CHANNEL_DEF={},
        ENV_PREFIX="WHOLEBODY_CT",
        HIGHRES=True,
        LOGGER=__import__("logging").getLogger("test"),
        OPT_AUTOCAST=False,
        OPT_COMPILE=False,
        OPT_COMPILE_MODE="max-autotune",
        OPT_DYNAMIC=False,
        OPT_SUMMARY={},
        SERVICE_TIMEOUT_SECONDS=1800,
    )

    # pipeline — stub every symbol imported by service.py
    _make_module(
        "pipeline",
        apply_compile=lambda net, **kw: net,
        build_device=lambda: "cpu",
        build_network=lambda dev: None,
        build_postprocessing=lambda pre, out: None,
        build_preprocessing=lambda dev: None,
        checkpoint_filename=lambda hr=True: "models/model.pt",
        ensure_checkpoint=lambda bd, cp: None,
        expected_output_path=lambda od, ip: od + "/out.nii.gz",
        load_checkpoint=lambda net, cp, dev: None,
    )

    # schemas
    _make_module(
        "schemas",
        InferenceResponse=dict,
        PredictResponse=dict,
    )

    # monai sub-modules referenced at import time
    for name in [
        "monai",
        "monai.data",
        "monai.engines",
        "monai.handlers",
        "monai.inferers",
        "monai.utils",
    ]:
        _make_module(
            name,
            Dataset=object,
            ThreadDataLoader=object,
            SupervisedEvaluator=object,
            StatsHandler=object,
            SlidingWindowInferer=object,
            set_determinism=lambda *a, **kw: None,
        )

    return displaced


_displaced_modules = _stub_heavy_deps()

_SRC = pathlib.Path(__file__).parents[2] / "src"

spec = importlib.util.spec_from_file_location("service", str(_SRC / "service.py"))
assert spec is not None and spec.loader is not None, "Could not load service module spec"
_service_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_service_mod)  # type: ignore[union-attr]

_validate_path_within = _service_mod._validate_path_within

# The same class object service.py imported, so pytest.raises matches on identity.
# Read before the stubs are withdrawn below.
InvalidArgument = sys.modules["bentoml.exceptions"].InvalidArgument

# service.py is loaded and holds its own references, so the stubs have done their
# job. Put sys.modules back exactly as it was -- leaving them registered changes
# what later-collected test modules import.
for _name, _original in _displaced_modules.items():
    if _original is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidatePathWithin:
    """_validate_path_within: path-containment assertion."""

    def test_direct_child_is_allowed(self, tmp_path):
        child = tmp_path / "scan.nii.gz"
        child.touch()
        result = _validate_path_within(child, tmp_path, "image_path")
        assert result == child.resolve()

    def test_nested_child_is_allowed(self, tmp_path):
        nested = tmp_path / "subdir" / "scan.nii.gz"
        nested.parent.mkdir(parents=True)
        nested.touch()
        result = _validate_path_within(nested, tmp_path, "image_path")
        assert result == nested.resolve()

    def test_path_equal_to_base_is_allowed(self, tmp_path):
        """The base directory itself should pass (edge case)."""
        result = _validate_path_within(tmp_path, tmp_path, "output_dir")
        assert result == tmp_path.resolve()

    def test_traversal_via_dotdot_is_rejected(self, tmp_path):
        outside = tmp_path / ".." / "escape.nii.gz"
        with pytest.raises(InvalidArgument, match="image_path must be a path under"):
            _validate_path_within(outside, tmp_path, "image_path")

    def test_absolute_path_outside_base_is_rejected(self, tmp_path):
        outside = pathlib.Path("/etc/passwd")
        with pytest.raises(InvalidArgument, match="image_path must be a path under"):
            _validate_path_within(outside, tmp_path, "image_path")

    def test_prefix_collision_is_rejected(self, tmp_path):
        """'/workspace/data2' must not match base '/workspace/data'."""
        base = tmp_path / "data"
        base.mkdir()
        sibling = tmp_path / "data2"
        sibling.mkdir()
        victim = sibling / "file.nii.gz"
        victim.touch()
        with pytest.raises(InvalidArgument, match="image_path must be a path under"):
            _validate_path_within(victim, base, "image_path")

    def test_error_message_includes_label_and_resolved_path(self, tmp_path):
        outside = pathlib.Path("/tmp")
        with pytest.raises(InvalidArgument) as exc_info:
            _validate_path_within(outside, tmp_path, "output_dir")
        msg = str(exc_info.value)
        assert "output_dir" in msg
        assert str(tmp_path.resolve()) in msg
