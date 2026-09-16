# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_runtime.evaluation.backends.base.

The seam itself has almost no behaviour; what matters is that it obliges an
adapter to answer all three questions, and that "ran but scored nothing" stays
expressible.
"""

from pathlib import Path

import pytest

from aim_runtime.evaluation.backends.base import BackendInfo, EvaluationBackend, RawRun
from aim_runtime.evaluation.results import MODALITY_TEXT


class TestEvaluationBackend:
    """Test suite for the backend interface."""

    def test_a_backend_must_answer_all_three_questions(self):
        class Partial(EvaluationBackend):
            name = "partial"

            def probe(self):
                return BackendInfo(name=self.name, available=True)

        with pytest.raises(TypeError, match="run"):
            Partial()  # type: ignore[abstract]

    def test_a_backend_implementing_the_interface_is_constructible(self):
        class Complete(EvaluationBackend):
            name = "complete"

            def probe(self):
                return BackendInfo(name=self.name, available=True)

            def run(self, settings, *, env, output_dir):
                return RawRun()

            def parse(self, raw):
                return []

        assert Complete().name == "complete"

    def test_a_backend_measures_text_unless_it_says_otherwise(self):
        class Segmentation(EvaluationBackend):
            name = "monai"
            modality = "image_3d"

            def probe(self):
                return BackendInfo(name=self.name, available=True)

            def run(self, settings, *, env, output_dir):
                return RawRun()

            def parse(self, raw):
                return []

        assert EvaluationBackend.modality == MODALITY_TEXT
        assert Segmentation().modality == "image_3d"


class TestRawRun:
    """Test suite for the opaque run handle."""

    def test_a_run_with_no_failure_succeeded(self):
        assert RawRun().succeeded is True

    def test_a_run_can_succeed_without_producing_a_document(self):
        """The backend finished, but left nothing to score — not a failure."""
        raw = RawRun()

        assert raw.succeeded is True
        assert raw.document is None

    def test_a_failed_run_carries_its_reason(self):
        raw = RawRun(failure_reason="lm_eval failed with code 1: boom")

        assert raw.succeeded is False

    def test_artifacts_and_extra_default_to_empty(self):
        raw = RawRun(document={"results": {}}, artifacts=(Path("results.json"),))

        assert raw.artifacts == (Path("results.json"),)
        assert raw.extra == {}


class TestBackendInfo:
    """Test suite for the probe's answer."""

    def test_an_available_backend_needs_no_detail(self):
        info = BackendInfo(name="lm-eval", available=True, command="lm_eval")

        assert info.detail is None
        assert info.version is None

    def test_an_unavailable_backend_explains_itself(self):
        info = BackendInfo(name="lm-eval", available=False, command="lm_eval", detail="not found")

        assert info.available is False
        assert info.detail == "not found"
