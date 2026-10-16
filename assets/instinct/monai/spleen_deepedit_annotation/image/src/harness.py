# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Per-model ModelHarness for the Spleen DeepEdit AIM.

The bulk of the implementation lives in ``monai_harness.MonaiHarness`` (the
shared base used by every MONAI bundle AIM). This file only declares the
leaf class with model-specific metadata.

DeepEdit-specific notes
-----------------------
The Tuner Step 6 evaluate path can drive this AIM in either "automatic
mode" (no clicks → empty heatmaps) or "interactive mode" (user-supplied
clicks). The default ``build_inference_payload`` on ``MonaiHarness``
sends only ``image_path`` to the service, which lands us in automatic
mode. **This is the project policy: interactive-segmentation AIMs
(DeepEdit, DeepGrow, MONAI Label) are benchmarked in automatic mode
only** — interactive mode is a runtime user surface, not a Tuner one.

The bundle's published 0.97 mean-Dice anchor (bundle README
§Performance) is reported "depending on the number of simulated
clicks" — i.e. **with clicks**, NOT automatic mode. The closest
published click-free anchor for DeepEdit on a comparable spleen task
is 0.835 Dice on BTCV single-label spleen (Diaz-Pinto et al. 2022,
DALI 2022, Table 3, DeepEdit-0.5 = same 50% click-free training mix
as the bundle's ``deepgrow_probability_train``; arXiv:2305.10655).
Our automatic-mode Dice on MSD Task09 Spleen is expected to fall
**between** those two anchors (above the BTCV click-free 0.835,
below the with-clicks 0.97). A delta from 0.97 on the
automatic-mode benchmark is an operating-mode distinction, not
platform numeric drift.

The Decathlon Spleen test/training split uses identical basenames
(``imagesTs/spleen_NN.nii.gz`` ↔ ``labelsTr/spleen_NN.nii.gz``) so the
shared ``_pair_samples`` default works without an override. Spleen
segmentation is binary single-class so the shared binary-Dice metric in
``_compute_dice`` matches the bundle's published math.

Why ``import monai_harness`` (not ``from monai_harness import MonaiHarness``)?

    ``aim-runtime`` discovers harnesses via
    ``inspect.getmembers(harness_module)`` and picks the alphabetically-first
    ``ModelHarness`` subclass it finds. If ``MonaiHarness`` were a direct
    attribute of this module (via ``from … import``) it would shadow the
    per-model leaf class. Using a module import keeps ``MonaiHarness`` out
    of this module's namespace.
"""

from __future__ import annotations

from typing import Any, ClassVar

import monai_harness
from constants import BUNDLE_NAME

from aim_runtime.harness import CheckInfo, CheckResultType, CheckScope


class SpleenDeepeditHarness(monai_harness.MonaiHarness):
    """Harness for the Spleen DeepEdit Interactive Segmentation AIM."""

    ENGINE: ClassVar[str] = "bentoml"
    MODEL_ID: ClassVar[str] = f"monai/{BUNDLE_NAME}"

    CHECKS: ClassVar[list[CheckInfo]] = [
        CheckInfo(
            "bentoml_health",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "BentoML readiness (GET /healthz returns 200)",
        ),
        CheckInfo(
            "predict_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "POST /v1/inference returns valid InferenceResponse with output_path",
        ),
        CheckInfo(
            "p50_latency_ms",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Median /v1/inference latency over num_requests requests (ms)",
        ),
        CheckInfo(
            "throughput_img_per_sec",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Measured throughput in images/sec from successful requests",
        ),
        CheckInfo(
            "success_rate",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Fraction of evaluate samples that scored without error",
        ),
        CheckInfo(
            "mean_dice",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Mean binary Dice on Decathlon Task09 Spleen labelled split "
            "in automatic mode (zero clicks, project policy for "
            "interactive AIMs). Bundle's 0.97 anchor is with-clicks per "
            "README §Performance; closest published click-free anchor "
            "is 0.835 on BTCV (Diaz-Pinto 2022, DALI Table 3).",
        ),
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.MODEL_ID,
            "checks": [
                {
                    "name": c.name,
                    "result_type": c.result_type.value,
                    "scope": c.scope.value,
                    "description": c.description,
                }
                for c in self.list_checks()
            ],
        }
