# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Per-model ModelHarness for the MONAI VISTA-3D AIM.

The bulk of the implementation lives in ``monai_harness.MonaiHarness`` (the
shared base used by every MONAI bundle AIM). This file declares the leaf
class with VISTA-3D-specific metadata and a per-sample Dice metric that
handles the VISTA-3D class-conditional semantic-mask output.

Output shape:
    VISTA-3D writes a per-case ``*_trans.nii.gz`` where voxel values are
    integer class IDs from the bundle's 133-entry channel_def
    (0 = background, 1..132 = anatomical labels). When the request supplies
    ``label_prompt=[3]`` (spleen), the predicted volume contains values
    in ``{0, 3}``; for ``label_prompt=[3, 5]`` it would contain
    ``{0, 3, 5}``; in automatic mode it contains all the ``everything_labels``
    IDs the bundle detected.

Per-sample metric:
    Binary Dice on ``>0`` masks. This matches the MSD-Spleen evaluation
    convention (single-class GT, class-conditional pred) and reduces
    cleanly to the right number when ``label_prompt`` is a single class
    and the GT is binary. For multi-class prompts against a multi-class
    GT, swap in a class-conditional per-ID Dice — see the override stub
    in ``_compute_metric``.

Image⇄label pairing:
    The MSD-Spleen evaluation split (and every other Decathlon CT/MR task)
    ships ``imagesTs/<case>.nii.gz`` + ``labelsTs/<case>.nii.gz`` with
    matching basenames, so the base-class filename-match
    ``_pair_samples`` works unchanged.

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


class Vista3dHarness(monai_harness.MonaiHarness):
    """Harness for the MONAI VISTA-3D Foundation Model AIM."""

    MODEL_ID: ClassVar[str] = f"monai/{BUNDLE_NAME}"

    INPUT_GLOB: ClassVar[str] = "*.nii.gz"

    METRIC_NAME: ClassVar[str] = "dice"

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
            "Mean per-case binary Dice on the prompted class (target ≥ 0.952, VISTA-3D paper Table 1, automatic mode)",
        ),
    ]

    def _compute_metric(self, prediction: Any, label: Any) -> float:
        """Per-sample binary Dice on ``>0`` masks.

        VISTA-3D writes class IDs as voxel values; the MSD-Spleen GT is
        binary ``{0, 1}``. Binarising both via ``>0`` reduces this to a
        standard single-class Dice — correct for the spleen-only eval
        contract this AIM ships with (``label_prompt=[3]``).

        Multi-class extension (not currently exercised — included as a
        documented stub for the next reviewer):

            class-conditional Dice for ``label_prompt=[3, 5]`` would
            compute one Dice per ID against a multi-class GT and average,
            e.g. ``mean({Dice(pred==id, gt==id) for id in label_prompt})``.
            That requires the harness to know which class IDs were
            requested (not currently surfaced into ``_compute_metric``),
            so the override would route on a profile field set by Tuner.
            For now the spleen-only single-class evaluation is the
            canonical contract.

        Both ``prediction`` and ``label`` are NIfTI file paths — the
        prediction is ``SaveImaged`` output from the service, the label
        was resolved by the base class's filename-match ``_pair_samples``.
        """
        import os

        if not isinstance(prediction, str) or not prediction:
            raise ValueError(
                f"_compute_metric expects a file path prediction, " f"got {type(prediction).__name__}={prediction!r}",
            )
        if not os.path.isfile(prediction):
            raise ValueError(f"Prediction file not on disk: {prediction}")
        return self._compute_dice(prediction, label)
