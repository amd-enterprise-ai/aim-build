# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Per-model ModelHarness for the Pancreas + Tumor DiNTS 3D CT segmentation AIM.

The bulk of the implementation lives in ``monai_harness.MonaiHarness`` (the
shared base used by every MONAI bundle AIM). This file declares the leaf
class with model-specific metadata and overrides ``_compute_metric`` so
Tuner-side ``evaluate`` produces the bundle's authoritative per-class mean
Dice (over classes {1=pancreas, 2=pancreatic tumor}) instead of the default
binary ``voxel > 0`` Dice.

Why ``import monai_harness`` (not ``from monai_harness import MonaiHarness``)?

    ``aim-runtime`` discovers harnesses via
    ``inspect.getmembers(harness_module)`` and picks the alphabetically-first
    ``ModelHarness`` subclass it finds. If ``MonaiHarness`` were a direct
    attribute of this module (via ``from … import``) it would shadow the
    per-model leaf class. Using a module import keeps ``MonaiHarness`` out
    of this module's namespace.

Why a ``_compute_metric`` override?

    metadata.json declares ``label_classes = "single channel data, 1 is
    pancreas, 2 is pancreatic tumor, 0 is everything else"`` and the bundle's
    reported ``eval_metrics.mean_dice = 0.62`` is computed as the **per-class
    mean** of Dice over class 1 (pancreas) and class 2 (pancreatic tumor) on
    the bundle's validation split. The default ``MonaiHarness._compute_dice``
    is **binary**: ``pred > 0`` vs ``label > 0``, which collapses pancreas +
    tumor into a single foreground mask and yields a much higher Dice
    (~0.83 on a 20-volume validation cohort using binary Dice — the two
    numbers are both valid but not the same metric).

    To match the bundle's published number, the leaf harness computes
    Dice separately on ``pred == 1`` vs ``label == 1`` (pancreas) and on
    ``pred == 2`` vs ``label == 2`` (pancreatic tumor) and returns the mean.
    ``pipeline.build_postprocessing`` already runs ``argmax`` over the
    network's 3-channel one-hot output, so the on-disk ``pred.nii.gz``
    voxels carry values in {0, 1, 2} and the per-class masks are clean
    binary comparisons.

``_pair_samples`` does NOT need an override: MSD Task07 Pancreas uses the
same basename in ``imagesTr/`` and ``labelsTr/`` (``pancreas_NNN.nii.gz`` on
both sides), so the default filename-match pairing in
``MonaiHarness._pair_samples`` works as-is. See ``tests/unit/test_harness_pancreas.py``
which exercises the pairing against the real naming convention.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import monai_harness
from constants import BUNDLE_NAME

from aim_runtime.harness import CheckInfo, CheckResultType, CheckScope

# Class indices that contribute to the per-class mean Dice. Must match
# ``config.FOREGROUND_CLASS_INDICES``; duplicated here so the harness has no
# import-time dependency on ``config`` (the harness ships in the AIM image
# and is also imported by Builder-side host tests where ``config`` would
# pull in torch + torch._dynamo unnecessarily).
_FOREGROUND_CLASS_INDICES = (1, 2)


class PancreasDintsHarness(monai_harness.MonaiHarness):
    """Harness for the Pancreas and Tumor DiNTS Segmentation AIM."""

    MODEL_ID: ClassVar[str] = f"monai/{BUNDLE_NAME}"

    # The base class' default METRIC_NAME = "dice" is correct — we still
    # compute Dice, just per-class then averaged. Keep the name so Tuner's
    # aggregate stats stay byte-identical with other segmentation AIMs.
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
            "Per-class mean Dice over (pancreas, pancreatic tumor) on the "
            "MSD Task07 evaluate split. Bundle's published target is 0.62.",
        ),
    ]

    def _compute_metric(self, prediction: Any, label: Any) -> float:
        """Per-class mean Dice over classes {1 (pancreas), 2 (pancreatic tumor)}.

        This replaces the base ``MonaiHarness._compute_metric`` which is
        binary (pred > 0 vs label > 0). For a 3-channel one-hot segmentation
        like this AIM, the per-class mean is the metric the bundle's
        ``metadata.json:eval_metrics.mean_dice = 0.62`` is reported on, so
        the override keeps Tuner-side evaluate numbers comparable with the
        bundle's published number (and with NVIDIA's reference runs).

        ``prediction`` and ``label`` are both NIfTI file paths whose voxels
        carry class indices in {0, 1, 2} after ``AsDiscreted(argmax=True)``
        in postprocessing.
        """
        if not isinstance(prediction, str) or not prediction:
            raise ValueError(
                f"PancreasDintsHarness._compute_metric expects a file path "
                f"prediction, got {type(prediction).__name__}={prediction!r}"
            )
        if not os.path.isfile(prediction):
            raise ValueError(f"Prediction file not on disk: {prediction}")
        return self._per_class_mean_dice(
            prediction,
            label,
            class_indices=_FOREGROUND_CLASS_INDICES,
        )

    @staticmethod
    def _per_class_mean_dice(
        pred_path: str,
        label_path: str,
        *,
        class_indices: tuple[int, ...],
    ) -> float:
        """Mean Dice over per-class binary masks.

        For each ``c`` in ``class_indices``, compute Dice on
        ``(pred == c)`` vs ``(label == c)``. Return the simple mean of the
        per-class scores. Empty/empty per-class pairs contribute 1.0 (same
        convention as the base ``_compute_dice``). If every class is
        empty/empty, the overall mean is 1.0.

        Pancreatic tumor (class 2) is the small, hard class — most of the
        signal in the bundle's 0.62 number comes from class 1 being
        relatively easy and class 2 being relatively hard. The per-class
        mean exposes that tradeoff; binary Dice does not.
        """
        import nibabel as nib  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        pred = nib.load(pred_path).get_fdata()
        gt = nib.load(label_path).get_fdata()
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch pred={pred.shape} vs gt={gt.shape} "
                f"— resampling produced a different grid than the GT."
            )

        scores: list[float] = []
        for c in class_indices:
            pred_c = pred == c
            gt_c = gt == c
            p_sum = float(pred_c.sum())
            g_sum = float(gt_c.sum())
            denom = p_sum + g_sum
            if denom == 0.0:
                scores.append(1.0)
            else:
                scores.append(2.0 * float(np.logical_and(pred_c, gt_c).sum()) / denom)

        return float(sum(scores) / len(scores))
