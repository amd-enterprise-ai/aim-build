# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Per-model ModelHarness for the wholebody_ct_segmentation AIM.

The bulk of the implementation lives in ``monai_harness.MonaiHarness`` (the
shared base used by every MONAI bundle AIM). This file declares the leaf
class with model-specific metadata and overrides ``_compute_metric`` so
Tuner-side ``evaluate`` returns the bundle's per-class mean Dice (over the
104 foreground anatomical structures) instead of the base-class binary Dice.

Why ``import monai_harness`` (not ``from monai_harness import MonaiHarness``)?

    ``aim-runtime`` discovers harnesses via
    ``inspect.getmembers(harness_module)`` and picks the alphabetically-first
    ``ModelHarness`` subclass it finds. If ``MonaiHarness`` were a direct
    attribute of this module (via ``from … import``) it would shadow the
    per-model leaf class. Using a module import keeps ``MonaiHarness`` out
    of this module's namespace.

Bundle metric (per-class mean Dice over 104 foreground classes)
---------------------------------------------------------------

The bundle's ``configs/metadata.json`` declares
``eval_metrics.mean_dice = 0.8``; combined with the bundle's 105-channel
output (background + 104 TotalSegmentator anatomical structures) this is
the **per-class mean Dice** computed over the 104 foreground class indices
on the bundle's TotalSegmentator validation split. The base
``MonaiHarness._compute_metric`` returns binary Dice on ``(pred > 0)`` vs
``(label > 0)``, which would silently inflate the number for a 105-class
bundle (KB
``compute-metric-override-for-multi-class-or-multi-channel-bundle-dice``);
we override on the leaf to mirror the bundle's math.

Eval-data status (HIGH severity human_review entries)
-----------------------------------------------------

The bundle is trained + validated on the **TotalSegmentator** dataset
(Wasserthal et al. 2022). Reproducing the published 0.8 mean Dice on
public data has two practical obstacles:

1. **Sample dataset is not programmatically fetchable.** The bundle's own
   sample set (``imagesTr``/``imagesTs`` + merged 0-104 labels) is hosted
   on Google Drive behind interactive download links (bundle README
   §"Data"), not on a registry the Builder can pull non-interactively.

2. **Labels need a merge step.** The public TotalSegmentator release
   (Zenodo, CC-BY-4.0) ships one binary NIfTI per structure; the bundle
   expects a single merged NIfTI per case whose voxels carry class indices
   ``{0, …, 104}`` (bundle README §"Data": "merge all annotated labels
   into one NIFTI file"). No pre-merged public mirror was located.

Consequences (stated truthfully — no fabricated accuracy claims):

* ``data/labelsTs/`` is intentionally absent. Staging merged-but-unverified
  GT here would silently produce meaningless Dice numbers.
* The ``_compute_metric`` override below is preserved (correct math per
  bundle) but is **dormant** under the current build: Tuner Step 6
  (``evaluate``) will surface "No (image, label) pairs found" until a
  label-aligned eval split is staged under ``data/labelsTs/``. The harness
  is correct; the data is what's missing.
* ``_FOREGROUND_CLASS_INDICES`` covers ``range(1, 105)`` (excluding
  background 0) — the structural-correctness contract for the bundle's
  105-channel spatial output.
* Re-introducing real per-class Dice is a *single* change at staging time:
  drop matching-basename merged labels into ``data/labelsTs/`` and Tuner
  Step 6 starts producing the bundle's metric without further code changes.
  See ``configs/metadata.json`` in the bundle for the labels schema.

The bundle's own ``intended_use`` is explicit: "This is an example, not to
be used for diagnostic purposes."

Pairing (``_pair_samples``)
---------------------------

Default basename match: ``imagesTs/<case>.nii.gz`` ↔
``labelsTs/<case>.nii.gz``. The TotalSegmentator sample cases share a
basename between image and merged label, so the inherited
``MonaiHarness._pair_samples`` matches if/when labels are staged — no
override needed.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import monai_harness
from constants import BUNDLE_NAME

from aim_runtime.harness import CheckInfo, CheckResultType, CheckScope

# Foreground class indices contributing to the bundle's published mean Dice.
# Bundle's metadata.json:network_data_format.outputs.pred.channel_def maps
# {0: background, 1..104: TotalSegmentator anatomical structures}. The
# bundle's eval_metrics.mean_dice = 0.8 is the per-class mean over indices
# 1-104.
#
# Duplicated here (instead of importing from ``config``) so the harness can
# be imported on Builder-side host tests without pulling in torch +
# torch._dynamo through ``config``. Keep in sync with
# ``config.FOREGROUND_CLASS_INDICES``.
_FOREGROUND_CLASS_INDICES = tuple(range(1, 105))


class WholeBodyCtHarness(monai_harness.MonaiHarness):
    """Harness for the Whole Body CT Segmentation AIM (104 structures)."""

    MODEL_ID: ClassVar[str] = f"monai/{BUNDLE_NAME}"

    # The base class' default METRIC_NAME = "dice" is correct — we still
    # compute Dice, just per-class and averaged. Keep the name so Tuner's
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
        # Eval data: the bundle's published 0.8 mean Dice is per-class over
        # the 104 TotalSegmentator structures on its own validation split.
        # No programmatically fetchable, pre-merged public GT was available
        # at build time, so mean_dice is DORMANT — Tuner Step 6 will surface
        # "No (image, label) pairs found" until merged 0-104 labels are
        # staged under data/labelsTs/. The per-class mean Dice math below is
        # ready for whenever that happens. See module docstring and
        # configs/metadata.json in the bundle for the full disclaimer; surface
        # this in release notes.
        CheckInfo(
            "mean_dice",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Per-class mean Dice over 104 foreground TotalSegmentator "
            "structures on a label-aligned eval split. Bundle's published "
            "target is 0.8 on its own TotalSegmentator validation split. "
            "Dormant in this build: no pre-merged 0-104 public GT was "
            "fetchable, so labels are not staged under data/labelsTs/. "
            "See configs/metadata.json in the bundle for the labels schema.",
        ),
    ]

    def _compute_metric(self, prediction: Any, label: Any) -> float:
        """Per-class mean Dice over 104 foreground TotalSegmentator structures.

        ``prediction`` and ``label`` are NIfTI file paths whose voxels carry
        class indices in ``{0, ..., 104}`` (prediction is the bundle's
        ``AsDiscreted(argmax=True)`` output; label is the merged 0-104
        TotalSegmentator annotation). For each ``c`` in
        ``_FOREGROUND_CLASS_INDICES`` (= ``range(1, 105)``):

            Dice_c = 2·|pred==c ∩ gt==c| / (|pred==c| + |gt==c|)

        Empty/empty per-class pairs contribute 1.0 (matches the base
        ``_compute_dice`` convention so a case missing a rare class the
        bundle correctly does not predict is not punished). Returns the
        simple mean of the 104 per-class Dice values — the bundle's
        published 0.8 mean Dice is computed the same way (see bundle's
        ``configs/metadata.json:eval_metrics.mean_dice`` and the per-class
        entries in
        ``configs/metadata.json:network_data_format.outputs.pred.channel_def``).

        The validation wrapper here (file-path checks, shape match, indexing)
        raises ``ValueError`` on usage errors; the arithmetic is split into
        ``_per_class_mean_dice`` so unit tests can exercise it independently
        of the file-system probe.
        """
        if not isinstance(prediction, str) or not prediction:
            raise ValueError(
                f"WholeBodyCtHarness._compute_metric expects a file path "
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

        For each ``c`` in ``class_indices``, compute Dice on ``(pred == c)``
        vs ``(label == c)``. Return the simple mean of the per-class scores.
        Empty/empty per-class pairs contribute 1.0 (matches base
        ``_compute_dice`` convention). Raises ``ValueError`` on shape
        mismatch (resampling regression).
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
