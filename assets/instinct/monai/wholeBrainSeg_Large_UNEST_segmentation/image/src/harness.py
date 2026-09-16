# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Per-model ModelHarness for the wholeBrainSeg_Large_UNEST_segmentation AIM.

The bulk of the implementation lives in ``monai_harness.MonaiHarness`` (the
shared base used by every MONAI bundle AIM). This file declares the leaf
class with model-specific metadata and overrides ``_compute_metric`` so
Tuner-side ``evaluate`` returns the bundle's per-class mean Dice (over the
132 foreground anatomical regions) instead of the base-class binary Dice.

Either import style works: ``aim-runtime`` discovery drops candidates that
another candidate inherits from, so ``MonaiHarness`` cannot shadow the leaf
class even when imported directly into this module's namespace.

Bundle metric (per-class mean Dice over 132 foreground classes)
---------------------------------------------------------------

The bundle's ``configs/metadata.json`` declares
``eval_metrics.mean_dice = 0.71``; combined with the bundle's 133-channel
output (background + 132 anatomical regions) this is the **per-class mean
Dice** computed over the 132 foreground class indices on the bundle's
held-out OASIS validation + CANDI test cohort. The base
``MonaiHarness._compute_metric`` returns binary Dice on
``(pred > 0)`` vs ``(label > 0)``, which would silently inflate the
number for a 133-class bundle (KB
``compute-metric-override-for-multi-class-or-multi-channel-bundle-dice``);
we override on the leaf to mirror the bundle's math.

Eval-data caveats (HIGH severity human_review entries)
------------------------------------------------------

The bundle's published number is on a private Vanderbilt + CANDI test
cohort, with all images **pre-registered to the MNI305 atlas template
using NiftyReg** before inference (bundle README §Important). Two
issues with reproducing this on public data:

1. **MNI305-registered public mirrors are scarce.** The substitute eval
   set staged for this AIM (5 OASIS-1 cases from
   ``radiata-ai/brain-structure``, CC-BY-SA 3.0 / OASIS DUA — see
   ``data/README.md``) is registered to **MNI152NLin2009cAsym** (CAT12
   + FSL FLIRT, 1.5 mm³), which is a *related but distinct* template
   from MNI305. Inference will run, but per-voxel anatomical alignment
   to the bundle's MNI305 prior is approximate, not bit-exact.

2. **No public 133-class GT.** The bundle's 133-class label space (the
   Vanderbilt segmentation protocol from Huo et al. 2019) is not
   reproduced on any public OASIS / CANDI mirror we could find. Tuner
   Step 6 (``evaluate``) therefore has nothing to score against.

Consequences:

* ``data/labelsTs/`` is intentionally absent — staging unaligned-GT here
  would silently produce meaningless Dice numbers.
* The ``_compute_metric`` override below is preserved (correct math
  per bundle) but is dormant under the current substitute data: Tuner
  Step 6 will surface "No (image, label) pairs found" until labels
  are staged. The harness is correct; the data is what's missing.
* ``_FOREGROUND_CLASS_INDICES`` covers ``range(1, 133)`` (excluding
  background 0) — the structural-correctness contract for the bundle's
  133-channel spatial output. Even with no labels, this is pinned by
  ``test_pipeline``/``test_inference`` against synthetic mocks so the
  bundle's output cardinality is verified.
* Re-introducing real per-class Dice is a *single* change at staging
  time: drop matching basename labels into ``data/labelsTs/`` and
  Tuner Step 6 starts producing the bundle's metric without further
  code changes. See ``data/README.md`` for the labels schema.

KB references: ``compute-metric-override-for-multi-class-or-multi-channel-bundle-dice``
(why this override is necessary), ``dataset-substitution-soft-flag-worked-example``
(equivalence-status semantics; this AIM's status is
``equivalent-task-no-labels``).

Pairing (``_pair_samples``)
---------------------------

Default basename match: ``imagesTs/<sid>.nii.gz`` ↔
``labelsTs/<sid>.nii.gz`` under
``${MONAI_DATA_ROOT}/wholeBrainSeg_Large_UNEST_segmentation/``. The
5 staged cases use OASIS-1 subject IDs verbatim (``OASIS10007``, …) so
the inherited ``MonaiHarness._pair_samples`` would match if/when
matching labels are staged — no override needed.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import monai_harness

from aim_runtime.harness import CheckInfo, CheckResultType, CheckScope

# Foreground class indices contributing to the bundle's published mean
# Dice. Bundle's metadata.json:network_data_format.outputs.pred.channel_def
# maps {0: background, 1..132: anatomical brain regions}. The bundle's
# eval_metrics.mean_dice = 0.71 is the per-class mean over indices 1-132.
#
# Duplicated here (instead of importing from ``config``) so the harness
# can be imported on Builder-side host tests without pulling in torch +
# torch._dynamo through ``config``. Keep in sync with
# ``config.FOREGROUND_CLASS_INDICES``.
_FOREGROUND_CLASS_INDICES = tuple(range(1, 133))


class WholebrainsegHarness(monai_harness.MonaiHarness):
    """Harness for the Whole Brain Segmentation Large UNEST AIM."""

    MODEL_ID: ClassVar[str] = "monai/wholeBrainSeg_Large_UNEST_segmentation"

    # The base class' default METRIC_NAME = "dice" is correct — we still
    # compute Dice, just per-class and averaged. Keep the name so
    # Tuner's aggregate stats stay byte-identical with other segmentation
    # AIMs.
    METRIC_NAME: ClassVar[str] = "dice"

    CHECKS: ClassVar[list[CheckInfo]] = [
        CheckInfo(
            "bentoml_health",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
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
        # Substitute eval data: 5 OASIS-1 T1w cases registered to
        # MNI152NLin2009cAsym (radiata-ai/brain-structure mirror). The
        # bundle was trained on MNI305-registered data and its
        # published 0.71 mean Dice is per-class over the 132 anatomical
        # regions on a private Vanderbilt + CANDI test split. Without
        # public 133-class GT, Tuner Step 6 will surface
        # "No (image, label) pairs found" — the per-class mean Dice
        # math below is ready for whenever a label-aligned dataset is
        # staged. See module docstring + data/README.md for the full
        # disclaimer; surface this in release notes.
        CheckInfo(
            "mean_dice",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Per-class mean Dice over 132 foreground brain regions on a "
            "label-aligned eval split. Bundle's published target is 0.71, "
            "measured on private Vanderbilt + CANDI MNI305-registered T1w. "
            "This AIM's substitute split is MNI152-registered OASIS-1 with "
            "no 133-class labels available publicly; mean_dice is dormant "
            "until labels are staged under data/labelsTs/. See "
            "data/README.md and state.yaml.dataset_origin_change.",
        ),
    ]

    def _compute_metric(self, prediction: Any, label: Any) -> float:
        """Per-class mean Dice over 132 foreground brain regions.

        ``prediction`` and ``label`` are NIfTI file paths whose voxels
        carry class indices in ``{0, ..., 132}`` (prediction is the
        bundle's ``AsDiscreted(argmax=True)`` output; label is expected
        to follow the same 133-class schema). For each ``c`` in
        ``_FOREGROUND_CLASS_INDICES`` (= ``range(1, 133)``):

            Dice_c = 2·|pred==c ∩ gt==c| / (|pred==c| + |gt==c|)

        Empty/empty per-class pairs contribute 1.0 (matches the base
        ``_compute_dice`` convention so a bundle that doesn't predict
        a rare class on a case missing that class doesn't get punished).
        Returns the simple mean of the 132 per-class Dice values — the
        bundle's published 0.71 mean Dice is computed the same way (see
        bundle's ``configs/metadata.json:eval_metrics.mean_dice`` and
        the per-class entries in
        ``configs/metadata.json:network_data_format.outputs.pred.channel_def``).

        The validation wrapper in this method (file-path checks, shape
        match, indexing) raises ``ValueError`` on usage errors; the
        arithmetic is split into ``_per_class_mean_dice`` so unit tests
        can exercise it independently of the file-system probe.
        """
        if not isinstance(prediction, str) or not prediction:
            raise ValueError(
                f"WholebrainsegHarness._compute_metric expects a file path "
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
        ``(pred == c)`` vs ``(label == c)``. Return the simple mean of
        the per-class scores. Empty/empty per-class pairs contribute 1.0
        (matches base ``_compute_dice`` convention). Raises ``ValueError``
        on shape mismatch (resampling regression).
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
