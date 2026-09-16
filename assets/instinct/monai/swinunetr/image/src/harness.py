# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Per-model ModelHarness for this AIM.

The bulk of the implementation lives in ``monai_harness.MonaiHarness`` (the
shared base used by every MONAI bundle AIM). This file only declares the
leaf class with model-specific metadata.

Either import style works: ``aim-runtime`` discovery drops candidates that
another candidate inherits from, so ``MonaiHarness`` cannot shadow the leaf
class even when imported directly into this module's namespace.
"""

from __future__ import annotations

import os
from typing import ClassVar

import monai_harness

from aim_runtime.harness import CheckInfo, CheckResultType, CheckScope

# BTCV layout convention: imagesTr/img<NNNN>.nii.gz paired with
# labelsTr/label<NNNN>.nii.gz. The default filename-match pairing in
# MonaiHarness assumes image and label share a basename, so SwinunetrHarness
# overrides _pair_samples to rewrite img<NNNN> → label<NNNN>.
_IMG_PREFIX = "img"
_LBL_PREFIX = "label"


class SwinunetrHarness(monai_harness.MonaiHarness):
    """Harness for the Swin UNETR BTCV Multi-organ Segmentation AIM."""

    MODEL_ID: ClassVar[str] = "monai/swinunetr"

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
        CheckInfo(
            "mean_dice",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Mean Dice over BTCV 9-case held-out split (target ≥ 0.82 per bundle metadata.json)",
        ),
    ]

    @classmethod
    def _pair_samples(
        cls,
        data_dir: str,
        labels_dir: str,
        *,
        limit: int = 0,
    ) -> list[dict[str, str]]:
        """Pair BTCV imagesTr/img<NNNN>.nii.gz with labelsTr/label<NNNN>.nii.gz.

        The base-class default (filename match) would look for
        ``labels_dir/img<NNNN>.nii.gz`` and find nothing because BTCV
        labels carry the ``label`` prefix instead of ``img``.
        """
        if not (os.path.isdir(data_dir) and os.path.isdir(labels_dir)):
            return []
        pairs: list[dict[str, str]] = []
        for img_path in cls.list_unique_inputs(data_dir):
            img_name = os.path.basename(img_path)
            if not img_name.startswith(_IMG_PREFIX):
                continue
            label_name = _LBL_PREFIX + img_name[len(_IMG_PREFIX) :]
            label_path = os.path.join(labels_dir, label_name)
            if not os.path.isfile(label_path):
                continue
            pairs.append(
                {
                    "name": img_name,
                    "image": img_path,
                    "label": label_path,
                }
            )
        if limit and limit > 0:
            pairs = pairs[:limit]
        return pairs
