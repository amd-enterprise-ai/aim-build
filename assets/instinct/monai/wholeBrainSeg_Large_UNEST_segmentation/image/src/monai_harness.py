# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MonaiHarness — shared ModelHarness base for all MONAI bundle AIMs.

This module lives next to ``harness.py`` and is imported by it via
``import monai_harness`` (note: NOT ``from monai_harness import MonaiHarness``).
Keeping the class out of ``harness.py``'s namespace is intentional: the
``aim-runtime`` harness discovery uses ``inspect.getmembers(harness_module)``
to enumerate ``ModelHarness`` subclasses and picks the alphabetically-first
one. If both classes were in the same module, ``MonaiHarness`` would shadow
the per-model leaf class.

Per-model harnesses subclass ``MonaiHarness`` in ``harness.py`` and only
declare ``MODEL_ID`` and ``CHECKS`` (plus any override seams). This shared
base is copied verbatim into each MONAI AIM and never edited in an AIM.

Profile / config sources (in order of precedence)
-------------------------------------------------
Per-run config like ``image_paths``, ``num_requests``, etc. is read from:

  1. ``config.profile[<key>]`` — populated from the static profile registry
     when ``aim-runtime`` resolves the active profile.
  2. ``os.environ['AIM_HARNESS_<KEY>']`` — fallback when (1) is absent.
     Used by the Tuner side via ``docker exec -e AIM_HARNESS_*=...``.
  3. Hard-coded defaults below.

Source (2) exists because ``_benchmark_via_harness`` in ``aim-build``'s
entrypoint does not currently forward ``--config <yaml>`` into
``HarnessConfig.profile`` — only static profile-registry fields make it
through. Once that upstream gap is closed, source (2) can be retired.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
)

logger = logging.getLogger(__name__)

# BentoML 1.x exposes two health surfaces:
#   GET  /healthz  — built-in readiness probe (200 OK, plain HTTP)
#   POST /health   — user-defined @bentoml.api endpoint with rich JSON
# All @bentoml.api-decorated routes (/health, /v1/models, /v1/inference) are
# POST-only — a BentoML invariant. Use /healthz for liveness/readiness polling.
READYZ_PATH = "/healthz"
INFERENCE_PATH = "/v1/inference"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_S = 300
DEFAULT_NUM_WARMUP = 1
DEFAULT_NUM_REQUESTS = 10


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def _post_json(service_url: str, body: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    """Low-level POST /v1/inference. Takes an already-built body dict."""
    req = Request(
        f"{service_url}{INFERENCE_PATH}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def compute_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Stats schema identical to aims-tuner/.../run_benchmark.py:compute_stats.

    Keeping the field names byte-identical lets the Tuner-side adapter be a
    passthrough rather than a remap layer.
    """
    if not latencies_ms:
        return {}
    total_s = sum(latencies_ms) / 1000.0
    p50 = percentile(latencies_ms, 50)
    return {
        "mean_ms": statistics.fmean(latencies_ms),
        "stdev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "p50_ms": p50,
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput_img_per_sec": (len(latencies_ms) / total_s) if total_s > 0 else 0.0,
        "throughput_img_per_sec_from_p50": (1000.0 / p50) if p50 > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Shared base class — copied verbatim into the AIM's src/, not per-AIM
# --------------------------------------------------------------------------- #


class MonaiHarness(ModelHarness):
    """Common HTTP harness for MONAI bundle AIMs (BentoML, ``/v1/inference``).

    The defaults assume the canonical 3D NIfTI segmentation shape:
    one ``*.nii.gz`` input per case, a ``/v1/inference`` body of
    ``{"input": {"image_path": <path>}}``, a response with
    ``output.output_path`` pointing to a predicted NIfTI on disk, and
    binary Dice as the accuracy metric.

    A minimal per-AIM subclass just declares ``MODEL_ID`` + ``CHECKS``.
    For AIMs that diverge from the segmentation default, override the
    narrow customization seams rather than the higher-level ``validate``
    / ``benchmark`` / ``evaluate`` methods. The seams are:

      ===========================  ================================================
      What differs                 Override
      ===========================  ================================================
      Input file pattern           ``INPUT_GLOB``  (ClassVar)
      Accuracy metric name         ``METRIC_NAME`` (ClassVar)
      Request body shape           ``build_inference_payload``
      Response shape               ``_extract_prediction``
      Per-sample metric math       ``_compute_metric``
      Dataset-level aggregation    ``_reduce_metrics`` (see contract below)
      Image⇄label pairing          ``_pair_samples``  (see contract below)
      ===========================  ================================================

    **Metric reduction contract (per-sample vs dataset-level).**
    Most segmentation AIMs publish a "mean Dice" where the bundle's
    own evaluation handler computes per-case Dice and then averages —
    so the harness's default ``_reduce_metrics`` (arithmetic mean of
    per-sample floats) reproduces the published number bit-for-bit.
    A subset of AIMs publish metrics that are *not* the mean of
    per-sample numbers: detection AIMs publish COCO mAP / FROC, which
    aggregate TP/FP/FN across the entire dataset before computing a
    single AP; some classification AIMs publish macro-F1 across
    classes, which also doesn't decompose as a mean of per-sample
    scores. For those AIMs the leaf harness must override BOTH
    ``_compute_metric`` (to return a per-sample *dict of components*
    instead of a float — e.g. matches + GT count + pred count) AND
    ``_reduce_metrics`` (to consume the list of those dicts and emit
    the dataset-level headline). The harness records the dict on the
    per-sample record under ``<metric>_components`` for forensics;
    the per-sample ``<metric>`` column is then ``None`` (no per-sample
    headline is mathematically meaningful in that regime).

    **Image⇄label pairing contract.** The default ``_pair_samples`` is
    a strict filename match: it looks for the image's basename verbatim
    under ``labels_dir``. **Override on the subclass whenever any of
    these are true** (single most common cause of evaluate returning
    "No (image, label) pairs found"):

    - ``INPUT_GLOB`` is overridden to anchor on one of several modality
      files per case (multi-modal AIMs like BraTS) — the anchor's
      basename won't match the label.
    - Image and label filenames differ by prefix or suffix
      (e.g. BTCV's ``imgNNNN.nii.gz`` ↔ ``labelNNNN.nii.gz``;
      BraTS's ``<case>_flair.nii.gz`` ↔ ``<case>_seg.nii.gz``).
    - Labels live in a CSV/JSON manifest rather than per-image files
      (classification AIMs).

    Every AIM whose data layout matches one of those bullets should cover
    its ``_pair_samples`` override with a test that exercises it on a
    synthetic mimic of the AIM's real on-disk label naming convention.

    For example, a 2D classification AIM would set
    ``INPUT_GLOB = "*.png"`` and ``METRIC_NAME = "accuracy"``, then
    override ``_extract_prediction`` to pull a class label out of the
    response and ``_compute_metric`` to return ``1.0`` on match / ``0.0``
    otherwise. The HTTP framing, warmup, statistics, and pass/fail
    accounting in ``benchmark`` and ``evaluate`` are unchanged.
    """

    MODEL_ID: ClassVar[str] = ""
    CHECKS: ClassVar[list[CheckInfo]] = []

    # ``INPUT_GLOB`` is the per-AIM **logical-case anchor** glob used by
    # ``list_unique_inputs`` to discover one path per logical inference
    # case under a data directory. The default ``*.nii.gz`` is correct
    # for the canonical "one NIfTI per case" convention (spleen,
    # vista3d, swinunetr, …). Override for AIMs whose logical case is
    # multi-file (e.g. BraTS: 4 modality files per case → set to
    # ``"*_flair.nii.gz"`` to anchor on FLAIR; ``build_inference_payload``
    # then expands an anchor into the full multi-file request body), or
    # whose file type differs (e.g. ``"*.png"`` for a 2D classification
    # AIM). Pairing in ``evaluate`` honours the same glob, so labels are
    # matched against image files discovered by it.
    INPUT_GLOB: ClassVar[str] = "*.nii.gz"

    # ``METRIC_NAME`` names the per-sample accuracy metric this AIM
    # produces in ``evaluate``. The base class computes binary Dice
    # (``"dice"``). Override for AIMs with a different metric:
    # ``"accuracy"`` for classification, ``"mean_distance_mm"`` for
    # landmark regression, ``"map_at_iou_0_5"`` for detection, etc.
    # The name shows up in the per-sample record and as a prefix in
    # the aggregate stats (``mean_<METRIC_NAME>`` etc.) so downstream
    # tools can key off it.
    METRIC_NAME: ClassVar[str] = "dice"

    # ----- per-AIM customization seams ---------------------------------- #
    #
    # Override one or more of these on the per-model subclass to handle
    # AIMs that don't match the 3D-NIfTI-segmentation defaults. Each
    # seam is independent — overriding ``build_inference_payload`` to
    # reshape the request does not require overriding the response side
    # or the metric, and vice versa. See the class docstring above for
    # the full table mapping "what differs" to "what to override".

    @classmethod
    def list_unique_inputs(cls, data_dir: str) -> list[str]:
        """Return sorted absolute paths of every logical inference case
        in ``data_dir``, one path per case.

        Default implementation globs ``cls.INPUT_GLOB`` (which is
        ``"*.nii.gz"`` on the base class) under ``data_dir`` and
        returns the result sorted. Override (or override
        ``INPUT_GLOB``) on the per-model subclass when:

          - the AIM is multi-file per case (BraTS — anchor on FLAIR,
            harness expands to {T1c, T1, T2, FLAIR} via
            ``build_inference_payload``);
          - the file extension differs (e.g. ``*.png`` for a 2D
            classification AIM);
          - case discovery is more complex than a single glob (e.g. a
            datalist JSON drives it).

        Used by consumers (the Tuner today; potentially other
        orchestrators) to enumerate the workload before invoking
        ``benchmark`` / ``evaluate``. Keeping the convention in the
        harness means each consumer doesn't have to re-derive
        per-AIM file conventions.
        """
        if not data_dir or not os.path.isdir(data_dir):
            return []
        from pathlib import Path

        return [str(p) for p in sorted(Path(data_dir).glob(cls.INPUT_GLOB)) if not p.name.startswith(".")]

    def build_inference_payload(
        self,
        image_path: str,
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct the /v1/inference request body for one image path.

        Override in subclasses to reshape the body for AIMs that don't
        match the default single-path convention. Example: BraTS
        expands a single anchor file into a 4-element `image_paths`
        list ordered (T1c, T1, T2, FLAIR).

        Default:

            {"model": <MODEL_ID>,
             "input": {"image_path": <path>, **extra_input},
             "parameters": {}}
        """
        payload_input: dict[str, Any] = {"image_path": image_path}
        if extra_input:
            payload_input.update(extra_input)
        return {"model": self.MODEL_ID, "input": payload_input, "parameters": {}}

    def post_inference(
        self,
        service_url: str,
        image_path: str,
        *,
        extra_input: dict[str, Any] | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """POST /v1/inference using `build_inference_payload`. Subclasses
        usually don't need to override this — override
        `build_inference_payload` instead.
        """
        body = self.build_inference_payload(image_path, extra_input)
        return _post_json(service_url, body, timeout_seconds)

    def _extract_prediction(self, resp: dict[str, Any]) -> Any:
        """Pull the prediction out of a ``/v1/inference`` response body.

        Default extracts ``resp["output"]["output_path"]`` — the file
        path on disk where a segmentation bundle wrote its predicted
        NIfTI mask. Returns ``None`` when the field is missing (the
        caller treats that as a malformed response).

        Override for AIMs whose response is shaped differently:

          - classification: return the predicted class label/index
            from e.g. ``resp["output"]["class_label"]``;
          - landmarks: return a list of ``(x, y, z)`` coords;
          - detection: return a list of ``{"box": ..., "score": ...}``.

        The returned value flows straight into ``_compute_metric`` as
        the ``prediction`` argument, so the two methods are usually
        overridden together.
        """
        return (resp.get("output") or {}).get("output_path")

    def _compute_metric(self, prediction: Any, label: Any) -> float | dict[str, Any]:
        """Compute the per-sample accuracy metric.

        Return type:

          * ``float`` — per-sample score for metrics where the bundle's
            published number is mean-of-per-sample. The harness records
            the value on the per-sample report under ``<METRIC_NAME>``
            and averages all per-sample floats in the default
            ``_reduce_metrics``. THIS IS THE DEFAULT AND COVERS
            SEGMENTATION + CLASSIFICATION + LANDMARK REGRESSION.

          * ``dict[str, Any]`` — per-sample *components* for metrics
            where the bundle's published number does NOT decompose as
            a mean-of-per-sample (detection's COCO mAP / FROC, multi-
            class macro-F1, etc.). The harness records the dict on the
            per-sample report under ``<METRIC_NAME>_components`` for
            forensics; the per-sample ``<METRIC_NAME>`` column is
            ``None``. The leaf harness MUST also override
            ``_reduce_metrics`` to consume the list of these dicts and
            return the dataset-level headline.

        Default expects both ``prediction`` and ``label`` to be NIfTI
        file paths and computes binary Dice via ``_compute_dice``.
        Raises ``ValueError`` when ``prediction`` is not a usable
        file path (missing field, non-string, file not on disk);
        ``evaluate`` catches that and records it on the per-sample
        record.

        Override for non-Dice metrics — usually paired with an
        ``_extract_prediction`` override that produces the right
        input shape. Examples:

          - classification (single label):
              ``return 1.0 if prediction == label else 0.0``
          - landmark regression (lists of coords):
              ``return _mean_euclidean(prediction, label)``
          - detection (lists of boxes with scores), per-sample float
            proxy: ``return _map_at_iou(prediction, label, iou=0.5)``
          - detection (lists of boxes with scores), dataset-level
            faithful pattern: return per-sample match components
            ``{"matches": ..., "pred_scores": ..., "gt_count": int,
            "pred_count": int}`` AND override ``_reduce_metrics`` to
            aggregate them into a single COCO AP.

        Subclasses that override may also override ``_pair_samples``
        if ``label`` is not naturally a file path (e.g. a class index
        loaded from a CSV).
        """
        if not isinstance(prediction, str) or not prediction:
            raise ValueError(
                f"Default _compute_metric expects a file path prediction, "
                f"got {type(prediction).__name__}={prediction!r}"
            )
        if not os.path.isfile(prediction):
            raise ValueError(f"Prediction file not on disk: {prediction}")
        return self._compute_dice(prediction, label)

    def _reduce_metrics(
        self,
        per_sample: list[float | dict[str, Any]],
        metric_name: str,
    ) -> tuple[float | None, dict[str, Any]]:
        """Aggregate per-sample metric outputs into the dataset-level headline.

        Inputs:
          * ``per_sample``: list of values returned by
            ``_compute_metric`` for samples that scored successfully
            (failed samples are dropped by the caller). Each entry is
            either a ``float`` (per-sample-mean regime) or a
            ``dict`` (dataset-level-aggregator regime). The harness
            does not mix the two within one evaluate call.
          * ``metric_name``: the ``METRIC_NAME`` (or runtime override)
            in effect for this evaluate call. Provided so subclasses
            can route on metric name without re-reading config.

        Returns ``(headline, extra)``:

          * ``headline``: the dataset-level metric value, stored as
            ``mean_<metric_name>`` in ``HarnessResult.metrics``. Use
            the ``mean_`` prefix even for non-mean reductions (mAP,
            FROC) to keep the field name stable for Tuner-side
            consumers — the value itself is whatever the bundle's
            published number is. ``None`` only when there are no
            valid per-sample outputs to reduce.

          * ``extra``: extra fields to merge into
            ``HarnessResult.metrics`` (per-class breakdowns, FROC at
            additional sensitivities, the full COCO scalar dict).
            Namespace keys with the metric prefix to avoid colliding
            with the existing evaluate stats namespace (``n``,
            ``n_pass``, ``n_fail``, ``success_rate``, ``metric``,
            ``data_dir``, ``labels_dir``, ``mean_<metric>``,
            ``stdev_<metric>``, ``min_<metric>``, ``max_<metric>``):
            e.g. ``{"coco_ap_iou_0_1_per_class": {...}}``.

        DEFAULT: arithmetic mean of float entries. This is correct
        when the bundle's published metric is itself a mean of
        per-case scores (every MONAI segmentation AIM today). Empty
        input → ``(None, {})``.

        OVERRIDE WHEN: the bundle's published metric aggregates
        across the dataset before reducing (detection mAP / FROC,
        macro-F1, calibration ECE, etc.). In that case
        ``_compute_metric`` must return dicts of components and this
        method consumes them. Example sketch for COCO mAP::

            def _reduce_metrics(self, per_sample, metric_name):
                if not per_sample or not isinstance(per_sample[0], dict):
                    return None, {}
                coco = COCOMetric(classes=["nodule"], iou_list=[0.1])
                matches = [s["matches"] for s in per_sample]
                coco_dict, _ = coco(matches)
                ap = float(coco_dict[f"AP_IoU_0.10_MaxDet_100"])
                return ap, {f"{metric_name}_per_class": coco_dict}
        """
        floats = [s for s in per_sample if isinstance(s, (int, float))]
        if not floats:
            return None, {}
        return statistics.fmean(floats), {}

    # ----- discovery / introspection ------------------------------------ #

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    # ----- health ------------------------------------------------------- #

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        """Poll BentoML's built-in /healthz (GET, 200 OK once the service is up)."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urlopen(Request(f"{service_url}{READYZ_PATH}"), timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError) as exc:
                logger.debug("Health check retry for %s failed: %s", f"{service_url}{READYZ_PATH}", exc)
            time.sleep(1.0)
        return False

    # ----- validate ----------------------------------------------------- #

    def validate(self, config: HarnessConfig) -> HarnessResult:
        url = self._service_url(config)
        checks: list[CheckResult] = []

        healthy = self.health_check(url, timeout_seconds=config.timeout_seconds)
        checks.append(
            CheckResult(
                name="bentoml_health",
                result_type=CheckResultType.PASS_FAIL,
                success=healthy,
                value=healthy,
                detail="" if healthy else f"Service unreachable at {url}{READYZ_PATH}",
            )
        )
        if not healthy:
            return HarnessResult(success=False, summary="Health check failed", checks=checks)

        # `bentoml_health` is a RUNTIME check (always run above). The predict
        # smoke test is OFFLINE — a RUNTIME-only deploy stops at health.
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=True,
                summary=f"{self._slug()} validate: health OK (RUNTIME scope; smoke skipped)",
                checks=checks,
                metrics={"service_url": url},
            )

        image_paths = self._image_paths(config)
        if not image_paths:
            checks.append(
                CheckResult(
                    name="predict_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=(
                        "No image_paths in profile and no AIM_HARNESS_DATA_DIR / "
                        "AIM_HARNESS_IMAGE_PATHS in env — cannot run predict smoke test"
                    ),
                )
            )
            return HarnessResult(
                success=False,
                summary="Cannot smoke-test predict without image_paths",
                checks=checks,
            )

        checks.append(self._predict_smoke(url, image_paths[0], config.timeout_seconds))
        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"{self._slug()} validate: {sum(c.success for c in checks)}/{len(checks)} checks passed",
            checks=checks,
            metrics={"service_url": url, "smoke_image": image_paths[0]},
        )

    # ----- benchmark ---------------------------------------------------- #

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        url = self._service_url(config)
        if not self.health_check(url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary=f"Service unreachable at {url}")

        image_paths = self._image_paths(config)
        if not image_paths:
            return HarnessResult(
                success=False,
                summary="profile.image_paths (or AIM_HARNESS_DATA_DIR / AIM_HARNESS_IMAGE_PATHS) is required for benchmark",
            )

        num_requests = self._profile_int(config, "num_requests", "AIM_HARNESS_NUM_REQUESTS", DEFAULT_NUM_REQUESTS)
        num_warmup = self._profile_int(config, "num_warmup", "AIM_HARNESS_NUM_WARMUP", DEFAULT_NUM_WARMUP)
        request_timeout = self._profile_int(
            config, "request_timeout_s", "AIM_HARNESS_REQUEST_TIMEOUT_S", DEFAULT_TIMEOUT_S
        )

        for i in range(num_warmup):
            path = image_paths[i % len(image_paths)]
            try:
                self.post_inference(url, path, timeout_seconds=request_timeout)
            except Exception as exc:  # pragma: no cover — warmup failures are informational
                logger.warning("Warmup request %d/%d failed: %s", i + 1, num_warmup, exc)

        latencies_ms: list[float] = []
        failures: list[str] = []
        for i in range(num_requests):
            path = image_paths[i % len(image_paths)]
            t0 = time.perf_counter()
            try:
                self.post_inference(url, path, timeout_seconds=request_timeout)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                failures.append(f"req {i}: {exc}")

        if not latencies_ms:
            return HarnessResult(
                success=False,
                summary=f"All {num_requests} benchmark requests failed",
                metrics={"failures": failures},
            )

        stats = compute_stats(latencies_ms)
        p50 = stats["p50_ms"]
        tput = stats["throughput_img_per_sec"]

        checks = [
            CheckResult(
                name="p50_latency_ms",
                result_type=CheckResultType.SCORE,
                success=True,
                value=round(p50, 1),
                detail=f"{len(latencies_ms)}/{num_requests} succeeded, p50={p50:.1f} ms",
            ),
            CheckResult(
                name="throughput_img_per_sec",
                result_type=CheckResultType.SCORE,
                success=True,
                value=round(tput, 4),
                detail=(
                    f"throughput={tput:.3f} img/s (measured), "
                    f"{stats['throughput_img_per_sec_from_p50']:.3f} img/s (from p50)"
                ),
            ),
        ]

        return HarnessResult(
            success=len(failures) == 0,
            summary=f"{self._slug()} benchmark: p50={p50:.1f} ms, throughput={tput:.3f} img/s",
            checks=checks,
            metrics={
                **stats,
                "latencies_ms": latencies_ms,
                "num_warmup": num_warmup,
                "num_requests": num_requests,
                "num_input_files": len(image_paths),
                "data_dir": config.profile.get("data_dir") or os.environ.get("AIM_HARNESS_DATA_DIR"),
                "service_url": url,
                "failures": failures,
            },
        )

    # ----- evaluate ----------------------------------------------------- #

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        """Score every (image, label) pair under the configured data +
        labels directories and aggregate the per-sample metric.

        Pairs are discovered by ``_pair_samples`` (filename match by
        default, honouring ``INPUT_GLOB``). For each pair the harness
        sends one ``/v1/inference`` request, pulls the prediction out
        of the response via ``_extract_prediction``, and scores it
        against the label via ``_compute_metric``. The aggregate stats
        key off ``METRIC_NAME`` (so a classification AIM produces
        ``mean_accuracy`` instead of ``mean_dice``, etc.).

        Inputs (in profile or via ``AIM_HARNESS_*`` env vars):
          * ``data_dir`` / ``AIM_HARNESS_DATA_DIR``
              Container-side dir of input images (scanned via
              ``cls.INPUT_GLOB``).
          * ``labels_dir`` / ``AIM_HARNESS_LABELS_DIR``
              Container-side dir of ground-truth labels. Default
              pairing is by filename match; override ``_pair_samples``
              if your dataset needs something fancier (CSV manifest,
              stem-and-suffix convention, etc.).
          * ``output_dir`` / ``AIM_HARNESS_OUTPUT_DIR``
              Container-side dir forwarded as ``input.output_dir`` so
              the bundle knows where to write predictions.
          * ``limit`` / ``AIM_HARNESS_LIMIT``
              Optional cap on number of pairs scored.
          * ``label_prompt`` / ``AIM_HARNESS_LABEL_PROMPT``
              Optional JSON list passed as ``input.label_prompt`` for
              prompt-driven bundles (vista3d, multi-organ, …).
          * ``metric`` / ``AIM_HARNESS_METRIC``
              Optional runtime override for ``METRIC_NAME``.
          * ``request_timeout_s`` / ``AIM_HARNESS_REQUEST_TIMEOUT_S``
              Per-request HTTP timeout.
        """
        url = self._service_url(config)
        if not self.health_check(url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary=f"Service unreachable at {url}")

        data_dir = self._profile_str(config, "data_dir", "AIM_HARNESS_DATA_DIR", "")
        labels_dir = self._profile_str(config, "labels_dir", "AIM_HARNESS_LABELS_DIR", "")
        output_dir = self._profile_str(config, "output_dir", "AIM_HARNESS_OUTPUT_DIR", "")
        if not (data_dir and labels_dir and output_dir):
            return HarnessResult(
                success=False,
                summary=(
                    "evaluate requires data_dir + labels_dir + output_dir "
                    "(via profile or AIM_HARNESS_DATA_DIR/LABELS_DIR/OUTPUT_DIR env)"
                ),
            )

        request_timeout = self._profile_int(
            config, "request_timeout_s", "AIM_HARNESS_REQUEST_TIMEOUT_S", DEFAULT_TIMEOUT_S
        )
        limit = self._profile_int(config, "limit", "AIM_HARNESS_LIMIT", 0)

        # Optional label_prompt forwarded as extra inference input. Stored as
        # JSON in env (e.g. AIM_HARNESS_LABEL_PROMPT='[3]') because env vars
        # are scalar by nature.
        label_prompt: list[int] | None = None
        lp_raw = config.profile.get("label_prompt") or os.environ.get("AIM_HARNESS_LABEL_PROMPT")
        if isinstance(lp_raw, list):
            label_prompt = [int(x) for x in lp_raw]
        elif isinstance(lp_raw, str) and lp_raw.strip():
            try:
                parsed = json.loads(lp_raw)
                if isinstance(parsed, list):
                    label_prompt = [int(x) for x in parsed]
            except (ValueError, TypeError):
                logger.warning("Ignoring unparseable AIM_HARNESS_LABEL_PROMPT=%r", lp_raw)

        pairs = self._pair_samples(data_dir, labels_dir, limit=limit)
        if not pairs:
            missing: list[str] = []
            if not os.path.isdir(data_dir):
                missing.append(f"data_dir={data_dir}")
            if not os.path.isdir(labels_dir):
                missing.append(f"labels_dir={labels_dir}")
            if missing:
                summary = (
                    "No (image, label) pairs found: the following path"
                    f"{'s' if len(missing) > 1 else ''} "
                    f"do{'es' if len(missing) == 1 else ''} not exist "
                    f"INSIDE the container: {', '.join(missing)}. Most "
                    "likely the AIM image was deployed without a "
                    "`-v <host>:<container>` mount for "
                    f"{'them' if len(missing) > 1 else 'it'} "
                    "(common cause: the AIM's evaluate-data split lives "
                    "in a different on-disk dir than the inference "
                    "inputs and the sidecar didn't declare the extra "
                    "mount). Verify with `docker exec <container> ls "
                    f"{missing[0].split('=', 1)[1]}` and re-deploy with "
                    "the missing mount."
                )
            else:
                summary = (
                    f"No (image, label) pairs found under {data_dir} / "
                    f"{labels_dir}. Both directories exist but no images "
                    f"could be paired with a label. If image and label "
                    f"filenames don't share a basename (multi-modal "
                    f"anchor, img/label prefix swap, manifest-driven "
                    f"labels), override "
                    f"{type(self).__name__}._pair_samples — see the "
                    f'MonaiHarness docstring "Image⇄label pairing contract".'
                )
            return HarnessResult(
                success=False,
                summary=summary,
                metrics={"data_dir": data_dir, "labels_dir": labels_dir},
            )

        metric_name = self._profile_str(config, "metric", "AIM_HARNESS_METRIC", self.METRIC_NAME) or self.METRIC_NAME

        samples: list[dict[str, Any]] = []
        per_sample_outputs: list[float | dict[str, Any]] = []
        float_scores: list[float] = []
        extra_input: dict[str, Any] = {"output_dir": output_dir}
        if label_prompt is not None:
            extra_input["label_prompt"] = label_prompt

        for pair in pairs:
            rec: dict[str, Any] = {
                "name": pair["name"],
                "image": pair["image"],
                "label": pair["label"],
                "status": "FAIL",
                metric_name: None,
                "latency_ms": None,
                "error": None,
                "prediction": None,
                "pred_path": None,
            }
            try:
                t0 = time.perf_counter()
                resp = self.post_inference(
                    url,
                    pair["image"],
                    extra_input=extra_input,
                    timeout_seconds=request_timeout,
                )
                rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                prediction = self._extract_prediction(resp)
                if prediction is None:
                    rec["error"] = (
                        f"Response missing prediction (override "
                        f"_extract_prediction if needed): keys={list(resp.keys())}"
                    )
                    samples.append(rec)
                    continue
                rec["prediction"] = prediction
                # Backward-compat field for tools that rescore predictions
                # from disk (Tuner-side). Only meaningful for file-path
                # predictions; left null for class labels / coord lists.
                if isinstance(prediction, str):
                    rec["pred_path"] = prediction
                result = self._compute_metric(prediction, pair["label"])
                rec["status"] = "PASS"
                if isinstance(result, dict):
                    rec[metric_name] = None
                    rec[f"{metric_name}_components"] = result
                elif isinstance(result, (int, float)):
                    rec[metric_name] = round(float(result), 6)
                    float_scores.append(float(result))
                else:
                    raise ValueError(
                        f"{type(self).__name__}._compute_metric must return "
                        f"float or dict, got {type(result).__name__}={result!r}"
                    )
                per_sample_outputs.append(result)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            samples.append(rec)

        n = len(samples)
        n_pass = sum(1 for s in samples if s["status"] == "PASS")
        success_rate = n_pass / n if n else 0.0
        stats: dict[str, Any] = {
            "n": n,
            "n_pass": n_pass,
            "n_fail": n - n_pass,
            "success_rate": round(success_rate, 4),
        }

        headline: float | None = None
        extra_stats: dict[str, Any] = {}
        if per_sample_outputs:
            headline, extra_stats = self._reduce_metrics(
                per_sample_outputs,
                metric_name,
            )

        stats[f"mean_{metric_name}"] = round(float(headline), 6) if isinstance(headline, (int, float)) else None
        if float_scores:
            stats.update(
                {
                    f"stdev_{metric_name}": (
                        round(statistics.stdev(float_scores), 6) if len(float_scores) > 1 else 0.0
                    ),
                    f"min_{metric_name}": round(min(float_scores), 6),
                    f"max_{metric_name}": round(max(float_scores), 6),
                }
            )
        else:
            stats.update(
                {
                    f"stdev_{metric_name}": None,
                    f"min_{metric_name}": None,
                    f"max_{metric_name}": None,
                }
            )
        for k, v in extra_stats.items():
            if k in stats:
                continue
            stats[k] = v

        checks = [
            CheckResult(
                name="success_rate",
                result_type=CheckResultType.SCORE,
                success=success_rate >= 0.9,
                value=round(success_rate, 4),
                detail=f"{n_pass}/{n} samples scored",
            )
        ]
        if per_sample_outputs and stats[f"mean_{metric_name}"] is not None:
            checks.append(
                CheckResult(
                    name=f"mean_{metric_name}",
                    result_type=CheckResultType.SCORE,
                    success=True,
                    value=stats[f"mean_{metric_name}"],
                    detail=(
                        f"{metric_name} over {len(per_sample_outputs)} samples "
                        f"(reduced via {type(self).__name__}._reduce_metrics)"
                    ),
                )
            )

        return HarnessResult(
            success=bool(per_sample_outputs) and success_rate >= 0.9,
            summary=(
                f"{self._slug()} evaluate: "
                f"{n_pass}/{n} samples scored, "
                f"mean_{metric_name}={stats.get(f'mean_{metric_name}')}"
            ),
            checks=checks,
            metrics={
                **stats,
                "metric": metric_name,
                "data_dir": data_dir,
                "labels_dir": labels_dir,
                "output_dir": output_dir,
                "label_prompt": label_prompt,
                "limit": limit or None,
                "service_url": url,
                "samples": samples,
            },
        )

    # ----- internal helpers --------------------------------------------- #

    def _slug(self) -> str:
        """Short label used in summary strings.

        ``monai/spleen_ct_segmentation`` → ``spleen_ct_segmentation``.
        """
        return self.MODEL_ID.split("/")[-1] or type(self).__name__

    @staticmethod
    def _service_url(config: HarnessConfig) -> str:
        explicit = config.profile.get("service_url")
        if explicit:
            return str(explicit).rstrip("/")
        port = int(config.profile.get("port", DEFAULT_PORT))
        return f"http://localhost:{port}"

    @classmethod
    def _image_paths(cls, config: HarnessConfig) -> list[str]:
        """Read image_paths from profile, then env, then a directory scan.

        Sources, in order of precedence:
          1. ``config.profile['image_paths']`` (list)
          2. env ``AIM_HARNESS_IMAGE_PATHS`` (':'-separated paths)
          3. env ``AIM_HARNESS_DATA_DIR`` (directory; scanned via
             ``cls.list_unique_inputs`` so per-AIM ``INPUT_GLOB``
             overrides apply — e.g. BraTS anchors on ``*_flair.nii.gz``,
             a hypothetical PNG-classification AIM would anchor on
             ``*.png``)
        """
        paths = config.profile.get("image_paths") or []
        if not paths:
            env_paths = os.environ.get("AIM_HARNESS_IMAGE_PATHS", "")
            if env_paths:
                paths = [p for p in env_paths.split(":") if p]
        if not paths:
            data_dir = os.environ.get("AIM_HARNESS_DATA_DIR", "")
            if data_dir:
                paths = cls.list_unique_inputs(data_dir)
        return [str(p) for p in paths]

    @staticmethod
    def _profile_int(config: HarnessConfig, key: str, env: str, default: int) -> int:
        val = config.profile.get(key)
        if val is not None:
            return int(val)
        env_val = os.environ.get(env)
        if env_val:
            return int(env_val)
        return default

    @staticmethod
    def _profile_str(config: HarnessConfig, key: str, env: str, default: str) -> str:
        val = config.profile.get(key)
        if val is not None and str(val) != "":
            return str(val)
        env_val = os.environ.get(env, "")
        return env_val if env_val else default

    @classmethod
    def _pair_samples(
        cls,
        data_dir: str,
        labels_dir: str,
        *,
        limit: int = 0,
    ) -> list[dict[str, str]]:
        """Filename-match pairing — same basename in data_dir and labels_dir.

        Note: when ``data_dir`` or ``labels_dir`` doesn't exist *inside the
        container* (most common cause: the AIM image was deployed without
        a ``-v <host>:<container>`` mount for that path; happens when an
        AIM's evaluate split lives in a different on-disk directory than
        the inference inputs), this returns ``[]`` and the calling
        ``evaluate`` flow surfaces a clear remediation message.

        Image discovery honours ``cls.INPUT_GLOB`` (so a PNG-classification
        AIM with ``INPUT_GLOB = "*.png"`` pairs ``case.png`` with
        ``labels_dir/case.png``, not ``.nii.gz`` only). Images with no
        matching label are skipped silently — the caller's per-sample
        report records the skip.

        Override on the subclass when image and label filenames do NOT
        share a basename. Common cases:

          - Multi-modal anchor: ``INPUT_GLOB = "*_flair.nii.gz"`` picks
            ``<case>_flair.nii.gz`` but the label is ``<case>_seg.nii.gz``.
          - Stem-prefix swap: ``imagesTr/img<NNNN>.nii.gz`` paired with
            ``labelsTr/label<NNNN>.nii.gz`` (BTCV).
          - Stem-suffix swap: ``case_001_img.nii.gz`` ↔ ``case_001_seg.nii.gz``.
          - Manifest-driven labels: ``label`` becomes a class index from
            a CSV/JSON, not a file path (override ``_compute_metric`` too).

        Forgetting this override is the #1 cause of evaluate returning
        "No (image, label) pairs found". When you override, add a test
        in ``tests/test_harness.py`` that exercises the override on a
        synthetic mimic of the real on-disk label naming convention.
        """
        if not (os.path.isdir(data_dir) and os.path.isdir(labels_dir)):
            return []
        pairs: list[dict[str, str]] = []
        for img_path in cls.list_unique_inputs(data_dir):
            name = os.path.basename(img_path)
            label_path = os.path.join(labels_dir, name)
            if not os.path.isfile(label_path):
                continue
            pairs.append(
                {
                    "name": name,
                    "image": img_path,
                    "label": label_path,
                }
            )
        if limit and limit > 0:
            pairs = pairs[:limit]
        return pairs

    @staticmethod
    def _compute_dice(pred_path: str, label_path: str) -> float:
        """Binary Dice on boolean masks. Empty/empty == 1.0.

        nibabel + numpy are MONAI-bundle dependencies, so they're already in
        every Layer-1 image we ship — no need to gate on import availability.
        """
        import nibabel as nib  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        pred = (nib.load(pred_path).get_fdata() > 0).astype(np.bool_)
        gt = (nib.load(label_path).get_fdata() > 0).astype(np.bool_)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch pred={pred.shape} vs gt={gt.shape} — resampling produced a different grid than the GT."
            )
        p_sum = float(pred.sum())
        g_sum = float(gt.sum())
        denom = p_sum + g_sum
        if denom == 0.0:
            return 1.0
        return 2.0 * float((pred & gt).sum()) / denom

    def _predict_smoke(self, service_url: str, image_path: str, timeout_seconds: int) -> CheckResult:
        try:
            resp = self.post_inference(service_url, image_path, timeout_seconds=timeout_seconds)
            prediction = self._extract_prediction(resp)
            if prediction is not None:
                detail = f"prediction={prediction!r}"
                if len(detail) > 200:
                    detail = detail[:197] + "..."
                return CheckResult(
                    name="predict_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail=detail,
                )
            return CheckResult(
                name="predict_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=(
                    f"Response missing prediction (override _extract_prediction if needed): keys={list(resp.keys())}"
                ),
            )
        except Exception as exc:
            return CheckResult(
                name="predict_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
