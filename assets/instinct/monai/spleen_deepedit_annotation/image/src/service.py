# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for Spleen DeepEdit interactive 3D segmentation.

Canonical AIM API endpoints:
  GET  /healthz        — BentoML built-in healthcheck probe
  POST /v1/models      — list available models
  POST /v1/inference   — synchronous inference

DeepEdit-specific request shape
-------------------------------
The /v1/inference request body's ``input`` carries the standard
``image_path`` plus two optional click lists in *original-image voxel
coordinates* (the AddGuidanceFromPointsDeepEditd transform rescales them
into the resized 128^3 frame at preprocess time):

  {
    "model": "monai/spleen_deepedit_annotation",
    "input": {
      "image_path": "/workspace/data/spleen_1.nii.gz",
      "foreground_clicks": [[20, 50, 30], [40, 60, 35]],
      "background_clicks": [[10, 10, 10]],
      "output_dir": "/workspace/eval"
    },
    "parameters": {}
  }

Both click lists are optional and default to ``[]``. Empty lists put the
service in "automatic mode": MONAI's ``AddGuidanceSignalDeepEditd``
produces all-zero guidance channels and the network behaves like a
plain segmentation model. Non-empty lists turn on the interactive
correction path the bundle was trained for.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any

import bentoml
from bentoml.exceptions import InvalidArgument
from config import (
    CHANNEL_DEF,
    ENV_PREFIX,
    LABEL_NAMES,
    LOGGER,
    OPT_AUTOCAST,
    OPT_COMPILE,
    OPT_COMPILE_MODE,
    OPT_DYNAMIC,
    OPT_SUMMARY,
    SERVICE_TIMEOUT_SECONDS,
)
from constants import BUNDLE_NAME
from monai.data import Dataset, ThreadDataLoader
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler
from monai.inferers import SimpleInferer
from monai.utils import set_determinism
from pipeline import (
    apply_compile,
    build_device,
    build_network,
    build_postprocessing,
    build_preprocessing,
    ensure_checkpoint,
    expected_output_path,
    load_checkpoint,
)
from schemas import InferenceResponse, PredictResponse

# ---------------------------------------------------------------------------
# Path containment helper
# ---------------------------------------------------------------------------


def _validate_path_within(path: Path, base: Path, label: str, must_exist: bool = False) -> Path:
    """Resolve *path*, assert it falls under *base*, and optionally assert it exists.

    Raises ``InvalidArgument`` so the caller gets HTTP 400 and the rule they
    broke; a plain ``ValueError`` becomes a 500 whose detail only reaches the
    server log.

    Containment and existence share one message on purpose: distinguishing them
    lets a caller enumerate the filesystem, which is why containment is checked
    first.

    Returns the resolved ``Path`` so callers can replace the raw value in one
    step.
    """
    resolved = path.resolve()
    base_resolved = base.resolve()
    # ``relative_to`` raises ValueError when resolved is not under base_resolved.
    # This avoids the startswith prefix-collision edge case (e.g. /workspace/data2
    # falsely matching /workspace/data).
    try:
        resolved.relative_to(base_resolved)
        contained = True
    except ValueError:
        contained = False
    if not contained or (must_exist and not resolved.is_file()):
        what = "an existing file" if must_exist else "a path"
        raise InvalidArgument(f"{label} must be {what} under {base_resolved}")
    return resolved


@bentoml.service(
    name="spleen-deepedit-inference",
    traffic={"timeout": SERVICE_TIMEOUT_SECONDS},
)
class SpleenDeepeditService:

    def __init__(self) -> None:
        self.bundle_dir = Path(
            os.environ.get(
                f"{ENV_PREFIX}_BUNDLE_DIR",
                "bundles/spleen_deepedit_annotation",
            )
        ).resolve()
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self.bundle_dir}")
        LOGGER.info("Initializing service  bundle_dir=%s", self.bundle_dir)

        # Weights are cache content, not bundle content: downloaded at first
        # start, kept under AIM_CACHE_PATH for the operator to mount a volume at.
        cache_root = Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()
        self.weights_dir = cache_root / BUNDLE_NAME
        ckpt_path = self.weights_dir / "models" / "model.pt"
        ensure_checkpoint(self.weights_dir, ckpt_path)

        self.model_id = os.environ.get(
            f"{ENV_PREFIX}_MODEL_ID",
            f"monai/{BUNDLE_NAME}",
        )

        # Allowed base directories for caller-supplied paths.
        # Configurable via env vars so integration tests and non-standard
        # deployments can override without code changes.
        self._data_root = Path(os.environ.get("AIM_INPUT_PATH", "/workspace/data")).resolve()
        self._output_root = Path(os.environ.get("AIM_OUTPUT_PATH", "/workspace/output")).resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        # AIM_OUTPUT_PATH is both the default and the containment root, so an
        # omitted output_dir always passes its own containment check.
        self.default_output_dir = str(self._output_root)

        self.device = build_device()
        self.use_amp = OPT_AUTOCAST and self.device.type == "cuda"
        self._lock = threading.Lock()
        LOGGER.info("Runtime  device=%s  amp=%s", self.device, self.use_amp)

        self.preprocessing = build_preprocessing(
            self.device,
            bundle_dir=self.bundle_dir,
        )

        self.network = build_network(self.device)

        self.inferer = SimpleInferer()

        load_checkpoint(self.network, ckpt_path, self.device)

        set_determinism(seed=123)
        self.network.eval()
        LOGGER.info("Model loaded.")

        self.network = apply_compile(
            self.network,
            enabled=OPT_COMPILE,
            mode=OPT_COMPILE_MODE,
            dynamic=OPT_DYNAMIC,
        )

        LOGGER.info("Service ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heartbeat(stop_event: threading.Event, interval_s: int = 10) -> None:
        elapsed = 0
        while not stop_event.wait(interval_s):
            elapsed += interval_s
            LOGGER.info("Inference still running … elapsed=%ss", elapsed)

    @staticmethod
    def _coerce_clicks(clicks: Any, label: str) -> list[list[int]]:
        """Validate that ``clicks`` is a list of ``[i, j, k]`` triplets and
        coerce to ``list[list[int]]``.

        Returns ``[]`` when ``clicks`` is ``None``/missing/empty — the
        "automatic mode" path. Raises ``ValueError`` with a clear message
        when a malformed click slips through.
        """
        if clicks is None or clicks == []:
            return []
        if not isinstance(clicks, list):
            raise ValueError(
                f"input.{label}_clicks must be a list of [i,j,k] triplets, " f"got {type(clicks).__name__}"
            )
        coerced: list[list[int]] = []
        for idx, pt in enumerate(clicks):
            if not isinstance(pt, (list, tuple)) or len(pt) != 3:
                raise ValueError(f"input.{label}_clicks[{idx}] must be [i,j,k]; got {pt!r}")
            try:
                coerced.append([int(pt[0]), int(pt[1]), int(pt[2])])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"input.{label}_clicks[{idx}] coords must be integers: {exc}") from None
        return coerced

    # ------------------------------------------------------------------
    # Core inference  (Pattern A — Evaluator-based, mirrors bundle inference.json)
    # ------------------------------------------------------------------

    def _run_predict(
        self,
        image_path: str,
        output_dir: str | None = None,
        foreground_clicks: list[list[int]] | None = None,
        background_clicks: list[list[int]] | None = None,
    ) -> PredictResponse:
        start_total = time.perf_counter()
        image_path_obj = Path(image_path)
        fg_clicks = foreground_clicks or []
        bg_clicks = background_clicks or []
        LOGGER.info(
            "_run_predict  image=%s  fg_clicks=%d  bg_clicks=%d",
            image_path_obj,
            len(fg_clicks),
            len(bg_clicks),
        )

        # Containment check must precede .exists() — checking existence first
        # makes the endpoint a filesystem oracle (allows enumerating paths by
        # distinguishing FileNotFoundError from ValueError).
        image_path_obj = _validate_path_within(image_path_obj, self._data_root, "image_path", must_exist=True)

        output_dir_path = Path(output_dir) if output_dir else Path(self.default_output_dir)
        output_dir = str(_validate_path_within(output_dir_path, self._output_root, "output_dir"))

        # The DeepEdit preprocessing chain reads click points from data
        # dict keys named after each label in ``LABEL_NAMES`` ("spleen",
        # "background"). Empty lists are valid and trigger the automatic
        # all-zero-guidance path.
        input_dict: dict[str, Any] = {
            "image": str(image_path_obj),
            "spleen": fg_clicks,
            "background": bg_clicks,
        }
        dataset = Dataset(data=[input_dict], transform=self.preprocessing)
        dataloader = ThreadDataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

        evaluator = SupervisedEvaluator(
            device=self.device,
            val_data_loader=dataloader,
            network=self.network,
            inferer=self.inferer,
            postprocessing=build_postprocessing(self.preprocessing, output_dir),
            val_handlers=[StatsHandler(iteration_log=False)],
            amp=self.use_amp,
        )

        with self._lock:
            LOGGER.info("Starting evaluator.run()")
            run_start = time.perf_counter()
            stop_event = threading.Event()
            heartbeat = threading.Thread(target=self._heartbeat, args=(stop_event,), daemon=True)
            heartbeat.start()
            try:
                evaluator.run()
            finally:
                stop_event.set()
            LOGGER.info("evaluator.run() completed in %.3fs", time.perf_counter() - run_start)

        out = expected_output_path(output_dir, str(image_path_obj))
        LOGGER.info("Prediction finished  total_time=%.3fs", time.perf_counter() - start_total)
        return PredictResponse(output_path=out, device=str(self.device), used_amp=self.use_amp)

    # ------------------------------------------------------------------
    # Canonical AIM API endpoints
    # ------------------------------------------------------------------

    @bentoml.api(route="/v1/inference")
    def v1_inference(
        self,
        model: str | None = None,
        input: dict | None = None,
        parameters: dict | None = None,
    ) -> InferenceResponse:
        """POST /v1/inference — canonical AIM API.

        Request body:
          { "model": "...",
            "input": {
              "image_path": <str>,
              "foreground_clicks": [[i,j,k], ...],     # optional
              "background_clicks": [[i,j,k], ...],     # optional
              "output_dir": <str>                       # optional
            },
            "parameters": {} }

        Click coordinates are in the ORIGINAL image's voxel frame; the
        preprocessing pipeline rescales them to the 128^3 model input
        frame automatically (``AddGuidanceFromPointsDeepEditd``).

        Empty / missing click lists → automatic-mode inference (all-zero
        guidance channels). Non-empty lists → interactive correction.
        """
        LOGGER.info("[v1/inference] model=%s  input=%s  parameters=%s", model, input, parameters)

        input_data: dict = input or {}
        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("input.image_path is required.")

        output_dir = input_data.get("output_dir")

        fg_clicks = self._coerce_clicks(
            input_data.get("foreground_clicks"),
            label="foreground",
        )
        bg_clicks = self._coerce_clicks(
            input_data.get("background_clicks"),
            label="background",
        )

        pred = self._run_predict(
            image_path=image_path,
            output_dir=output_dir,
            foreground_clicks=fg_clicks,
            background_clicks=bg_clicks,
        )

        return InferenceResponse(
            model=model or self.model_id,
            output={"output_path": pred.output_path},
            metadata={
                "device": pred.device,
                "used_amp": pred.used_amp,
                "parameters": parameters or {},
                "optimizations": OPT_SUMMARY,
                "interactive": {
                    "foreground_clicks": len(fg_clicks),
                    "background_clicks": len(bg_clicks),
                    "mode": "interactive" if (fg_clicks or bg_clicks) else "automatic",
                },
            },
        )

    @bentoml.api(route="/v1/models")
    def v1_models(self) -> dict[str, Any]:
        """POST /v1/models — list available models."""
        return {
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "owned_by": "MONAI",
                    "channel_def": CHANNEL_DEF,
                    "interactive": {
                        "label_names": LABEL_NAMES,
                        "foreground_label": "spleen",
                        "background_label": "background",
                        "click_coord_frame": "original_image_voxel_ijk",
                    },
                }
            ],
            "object": "list",
        }
