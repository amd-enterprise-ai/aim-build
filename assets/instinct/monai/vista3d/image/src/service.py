# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for the MONAI VISTA-3D AIM.

3D promptable segmentation foundation model. The service supports BOTH
inference modes of the bundle in a single endpoint:

  * automatic: omit prompts → ``everything_labels`` runs over 117 classes;
  * interactive class-prompt: supply ``label_prompt: [<class_ids>]`` to
    target specific classes (e.g. ``[3]`` for spleen);
  * interactive point-prompt: supply ``points: [[x,y,z], ...]`` +
    ``point_labels: [1, 0, ...]`` (xyz in the ORIGINAL image frame,
    0/1/-1 = bg/fg/ignore). Point mode requires ``label_prompt`` of
    length 1 — this is a bundle-level constraint enforced by
    ``Vista3dInferer``.

Canonical AIM API endpoints:
  GET  /healthz        — BentoML built-in healthcheck probe
  POST /v1/models      — list available models (incl. 133-entry channel_def)
  POST /v1/inference   — synchronous inference
"""

import os
import threading
import time
from pathlib import Path
from typing import Any

import bentoml
from bentoml.exceptions import InvalidArgument
from config import (
    BUNDLE_NAME,
    CHANNEL_DEF,
    ENV_PREFIX,
    LOGGER,
    OPT_AUTOCAST,
    OPT_COMPILE,
    OPT_COMPILE_MODE,
    OPT_DYNAMIC,
    OPT_SUMMARY,
    SERVICE_TIMEOUT_SECONDS,
)
from monai.data import Dataset, ThreadDataLoader
from monai.handlers import StatsHandler
from monai.utils import set_determinism
from pipeline import (
    apply_compile,
    build_device,
    build_everything_labels,
    build_inferer,
    build_network,
    build_postprocessing,
    build_preprocessing,
    build_subclass,
    ensure_checkpoint,
    expected_output_path,
    load_bundle_components,
    load_checkpoint,
    load_labels_dict,
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
    name=f"{BUNDLE_NAME}-inference",
    traffic={"timeout": SERVICE_TIMEOUT_SECONDS},
)
class Vista3dService:

    def __init__(self) -> None:
        self.bundle_dir = Path(os.environ.get(f"{ENV_PREFIX}_BUNDLE_DIR", f"bundles/{BUNDLE_NAME}")).resolve()
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self.bundle_dir}")
        LOGGER.info("Initializing service  bundle_dir=%s", self.bundle_dir)

        # Load the custom evaluator class from bundle scripts.
        _, Vista3dEvaluator = load_bundle_components(self.bundle_dir)
        self._Vista3dEvaluator = Vista3dEvaluator

        # Weights are cache content, not bundle content: downloaded at first
        # start, kept under AIM_CACHE_PATH for the operator to mount a volume at.
        cache_root = Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()
        self.weights_dir = cache_root / BUNDLE_NAME
        ckpt_path = self.weights_dir / "models" / "model.pt"
        ensure_checkpoint(self.weights_dir, ckpt_path)

        # VISTA-3D metadata helpers — needed by both preprocessing and the
        # evaluator's hyper_kwargs (everything_labels for the automatic
        # branch).
        self.labels_dict = load_labels_dict(self.bundle_dir)
        self.subclass = build_subclass()
        self.everything_labels = build_everything_labels()

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
            self.subclass,
            self.labels_dict,
        )

        self.network = build_network(self.device)

        self.inferer = build_inferer(bundle_dir=self.bundle_dir)

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

    # ------------------------------------------------------------------
    # Core inference  (Pattern A — Vista3dEvaluator with hyper_kwargs)
    # ------------------------------------------------------------------

    def _run_predict(
        self,
        image_path: str,
        label_prompt: list[int] | None = None,
        points: list[list[int]] | None = None,
        point_labels: list[int] | None = None,
        output_dir: str | None = None,
    ) -> PredictResponse:
        """Run one VISTA-3D inference.

        ``label_prompt`` / ``points`` / ``point_labels`` mirror the bundle's
        ``input_dict`` schema (see configs/inference.json + bundle
        README §Inference). ``None`` for all three triggers automatic mode
        via ``everything_labels``.

        ``points`` and ``point_labels`` must come together or not at all,
        match length, and (when present) ``label_prompt`` must have length
        1 — these are bundle-side invariants enforced by Vista3dInferer.
        """
        start_total = time.perf_counter()

        image_path_obj = Path(image_path)
        LOGGER.info(
            "_run_predict  image=%s  label_prompt=%s  points=%s",
            image_path_obj,
            label_prompt,
            "(omitted)" if not points else f"{len(points)} pts",
        )

        # Containment check must precede .exists() — checking existence first
        # makes the endpoint a filesystem oracle (allows enumerating paths by
        # distinguishing FileNotFoundError from ValueError).
        image_path_obj = _validate_path_within(image_path_obj, self._data_root, "image_path", must_exist=True)

        # Bundle invariants surfaced as explicit ValueErrors so the HTTP
        # caller sees a 400 rather than a deep traceback from inside
        # Vista3dInferer.
        if (points is None) != (point_labels is None):
            raise ValueError(
                "`points` and `point_labels` must be provided together " "(or both omitted).",
            )
        if points is not None and point_labels is not None:
            if len(points) != len(point_labels):
                raise ValueError(
                    f"`points` (n={len(points)}) and `point_labels` " f"(n={len(point_labels)}) must match in length.",
                )
            if label_prompt is not None and len(label_prompt) != 1:
                raise ValueError(
                    "When `points` are provided, `label_prompt` must have "
                    "length 1 (single-object point-prompt segmentation).",
                )

        output_dir_path = Path(output_dir) if output_dir else Path(self.default_output_dir)
        output_dir = str(_validate_path_within(output_dir_path, self._output_root, "output_dir"))

        # Build the input dict matching the bundle's input_dict schema. We
        # only add prompt keys when they are non-None — VistaPreTransformd
        # branches on key presence.
        input_dict: dict[str, Any] = {"image": str(image_path_obj)}
        if label_prompt is not None:
            input_dict["label_prompt"] = label_prompt
        if points is not None:
            input_dict["points"] = points
            input_dict["point_labels"] = point_labels

        dataset = Dataset(data=[input_dict], transform=self.preprocessing)
        dataloader = ThreadDataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        evaluator = self._Vista3dEvaluator(
            device=self.device,
            val_data_loader=dataloader,
            network=self.network,
            inferer=self.inferer,
            postprocessing=build_postprocessing(self.preprocessing, output_dir),
            val_handlers=[StatsHandler(iteration_log=False)],
            amp=self.use_amp,
            hyper_kwargs={
                "user_prompt": True,
                "everything_labels": self.everything_labels,
            },
        )

        with self._lock:
            LOGGER.info("Starting evaluator.run()")
            run_start = time.perf_counter()
            stop_event = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat,
                args=(stop_event,),
                daemon=True,
            )
            heartbeat.start()
            try:
                evaluator.run()
            finally:
                stop_event.set()
            LOGGER.info(
                "evaluator.run() completed in %.3fs",
                time.perf_counter() - run_start,
            )

        out = expected_output_path(output_dir, str(image_path_obj))
        LOGGER.info(
            "Prediction finished  total_time=%.3fs",
            time.perf_counter() - start_total,
        )
        return PredictResponse(
            output_path=out,
            device=str(self.device),
            used_amp=self.use_amp,
        )

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

            {
              "model": "monai/vista3d",
              "input": {
                "image_path": "<absolute path to .nii.gz>",
                "label_prompt": [3],                       (optional)
                "points": [[138, 245, 18], [271, 343, 27]], (optional, xyz)
                "point_labels": [1, 0],                    (optional, 0/1/-1)
                "output_dir": "<dir>"                      (optional)
              },
              "parameters": {}
            }

        Default (no prompts) → automatic 117-class everything-mode.
        """
        LOGGER.info(
            "[v1/inference] model=%s  input=%s  parameters=%s",
            model,
            input,
            parameters,
        )

        input_data: dict = input or {}
        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("input.image_path is required.")

        label_prompt = input_data.get("label_prompt")
        points = input_data.get("points")
        point_labels = input_data.get("point_labels")
        output_dir = input_data.get("output_dir")

        pred = self._run_predict(
            image_path=image_path,
            label_prompt=label_prompt,
            points=points,
            point_labels=point_labels,
            output_dir=output_dir,
        )

        return InferenceResponse(
            model=model or self.model_id,
            output={"output_path": pred.output_path},
            metadata={
                "device": pred.device,
                "used_amp": pred.used_amp,
                "parameters": parameters or {},
                "optimizations": OPT_SUMMARY,
            },
        )

    @bentoml.api(route="/v1/models")
    def v1_models(self) -> dict[str, Any]:
        """POST /v1/models — list available models.

        Exposes the bundle's full 133-entry ``channel_def`` so Tuner-side
        consumers can render class names without re-reading the bundle.
        """
        return {
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "owned_by": "MONAI",
                    "channel_def": CHANNEL_DEF,
                }
            ],
            "object": "list",
        }
