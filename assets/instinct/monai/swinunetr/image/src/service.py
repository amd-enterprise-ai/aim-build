# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for a MONAI model AIM.

Canonical AIM API endpoints (all POST — @bentoml.api defaults to POST):
  POST /health         — service health check (rich JSON)
  POST /v1/models      — list available models
  POST /v1/inference   — synchronous inference

Readiness/liveness is BentoML's built-in GET /healthz. Any GET aliases for
/health and /v1/models are provided by the BentoML engine integration in
aim-runtime.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any

import bentoml
from config import (
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
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler
from monai.inferers import SlidingWindowInferer
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


@bentoml.service(name="swinunetr-inference", traffic={"timeout": SERVICE_TIMEOUT_SECONDS})
class SwinunetrService:

    def __init__(self) -> None:
        self.bundle_dir = Path(
            os.environ.get(
                f"{ENV_PREFIX}_BUNDLE_DIR",
                "bundles/swin_unetr_btcv_segmentation",
            )
        ).resolve()
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self.bundle_dir}")
        LOGGER.info("Initializing service  bundle_dir=%s", self.bundle_dir)

        # SwinUNETR uses stock MONAI components (SlidingWindowInferer); no
        # bundle-specific scripts to import.

        # ── Checkpoint ──
        ckpt_path = self.bundle_dir / "models" / "model.pt"
        ensure_checkpoint(self.bundle_dir, ckpt_path)

        # ── Output dir and model ID ──
        self.default_output_dir = str((self.bundle_dir / "eval").resolve())
        self.model_id = os.environ.get(f"{ENV_PREFIX}_MODEL_ID", "MONAI/swin_unetr_btcv_segmentation")

        # ── Device ──
        self.device = build_device()
        self.use_amp = OPT_AUTOCAST and self.device.type == "cuda"
        self._lock = threading.Lock()
        LOGGER.info("Runtime  device=%s  amp=%s", self.device, self.use_amp)

        # ── Preprocessing ──
        self.preprocessing = build_preprocessing(self.device)

        # ── Network ──
        self.network = build_network(self.device)

        # ── Inferer (mirrors bundle inference.json) ──
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            overlap=0.5,
        )

        # ── Load checkpoint ──
        # Shared helper handles the common state-dict wrappings ("state_dict",
        # "model", "net") plus a bare state-dict.
        load_checkpoint(self.network, ckpt_path, self.device)

        set_determinism(seed=123)
        self.network.eval()
        LOGGER.info("Model loaded.")

        # ── torch.compile ──
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
    # Core inference — Pattern A (SupervisedEvaluator, mirrors bundle
    # inference.json). Pattern B (direct SlidingWindowInferer) was tried
    # first but broke MONAI Invertd's invertible-transform tracking when
    # the MetaTensor was moved across devices in a plain `.to(...)`. The
    # bundle ships an explicit SupervisedEvaluator with the same
    # data_loader / inferer / postprocessing chain, so we mirror it verbatim.
    # ------------------------------------------------------------------

    def _run_predict(
        self,
        image_path: str,
        output_dir: str | None = None,
    ) -> PredictResponse:
        start_total = time.perf_counter()
        image_path_obj = Path(image_path)
        LOGGER.info("_run_predict  image=%s", image_path_obj)

        if not image_path_obj.exists():
            raise FileNotFoundError(f"Input image not found: {image_path_obj}")

        output_dir = output_dir or self.default_output_dir
        output_dir = str(Path(output_dir).resolve())

        input_dict = {"image": str(image_path_obj)}
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

        Request body: { "model": "...", "input": {...}, "parameters": {...} }
        """
        LOGGER.info("[v1/inference] model=%s  input=%s  parameters=%s", model, input, parameters)

        input_data: dict = input or {}
        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("input.image_path is required.")

        output_dir = input_data.get("output_dir")

        pred = self._run_predict(image_path=image_path, output_dir=output_dir)

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
        """GET /v1/models — list available models."""
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

    @bentoml.api(route="/health")
    def health(self) -> dict[str, Any]:
        """GET /health — service health check."""
        return {
            "status": "ok",
            "device": str(self.device),
            "amp": self.use_amp,
            "bundle_dir": str(self.bundle_dir),
            "model_loaded": self.network is not None,
            "optimizations": OPT_SUMMARY,
        }
