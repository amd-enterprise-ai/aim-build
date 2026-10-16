# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""DINOv2 model loading and inference without AIM runtime dependencies."""

from __future__ import annotations

import base64
import os
import time
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any

MODEL_NAME = "dinov2_vitl14"
EMBED_DIM = 1024
DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESIZE_SIZE = 256
SUPPORTED_PRECISIONS = {"fp32", "fp16"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV2_SOURCE_DIR = Path("/opt/dinov2")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Dinov2Predictor:
    """Load DINOv2 and produce CLS-token image embeddings."""

    def __init__(self) -> None:
        self.model_name = MODEL_NAME
        self.image_size = int(os.environ.get("DINOV2_IMAGE_SIZE", str(DEFAULT_IMAGE_SIZE)))
        self.embed_dim = EMBED_DIM
        self.precision = os.environ.get("DINOV2_PRECISION", "fp16").strip().lower()
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"DINOV2_PRECISION must be one of {sorted(SUPPORTED_PRECISIONS)}, got {self.precision!r}")
        self.torch_compile = _env_bool("DINOV2_TORCH_COMPILE", default=True)
        self.compile_mode = os.environ.get("DINOV2_COMPILE_MODE", "default").strip() or "default"
        self.gpu_preprocess = _env_bool("DINOV2_GPU_PREPROCESS", default=True)
        if os.environ.get("DINOV2_DISABLE_XFORMERS", "true").lower() in {"1", "true", "yes"}:
            os.environ.setdefault("XFORMERS_DISABLED", "1")
        self._model: Any | None = None
        self._device: Any | None = None
        self._forward: Any | None = None
        self._normalization_tensors: tuple[Any, Any] | None = None

    def predict_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        include_timings = bool(payload.get("include_timings", False))
        image_mod = self._image_module()
        if payload.get("image_base64") and payload.get("image_path"):
            raise ValueError("request must include only one of image_path or image_base64")
        if payload.get("image_base64"):
            raw = base64.b64decode(payload["image_base64"])
            image = image_mod.open(BytesIO(raw))
            source = "base64"
        elif payload.get("image_path"):
            source = str(payload["image_path"])
            image = image_mod.open(self._resolve_input_path(source))
        else:
            raise ValueError("request must include image_path or image_base64")
        return self._predict_image(image, source=source, include_timings=include_timings)

    def warmup(self) -> None:
        """Run one image through the full request path so compilation happens before serving."""
        # A synthetic tensor would not match the strides produced by preprocessing,
        # and torch.compile would recompile on the first real request.
        image = self._image_module().new("RGB", (self.image_size, self.image_size))
        self._predict_image(image, source="warmup", include_timings=False)

    def predict_image_path(self, image_path: str, include_timings: bool = False) -> dict[str, Any]:
        return self.predict_request({"image_path": image_path, "include_timings": include_timings})

    def _predict_image(self, image: Any, source: str, include_timings: bool) -> dict[str, Any]:
        preprocessing_start = time.perf_counter()
        tensor = self._preprocess_batch([image])
        preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000.0
        tensor = tensor.to(self._get_device())
        forward = self._get_forward()
        torch = self._torch()
        np = self._numpy()
        start_event = torch.cuda.Event(enable_timing=True) if include_timings else None
        end_event = torch.cuda.Event(enable_timing=True) if include_timings else None
        with torch.inference_mode():
            with self._autocast_context():
                if start_event is not None:
                    start_event.record()
                cls_tokens = forward(tensor)
                if end_event is not None:
                    end_event.record()
        gpu_inference_ms = None
        if start_event is not None and end_event is not None:
            end_event.synchronize()
            gpu_inference_ms = start_event.elapsed_time(end_event)
        cls_token = cls_tokens.detach().float().cpu().numpy()[0]
        result: dict[str, Any] = {
            "model": self.model_name,
            "source": source,
            "embedding_dim": int(cls_token.shape[-1]),
            "embedding": cls_token.tolist(),
            "norm": float(np.linalg.norm(cls_token)),
        }
        if include_timings:
            if gpu_inference_ms is None:
                raise RuntimeError("GPU timing was requested but no CUDA events were recorded")
            result["timings"] = {
                "preprocessing_ms": round(preprocessing_ms, 6),
                "gpu_inference_ms": round(float(gpu_inference_ms), 6),
            }
        return result

    def _get_device(self) -> Any:
        if self._device is None:
            torch = self._torch()
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DINOv2 ViT-L/14 requires a ROCm-compatible GPU, but "
                    "torch.cuda.is_available() returned false. Verify that the "
                    "container has access to the ROCm devices and runtime."
                )
            self._device = torch.device("cuda:0")
        return self._device

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        torch = self._torch()
        source_dir = self._dinov2_source_dir()
        model = torch.hub.load(str(source_dir), self.model_name, source="local", pretrained=True)
        model.eval().to(self._get_device())
        self._model = model
        return model

    def _get_forward(self) -> Any:
        if self._forward is not None:
            return self._forward
        torch = self._torch()
        model = self._get_model()
        forward = model.forward
        if self.torch_compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("DINOV2_TORCH_COMPILE=true requires torch.compile support")
            if self._get_device().type != "cuda":
                raise RuntimeError("DINOV2_TORCH_COMPILE=true is only supported on CUDA/ROCm devices")
            forward = torch.compile(forward, mode=self.compile_mode, dynamic=False)
        self._forward = forward
        return forward

    def _autocast_context(self) -> Any:
        torch = self._torch()
        if self.precision == "fp16" and self._get_device().type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _dinov2_source_dir(self) -> Path:
        if not (DINOV2_SOURCE_DIR / "hubconf.py").is_file():
            raise RuntimeError(f"pinned DINOv2 source is missing from {DINOV2_SOURCE_DIR}")
        return DINOV2_SOURCE_DIR

    def _preprocess(self, image: Any) -> Any:
        np = self._numpy()
        torch = self._torch()
        image = self._resize_and_crop(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        chw = np.transpose(arr, (2, 0, 1)).copy()
        return torch.from_numpy(chw).unsqueeze(0)

    def _resize_and_crop(self, image: Any) -> Any:
        image_ops = self._image_ops()
        image = image_ops.exif_transpose(image).convert("RGB")
        width, height = image.size
        resize_size = round(DEFAULT_RESIZE_SIZE * self.image_size / DEFAULT_IMAGE_SIZE)
        if width <= height:
            resized_size = (resize_size, int(resize_size * height / width))
        else:
            resized_size = (int(resize_size * width / height), resize_size)
        image = image.resize(resized_size, self._image_module().Resampling.BICUBIC)
        left = round((image.width - self.image_size) / 2)
        top = round((image.height - self.image_size) / 2)
        return image.crop((left, top, left + self.image_size, top + self.image_size))

    def _preprocess_batch(self, images: list[Any]) -> Any:
        torch = self._torch()
        if not self.gpu_preprocess or not torch.cuda.is_available():
            return torch.cat([self._preprocess(image) for image in images], dim=0)

        np = self._numpy()
        arrays = np.stack(
            [np.asarray(self._resize_and_crop(image), dtype=np.uint8) for image in images],
            axis=0,
        )
        device = self._get_device()
        batch = torch.from_numpy(arrays).permute(0, 3, 1, 2)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        mean, std = self._get_normalization_tensors()
        batch.sub_(mean).div_(std)
        torch.cuda.synchronize(device)
        return batch

    def _get_normalization_tensors(self) -> tuple[Any, Any]:
        if self._normalization_tensors is None:
            torch = self._torch()
            device = self._get_device()
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            self._normalization_tensors = mean, std
        return self._normalization_tensors

    def _data_dir(self) -> Path:
        return Path(os.environ.get("AIM_DATA_DIR", "/workspace/data"))

    def _resolve_input_path(self, raw_path: str) -> Path:
        # Containment must be checked before the file is opened; otherwise the
        # differing errors reveal which paths exist in the container.
        inputs_dir = (self._data_dir() / "inputs").resolve()
        candidate = Path(raw_path).resolve()
        if not candidate.is_relative_to(inputs_dir):
            raise ValueError(f"image path must be under {inputs_dir}")
        return candidate

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

    @staticmethod
    def _numpy() -> Any:
        import numpy as np

        return np

    @staticmethod
    def _image_module() -> Any:
        from PIL import Image

        return Image

    @staticmethod
    def _image_ops() -> Any:
        from PIL import ImageOps

        return ImageOps
