# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MONAI pipeline construction: network, transforms, checkpoint management.

This is the most model-specific file. The overall structure (function names,
GPU-transforms pattern, checkpoint download) is shared, but the contents of
each function body vary per model. See agent instructions for full guidance.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from config import LOGGER, OPT_GPU_TRANSFORMS
from huggingface_hub import hf_hub_download
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def ensure_checkpoint(bundle_dir: Path, ckpt_path: Path) -> None:
    """Download model weights from Hugging Face if not present locally."""
    if ckpt_path.exists():
        return

    repo_id = os.environ.get("SWINUNETR_HF_REPO", "MONAI/swin_unetr_btcv_segmentation")
    filename = os.environ.get("SWINUNETR_HF_FILENAME", "models/model.pt")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Checkpoint not found at %s — downloading %s from %s …", ckpt_path, filename, repo_id)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(bundle_dir),
    )
    LOGGER.info("Downloaded checkpoint to %s", downloaded)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Download succeeded but checkpoint not at expected path: {ckpt_path}. " f"Downloaded file: {downloaded}"
        )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def build_device() -> torch.device:
    """Pick cuda:0 if available, else cpu. Single source of truth."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(device: torch.device) -> torch.nn.Module:
    """Instantiate the model architecture and move to device.

    Mirrors `bundles/swin_unetr_btcv_segmentation/configs/inference.json`
    network_def. NOTE: the bundle's inference.json (written for monai 1.4)
    also passes ``img_size=96``; MONAI 1.5 removed that kwarg (input
    spatial extent is now inferred at runtime). The trained weights are
    identical — same depths/num_heads/feature_size — so the bundle's
    model.pt loads into this constructor verbatim. See agent journal
    DECISION 2026-05-21T05:56:05Z (risk=medium).
    """
    return SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    network: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> None:
    """Load weights into ``network`` in place. Handles the common state-dict
    wrappings ("model", "state_dict", "net") plus a bare state-dict.

    Shared between the HTTP service and the internal benchmark script so the
    two never diverge on checkpoint handling.
    """
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        network.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        network.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "net" in checkpoint:
        network.load_state_dict(checkpoint["net"])
    elif isinstance(checkpoint, dict):
        network.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")


# ---------------------------------------------------------------------------
# torch.compile wrapper
# ---------------------------------------------------------------------------


def apply_compile(
    network: torch.nn.Module,
    *,
    enabled: bool,
    mode: str,
    dynamic: bool,
) -> torch.nn.Module:
    """Optionally wrap ``network`` with ``torch.compile``.

    Single source of truth for compile decisions. Used by both the HTTP
    service and the internal benchmark script. Returns ``network`` unchanged
    when compile is disabled or the model is not on CUDA.
    """
    if not enabled:
        LOGGER.info("torch.compile disabled")
        return network

    try:
        param = next(network.parameters())
        is_cuda = param.device.type == "cuda"
    except StopIteration:
        is_cuda = False

    if not is_cuda:
        LOGGER.info("torch.compile skipped (model is not on CUDA)")
        return network

    LOGGER.info("Compiling model with mode=%s dynamic=%s …", mode, dynamic)
    t0 = time.perf_counter()
    compiled = torch.compile(network, mode=mode, dynamic=dynamic)
    LOGGER.info(
        "torch.compile returned in %.1fs (actual compile cost deferred to first forward)", time.perf_counter() - t0
    )
    return compiled


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_preprocessing(device: torch.device, **model_kwargs) -> Compose:
    """Build the SwinUNETR preprocessing pipeline.

    Mirrors `bundles/swin_unetr_btcv_segmentation/configs/inference.json`
    `preprocessing#transforms` verbatim: LoadImaged(ITKReader) →
    EnsureChannelFirstd → Orientationd(RAS) → Spacingd(1.5,1.5,2.0 bilinear)
    → ScaleIntensityRanged(-175..250 → 0..1, clip) → EnsureTyped.

    GPU transforms optimization: when enabled, data is moved to GPU early
    (right after EnsureChannelFirstd) so subsequent transforms run on GPU.
    """
    transforms: list = [
        LoadImaged(keys="image", reader="ITKReader", image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if OPT_GPU_TRANSFORMS:
        LOGGER.info("GPU transforms enabled — inserting EnsureTyped(device=%s) early", device)
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    transforms.extend(
        [
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )

    if not OPT_GPU_TRANSFORMS:
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    return Compose(transforms)


def build_postprocessing(preprocessing: Compose, output_dir: str) -> Compose:
    """Build the SwinUNETR postprocessing pipeline.

    Mirrors `bundles/swin_unetr_btcv_segmentation/configs/inference.json`
    `postprocessing#transforms`: Activationsd(softmax) → Invertd(preprocessing) →
    AsDiscreted(argmax) → SaveImaged.

    `preprocessing` is passed BY REFERENCE to Invertd (do not deepcopy — MONAI
    matches inverse transforms by id(transform); see KB
    `vista3d-deepcopy-is-not-generalizable`).
    """
    return Compose(
        [
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(
                keys="pred",
                output_dir=output_dir,
                output_ext=".nii.gz",
                output_dtype=np.float32,
                output_postfix="trans",
                separate_folder=True,
            ),
        ]
    )


def expected_output_path(output_dir: str, image_path: str) -> str:
    """Predict the SaveImaged output path from the input image name.

    SwinUNETR uses SaveImaged with output_postfix="trans",
    separate_folder=True, output_ext=".nii.gz".
    """
    output_ext = ".nii.gz"

    stem = Path(image_path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    return str(Path(output_dir) / stem / f"{stem}_trans{output_ext}")
