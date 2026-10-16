# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MONAI pipeline construction: network, transforms, checkpoint management.

Spleen DeepEdit is an interactive 3D NIfTI segmentation model. It departs
from the canonical 3D-segmentation pattern in two important ways:

  1. Three input channels — the raw CT image plus two click-prompt heatmaps
     (positive/foreground "spleen" clicks and negative/background clicks).
     The two heatmap channels are appended at preprocess time by MONAI's
     ``AddGuidanceSignalDeepEditd`` transform; an empty click list yields an
     all-zero heatmap, which is the "automatic mode" the bundle's
     ``inference.json`` supports out of the box.

  2. Full-volume one-shot inference — the preprocessing pipeline ends in a
     ``Resized([128,128,128])`` step, so the network ingests the entire
     resampled volume in a single forward pass (``SimpleInferer``). There
     is no sliding-window step. The postprocessing ``Invertd`` undoes the
     resize and restores the original spacing/orientation.

  3. The bundle ships one custom transform — ``scripts.transforms.
     OrientationGuidanceMultipleLabelDeepEditd`` (einops-based). Per the
     ``unest-network-is-bundle-shipped-not-monai-nets`` KB pattern,
     ``build_preprocessing`` puts the bundle directory on ``sys.path``
     before importing ``scripts.transforms``.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from config import (
    ENV_PREFIX,
    LABEL_NAMES,
    LOGGER,
    NUMBER_INTENSITY_CH,
    OPT_GPU_TRANSFORMS,
    SPATIAL_SIZE,
)
from huggingface_hub import hf_hub_download
from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    ResizeGuidanceMultipleLabelDeepEditd,
)
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityRanged,
)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def ensure_checkpoint(weights_dir: Path, ckpt_path: Path) -> None:
    """Download model weights from Hugging Face if not present locally."""
    if ckpt_path.exists():
        return

    repo_id = os.environ.get(f"{ENV_PREFIX}_HF_REPO", "MONAI/spleen_deepedit_annotation")
    filename = os.environ.get(f"{ENV_PREFIX}_HF_FILENAME", "models/model.pt")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Checkpoint not found at %s — downloading %s from %s …", ckpt_path, filename, repo_id)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(weights_dir),
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
    """Return cuda:0. Raises RuntimeError if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. spleen_deepedit_annotation requires a ROCm/CUDA GPU (MI300X). "
            "Ensure the GPU is mounted and the ROCm driver is accessible."
        )
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(device: torch.device) -> torch.nn.Module:
    """Instantiate the DeepEdit DynUNet (mirrors bundle inference.json
    ``network_def``).

    ``in_channels = NUMBER_INTENSITY_CH + len(LABEL_NAMES)`` = 1 + 2 = 3.
    The extra two channels are the foreground/background click heatmaps
    appended by ``AddGuidanceSignalDeepEditd`` at preprocess time. Output
    head is 2 channels (background + spleen).
    """
    in_channels = NUMBER_INTENSITY_CH + len(LABEL_NAMES)
    out_channels = len(LABEL_NAMES)
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
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

    The spleen DeepEdit bundle ships a bare ``OrderedDict`` (no wrapper key) —
    the final ``elif isinstance(checkpoint, dict)`` branch handles that.
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
# Bundle-shipped custom transform
# ---------------------------------------------------------------------------


def _import_bundle_orientation_guidance(bundle_dir: Path | None):
    """Import ``scripts.transforms.OrientationGuidanceMultipleLabelDeepEditd``
    from the bundle directory.

    The bundle ships this transform under ``<bundle>/scripts/transforms.py``;
    it is not part of ``monai.apps.deepedit``. We mirror the
    ``unest-network-is-bundle-shipped-not-monai-nets`` pattern: put the
    bundle dir on ``sys.path`` BEFORE the import, then resolve. Caller
    contract: pass ``bundle_dir`` explicitly when the AIM root differs
    from the default ``bundles/spleen_deepedit_annotation`` path (CLI /
    tests / harness); the service threads its own ``self.bundle_dir``.
    """
    if bundle_dir is None:
        env_dir = os.environ.get(f"{ENV_PREFIX}_BUNDLE_DIR")
        if env_dir:
            bundle_dir = Path(env_dir)
        else:
            bundle_dir = Path("bundles/spleen_deepedit_annotation")
    bundle_dir = Path(bundle_dir).resolve()
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    from scripts.transforms import OrientationGuidanceMultipleLabelDeepEditd  # noqa: PLC0415

    return OrientationGuidanceMultipleLabelDeepEditd


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_preprocessing(
    device: torch.device,
    *,
    bundle_dir: Path | None = None,
) -> Compose:
    """Build the preprocessing pipeline (mirrors bundle inference.json
    ``preprocessing_transforms + deepedit_transforms + extra_transforms``).

    The pipeline is a concatenation of three stages:

      1. Standard image preprocessing — load, channel-first, orientation,
         intensity scaling.
      2. DeepEdit click-conditioning — reorient any user-supplied click
         points into the model's RAS frame, encode them as voxel-coord
         guidance points, resize the image to 128^3, rescale the guidance
         points to the resized frame, then synthesise two Gaussian
         heatmap channels and concatenate them onto the image. Empty
         click lists produce all-zero heatmaps (automatic mode).
      3. EnsureTyped — final tensor conversion + device placement.

    GPU transforms optimization: when enabled, ``EnsureTyped(device=...)``
    is inserted right after ``EnsureChannelFirstd`` so the image-only
    portion of the chain (``Orientationd``, ``ScaleIntensityRanged``,
    ``OrientationGuidanceMultipleLabelDeepEditd``) runs on GPU. The
    DeepEdit click transforms downstream (``AddGuidanceFromPointsDeepEditd``,
    ``Resized``, ``ResizeGuidanceMultipleLabelDeepEditd``,
    ``AddGuidanceSignalDeepEditd``) are CPU-only — they operate on
    numpy arrays / unaware torch tensors and pull the image back to
    CPU. A final ``EnsureTyped(device=...)`` therefore runs at the
    tail of the chain on either branch so the output tensor is always
    on the target device.

    Net effect with OPT_GPU_TRANSFORMS=True: two ``EnsureTyped``
    instances (early + late). This is *intentionally* not deduped — the
    early one is a real optimization for the leading image-only
    transforms; the late one is needed to undo the CPU detour the
    click transforms force. With OPT_GPU_TRANSFORMS=False: only the
    late ``EnsureTyped``, matching the bundle's reference behaviour.
    """
    OrientationGuidanceMultipleLabelDeepEditd = _import_bundle_orientation_guidance(bundle_dir)

    transforms: list = [
        LoadImaged(keys="image", image_only=True, reader="ITKReader"),
        EnsureChannelFirstd(keys="image"),
    ]

    if OPT_GPU_TRANSFORMS:
        LOGGER.info("GPU transforms enabled — inserting EnsureTyped(device=%s) early", device)
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    transforms.extend(
        [
            Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            OrientationGuidanceMultipleLabelDeepEditd(
                ref_image="image",
                label_names=LABEL_NAMES,
            ),
            AddGuidanceFromPointsDeepEditd(
                ref_image="image",
                guidance="guidance",
                label_names=LABEL_NAMES,
            ),
            Resized(keys="image", spatial_size=list(SPATIAL_SIZE), mode="area"),
            ResizeGuidanceMultipleLabelDeepEditd(
                guidance="guidance",
                ref_image="image",
            ),
            AddGuidanceSignalDeepEditd(
                keys="image",
                guidance="guidance",
                number_intensity_ch=NUMBER_INTENSITY_CH,
            ),
            EnsureTyped(keys="image", device=device, track_meta=True),
        ]
    )

    return Compose(transforms)


def build_postprocessing(preprocessing: Compose, output_dir: str) -> Compose:
    """Build the postprocessing pipeline (mirrors bundle inference.json
    ``postprocessing#transforms``): ensure-typed → softmax → invert → argmax → save.

    ``Invertd`` must receive the *same* ``preprocessing`` Compose instance
    that was applied to the data — MONAI's invertible transform tracking
    matches by ``id(transform)``. A deep-copy here breaks the match and
    raises ``Error SpatialResample getting the most recently applied
    invertible transform …`` at runtime.

    For DeepEdit specifically, ``Invertd`` reverses the ``Resized(128^3)``
    step (and the upstream orientation flip) so the saved prediction is
    aligned with the input volume's original grid.
    """
    return Compose(
        [
            EnsureTyped(keys="pred"),
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
    """Predict the SaveImaged output path from the input image name."""
    output_ext = ".nii.gz"

    stem = Path(image_path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    return str(Path(output_dir) / stem / f"{stem}_trans{output_ext}")
