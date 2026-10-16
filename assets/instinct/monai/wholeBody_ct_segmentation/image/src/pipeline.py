# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MONAI pipeline construction: network, transforms, checkpoint management.

wholeBody_ct_segmentation is a standard 3D NIfTI segmentation model:
- SegResNet network from monai.networks.nets (init_filters=32, blocks_down=[1,2,2,4],
  blocks_up=[1,1,1], dropout=0.2, 105 output channels)
- Default NIfTI reader for LoadImaged (matches bundle inference.json)
- Stock MONAI transforms; no custom bundle scripts
- Two model variants: ``model.pt`` (highres, pixdim 1.5 mm — default) and
  ``model_lowres.pt`` (lowres, pixdim 3.0 mm). The variant is picked via the
  ``WHOLEBODY_CT_HIGHRES`` env var (config.HIGHRES). Mirrors the bundle's
  ``displayable_configs#highres`` toggle.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from config import HIGHRES, LOGGER, OPT_GPU_TRANSFORMS
from huggingface_hub import hf_hub_download
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SaveImaged,
    ScaleIntensityd,
    Spacingd,
)

# ---------------------------------------------------------------------------
# Variant helpers
# ---------------------------------------------------------------------------


def checkpoint_filename(highres: bool = HIGHRES) -> str:
    """Return the bundle-relative checkpoint filename for the active variant."""
    return "models/model.pt" if highres else "models/model_lowres.pt"


def pixdim(highres: bool = HIGHRES) -> list[float]:
    """Return the Spacingd target pixdim for the active variant.

    Matches bundle inference.json:
      ``"pixdim": "$[1.5, 1.5, 1.5] if @displayable_configs#highres else [3.0, 3.0, 3.0]"``
    """
    return [1.5, 1.5, 1.5] if highres else [3.0, 3.0, 3.0]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def ensure_checkpoint(weights_dir: Path, ckpt_path: Path) -> None:
    """Download model weights from Hugging Face if not present locally.

    The bundle hosts both variant checkpoints under the same HF repo
    (``MONAI/wholeBody_ct_segmentation``). ``ckpt_path`` points at the
    variant-specific filename (model.pt or model_lowres.pt); we derive the
    in-repo path by reading its name.
    """
    if ckpt_path.exists():
        return

    repo_id = os.environ.get("WHOLEBODY_CT_HF_REPO", "MONAI/wholeBody_ct_segmentation")
    filename = os.environ.get("WHOLEBODY_CT_HF_FILENAME", f"models/{ckpt_path.name}")

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
            "No GPU detected. wholeBody_ct_segmentation requires a ROCm/CUDA GPU (MI300X). "
            "Ensure the GPU is mounted and the ROCm driver is accessible."
        )
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(device: torch.device) -> torch.nn.Module:
    """Instantiate the wholeBody SegResNet (matches bundle inference.json network_def).

    Architecture matches both variants (highres and lowres share the same
    network definition; only weights and Spacingd pixdim differ).
    """
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
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

    The wholeBody bundle ships a bare ``OrderedDict`` (no wrapper key) — the
    final ``elif isinstance(checkpoint, dict)`` branch handles that.
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


def build_preprocessing(device: torch.device) -> Compose:
    """Build the preprocessing pipeline (mirrors bundle inference.json
    ``preprocessing#transforms``).

    Order: LoadImaged → EnsureChannelFirstd → EnsureTyped (early, w/
    optional GPU placement) → Orientationd(RAS) → Spacingd(pixdim) →
    NormalizeIntensityd(nonzero=True) → ScaleIntensityd(-1.0, 1.0).

    GPU transforms optimization: when enabled, data is moved to GPU early
    (right after EnsureChannelFirstd at index 2) so subsequent transforms
    run on GPU. When disabled, ``EnsureTyped`` is appended at the end of
    the chain (late CPU→device hop).

    Note: the upstream bundle inserts ``EnsureTyped`` before
    ``EnsureChannelFirstd``. The order swap here is functionally
    equivalent (channel-dim placement and device placement commute on
    these tensors) and aligns with the universal AIM template pattern,
    where ``test_gpu_transforms_moves_device_early`` asserts on the
    fixed index-2 location.
    """
    transforms: list = [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if OPT_GPU_TRANSFORMS:
        LOGGER.info("GPU transforms enabled — inserting EnsureTyped(device=%s) early", device)
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    transforms.extend(
        [
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=pixdim(), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]
    )

    if not OPT_GPU_TRANSFORMS:
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    return Compose(transforms)


def build_postprocessing(preprocessing: Compose, output_dir: str) -> Compose:
    """Build the postprocessing pipeline (mirrors bundle inference.json
    ``postprocessing#transforms``): softmax → argmax → invert → save.

    ``Invertd`` must receive the *same* ``preprocessing`` Compose instance
    that was applied to the data — MONAI's invertible transform tracking
    matches by ``id(transform)``. A deep-copy here breaks the match and
    raises ``Error SpatialResample getting the most recently applied
    invertible transform …`` at runtime. (See KB entry
    ``vista3d-deepcopy-is-not-generalizable``.)

    The bundle uses ``nearest_interp=True`` for ``Invertd`` because the
    post-argmax discrete labels must not be linearly interpolated.
    """
    return Compose(
        [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
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
