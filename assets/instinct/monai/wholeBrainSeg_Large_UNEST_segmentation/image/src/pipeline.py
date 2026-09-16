# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MONAI pipeline construction: network, transforms, checkpoint management.

wholeBrainSeg_Large_UNEST_segmentation is a 3D NIfTI single-channel T1w
MRI 133-class brain parcellation model with a transformer-based,
bundle-shipped network.

Architectural notes (pulled from the bundle's ``configs/inference.json``
and bundle ``scripts/networks/``):

* The network class ``UNesT`` lives at
  ``<bundle_dir>/scripts/networks/unest_base_patch_4.py`` (NOT
  ``unest.py`` — the renal-UNEST sibling AIM uses a different leaf
  module name; see KB ``unest-network-is-bundle-shipped-not-monai-nets``).
  It pulls in three sibling helper files
  (``nest_transformer_3D.py``, ``unest_block.py``, ``patchEmbed3D.py``)
  via relative ``scripts.networks.*`` imports, so ``build_network``
  needs ``sys.path.insert(0, str(bundle_dir))`` before the import.
* The bundle's UNesT constructor is the **3-level** variant:
  ``in_channels=1, out_channels=133, patch_size=4,
  depths=[2,2,8], embed_dim=[128,256,512], num_heads=[4,8,16]``. The
  internal ``NestTransformer3D`` hardcodes ``img_size=96``, so the
  ``SlidingWindowInferer`` ROI MUST stay at ``96^3`` (the bundle's own
  ``inferer.roi_size`` is also fixed at ``[96,96,96]``).
* Preprocessing is *minimal*: ``LoadImaged → EnsureChannelFirstd →
  NormalizeIntensityd(nonzero=True, channel_wise=True) → EnsureTyped``.
  No ``Spacingd``, no ``Orientationd``, no intensity-clip step. The
  bundle README is explicit that **inputs must already be in MNI305
  space**; if they are not, registration is the user's responsibility
  (see the bundle README §Important and the high-severity human_review
  entry on this AIM).
* Postprocessing matches the bundle exactly: softmax → invert → argmax
  → save NIfTI.
* The checkpoint ships under the top-level ``"model"`` key
  (``CheckpointLoader.load_dict={"model": @network}``); the shared
  ``load_checkpoint`` helper handles that wrapper natively.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from config import LOGGER, OPT_GPU_TRANSFORMS
from huggingface_hub import hf_hub_download
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    SaveImaged,
)

# ---------------------------------------------------------------------------
# Bundle constants
# ---------------------------------------------------------------------------

INPUT_CHANNELS = 1
OUTPUT_CLASSES = 133  # background + 132 anatomical brain regions


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def ensure_checkpoint(bundle_dir: Path, ckpt_path: Path) -> None:
    """Download model weights from Hugging Face if not present locally."""
    if ckpt_path.exists():
        return

    repo_id = os.environ.get("BRAINSEG_HF_REPO", "MONAI/wholeBrainSeg_Large_UNEST_segmentation")
    filename = os.environ.get("BRAINSEG_HF_FILENAME", "models/model.pt")

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


def build_network(device: torch.device, bundle_dir: Path | None = None) -> torch.nn.Module:
    """Instantiate the UNesT-Large network from the bundle's scripts/.

    Mirrors bundle ``configs/inference.json`` ``network_def``:

        scripts.networks.unest_base_patch_4.UNesT(
            in_channels=1,
            out_channels=133,
            patch_size=4,
            depths=[2, 2, 8],
            embed_dim=[128, 256, 512],
            num_heads=[4, 8, 16],
        )

    The bundle's UNesT-Large variant is 3-level (depths length=3, embed_dim
    length=3), patch_size=4 — distinct from the renal-UNEST sibling AIM
    which uses the 4-level variant. UNesT internally hardcodes
    ``img_size=96`` so the ``SlidingWindowInferer`` ROI MUST stay at
    96^3 (the bundle's ``inferer.roi_size`` is also fixed at [96,96,96]).

    The bundle ships UNesT and three sibling helper modules
    (``nest_transformer_3D.py``, ``unest_block.py``, ``patchEmbed3D.py``)
    under ``<bundle_dir>/scripts/networks/``; they use relative
    ``scripts.networks.*`` imports, so we must put ``bundle_dir`` on
    ``sys.path`` before importing. When ``bundle_dir`` is ``None`` we
    fall back to the ``BRAINSEG_BUNDLE_DIR`` env var (default
    ``bundles/wholeBrainSeg_Large_UNEST_segmentation``), which keeps test
    fixtures that only pass ``device`` working.

    KB references: ``unest-network-is-bundle-shipped-not-monai-nets``.
    """
    if bundle_dir is None:
        bundle_dir = Path(
            os.environ.get(
                "BRAINSEG_BUNDLE_DIR",
                "bundles/wholeBrainSeg_Large_UNEST_segmentation",
            )
        )
    bundle_dir = Path(bundle_dir).resolve()

    bundle_dir_str = str(bundle_dir)
    if bundle_dir_str not in sys.path:
        sys.path.insert(0, bundle_dir_str)
    from scripts.networks.unest_base_patch_4 import UNesT  # type: ignore[import-not-found]  # noqa: PLC0415

    network = UNesT(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CLASSES,
        patch_size=4,
        depths=[2, 2, 8],
        embed_dim=[128, 256, 512],
        num_heads=[4, 8, 16],
    ).to(device)
    return network


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    network: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> None:
    """Load weights into ``network`` in place.

    Handles the common state-dict wrappings ("model", "state_dict", "net")
    plus a bare state-dict. The wholeBrainSeg_Large_UNEST_segmentation
    bundle ships its weights under the top-level ``"model"`` key (its
    bundle ``CheckpointLoader.load_dict={"model": @network}``); the
    ``elif "model" in checkpoint`` branch handles that.
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
    service and the internal benchmark script. Returns ``network``
    unchanged when compile is disabled or the model is not on CUDA.

    UNesT compile-compat has been validated on the renal-UNEST sibling
    AIM under ROCm 7.2 + PyTorch 2.9.1 + MI300X (KB
    ``unest-torch-compile-rocm-inductor-ok-on-mi300x``); a 96^3
    autocast forward through the compiled graph passes there. The
    wholeBrainSeg variant is the same architecture family with a
    different head depth (3-level vs 4-level) and 133 output channels;
    we expect compile to remain compatible. The unit/GPU suite's
    ``test_compiled_forward_shape`` exercises compile=true on a
    representative ``(1,1,96,96,96)`` input under autocast — Tuner
    should treat any failure there as a stronger signal than HTTP-500
    in the sweep. Compile remains OFF by default; Tuner toggles it via
    the ``BRAINSEG_COMPILE`` env var.
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

    The bundle's preprocessing chain is *minimal*:

      1. ``LoadImaged(keys="image")`` — read NIfTI from disk.
      2. ``EnsureChannelFirstd(keys="image")`` — promote single-channel
         volume to (C=1, H, W, D).
      3. ``NormalizeIntensityd(keys="image", nonzero=True,
         channel_wise=True)`` — zero-mean / unit-std intensity
         normalization computed only over non-zero voxels (the bundle
         expects brain-extracted T1w MRI, so non-zero ≈ brain). This is
         the only intensity step; there is **no** clip / rescale to a
         fixed Hounsfield range like CT bundles use.
      4. ``EnsureTyped(keys="image", device=device)`` — coerce to torch
         tensor on the chosen device.

    There is **no Spacingd, no Orientationd, no Resized** — the bundle
    explicitly assumes inputs are already in MNI305 space at the
    bundle's expected grid (see bundle README §Important and the
    high-severity human_review entry recorded for this AIM). Any
    upstream registration is the user's responsibility.

    GPU transforms optimization: when enabled, data is moved to GPU
    early (right after ``EnsureChannelFirstd`` at index 2) so the
    ``NormalizeIntensityd`` runs on GPU; when disabled, ``EnsureTyped``
    is appended at the end of the chain (late CPU→device hop). The
    bundle's own trailing ``EnsureTyped`` (no device arg) is replaced
    by the device-aware variant so the GPU transforms optimization can
    do real work.
    """
    transforms: list = [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if OPT_GPU_TRANSFORMS:
        LOGGER.info("GPU transforms enabled — inserting EnsureTyped(device=%s) early", device)
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    transforms.append(
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    )

    if not OPT_GPU_TRANSFORMS:
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    return Compose(transforms)


def build_postprocessing(preprocessing: Compose, output_dir: str) -> Compose:
    """Build the postprocessing pipeline (mirrors bundle inference.json
    ``postprocessing#transforms``): softmax → invert-preprocessing →
    argmax → save.

    Order matches the upstream bundle verbatim:

      1. ``Activationsd(softmax=True)`` — normalises 133-channel logits
         to per-voxel class probabilities.
      2. ``Invertd`` — undoes any spatial transforms in preprocessing.
         For wholeBrainSeg the bundle's preprocessing has no spatial
         transforms, so Invertd is a no-op spatially; it still copies
         the prediction back onto the original-image meta_dict so the
         saved NIfTI carries the input's affine. ``nearest_interp=False``
         matches the bundle (probabilities are continuous).
      3. ``AsDiscreted(argmax=True)`` — collapses the 133-channel
         probabilities to a single-channel integer label map in
         ``{0, ..., 132}``.
      4. ``SaveImaged`` — writes the per-case NIfTI mask under
         ``output_dir``. Default conventions
         (``output_postfix="trans"``, ``separate_folder=True``,
         ``output_dtype=float32``) match every other MONAI segmentation
         AIM so ``expected_output_path`` can predict the saved path.

    ⚠ KNOWN PITFALL (KB ``vista3d-deepcopy-is-not-generalizable``):
    ``Invertd`` must receive the *same* ``preprocessing`` Compose
    instance that was applied to the data — MONAI matches invertible
    transforms by ``id(transform)``. A deep-copy here breaks the match
    and raises ``RuntimeError: Error SpatialResample getting the most
    recently applied invertible transform``.
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
                resample=False,
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
