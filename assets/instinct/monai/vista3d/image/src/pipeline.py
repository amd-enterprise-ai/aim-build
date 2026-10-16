# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MONAI pipeline construction for the VISTA-3D AIM: network, transforms,
inferer, evaluator loading, and checkpoint management.

VISTA-3D is MONAI's 3D promptable segmentation foundation model. It maps
3D CT volumes to a 132-class semantic mask, supporting two inference modes
in a single forward graph:

  * automatic: ``label_prompt=None`` triggers the bundle's 117-class
    everything-mode subset (``build_everything_labels`` below);
  * interactive class-prompt: ``label_prompt=[<class_ids>]`` (e.g. ``[3]``
    for spleen) runs targeted multi-class segmentation;
  * interactive point-prompt: ``points`` + ``point_labels`` (xyz coords in
    the ORIGINAL image frame, 0/1/-1 = bg/fg/ignore) — single-object only,
    requires ``label_prompt`` of length 1.

The network constructor is ``monai.networks.nets.vista3d132(in_channels=1)``,
loaded via the standard checkpoint helper (the bundle's
``configs/inference.json`` declares ``load_dict={"model": @network}``).

Bundle-shipped components are loaded with ``sys.path.insert(0, bundle_dir)``
(KB ``unest-network-is-bundle-shipped-not-monai-nets``):

  * ``scripts.inferer.Vista3dInferer`` — wraps SlidingWindowInfererAdapt
    plus ``monai.apps.vista3d.inferer.point_based_window_inferer`` so the
    same inferer object can serve both auto and point-prompt modes.
  * ``scripts.evaluator.Vista3dEvaluator`` — SupervisedEvaluator subclass
    that consumes ``hyper_kwargs={user_prompt, everything_labels}`` to wire
    the prompt branching at evaluator.run time.

Preprocessing mirrors the bundle's ``configs/inference.json`` verbatim:

    LoadImaged → EnsureChannelFirstd → [EnsureTyped(device) when GPU xforms]
    → EnsureTyped(track_meta) → Spacingd(1.5mm, bilinear)
    → CropForegroundd(margin=10) → VistaPreTransformd(subclass, labels_dict)
    → ScaleIntensityRanged(-963.82, 1053.68, [0,1], clip)
    → Orientationd(RAS) → CastToTyped(float32)
    → [EnsureTyped(device) when GPU xforms off]

Postprocessing also mirrors the bundle:

    VistaPostTransformd → Invertd(transform=preprocessing, …) → Lambdad(nan→255)
    → SaveImaged(.nii.gz, output_postfix=trans)

Per KB ``vista3d-deepcopy-is-not-generalizable``, we pass ``preprocessing``
to ``Invertd`` BY REFERENCE — not via ``copy.deepcopy``. The VISTA-3D bundle's
own inference.json deepcopies because ``VistaPostTransformd`` normalises the
prediction first, but the by-reference form is strictly safer and works
identically here because the pred has already been normalised before
``Invertd`` sees it.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from config import (
    BUNDLE_NAME,
    ENV_PREFIX,
    HF_TOKEN,
    LOGGER,
    NETWORK_IN_CHANNELS,
    OPT_GPU_TRANSFORMS,
    ROI_SIZE,
    SW_BATCH_SIZE,
    SW_OVERLAP,
    USE_POINT_WINDOW,
)
from huggingface_hub import hf_hub_download
from monai.apps.vista3d.transforms import VistaPostTransformd, VistaPreTransformd
from monai.inferers import Inferer
from monai.networks.nets import vista3d132
from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    Lambdad,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)

# ---------------------------------------------------------------------------
# Bundle-shipped components loader
#
# The bundle's ``scripts/inferer.py`` and ``scripts/evaluator.py`` use
# top-level imports like ``from scripts.inferer import Vista3dInferer`` that
# only resolve when the bundle root is on ``sys.path``. We expose a single
# helper so callers (service, tests, benchmark, harness) all use the same
# insertion point. Same pattern as VISTA-2D / renalStructures_UNEST.
# ---------------------------------------------------------------------------


def _ensure_bundle_on_syspath(bundle_dir: Path) -> None:
    """Insert ``bundle_dir`` at the front of ``sys.path`` if not already there."""
    bundle_str = str(Path(bundle_dir).resolve())
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)


def load_bundle_components(bundle_dir: Path):
    """Import the bundle's custom ``Vista3dInferer`` + ``Vista3dEvaluator``."""
    _ensure_bundle_on_syspath(bundle_dir)
    from scripts.evaluator import Vista3dEvaluator
    from scripts.inferer import Vista3dInferer

    return Vista3dInferer, Vista3dEvaluator


# ---------------------------------------------------------------------------
# VISTA-3D metadata helpers
#
# These are specific to VISTA-3D's 132-class prompt-driven architecture and
# are read straight from the bundle's ``configs/inference.json``.
# ---------------------------------------------------------------------------


def build_subclass() -> dict[str, list[int]]:
    """Return the bundle's class → sub-class expansion table.

    The keys map a coarse VISTA-3D class to the global IDs of its
    sub-structures (e.g. ``"2"`` = kidney → ``[14, 5]`` = left/right
    kidney). VistaPreTransformd consumes this when ``label_prompt``
    references one of the coarse classes.

    Source: ``app/bundles/vista3d/configs/inference.json#subclass``.
    """
    return {
        "2": [14, 5],
        "20": [28, 29, 30, 31, 32],
        "21": list(range(33, 57)) + list(range(63, 98)) + [114, 120, 122],
    }


def build_everything_labels() -> list[int]:
    """Return the 117-class "everything" subset used when ``label_prompt`` is None.

    Equivalent to the bundle's ``configs/inference.json#everything_labels``:
    every global ID 1..132 EXCEPT the special classes (kidney/lung/bone
    composites, lesion classes, and out-of-range placeholders) that would
    duplicate or over-segment the result in automatic mode.
    """
    excluded = {2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132}
    return sorted(set(range(1, 133)) - excluded)


def load_labels_dict(bundle_dir: Path) -> dict:
    """Read ``channel_def`` from the bundle's ``configs/metadata.json``.

    Returns the ``{<string-int-class-id>: <label name>}`` map for all 133
    entries (background + 132 anatomical labels). VistaPreTransformd needs
    the keys to validate label_prompt; the service's /v1/models reflects
    the same map via ``config.CHANNEL_DEF``.
    """
    metadata_path = bundle_dir / "configs" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing bundle metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    return metadata["network_data_format"]["outputs"]["pred"]["channel_def"]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def ensure_checkpoint(weights_dir: Path, ckpt_path: Path) -> None:
    """Download VISTA-3D weights from Hugging Face if not present locally.

    VISTA-3D ships a single checkpoint ``models/model.pt`` with the
    encoder + decoder + prompt encoder weights. The bundle's
    ``configs/inference.json`` declares ``load_dict={"model": @network}``,
    so this is loaded via the shared ``load_checkpoint`` helper's ``model``
    branch.

    Weights are governed by the NVIDIA OneWay Noncommercial License.
    ``HF_TOKEN`` (from ``config.py``) must be set to a Hugging Face token
    whose account has accepted the license at
    https://huggingface.co/MONAI/vista3d.
    """
    if ckpt_path.exists():
        return

    if not HF_TOKEN:
        raise RuntimeError(
            "VISTA-3D model weights are governed by the NVIDIA OneWay Noncommercial "
            "License. Set the HF_TOKEN environment variable to a Hugging Face token "
            "whose account has accepted the license at "
            "https://huggingface.co/MONAI/vista3d before starting the service."
        )

    repo_id = os.environ.get(f"{ENV_PREFIX}_HF_REPO", f"MONAI/{BUNDLE_NAME}")
    filename = os.environ.get(f"{ENV_PREFIX}_HF_FILENAME", "models/model.pt")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Checkpoint not found at %s — downloading %s from %s …",
        ckpt_path,
        filename,
        repo_id,
    )
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(weights_dir),
        token=HF_TOKEN,
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
            f"No GPU detected. {BUNDLE_NAME} requires a ROCm/CUDA GPU (MI300X). "
            "Ensure the GPU is mounted and the ROCm driver is accessible."
        )
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(device: torch.device) -> torch.nn.Module:
    """Instantiate the ``vista3d132`` architecture (in_channels=1) and move to device.

    ``vista3d132`` lives in ``monai.networks.nets`` (no bundle-side scripts
    dependency for construction — unlike UNEST/CellSamWrapper). The "132" in
    the name refers to the maximum class count; the actual classes returned
    are gated by the ``label_prompt`` at inference time.
    """
    return vista3d132(in_channels=NETWORK_IN_CHANNELS).to(device)


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

    VISTA-3D's ``model.pt`` matches the ``"model"`` branch (the bundle's
    ``configs/inference.json`` declares ``load_dict={"model": @network}``).
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
        "torch.compile returned in %.1fs (actual compile cost deferred to first forward)",
        time.perf_counter() - t0,
    )
    return compiled


# ---------------------------------------------------------------------------
# Inferer
# ---------------------------------------------------------------------------


def build_inferer(bundle_dir: Path | None = None) -> Inferer:
    """Construct the bundle's ``Vista3dInferer`` with the inference.json knobs.

    ``Vista3dInferer`` ships in ``<bundle>/scripts/inferer.py``. It wraps
    ``SlidingWindowInfererAdapt`` (for class-prompt/everything-mode forwards)
    and ``monai.apps.vista3d.inferer.point_based_window_inferer`` (for the
    point-prompt branch). The knob values mirror the bundle's
    ``configs/inference.json#inferer`` block verbatim.
    """
    if bundle_dir is None:
        bundle_dir = Path(os.environ.get(f"{ENV_PREFIX}_BUNDLE_DIR", f"bundles/{BUNDLE_NAME}"))
    bundle_dir = Path(bundle_dir).resolve()

    Vista3dInferer, _ = load_bundle_components(bundle_dir)
    return Vista3dInferer(
        roi_size=ROI_SIZE,
        overlap=SW_OVERLAP,
        sw_batch_size=SW_BATCH_SIZE,
        use_point_window=USE_POINT_WINDOW,
    )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_preprocessing(
    device: torch.device,
    subclass: dict[str, list[int]],
    labels_dict: dict,
) -> Compose:
    """Build the preprocessing pipeline (mirrors bundle's
    ``configs/inference.json#preprocessing_transforms`` exactly).

    GPU transforms optimization: when enabled, data is moved to GPU early
    (right after EnsureChannelFirstd) so subsequent transforms run on GPU.
    The default ``EnsureTyped(track_meta=True)`` insertion-point matches
    the bundle's own placement so per-step ApplyTransform/Invertd matching
    by ``id(transform)`` keeps working downstream.
    """
    transforms: list = [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if OPT_GPU_TRANSFORMS:
        LOGGER.info(
            "GPU transforms enabled — inserting EnsureTyped(device=%s) early",
            device,
        )
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    transforms.extend(
        [
            Spacingd(keys="image", pixdim=[1.5, 1.5, 1.5], mode="bilinear"),
            CropForegroundd(
                keys="image",
                allow_smaller=True,
                margin=10,
                source_key="image",
            ),
            VistaPreTransformd(
                keys="image",
                subclass=subclass,
                labels_dict=labels_dict,
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=-963.8247715525971,
                a_max=1053.678477684517,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys="image", axcodes="RAS"),
            CastToTyped(keys="image", dtype=torch.float32),
        ]
    )

    if not OPT_GPU_TRANSFORMS:
        transforms.append(EnsureTyped(keys="image", device=device, track_meta=True))

    return Compose(transforms)


def build_postprocessing(
    preprocessing: Compose,
    output_dir: str,
) -> Compose:
    """Build the postprocessing pipeline (mirrors bundle's
    ``configs/inference.json#postprocessing#transforms``).

    Chain:

        VistaPostTransformd → Invertd → Lambdad(nan→255) → SaveImaged

    ⚠ KNOWN PITFALL — ``Invertd + preprocessing``
    We pass ``preprocessing`` BY REFERENCE (no ``copy.deepcopy``). The
    bundle's own inference.json ``deepcopy``s the preprocessing Compose
    before handing it to Invertd, and that *works* in the bundle only
    because VistaPostTransformd runs first and normalises the prediction
    before Invertd's id-based matching gets exercised. The by-reference
    form is strictly safer (every other AIM in this repo uses it) and
    behaves identically here because of the same Vista3dPost-first
    ordering. See KB ``vista3d-deepcopy-is-not-generalizable``.
    """
    return Compose(
        [
            VistaPostTransformd(keys="pred"),
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
            Lambdad(keys="pred", func=lambda x: torch.nan_to_num(x, nan=255)),
            SaveImaged(
                keys="pred",
                resample=False,
                output_dir=output_dir,
                output_ext=".nii.gz",
                output_dtype=np.float32,
                output_postfix="trans",
                separate_folder=True,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Output-path predictor
# ---------------------------------------------------------------------------


def expected_output_path(output_dir: str, image_path: str) -> str:
    """Predict the ``SaveImaged`` output path from the input image name.

    ``SaveImaged`` with ``separate_folder=True`` writes
    ``<output_dir>/<stem>/<stem>_trans.nii.gz``. Same logic as every
    other 3D-segmentation AIM in this repo.
    """
    stem = Path(image_path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    return str(Path(output_dir) / stem / f"{stem}_trans.nii.gz")
