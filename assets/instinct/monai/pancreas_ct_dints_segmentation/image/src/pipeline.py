# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MONAI pipeline construction: network, transforms, checkpoint management.

Pancreas DiNTS is a 3D NIfTI segmentation model with a non-standard
architecture story:

* The bundle's ``configs/inference.yaml`` constructs a ``TopologyInstance``
  from a saved architecture artifact (``models/search_code_18590.pt``) that
  encodes the cell-level operations and node connectivity discovered by
  Differentiable Neural Architecture Search (DiNTS). The ``DiNTS`` network
  then wraps that topology and exposes the standard 3D segmentation forward
  signature.
* Both the model weights (``models/model.pt``) AND the architecture artifact
  must be present at service-startup time. ``ensure_checkpoint`` fetches both
  from Hugging Face if they are missing locally.
* Preprocessing / postprocessing follow the bundle's inference.yaml exactly:
  Spacingd 1mm iso, ScaleIntensityRanged [-87, 199] -> [0, 1], sliding-window
  inference 96x96x96 with overlap 0.625, softmax -> invert -> argmax -> save.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from config import LOGGER, OPT_GPU_TRANSFORMS
from huggingface_hub import hf_hub_download
from monai.networks.nets import DiNTS, TopologyInstance
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
# Bundle paths and constants
# ---------------------------------------------------------------------------

ARCH_FILENAME = "search_code_18590.pt"
WEIGHTS_FILENAME = "model.pt"

INPUT_CHANNELS = 1
OUTPUT_CLASSES = 3


def _arch_ckpt_path(weights_dir: Path) -> Path:
    return weights_dir / "models" / ARCH_FILENAME


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def ensure_checkpoint(weights_dir: Path, ckpt_path: Path) -> None:
    """Download model weights + architecture artifact from HF if missing.

    DiNTS requires two files in ``<weights_dir>/models/``:
      * ``model.pt``               — model state_dict (torch weights)
      * ``search_code_18590.pt``   — architecture (arch_code_a, arch_code_c, node_a)

    Neither is baked into the image: the Dockerfile deletes both in the same
    layer as the bundle download, so they are re-fetched here at service
    startup.
    """
    repo_id = os.environ.get("PANCREAS_DINTS_HF_REPO", "MONAI/pancreas_ct_dints_segmentation")
    weights_filename = os.environ.get(
        "PANCREAS_DINTS_HF_FILENAME",
        f"models/{WEIGHTS_FILENAME}",
    )
    arch_filename = os.environ.get(
        "PANCREAS_DINTS_HF_ARCH_FILENAME",
        f"models/{ARCH_FILENAME}",
    )

    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        LOGGER.info(
            "Weights not found at %s — downloading %s from %s …",
            ckpt_path,
            weights_filename,
            repo_id,
        )
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            local_dir=str(weights_dir),
        )
        LOGGER.info("Downloaded weights to %s", downloaded)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Download succeeded but weights not at expected path: {ckpt_path}. " f"Downloaded file: {downloaded}"
            )

    arch_path = _arch_ckpt_path(weights_dir)
    if not arch_path.exists():
        LOGGER.info(
            "Architecture artifact not found at %s — downloading %s from %s …",
            arch_path,
            arch_filename,
            repo_id,
        )
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=arch_filename,
            local_dir=str(weights_dir),
        )
        LOGGER.info("Downloaded architecture artifact to %s", downloaded)
        if not arch_path.exists():
            raise FileNotFoundError(
                f"Download succeeded but architecture artifact not at expected path: "
                f"{arch_path}. Downloaded file: {downloaded}"
            )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def build_device() -> torch.device:
    """Return cuda:0. Raises RuntimeError if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. pancreas_ct_dints_segmentation requires a ROCm/CUDA GPU (MI300X). "
            "Ensure the GPU is mounted and the ROCm driver is accessible."
        )
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(device: torch.device, weights_dir: Path | None = None) -> torch.nn.Module:
    """Instantiate the DiNTS network from the saved architecture artifact.

    Mirrors bundle ``configs/inference.yaml``:

      dints_space = TopologyInstance(channel_mul=1, num_blocks=12, num_depths=4,
                                     use_downsample=True,
                                     arch_code=[arch_code_a, arch_code_c],
                                     device=device)
      network    = DiNTS(dints_space=dints_space,
                         in_channels=1, num_classes=3, use_downsample=True,
                         node_a=torch.from_numpy(arch_ckpt['node_a'])).to(device)

    The architecture artifact is a downloaded weight, not bundle content, so it is
    resolved from ``<weights_dir>/models/search_code_18590.pt`` on the cache volume.
    When ``weights_dir`` is None we fall back to ``$AIM_CACHE_PATH`` (defaulting to
    ``/workspace/model-cache/pancreas_ct_dints_segmentation``), matching where
    ``ensure_checkpoint`` puts it. ``PANCREAS_DINTS_BUNDLE_DIR`` still points at the
    read-only bundle content in the image and is unrelated to this path.
    """
    if weights_dir is None:
        cache_root = Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache"))
        weights_dir = cache_root / "pancreas_ct_dints_segmentation"
    weights_dir = Path(weights_dir).resolve()

    arch_path = _arch_ckpt_path(weights_dir)
    if not arch_path.exists():
        raise FileNotFoundError(
            f"DiNTS architecture artifact not found at {arch_path}. " "Run ensure_checkpoint() before build_network()."
        )

    map_loc = device if device.type == "cuda" else torch.device("cpu")
    # weights_only=False is required: the arch artifact is a pickled dict
    # containing numpy arrays (arch_code_a / arch_code_c / node_a) and the
    # default-True loader in PyTorch >=2.6 refuses to unpickle numpy globals.
    # The artifact ships as part of the trusted MONAI bundle distribution.
    arch_ckpt = torch.load(str(arch_path), map_location=map_loc, weights_only=False)

    arch_code = [arch_ckpt["arch_code_a"], arch_ckpt["arch_code_c"]]
    node_a = torch.from_numpy(arch_ckpt["node_a"])

    dints_space = TopologyInstance(
        channel_mul=1,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        arch_code=arch_code,
        device=device,
    )
    network = DiNTS(
        dints_space=dints_space,
        in_channels=INPUT_CHANNELS,
        num_classes=OUTPUT_CLASSES,
        use_downsample=True,
        node_a=node_a,
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
    """Load weights into ``network`` in place. Handles the common state-dict
    wrappings ("model", "state_dict", "net") plus a bare state-dict.

    Shared between the HTTP service and the internal benchmark script so the
    two never diverge on checkpoint handling.

    The DiNTS bundle ships its weights under the top-level ``"model"`` key
    (mirroring its bundle ``CheckpointLoader.load_dict={'model': @network}``);
    the ``elif "model" in checkpoint`` branch handles that.
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

    KNOWN ISSUE — DiNTS + torch.compile.
    The DiNTS-searched topology + 3-channel one-hot SlidingWindowInferer
    (roi=(96,96,96), sw_batch_size=4, overlap=0.625) is hostile to
    dynamo / Inductor on ROCm: every PANCREAS_DINTS_COMPILE=true combination
    returned HTTP 500 on the first warmup request during benchmarking.
    The unit/GPU suite's ``test_compiled_forward_shape`` exercises compile=true
    on a representative (1,1,96,96,96) input under autocast — treat any
    failure there as a strong signal. Compile remains OFF by default;
    toggle via PANCREAS_DINTS_COMPILE.
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
    """Build the preprocessing pipeline (mirrors bundle inference.yaml
    ``preprocessing#transforms``).

    GPU transforms optimization: when enabled, data is moved to GPU early
    (right after EnsureChannelFirstd) so subsequent transforms run on GPU.
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
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0], mode="bilinear"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-87,
                a_max=199,
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
    """Build the postprocessing pipeline (mirrors bundle inference.yaml
    ``postprocessing#transforms``): softmax -> invert -> argmax -> save.

    ``Invertd`` must receive the *same* ``preprocessing`` Compose instance
    that was applied to the data — MONAI's invertible transform tracking
    matches by ``id(transform)``. A deep-copy here breaks the match and
    raises ``Error SpatialResample getting the most recently applied
    invertible transform …`` at runtime. See KB
    `vista3d-deepcopy-is-not-generalizable`.
    """
    return Compose(
        [
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
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
