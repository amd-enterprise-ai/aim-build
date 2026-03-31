#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Pre-build AITER JIT attention kernels for vLLM 0.16 profiles.

AITER dynamically compiles (JIT) MHA attention kernels on first use,
causing 25-40s stalls per kernel variant. This script pre-builds all
required variants via cross-compilation, so they can be shipped in the
container image and eliminate runtime JIT entirely.

Two kernel types are built:
  - mha_varlen_fwd_*: 9 variants covering all dtype × causal × LSE
    combinations used by vLLM 0.16 model profiles.
  - module_fmha_v3_varlen_fwd: Flash Attention v3 ASM fast-path for
    bf16 models with head_dim=128. Uses hand-written GPU ASM, so it
    must be compiled separately per architecture (cannot be fat binary).

Each kernel build is isolated in its own subprocess for two reasons:
  1. AITER caches loaded modules in sys.modules; a fresh process ensures
     the JIT decorator triggers compilation for each target architecture.
  2. Cross-compiled kernels may segfault on dispatch (e.g. gfx950 code
     on gfx942 hardware). Subprocess isolation ensures one crash doesn't
     prevent building the remaining kernels — the .so is already on disk
     before dispatch occurs.

Requires:
  - A GPU with ROCm (tensor creation needs CUDA device)
  - AITER installed (ships with the vLLM ROCm base image)

Usage:
    # Auto-detect architectures from all profile filenames:
    python3 prebuild_aiter_kernels.py --profile-dir assets/

    # Build for specific architectures (overrides auto-detection):
    python3 prebuild_aiter_kernels.py --archs gfx942 gfx950

    # Custom output directory:
    python3 prebuild_aiter_kernels.py --output-dir ./prebuilt

    # List target kernels without building:
    python3 prebuild_aiter_kernels.py --list
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# GPU model name (as used in profile filenames) → ROCm gfx architecture.
# Keep in sync with GPU_TO_GFX_ARCH in src/aim_runtime/gpu_detector.py
GPU_MODEL_TO_GFX = {
    "mi300a": "gfx942",
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi350x": "gfx950",
    "mi355x": "gfx950",
}

DEFAULT_ARCHS = ["gfx942", "gfx950"]


def infer_archs_from_profiles(assets_dir: Path) -> list[str]:
    """Derive the set of gfx architectures from GPU models in profile filenames."""
    archs: set[str] = set()
    for f in assets_dir.rglob("profiles/**/*.yaml"):
        for part in f.stem.split("-"):
            gfx = GPU_MODEL_TO_GFX.get(part)
            if gfx:
                archs.add(gfx)
    return sorted(archs) if archs else DEFAULT_ARCHS


def _get_aiter_jit_dir() -> Path:
    """Dynamically resolve the AITER JIT directory from the installed package.

    This avoids hard-coding paths that can break across Python versions,
    site-packages vs dist-packages layouts, or virtualenv/conda installs.
    """
    try:
        import aiter

        aiter_pkg_path = Path(aiter.__file__).parent
        jit_dir = aiter_pkg_path / "jit"
        if jit_dir.is_dir():
            return jit_dir
        raise FileNotFoundError(f"AITER jit directory not found at {jit_dir}")
    except ImportError as e:
        raise RuntimeError("AITER package not found. Ensure it is installed.") from e


# Each tuple defines one mha_varlen_fwd variant to build.
# Fields: (dtype, is_causal, return_softmax_lse, min_seqlen_q, label)
#
# The kernel name encodes these flags:
#   mha_varlen_fwd_{dtype}_{logits}_{bias}_{causal}_{lse}_{dropout}_{skip}_{qscale}
#
# All v0.16 profiles use: nlogits, nbias, ndropout, nqscale (fixed).
# The varying flags are: dtype (fp16/bf16), causal (mask/nmask),
# lse (lse/nlse), and skip (skip if min_seqlen_q>0, nskip if 0).
MHA_VARIANTS = [
    # fp16 attention — used by fp16 model profiles
    ("fp16", True, False, 128, "fp16_mask_nlse"),
    ("fp16", True, True, 128, "fp16_mask_lse"),
    ("fp16", False, True, 128, "fp16_nmask_lse"),
    ("fp16", False, False, 128, "fp16_nmask_nlse"),
    # bf16 attention — used by bf16 and fp8 profiles (fp8 runs attention in bf16)
    ("bf16", True, False, 128, "bf16_mask_nlse"),
    ("bf16", True, True, 128, "bf16_mask_lse"),
    ("bf16", False, True, 128, "bf16_nmask_lse"),
    ("bf16", False, False, 128, "bf16_nmask_nlse"),
    ("bf16", False, False, 0, "bf16_nmask_nlse_nskip"),  # gemma-3 (min_seqlen_q=0)
]

# Expected .so filenames corresponding to MHA_VARIANTS (same order).
MHA_SO_NAMES = [
    "mha_varlen_fwd_fp16_nlogits_nbias_mask_nlse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_fp16_nlogits_nbias_mask_lse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_fp16_nlogits_nbias_nmask_lse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_fp16_nlogits_nbias_nmask_nlse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_bf16_nlogits_nbias_mask_lse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_bf16_nlogits_nbias_nmask_lse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_skip_nqscale.so",
    "mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so",
]

FMHA_V3_SO = "module_fmha_v3_varlen_fwd.so"


def _clean_jit_artifacts(jit_dir: Path, prefix: str):
    """Remove .so files, build dirs, and lock files matching a name prefix.

    Called before each arch build to ensure AITER sees no cached artifacts
    and triggers a fresh compilation for the new target architecture.
    """
    for so in jit_dir.glob(f"{prefix}*.so"):
        so.unlink(missing_ok=True)
    build_dir = jit_dir / "build"
    if build_dir.exists():
        for d in build_dir.glob(f"{prefix}*"):
            if d.is_dir():
                shutil.rmtree(d)
            else:
                d.unlink(missing_ok=True)
        for lock in build_dir.glob(f"lock_{prefix}*"):
            lock.unlink(missing_ok=True)


def _trigger_mha_variant(variant_index: int):
    """Trigger AITER JIT for a single mha_varlen_fwd variant (runs in subprocess).

    Called via --_trigger-mha <index>. Creates tensors on GPU and calls
    mha_varlen_fwd() which triggers AITER's JIT compilation. The .so file
    is written to disk during compilation, before kernel dispatch. If dispatch
    crashes (cross-compiled arch mismatch), the .so is still valid.
    """
    import torch
    from aiter.ops.mha import mha_varlen_fwd

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype_str, is_causal, return_lse, min_sq, _ = MHA_VARIANTS[variant_index]

    dt = dtype_map[dtype_str]
    seq, d = 256, 128
    q = torch.randn(seq, 1, d, dtype=dt, device="cuda")
    k = torch.randn(seq, 1, d, dtype=dt, device="cuda")
    v = torch.randn(seq, 1, d, dtype=dt, device="cuda")
    cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")

    mha_varlen_fwd(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        min_seqlen_q=min_sq,
        dropout_p=0.0,
        softmax_scale=1.0 / (d**0.5),
        logits_soft_cap=0.0,
        zero_tensors=False,
        is_causal=is_causal,
        window_size_left=-1,
        window_size_right=-1,
        return_softmax_lse=return_lse,
        return_dropout_randval=False,
    )


def _trigger_fmha_v3():
    """Trigger AITER JIT for the fmha_v3 kernel (runs in subprocess).

    Called via --_trigger-fmha-v3. The fmha_v3 kernel uses hand-written
    GPU ASM. Cross-compilation produces a valid .so but dispatch raises
    RuntimeError (caught here) or segfaults (caught by subprocess isolation).
    """
    import torch
    from aiter.ops.mha import fmha_v3_varlen_fwd

    seq, d = 256, 128
    q = torch.randn(seq, 1, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq, 1, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(seq, 1, d, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")

    try:
        fmha_v3_varlen_fwd(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=seq,
            max_seqlen_k=seq,
            min_seqlen_q=seq,
            dropout_p=0.0,
            softmax_scale=1.0 / (d**0.5),
            logits_soft_cap=0.0,
            zero_tensors=False,
            is_causal=True,
            window_size_left=-1,
            window_size_right=-1,
            return_softmax_lse=False,
            return_dropout_randval=False,
            how_v3_bf16_cvt=0,
        )
    except RuntimeError:
        pass  # Expected when cross-compiling for a non-native arch


def _build_kernel_in_subprocess(jit_dir: Path, so_name: str, label: str, trigger_args: list[str]):
    """Build a single kernel by triggering JIT in an isolated subprocess.

    The subprocess may crash (segfault) when dispatching a cross-compiled
    kernel — this is expected. We only care whether the .so file was
    written to disk, which happens during compilation before dispatch.

    Returns True if the .so was produced, False otherwise.
    """
    so_path = jit_dir / so_name

    if so_path.exists():
        print(f"    SKIP  {label}")
        return True

    print(f"    BUILD {label} ...", end="", flush=True)
    start = time.time()

    result = subprocess.run(
        [sys.executable, __file__] + trigger_args,
        env=os.environ.copy(),
        capture_output=True,
        timeout=300,
    )

    elapsed = time.time() - start
    if so_path.exists():
        size_k = so_path.stat().st_size / 1024
        status = "OK" if result.returncode == 0 else "OK (dispatch crashed, .so valid)"
        print(f" {status} ({elapsed:.0f}s, {size_k:.0f}K)")
        return True
    else:
        print(f" FAILED ({elapsed:.0f}s, exit={result.returncode})")
        if result.stderr:
            for line in result.stderr.decode(errors="replace").strip().splitlines()[-3:]:
                print(f"      {line}")
        return False


def build_kernels(archs: list[str], output_dir: Path):
    """Build all kernel variants for each target architecture.

    For each arch:
      1. Set GPU_ARCHS=<arch> so hipcc cross-compiles for that target
      2. Clean any cached .so / build artifacts from the previous arch
      3. Build each kernel in an isolated subprocess (9 mha + 1 fmha_v3)
      4. Copy the resulting .so files into <output_dir>/<arch>/

    The output directories are COPY'd into the Docker image at
    /workspace/aiter-jit-prebuilt/<arch>/. At runtime, the matching
    arch's kernels are copied into AITER's system JIT directory.
    """
    jit_dir = Path(os.environ.get("AITER_JIT_DIR", str(_get_aiter_jit_dir())))
    total_failures = 0

    for arch in archs:
        print(f"\n=== Building for {arch} ===")
        arch_dir = output_dir / arch
        arch_dir.mkdir(parents=True, exist_ok=True)

        os.environ["GPU_ARCHS"] = arch

        _clean_jit_artifacts(jit_dir, "mha_varlen_fwd_")
        _clean_jit_artifacts(jit_dir, "module_fmha_v3_varlen_fwd")

        start = time.time()
        built = 0

        for i, (_, _, _, _, label) in enumerate(MHA_VARIANTS):
            if _build_kernel_in_subprocess(jit_dir, MHA_SO_NAMES[i], label, ["--_trigger-mha", str(i)]):
                built += 1
            else:
                total_failures += 1

        if _build_kernel_in_subprocess(jit_dir, FMHA_V3_SO, "fmha_v3", ["--_trigger-fmha-v3"]):
            built += 1
        else:
            total_failures += 1

        elapsed = time.time() - start

        collected = 0
        for so_name in MHA_SO_NAMES + [FMHA_V3_SO]:
            src = jit_dir / so_name
            if src.exists():
                shutil.copy2(src, arch_dir / so_name)
                collected += 1

        print(f"    {arch}: {collected}/10 kernels in {elapsed:.0f}s → {arch_dir}")

    print("\n=== Summary ===")
    for arch in archs:
        arch_dir = output_dir / arch
        so_files = list(arch_dir.glob("*.so"))
        total_size = sum(f.stat().st_size for f in so_files)
        print(f"  {arch}: {len(so_files)} kernels, {total_size / 1024:.0f}K total")
        for f in sorted(so_files):
            print(f"    {f.name}: {f.stat().st_size / 1024:.0f}K")

    if total_failures > 0:
        print(f"\nERROR: {total_failures} build failure(s), exiting...")
        sys.exit(1)


def list_kernels():
    """Print the full list of kernels this script builds."""
    print("AITER kernels pre-built for vLLM 0.16 profiles:\n")
    print("mha_varlen_fwd variants (9):")
    for name in MHA_SO_NAMES:
        print(f"  {name}")
    print("\nfmha_v3 (1):")
    print(f"  {FMHA_V3_SO}")
    print(f"\nTotal: {len(MHA_SO_NAMES) + 1} kernels per architecture")
    print(f"Default architectures: {', '.join(DEFAULT_ARCHS)}")
    print("\nAlready pre-compiled in base image (52 modules, no action needed):")
    print("  module_rmsnorm.so, module_pa.so, module_moe_ck2stages.so, etc.")


def main():
    parser = argparse.ArgumentParser(description="Pre-build AITER JIT attention kernels for vLLM 0.16 profiles")
    parser.add_argument("--list", action="store_true", help="List target kernel variants without building")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="GPU architectures to cross-compile for (overrides --profile-dir inference)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Assets root directory to scan for profile filenames (e.g. assets/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/aiter-jit-prebuilt"),
        help="Output directory (kernels stored in <dir>/<arch>/)",
    )
    parser.add_argument("--_trigger-mha", type=int, metavar="INDEX", help=argparse.SUPPRESS)
    parser.add_argument("--_trigger-fmha-v3", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._trigger_mha is not None:
        _trigger_mha_variant(args._trigger_mha)
        return

    if args._trigger_fmha_v3:
        _trigger_fmha_v3()
        return

    if args.list:
        list_kernels()
        return

    if args.archs:
        archs = args.archs
    elif args.profile_dir:
        archs = infer_archs_from_profiles(args.profile_dir)
    else:
        archs = DEFAULT_ARCHS

    build_kernels(archs, args.output_dir)


if __name__ == "__main__":
    main()
