<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Pre-built AITER JIT Kernels

Pre-compiled AITER attention kernels that eliminate ~280s of JIT
compilation on first inference request.

## How it works

1. **CI builds kernels** on a GPU pod using `prebuild_aiter_kernels.py`,
   cross-compiling for each target architecture (gfx942, gfx950).
2. **Docker build** copies per-arch kernel directories into the image
   at `/workspace/aiter-jit-prebuilt/<arch>/`.
3. **At runtime**, `AIMRuntime.serve()` detects the GPU model from the
   selected profile, resolves its GFX architecture, and copies the
   matching pre-built `.so` files into AITER's system JIT directory
   before command generation runs. AITER finds both pre-built and system
   modules in one place — no `AITER_JIT_DIR` override or symlinks needed.

The `.so` files are NOT stored in git — they are built fresh in CI.

## Directory structure (in image)

```
/workspace/aiter-jit-prebuilt/
├── gfx942/
│   ├── mha_varlen_fwd_fp16_*.so          (pre-built, ~1.2M each)
│   ├── mha_varlen_fwd_bf16_*.so          (pre-built, ~1.2M each)
│   └── module_fmha_v3_varlen_fwd.so      (pre-built, ~500K)
└── gfx950/
    └── (same structure, cross-compiled for gfx950)
```

## Kernel inventory (10 per arch)

| Kernel | Triggered by |
|--------|-------------|
| `mha_varlen_fwd_fp16_*_mask_nlse_*` | fp16 causal attention |
| `mha_varlen_fwd_fp16_*_mask_lse_*` | fp16 causal + LSE return |
| `mha_varlen_fwd_fp16_*_nmask_lse_*` | fp16 non-causal + LSE |
| `mha_varlen_fwd_fp16_*_nmask_nlse_*` | fp16 non-causal attention |
| `mha_varlen_fwd_bf16_*_mask_nlse_*` | bf16/fp8 causal attention |
| `mha_varlen_fwd_bf16_*_mask_lse_*` | bf16 causal + LSE return |
| `mha_varlen_fwd_bf16_*_nmask_lse_*` | bf16 non-causal + LSE |
| `mha_varlen_fwd_bf16_*_nmask_nlse_…skip_*` | bf16 non-causal attention |
| `mha_varlen_fwd_bf16_*_nmask_nlse_…nskip_*` | gemma-3 (min_seqlen_q=0) |
| `module_fmha_v3_varlen_fwd` | Flash Attention v3 ASM (bf16, head_dim=128) |

## Building manually

On any machine with GPU + ROCm + AITER:

```bash
python3 docker/prebuild_aiter_kernels.py --archs gfx942 gfx950
```

## Validation

Validated on `aim-base:0.11-rc21` (vLLM 0.16.0) with 13 model profiles.
100% JIT elimination, zero accuracy impact, up to 47% faster initial inference.
