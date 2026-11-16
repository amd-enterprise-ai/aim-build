<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

- The FP8 profiles for MI350P, MI350X and MI355X serve the BF16 checkpoint (`Qwen/Qwen3.8-27B`) with FP8 quantization applied at runtime. The FP8 profiles for MI300X and MI325X serve the pre-quantized `Qwen/Qwen3.8-27B-FP8` checkpoint.
- MXFP4 profiles are not included in this release because of a bug in the vLLM 0.33 base image that prevents them from starting.
