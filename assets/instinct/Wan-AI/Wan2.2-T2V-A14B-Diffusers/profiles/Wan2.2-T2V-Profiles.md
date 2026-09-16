<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Wan2.2-T2V-A14B USP Profiles (vLLM-Omni)

Documentation for the USP-based profiles used by the
`Wan-AI/Wan2.2-T2V-A14B-Diffusers` AIM running on the **vLLM-Omni** engine on
AMD Instinct GPUs (**MI300X, MI325X, MI350X, MI355X**).

All profiles for this model follow the **same structure** and only differ in
**scale** — the number of GPUs (`usp` / `vae-patch-parallel-size` / `gpu_count`),
which are always kept equal to each other — and the **GPU model** they target.

For each supported GPU we ship the same 8-profile set (4 GPU counts ×
latency/throughput). Using MI300X as the example:

| Profile | GPUs | `usp` | `vae-patch-parallel-size` | metric |
|---|---|---|---|---|
| `vllm_omni-mi300x-fp16-tp1-latency-usp1.yaml`      | 1 | 1 | 1 | latency |
| `vllm_omni-mi300x-fp16-tp1-throughput-usp1.yaml`   | 1 | 1 | 1 | throughput |
| `vllm_omni-mi300x-fp16-tp2-latency-usp2.yaml`      | 2 | 2 | 2 | latency |
| `vllm_omni-mi300x-fp16-tp2-throughput-usp2.yaml`   | 2 | 2 | 2 | throughput |
| `vllm_omni-mi300x-fp16-tp4-latency-usp4.yaml`      | 4 | 4 | 4 | latency |
| `vllm_omni-mi300x-fp16-tp4-throughput-usp4.yaml`   | 4 | 4 | 4 | throughput |
| `vllm_omni-mi300x-fp16-tp8-latency-usp8.yaml`      | 8 | 8 | 8 | latency |
| `vllm_omni-mi300x-fp16-tp8-throughput-usp8.yaml`   | 8 | 8 | 8 | throughput |

The same eight files exist for the other GPUs with the `mi300x` slug replaced
by `mi325x`, `mi350x`, or `mi355x` (and `metadata.accelerator_model` set to
`MI325X` / `MI350X` / `MI355X` accordingly) — 32 profile files in total. The
engine args and env vars are currently identical across GPUs (all
`type: unoptimized`); per-GPU tuning can be applied later.

The `tp1` (usp1) profiles are marked `primary: true` — the default selection
for each GPU model.

> **Model scope:** Wan2.2-T2V-A14B is a **text-to-video (T2V)** model. The same
> profiling approach (USP-based parallelism, VAE parallelism, the vLLM-Omni
> diffusion serving path) is intended to be reused for the **image-to-video
> (I2V)** variants — the knobs and structure will look essentially the same,
> only the model and input modality change.

---

## What is USP?

**USP = Unified Sequence Parallelism.**

For diffusion / video-generation transformers, the heavy cost is the very long
token **sequence** (all the latent patches across every video frame). Instead of
splitting the model *weights* across GPUs (as tensor parallelism does), USP
splits the **sequence dimension** — each GPU processes a slice of the latent
sequence and the ranks exchange information (via ring/Ulysses-style attention)
so the result is mathematically equivalent to running on a single device.

`usp: N` means the sequence is sharded across **N GPUs**. This is the primary
scaling mechanism for this model because diffusion inference is dominated by the
denoising loop over that long sequence, and USP parallelizes exactly that work.

Alongside USP, the profiles also parallelize the VAE decode stage:

- `vae-patch-parallel-size: N` — splits the VAE decode across the same N GPUs so
  the final decode step doesn't become a single-GPU bottleneck.

### The scaling invariant

Across **every** profile in this family the three scale values are kept equal:

```
usp == vae-patch-parallel-size == gpu_count
```

So `usp1` runs on 1 GPU, `usp2` on 2, `usp4` on 4, `usp8` on 8 — and in each
case the VAE decode is split the same number of ways. To add a new scale you
simply pick a GPU count and set all three to that value; nothing else in the
profile needs to change.

---

## We are NOT using tensor parallelism (TP)

Even though the filenames contain `tpN`, these profiles **do not set a
tensor-parallel size** — there is no `tensor-parallel-size` in `engine_args`.
The `tpN` in the filename simply reflects the **number of GPUs used**, and the
actual parallelism is delivered by **USP + VAE patch parallelism**, not TP. The
`uspN` suffix names the real strategy.

Why USP instead of TP here:

- The bottleneck for video diffusion is the long **sequence**, not the model
  weights — so sharding the sequence (USP) scales better than sharding weights
  (TP).
- USP keeps each GPU running the full model on a slice of the sequence, which
  maps cleanly onto the diffusion denoising loop and the VAE decode.

So: **GPU count = N, parallelism strategy = USP(N) + VAE patch parallel(N), TP =
not used.**

---

## Main differences vs a "formal" AIM (non-vLLM-Omni)

A standard LLM AIM (regular vLLM, text generation) differs from these
vLLM-Omni diffusion profiles in several important ways:

| Aspect | Formal AIM (regular vLLM) | This AIM (vLLM-Omni diffusion) |
|---|---|---|
| Engine | `vllm` (LLM serving) | `vllm_omni` (diffusion / video) |
| Workload | Token-by-token text generation | Denoising loop that produces video frames |
| Parallelism | `tensor-parallel-size` (TP), sometimes PP | **USP** (sequence parallel) + **VAE patch parallel** |
| Extra stages | — | Dedicated **VAE decode** stage (`vae-*` args) |
| Output | Text tokens | Video (served via the `/v1/videos` endpoint by the diffusion harness) |
| Benchmark path | Standard token throughput/latency | vLLM-Omni diffusion benchmark script |

In short: the formal AIM parallelizes **model weights** for **text**; this AIM
parallelizes the **sequence and VAE** for **video**, and it carries
diffusion-specific engine args (`usp`, `vae-patch-parallel-size`,
`vae-use-tiling`) that simply don't exist in a text LLM profile.

---

## Latency vs Throughput

Within a given scale, the latency and throughput profiles differ only in the
fields below. **These differences are the same at every USP scale (usp1 → usp8):**

| Field | `...-throughput-uspN` | `...-latency-uspN` |
|---|---|---|
| `metadata.metric` | `throughput` | `latency` |
| `engine_args.vae-use-tiling` | _omitted_ | `null` |
| `env_vars.VLLM_ROCM_USE_AITER` | `'1'` (AITER on) | `'0'` (AITER off) |

Everything else is shared across the whole family (only `usp` /
`vae-patch-parallel-size` change with scale):

```yaml
engine_args:
  gpu-memory-utilization: 0.95
  no-async-scheduling: null
  usp: N                      # == gpu_count
  vae-patch-parallel-size: N  # == gpu_count
  disable-uvicorn-access-log: null
  distributed_executor_backend: mp
  no-enable-log-requests: null
env_vars:
  NCCL_MIN_NCHANNELS: '112'
  TORCH_BLAS_PREFER_HIPBLASLT: '1'
  VLLM_DO_NOT_TRACK: '1'
```

> **Note (work in progress):** several of these values are still being tuned.
> All profiles are currently `type: unoptimized`, and the throughput profiles
> carry a `#remove later` comment on `manual_selection_only`; these are temporary
> and should move to `type: preview` (and drop the comment) before release. Keys
> set to `null` are rendered as valueless engine flags (e.g. `--no-async-scheduling`).

### Field notes

- **`gpu-memory-utilization: 0.95`** — reserve 95% of VRAM for the engine.
- **`usp: N`** — N-way Unified Sequence Parallelism (see above).
- **`vae-patch-parallel-size: N`** — N-way VAE decode parallelism.
- **`vae-use-tiling`** — controls tiled VAE decode. Present as `null` in the
  **latency** profiles and omitted entirely from the **throughput** profiles.
- **`no-async-scheduling: null`** — passed as a valueless flag to keep the
  scheduler synchronous.
- **`disable-uvicorn-access-log: null`** — silence the Uvicorn per-request
  access log.
- **`distributed_executor_backend: mp`** — use the multiprocessing executor for
  the distributed (multi-GPU) workers.
- **`no-enable-log-requests: null`** — disable per-request logging in the engine.
- **`VLLM_ROCM_USE_AITER`** — enable AMD AITER kernels. On (`'1'`) for the
  throughput profiles, off (`'0'`) for the latency profiles.
- **`NCCL_MIN_NCHANNELS: '112'`** — tune NCCL channel count for the multi-GPU
  collective traffic that USP/VAE parallelism generates.
- **`TORCH_BLAS_PREFER_HIPBLASLT: '1'`** — prefer hipBLASLt for GEMMs on ROCm.
- **`metadata.type`** — currently `unoptimized` for all profiles (WIP; should
  become `preview` before release).

---

## How this integrates with CI

These profiles are not wired up by hand — the build system discovers and
validates them automatically:

1. **Discovery.** The tooling globs `profiles/*.yaml` for the model
   (`ProfileManager.get_yamls`). Only `*.yaml` files are treated as profiles, so
   this `.md` doc is ignored. Adding/removing a profile is just adding/removing a
   YAML file here — no registry to update.

2. **Name ↔ metadata validation.** A CI check
   (`_check_profile_metadata` in `src/aim_utils/profile_utils.py`) parses each
   filename stem as `{engine}-{accelerator}-{precision}-tp{count}-{metric}` plus
   an optional `-{variant}` suffix, and asserts it matches the file's `metadata`
   block. Every part is validated: `tp{count}` must equal `gpu_count`, `{metric}`
   must equal `metadata.metric`, and the `-{variant}` suffix must equal
   `metadata.variant`. That is why each profile here declares `variant: uspN`
   in its metadata to match its `...-uspN.yaml` filename (and why the keys are
   kept alphabetically sorted with single-quoted env values — the
   `sort-profile-keys` pre-commit hook enforces that).

3. **What triggers a build.** The PR orchestrator
   (`.github/workflows/aim-pr-pipeline-orchestrator.yaml`) runs
   `inspect-changes` to decide what changed:
   - Editing `config.yaml` / `profiles/*.yaml` / `metadata.yaml` →
     `RUN_BUILD_IMAGE` (build the **model** image).
   - Editing `image/` (the specialized base) or a shared base →
     `RUN_BUILD_BASE_IMAGE` (build the **base** image).
   - If a change would set **both**, the base build wins and `RUN_BUILD_IMAGE` is
     dropped; the model image is (re)built in a follow-up run once the new base is
     published. So a profile-only PR builds the runnable model image, while a PR
     that also touches `image/` only rebuilds the base.

4. **Base vs model image.** `config.yaml` marks this model `specialized: true`,
   so the build chain inserts `image/Dockerfile` as a layer that drops the
   harness into `/workspace/model/src/`. The base image pipeline publishes
   `aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers-base`; the model-processing
   pipeline then layers these profiles + `AIM_ID` on top to produce the runnable
   `aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers` image.

5. **Harness auto-discovery & validation.** At runtime
   `aim_runtime.harness.discovery` finds `VllmOmniDiffusionHarness` from the
   copied sources. CI's model-service validation exercises it: a `health` check
   (`GET /v1/models`) and a `smoke_video` check (a tiny `POST /v1/videos` job),
   with an optional offline `benchmark` recipe.

---

## Running it (minimal example)

> Replace `<TAG>` with the published tag. The runnable model image is
> `docker.io/silogenai/aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers:<TAG>`
> (note: **no** `-base` suffix — the `-base` image is the engine layer, not the
> model AIM).

### 1. Serve on a single GPU (auto-selects the `usp1` primary profile)

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  docker.io/silogenai/aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers:<TAG>
```

### 2. Multi-GPU with USP (e.g. 4 GPUs → the `usp4` profile)

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_GPU_COUNT=4 \
  -e AIM_METRIC=latency \
  --device=/dev/kfd --device=/dev/dri \
  --shm-size=32g \
  -p 8000:8000 \
  docker.io/silogenai/aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers:<TAG>
```

`AIM_GPU_COUNT` selects the matching `uspN` profile; `AIM_METRIC` picks
latency vs throughput. `--shm-size` (or `--ipc=host`) is required for multi-GPU
so the USP/VAE workers can use shared memory. To pin a profile explicitly,
set `-e AIM_PROFILE_ID=vllm_omni-mi300x-fp16-tp4-latency-usp4`.

Check what would be selected without serving:

```bash
docker run --rm \
  -e AIM_GPU_COUNT=4 -e AIM_METRIC=latency -e AIM_ACCELERATOR_MODEL=MI300X \
  docker.io/silogenai/aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers:<TAG> \
  dry-run
```

### 3. Generate a video (`/v1/videos`)

The server exposes the vLLM-Omni diffusion API. Submit a job, poll for
completion, then download the clip:

```bash
# Submit
JOB=$(curl -s http://localhost:8000/v1/videos \
  -F prompt="a red panda surfing a wave, cinematic" \
  -F size=832x480 -F seconds=5 -F fps=16 -F num_inference_steps=18 \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

# Poll until status == completed
curl -s http://localhost:8000/v1/videos/$JOB

# Download the result
curl -s http://localhost:8000/v1/videos/$JOB/content -o out.mp4
```

### 4. Run the benchmark harness (optional)

The `VllmOmniDiffusionHarness` wraps the upstream
`diffusion_benchmark_serving.py`. Pick a recipe from `image/src/recipes.py`
(`smoke`, `dataset_a_480p`, `dataset_b_720p`, `dataset_c_mix`):

```bash
# bench.yaml
recipe: dataset_a_480p
```

```bash
docker run --rm \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -v "$PWD/bench.yaml:/tmp/bench.yaml" \
  docker.io/silogenai/aim-instinct-wan-ai-wan2-2-t2v-a14b-diffusers:<TAG> \
  benchmark --config /tmp/bench.yaml
```

This reports `throughput_qps` and latency percentiles (`latency_mean`,
`latency_p99`, …) for the chosen recipe.

---

## Standardization / future work

Wan2.2 is the **first** vLLM-Omni (diffusion/video) model in this repo, so parts
of it are currently one-off patterns that should be generalized before more omni
models (I2V, other T2V variants) are added. Two areas in particular:

### 1. Move the harness to a shared vLLM-Omni base

Today the harness ships per-model at
`assets/instinct/Wan-AI/Wan2.2-T2V-A14B-Diffusers/image/src/harness.py`, even
though `VllmOmniDiffusionHarness` is deliberately **engine-generic** — it has no
Wan-specific paths (only the *default* recipes in `recipes.py` are model-flavored).
Its own docstring notes it "can later be lifted into a shared
`assets/instinct/base/vllm-omni/image/src/` without changes."

**Standardized target:** host the harness (and the generic recipe scaffolding)
once in a shared `base/vllm-omni/` image layer, and have each omni model image
build `FROM` that base. New omni models then only supply model-specific bits
(profiles, `AIM_ID`, any model-specific recipes) instead of copying the harness.
This is the structural refactor the reviewers have flagged; it keeps a single
source of truth for the `/v1/videos` serving + benchmark logic.

### 2. Standardize the omni profile conventions

The current profiles carry two conventions that are worth ratifying (and
documenting) so every omni model follows them:

- **USP vs TP naming.** Filenames use `tpN` for historical/tooling reasons, but
  these profiles use **no tensor parallelism** — `N` is the GPU count and the
  real strategy is USP (`usp: N`) + VAE patch parallelism. The `-uspN` variant
  suffix (declared as `metadata.variant`) disambiguates this. A standard should
  state clearly whether `tpN` stays as "GPU count" or whether omni profiles get
  a dedicated segment, and whether `variant` is the right home for `uspN`.
- **The knob set.** Every profile shares the same shape and differs only by
  scale (`usp == vae-patch-parallel-size == gpu_count`) and the latency/throughput
  knobs (`vae-use-tiling`, `VLLM_ROCM_USE_AITER`). Codifying this shape — ideally
  as a documented template or generator — means future omni profiles are
  consistent by construction rather than hand-copied.

Until these are standardized, treat this model's layout as the reference example
and keep the harness engine-generic (no Wan-specific logic) so the eventual lift
into a shared base stays a move, not a rewrite.
