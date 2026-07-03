<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# OpenFold3 Specialized Base Image (model-level)

Build context for the OpenFold3 (OF3) specialized base image — the Layer 1
container for the OF3 AIM, served via BentoML on ROCm.

## Layout

- `Dockerfile`          — builds on top of the upstream vendor image
                           (rocm/pytorch, pinned in the `FROM`),
                           installs OF3 Python deps, clones OpenFold3 at the
                           pinned ref, installs the AIM wrapper package, and
                           COPYs `examples/benchmark_sequences/` into the
                           image at `/workspace/model/benchmarks/` for the
                           harness to read at benchmark time. Also COPYs the
                           engine config and OF3 profile into the aim-runtime
                           well-known paths so `aim-runtime serve` can resolve
                           them.
- `pyproject.toml`      — wrapper package metadata. Pins `bentoml==1.4.35`
                           and `pydantic>=2.0`. Uses a src layout so
                           `pip install -e .` discovers the modules under
                           `src/`.
- `src/service.py`      — BentoML `OpenFold3Prediction` service exposing a
                           single `/predict` endpoint.
- `src/runner.py`       — bridge to OF3's `InferenceExperimentRunner`. Owns
                           the model checkpoint cache (`~/.openfold3`), the
                           ROCm-specific runner args (Triton triangle kernels
                           on, DeepSpeed evo-attention and cuEquivariance
                           kernels off), and output-directory parsing.
- `src/harness.py`      — `OpenFold3Harness(ModelHarness)`. Discovered at
                           runtime by `aim_runtime.harness.discovery` from
                           `/workspace/model/src/harness.py`. Implements
                           `validate`, `benchmark`, `evaluate`, `list_checks`
                           for the `aim-runtime` CLI.
- `examples/benchmark_sequences/<pdb>/<pdb>.json` — 21 per-PDB request
                           payloads (NVIDIA-reference benchmark set),
                           COPY'd into the image as
                           `/workspace/model/benchmarks/` and read by the
                           harness's `benchmark_21_pdb` check.
- `patches/`            — source patches `git apply`'d to `/opt/openfold3`
                           at build time. Currently:
                           `of3_torch_compile_champion.patch` — adds the
                           `OPENFOLD3_COMPILE_*` `torch.compile` dispatcher
                           to `model.py` and a stride-only `.expand()`
                           guard in `atomize_utils.py` (required for the
                           small-recipe compile path).

## Build args

The Layer 1 build runs a plain `docker build` with no build-args (the
upstream ref is pinned in the `FROM`). The only overridable args:

- `OPENFOLD3_REPO`      — defaults to `https://github.com/aqlaboratory/openfold-3.git`.
- `OPENFOLD3_REF`       — defaults to `0.4.1`.

## How CI resolves this directory

`aim_utils.specialized_utils.enumerate_specialized_base_targets` discovers
this `image/` directory as a **model-level** specialized base target
(`assets/<acc>/<org>/<model>/image/`), built as the model-dedicated
Layer 2 base `aim-instinct-openfold-openfold3-base`. The pipeline is
implemented in
[.github/workflows/base-image-pipeline.yaml](../../../../../.github/workflows/base-image-pipeline.yaml).

## Running the image

The canonical entrypoint is `aim-runtime serve`, which reads the engine
config + OF3 profile baked into the image, generates the BentoML launch
command, and execs it.

### Required environment

| Variable | Value | Why |
|---|---|---|
| `AIM_ID` | `openfold/openfold3` | Selects this model's profile dir. |
| `AIM_PROFILE_ID` | `bentoml-mi300x-fp16-tp1-latency` | The OF3 profile is `manual_selection_only: true` while OF3 is in preview, so auto-selection won't pick it; specify explicitly. See "TorchInductor" below for what this profile compiles. |
| `AIM_ACCELERATOR_COUNT` | `1` | Match the profile's `gpu_count: 1`. The AIM accelerator detector reads sysfs and may report all physical GPUs on the host; this override constrains profile selection. |
| `AIM_PORT` | `8000` (default) | Port BentoML binds. |

### TorchInductor

The shipped profile is the **long-sequence champion**: it compiles
`PairFormerBlock` (`max-autotune-no-cudagraphs`), `diffusion_transformer`,
and `diffusion_conditioning` (default mode) via `torch.compile`
(Inductor), with global Triton GEMM max-autotune
(`OPENFOLD3_INDUCTOR_GEMM_BACKENDS=TRITON`). `PairFormerBlock` is the
dominant compute bucket on medium+ sequences (the pair stack overtakes
the diffusion rollout between the medium and med_high tiers), so wrapping
it is the main delta over the older diff-only recipe. Expected warm
e2e-forward gain vs eager is ~−13 % to −18 % on the medium / med_high /
high tiers. `CD_TUNE` is intentionally
omitted — it erodes the gain at the longest sequences.

The recipe is a `torch.compile` dispatcher (`OPENFOLD3_COMPILE_*` env
vars read at first `forward()`) plus a one-line raise of
`torch._dynamo.config.recompile_limit` from 8 to 64 so static-shape
compiles for many distinct sequence lengths all fit in dynamo's
per-frame cache. The source patches live in
`patches/of3_torch_compile_champion.patch` and are applied at build
time — including the trunk-unblock patch to
`triangular_multiplicative_update.py` (makes the hand-written triangle
Triton kernels opaque to Inductor so the `PairFormerBlock` wrap does not
LDS-OOM on MI300). Other recipes (small champion, diff-only) remain
*available* via different `OPENFOLD3_COMPILE_*` env vars; a future
multi-profile split will ship per-tier recipes.

**Cold start per distinct sequence length** spends tens of seconds (up
to ~80 s on large shapes) in `torch.compile` + Triton/GEMM autotune on
the first `/predict` for that shape; subsequent same-shape requests run
at the compiled latency. Distinct shapes each pay their own one-time
tax, up to the `recompile_limit=64` cap. The harness benchmark's
mean-of-3 absorbs one cold compile per PDB, so its reported ratio runs
higher than the steady-state (`min_s`) latency. All harness `/predict`
checks use a 600 s client timeout, matching the service-side
`traffic.timeout`, so one cold compile plus prediction fits within the
window.

**TunableOp** (`PYTORCH_TUNABLEOP_*`, runtime rocBLAS/hipBLASLt GEMM
tuning) is a further ~−6 % long-sequence lever but requires a per-shape
tuned CSV on a writable, persistent mount; it is deferred until the
serving image ships a tuning/cache path.

**Persisting the compile cache.** To skip the warm-up across
container restarts, point `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` at a writable host path mounted into the
container — for example NVMe on a docker host, or `$HOME` on Core42 /
enroot:

```bash
-v $HOME/of3_compile_cache:/cache \
-e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
-e TRITON_CACHE_DIR=/cache/triton
```

These vars are intentionally **not** baked into the profile YAML
because the right path depends on the deployment environment.

### Minimum docker invocation

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $HOME/openfold_cache:/root/.openfold3 \
  -p 8000:8000 \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  aim-bentoml-openfold-openfold3-upstream:<version> \
  aim-runtime serve
```

### Driving the harness from a second shell

Once the container is running, exec in and call the harness CLI:

```bash
docker exec <container> aim-runtime list-checks
docker exec <container> aim-runtime validate --scope runtime --service-url http://localhost:8000
docker exec <container> aim-runtime validate --scope offline --service-url http://localhost:8000
docker exec <container> aim-runtime evaluate                 --service-url http://localhost:8000
docker exec <container> aim-runtime benchmark                --service-url http://localhost:8000 --output-dir /artifacts
```

## Runtime dependencies (not provisioned at build time)

- **Model weights** (~2.3 GB, `of3-p2-155k.pt`) — downloaded from
  HuggingFace on first `/predict` call by
  `openfold3.entry_points.parameters.download_model_parameters`, cached at
  `$OPENFOLD3_CACHE` or `~/.openfold3`. Mount a host directory at
  `~/.openfold3` (or override `$OPENFOLD3_CACHE`) to keep the checkpoint
  across container restarts.
- **ColabFold MSA server** — used when `use_msa_server=true` (default in
  `service.py`). The harness smoke / benchmark payloads pin
  `use_msa_server: false` to avoid the dependency on a public service
  during automated checks.
- **PDB structure templates** — fetched on-demand when `use_templates=true`.
  The harness payloads pin this `false` for the same reason.

### Inline precomputed MSAs

A chain may carry its precomputed MSA inline in the `/predict` body instead of
relying on the MSA server — no mount needed, the a3m text travels in the
request. Set `main_msa` and/or `paired_msa` on the chain to a3m text (a single
string, or a list of strings for multiple alignments), and pass
`use_msa_server: false`. The wrapper writes the content to `.a3m` files and
hands their paths to OF3.

Server-side `main_msa_file_paths` / `paired_msa_file_paths` still work and
**win** over inline content if both are supplied on the same chain.

```json
{
  "data": {
    "queries": {
      "my_query": {
        "chains": [
          {
            "molecule_type": "protein",
            "chain_ids": ["A"],
            "sequence": "MSDKIIHLTDDSFDTDVLKAD...",
            "main_msa": ">query\nMSDKIIHLTDDSFDTDVLKAD...\n"
          }
        ]
      }
    },
    "use_msa_server": false
  }
}
```

### Per-atom confidences and timing

Set `include_atom_confidences: true` in the `/predict` body (default `false`)
to add an `atom_confidence` map to the response, keyed by `sample_id`. Each
entry is OF3's `*_confidences.json` verbatim:

- `plddt` — per-**atom** list (one score per atom).
- `pae` / `pde` — per-**token** T×T matrices (predicted aligned / distance
  error). These scale as T² and can be large, which is why this is opt-in.

The response always carries a top-level `timing` object **keyed by `sample_id`**:
each sample's value is its seed's `timing.json` (`{"runtime_s": <float>}`). It is
always present — an empty `{}` when OF3 didn't emit timing.

Default responses are backward compatible: with the flag off, no
`atom_confidence` key is added and only the new always-present `timing` key
appears alongside the existing `structures` / `confidence` / `error` keys.

## CI invocation (when wired)

The same `aim-runtime` subcommands above are what the orchestrator will
call once the harness invocation step lands in the base/model image
pipeline. CI integration is a follow-up. Expected per-job shape:

```bash
docker run -d --name $JOB \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  ... aim-bentoml-openfold-openfold3-upstream:$VERSION \
  aim-runtime serve
docker exec $JOB aim-runtime validate --scope runtime --service-url http://localhost:8000
docker exec $JOB aim-runtime benchmark --service-url http://localhost:8000 --output-dir /artifacts
docker cp $JOB:/artifacts ./artifacts/
docker stop $JOB
```
