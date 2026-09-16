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
                           the model checkpoint cache (`$AIM_CACHE_PATH`), the
                           ROCm-specific runner args (Triton triangle kernels
                           on, DeepSpeed evo-attention and cuEquivariance
                           kernels off), and output-directory parsing.
- `src/gpu_affinity.py` — maps each BentoML worker onto one accelerator and
                           restricts the worker process to it. See
                           "Concurrency" below.
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
                           at build time, in the order the Dockerfile applies
                           them. Each patch's purpose and the condition for
                           dropping it are documented at its `git apply` site
                           in the Dockerfile.

## Build args

The Layer 1 build runs a plain `docker build` with no build-args (the
upstream ref is pinned in the `FROM`). The only overridable args:

- `OPENFOLD3_REPO`      — defaults to `https://github.com/aqlaboratory/openfold-3.git`.
- `OPENFOLD3_SHA`       — defaults to `c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`,
                          the commit tag `v0.5.0` resolves to. Selects the source;
                          pinned by commit because tags are mutable.
- `OPENFOLD3_VERSION`   — defaults to `0.5.0`, the release `OPENFOLD3_SHA` points to.
                          Only names the install: a commit-pinned checkout has no
                          tag for `setuptools-scm` to read, and OF3 gates
                          checkpoint compatibility on the reported version.
                          Update together with `OPENFOLD3_SHA`.

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
| `AIM_PROFILE_ID` | `bentoml-mi300x-fp16-tp<N>-latency`, N ∈ {1, 2, 4, 8} | Pins one OF3 profile explicitly, so selection cannot depend on detected hardware. All four share the same recipe and differ only in accelerator count. See "TorchInductor" below for what they compile. |
| `AIM_ACCELERATOR_COUNT` | `1`, `2`, `4` or `8` | Must match the chosen profile's `accelerator_count`, and also fixes the worker count. The AIM accelerator detector reads sysfs and may report all physical GPUs on the host; this override constrains profile selection. |
| `AIM_PORT` | `8000` (default) | Port BentoML binds. |

### Concurrency

OpenFold3 has no tensor parallelism, so the `tpN` profiles are **data
parallel**: `AIM_ACCELERATOR_COUNT=N` runs N BentoML
[workers](https://docs.bentoml.com/en/latest/build-with-bentoml/parallelize-requests.html)
— separate processes, each with one accelerator pinned — and the deployment
serves **up to** N predictions concurrently. A worker handles one request at a
time, because `/predict` is a sync endpoint and
[`threads`](https://docs.bentoml.com/en/latest/reference/bentoml/configurations.html)
defaults to 1, so the worker count is the ceiling on concurrency. The `tpN`
filename stem is the enforced naming convention for `accelerator_count`, not a
claim of tensor parallelism.

N concurrent requests do **not** reliably land on N different workers. All
workers share one listening socket and the kernel picks whichever wins
`accept()`, with no knowledge of who is busy; a worker mid-prediction still
accepts, because the one-at-a-time limit is applied inside the request handler
rather than at the socket. So a request can queue behind a running prediction
while another accelerator sits idle. Distributing by load would need a
least-busy router in front of the workers, which BentoML does not provide — its
worker model is process-level parallelism, with no dispatch layer between the
socket and the workers. The queue that forms behind a busy worker is bounded by
`traffic.max_concurrency` (see below), so waiting no longer consumes a request's
whole time budget.

Pinning happens in `OpenFold3Prediction.__init__` via `src/gpu_affinity.py`,
before any torch import, by narrowing `HIP_VISIBLE_DEVICES` /
`CUDA_VISIBLE_DEVICES` to this worker's device. This is BentoML's own
[multi-GPU pattern](https://docs.bentoml.com/en/latest/build-with-bentoml/gpu-inference.html)
— `worker_index - 1` selects the device, worker 1 taking GPU 0 — applied to the
environment rather than to a `torch.device`, so it also constrains anything the
model does internally. Without it a single prediction could spread across every
visible accelerator. BentoML's own GPU allocator is switched off in the profiles
(`BENTOML_DISABLE_GPU_ALLOCATION=1`) — it probes via `pynvml`, which finds
nothing on ROCm, and would otherwise hand every worker the same device list.

Two costs scale with N, since each worker holds its own copy of the model:

- **Host memory** — roughly N× the single-accelerator footprint. The generated
  Kubernetes pod spec already scales memory and CPU with `accelerator_count`.
- **Cold start** — each worker compiles independently on its first request per
  shape. The on-disk Inductor and Triton caches are already shared, since all
  workers run in one container against the same filesystem, so a shape another
  worker has compiled is much cheaper the second time. What is not shared is
  Dynamo's in-process tracing. To keep the caches across restarts as well, see
  "Persisting the compile cache" below.

Model weights are downloaded once: `_ensure_model_parameters` takes a lock file
in the checkpoint cache, so a cold multi-worker start does not race on the same
file.

### Request budget and load shedding

`traffic.timeout` cannot stop a running prediction. `/predict` is a sync
endpoint, so BentoML runs it through
`anyio.to_thread.run_sync(..., abandon_on_cancel=False)`, and the timeout
middleware cancels only the *client* side: the cancelled task does not unwind
until the worker thread returns, and a Python thread cannot be killed. Left to
fire, the cap answers 504 while the worker keeps grinding on a result nobody
will receive — the service then accepts requests it cannot start, which looks
like a hang. This bit hardest with `use_msa_server=true`, because OF3's ColabFold
client retries forever: `submit`, `status` and `download` each loop on
`while True`, and a `RATELIMIT` response is resubmitted indefinitely.

Three mechanisms keep that bounded, so the cap should never fire:

1. **Every request carries a deadline** set inside `traffic.timeout` and
   measured from *arrival* (stamped by `ArrivalStampMiddleware`, since queue
   time is already spent against the cap). `src/request_budget.py` holds the
   arithmetic.
2. **The MSA stage is capped separately**, at whichever comes first: its own
   timeout, or the point where continuing would eat the reserve held back for
   inference. `patches/of3_msa_server_deadline.patch` gives the ColabFold retry
   loops a wall clock to respect; exceeding it raises `MsaServerTimeout` rather
   than retrying. Exhausting the five-attempt retry cap against an unreachable
   server raises the same type, so both ways of giving up on the MSA server look
   alike to a caller instead of one arriving as a generic failure. The service
   fails to start if that patch is missing, rather than serving with the bound
   silently gone.

   The patch only checks the deadline between retries, so it bounds the *wait*
   before a `requests.get()` call, not a call already in flight — the MSA
   result tar download and the streamed template fetch can each run past the
   deadline, held only by their own ~6.02s per-socket `requests` timeout rather
   than a total-transfer cap. A self-hosted MMseqs2 server (see below) removes
   this exposure along with the throttling it is meant to survive.
3. **Excess work is turned away**, not queued: `traffic.max_concurrency`
   defaults to three in flight per worker (one running, two queued), past which
   a client gets an immediate `429 {"error": "Too many requests"}`. The
   "one running" half of that holds because `threads` is left at its default of
   1; raising `threads` would put several predictions on one accelerator and
   invalidate the inference reserve below.

| Env var | Default | Effect |
| --- | --- | --- |
| `OPENFOLD3_REQUEST_TIMEOUT_SECONDS` | `600` | Whole-request cap (`traffic.timeout`). |
| `OPENFOLD3_MSA_TIMEOUT_SECONDS` | `300` | Cap on MSA-server work. Also read by the patched client as its own fallback. |
| `OPENFOLD3_INFERENCE_RESERVE_SECONDS` | `120` | Budget withheld from the MSA stage for GPU work. |
| `OPENFOLD3_MAX_CONCURRENCY` | `3 × accelerator_count` | Service-wide in-flight cap; BentoML divides it by the worker count. |
| `OPENFOLD3_MSA_USER_AGENT` | `aim-openfold3 (+…)` | Sent to the MSA server. OF3 defaults to a bare `openfold`, which its own client warns about. |

**Status codes.** Prediction failures keep answering HTTP 200 with
`{"error": true, "message": …}`, as before. The *transient* conditions answer
**503** with the same body plus `Retry-After`: the MSA server not finishing
within its budget or being unreachable across its retries, and a request that
queued so long it can no longer finish. Overload answers **429**. Those are
worth retrying; a 200-with-error is not.

**For sustained throughput, stop using the public MSA server.** The 503 above
means `api.colabfold.com` is rate-limiting the deployment — every worker calls
it from one pod IP, so fanning out makes throttling *more* likely, and the
accelerators sit idle while workers block on HTTP. `mmseqs2` is already
installed in this image; point `msa_computation_settings.server_url` at a
self-hosted ColabFold/MMseqs2 server, or supply
[inline MSAs](#inline-precomputed-msas) with `use_msa_server: false`, and the
throttling disappears. The bounds above make the public-server path fail
honestly; they do not make it fast.

### TorchInductor

The shipped profiles all use the same **long-sequence champion** recipe: it compiles
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
checks use a 600 s client timeout, matching the default service-side
`traffic.timeout`, so one cold compile plus prediction fits within the
window. Raising `OPENFOLD3_REQUEST_TIMEOUT_SECONDS` past that needs a
matching `predict_timeout_seconds` in the harness config, or the client
gives up first.

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
  -v $HOME/openfold_cache:/workspace/model-cache \
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

- **Model weights** (~2.2 GB, `of3-ob-2025-06-30-174k.pt`, registry name
  `openbind-2025-06-30-174k`) — downloaded during service startup from
  `s3://openfold3-data` by
  `openfold3.entry_points.parameters.download_model_parameters`, cached at
  `$AIM_CACHE_PATH` (default `/workspace/model-cache`, the standard AIM cache
  path that deployments already mount a volume at). Mount a host directory or
  volume there to keep the checkpoint across restarts, or point
  `$AIM_CACHE_PATH` elsewhere as with any AIM.
- **ColabFold MSA server** — used when `use_msa_server=true` (default in
  `service.py`). The harness smoke / benchmark payloads pin
  `use_msa_server: false` to avoid the dependency on a public service
  during automated checks. The public server rate-limits by caller, and this
  path is bounded by `OPENFOLD3_MSA_TIMEOUT_SECONDS` — see
  [Request budget and load shedding](#request-budget-and-load-shedding).
- **PDB structure templates** — fetched on-demand when `use_templates=true`.
  The harness payloads pin this `false` for the same reason. OF3's template
  preprocessing forks an `mp.Pool` (and an `mp.Manager`) even for its default
  `n_processes=1`; since `/predict` is sync, that forks off an AnyIO worker
  thread and the pool's `join()` never returns, wedging the worker for good.
  `patches/of3_template_no_fork.patch` gives the three classes that pool
  unconditionally the same `if workers > 1` guard the module's four batch
  functions already carry, so the default runs in-process. Only
  `TemplatePreprocessor` is on the served path; the other two are dataset-prep
  CLIs, patched for consistency. Dropping the pool also drops a process
  boundary — a `func_timeout` thread that outlives its 60 s deadline, or a
  crash in native parsing, now lands in the serving worker rather than in a
  disposable child.

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
