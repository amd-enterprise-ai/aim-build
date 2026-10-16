<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# DINOv2 ViT-L/14 Specialized Base Image (model-level)

Build context for the DINOv2 ViT-L/14 specialized base image — the Layer 1
container for the DINOv2 image-embedding AIM, served via BentoML on ROCm. The
service returns the 1024-dimensional CLS-token embedding of one RGB image per
request.

## Layout

- `Dockerfile`          — builds the service image from the upstream ROCm
                           PyTorch image, the DINOv2 source and `src/`.
- `requirements.txt`    — exact-pinned serving and DINOv2 runtime
                           dependencies. `torch` and `numpy` come from the base
                           image.
- `src/service.py`      — BentoML `Dinov2Vitl14Service` exposing a single
                           `/predict` endpoint. Compiles and warms up the model
                           in `__init__`, so `/healthz` reports ready only once
                           compilation has finished.
- `src/inference.py`    — `Dinov2Predictor`: model loading through
                           `torch.hub` from the pinned source, preprocessing,
                           `torch.compile`, and the forward pass. Has no
                           aim-runtime dependency.
- `src/harness.py`      — `Dinov2Vitl14Harness(ModelHarness)`. Discovered at
                           runtime by `aim_runtime.harness.discovery` from
                           `/workspace/model/src/harness.py`.
- `src/golden_fixtures.py` — five deterministic synthetic images (gradient,
                           checkerboard, rings, quadrants, diagonal stripes)
                           generated in code, so no image files ship in the
                           repository.
- `src/dinov2_golden_embeddings.json` — reviewed reference embeddings for the
                           five fixtures, captured on MI300X at fp16, with the
                           cosine-similarity threshold `evaluate` applies.
- `src/capture_golden_embeddings.py` — regenerates the golden manifest from a
                           running, approved AIM. See
                           [Refreshing the golden embeddings](#refreshing-the-golden-embeddings).
- `tests/`              — unit tests for the harness, request schema,
                           preprocessing and fixtures, discovered by CI.

## Build args

The Layer 1 build runs a plain `docker build` with no build-args. The only
overridable args:

- `DINOV2_REPO`   — defaults to `https://github.com/facebookresearch/dinov2.git`.
- `DINOV2_COMMIT` — defaults to `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
                    The build fails if the checked-out commit does not match.

## How CI resolves this directory

`aim_utils.specialized_utils.enumerate_specialized_base_targets` discovers
this `image/` directory as a **model-level** specialized base target
(`assets/<acc>/<org>/<model>/image/`), built as the model-dedicated
Layer 2 base `aim-instinct-facebookresearch-dinov2-vitl14-embedding-base`.
The pipeline is implemented in
[.github/workflows/base-image-pipeline.yaml](../../../../../.github/workflows/base-image-pipeline.yaml).

## Running the image

The container entrypoint is `/workspace/entrypoint.py`, the `aim-runtime` CLI.
The `aim-runtime` name itself is not on the image's `PATH`, so call the script
by its path when you `docker exec` into the container. Its `serve` subcommand
reads the engine config and the DINOv2 profile baked into the image, generates
the `bentoml serve` command, and execs it.

### Required environment

| Variable | Value | Why |
|---|---|---|
| `AIM_PROFILE_ID` | `bentoml-mi300x-fp16-tp1-latency` | Pins the only DINOv2 profile explicitly, so selection cannot depend on detected hardware. |
| `AIM_ACCELERATOR_COUNT` | `1` | The service uses one accelerator. The AIM accelerator detector may report every physical GPU on the host; this override constrains profile selection. |
| `AIM_ACCELERATOR_MODEL` | `MI300X` | Matches the profile's `accelerator_model` without relying on detection. |
| `AIM_PORT` | `8000` (default) | Port BentoML binds. `aim-runtime` passes it to the service as `--arg port=…`. |
| `HIP_VISIBLE_DEVICES` | GPU index, e.g. `0` | The service runs on the first GPU it sees. On a shared or multi-GPU host, set this to the GPU allocated to you. |

### Minimum docker invocation

```bash
docker run -d --name dinov2 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 \
  -v $HOME/dinov2_cache:/workspace/model-cache \
  -v $HOME/dinov2_inputs:/workspace/data/inputs:ro \
  -p 8000:8000 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  <dinov2-aim-image>:<tag> \
  serve

until curl -fsS http://localhost:8000/healthz >/dev/null 2>&1; do sleep 2; done; echo ready
```

Startup takes about 30 seconds once the weights are cached, most of it spent
compiling the model. The first start also downloads the weights (about
1.2 GB).

- `/workspace/model-cache` — `TORCH_HOME` points here. Mount a volume to keep
  the downloaded weights across restarts. The container writes the cache as
  root, so remove it through the image rather than with a plain `rm`:

  ```bash
  docker run --rm --entrypoint rm -v $HOME/dinov2_cache:/c <dinov2-aim-image>:<tag> -rf /c/torch
  ```
- `/workspace/data/inputs` — the only directory `image_path` requests may read
  from. Optional if you only send base64 images.

### The `/predict` API

Every request carries exactly one image, given as **either** a path inside the
container **or** base64-encoded bytes. The body is wrapped in `request`. The
examples assume an image of your own, `cat.jpg`, in the current directory and in
`$HOME/dinov2_inputs`:

```bash
# Base64-encoded image
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"request\": {\"image_base64\": \"$(base64 -w0 cat.jpg)\", \"include_timings\": true}}"

# Image already in the mounted inputs directory
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"request": {"image_path": "/workspace/data/inputs/cat.jpg"}}'
```

| Request field | Type | Meaning |
|---|---|---|
| `image_path` | string | Path under `/workspace/data/inputs` inside the container. |
| `image_base64` | string | Base64-encoded image bytes (JPEG, PNG, or anything Pillow reads). |
| `include_timings` | bool, default `false` | Add server-side preprocessing and GPU inference timings to the response. |

The response:

```json
{
  "model": "dinov2_vitl14",
  "source": "base64",
  "embedding_dim": 1024,
  "embedding": [0.0123, -0.0456, "…"],
  "norm": 47.21,
  "timings": {"preprocessing_ms": 1.6, "gpu_inference_ms": 3.6}
}
```

`embedding` is the raw, un-normalized CLS token; divide by `norm` for a unit
vector before computing cosine similarity. `timings` is `null` unless
requested.

**Status codes.** A request with neither or both image fields, a path outside
`/workspace/data/inputs`, or bytes that are not a readable image answers
**400** with an error message. Batches are not supported; send one request
per image.

### Preprocessing and compilation

Images are converted to RGB, resized so the short side is 256 pixels (bicubic),
center-cropped to 224×224, and normalized with the ImageNet mean and standard
deviation — the standard DINOv2 evaluation transform. The resize and crop run
on the CPU; conversion to float and normalization run on the GPU.

The forward pass runs under `torch.compile` (Inductor, static shapes) with fp16
autocast. Because every request is one 224×224 image, the model compiles once,
during startup, and never recompiles while serving.

| Env var | Profile value | Effect |
|---|---|---|
| `DINOV2_PRECISION` | `fp16` | `fp16` runs under autocast; `fp32` disables it. The golden embeddings are fp16. |
| `DINOV2_TORCH_COMPILE` | `true` | Set `false` to run eager, for debugging. |
| `DINOV2_COMPILE_MODE` | `default` | `torch.compile` mode. |
| `DINOV2_GPU_PREPROCESS` | not set (default `true`) | Set `false` to normalize on the CPU. |
| `DINOV2_IMAGE_SIZE` | `224` | Crop size. The golden embeddings assume 224. |

## Checking, benchmarking and evaluating with `aim-runtime`

With the service running, exec into the container and call the harness.
`list-checks` prints a table (add `--output-format json` for JSON); the other
commands print a JSON result and exit non-zero on failure.

```bash
# What the harness checks
docker exec dinov2 /workspace/entrypoint.py list-checks

# Service health: BentoML /healthz returns 200
docker exec dinov2 /workspace/entrypoint.py validate --service-url http://localhost:8000

# Latency and throughput: 5 warmup + 100 measured requests at batch size 1
docker exec dinov2 /workspace/entrypoint.py benchmark \
  --service-url http://localhost:8000 --output-dir /tmp/results

# Accuracy: compare the five golden fixtures with the reference embeddings
docker exec dinov2 /workspace/entrypoint.py evaluate \
  --service-url http://localhost:8000 --output-dir /tmp/results

docker cp dinov2:/tmp/results ./results
```

### Benchmark

The benchmark sends the five golden fixtures round-robin as base64 requests
with `include_timings: true`, and reports the p50 of:

- **HTTP end-to-end latency** — measured by the client, request to response.
- **Preprocessing** — decode, resize, crop and normalization, from the
  response's `timings`.
- **GPU inference** — the forward pass, measured with GPU events.

Also reported: mean, standard deviation, p95, p99, min, max and throughput in
images per second. Results are written to
`harness_benchmark_results.json` in `--output-dir`. Every measured request must
return a valid 1024-value embedding and timings, or the run fails.

To change the request counts, write a YAML file such as `bench.yaml`:

```yaml
num_requests: 100
num_warmup: 5
batch_sizes:
  - 1
request_timeout_s: 300
```

`--config` is read inside the container, so copy the file in first:

```bash
docker cp bench.yaml dinov2:/tmp/bench.yaml
docker exec dinov2 /workspace/entrypoint.py benchmark \
  --service-url http://localhost:8000 --config /tmp/bench.yaml --output-dir /tmp/results
```

Only batch size 1 is accepted.

### Evaluation

`evaluate` sends each of the five fixtures once and computes the cosine
similarity with its reference in `dinov2_golden_embeddings.json`. It passes
only if all five reach the manifest's threshold (0.999). The result, written to
`evaluate_results.json`, includes the mean and minimum similarity and the
worst fixture. An MI300X build of this image scores about 0.999999 on every
fixture.

### Refreshing the golden embeddings

Regenerate the manifest only after reviewing a change that is expected to move
the embeddings, such as a new DINOv2 commit or base image:

```bash
docker exec dinov2 python3 /workspace/model/src/capture_golden_embeddings.py \
  --service-url http://localhost:8000 --output /tmp/dinov2_golden_embeddings.json
docker cp dinov2:/tmp/dinov2_golden_embeddings.json src/dinov2_golden_embeddings.json
```

## Runtime dependencies (not provisioned at build time)

- **Model weights** — the official `dinov2_vitl14` checkpoint, downloaded by
  `torch.hub` from Meta's public URL on first start and cached under
  `TORCH_HOME=/workspace/model-cache/torch`. No Hugging Face token is needed.
  The model code itself is baked into the image at `/opt/dinov2`.
