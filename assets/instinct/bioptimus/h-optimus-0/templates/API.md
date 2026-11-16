<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 Endpoints

This AIM serves one model endpoint and the shared AIM health and model-listing endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Return the H-optimus-0 embedding of one 224×224 histopathology tile. |
| `GET` | `/health` | Readiness. Forwards to the BentoML `/healthz` check of the same service and returns its status code. |
| `GET` | `/v1/models` | Lists the model the container serves. |
| `GET` | `/healthz` | BentoML built-in health check. The service also keeps BentoML's `/livez`, `/readyz` and `/metrics`. |

There is no `/predict` endpoint. Requests to `/predict` are not served; use `/v1/inference`.

`GET /health` returns `{"status": "ok"}` with status `200` when the BentoML health check succeeds. It returns
`{"status": "unavailable"}` with the upstream status code if the check fails, and with `503` if the service cannot reach
its own health check. Use it as a readiness signal, not as a liveness signal.

`GET /v1/models` returns an OpenAI-style list with the single served model. `metadata.dtype` is the precision the model
runs in, and `metadata.accelerator` is the name PyTorch reports for the first visible GPU (`null` if it cannot be read):

```json
{
  "object": "list",
  "data": [
    {
      "id": "bioptimus/H-optimus-0",
      "object": "model",
      "metadata": {"dtype": "fp16", "accelerator": "<GPU name reported by PyTorch>"}
    }
  ]
}
```

### 4.3 Request body

The request body is a flat JSON object. Each request carries exactly one image:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_base64` | string | not set | Base64-encoded bytes of a PNG or JPEG image. |
| `image_path` | string | not set | Path to an image file inside the container, under the input directory (`AIM_INPUT_PATH`, default `/workspace/data`). |

Provide exactly one of the two fields. If you provide both or neither, the service answers `400` and does not run the
model.

The tile must be a hematoxylin and eosin (H&E) stained histopathology image of exactly 224×224 pixels, extracted at
0.5 microns per pixel, as the model card requires. The service does not resize or crop: an image of any other size is
rejected with `400`, because resizing a tile of another size or magnification would give a plausible but meaningless
embedding. Resize or re-tile your images before you send them.

The service converts the image to RGB, converts it to a tensor and normalizes each colour channel with mean
`(0.707223, 0.578729, 0.703617)` and standard deviation `(0.211883, 0.230117, 0.177517)`, as specified in the model
card. This is a fixed per-channel scaling, not stain normalisation, and it does not change the stain appearance of the
tile. Stain normalisation is not required. H-optimus-0 was trained on H&E-stained whole-slide images. Other stains (IHC,
or special stains such as PAS, trichrome, Giemsa) are outside that training distribution. The service still returns an
embedding, but the model card gives no basis for interpreting it; validate before relying on it. Frozen sections are
H&E-stained, so they are not a different stain, but their tissue preparation differs from the usual slides: expect a
domain shift.

H-optimus-0 is a tile encoder: one request embeds one tile. A slide-level task needs you to cut the slide into tiles,
embed each one and aggregate the embeddings yourself (for example with multiple-instance learning). There is no batch
form, so a slide of thousands of tiles takes thousands of requests.


### 4.4 Example request

The examples assume a tile of your own, `tile.png`, in the current directory, and that the service is reachable at
`localhost:8000`; adjust the port to the one you published.

Check readiness and the served model:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the embedding request. The base64 text of a tile can exceed the operating system's limit on a single command-line
argument (128 KiB on Linux, reached by a PNG of about 96 KB), so write the request body to a file and send the file:

```bash
printf '{"image_base64": "%s"}' "$(base64 -w0 tile.png)" > request.json
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

The same request in Python:

```python
import base64

import requests

with open("tile.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"image_base64": image_base64},
    timeout=120,
)
response.raise_for_status()

result = response.json()
print(result["embedding_dim"], len(result["embedding"]))
```

### 4.5 Request with an image path

To read tiles from storage instead of sending their bytes, mount the directory into the container when you start it,
for example with `-v /data/tiles:/workspace/data:ro`, and give the path inside the container. The service resolves
symbolic links and accepts only files under the input directory, `AIM_INPUT_PATH`, which defaults to `/workspace/data`.
A path outside it, or a file that does not exist, is rejected with `400`. To use another directory, set `AIM_INPUT_PATH`
and mount the volume at the same path.

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/workspace/data/tile.png"}'
```

### 4.6 Response

A successful response has this shape:

```json
{
  "embedding": [0.0123, -0.0456, "..."],
  "embedding_dim": 1536,
  "model": "bioptimus/H-optimus-0"
}
```

| Key | Purpose |
|---|---|
| `embedding` | The CLS-token representation of the tile, as a list of 1536 floats. It is not normalized to unit length. |
| `embedding_dim` | The constant `1536`. The service does not compute it from `embedding`. |
| `model` | The constant `"bioptimus/H-optimus-0"`. |

The model runs in fp16 under `torch.autocast`, and the service returns the embedding as float32 values converted from
the fp16 result.

### 4.7 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | Both or neither of `image_base64` and `image_path` was given, `image_path` is not an existing file under the input directory, the input is not valid base64 or not a readable image, or the image is not exactly 224×224 pixels. | Correct the request. |
| `503` | Returned by `GET /health` when the service cannot reach its own health check. | Retry later. |
| `5xx` | A server-side failure, for example during inference. The response body is generic. | Check the container log. |

Errors are never returned as a `200` response with an error flag. The `400` response names the rule that was broken but
not the contents of the underlying error. The server-side request timeout is 120 seconds.

### 4.8 What to know before you deploy

**A Hugging Face token is required.** The H-optimus-0 weights are in a gated Hugging Face repository,
`bioptimus/H-optimus-0`. Request access to it with the Hugging Face account that owns the token, and pass the token to
the container as `HF_TOKEN`, as shown in section 1. The service downloads the weights with `timm` at startup, so
without access to the repository, the service does not become ready: the weights error is logged and the container keeps
running (BentoML restarts the worker), so check the container logs. The profiles do not carry the
token; you always supply it at container start.

**The service needs a GPU.** At startup the service checks `torch.cuda.is_available()` and fails to start with a
`RuntimeError` if no GPU is visible, instead of falling back to the CPU. The container keeps running and never becomes
ready, so check the container logs. It uses the first GPU visible to the
container. Both profiles, `bentoml-mi300x-fp16-tp1-throughput` and
`bentoml-mi355x-fp16-tp1-throughput`, use one accelerator, an MI300X and an MI355X respectively. On a host with several
accelerators, restrict the container to the one it should use. To serve more tiles per second, run more replicas.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**The service loads the model once, at startup.** The service reports ready on `GET /health` (and `GET /healthz`) only after its
initialization, which includes downloading and loading the weights, has finished. The first start downloads the
weights, so allow it more time than later starts. Mount a persistent volume for the model cache, as described in
section 2. Weights are read and written only under `AIM_CACHE_PATH`, in the Hugging Face hub layout. At startup the
service logs that path and whether `HF_TOKEN` is set. The repository is gated, so if the weights are not already in the
cache and `HF_TOKEN` is not set, or the download fails, the service stops with an error naming `AIM_CACHE_PATH` and
`HF_TOKEN`; the container keeps running and never becomes ready.

**Every request uses the same input shape.** Each image must be 224×224 pixels, so the model always sees a
tensor of shape 1×3×224×224.

**Benchmark metric.** `aim-runtime benchmark` reports `images_per_sec`: one request is one 224×224 image, sent
sequentially (concurrency 1), and the value is the number of successful requests divided by the sum of their
latencies, including the HTTP round trip and excluding warm-up. If any timed request fails, the benchmark is reported
as failed. The profile `metric` field is `throughput`.

### 4.9 Model environment variables

The service reads these environment variables:

| Variable | Profile value | Effect |
|---|---|---|
| `HF_TOKEN` | secret, you provide it | Authenticates the download of the gated model weights. Required. |
| `CUDA_VISIBLE_DEVICES` | not set by the profile | Restricts which GPUs the container sees. The service uses the first visible GPU. |
| `AIM_INPUT_PATH` | `/workspace/data` | The only directory `image_path` may read from. |
| `AIM_CACHE_PATH` | `/workspace/model-cache` | Where the model weights are downloaded and cached. |
| `HOPTIMUS0_NUM_THREADS` | `min(8, usable CPUs)` | CPU threads PyTorch uses (`torch.set_num_threads`), applied at startup. Each request decodes and normalises the image on the CPU, and a large default thread pool (one per CPU) stalls that step in some requests. The default is a small fixed count because a large default pool stalls this step in some requests. Must be a positive integer, otherwise startup stops. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` default to the same value unless you set them. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` in the MI300X profile, not set in the MI355X profile | Selects the hipBLASLt backend for matrix multiplication in PyTorch. |

The service has no other model-specific settings. The precision (fp16), input size (224×224) and embedding size (1536)
are fixed in the service code.
