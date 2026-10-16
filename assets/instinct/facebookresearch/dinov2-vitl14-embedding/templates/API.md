<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The prediction endpoint

This AIM serves one prediction endpoint:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Return the DINOv2 ViT-L/14 embedding of one image. |

The service also answers `GET /healthz`, `/livez` and `/readyz` for probes, `GET /metrics` in Prometheus format, and
`GET /docs.json` with the OpenAPI schema.

### 4.3 Request body

The whole request is nested under a top-level `request` object. Each request carries exactly one image:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_base64` | string | not set | Base64-encoded image bytes. Give the plain base64 text, without a `data:image/...;base64,` prefix. |
| `image_path` | string | not set | Absolute path to an image inside the container, under `/workspace/data/inputs`. |
| `include_timings` | boolean | `false` | Add server-side preprocessing and accelerator timings to the response. |

The service accepts any image format that Pillow reads, for example JPEG, PNG or WebP, in any size and colour mode.
It applies the EXIF orientation, converts the image to RGB, discarding any alpha channel, resizes the short side to 256
pixels (bicubic), center-crops 224×224 pixels and normalizes with the ImageNet mean and standard deviation. This is the
standard DINOv2 evaluation transform, so send the original image and not a preprocessed one.

The service rejects these requests with HTTP 400, before the model runs:

* Both `image_base64` and `image_path`, or neither of them, or an empty string as the only image field.
* An `image_path` that is relative, or that resolves outside `/workspace/data/inputs`. `..` segments and symbolic
  links are resolved first.
* An `image_path` that does not exist, or bytes that are not valid base64 or not a readable image.

Fields the service does not know are ignored. There is no batch form: send one request per image.

### 4.4 Example request

The examples assume an image of your own, `cat.jpg`, in the current directory. Any format listed in section 4.3
works.

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"request\": {\"image_base64\": \"$(base64 -w0 cat.jpg)\", \"include_timings\": true}}"
```

The same request in Python:

```python
import base64

import requests

with open("cat.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/predict",
    json={"request": {"image_base64": image_base64, "include_timings": True}},
    timeout=60,
)
response.raise_for_status()

result = response.json()
unit_embedding = [x / result["norm"] for x in result["embedding"]]
print(result["embedding_dim"], result["timings"])
```

### 4.5 Request with an image path

To read images from storage instead of sending their bytes, mount the directory read-only at
`/workspace/data/inputs` when you start the container, for example with
`-v /data/images:/workspace/data/inputs:ro`, and give the path inside the container. With `cat.jpg`
in `/data/images`:

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"request": {"image_path": "/workspace/data/inputs/cat.jpg"}}'
```

The directory does not exist in the image. Without the mount, every `image_path` request fails with HTTP 400.

### 4.6 Response

A successful response has this shape:

```json
{
  "model": "dinov2_vitl14",
  "source": "base64",
  "embedding_dim": 1024,
  "embedding": [0.0123, -0.0456, "..."],
  "norm": 45.83,
  "timings": {"preprocessing_ms": 1.6, "gpu_inference_ms": 3.6}
}
```

| Key | Purpose |
|---|---|
| `model` | The DINOv2 model name, `dinov2_vitl14`. |
| `source` | `"base64"`, or the `image_path` exactly as given in the request. |
| `embedding_dim` | Length of `embedding`, always `1024`. |
| `embedding` | The CLS token after the model's final layer norm, as 1024 floats. It is not normalized to unit length. |
| `norm` | The L2 norm of `embedding`. Divide `embedding` by `norm` for a unit vector before computing cosine similarity. |
| `timings` | `null`, unless the request sets `include_timings: true`. Then `preprocessing_ms` covers image decoding, resizing, cropping, copying to the accelerator and normalization, and `gpu_inference_ms` covers the model forward pass, measured on the accelerator. Base64 decoding and HTTP handling are not included. |

The model runs in fp16. Embeddings of the same image agree to a cosine similarity of about 0.999999 between runs on
the same accelerator model; they are not bit-identical across accelerator models or software versions.

### 4.7 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | The request is invalid. See section 4.3. | No. Correct the request. |
| `500` | Unexpected server error, including an image larger than 178,956,970 pixels, which Pillow refuses to decode. | No, for an oversized image. Downscale it first. |

A `400` body has one of two shapes. A request that does not match the schema, for example with both image fields,
without the `request` wrapper, or with malformed JSON, returns an object:

```json
{"error": "1 validation error for Input", "detail": [{"loc": ["request"], "msg": "Value error, provide exactly one of image_path or image_base64", "...": "..."}]}
```

A request that matches the schema but whose image cannot be used, for example an unreadable file or a path outside
`/workspace/data/inputs`, returns a list with one object:

```json
[{"error": "image path must be under /workspace/data/inputs"}]
```

Read the `error` field from either shape.

### 4.8 What to know before you deploy

**The service runs on one accelerator.** The only profile, `bentoml-mi300x-fp16-tp1-latency`, uses one MI300X. On a
host with several accelerators, restrict the container to the accelerator it should use. To serve more images per
second, run more replicas.

**The service accepts connections only once it is ready.** At startup it loads the model and compiles it with
`torch.compile`, running one image through the full request path. This takes about 30 seconds. The HTTP port opens
only after that, so no request waits for compilation. Use `GET /readyz` as the readiness probe.

**Every request uses the same input shape.** Because each image is cropped to 224×224 pixels, the model compiles
once at startup and never recompiles while serving. On an MI300X, a request takes about 6 ms end to end, of which
about 3.5 ms is the forward pass.

**The checkpoint is about 1.2 GB.** It is downloaded from Meta's public server at the first start and kept in
`/workspace/model-cache/torch`. Mount a persistent volume at `/workspace/model-cache`, as described in section 2.
The location follows `TORCH_HOME`, not `AIM_CACHE_PATH`: changing `AIM_CACHE_PATH` does not move this checkpoint.
The container writes the cache as root.

**This model does not need a Hugging Face token.**

### 4.9 Model environment variables

The profile sets these values. Change them only for debugging or experiments:

| Variable | Profile value | Effect |
|---|---|---|
| `DINOV2_PRECISION` | `fp16` | `fp16` runs the model under fp16 autocast. `fp32` runs it in full precision. |
| `DINOV2_TORCH_COMPILE` | `true` | Set `false` to run the model without `torch.compile`. |
| `DINOV2_COMPILE_MODE` | `default` | The `torch.compile` mode. |
| `DINOV2_GPU_PREPROCESS` | not set, which means `true` | Set `false` to normalize the image on the CPU. |
| `DINOV2_IMAGE_SIZE` | `224` | Crop size in pixels, a multiple of 14. The resize scales with it. Other values change the embeddings. |
| `AIM_DATA_DIR` | not set, which means `/workspace/data` | Parent of the `inputs` directory that `image_path` requests may read. |

The server-side request timeout is 300 seconds. A client timeout of 60 seconds is ample for one image.
