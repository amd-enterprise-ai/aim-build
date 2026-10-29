<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
facebookresearch/dinov2-vitl14-embedding model in different environments. Follow these instructions to run the model on AMD
accelerators. This guide assumes a MI300X accelerator on the target system.

This AIM has an API of its own, shaped by what the model takes and returns. See
[Testing your deployment](#4-testing-your-deployment) for the endpoints and payloads of this model.

## Prerequisites

* AMD Instinct™ GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support
* A host directory for the model cache, if the model downloads its weights at first start

## 1. Docker deployment

### 1.1 Running the container

Pin the profile explicitly, so that the selection cannot depend on the hardware the runtime detects. Set
`AIM_PROFILE_ID` to one of the profiles listed in the [README](README.md), and set `AIM_ACCELERATOR_COUNT` to the
accelerator count of that profile.

```bash
mkdir -p /path/to/model-cache

docker run \
  -e AIM_ID=facebookresearch/dinov2-vitl14-embedding \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview
```

`AIM_ACCELERATOR_COUNT` must agree with the `accelerator_count` of the selected profile. Set an integer, not `auto`,
because some AIMs read the value directly.

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=facebookresearch/dinov2-vitl14-embedding \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview
```

## 2. Model caching for production

This AIM has no separate model download step. If the model downloads its weights, it does so at service
startup and keeps them in `$AIM_CACHE_PATH`, which defaults to `/workspace/model-cache`.

Mount a persistent volume at that path, as shown in section 1.1. Without the mount, the container downloads the
weights again at each start.

If the cache must be at a different path, set `AIM_CACHE_PATH` and mount the volume at the same path:

```bash
docker run \
  -e AIM_ID=facebookresearch/dinov2-vitl14-embedding \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview
```

The first start is slower than later starts, because it downloads the weights. Size the volume for the model
checkpoint, and keep it writable.

## 3. Kubernetes deployment

### 3.1 Deployment

It is possible to deploy AIM with Kubernetes. This section gives a sample Kubernetes deployment manifest.

Create `deployment.yaml` with the following content:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minimal-aim-deployment
  labels:
    app: minimal-aim-deployment
spec:
  progressDeadlineSeconds: 3600
  replicas: 1
  selector:
    matchLabels:
      app: minimal-aim-deployment
  template:
    metadata:
      labels:
        app: minimal-aim-deployment
    spec:
      containers:
        - name: minimal-aim-deployment
          image: amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "facebookresearch/dinov2-vitl14-embedding"
            - name: AIM_PROFILE_ID
              value: "bentoml-mi300x-fp16-tp1-latency"
            - name: AIM_ACCELERATOR_COUNT
              value: "1"
            - name: AIM_CACHE_PATH
              value: "/workspace/model-cache"
            - name: AIM_LOG_LEVEL_ROOT
              value: "INFO"
            - name: AIM_LOG_LEVEL
              value: "INFO"
            - name: AIM_PORT
              value: "8000"
          ports:
            - name: http
              containerPort: 8000
          resources:
            requests:
              memory: "64Gi"
              cpu: "8"
              amd.com/gpu: "1"
            limits:
              memory: "64Gi"
              cpu: "8"
              amd.com/gpu: "1"
          startupProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 30
          volumeMounts:
            - name: model-cache
              mountPath: /workspace/model-cache
            - name: ephemeral-storage
              mountPath: /tmp
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: minimal-aim-model-cache
        - name: ephemeral-storage
          emptyDir:
            sizeLimit: 256Gi
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
```

The startup probe gives the start up to one hour: 360 checks at 10-second intervals. That covers a first start that
downloads weights, as described in section 2. The pod stays out of the service endpoints until this probe succeeds,
so no request reaches a model that is still loading. A later start reads the cache and succeeds much sooner.

Kubernetes runs the liveness probe only after the startup probe succeeds. The liveness probe then accepts 30 failed
checks, which is five minutes, before it restarts the container. This tolerates a model that does more work after the
health endpoint answers.

The `memory` and `cpu` values above are a starting point. Raise them if the profile uses more than one accelerator,
because this AIM starts one worker process per accelerator, and each worker holds its own copy of
the model.

Create the cache volume claim in `pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minimal-aim-model-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

Create a service configuration in the `service.yaml` file:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: minimal-aim-deployment
  labels:
    app: minimal-aim-deployment
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 80
      targetPort: 8000
  selector:
    app: minimal-aim-deployment
```

Customize the deployment further to fit your needs. Apply it to the Kubernetes cluster with `kubectl`:

```bash
kubectl apply -f . -n <YOUR_K8S_NAMESPACE>
```

## 4. Testing your deployment

To test the deployment, execute an API call. First find the pod, then set up port forwarding.

```bash
kubectl port-forward service/minimal-aim-deployment 8000:80 -n <YOUR_K8S_NAMESPACE>
```

After port forwarding starts, the API is available locally at `http://localhost:8000/`.

### 4.1 Health check

```bash
curl http://localhost:8000/healthz
```

The service answers with HTTP 200 when it is ready to accept requests. The startup probe and the liveness probe in
section 3.1 use it.

`/healthz` answers a `GET`. An AIM can also give an endpoint that reports the state in more detail. That endpoint
takes a `POST`, and the section below names it if this AIM has one.

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

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=facebookresearch/dinov2-vitl14-embedding \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -e AIM_ACCELERATOR_COUNT=<ACCELERATOR_COUNT_OF_THAT_PROFILE> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview
```

`AIM_ACCELERATOR_COUNT` must match the `accelerator_count` of the selected profile. If the two disagree, the
container stops at startup with a profile selection error.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=facebookresearch/dinov2-vitl14-embedding \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=facebookresearch/dinov2-vitl14-embedding \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-facebookresearch-dinov2-vitl14-embedding:2026.14.0-preview \
  list-profiles
```

### 6.5 Common problems

* **The container restarts during the first start.** The manifest in section 3.1 gives the start one hour. If the
  model needs more, raise `failureThreshold` on the startup probe. Mount a persistent cache, so that later starts
  skip the download.
* **The container restarts after it answers on `/healthz`.** The liveness probe accepts five minutes of failed
  checks. If the model does more work after the health endpoint answers, and that work takes longer, raise
  `failureThreshold` on the liveness probe.
* **The container stops with a profile selection error.** `AIM_PROFILE_ID` and `AIM_ACCELERATOR_COUNT` must agree.
* **`/healthz` does not answer yet.** The service binds the port only after the model is loaded. Read the container
  logs to see how far startup got.

## 7. Security considerations

* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment
* The model cache holds downloaded model weights. Restrict write access to it