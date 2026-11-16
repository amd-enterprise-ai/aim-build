<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
bioptimus/h-optimus-0 model in different environments. Follow these instructions to run the model on AMD
accelerators. This guide assumes a MI300X accelerator on the target system.

This AIM has an API of its own, shaped by what the model takes and returns. See
[Testing your deployment](#4-testing-your-deployment) for the endpoints and payloads of this model.

## Prerequisites

* AMD Instinct™ GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support
* A host directory for the model cache, if the model downloads its weights at first start
* Access to model repositories (Hugging Face account with appropriate permissions for gated models)

## 1. Docker deployment

### 1.1 Running the container

Set `AIM_PROFILE_ID` to one of the profiles listed in the [README](README.md). The runtime then uses that profile
and does not select a profile from the detected hardware.

```bash
mkdir -p /path/to/model-cache

docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_ID=bioptimus/h-optimus-0 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview
```

Where <YOUR_HUGGINGFACE_TOKEN> is your Hugging Face access token (required for gated models).

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_ID=bioptimus/h-optimus-0 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview
```

Without `AIM_PROFILE_ID`, the runtime selects a profile from the detected hardware.
`AIM_ACCELERATOR_COUNT` changes the accelerator count that the selection uses. The runtime does not use it when
`AIM_PROFILE_ID` is set.

## 2. Model caching for production

This AIM has no separate model download step. If the model downloads its weights, it does so at service
startup and keeps them in `$AIM_CACHE_PATH`, which defaults to `/workspace/model-cache`.

Mount a persistent volume at that path, as shown in section 1.1. Without the mount, the container downloads the
weights again at each start.

If the cache must be at a different path, set `AIM_CACHE_PATH` and mount the volume at the same path:

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_ID=bioptimus/h-optimus-0 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview
```

The first start is slower than later starts, because it downloads the weights. Size the volume for the model
checkpoint, and keep it writable.

## 3. Kubernetes deployment

### 3.1 Deployment

It is possible to deploy AIM with Kubernetes. This section gives a sample Kubernetes deployment manifest.

Create a secret for the Hugging Face token used to download the model:

```bash
kubectl create secret generic hf-token \
    --from-literal="hf-token=<YOUR_HUGGINGFACE_TOKEN>" \
    -n <YOUR_K8S_NAMESPACE>
```

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
          image: amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "bioptimus/h-optimus-0"
            - name: AIM_PROFILE_ID
              value: "bentoml-mi300x-fp16-tp1-throughput"
            - name: AIM_CACHE_PATH
              value: "/workspace/model-cache"
            - name: AIM_LOG_LEVEL_ROOT
              value: "INFO"
            - name: AIM_LOG_LEVEL
              value: "INFO"
            - name: AIM_PORT
              value: "8000"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: hf-token
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
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
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

The readiness probe checks `/readyz` every 10 seconds. After three failed checks (30 seconds) Kubernetes takes the
pod out of the service endpoints, so a worker that fails after a successful start stops receiving requests until it
recovers. The liveness probe below is slower on purpose and only restarts the container.

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

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_ID=bioptimus/h-optimus-0 \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_ID=bioptimus/h-optimus-0 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=bioptimus/h-optimus-0 \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-bioptimus-h-optimus-0:2026.24.0-preview \
  list-profiles
```

### 6.5 Common problems

* **The container restarts during the first start.** The manifest in section 3.1 gives the start one hour. If the
  model needs more, raise `failureThreshold` on the startup probe. Mount a persistent cache, so that later starts
  skip the download.
* **The container restarts after it answers on `/healthz`.** The liveness probe accepts five minutes of failed
  checks. If the model does more work after the health endpoint answers, and that work takes longer, raise
  `failureThreshold` on the liveness probe.
* **The container stops with a profile selection error.** The error message gives the cause.
  * If `AIM_PROFILE_ID` is set, the ID matches no profile of this AIM. Compare it with the [README](README.md) or the
    output of `list-profiles` (see section 6.4).
  * If the error lists profile IDs, the matching profiles must be selected explicitly. Set `AIM_PROFILE_ID` to one of
    them.
  * If no profile is compatible, the error counts the profiles per reason, such as `accelerator_mismatch` or
    `precision_mismatch`. Compare the `Detected Accelerator` and `Accelerator Count` lines in the log, and any
    `AIM_PRECISION` or `AIM_METRIC` you set, with the profile table in the [README](README.md). Or set
    `AIM_PROFILE_ID` to a profile from the README (see section 1.1).
* **`/healthz` does not answer yet.** The service binds the port only after the model is loaded. Read the container
  logs to see how far startup got.

## 7. Security considerations

* Never include HF_TOKEN in Dockerfiles or commit it to version control
* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment
* The model cache holds downloaded model weights. Restrict write access to it