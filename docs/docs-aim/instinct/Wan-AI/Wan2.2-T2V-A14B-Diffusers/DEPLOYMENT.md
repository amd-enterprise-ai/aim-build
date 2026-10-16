<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide provides step-by-step instructions for deploying AMD Inference Microservice (AIM) container for
 Wan-AI/Wan2.2-T2V-A14B-Diffusers model in
various environments. Follow these instructions to quickly get started with running an AI model on AMD accelerators.
This guide assumes MI300X accelerator on the target system.

## Prerequisites

* AMD Instinct™ GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support

## 1. Docker deployment

### 1.1 Running the container

```bash
docker run \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
```


### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_METRIC=throughput \
  -e AIM_PORT=8080 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
```

## 2. Model caching for production

For production environments, pre-download models to a persistent cache:

### 2.1 Download model to cache

```bash
# Create persistent cache directory
mkdir -p /path/to/model-cache

# Download model using the download-to-cache command
docker run --rm \
  -v /path/to/model-cache:/workspace/model-cache \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0 \
  download-to-cache --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

### 2.2 Run with pre-cached model

```bash
docker run \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
```

## 3. Kubernetes deployment

### 3.1 Deployment

It is possible to deploy AIM using Kubernetes. In this doc a sample Kubernetes deployment manifest is provided.

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
          image: amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
          imagePullPolicy: Always
          env:
            - name: AIM_PRECISION
              value: "auto"
            - name: AIM_ACCELERATOR_COUNT
              value: "1"
            - name: AIM_ACCELERATOR_MODEL
              value: "auto"
            - name: AIM_ENGINE
              value: "vllm_omni"
            - name: AIM_METRIC
              value: "latency"
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
              memory: "16Gi"
              cpu: "4"
              amd.com/gpu: "1"
            limits:
              memory: "16Gi"
              cpu: "4"
              amd.com/gpu: "1"
          startupProbe:
            httpGet:
              path: /v1/models
              port: http
            periodSeconds: 10
            failureThreshold: 360
          livenessProbe:
            httpGet:
              path: /health
              port: http
          readinessProbe:
            httpGet:
              path: /v1/models
              port: http
          volumeMounts:
            - name: ephemeral-storage
              mountPath: /tmp
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: ephemeral-storage
          emptyDir:
            sizeLimit: 256Gi
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
```

Create a service configuration in `service.yaml` file:

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

The deployment can be customized further based on your requirements. It can be applied to Kubernetes cluster using `kubectl`:

```bash
kubectl apply -f . -n <YOUR_K8S_NAMESPACE>
```

## 4. Testing your deployment

To test the deployment, an API call can be executed. To do that a pod should be found and then port forwarding should be
set up.

```bash
kubectl port-forward service/minimal-aim-deployment 8000:80 -n <YOUR_K8S_NAMESPACE>
```

After executing port forwarding, the API becomes accessible locally at `http://localhost:8000/v1/`.

### 4.1 Health check

This AIM serves the vLLM-Omni diffusion API, not the OpenAI text-completion API. Confirm the service is ready before
sending requests:

```bash
curl http://localhost:8000/v1/models
```

The service answers with HTTP 200 and the served model once it is ready to accept requests. The Kubernetes startup and
readiness probes in section 3.1 use this endpoint.

### 4.2 The endpoints

This AIM serves the vLLM-Omni diffusion API for text-to-video generation. Generation is asynchronous: submit a job,
poll its status, then download the clip.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/videos` | Submit a text-to-video generation job. Returns a job id. |
| `GET` | `/v1/videos/{id}` | Report the status of a job (`queued`, `completed`, `failed`). |
| `GET` | `/v1/videos/{id}/content` | Download the generated video once the job is `completed`. |
| `GET` | `/v1/models` | Report the served model. Used for the health check in section 4.1. |

This AIM does **not** serve the OpenAI `/v1/completions` or `/v1/chat/completions` endpoints.

### 4.3 Request body

`POST /v1/videos` takes `multipart/form-data` fields (send them as form fields, not JSON):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `prompt` | string | required | Text description of the video to generate. |
| `negative_prompt` | string | not set | Text to steer generation away from. The model runs classifier-free guidance, so a negative prompt is supported. |
| `size` | string | model default | Frame resolution as `WIDTHxHEIGHT`, for example `832x480`. |
| `seconds` | integer | model default | Clip length in seconds. |
| `fps` | integer | model default | Frames per second. |
| `num_inference_steps` | integer | model default | Number of denoising steps. Fewer steps are faster and lower quality; this dominates wall-clock time. |
| `seed` | integer | not set | Random seed for reproducible output. |

### 4.4 Example request

Submit a job, poll until it completes, then download the clip:

```bash
# Submit the job and capture its id
JOB=$(curl -s http://localhost:8000/v1/videos \
  -F prompt="a red panda surfing a wave, cinematic" \
  -F size=832x480 -F seconds=5 -F fps=16 -F num_inference_steps=18 \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

# Poll until the status is "completed" (or "failed")
curl -s http://localhost:8000/v1/videos/$JOB

# Download the finished clip
curl -s http://localhost:8000/v1/videos/$JOB/content -o out.mp4
```

The same flow in Python:

```python
import time
import requests

base = "http://localhost:8000/v1/videos"

job = requests.post(
    base,
    files={
        "prompt": (None, "a red panda surfing a wave, cinematic"),
        "size": (None, "832x480"),
        "seconds": (None, "5"),
        "fps": (None, "16"),
        "num_inference_steps": (None, "18"),
    },
    timeout=300,
).json()["id"]

while True:
    status = requests.get(f"{base}/{job}", timeout=30).json()["status"]
    if status in ("completed", "failed"):
        break
    time.sleep(2)

if status == "completed":
    content = requests.get(f"{base}/{job}/content", timeout=120).content
    with open("out.mp4", "wb") as fh:
        fh.write(content)
```

### 4.5 Response

`POST /v1/videos` answers with a job id and its initial status:

```json
{
  "id": "vid-abc123",
  "status": "queued"
}
```

`GET /v1/videos/{id}` reports the current job state:

```json
{
  "id": "vid-abc123",
  "status": "completed"
}
```

| `status` | Meaning |
|---|---|
| `queued` / `in_progress` | The job is still generating. Keep polling. |
| `completed` | The clip is ready. Download it from `/v1/videos/{id}/content`. |
| `failed` | Generation failed. The `error` field holds the reason. |

`GET /v1/videos/{id}/content` streams the generated video (an MP4) as the response body.

### 4.6 What to know before you deploy

**Multi-GPU profiles need shared memory.** A profile named `...-uspN` with N greater than one shards the latent
sequence across N accelerators with Unified Sequence Parallelism, and the workers exchange data over shared memory.
Run the container with `--ipc=host` (or `--shm-size=32g`) and set `AIM_ACCELERATOR_COUNT` to N, or the workers fail to
start. The [README](README.md) lists the available profiles and their accelerator counts.

**`num_inference_steps` dominates latency.** Each step runs the full diffusion transformer, and the model runs two
passes per step for classifier-free guidance. Lower the step count for faster, lower-quality drafts and raise it for
final renders.

**The checkpoint is large and downloaded at first start.** It is kept in `$AIM_CACHE_PATH`, which defaults to
`/workspace/model-cache`. Mount a persistent volume there, as described in section 2, so that later starts skip the
download.

**Submission can be slow.** Some builds generate the clip synchronously on `POST /v1/videos` rather than returning a
job id immediately. Set a client timeout long enough for a short generation, not just an enqueue round-trip.

### 4.7 Validating and benchmarking the deployment

The image ships a harness that the `aim-runtime` CLI drives. Against the container started in section 1.1 (here named
`<CONTAINER>`), run the checks to confirm health and a smoke generation, or to benchmark throughput and latency:

```bash
# List the checks this image exposes
docker exec <CONTAINER> aim-runtime list-checks

# Validate against the running service
docker exec <CONTAINER> aim-runtime validate \
  --service-url http://localhost:8000 --scope runtime --scope offline

# Benchmark against the running service
docker exec <CONTAINER> aim-runtime benchmark \
  --service-url http://localhost:8000 \
  --timeout-seconds 1800
```

The 1800-second timeout applies to the whole benchmark. Increase it if the run needs more time.

`validate` reports `health` and `smoke_video`; `benchmark` reports `throughput_qps` and latency percentiles for the
chosen recipe.

## 5. Advanced deployment scenarios

### 5.1 Using custom profiles

```bash
# Create custom profile directory
mkdir -p custom-profiles

# Add your custom profile YAML
cat > custom-profiles/vllm-custom-profile.yaml << EOF
aim_id: Wan-AI/Wan2.2-T2V-A14B-Diffusers
model_id: Wan-AI/Wan2.2-T2V-A14B-Diffusers
metadata:
  accelerator_count: 1
  accelerator_model: MI300X
  accelerator_type: gpu
  engine: vllm_omni
  metric: throughput
  precision: fp16
  type: unoptimized

engine_args:
  gpu-memory-utilization: 0.95
  dtype: float16
  tensor-parallel-size: 1
  max-num-batched-tokens: 1024
  max-model-len: 2048

env_vars:
  VLLM_DO_NOT_TRACK: "1"
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
EOF

# Run with custom profile
docker run \
  -v $(pwd)/custom-profiles:/workspace/aim-runtime/profiles/custom \
  -e AIM_METRIC=throughput \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
```

### 5.2 Using profiles excluded from automatic selection

To use a profile excluded from automatic selection, an environment variable `AIM_PROFILE_ID` should be set with the
desired profile identifier. Profile identifier is the filename of the profile without the `.yaml` extension.

```bash
docker run \
  -e AIM_PROFILE_ID=vllm_omni-mi300x-fp16-tp2-latency-usp2 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
```

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0 \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects based on the provided environment variables.

```bash
docker run \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0 \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-wan-ai-wan2-2-t2v-a14b-diffusers:2026.9.0 \
  list-profiles
```

## 7. Security considerations

* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment