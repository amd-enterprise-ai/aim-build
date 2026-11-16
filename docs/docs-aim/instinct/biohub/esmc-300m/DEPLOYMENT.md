<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
biohub/esmc-300m model in different environments. Follow these instructions to run the model on AMD
accelerators. This guide assumes a MI300X accelerator on the target system.

This AIM has an API of its own, shaped by what the model takes and returns. See
[Testing your deployment](#4-testing-your-deployment) for the endpoints and payloads of this model.

## Prerequisites

* AMD Instinct™ GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support
* A host directory for the model cache, if the model downloads its weights at first start

## 1. Docker deployment

### 1.1 Running the container

Set `AIM_PROFILE_ID` to one of the profiles listed in the [README](README.md). The runtime then uses that profile
and does not select a profile from the detected hardware.

```bash
mkdir -p /path/to/model-cache

docker run \
  -e AIM_ID=biohub/esmc-300m \
  -e AIM_PROFILE_ID=bentoml-mi300x-bf16-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview
```

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=biohub/esmc-300m \
  -e AIM_PROFILE_ID=bentoml-mi300x-bf16-tp1-throughput \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview
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
  -e AIM_ID=biohub/esmc-300m \
  -e AIM_PROFILE_ID=bentoml-mi300x-bf16-tp1-throughput \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview
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
          image: amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "biohub/esmc-300m"
            - name: AIM_PROFILE_ID
              value: "bentoml-mi300x-bf16-tp1-throughput"
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

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Embed one protein sequence (default), or score it with a masked-token loss and perplexity (`mode` `masked_lm`). |
| `GET` | `/v1/models` | Report the served model. |
| `GET` | `/health` | Check whether the service is ready to accept requests. |

`/healthz` also answers with HTTP 200 once the model is loaded; see section 4.1.

### 4.3 Request body

Request bodies are flat JSON objects. Do not nest the fields under another key.

`/v1/inference` takes `sequence` and three optional fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sequence` | string | required | Protein sequence in one-letter amino-acid codes. Whitespace (including inside the sequence) is removed and lowercase is converted to uppercase. |
| `mode` | string | `embedding` | `embedding` returns the embedding of the sequence; `masked_lm` returns the masked-token loss and perplexity. Any other value is rejected with HTTP 400. |
| `mask_fraction` | number | `0.15` | `masked_lm` only: fraction of residues to mask. Must be greater than 0 and less than 1. Not used in `embedding` mode, but the range is still checked in every mode: a value outside it is rejected with 400. |
| `seed` | integer | `0` | `masked_lm` only: seed that selects which residues are masked. The same seed gives the same loss. Not used in `embedding` mode. |

`/v1/models` and `/health` take no request body.

A sequence can hold at most **2046 residues**: the 2048-token context window minus the start and end tokens. A longer
sequence, or an empty one, is rejected with HTTP 400 and the service does not truncate it.

**Which characters are accepted.** After whitespace removal and upper-casing, every character must be a one-letter
residue code in the tokenizer's vocabulary (the 20 standard amino acids plus the ambiguity and rare codes the
tokenizer lists, including `X` for an unknown residue). Any other character, such as a digit or a symbol (for example
`MKTV1RQE`), is rejected with HTTP 400 and a message that lists the first few unsupported characters; the sequence is
never silently turned into an unknown token.

### 4.4 Example request

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"},
    timeout=60,
)

embedding = response.json()["embedding"]
print(len(embedding))
```

Score a sequence with the masked-token objective:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "mode": "masked_lm", "mask_fraction": 0.15, "seed": 0}'
```

### 4.5 Response

`/v1/inference` in the default `embedding` mode returns the embedding, its length and the served model:

```json
{
  "embedding": [0.0123, -0.0456, "... 960 values in total ..."],
  "embedding_dim": 960,
  "model": "biohub/ESMC-300M"
}
```

The embedding is the mean of the final hidden states over the residue tokens. The start and end tokens are left out.

In `masked_lm` mode it returns the loss and perplexity over the masked residues instead:

```json
{
  "masked_positions": 10,
  "sequence_length": 65,
  "loss": 1.5559,
  "perplexity": 4.7396,
  "model": "biohub/ESMC-300M"
}
```

These `masked_lm` values are an example for the request in section 4.4 (`mask_fraction` 0.15, `seed` 0): other
hardware or library versions can differ. The example values are reproduced within a couple of percent; when you
check your deployment against them, allow a tolerance of 0.03 on the loss and 0.15 on the perplexity.

`/v1/models` lists the served model:

```json
{
  "object": "list",
  "data": [{"id": "biohub/ESMC-300M", "object": "model"}]
}
```

`/health` returns `{"status": "ok"}` with HTTP 200 once the model is loaded. Before that the port is closed (connection
refused), so there is no 503 body to read while the model loads; poll until the connection succeeds. `/health` answers
`{"status": "unavailable"}` with HTTP 503 only when the service is up but its BentoML health check fails.

### 4.6 What to know before you deploy

**The model is a protein language model.** It takes amino-acid sequences and nothing else. The embedding is a
per-sequence summary, not per-residue; use it for similarity search, clustering or as input to a downstream predictor.

**The perplexity depends on the mask.** A masked-token loss from the `masked_lm` mode varies with `mask_fraction` and
`seed`. Compare scores only when both are the same. A single seed gives a noisy estimate of the model's fit to a sequence:
average the loss over several seeds (for example 5 to 10) before comparing sequences.

**The service needs a GPU.** It fails at startup with an error when no GPU is visible (see the startup failure note
below), instead of falling back to CPU.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**The weights live under `AIM_CACHE_PATH`.** The service reads `biohub/ESMC-300M` (about 620 MB) from the Hugging Face
hub cache layout under `AIM_CACHE_PATH` and downloads it there on first start when absent. At startup it logs a
`Weights pre-flight` line (to the container's stderr, so it shows in `docker logs`) with the cache path, whether
weights are already cached and whether `HF_TOKEN` is set. The model is public and needs no `HF_TOKEN`. Mount a
persistent volume at `AIM_CACHE_PATH` to avoid downloading on every start.

**Startup failure.** If no GPU is visible, or the weights cannot be loaded (not cached and the download fails, or
`ESMC_HF_REPO` names a repo that does not exist), the service logs a critical `ESMC-300M startup failed: ...` message
and the BentoML worker exits. The message for a weights failure names `AIM_CACHE_PATH` and notes that `HF_TOKEN` is not
required; the no-GPU message is `No GPU found. ESMC requires a CUDA/ROCm GPU.` and names neither. The container does **not** exit: BentoML's
supervisor restarts the worker about every 10 seconds, so the message repeats, the container stays running and the
service never becomes ready (`/health` is unreachable and can hang until the client times out). Use a readiness probe
(for example `/readyz`) and watch the container logs to detect this; the container exit code and a restart policy will
not.

**One optional variable selects the Hub repo.** `ESMC_HF_REPO` (default `biohub/ESMC-300M`) overrides the repo the
service loads. Only the default is tested; other repos can break the 960-dimension and 2048-token assumptions. The
service does not read `AIM_MODEL_ID`, which the AIM runtime reserves for catalog model selection.

**Use is subject to the Biohub Acceptable Use Policy.** The weights are released under the MIT license, and the model
card prohibits uses banned by the Biohub Acceptable Use Policy.

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=biohub/esmc-300m \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=biohub/esmc-300m \
  -e AIM_PROFILE_ID=bentoml-mi300x-bf16-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=biohub/esmc-300m \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=bf16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-biohub-esmc-300m:2026.24.0-preview \
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

* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment
* The model cache holds downloaded model weights. Restrict write access to it