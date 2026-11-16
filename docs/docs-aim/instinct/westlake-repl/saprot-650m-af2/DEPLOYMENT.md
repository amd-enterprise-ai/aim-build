<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
westlake-repl/saprot-650m-af2 model in different environments. Follow these instructions to run the model on AMD
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
  -e AIM_ID=westlake-repl/saprot-650m-af2 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview
```

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=westlake-repl/saprot-650m-af2 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview
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
  -e AIM_ID=westlake-repl/saprot-650m-af2 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview
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
          image: amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "westlake-repl/saprot-650m-af2"
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
| `POST` | `/v1/inference` | Return the SaProt 650M AF2 embedding of one structure-aware protein sequence. |
| `GET` | `/health` | Readiness. Forwards to the BentoML `/healthz` check of the same service and returns its status code. |
| `GET` | `/v1/models` | Lists the model the container serves. |
| `GET` | `/healthz` | BentoML built-in health check. The service also keeps BentoML's `/livez`, `/readyz`, `/metrics` and `/docs.json` (OpenAPI schema). |

There is no `/predict` endpoint. Requests to `/predict` are not served; use `/v1/inference`.

`GET /health` returns `{"status": "ok"}` with status `200` when the BentoML health check succeeds. It returns
`{"status": "unavailable"}` with the upstream status code if the check fails, and with `503` if the service cannot reach
its own health check. Use it as a readiness signal, not as a liveness signal.

`GET /v1/models` returns an OpenAI-style list with the single served model. `id` is the AIM name,
`metadata.dtype` is the precision the model runs in, and `metadata.accelerator` is the name PyTorch reports for the first
visible GPU (`null` if it cannot be read):

```json
{
  "object": "list",
  "data": [
    {
      "id": "westlake-repl/saprot-650m-af2",
      "object": "model",
      "metadata": {"dtype": "fp16", "accelerator": "<GPU name reported by PyTorch>"}
    }
  ]
}
```

### 4.3 Request body

The whole request is nested under a top-level `request` object:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sequence` | string | required | A structure-aware sequence in SaProt format. |

SaProt reads a sequence that interleaves one amino-acid letter and one 3Di structural token per residue, for example
`M#A#K#T#`. The `#` character is the placeholder for a residue with no structural information, so a plain
amino-acid sequence is sent as each residue followed by `#`. A sequence of `N` residues is therefore `2N` characters
long.

The model context is 1024 tokens. SaProt has one token per residue (amino-acid letter plus 3Di token), and the start
and end tokens take two positions, so the service accepts at most **1022 residues** (2044 characters). A longer
sequence is rejected with status `400` before the model runs. The service never truncates the input, so a `200` response
always embeds the whole sequence you sent. The service also rejects an empty or whitespace-only `sequence` and any `sequence` that is not strictly paired.
There is no batch form: send one request per sequence.

### 4.4 Example request

The examples assume that the service is reachable at `localhost:8000`; adjust the port to the one you published.

Check readiness and the served model:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the embedding request:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"request": {"sequence": "M#A#K#T#A#Y#I#A#K#Q#"}}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"request": {"sequence": "M#A#K#T#A#Y#I#A#K#Q#"}},
    timeout=60,
)
response.raise_for_status()

result = response.json()
print(result["embedding_dim"], result["embedding"][:5])
```

A sequence over the limit returns `400` with a message that states the limit and the received length, for example for
a sequence of 1023 residues:

```text
'sequence' is too long: 1023 residues received, the maximum is 1022 residues (1024 tokens including start and end tokens). Shorten the sequence; it is not truncated.
```

The `400` body has one of two shapes. Errors raised by the service itself (this message, the empty-sequence
message, the format messages) are a JSON array, `[{"error": "..."}]`. Request-schema failures (missing `request`
wrapper or `sequence`, not valid JSON, `sequence` not a string) are an object,
`{"error": "1 validation error for Input", "detail": [...]}`. Clients must handle both. Residues are counted as half the
number of characters of the stripped `sequence`, rounded up. The service also keeps a defensive check of the
real tokenized length against the 1024-token maximum; the paired-format and residue limits above already prevent it
from triggering.

The `sequence` must be strictly paired: an uppercase amino-acid letter followed by a lowercase 3Di letter or `#`,
so its stripped length is even. The amino-acid letter must be one of `ACDEFGHIKLMNPQRSTVWY` and the 3Di letter one of
`acdefghiklmnpqrstvwy` or `#`, the model's real vocabulary (446 entries: 5 special tokens and 441 other tokens).
The letters `B J O U X Z` and `b j o u x z` are not in the vocabulary and return `400` naming the offending
characters, instead of being silently turned into an unknown token.
This includes `X` (unknown residue) and `U` (selenocysteine), which do occur in real sequences. The real SaProt
tokenizer maps any amino-acid and 3Di pair that is not in the vocabulary, such as `Xd` or `Ud`, to the `<unk>` token
without an error, and the model would then embed that position as unknown. The service does not allow this: it returns
`400`, and it does not replace or drop such residues for you. A plain amino-acid sequence, an odd length,
internal whitespace, uppercase 3Di letters, or tokens that start with `#` (such as `#x` or `##`) also return `400`
with a message naming the offending characters.

### 4.5 Sequences with structure tokens

Embeddings from SaProt are meant to be computed from a sequence with real 3Di tokens. An all-`#` sequence carries no
structural information, so use `#` only for residues that have no structure or a low-confidence one (the upstream
model card uses `#` for AlphaFold2 regions with pLDDT below 70).

An example with real 3Di letters, taken from the upstream model card
(<https://huggingface.co/westlake-repl/SaProt_650M_AF2>, the `seq` variable in its usage example), where `#` marks
low-pLDDT residues:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"request": {"sequence": "M#EvVpQpL#VyQdYaKv"}}'
```

This is 9 residues (18 characters): `M#`, `Ev`, `Vp`, `Qp`, `L#`, `Vy`, `Qd`, `Ya`, `Kv`.

When you have 3Di tokens for a structure, for example from Foldseek, put each residue's 3Di token after its amino-acid
letter in place of `#`. The service takes the sequence as given and does not compute structure tokens itself.
Foldseek prints the 3Di string in uppercase; lowercase it before interleaving (for example `Ma` for residue `M` with
3Di `A`), otherwise the request returns `400`.

### 4.6 Response

A successful response has this shape:

```json
{
  "embedding": [0.0123, -0.0456, "..."],
  "embedding_dim": 1280,
  "model": "westlake-repl/SaProt_650M_AF2"
}
```

| Key | Purpose |
|---|---|
| `embedding` | The last hidden state of the model, averaged over the residue positions (the start and end tokens are not part of the average), as 1280 floats. |
| `embedding_dim` | Length of `embedding`, `1280`. |
| `model` | The Hugging Face model ID the service loaded. |

Only this pooled, sequence-level vector is returned. The service does not return per-residue embeddings, which tasks
such as per-residue or variant-effect analysis need; that is not supported by this AIM.

The model runs in fp16.

### 4.7 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | The request does not match the schema, for example it lacks the `request` wrapper or the `sequence` field, the body is not valid JSON, `sequence` is empty, `sequence` is not strictly paired amino-acid and 3Di characters (odd length, plain amino-acid sequence), `sequence` contains a letter outside the model vocabulary (`B J O U X Z`, `b j o u x z`), or `sequence` has more than 1022 residues (the length message states the limit and the received count). | No. Correct the request. |
| `503` | Returned by `GET /health` when the service cannot reach its own health check. | Retry later. |
| `500` | Inference failed on the server. The response carries a generic message; details are in the container log only. | Not without a change to the request or the deployment. |

The server-side request timeout is 60 seconds.

### 4.8 What to know before you deploy

**The service runs on one accelerator.** The profiles, `bentoml-mi300x-fp16-tp1-throughput` and
`bentoml-mi355x-fp16-tp1-throughput`, each use one accelerator. On a host with several accelerators, restrict the
container to the accelerator it should use. To serve more sequences per second, run more replicas.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**The service accepts connections only once it is ready.** At startup it loads the tokenizer and the model weights
and moves the model to the accelerator. Use `GET /health` (or `GET /readyz`) as the readiness probe.

**The weights are downloaded at the first start.** The checkpoint is fetched from Hugging Face. The service stores it under `AIM_CACHE_PATH` (default `/workspace/model-cache`). Mount a persistent
volume there, as described in section 2, so a restart does not download it again.

**The service needs an accelerator.** If none is visible to the container, the service does not fall back to the CPU.
It logs `RuntimeError: SaProt 650M AF2 requires a ROCm GPU ...` and `Application startup failed`, but the container does
not exit: BentoML retries the service initialisation about every 10 seconds, and `/health` stays unreachable. Use a
readiness probe on `/health` (not process exit) to detect this.

**This model does not need a Hugging Face token.** The weights are public and not gated.

**Benchmark metric.** `aim-runtime benchmark` reports `sequences_per_sec` = 1 / mean latency. Requests are sent
sequentially (concurrency 1), per length bucket, and the latency includes the HTTP round trip and tokenization.
Warm-up requests are excluded. If any timed request fails, the benchmark is reported as failed.

### 4.9 Model environment variables

| Variable | Profile value | Effect |
|---|---|---|
| `AIM_CACHE_PATH` | not set, which means `/workspace/model-cache` | Directory where the weights are cached. |

The service always loads `westlake-repl/SaProt_650M_AF2`. Do not set `AIM_MODEL_ID`: the runtime reserves it for
model selection, and together with `AIM_ID` it stops the container at startup (`Cannot set both AIM_ID and
AIM_MODEL_ID`), so it cannot be used to load other weights.

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=westlake-repl/saprot-650m-af2 \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=westlake-repl/saprot-650m-af2 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=westlake-repl/saprot-650m-af2 \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-westlake-repl-saprot-650m-af2:2026.24.0-preview \
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