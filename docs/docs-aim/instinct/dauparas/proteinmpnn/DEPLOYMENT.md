<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
dauparas/proteinmpnn model in different environments. Follow these instructions to run the model on AMD
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
  -e AIM_ID=dauparas/proteinmpnn \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview
```

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=dauparas/proteinmpnn \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview
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
  -e AIM_ID=dauparas/proteinmpnn \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview
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
          image: amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "dauparas/proteinmpnn"
            - name: AIM_PROFILE_ID
              value: "bentoml-mi300x-fp32-tp1-throughput"
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
| `POST` | `/v1/inference` | Design amino acid sequences for the backbone in one PDB file. |
| `GET` | `/health` | Readiness. Forwards to the BentoML `/healthz` check of the same service and returns its status code. |
| `GET` | `/v1/models` | Lists the model the container serves. |
| `GET` | `/healthz` | BentoML built-in health check. The service also keeps BentoML's `/livez`, `/readyz` and `/metrics`. |

The service also answers `GET /docs.json` with the OpenAPI schema.

There is no `/predict` endpoint. Requests to `/predict` are not served; use `/v1/inference`.

The service answers no request, `/health` included, until the model is loaded and warmed up (see section 4.7, "Startup
and readiness"); before that the port is closed.

`GET /health` returns `{"status": "ok"}` with status `200` when the BentoML health check succeeds. It returns
`{"status": "unavailable"}` with the upstream status code if the check fails, and with `503` if the service cannot reach
its own health check. Use it as a readiness signal, not as a liveness signal.

`GET /v1/models` returns an OpenAI-style list with the single served model. `metadata.accelerator` is the name PyTorch
reports for the first visible GPU (`null` if it cannot be read):

```json
{
  "object": "list",
  "data": [
    {
      "id": "dauparas/proteinmpnn",
      "object": "model",
      "metadata": {"accelerator": "<GPU name reported by PyTorch>"}
    }
  ]
}
```

### 4.3 Request body

The whole request is nested under a top-level `request` object:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `pdb_path` | string | required | Absolute path to the input PDB file inside the container. It must resolve to an existing file under `AIM_INPUT_PATH`. |
| `num_seq_per_target` | integer | `4` | Number of sequences to design, from 1 to 64. For larger sets, send repeated requests (see below). |
| `sampling_temp` | number | `0.1` | Sampling temperature, greater than 0.0 and at most 1.0. Higher values give more diverse sequences. |
| `model_name` | string | `v_48_020` | ProteinMPNN model variant (the `--model_name` of `protein_mpnn_run.py`). One of `v_48_002`, `v_48_010`, `v_48_020`, `v_48_030`. The number in the name is the standard deviation, in Å, of the Gaussian backbone noise added during the model's training (0.02, 0.10, 0.20, 0.30 Å); it is not noise added to your structure at inference time (the service always uses upstream's `--backbone_noise 0.0` and does not expose it). |
| `seed` | integer | `37` | Random seed (the `--seed` of `protein_mpnn_run.py`). From 0 to 4294967295 (2**32 - 1); `0` lets ProteinMPNN draw a random seed from 0 to 998 on each call, so results usually differ between calls but can repeat. |

**More than 64 sequences.** One request returns at most 64 sequences. For a larger set, send repeated requests for the
same structure and concatenate the results. The seed must differ between the requests: ProteinMPNN seeds its random
generator from `seed` on every call, so the same structure, model, temperature and seed always return the same
sequences, whatever the temperature. Use a different `seed` for each request (or `0`, which draws a random one from only
999 values, so duplicates across many requests are possible). Changing the temperature is not needed to get different
sequences, but it does change how diverse they are. Remove duplicate sequences after merging.

**Limitation: all chains and all positions are designed.** The service designs every chain of the input structure and
every residue position. It has no chain selection and no fixed positions: it cannot keep a chain, a binder interface or
a catalytic site unchanged while redesigning the rest. Upstream ProteinMPNN supports both (`--pdb_path_chains`,
`--chain_id_jsonl`, `--fixed_positions_jsonl`); they are not exposed by this AIM.

The service does not accept the PDB contents in the request. It reads the file from `pdb_path`, so mount the directory
that holds your structures into the container, and give the path as seen inside the container. `AIM_INPUT_PATH`
(default `/workspace/data`) is the root the service reads from, so mount the directory there, for example
`-v /data/pdbs:/workspace/data:ro`. Symbolic links are followed before the check, so a link that points outside the
root is rejected, as is any path with `..` that leaves it.

Values outside the ranges above, an unknown `model_name`, a missing `pdb_path`, a `pdb_path` that is outside
`AIM_INPUT_PATH` or does not name an existing file, and malformed JSON are rejected with HTTP 400 before ProteinMPNN
runs. There is no batch form: send one request per structure.

### 4.4 Example request

The example assumes a structure `1ubq.pdb` in `/data/pdbs` on the host, mounted at `/workspace/data` in the container
(`-v /data/pdbs:/workspace/data:ro`):

Check readiness and the served model (the examples assume the service is reachable at `localhost:8000`; adjust the port
to the one you published):

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the design request:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"request": {"pdb_path": "/workspace/data/1ubq.pdb", "num_seq_per_target": 8, "sampling_temp": 0.1, "seed": 37}}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={
        "request": {
            "pdb_path": "/workspace/data/1ubq.pdb",
            "num_seq_per_target": 8,
            "sampling_temp": 0.1,
            "seed": 37,
        }
    },
    timeout=700,
)
response.raise_for_status()

result = response.json()
for sequence, score in zip(result["sequences"], result["scores"]):
    print(score, sequence)
```

### 4.5 Response

A successful response has this shape:

```json
{
  "sequences": ["MQIFVKTLTGK...", "MQIFVKTLTGK..."],
  "scores": [1.23, 1.19],
  "seq_recovery": [0.41, 0.39],
  "native_score": 1.31
}
```

| Key | Purpose |
|---|---|
| `sequences` | The designed sequences, in the order ProteinMPNN sampled them. The native (input) sequence that ProteinMPNN writes as the first FASTA record is not included. |
| `scores` | The `score` values from the `.npz` score file that ProteinMPNN writes (`--save_score 1`), one per designed sequence. `scores[i]` belongs to `sequences[i]`. Each is the negative log-probability of the designed sequence given the backbone, averaged over the designed positions (every position, since this AIM designs all of them; so it equals upstream's `global_score`). Lower means the model considers the sequence more likely. Compare scores only for the same structure and the same settings (`model_name`, `sampling_temp`); they are not comparable across different structures. If a score file is missing or has a different length, the entries are padded with NaN or truncated to match the sequences. |
| `seq_recovery` | The `seq_recovery` value from the header of each designed FASTA record: the fraction of designed positions where the designed residue equals the native residue. `seq_recovery[i]` belongs to `sequences[i]`. Padded with NaN, like `scores`, when a header has no value. |
| `native_score` | The `score` in the header of the native (input) sequence record, which the service does not return in `sequences`. It is on the same scale as `scores`, so it shows how likely the model finds the native sequence compared with the designs. With several input structures it is the one of the first structure in file-name order. NaN when the header has no value. |

The service runs ProteinMPNN's own `protein_mpnn_run.py` code (`main`) with `batch_size 1` and `save_score 1`, in the
service process. For the same request (including `seed`) the sequences are the ones the script gives on the command line.

### 4.6 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | The request does not match section 4.3: a field is missing or out of range (including a `seed` below 0 or above 4294967295), `model_name` is not one listed, or `pdb_path` is not an existing file under `AIM_INPUT_PATH`. | No. Correct the request. |
| `500` | The design run raised an error, for example because the file is not a valid PDB, or it produced no sequences. The response body carries a generic message; the detail is in the container log. | Only after fixing the cause. |
| `503` | The request did not finish within 600 seconds, counted from its arrival and including the time it waited for earlier requests (`Sequence design exceeded 600 seconds`, or `did not start within 600 seconds (service busy)` when it never started), or a request that timed out earlier is still running (`A previous request that timed out is still running on the GPU`). `GET /health` also returns `503` when the service cannot reach its own health check. | Yes, later, with a smaller `num_seq_per_target` or a shorter structure; for `/health`, retry later. |

Errors are never returned as a `200` response with an error flag.

### 4.7 What to know before you deploy

**The service runs on one accelerator.** The profiles `bentoml-mi300x-fp32-tp1-throughput` and
`bentoml-mi355x-fp32-tp1-throughput` use one accelerator each, in fp32 and without `torch.compile`. To serve more
requests per second, run more replicas.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**Startup and readiness.** The service loads the model in its own process and does not start a process per request. At
startup it loads ProteinMPNN, then runs one warm-up design with each of the four weight files on a generated 64-residue
backbone (a scoring pass and sampling), and only then opens its port. `/health`, `/healthz` and `/readyz` therefore
answer only after the warm-up, and the first request is as fast as the later ones. The warm-up pays the one-time GPU
start-up cost (the first forward pass in a process). Use a readiness probe and wait for it before sending traffic. The warm-up
is logged to the container's stderr (`warm-up <model> done in ...`, `ProteinMPNN resident model ready on <device>`).

**Your input structures must be mounted; readiness does not check them.** The service reads each structure from a file
inside the container, so mount a directory of PDB files under `AIM_INPUT_PATH` (default `/workspace/data`). The example of
section 4.4 uses `1ubq.pdb` from that directory. The service does not look at the input directory when it starts, so
`/readyz` passes without it, and a request then fails with `400`. With Docker, add this line to the command of section
1.1 or 1.2:

```bash
  -v /path/to/pdbs:/workspace/data:ro \
```

With Kubernetes, create a persistent volume claim `proteinmpnn-data` that holds your PDB files, and add these entries to
the existing `volumeMounts` and `volumes` lists of the manifest in section 3.1:

```yaml
          volumeMounts:
            - name: proteinmpnn-data
              mountPath: /workspace/data
              readOnly: true
      volumes:
        - name: proteinmpnn-data
          persistentVolumeClaim:
            claimName: proteinmpnn-data
```

**Requests run one at a time, in arrival order.** There is one model instance and one worker thread for all GPU work, so
concurrent requests queue; they are not run in parallel on the GPU. Do not set more than one BentoML worker for this
service: each worker would load its own copy of the model and warm up separately. To serve more
traffic, run more replicas.

**Memory.** The four weight files are kept in memory, on the accelerator (about 7 MB each as stored; a small amount, tens of MiB, after the warm-up), and PyTorch's caching allocator keeps the
memory it needed for the largest structure served so far. Long structures use more memory per request; the memory is
not given back between requests. A request that runs out of accelerator memory returns `500` and the cache is released.

**Timeouts.** A request waits for its result for at most 600 seconds in total, queueing time included. The model now
runs inside the service process, so a run cannot be killed the way the earlier one-process-per-request design killed
its process:

- A request that has not started when the 600 seconds run out is cancelled and never runs (`503`).
- A request that is already running when the 600 seconds run out gets `503` at once, but its run goes on until it ends
  on its own and keeps the GPU until then. While such a run is active, every new request gets `503`
  (`A previous request that timed out is still running on the GPU`) immediately instead of queueing. When the run ends,
  the service accepts requests again.
- A run that never ends can be stopped only by restarting the container.

The service raises BentoML's request timeout above this limit, so the `503` is the answer for a run that is too long;
BentoML's default 60 second timeout, which would answer `504` earlier, does not apply. Set a client timeout above 600
seconds. Input size is the only lever on run time: it is not limited other than by this timeout and by accelerator
memory.

**Output.** The designed sequences and scores for the same request and seed are identical to those of the former
one-process-per-request service.

**The service needs a GPU and refuses to become ready without one.** If PyTorch sees no accelerator, a weights file
cannot be provided (missing and the download fails, or the size or sha256 does not match), or the warm-up fails, the
service logs a critical error naming the cause (for example `No GPU visible to PyTorch`, or `Could not provide ProteinMPNN weight <path> (source
<URL>)` or `Model warm-up failed on device cuda: ...`) and the BentoML worker exits. The container does **not** exit: BentoML's supervisor restarts the worker about
every 10 seconds, so the message repeats, the container stays running and the service never becomes ready (`/health`
is unreachable). Use a readiness probe (for example `/readyz`) and watch the container logs to detect this; the exit
code and a restart policy will not. It does not fall back to the CPU unless you ask for it (`PROTEINMPNN_DEVICE=cpu`, see section 4.8).

**The weights are not in the image; they live under `AIM_CACHE_PATH`.** The four vanilla weight files
(`v_48_002`, `v_48_010`, `v_48_020`, `v_48_030`, about 27 MB in total, 6681301 bytes each) are stored in
`$AIM_CACHE_PATH/proteinmpnn/vanilla_model_weights/` (default `/workspace/model-cache/proteinmpnn/vanilla_model_weights/`).
At startup the service logs this directory (to the container's stderr, so it shows in `docker logs`) and checks each file (size and sha256). A missing file is downloaded on
first start from
`https://raw.githubusercontent.com/dauparas/ProteinMPNN/905b0081f3aa3404d7ac22174bc25710f9995ff8/vanilla_model_weights/<name>.pt`
(upstream commit of tag `v1.0.1`; the upstream repository is MIT-licensed), verified, and renamed into place
atomically; each download is logged. The first download takes tens of seconds (depending on the network). No Hugging Face token is needed. The first start therefore needs network access to
`raw.githubusercontent.com`, or pre-staged files. Mount a persistent volume at `AIM_CACHE_PATH` so the download
happens once.

**Air-gapped use.** Copy the four files from that upstream commit into
`<cache volume>/proteinmpnn/vanilla_model_weights/` before starting; the service then makes no network request. A file
with the wrong size or sha256 stops startup.

**Weights failure.** If a download fails or a file does not verify, the container logs
`ProteinMPNN startup failed: Could not provide ProteinMPNN weight <path> (source <URL>): <reason>`. As described above,
the worker is restarted about every 10 seconds and the container keeps running without becoming ready.

### 4.8 Model environment variables

| Variable | Default | Effect |
|---|---|---|
| `PROTEINMPNN_PATH` | `/opt/proteinmpnn` | Directory that holds `protein_mpnn_run.py` and `protein_mpnn_utils.py`. |
| `PROTEINMPNN_DEVICE` | `cuda` | Device of the resident model: `cuda` (the accelerator; also `cuda:0`) or `cpu`. Any other value stops startup. A GPU request with no visible GPU stops startup; the service does not fall back to the CPU. To use a particular GPU, set `HIP_VISIBLE_DEVICES` (ROCm) or `CUDA_VISIBLE_DEVICES`, as upstream always uses the first visible GPU. |
| `AIM_CACHE_PATH` | `/workspace/model-cache` | Root of the model cache; weights are under `proteinmpnn/vanilla_model_weights/`. Mount a persistent volume here. |
| `AIM_INPUT_PATH` | `/workspace/data` | Root directory that `pdb_path` must be under. |
| `PROTEINMPNN_FIXTURE_DIR` | `/workspace/model/fixtures` | Second directory that `pdb_path` may be under. The benchmark and evaluate harness writes its small smoke structure here when it runs, so point it at a writable directory when the container root filesystem is read-only. |

`PROTEINMPNN_DEVICE=cpu` runs the model on the CPU and logs a warning. It is not the default and is only worth it for
very small inputs. With `cpu`, `/v1/models` reports
`"accelerator": null`.

The profiles also set `TORCH_BLAS_PREFER_HIPBLASLT=1`. Leave it as it is.

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=dauparas/proteinmpnn \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=dauparas/proteinmpnn \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=dauparas/proteinmpnn \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp32 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-dauparas-proteinmpnn:2026.24.0-preview \
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