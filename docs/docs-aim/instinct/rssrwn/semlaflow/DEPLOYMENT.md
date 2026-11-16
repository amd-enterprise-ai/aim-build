<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
rssrwn/semlaflow model in different environments. Follow these instructions to run the model on AMD
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
  -e AIM_ID=rssrwn/semlaflow \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview
```

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=rssrwn/semlaflow \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview
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
  -e AIM_ID=rssrwn/semlaflow \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview
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
          image: amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "rssrwn/semlaflow"
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
| `POST` | `/v1/inference` | Generate small molecules; the pre-staged dataset split sets the molecule sizes; returns SMILES strings and the 3D structure of each molecule. |
| `GET` | `/health` | Readiness. Forwards to the BentoML `/healthz` check of the same service and returns its status code. |
| `GET` | `/v1/models` | Lists the model the container serves. |
| `GET` | `/healthz` | BentoML built-in health check. The service also keeps BentoML's `/livez`, `/readyz` and `/metrics`, and `GET /docs.json` with the OpenAPI schema. |

There is no `/predict` endpoint. Requests to `/predict` are not served; use `/v1/inference`.

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
      "id": "rssrwn/semlaflow",
      "object": "model",
      "metadata": {"accelerator": "<GPU name reported by PyTorch>"}
    }
  ]
}
```

### 4.3 Request body

The fields are top-level keys of the JSON body. There is no wrapper object.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `data_path` | string | required | Path inside the container to the directory that holds the GEOM-Drugs dataset in the SemlaFlow `Smol` format (`.smol` files; see section 4.7). The directory must contain `<dataset_split>.smol` for the requested split (`train.smol`, `val.smol` or `test.smol`); `train.smol` is needed only when `dataset_split` is `train`, because the service does not read it for other splits. There is no `data.smol` file. It must resolve (symlinks followed) to a directory under `AIM_INPUT_PATH`. |
| `n_molecules` | integer | `100` | Number of molecules to generate. |
| `integration_steps` | integer | `100` | Number of ODE integration steps. |
| `dataset_split` | string | `"test"` | Dataset split that the generation script reads: `"train"`, `"val"` or `"test"`. The split is not a conditioning signal for the model; it is the data from which the generation script takes the molecule sizes to sample. The service reads `<dataset_split>.smol` from `data_path`. The service default is `"test"` and is always passed to the generation script explicitly, so a request that omits the field reads `test.smol`. |
| `seed` | integer | omitted | Optional random seed, from 0 to 2147483647. When omitted, the request draws a fresh random seed. |

**The model is loaded once and kept resident.** At service start the service loads the checkpoint onto the
accelerator and runs one small warm-up generation (a built-in single molecule, so it needs no dataset) before the port
opens, so `/health` answers only once the service is ready. The start-up log shows `SemlaFlow device: ...` and
`SemlaFlow resident model ready on ... after warm-up in ... s`. Requests are then generated in the service process, with
the SemlaFlow code of the `semlaflow.predict` script, using the GEOM-Drugs dataset format. The dataset is not part of the
image: mount it into the container and give its path in `data_path`. A request returns only when generation has
finished, so the request duration grows with `n_molecules` and `integration_steps`. A request no longer pays for Python
imports or a model load. The first request that names a dataset split reads and parses that `.smol` file; the parsed
split is then kept in memory (up to three splits; a file that is replaced on disk is read again).

### 4.4 Example request

The example assumes the dataset directory is mounted at `/workspace/data/inputs` and that the service is reachable at
`localhost:8000`; adjust the port to the one you published.

Check readiness and the served model:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the generation request:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"data_path": "/workspace/data/inputs", "n_molecules": 5, "integration_steps": 20}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"data_path": "/workspace/data/inputs", "n_molecules": 5, "integration_steps": 20},
    timeout=600,
)
response.raise_for_status()

result = response.json()
print(result["n_molecules"], result["elapsed_s"])
```

### 4.5 Response

The response holds the canonical SMILES strings and, for each of them, the 3D structure the model generated, as MolBlock text. The 3D structures are always returned; there is no option to switch them off.

A successful response has this shape:

```json
{
  "smiles": ["...", "..."],
  "molblocks": ["...", "..."],
  "n_molecules": 5,
  "num_requested": 5,
  "num_valid": 5,
  "num_invalid": 0,
  "elapsed_s": 12.345
}
```

| Key | Purpose |
|---|---|
| `smiles` | Canonical SMILES strings (RDKit), one per generated molecule that passed RDKit sanitisation. Molecules that fail sanitisation are left out and counted in `num_invalid` (and in the service log); hydrogens are removed from the string. |
| `molblocks` | The 3D structure of each molecule in `smiles`, as MolBlock text (V2000) with the atoms and 3D coordinates of the molecule as the model generated it. Same order and same length as `smiles`: `molblocks[i]` belongs to `smiles[i]`. Read an entry with RDKit `Chem.MolFromMolBlock`. |
| `n_molecules` | The number of entries in `smiles` (so it can be lower than the requested `n_molecules`); equal to `num_valid`. |
| `num_requested` | The `n_molecules` value of the request. |
| `num_valid` | The number of molecules returned in `smiles`. |
| `num_invalid` | The number of requested molecules that are not in `smiles`: molecules upstream could not build or that failed RDKit sanitisation. `num_valid + num_invalid` equals `num_requested`. |
| `elapsed_s` | Wall-clock seconds the service spent on the request, including any wait for an earlier request, rounded to three decimals. |

### 4.6 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | The request is invalid: the body does not match the schema, `data_path` is not an existing directory under `AIM_INPUT_PATH`, `n_molecules` or `integration_steps` is below 1, or `dataset_split` is not `train`, `val` or `test`. | No. Correct the request. |
| `500` | The generation run failed, for example because `data_path` does not contain a readable `<dataset_split>.smol`, the model raised an error or ran out of accelerator memory, or no generated molecule passes sanitisation. The body is BentoML's generic error text; the details (path, traceback) are only in the service log. | Only after fixing the cause. |
| `503` | The request did not finish within `SEMLAFLOW_TIMEOUT_SECONDS` counted from its arrival, queue time included (`Generation exceeded N seconds`, or `did not start within N seconds (service busy)` when it never started), or a request that timed out earlier is still running (`A previous request that timed out is still running`). Also returned by `GET /health` when the service cannot reach its own health check. | Yes, later; for a timeout, with a smaller `n_molecules` or `integration_steps`. |

Errors are always returned with one of these status codes, never as a `200` response.

**What a timeout does.** A run on the resident model cannot be killed. A request that has not started when the limit
expires is cancelled and never runs. A request that is already running gets `503` at once, but its run continues until
it ends on its own and keeps the accelerator until then; every new request gets `503` ("still running") until that run
ends. Only a restart of the container stops a run that never ends. The BentoML traffic timeout is set to
`SEMLAFLOW_TIMEOUT_SECONDS` plus 60 seconds, so a slow run ends in this `503`, not in a gateway `504`.

### 4.7 What to know before you deploy

**The service needs a checkpoint that you supply.** The checkpoint is not part of the image. At startup the service
looks for `model.ckpt` in `AIM_CACHE_PATH`. If the file is
missing, the service logs a CRITICAL message `SemlaFlow startup failed: SemlaFlow checkpoint not found at <path>. Weights are read from AIM_CACHE_PATH only ...` and ends its worker; the message names the expected path. BentoML restarts the worker about every 10 seconds, so the message repeats and the container keeps running without becoming ready: use a readiness probe on `/health` and watch the log. Mount the directory that holds the
checkpoint at `AIM_CACHE_PATH`, as described in section 2. Place `model.ckpt` in that host directory before the first
launch: a new empty cache never gives a ready service, because this AIM does not download the checkpoint. With Docker,
this is the `-v /path/to/model-cache:/workspace/model-cache` mount of sections 1.1 and 1.2. With Kubernetes, copy
`model.ckpt` into the persistent volume claim `minimal-aim-model-cache` (for example with a temporary pod that mounts
it and `kubectl cp`) before you apply the manifest of section 3.1.

**The service needs a dataset that you supply.** The checkpoint and the processed GEOM-Drugs data come from the official [rssrwn/semla-flow](https://github.com/rssrwn/semla-flow) repository; its README links the pretrained checkpoints and the processed datasets (Google Drive). `Smol` is that repository's internal molecule dataset representation, written by its `preprocess` script from the GEOM-Drugs data into `train.smol`, `val.smol` and `test.smol`. Mount the directory that holds the `.smol` files under
`AIM_INPUT_PATH` (default `/workspace/data`) and pass that directory as `data_path`; it must contain the `<dataset_split>.smol` file for the split you request (`test.smol` by default). Paths outside `AIM_INPUT_PATH`
are rejected with `400`.

**The dataset is needed for inference, not for startup.** The service checks only the checkpoint when it starts, and
checks `data_path` when a request arrives. A container without any `.smol` files can therefore pass `/readyz` and
`/healthz` and still fail the request of section 4.4 with `400` or `500`. Do not treat readiness as confirmation that the
dataset is staged: check that `<dataset_split>.smol` is present under `AIM_INPUT_PATH` yourself. Mount the dataset
directory read-only. With Docker, add this line to the command of section 1.1 or 1.2:

```bash
  -v /path/to/semlaflow-data:/workspace/data/inputs:ro \
```

With Kubernetes, create a persistent volume claim `semlaflow-data` that holds the `.smol` files, and add these entries
to the existing `volumeMounts` and `volumes` lists of the manifest in section 3.1:

```yaml
          volumeMounts:
            - name: semlaflow-data
              mountPath: /workspace/data/inputs
              readOnly: true
      volumes:
        - name: semlaflow-data
          persistentVolumeClaim:
            claimName: semlaflow-data
```

The request in section 4.4 uses `/workspace/data/inputs` as `data_path`.

**The service needs a GPU, and does not fall back to the CPU.** The resident model is placed on `cuda` (ROCm builds of
PyTorch use the name `cuda` for AMD GPUs). If no GPU is visible, or `SEMLAFLOW_DEVICE` is invalid, or the start-up
warm-up fails, the service logs `SemlaFlow startup failed: ...` (for example `No GPU detected (torch.cuda.is_available()
is False). SemlaFlow requires an AMD GPU ...`) and ends its worker, with the same restart behaviour as above. To use
another visible GPU set `SEMLAFLOW_DEVICE=cuda:<index>`; `SEMLAFLOW_DEVICE=cpu` runs on the CPU on purpose (far slower;
a warning is logged).

**The service runs on one accelerator.** Two profiles are shipped, both fp32 on one accelerator:
`bentoml-mi300x-fp32-tp1-throughput` (one MI300X, gfx942) and `bentoml-mi355x-fp32-tp1-throughput` (one MI355X, gfx950,
the same settings with `PYTORCH_ROCM_ARCH` set to `gfx950`); no profile is shipped for other accelerators. To generate
more molecules per second, run more replicas.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**Requests are not batched across calls.** One resident model serves one request at a time, in arrival order, so a
request sent while a generation is running waits for it (up to `SEMLAFLOW_TIMEOUT_SECONDS`, then `503`). A request that
is rejected with `400` is checked before it queues. Large `n_molecules` values need more accelerator memory
(the molecules of one request are generated in batches on the resident model, and the whole request's output is held
before the SMILES are returned); send large jobs as several smaller requests. Unless you send a `seed`, each request samples with a fresh random seed (the upstream fixed seed is overridden), so
repeated or split requests give independent molecules. With a `seed`, the same request on the same image and hardware is
expected to give the same molecules; exact reproducibility on a GPU is not guaranteed.

**This model does not need a Hugging Face token.**

### 4.8 Model environment variables

| Variable | Default | Effect |
|---|---|---|
| `AIM_CACHE_PATH` | `/workspace/model-cache` | Directory that holds `model.ckpt`. |
| `AIM_INPUT_PATH` | `/workspace/data` | Root directory that `data_path` must be under. |
| `SEMLAFLOW_TIMEOUT_SECONDS` | `1800` | Maximum time a `/v1/inference` request may take, counted from its arrival and including queue time; longer requests get `503` (see section 4.6). |
| `SEMLAFLOW_DEVICE` | `cuda` | Torch device of the resident model: `cuda`, `cuda:<index>` or `cpu`. Anything else fails start-up. |

### 4.9 Client timeouts

Choose a client timeout that covers the longest generation you request, and keep it at or above `SEMLAFLOW_TIMEOUT_SECONDS` if you want to see the service's own `503` instead of a client-side timeout. Generation time depends on `n_molecules`,
`integration_steps` and the accelerator, so measure it for your own settings.

### 4.10 Harness checks

The `aim-runtime` benchmark reports `throughput_mol_per_min`, the molecules returned (not requested; the service drops
molecules that fail sanitisation) per minute measured over whole `/v1/inference` calls on the resident model. It is an
end-to-end HTTP figure: it excludes the one-time model load, which happens at start-up, but includes the per-request
dataset sampling, generation and SMILES conversion. The benchmark metrics also report `molecules_requested`,
`molecules_returned` (mean per timed call) and `mean_service_elapsed_s`.

The `evaluate` run mode makes one `/v1/inference` call (`evaluate_n_molecules` molecules, default 500, with
`evaluate_n_steps` steps, default 100) and reports two scores between 0 and 1: `validity` (molecules returned divided by
molecules requested, that is the share that passed RDKit sanitisation) and `uniqueness` (distinct canonical SMILES divided
by molecules returned). Novelty, energy and stability metrics of the upstream evaluation are not computed. The run
succeeds when the response holds well-formed SMILES and both scores are above the optional profile values
`evaluate_min_validity` and `evaluate_min_uniqueness` (default 0, report only).

The harness checks `health`, `smoke_inference`, `throughput_mol_per_min`, `validity` and `uniqueness` call the service
and have scope OFFLINE; when OFFLINE is not requested they are reported as skipped and the service is not contacted.
`input_data_check` is a local file check (RUNTIME).

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=rssrwn/semlaflow \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=rssrwn/semlaflow \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=rssrwn/semlaflow \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp32 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-rssrwn-semlaflow:2026.24.0-preview \
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