<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
molecularai/reinvent4 model in different environments. Follow these instructions to run the model on AMD
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
  -e AIM_ID=molecularai/reinvent4 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview
```

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=molecularai/reinvent4 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview
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
  -e AIM_ID=molecularai/reinvent4 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview
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
          image: amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "molecularai/reinvent4"
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
| `POST` | `/v1/inference` | Sample novel molecules, as SMILES strings, from a REINVENT4 prior model (the default prior unless you select another, see 4.7). |
| `GET` | `/health` | Readiness. Forwards to the BentoML `/healthz` check of the same service and returns its status code. |
| `GET` | `/v1/models` | Lists the model the container serves, the default prior and the selectable priors. |
| `GET` | `/healthz` | BentoML built-in health check. The service also keeps BentoML's `/livez`, `/readyz` and `/metrics`. |

The service also answers `GET /docs.json` with the OpenAPI schema.

There is no `/predict` endpoint. It has been removed, with no alias; requests to `/predict` are not served. Use
`/v1/inference`.

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
      "id": "molecularai/reinvent4",
      "object": "model",
      "metadata": {
        "accelerator": "<GPU name reported by PyTorch>",
        "default_prior": "reinvent_pubchem",
        "priors": [
          {"name": "reinvent_pubchem", "source": "built-in", "checksum": "md5:75052d5e93b2470e9c6684d369704f5f", "present": true}
        ]
      }
    }
  ]
}
```

`metadata.priors` lists every prior a request can select: its `name`, its `source` (`built-in` or `staged`), the pinned
`checksum` and whether the file is `present` in the cache now. A built-in prior that is not yet present is downloaded on
its first use (see 4.7).

### 4.3 Request body

The whole request is nested under a top-level `data` object (required, but it may be empty: `{"data": {}}`). Every field inside it is optional:

| Field | Type | Default | Allowed values | Purpose |
|---|---|---|---|---|
| `num_smiles` | integer | `100` | 1 to 10000 | Number of SMILES to sample. A single REINVENT run typically samples a few thousand molecules in total, in batches of tens of molecules per step, so 10000 per request is a generous cap. The default of 100 suits interactive use; for bulk sampling choose a multiple of `REINVENT4_SAMPLE_BATCH_SIZE` (see the settings table in 4.8), for example `1024` with the shipped profiles. |
| `unique_molecules` | boolean | `true` | | Keep only valid, non-repeated molecules (see "What unique means" in 4.5). `false` returns every sampled string as it is, invalid and repeated ones included. |
| `prior` | string | the service default (`reinvent_pubchem` unless `REINVENT4_PRIOR` is set) | a `name` listed in `GET /v1/models` | Prior model to sample from. A name only, never a file path. |

The service generates molecules de novo from the prior model: it samples from the prior's learned distribution of molecules. It does not take a scaffold, a seed molecule or a scoring
function, and it does no reinforcement learning or transfer learning. REINVENT4's LibInvent (scaffold decoration), LinkInvent (linker design between two fragments) and PepInvent (peptides) run modes are not covered by this AIM. A request with a value outside the allowed range, or of the wrong type, is rejected with HTTP 400 before the
model runs. A `prior` that is not in the list is rejected with HTTP 400 and a message listing the allowed names. Fields the service does not know are ignored (they are not rejected and have no effect).

**Removed request fields.** `temperature` and `randomize_smiles` were accepted by earlier builds of this AIM but had no effect on the result: upstream REINVENT4 (v4.8) applies `temperature` only to transformer models and `randomize_smiles` only to samplers that take SMILES as input, and this AIM serves only the de novo Reinvent-type priors. They have been removed. A client that still sends either of them is not rejected: the request is answered as if the field were absent (HTTP `200`, no `422`, no warning), so existing clients keep working. Remove them from your requests; they are no longer part of the schema.

### 4.4 Example request

Check readiness and the served model (the examples assume the service is reachable at `localhost:8000`; adjust the port
to the one you published):

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the sampling request:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"data": {"num_smiles": 10}}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"data": {"num_smiles": 10}},
    timeout=600,
)
response.raise_for_status()

result = response.json()
print(result["num_generated"])
for smiles in result["smiles"]:
    print(smiles)
```

### 4.5 Response

A successful response has this shape:

```json
{
  "smiles": ["CCOc1ccc(...)cc1", "..."],
  "num_generated": 10,
  "num_requested": 10,
  "num_invalid": 0,
  "num_duplicates": 0
}
```

| Key | Purpose |
|---|---|
| `smiles` | The generated SMILES strings, one per molecule. |
| `num_generated` | The length of `smiles`. |
| `num_requested` | The `num_smiles` of the request. |
| `num_invalid` | How many of the sampled strings RDKit could not parse or sanitise (the validity check). |
| `num_duplicates` | How many valid sampled strings repeated an earlier valid one (same canonical SMILES, see below). |

The three counts describe the whole sampled batch of `num_requested` strings, whatever `unique_molecules` is. With
`unique_molecules: true`, `num_generated` is `num_requested - num_invalid - num_duplicates` (strings that are empty
after trimming are not counted in `num_generated`). With `unique_molecules: false` the invalid and repeated strings stay in
`smiles`, so `num_generated` can reach `num_requested` and the counts tell you how many of them are not usable.

A run that yields no SMILES at all is reported as HTTP 500, not as an empty `200` response.

**What unique means.** The service uses REINVENT4's own validation step. Each sampled string is parsed and sanitised
with RDKit and written as a canonical, non-isomeric SMILES (stereochemistry is not part of the sampled strings, which the
upstream Reinvent sampler writes without it). A string that fails is invalid. A valid string is a duplicate when its
canonical SMILES equals that of an earlier valid string in the same request, so two different writings of the same
structure count as one molecule. Uniqueness is within one request only.

This is not an InChIKey comparison. The two agree on writing differences (atom order, aromatic or Kekule form, charge-separated
forms such as a nitro group). They differ for tautomers: for example `O=c1cccc[nH]1` and `Oc1ccccn1` (2-pyridone and
2-hydroxypyridine) have the same standard InChIKey but different canonical SMILES, so this service returns both, as two
molecules. Deduplicate the output on InChIKey yourself (RDKit `rdkit.Chem.inchi.MolToInchiKey`) if you want tautomers merged.

`num_generated` can be smaller than the requested `num_smiles`: with `unique_molecules` `true`, invalid and repeated strings
are removed. Sampling is stochastic: two identical requests return different molecules.

### 4.6 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `404` | The path is not served, for example the removed `/predict`. | No. Use `/v1/inference`. |
| `400` | The request does not match the schema in section 4.3, or `prior` names an unknown prior. The body has two shapes: a schema error is an object, `{"error": "1 validation error for Input", "detail": [...]}`; an unknown or invalid prior name is an array, `[{"error": "Unknown prior 'nope'. Allowed priors: reinvent_pubchem"}]`. | No. Correct the request. |
| `503` | The requested built-in prior is missing and could not be downloaded (the path and URL are in the server log only), or the request did not finish within 600 seconds counted from its arrival, queue time included (`Sampling exceeded 600 seconds`, or `did not start within 600 seconds (service busy)` when it never started), or a request that timed out earlier is still running (`A previous request that timed out is still running`). | Yes, later; for a timeout, with a smaller `num_smiles`. |
| `500` | A prior file failed its integrity check (built-in size/MD5 or staged sha256; checked before every use), or the sampling run failed or produced no SMILES (for example the model raised an error or ran out of accelerator memory, or the prior is not a Reinvent-type model). The response body for 500 and 503 is BentoML's generic text, `{"error": "An unexpected error has occurred, please check the server log."}`; a client cannot tell an integrity failure from a sampling failure or a 503 by the body. The detailed message (path, URL, hashes, the sampling error and its traceback) is only in the server log and is never returned. | Only after you have fixed the cause (for an integrity failure, restore or re-stage the file). |

### 4.7 What to know before you deploy

**The service runs on one accelerator.** The profiles, one for MI300X and one for MI355X, each use one accelerator. On
a host with several accelerators, restrict the container to the accelerator it should use. To serve more molecules per
second, run more replicas. The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the
throughput profiles.

**The model is loaded once and kept resident.** At service start the default prior is loaded onto the accelerator
and one warm-up sampling call is made (128 molecules) before the port opens, so `/health` answers only once the service
is ready and no request pays the process start, the library imports, the accelerator initialisation or the prior load.
The start-up log shows `prior reinvent.prior loaded on cuda:0 in ... s` and `REINVENT4 resident model ready on cuda:0 after
warm-up in ... s`. Requests are then sampled in the service process, one at a time in arrival order, with the same
upstream steps as the `reinvent` sampling mode (`create_adapter` once, then `setup_sampler`, `sampler.sample` and, for
`unique_molecules`, upstream's `filter_valid` per request). A staged prior is loaded on its first use (or after its file
changed) and then also stays resident.

**SMILES validation runs in parallel worker processes.** After the model has sampled, every SMILES string is parsed,
sanitised and canonicalised with RDKit (upstream's `validate_smiles`), which is single threaded. The service therefore starts a pool of worker processes (the `spawn` start method) that
validate the sampled strings in parallel. The pool is created and warmed up at start-up, before the port opens, and the
start-up log shows `validation pool ready: <n> workers`. The result is the same as the in-process path: the same
SMILES, in the same order, with the same duplicate marking, so a fixed seed gives identical output with and without the
pool. The workers load only RDKit and NumPy: no PyTorch, no accelerator context and no accelerator memory (a small amount of
host memory per worker). The number of workers is `min(8, usable CPUs)`, where the usable CPUs are the
ones the container may run on (the CPU affinity, for example `docker run --cpuset-cpus`), and can be set with
`REINVENT4_VALIDATION_WORKERS` (see 4.8): `0` or `1` turns the pool off and validates in the service process. A CPU
quota (a Kubernetes `limits.cpu` or `docker run --cpus`) is not visible as affinity, so on a quota-limited container set
`REINVENT4_VALIDATION_WORKERS` to the quota. The pool is stopped, and its processes reaped, when the service shuts down.
If the pool cannot start, the service logs the reason and validates in-process; if a worker dies or hangs during a
request, that request gets the generic HTTP 500 described in 4.6, the pool is shut down, and every later request is
validated in-process (the service keeps working, more slowly, until it is restarted). The 600 second limit of the
paragraph below applies to the whole request, validation included.

The workers take the SMILES validation off the sampling path. With the model on the CPU they help little, because the
CPU is then busy sampling.

**Timeouts.** A request waits for its result for at most 600 seconds in total, queue time included. A thread cannot be
killed the way the former per-request process was: a request that has not started when the 600 seconds run out is
cancelled and never runs (HTTP 503); a request that is already running gets HTTP 503 at once but its run continues
until it ends on its own and keeps the accelerator until then, and every new request gets HTTP 503 ("still running")
in the meantime. Only a container restart stops a run that never ends. BentoML's own request timeout is raised to 660
seconds so the 503 above is the answer for a slow run. Set a client timeout of at least 600 seconds and keep large
`num_smiles` values in mind when you choose it.

**Sampling runs on the GPU.** The resident model is placed on `cuda:0` (ROCm
builds of PyTorch use the name `cuda` for AMD GPUs), and the service logs `REINVENT4 sampling device: cuda:0` at startup. REINVENT4
itself would sample on the CPU when no device is given, even on a GPU node. To use another visible GPU, set `REINVENT4_DEVICE=cuda:<index>`;
`REINVENT4_DEVICE=cpu` runs on the CPU on purpose (slow; a warning is logged). Any other value, or a `cuda` index
that is not visible, fails startup like a missing accelerator.

**The service fails instead of falling back to the CPU.** If no accelerator is visible, or `REINVENT4_DEVICE` is invalid, or a prior cannot be found,
verified or downloaded (see below), or `REINVENT4_PRIOR` names an unknown prior, or the start-up warm-up fails, the service logs a CRITICAL message
naming the problem (for example `REINVENT4 startup failed: No GPU found. REINVENT4 requires a CUDA/ROCm GPU (set REINVENT4_DEVICE=cpu to run on the CPU on purpose).`, or a
message naming the prior file and the expected and actual hash) and ends its worker process. The container does not
exit: the worker is restarted about every 10 seconds, the same message is logged on every restart, the container stays
running, and `/health` is never reachable. A request sent while the service is in this state may get no reply at all (with
no GPU, a POST received none for 650 seconds when BentoML's request timeout was 600 seconds; it is now 660). Do not rely on the container exit code or on `docker run` to detect this.
Use a readiness probe on `GET /health` (in Kubernetes, a `readinessProbe`), set a client timeout, and read the container
log for the cause.

**Where the prior model lives, and the first-start download.** The prior (model weights) is not part of the image. It
lives at `$AIM_CACHE_PATH/reinvent4/priors/reinvent.prior` (`AIM_CACHE_PATH` defaults to `/workspace/model-cache`).
At startup the service logs that path. If the file is missing it downloads `reinvent_pubchem.prior` (about 23.5 MB,
no token needed) from the pinned Zenodo record 20701824 into a temporary file, checks its size and MD5, and renames
it into place. If the download or the check fails, startup fails as described above, with a message naming the
path and the URL. The progress lines (cache path, source URL, `verified`) are logged at INFO level to the container log. Mount a persistent volume at `AIM_CACHE_PATH` so the download happens once. A downloaded prior is private to the service user (mode 0600, owner root); the service is the only reader that needs it. To copy it out of the volume, read it as root, for example `sudo cp <cache volume>/reinvent4/priors/reinvent.prior my_prior.prior`.

**Origin and licence of the default prior.** The default prior, `reinvent_pubchem.prior`, comes from the public REINVENT4
prior models that the upstream REINVENT4 README links on Zenodo: the record "REINVENT4 priors" ("Prior models (v2) for
REINVENT4", by Hannes Löffler, published 2026-06-15; version DOI 10.5281/zenodo.20701824, concept DOI
10.5281/zenodo.15641296, <https://zenodo.org/records/20701824>). The record is open access and states the licence
`Apache-2.0` in its metadata. This image does not contain the file and does not redistribute it: each deployment
downloads it from Zenodo into its own cache. If you redistribute the file yourself, keep the Apache-2.0 notice. The
record page does not describe the training data; the only statement about it is the metadata stored inside the file (see "What the default prior is for" below). Check the record page for its current terms before relying on it.

**If you do not provide a prior.** With no `REINVENT4_PRIOR`, no `prior` request field and nothing staged, the service uses
the default `reinvent_pubchem`: it loads it from `$AIM_CACHE_PATH/reinvent4/priors/reinvent.prior` and, if that file is
missing, downloads it from the Zenodo record above at the first start.

**Pre-staging and air-gapped use.** To avoid the first-start download, or when the container has no internet access,
place the file `reinvent_pubchem.prior` (MD5 `75052d5e93b2470e9c6684d369704f5f`) from
`https://zenodo.org/records/20701824/files/reinvent_pubchem.prior?download=1` at
`<cache volume>/reinvent4/priors/reinvent.prior` before starting the container. A file already at that path is verified
(size and MD5) before every use, and a mismatch is refused with HTTP 500. To avoid hashing 23 MB on every request, a passing result is cached until the file's size, modification time, inode or change time differs; the change time cannot be set from outside, so a rewrite of the file is detected on the next request. The cache lives in the server process, so a restart always re-verifies.

**Selecting the prior.** The default is `reinvent_pubchem`. Two ways to change it: set the service-level default with the
environment variable `REINVENT4_PRIOR=<name>` (for example through a profile `env_vars` entry or `docker run -e`), and/or
send the optional request field `prior` (a name; it overrides the default for that request). A name that is not in the
registry gives HTTP 400 listing the allowed names; an unknown `REINVENT4_PRIOR` fails startup (logged, worker restarted, never ready, see above). The selectable priors
are those in `metadata.priors` of `GET /v1/models`. Names are strict identifiers (letters, digits, `_`, `-`, at most 64
characters, starting with a letter or digit); no endpoint or field takes a file path.

**What the default prior is for.** `reinvent_pubchem` is a de novo prior: it samples drug-like molecules from the
chemical space of its training data. the metadata stored inside the prior file gives the data source as `PubChem 2024-06-03`. The sampled molecules are not guaranteed to be synthesizable, and the
chemistry is limited to what the prior was trained on. The prior does not score or optimise towards a property.

**Built-in priors.** The registry contains only `reinvent_pubchem` (checkpoint `model_type` `Reinvent`). The other files of
Zenodo record 20701824 (`libinvent`, `linkinvent`, `pepinvent`, the `*_transformer_pubchem` files and
`pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc`, which upstream maps to Mol2Mol) are for the Lib-, Link- and
Pep-Invent and Mol2Mol run modes, which need scaffolds, linkers or seed molecules as input; they cannot be used by the
de novo `sampling` run mode this AIM exposes, so they are not offered. Other built-ins are downloaded on first use
(under a lock, to a temporary file, size and MD5 checked, then renamed atomically), so the first request that selects one
pays the download time; a download failure returns HTTP 503 and logs the cache path and URL. Only the default prior is fetched at
startup.

**Bring your own prior (for example a REINVENT4 model you trained or fine-tuned).** Prior files are pickled torch
checkpoints: loading one runs code, so an unverified file is never loaded, and you should only use files you trust.

1. Get a `Reinvent`-type prior for the de novo `sampling` run mode (not a Lib-, Link-, Pep-Invent or Mol2Mol prior).
2. Compute its sha256: `sha256sum my_prior.prior`.
3. Copy it to `<cache volume>/reinvent4/priors/my_prior.prior`, where `<cache volume>` is the volume you mount at
   `AIM_CACHE_PATH` (the name is letters, digits, `_` or `-`, at most 64 characters, and must not be a built-in name or `reinvent`, whose file `reinvent.prior` belongs to `reinvent_pubchem`).
4. Declare it with its hash when you start the container:
   `-e REINVENT4_EXTRA_PRIORS='{"my_prior": "<64 hex digits>"}'`. Add `-e REINVENT4_PRIOR=my_prior` to make it the default
   instead of `reinvent_pubchem`.
5. Check that `GET /v1/models` lists it under `metadata.priors` with `source` `staged` and `present` `true`.
6. Select it per request with `"prior": "my_prior"`, or leave `prior` out if you made it the default.

At startup the service hashes every declared file and, on a missing file or a mismatch, fails startup (logged, worker restarted, never ready) with a message naming the
file and the expected and actual hash. The hash is checked again on every use, so a file replaced after startup is
refused (HTTP 500) until you update the declared hash. A file in the cache that is not declared is ignored. The file must
be a `Reinvent`-type prior; this is not checked, so a wrong model type fails the sampling run (HTTP 500).

Examples:

```bash
curl http://localhost:8000/v1/inference -H "Content-Type: application/json" \
  -d '{"data": {"num_smiles": 10, "prior": "my_prior"}}'
curl http://localhost:8000/v1/inference -H "Content-Type: application/json" \
  -d '{"data": {"prior": "nope"}}'   # HTTP 400, lists the allowed names
```

**Cost per request.** Whichever prior is used, it stays resident after its first load, so only the sampling itself is paid per request. Only Reinvent-type priors can be served; a staged prior of another model type fails with HTTP 500.

**This model does not need a Hugging Face token.**

### 4.8 Model environment variables

The only model-specific environment variable the shipped profiles set is `REINVENT4_SAMPLE_BATCH_SIZE` (`1024`). The
service reads five optional ones. Variables other than `REINVENT4_SAMPLE_BATCH_SIZE` can be added to a profile's
`env_vars` or passed to the container with `docker run -e`. A value a profile sets takes precedence over `docker run -e`
for the same name (the runtime applies the profile's `env_vars` last, and logs `Set env var: <name>=<value>`), so to change
`REINVENT4_SAMPLE_BATCH_SIZE` use a custom profile whose `env_vars` carries a different value (a copy of the shipped
profile YAML), not `-e`:

| Variable | Purpose |
|---|---|
| `REINVENT4_DEVICE` | Torch device for sampling: `cuda:0` (default), `cuda`, `cuda:<index>` or `cpu`. Anything else fails startup. |
| `REINVENT4_VALIDATION_WORKERS` | Number of worker processes that validate the sampled SMILES (0 to 64). Default: `min(8, usable CPUs)`. `0` or `1` disables the pool and validates in the service process. Anything else fails startup. |
| `REINVENT4_SAMPLE_BATCH_SIZE` | Rows sampled per model call when the validation pool is on (1 to 65536). Default: upstream's value, `1000`; the shipped profiles set `1024`. A request is sampled in calls of this size plus one call for the remainder, so a `num_smiles` that is a multiple of it (for bulk sampling, `1024` with the shipped profiles) runs in whole calls; with the default `1000`, a request of 1024 is split into 1000 and 24. Anything else fails startup. |
| `REINVENT4_PRIOR` | Name of the default prior (default `reinvent_pubchem`). |
| `REINVENT4_EXTRA_PRIORS` | JSON object `{"name": "sha256"}` declaring staged priors under `$AIM_CACHE_PATH/reinvent4/priors/<name>.prior`. |

### 4.9 Model licence

REINVENT4 is published by AstraZeneca (MolecularAI) under the Apache 2.0 licence.

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=molecularai/reinvent4 \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=molecularai/reinvent4 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp32-tp1-throughput \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=molecularai/reinvent4 \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp32 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-molecularai-reinvent4:2026.24.0-preview \
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