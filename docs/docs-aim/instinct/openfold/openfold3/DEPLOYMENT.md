<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
openfold/openfold3 model in different environments. Follow these instructions to run the model on AMD
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
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0
```

`AIM_ACCELERATOR_COUNT` must agree with the `accelerator_count` of the selected profile. Set an integer, not `auto`,
because some AIMs read the value directly.

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0
```

## 2. Model caching for production

This AIM has no separate model download step. If the model downloads its weights, it does so at service
startup and keeps them in `$AIM_CACHE_PATH`, which defaults to `/workspace/model-cache`.

Mount a persistent volume at that path, as shown in section 1.1. Without the mount, the container downloads the
weights again at each start.

If the cache must be at a different path, set `AIM_CACHE_PATH` and mount the volume at the same path:

```bash
docker run \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0
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
          image: amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "openfold/openfold3"
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
| `POST` | `/predict` | Predict the structure of one or more biomolecular queries. |

### 4.3 Request body

The whole request is nested under a top-level `data` object:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `queries` | object | required | Map of query name to `{"chains": [...]}`. |
| `seeds` | list of integers | `[42]` | Explicit random seeds. One prediction runs per seed. |
| `num_diffusion_samples` | integer | `1` | Diffusion samples per seed. |
| `num_model_seeds` | integer | not set | Generate this many seeds instead of listing them. Cannot be combined with `seeds`. |
| `use_msa_server` | boolean | `true` | Build the MSA with the ColabFold MSA server. |
| `use_templates` | boolean | `true` | Fetch PDB structure templates. Needs `use_msa_server: true`. |
| `output_format` | string | `"mmcif"` | `"mmcif"` or `"pdb"`. The service does not validate this field: only `"pdb"` gives PDB, and every other value gives mmCIF. |
| `include_atom_confidences` | boolean | `false` | Add per-atom confidences to the response. |

Each entry of `chains` describes one molecule:

| Field | Type | Purpose |
|---|---|---|
| `molecule_type` | string | `protein`, `dna`, `rna` or `ligand`. |
| `chain_ids` | list of strings | Chain identifiers, for example `["A"]`. |
| `sequence` | string | Amino-acid or nucleotide sequence for `protein`, `dna`, and `rna`. |
| `smiles` | string | SMILES string for a `ligand`. |
| `ccd_codes` | string or list of strings | One Chemical Component Dictionary code for a `ligand`, for example `"ATP"`. |
| `main_msa` | string or list of strings | Optional precomputed MSA as a3m text. |
| `paired_msa` | string or list of strings | Optional precomputed paired MSA as a3m text. |

A `ligand` chain uses `smiles` or `ccd_codes` instead of `sequence`. A list given as `ccd_codes` must hold a single code. If a chain sets both `smiles` and `ccd_codes`, `smiles` is used.

The service rejects these requests with HTTP 400, before any computation starts:

* `queries` set to `{}`. There is nothing to fold.
* `main_msa` or `paired_msa` together with `use_msa_server: true`. The server MSA would overwrite the inline one.
* `seeds` together with `num_model_seeds`. One generates seeds, the other lists them.
* `use_templates: true` together with `use_msa_server: false`. Templates are fetched through the same server.

Because `use_templates` defaults to `true`, a request that sets `use_msa_server: false` must also set
`use_templates: false`.

### 4.4 Example request

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "queries": {
        "ubiquitin": {
          "chains": [
            {
              "molecule_type": "protein",
              "chain_ids": ["A"],
              "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            }
          ]
        }
      },
      "seeds": [42],
      "num_diffusion_samples": 1,
      "output_format": "mmcif"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "data": {
            "queries": {
                "ubiquitin": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                        }
                    ]
                }
            },
            "seeds": [42],
            "num_diffusion_samples": 1,
            "output_format": "mmcif",
        }
    },
    timeout=600,
)

print(response.json()["structures"][0]["content"][:200])
```

### 4.5 Request with a precomputed MSA

To avoid the MSA server, give the alignment in the request and switch off both the server and the templates:

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "queries": {
        "ubiquitin": {
          "chains": [
            {
              "molecule_type": "protein",
              "chain_ids": ["A"],
              "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
              "main_msa": ">query\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n"
            }
          ]
        }
      },
      "use_msa_server": false,
      "use_templates": false
    }
  }'
```

### 4.6 Request with a ligand

A ligand chain carries `smiles` or `ccd_codes` instead of `sequence`. This folds ubiquitin together with toluene. The protein chain still uses the MSA server:

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "queries": {
        "ubiquitin": {
          "chains": [
            {
              "molecule_type": "protein",
              "chain_ids": ["A"],
              "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            },
            {
              "molecule_type": "ligand",
              "chain_ids": ["L"],
              "smiles": "Cc1ccccc1"
            }
          ]
        }
      }
    }
  }'
```

To name the ligand with a Chemical Component Dictionary code instead, omit `smiles` and set `"ccd_codes": "ATP"` on that chain. `"ccd_codes": ["ATP"]` is the same request.

### 4.7 Response

A successful response has this shape:

```json
{
  "error": false,
  "structures": [
    {
      "record_id": "ubiquitin_seed_42_sample_1",
      "format": "mmcif",
      "content": "data_ubiquitin\n#\n_entry.id ubiquitin\n..."
    }
  ],
  "confidence": {
    "ubiquitin_seed_42_sample_1": { "...": "..." }
  },
  "timing": {
    "ubiquitin_seed_42_sample_1": { "runtime_s": 42.0 }
  }
}
```

| Key | Purpose |
|---|---|
| `error` | `false` on success, `true` on a prediction failure. |
| `structures` | One entry per sample. `content` holds the mmCIF or PDB text. |
| `confidence` | OF3 confidence metrics, keyed by sample identifier. |
| `timing` | `{"runtime_s": <float>}` per sample, keyed by sample identifier. Always present, `{}` when OF3 emits no timing. |
| `atom_confidence` | Present only with `include_atom_confidences: true`. Per-atom `plddt` and per-token `pae` / `pde` matrices, keyed by sample identifier. The matrix size increases as the square of the token count. |

When a prediction fails, the service answers HTTP 200 with `"error": true`, a `message` field, and empty `structures` and
`confidence`. `timing` is then `{}`, because the service records a runtime only for a sample that produced a structure.

### 4.8 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` with `"error": false` | Success. | — |
| `200` with `"error": true` | The prediction failed. `message` holds the reason. | No. The same request fails again. |
| `400` | The request is not valid. See section 4.3. | No. Correct the request. |
| `429` | Too many requests in flight. | Yes. |
| `503` | The MSA server did not answer within its budget, or the request queued too long. The `Retry-After` header gives the interval. | Yes. |

### 4.9 What to know before you deploy

**The `tpN` in a profile name is the number of data-parallel workers, not tensor parallelism.** OpenFold3 has no
tensor parallelism. A profile named `bentoml-mi300x-fp16-tp4-latency` runs four workers, each pinned to one
accelerator, and each holding its own copy of the model. The deployment then serves up to four predictions at the
same time. It does not split one prediction across four accelerators, and one prediction is not faster with a larger
profile.

Two costs grow with the worker count: host memory, which is roughly N times the single-accelerator footprint, and
cold start, because each worker compiles on its own first request per sequence length.

**`AIM_ACCELERATOR_COUNT` must be an integer that matches the profile.** The service reads the value to size its
worker pool and rejects `auto`.

**The default MSA path calls a public third-party server.** With `use_msa_server: true`, which is the default, the
service calls the public ColabFold server at `api.colabfold.com`. That server limits the rate per caller address, and
every worker in the deployment calls it from the same address, so a larger deployment is throttled sooner. Throttling
gives HTTP 503. For sustained use, set the service to use a self-hosted ColabFold or MMseqs2 server, or give
precomputed MSAs in the request as shown in section 4.5.

**The first request for each sequence length is slow.** The service compiles kernels for each distinct sequence
length, which takes tens of seconds. Later requests with the same length run at the compiled speed.

**The checkpoint is about 2.2 GB.** It is downloaded at the first start and kept in `$AIM_CACHE_PATH`. Mount a
persistent volume there, as described in section 2.

**This model does not need a Hugging Face token.**

### 4.10 Request budget environment variables

| Variable | Default | Effect |
|---|---|---|
| `OPENFOLD3_REQUEST_TIMEOUT_SECONDS` | `600` | Whole-request cap. |
| `OPENFOLD3_MSA_TIMEOUT_SECONDS` | `300` | Cap on MSA-server work. |
| `OPENFOLD3_INFERENCE_RESERVE_SECONDS` | `120` | Budget held back from the MSA stage for accelerator work. |
| `OPENFOLD3_MAX_CONCURRENCY` | `0` | In-flight requests before the service answers 429. The value `0` means "derive from the accelerator count", which gives 3 per accelerator. |

Set a client timeout no shorter than `OPENFOLD3_REQUEST_TIMEOUT_SECONDS`, or the client gives up before the service
answers.

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -e AIM_ACCELERATOR_COUNT=<ACCELERATOR_COUNT_OF_THAT_PROFILE> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0
```

`AIM_ACCELERATOR_COUNT` must match the `accelerator_count` of the selected profile. If the two disagree, the
container stops at startup with a profile selection error.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0 \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=openfold/openfold3 \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0 \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-target-openfold-openfold3-model-openfold-openfold3:0.13.0 \
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