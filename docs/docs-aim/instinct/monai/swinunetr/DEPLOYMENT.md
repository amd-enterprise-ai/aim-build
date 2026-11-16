<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
monai/swinunetr model in different environments. Follow these instructions to run the model on AMD
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
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0
```

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0
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
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0
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
          image: amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "monai/swinunetr"
            - name: AIM_PROFILE_ID
              value: "bentoml-mi300x-fp16-tp1-latency"
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
| `POST` | `/v1/inference` | Segment one CT volume. |
| `POST` | `/v1/models` | Report the served model and its label map. |
| `GET` | `/healthz` | Check whether the service is ready to accept requests. |

`/v1/inference` and `/v1/models` accept `POST` only. `/healthz` accepts `GET`; see section 4.1.

### 4.3 Request body

`/v1/inference` takes three top-level fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | string | the served model | Model identifier. |
| `input` | object | required | Holds the input and output paths. See below. |
| `parameters` | object | empty | Returned unchanged in the response metadata. |

`input` holds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_path` | string | required | Path of the input NIfTI volume, **inside the container**. |
| `output_dir` | string | `$AIM_OUTPUT_PATH` (default `/workspace/output`) | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body. `/healthz` needs no request body.

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask and not the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/ct-data:/workspace/data` for the input and `-v /path/to/output:/workspace/output` for the masks to the
`docker run` command of section 1.1. Without the output mount, the masks stay in the container and are lost when it is
removed. In the Kubernetes deployment of section 3, add both volumes to the existing `volumeMounts` and `volumes`
lists, and create the two claims like the cache claim in `pvc.yaml`:

```yaml
volumeMounts:
  - name: input-data
    mountPath: /workspace/data
  - name: output-data
    mountPath: /workspace/output
volumes:
  - name: input-data
    persistentVolumeClaim:
      claimName: minimal-aim-input-data
  - name: output-data
    persistentVolumeClaim:
      claimName: minimal-aim-output-data
```

The examples below use `/workspace/data` for input and `/workspace/output` for output, which are the defaults.
`AIM_INPUT_PATH` and `AIM_OUTPUT_PATH` change them; a path outside those roots is rejected. The file name stands for a
volume of your own.

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/abdomen_001.nii.gz",
      "output_dir": "/workspace/output"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"input": {"image_path": "/workspace/data/abdomen_001.nii.gz", "output_dir": "/workspace/output"}},
    timeout=1800,
)

print(response.json()["output"]["output_path"])
```

### 4.5 Response

```json
{
  "model": "MONAI/swin_unetr_btcv_segmentation",
  "output": {
    "output_path": "/workspace/output/abdomen_001/abdomen_001_trans.nii.gz"
  },
  "metadata": {
    "device": "cuda:0",
    "used_amp": true,
    "parameters": {},
    "optimizations": {
      "compile": true,
      "compile_mode": "max-autotune",
      "dynamic": false,
      "autocast": true,
      "gpu_transforms": true
    }
  }
}
```

The mask is written to `<output_dir>/<stem>/<stem>_trans.nii.gz`, where `<stem>` is the input filename without its
extensions. Read the file from the volume mounted at `/workspace/output`, as described in section 4.4. A later request
for the same input writes this same path and overwrites the earlier mask. To keep both, give each request its own
`output_dir` under `/workspace/output`.

`/v1/models` answers with the served model and the 14-entry label map, one entry for the background and one for each
of the 13 segmented organs:

```json
{
  "object": "list",
  "data": [
    {
      "id": "MONAI/swin_unetr_btcv_segmentation",
      "object": "model",
      "owned_by": "MONAI",
      "channel_def": {
        "0": "background",
        "1": "spleen",
        "2": "Right Kidney",
        "3": "Left Kidney",
        "4": "Gallbladder",
        "5": "Esophagus",
        "6": "Liver",
        "7": "Stomach",
        "8": "Aorta",
        "9": "IVC",
        "10": "Portal and Splenic Veins",
        "11": "Pancreas",
        "12": "Right adrenal gland",
        "13": "Left adrenal gland"
      }
    }
  ]
}
```

`/healthz` returns HTTP 200 when the service is ready. It does not return the model details or optimization settings.

### 4.6 What to know before you deploy

**The model expects abdominal CT.** CT is Computed Tomography, an X-ray imaging method that gives a 3D volume.
Preprocessing resamples to 1.5 × 1.5 × 2.0 mm, reorients to RAS, and clips intensity to the −175 to 250 Hounsfield
range. Other modalities, such as MRI, give meaningless output.

**The service always listens on port 8000.** The image sets `BENTOML_PORT=8000`, and the service does not read
`AIM_PORT`, so the `AIM_PORT=8080` example in section 1.2 does not work for this AIM: nothing listens on port 8080.
Leave `AIM_PORT` unset and map the host port you want to 8000, for example `-p 8080:8000`. In Kubernetes, keep
`containerPort` and `targetPort` at 8000, as in section 3.

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `SWINUNETR_TIMEOUT_SECONDS` changes it. When compile is on, as in the
`bentoml-mi300x-fp16-tp1-latency` profile (see section 4.7), the network compiles on the first request and not at
startup. That request takes much longer than later ones, so a client timeout sized for later requests can stop it.

**The weights are fetched at startup.** They are not baked into the image. They are downloaded from
`MONAI/swin_unetr_btcv_segmentation` on Hugging Face and kept in `$AIM_CACHE_PATH`. Mount a persistent volume there,
as described in section 2, or they are downloaded again on every start.

**This model does not need a Hugging Face token.**

### 4.7 Optimization environment variables

A profile can set these variables in its `env_vars`. A value that the profile sets overrides the code default below
and any `-e`. The `bentoml-mi300x-fp16-tp1-latency` profile sets `SWINUNETR_COMPILE`, `SWINUNETR_AUTOCAST` and
`SWINUNETR_GPU_TRANSFORMS` to `true`, which increases the startup time and decreases the inference time. The example
response in section 4.5 comes from that profile. To see what another profile sets, run the `dry-run` command of
section 6.3 with `AIM_PROFILE_ID` set, and read its `env_vars`.

| Variable | Code default | Effect |
|---|---|---|
| `SWINUNETR_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `SWINUNETR_COMPILE_MODE` | `max-autotune` | Compile mode used when `SWINUNETR_COMPILE` is on. |
| `SWINUNETR_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `SWINUNETR_AUTOCAST` | `false` | Run inference in mixed precision. |
| `SWINUNETR_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `SWINUNETR_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `SWINUNETR_LOG_LEVEL` | `INFO` | Log level of the service. |

A variable that the profile does not set can be changed with `-e`, as in section 1.2. To change a variable that the
profile sets, write a custom profile with the changed `env_vars`, mount its directory at
`/workspace/aim-runtime/profiles/custom`, and set `AIM_PROFILE_ID` to its filename without the `.yaml` extension.

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0
```

Section 1.1 describes how the runtime uses `AIM_PROFILE_ID`. If the profile uses a different number of accelerators
than the default profile, change the accelerator count under `resources` in the section 3.1 manifest to match.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0 \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0 \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-model-monai-swinunetr:2026.14.0 \
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