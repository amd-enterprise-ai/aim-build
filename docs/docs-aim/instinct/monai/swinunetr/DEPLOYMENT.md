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

Pin the profile explicitly, so that the selection cannot depend on the hardware the runtime detects. Set
`AIM_PROFILE_ID` to one of the profiles listed in the [README](README.md), and set `AIM_ACCELERATOR_COUNT` to the
accelerator count of that profile.

```bash
mkdir -p /path/to/model-cache

docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0
```

`AIM_ACCELERATOR_COUNT` must agree with the `accelerator_count` of the selected profile. Set an integer, not `auto`,
because some AIMs read the value directly.

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8080:8080 \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0
```

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
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0
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
          image: amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "monai/swinunetr"
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
| `output_dir` | string | the bundle `eval` directory | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body. `/healthz` needs no request body.

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask and not the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/ct-data:/data` to the `docker run` command of section 1.1, or mount the same directory in the
Kubernetes deployment of section 3. The examples below use `/data` for that mount.

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/data/abdomen_001.nii.gz",
      "output_dir": "/data/out"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"input": {"image_path": "/data/abdomen_001.nii.gz", "output_dir": "/data/out"}},
    timeout=1800,
)

print(response.json()["output"]["output_path"])
```

### 4.5 Response

```json
{
  "model": "MONAI/swin_unetr_btcv_segmentation",
  "output": {
    "output_path": "/data/out/abdomen_001/abdomen_001_trans.nii.gz"
  },
  "metadata": {
    "device": "cuda:0",
    "used_amp": false,
    "parameters": {},
    "optimizations": {
      "compile": false,
      "compile_mode": null,
      "dynamic": null,
      "autocast": false,
      "gpu_transforms": false
    }
  }
}
```

The mask is written to `<output_dir>/<stem>/<stem>_trans.nii.gz`, where `<stem>` is the input filename without its
extensions. Read the file from the mounted directory.

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

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `SWINUNETR_TIMEOUT_SECONDS` changes it.

**The weights ship in the image.** They are downloaded from Hugging Face at startup only if they are missing from the
bundle directory.

**This model does not need a Hugging Face token.**

### 4.7 Optimization environment variables

These are off by default. Switch them on to increase the startup time and decrease the inference time.

| Variable | Default | Effect |
|---|---|---|
| `SWINUNETR_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `SWINUNETR_COMPILE_MODE` | `max-autotune` | Compile mode used when `SWINUNETR_COMPILE` is on. |
| `SWINUNETR_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `SWINUNETR_AUTOCAST` | `false` | Run inference in mixed precision. |
| `SWINUNETR_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `SWINUNETR_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `SWINUNETR_LOG_LEVEL` | `INFO` | Log level of the service. |

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -e AIM_ACCELERATOR_COUNT=<ACCELERATOR_COUNT_OF_THAT_PROFILE> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0
```

`AIM_ACCELERATOR_COUNT` must match the `accelerator_count` of the selected profile. If the two disagree, the
container stops at startup with a profile selection error.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0 \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_ID=monai/swinunetr \
  -e AIM_PROFILE_ID=bentoml-mi300x-fp16-tp1-latency \
  -e AIM_ACCELERATOR_COUNT=1 \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID=monai/swinunetr \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0 \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-instinct-target-monai-swinunetr-model-monai-swinunetr:0.13.0 \
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