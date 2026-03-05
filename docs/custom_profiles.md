<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Custom Profiles

**AMD Inference Microservice (AIM)** supports custom profile configurations that extend beyond the built-in optimized
and general profiles. Custom profiles enable users to define specialized configurations for unique hardware setups,
model variants not supported by AIM, or specific performance requirements not covered by standard profiles.

## Overview

Four kinds of profiles can be distinguished based on their structure and location (in order of selection precedence,
from highest to lowest):

* Custom model-specific profiles
* Custom general profiles
* Built-in model-specific profiles
* Built-in general profiles

Custom profiles follow the same YAML structure as standard profiles but are placed in the
`/workspace/aim-runtime/profiles/custom/` directory within the container. On the users' side, custom profiles can be
placed in a folder that must be mounted to the container at the path specified above. When AIM starts, it scans the
custom profiles directory first, so custom profiles take precedence over built-in profiles.

**Key Features:**
- **Highest Search Precedence**: Custom profiles are prioritized over built-in profiles
- **Flexible Deployment**: Mount custom profiles via volumes
- **Experimental Safe**: Test new configurations without building new AIM images

Custom profiles are ideal for performance tuning, hardware-specific optimizations, or deploying models that are not yet
supported by AIM but are compatible with supported engines.

## Creating Custom Profiles

A profile can be defined as a YAML file. The file should adhere to the AIM profile schema. Please refer to the
[existing profiles](https://github.com/amd-enterprise-ai/aim-build/tree/main/profiles) for more examples.

## Using Custom Profiles

Assume you have a custom profile YAML for `DeepSeek R1 Distill Qwen 32B` model named `vllm-mi300x-fp16-tp1-latency.yaml` placed in
the folder `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`.

It contains the following:

```yaml
aim_id: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
model_id: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
metadata:
  engine: vllm
  gpu: MI300X
  gpu_count: 1
  manual_selection_only: false
  metric: latency
  precision: fp16
  type: unoptimized
engine_args:
  distributed_executor_backend: mp
  dtype: float16
  gpu-memory-utilization: 0.95
  tensor-parallel-size: 1
env_vars:
  HIP_FORCE_DEV_KERNARG: '1'
  NCCL_MIN_NCHANNELS: '112'
  PYTORCH_TUNABLEOP_ENABLED: '1'
  PYTORCH_TUNABLEOP_TUNING: '0'
  PYTORCH_TUNABLEOP_VERBOSE: '1'
  TORCH_BLAS_PREFER_HIPBLASLT: '1'
  VLLM_DO_NOT_TRACK: '1'
```

A custom general profile can look the same, but it should not contain `aim_id` and `model_id`. Also, `metadata.type`
should be set to `general`. See [Profile Structure](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md#32-profile-structure)
chapter in the development documentation for details on each field.

### Usage with Docker

To use a custom profile with Docker, the directory containing the profile has to be mounted to the container. The target
path is `/workspace/aim-runtime/profiles/custom/`. The mapped directory must have a certain structure (assuming that the
directory to map is in the current working directory and is named `custom-profiles`):

```./custom-profiles/
    ├── org/
    │   └── model/
    │       └── profile.yaml
    └── general/
        └── profile.yaml
```

All profiles, including the custom ones, are validated against the [AIM profile schema](https://github.com/amd-enterprise-ai/aim-build/tree/main/schemas) at runtime.

#### Running base image with custom general profile

`AIM_MODEL_ID` environment variable is set to ensure the correct profile is selected.

```bash
docker run \
  -e AIM_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  -e AIM_GPU_MODEL=MI300X \
  -v $(pwd)/custom-profiles:/workspace/aim-runtime/profiles/custom \
  --device=/dev/kfd --device=/dev/dri \
  amdenterpriseai/aim-base:0.10 list-profiles
```

As a result, a custom general profile will be selected if it is present in the mounted directory. If there is no custom
general profile, AIM will fall back to built-in general profiles and select the best match.

#### Running base image with custom model-specific profile

`AIM_ID` environment variable is set to ensure the correct profile is selected.

```bash
docker run \
  -e AIM_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  -v $(pwd)/custom-profiles:/workspace/aim-runtime/profiles/custom \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-base:0.10
```

As a result, a custom model-specific profile will be selected if it is present in the mounted directory. If there is no
such profile, AIM will fall back to built-in general profiles and select the best match.

### Usage with Kubernetes

To use custom profiles in Kubernetes, you need to create a ConfigMap or volume containing your custom profiles and mount
it to the `/workspace/aim-runtime/profiles/custom/` path in the container.

#### Creating ConfigMap with Custom Profile

First, create a ConfigMap containing your custom profile:

```bash
kubectl create configmap custom-profiles \
  --from-file=profiles/custom/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/vllm-mi300x-fp16-tp1-latency.yaml \
  -n YOUR_K8S_NAMESPACE
```

#### Example Deployment with Custom Profile

Here's an example Kubernetes deployment that uses a custom model-specific profile:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aim-custom-profile-deployment
  labels:
    app: aim-custom-profile
spec:
  progressDeadlineSeconds: 3600
  replicas: 1
  selector:
    matchLabels:
      app: aim-custom-profile
  template:
    metadata:
      labels:
        app: aim-custom-profile
    spec:
      containers:
        - name: aim-custom-profile
          image: "amdenterpriseai/aim-base:0.10"
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
          ports:
            - name: http
              containerPort: 8000
          resources:
            requests:
              memory: "32Gi"
              cpu: "4"
              amd.com/gpu: "1"
            limits:
              memory: "32Gi"
              cpu: "4"
              amd.com/gpu: "1"
          startupProbe:
            httpGet:
              path: /v1/models
              port: http
            periodSeconds: 10
            failureThreshold: 120
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
            - name: custom-profiles
              mountPath: /workspace/aim-runtime/profiles/custom
              readOnly: true
      volumes:
        - name: ephemeral-storage
          emptyDir:
            sizeLimit: 512Gi
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 64Gi
        - name: custom-profiles
          configMap:
            name: custom-profiles
```

Use `AIM_MODEL_ID` environment variable instead of `AIM_ID` if you want to use a custom general profile.

#### Example Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: aim-custom-profile-service
  labels:
    app: aim-custom-profile
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 80
      targetPort: 8000
  selector:
    app: aim-custom-profile
```

#### Deployment and test commands

Deploy pod and service configured on the previous step

```bash
kubectl apply -f . -n YOUR_K8S_NAMESPACE
```

Port forward the service to access it locally

```bash
kubectl port-forward service/aim-custom-profile-service 8000:80 -n YOUR_K8S_NAMESPACE
```

Test the inference endpoint

Make a request to the inference endpoint using `curl`:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

Remove the deployment and service:

```bash
kubectl delete -f . -n YOUR_K8S_NAMESPACE
```

Remove config map:

```bash
kubectl delete configmap custom-profiles -n YOUR_K8S_NAMESPACE
```
