<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide provides step-by-step instructions for deploying AMD Inference Microservice (AIM) container for
 mistralai/Mixtral-8x7B-Instruct-v0.1 model  in
various environments. Follow these instructions to quickly get started with running an AI model on AMD GPUs.

## Prerequisites

* AMD GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support

## 1. Docker deployment

### 1.1 Running the container

```bash
docker run \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4
```


### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_PRECISION=fp16 \
  -e AIM_GPU_COUNT=1 \
  -e AIM_METRIC=throughput \
  -e AIM_PORT=8080 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8080:8080 \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4
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
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4 \
  download-to-cache --model-id mistralai/Mixtral-8x7B-Instruct-v0.1
```

### 2.2 Run with pre-cached model

```bash
docker run \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4
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
          image: amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4
          imagePullPolicy: Always
          env:
            - name: AIM_PRECISION
              value: "auto"
            - name: AIM_GPU_COUNT
              value: "1"
            - name: AIM_GPU_MODEL
              value: "auto"
            - name: AIM_ENGINE
              value: "vllm"
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

### 4.1 Using curl

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "Once upon a time,",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 4.2 Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": "Once upon a time,",
        "max_tokens": 50,
        "temperature": 0.7
    }
)

print(response.json())
```

## 5. Advanced deployment scenarios

### 5.1 Using custom profiles

```bash
# Create custom profile directory
mkdir -p custom-profiles

# Add your custom profile YAML
cat > custom-profiles/vllm-custom-profile.yaml << EOF
aim_id: mistralai/Mixtral-8x7B-Instruct-v0.1
model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
metadata:
  engine: vllm
  gpu: MI300X
  precision: fp16
  gpu_count: 1
  metric: throughput
  manual_selection_only: false
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
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4
```



## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4 \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects based on the provided environment variables.

```bash
docker run \
  -e AIM_GPU_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_GPU_MODEL=MI300X \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4 \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1:0.8.4 \
  list-profiles
```

## 7. Security considerations

* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment