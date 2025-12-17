<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide provides step-by-step instructions for deploying AMD Inference Microservice (AIM) container for
{% if aim_deployment.is_base %} any supported model {% else %} {{ aim_deployment.model_name }} model {% endif %} in
various environments. Follow these instructions to quickly get started with running an AI model on AMD GPUs.

## Prerequisites

* AMD GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support
{% if aim_deployment.hf_token %}
* Access to model repositories (Hugging Face account with appropriate permissions for gated models)
{% endif %}

## 1. Docker deployment

### 1.1 Running the container

```bash
docker run \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
{% endif %}
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
```

{% if aim_deployment.hf_token %}
Where <YOUR_HUGGINGFACE_TOKEN> is your Hugging Face access token (required for gated models)
{% endif %}
{% if aim_deployment.is_base %}
Where <ANY_SUPPORTED_MODEL> is the model ID of any supported model (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
{% endif %}

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_PRECISION=fp16 \
  -e AIM_GPU_COUNT={{ aim_deployment.gpus }} \
  -e AIM_METRIC=throughput \
  -e AIM_PORT=8080 \
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
{% endif %}
  --device=/dev/kfd --device=/dev/dri \
  -p 8080:8080 \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
```

## 2. Model caching for production

For production environments, pre-download models to a persistent cache:

### 2.1 Download model to cache

```bash
# Create persistent cache directory
mkdir -p /path/to/model-cache

# Download model using the download-to-cache command
docker run --rm \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
{% endif %}
  -v /path/to/model-cache:/workspace/model-cache \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }} \
{% if aim_deployment.is_base %}
  download-to-cache --model-id <ANY_SUPPORTED_MODEL>
{% else %}
  download-to-cache --model-id {{ aim_deployment.model_name }}
{% endif %}
```

### 2.2 Run with pre-cached model

```bash
docker run \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
{% endif %}
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
```

## 3. Kubernetes deployment

### 3.1 Deployment

It is possible to deploy AIM using Kubernetes. In this doc a sample Kubernetes deployment manifest is provided.

{% if aim_deployment.is_base %}

Create secret for Hugging Face token to download models:

```bash
kubectl create secret generic hf-token \
    --from-literal="hf-token=<YOUR_HUGGINGFACE_TOKEN>" \
    -n <YOUR_K8S_NAMESPACE>
```

{% endif %}

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
          image: {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
          imagePullPolicy: Always
          env:
            - name: AIM_PRECISION
              value: "auto"
            - name: AIM_GPU_COUNT
              value: "{{ aim_deployment.gpus }}"
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
            {% if aim_deployment.hf_token %}
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: hf-token
          {% endif %}
          {% if aim_deployment.is_base %}
            - name: AIM_MODEL_ID
              value: <ANY_SUPPORTED_MODEL>
          {% endif %}
          ports:
            - name: http
              containerPort: 8000
          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              amd.com/gpu: "{{ aim_deployment.gpus }}"
            limits:
              memory: "16Gi"
              cpu: "4"
              amd.com/gpu: "{{ aim_deployment.gpus }}"
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
    {% if aim_deployment.is_base %}
    "model": "<ANY_SUPPORTED_MODEL>",
    {% else %}
    "model": "{{ aim_deployment.model_name }}",
    {% endif %}
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
        {% if aim_deployment.is_base %}
        "model": "<ANY_SUPPORTED_MODEL>",
        {% else %}
        "model": "{{ aim_deployment.model_name }}",
        {% endif %}
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
{% if not aim_deployment.is_base %}
aim_id: {{ aim_deployment.model_name }}
model_id: {{ aim_deployment.model_name }}
{% endif %}
metadata:
  engine: vllm
  gpu: MI300X
  precision: fp16
  gpu_count: {{ aim_deployment.gpus }}
  metric: throughput
  manual_selection_only: false
  {% if aim_deployment.is_base %}
  type: general
  {% else %}
  type: unoptimized
  {% endif %}

engine_args:
  gpu-memory-utilization: 0.95
  dtype: float16
  tensor-parallel-size: {{ aim_deployment.gpus }}
  max-num-batched-tokens: 1024
  max-model-len: 2048

env_vars:
  VLLM_DO_NOT_TRACK: "1"
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
EOF

# Run with custom profile
docker run \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
  -v $(pwd)/custom-profiles:/workspace/aim-runtime/profiles/custom/general \
{% else %}
  -v $(pwd)/custom-profiles:/workspace/aim-runtime/profiles/custom \
{% endif %}
  -e AIM_METRIC=throughput \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
```

{% if aim_deployment.is_base %}
### 5.2 Using S3-hosted models

```bash
docker run \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
  -e AIM_MODEL_ID=s3://my-bucket/models/<ANY_SUPPORTED_MODEL> \
  -e AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY> \
  -e AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY> \
  -e AWS_DEFAULT_REGION=<YOUR_BUCKET_REGION> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
```
{% endif %}


## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }} \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }} \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
{% endif %}
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }}
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects based on the provided environment variables.

```bash
docker run \
  -e AIM_GPU_COUNT={{ aim_deployment.gpus }} \
  -e AIM_PRECISION=fp16 \
  -e AIM_GPU_MODEL=MI300X \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
{% if aim_deployment.is_base %}
  -e AIM_MODEL_ID=<ANY_SUPPORTED_MODEL> \
{% endif %}
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }} \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  {{ aim_deployment.organization }}/{{ aim_deployment.image_repository }}:{{ aim_deployment.image_version }} \
  list-profiles
```

## 7. Security considerations

{% if aim_deployment.hf_token %}
* Never include HF_TOKEN in Dockerfiles or commit it to version control
{% endif %}
* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment
