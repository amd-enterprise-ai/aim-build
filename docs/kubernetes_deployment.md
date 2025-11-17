<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Pure Kubernetes Deployment

This guide provides step-by-step instructions for deploying the AIM on a Kubernetes cluster. By following these steps,
you will be able to use the provided example to set up and run an AI model on your own Kubernetes infrastructure. This
document covers the prerequisites, deployment process, and how to test the endpoint to ensure everything is working
correctly.

## Prerequisites

- Kubernetes cluster with kubectl configured (v1.32.8+rke2r1)
- AMD GPU with ROCm support (e.g., MI300X)

## Deployment

### 1. Create secret

AIM uses DockerHub container registry to host its images. The images are public and no authentication is required to pull.
However, some models are gated and require authentication to download them from Hugging Face. Therefore, you need to
create a Kubernetes secret.

Create secret for Hugging Face token to download models:

```bash
kubectl create secret generic hf-token \
    --from-literal="hf-token=YOUR_HUGGINGFACE_TOKEN" \
    -n YOUR_K8S_NAMESPACE
```

Expected output:

```
secret/hf-token created
```

### 2. Install AMD device plugin if it is not already in place

Fetch plugin manifest and create the DaemonSet:

```bash
kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
```

Expected output:

```
daemonset.apps/amdgpu-device-plugin-daemonset created
```

### 3. Deploy Kubernetes manifest

#### Example of deployment manifest

Here is an example of `deployment.yaml` for deploying AIM with a specific model. See a corresponding `service.yaml` below.

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
          image: "amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.4"
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
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: hf-token
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
            failureThreshold: 60
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

#### Example of service.yaml

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

#### Deployment

Deploy AIM with specific model using Kubernetes deployment:

```bash
kubectl apply -f . -n YOUR_K8S_NAMESPACE
```

Expected output:

```
deployment.apps/minimal-aim-deployment created
service/minimal-aim-deployment created
```

## Testing

### 1. Port forward the service to access it locally

Do port forwarding

```bash
kubectl port-forward service/minimal-aim-deployment 8000:80 -n YOUR_K8S_NAMESPACE
```

Expected output:

```
Forwarding from 127.0.0.1:8000 -> 8000
Forwarding from [::1]:8000 -> 8000
```

### 2. Test the inference endpoint

Make a request to the inference endpoint using `curl`:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

Expected output:

```json
{
  "id": "cmpl-703ff7b124a944849d64d063720a28f4",
  "object": "text_completion",
  "created":1758657978,
  "model":"meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "text":" city that is known for its v",
      "logprobs": null,
      "finish_reason":"length",
      "stop_reason":null,
      "prompt_logprobs":null,
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 12,
    "completion_tokens": 7,
    "prompt_tokens_details": null,
  },
  "kv_transfer_params": null
}
```

## Removing the deployment

To remove the deployment and service, run:

```bash
kubectl delete -f . -n YOUR_K8S_NAMESPACE
```

Expected output:

```
deployment.apps "minimal-aim-deployment" deleted
service "minimal-aim-deployment" deleted
```
