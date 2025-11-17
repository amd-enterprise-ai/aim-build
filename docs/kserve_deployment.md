<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# KServe Deployment

This guide provides step-by-step instructions for deploying the AIM inference server on a Kubernetes cluster using KServe. By following these steps, you will be able to deploy AI models using KServe's inference service abstraction, which simplifies model deployment and management on Kubernetes infrastructure. This document covers the prerequisites, deployment process, and how to test the endpoint to ensure everything is working correctly.

## Overview

KServe deployment uses two main components:

1. **ClusterServingRuntime** - Defines the container image, ports, and configuration for the AIM inference engine
2. **InferenceService** - Declares which model to serve, resource requirements, and scaling policies

This approach separates the runtime configuration from the model deployment, allowing you to reuse the same runtime for multiple models and manage them independently.

## Prerequisites

- Kubernetes cluster with kubectl configured (v1.32.8+rke2r1)
- KServe installed on the cluster (v0.15.2)
- AMD GPU with ROCm support (e.g., MI300X)

## Deployment

### 1. Install the Serving Runtime

The ClusterServingRuntime defines the AIM container image and configuration used by the inference service.

Create a file named `servingruntime-aim-qwen3-32b.yaml` with the following contents:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterServingRuntime
metadata:
  name: aim-qwen3-32b-runtime
spec:
  supportedModelFormats:
  - name: aim-qwen3-32b
  containers:
  - name: kserve-container
    image: amdenterpriseai/aim-qwen-qwen3-32b:0.8.4
    imagePullPolicy: Always
    ports:
    - name: http
      containerPort: 8000
      protocol: TCP
    volumeMounts:
    - mountPath: /dev/shm
      name: dshm
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
```

Deploy the serving runtime:

```bash
kubectl apply -f servingruntime-aim-qwen3-32b.yaml
```

Expected output:

```
clusterservingruntime.serving.kserve.io/aim-qwen3-32b-runtime created
```

### 2. Deploy the Inference Service

The InferenceService creates a model serving endpoint using the pre-built Qwen3 32B model.

Create a file named `aim-qwen3-32b.yaml` with the following contents:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: aim-qwen3-32b
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      runtime: aim-qwen3-32b-runtime
      modelFormat:
        name: aim-qwen3-32b
      resources:
        limits:
          memory: "128Gi"
          cpu: "8"
          amd.com/gpu: "1"
        requests:
          memory: "64Gi"
          cpu: "4"
          amd.com/gpu: "1"
```

Deploy the inference service:

```bash
kubectl apply -f aim-qwen3-32b.yaml
```

Expected output:

```
inferenceservice.serving.kserve.io/aim-qwen3-32b created
```

### 3. Port forward the service to access it locally

KServe automatically creates a service with the name `<inferenceservice-name>-predictor` (in this case `aim-qwen3-32b-predictor`) that exposes port 80 by default.

```bash
kubectl port-forward service/aim-qwen3-32b-predictor 8000:80
```

Expected output:

```
Forwarding from 127.0.0.1:8000 -> 8000
Forwarding from [::1]:8000 -> 8000
```

### 4. Test the inference endpoint

Make a request to the inference endpoint using `curl`:

```bash
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

Expected output:

```json
{
    "id": "cmpl-bfb8650209b74010b2a89489b31d8c7c",
    "object": "text_completion",
    "created": 1762361538,
    "model": "Qwen/Qwen3-32B-FP8",
    "choices": [
        {
            "index": 0,
            "text": " city that has long been a beacon",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "token_ids": null,
            "prompt_logprobs": null,
            "prompt_token_ids": null
        }
    ],
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 11,
        "completion_tokens": 7,
        "prompt_tokens_details": null
    },
    "kv_transfer_params": null
}
```

## Cleanup

Remove the deployed resources:

```bash
kubectl delete inferenceservice aim-qwen3-32b
kubectl delete clusterservingruntime aim-qwen3-32b-runtime
```
