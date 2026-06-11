<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Deployment Overview

The main use-case for AIM is to provide optimized microservice for large AI models inference on AMD GPUs with simplified
deployment process. AIM can be deployed in multiple ways depending on the use-case. The following options are available:

- Kubernetes deployment with AIM-Engine (recommended)
- Raw Kubernetes Deployment without additional functionality/dependencies
- KServe deployment without AIM-Engine
- Docker (for development and low-scale use-cases)

## AIM-Engine

AIM (AMD Inference Microservice) Engine is a Kubernetes operator that simplifies the deployment and management of AI inference workloads on AMD GPUs. It provides a declarative, cloud-native approach to running ML models at scale. AIM Engine automatically resolves the AIM model, selects an optimal runtime configuration for your hardware, deploys a KServe InferenceService, and optionally creates HTTP routing through Gateway API. In addition to standard KServe deployment, AIM-Engine offers:

- Automatic discovery of AIMs catalog and images
- Automatic selection of optimized images and profiles to match discovered accelerators
- Automatic model lifecycle management of model caches
- Automatic resource requests/allocation by model type
- Simple, single CRD to handle everything from routing to auto-scaling

## Raw Kubernetes deployment

AIMs can be deployed with Kubernetes without additional dependencies. Raw Kubernetes deployment may serve basic requirements where aforementioned features of AIM-Engine are unnecessary. Kubernetes requires specifying an image and parameters that the image may require. AIM
supports many models but their deployment process is fairly similar.

### Configuration

The main configuration is done through the deployment manifest. Key parameters include:

- `amd.com/gpu`: Number of GPUs to allocate
- `memory`: Memory allocation
- `cpu`: CPU allocation
- `image`: The AIM image to use
- `env`: Environment variables for AIM configuration

See [Kubernetes deployment documentation](./kubernetes_deployment.md) for more information.

## KServe

KServe is a Kubernetes-based platform for model serving that provides standardized APIs and advanced features like
autoscaling, canary deployments, and multi-framework support. It simplifies the deployment and management of machine
learning models at scale. AIM can be integrated with KServe to provide optimized inference services for large AI models
on AMD GPUs.

KServe offers several advantages over standard Kubernetes deployments such as:
- Automatic scaling based on traffic
- Built-in monitoring and logging
- Support for multiple model serving protocols

See [KServe deployment documentation](./kserve_deployment.md) for detailed setup instructions.

## Docker

It is possible to run AIM with Docker just as a regular image. Please refer to [Docker documentation](https://docs.docker.com/reference/cli/docker/)
on how to use Docker. Also, see [AIM Docker deployment documentation](./docker_deployment.md) for AIM-specific instructions.
