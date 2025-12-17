<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Deployment Overview

The main use-case for AIM is to provide optimized microservice for large AI models inference on AMD GPUs with simplified
deployment process. AIM can be deployed in multiple ways depending on the use-case. The following options are available:

* Kubernetes (for large-scale deployments)
* KServe (for large-scale deployments tailored for model serving)
* Docker (for development and low-scale use-cases)

## Kubernetes

Kubernetes is an open-source container orchestration system for automating software deployment, scaling, and management.
Think of it as the operating system for your cloud-native applications, managing how and where they run. AIMs can be
deployed with Kubernetes. Kubernetes requires specifying an image and parameters that the image may require. AIM
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

KServe offers several advantages over standard Kubernetes deployments:
- Automatic scaling based on traffic
- Built-in monitoring and logging
- Support for multiple model serving protocols

See [KServe deployment documentation](./kserve_deployment.md) for detailed setup instructions.

## Docker

It is possible to run AIM with Docker just as a regular image. Please refer to [Docker documentation](https://docs.docker.com/reference/cli/docker/)
on how to use Docker. Also, see [AIM Docker deployment documentation](./docker_deployment.md) for AIM-specific instructions.
