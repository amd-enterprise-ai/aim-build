<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIMs Overview

AIM stands for AMD Inference Microservice. AIMs provide standardized, portable inference microservices for serving AI
models on AMD Instinct™ GPUs, AMD Radeon™ Pro GPUs, and EPYC™ CPUs. AIMs use ROCm under the hood.

AIMs are distributed as Docker images, making them easy to deploy and manage in various environments. Serving AI models
in general and LLMs in particular is not a trivial task. AIMs abstract away the complexities involved in configuring
and serving AI models by providing a mechanism to automatically choose optimal runtime parameters based on the user's
input, hardware, and model specifications.

AIM exposes an [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/introduction) for LLMs, making it
easy to integrate with existing applications and services.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} AIMs Catalog
:link: /aims/catalog/models
:link-type: doc
Browse the available AIM containers for AMD Instinct™, Radeon™, and EPYC™ hardware.
:::

:::{grid-item-card} Quick Start
:link: /aims/guides/deployment-guide
:link-type: doc
AIM Quick Start Guide: From discovery to first deployment
:::

:::{grid-item-card} Source Code
:link: https://github.com/amd-enterprise-ai/aim-build
:link-type: url
Explore the AIM build system, profiles, and runtime on GitHub.
:::

::::

## Key Features

* **Broad model support**
  * Including community models, custom fine-tuned models, and popular foundation models.
* **Intelligent Configuration based on profiles**.
  * Profiles are predefined configurations optimized for specific models and hardware.
  * Profile selection is an automated process of choosing the best profile based on the user's input, hardware, and model.
    * It is possible to bypass automatic selection and specify a particular profile directly using an environment variable.
    * Custom profiles can be created by users to suit their specific needs.
  * All published profiles are validated, tested on the target hardware, and optimized for throughput or latency.
* **Models downloading and caching**
  * Models can be downloaded from Hugging Face.
  * Downloaded models can be cached in different ways to speed-up subsequent runs.
  * Downloading gated models from Hugging Face is supported.
* **Integration**
  * Logging is available on the container level and can be used by orchestrating frameworks.
  * AIM Runtime CLI simplifies the integration with orchestrating frameworks, such as Kubernetes.
  * AIM exposes OpenAI-compatible API for LLMs.

## Terminology reference

| Word    | Explanation                                                                                         |
|---------|-----------------------------------------------------------------------------------------------------|
| AIM     | AMD Inference Microservice                                                                          |
| CPU     | Central processing unit. AIMs support running on AMD EPYC™ CPUs without a GPU                       |
| Docker  | A platform for developing, shipping, and running applications in containers                         |
| GPU     | A graphics processing unit. Essential hardware for running AI models                                |
| HF      | Hugging Face, a popular platform for sharing machine learning models and datasets                   |
| IPC     | Inter-Process Communication. In Docker, `--ipc=host` shares the host's IPC namespace (including shared memory) with the container |
| LLM     | Large Language Model                                                                                |
| Profile | A predefined AIM run configuration that can be optimized for specific models, compute, or use cases |
| ROCm    | Radeon Open Compute, AMD's open software platform for GPU computing                                 |
| YAML    | A human-readable data serialization format often used for configuration files                       |
