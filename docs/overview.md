<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIMs Overview

AIM stands for AMD Inference Microservice. AIMs provide a standardized, portable inference microservices for serving AI
models on AMD Instinct™ GPUs. AIMs use ROCm 7 under the hood.

AIMs are distributed as Docker images, making them easy to deploy and manage in various environments. Serving AI models
in general and LLMs in particular is not a trivial task. AIMs abstract away the complexities involved in configuring
and serving AI models by providing a mechanism to automatically choose optimal runtime parameters based on the user's
input, hardware, and model specifications.

AIM exposes an [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/introduction) for LLMs, making it
easy to integrate with existing applications and services.

## Features

* **Broad model support**
  * Including community models, custom fine-tuned models, and popular foundation models.
* **Intelligent Configuration based on profiles**.
  * Profiles are predefined configurations optimized for specific models and hardware.
  * Profile selection is an automated process of choosing the best profile based on the user's input, hardware, and model.
    * It is possible to bypass automatic selection and specify a particular profile directly using an environment variable.
    * Custom profiles can be created by users to suit their specific needs.
  * All published profiles are validated against the schema, tested on the target hardware, and optimized for throughput or latency.
* **Models downloading and caching**
  * Models can be downloaded from various sources, including HuggingFace and S3.
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
| Docker  | A platform for developing, shipping, and running applications in containers                         |
| GPU     | A graphics processing unit. Essential hadrware for running AI models                                |
| HF      | Hugging Face, a popular platform for sharing machine learning models and datasets                   |
| LLM     | Large Language Model                                                                                |
| Profile | A predefined AIM run configuration that can be optimized for specific models, compute, or use cases |
| ROCm    | Radeon Open Compute, AMD's open software platform for GPU computing                                 |
| S3      | Amazon Simple Storage Service, a scalable object storage service                                    |
| YAML    | A human-readable data serialization format often used for configuration files                       |
