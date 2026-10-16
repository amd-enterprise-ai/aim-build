<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIMs Overview

AIM stands for AMD Inference Microservice. AIMs provide standardized, portable inference microservices for serving AI
models on AMD Instinct™ GPUs, AMD Radeon™ Pro GPUs, and EPYC™ CPUs. AIMs use ROCm under the hood.

AIMs are distributed as Docker images, making them easy to deploy and manage in various environments. AIMs abstract
away the complexities involved in configuring and serving AI models by providing a mechanism to automatically choose
optimal runtime parameters based on the user's input, hardware, and model specifications.

AIM exposes an HTTP API for every model. LLMs use an
[OpenAI-compatible API](https://platform.openai.com/docs/api-reference/introduction), which makes them easy to
integrate with existing applications and services. Models for other use cases, such as biomolecular structure
prediction, medical image segmentation and text-to-video generation, expose task-specific APIs. The deployment guide
of each AIM documents its endpoints and its request and response formats.

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
  * Beyond language models, the catalog covers tasks in life sciences and health care, generative media, and weather
    and earth sciences.
* **Intelligent Configuration based on profiles**.
  * Profiles are predefined configurations optimized for specific models and hardware.
  * Profile selection is an automated process of choosing the best profile based on the user's input, hardware, and model.
    * It is possible to bypass automatic selection and specify a particular profile directly using an environment variable.
    * Custom profiles can be created by users to suit their specific needs.
  * All published profiles are validated, tested on the target hardware, and optimized for throughput or latency.
* **Automatic engine selection**
  * AIM runs language models on vLLM. Models that need a runner of their own use BentoML or vLLM-Omni.
  * The profile gives the engine, and AIM starts it. AIM abstracts the engine away, so the deployment steps are the
    same for all engines.
* **Models downloading and caching**
  * Models can be downloaded from Hugging Face, or from another source that the model publisher provides.
  * Downloaded models can be cached in different ways to speed-up subsequent runs.
  * Downloading gated models from Hugging Face is supported.
* **Integration**
  * Logging is available on the container level and can be used by orchestrating frameworks.
  * AIM Runtime CLI simplifies the integration with orchestrating frameworks, such as Kubernetes.
  * AIM exposes an OpenAI-compatible API for LLMs, and a task-specific HTTP API for every other model family.

## Terminology Reference

| Word     | Explanation                                                                                        |
|----------|----------------------------------------------------------------------------------------------------|
| AIM      | AMD Inference Microservice                                                                          |
| CPU      | Central processing unit. AIMs support running on AMD EPYC™ CPUs without a GPU                       |
| Docker   | A platform for developing, shipping, and running applications in containers                         |
| Engine   | The inference engine that runs the model in the AIM container                                       |
| GPU      | A graphics processing unit. Essential hardware for running AI models                                |
| HF       | Hugging Face, a popular platform for sharing machine learning models and datasets                   |
| IPC      | Inter-Process Communication. In Docker, `--ipc=host` shares the host's IPC namespace (including shared memory) with the container |
| LLM      | Large Language Model                                                                                |
| MONAI    | Medical Open Network for AI — a framework/bundle format for medical-imaging models                  |
| Profile  | A predefined AIM run configuration that can be optimized for specific models, compute, or use cases |
| ROCm     | Radeon Open Compute, AMD's open software platform for GPU computing                                 |
| YAML     | A human-readable data serialization format often used for configuration files                       |


# Model-specific AIM

This AIM allows to deploy Qwen/Qwen3.6-35B-A3B with a tailored set of profiles.

* Model name: Qwen/Qwen3.6-35B-A3B
* Description: Qwen3.6-35B-A3B is a Mixture-of-Experts LLM with 35B total parameters and ~3B active per token, balancing high quality with efficient inference.
* Capabilities:
  * text-generation
  * chat
  * reasoning
  * instruction
  * moe


## Available Profiles

The following profiles are available for this model:

|Profile| Accelerator model |Precision|Engine|Accelerator count|Metric|Type|
|-------|-------------------|---------|------|---------|------|----|
|vllm-r9700-bf16-tp4-latency|R9700|bf16|vllm|4|latency|preview|
|vllm-r9700-bf16-tp4-throughput|R9700|bf16|vllm|4|throughput|preview|
|vllm-w7900-bf16-tp4-latency|W7900|bf16|vllm|4|latency|optimized|
|vllm-w7900-bf16-tp4-throughput|W7900|bf16|vllm|4|throughput|optimized|

The columns should be read as follows:
* **Profile**: Name of the deployment profile.
* **Accelerator model**: Target accelerator model for the profile.
* **Precision**: Numerical precision used for model inference. Most common precisions are `fp16` (half-precision floating point) and `fp8` (8-bit floating point).
* **Engine**: Inference engine used to run the model.
* **Accelerator count**: Number of accelerators utilized in the profile.
* **Metric**: Performance metric optimized the profile is optimized for. Common metrics are `latency` (time taken to generate a response) and `throughput` (number of requests handled per second).
* **Type**: Indicates whether the profile is `optimized`, `unoptimized`, or `general`.
  * `"optimized"`: Performance-tuned profiles with benchmarked configurations for specific model/hardware combinations
  * `"unoptimized"`: Basic profiles with default or minimal tuning, suitable as starting points for experimentation; these are never selected automatically and must be requested explicitly
  * `"general"`: Generic profiles applicable across multiple models, providing baseline configurations when model-specific profiles are unavailable
  * `"preview"`: Performance-tuned profiles which do not reach the same level of performance as "optimized" profiles, intended for early access to new configurations

## Getting Started

See the [deployment guide](DEPLOYMENT.md) for Docker and Kubernetes instructions.


# Terms of Use


This AIM can be used in accordance with the following licenses: Apache-2.0, MIT.



This model does not require a Hugging Face authentication.


