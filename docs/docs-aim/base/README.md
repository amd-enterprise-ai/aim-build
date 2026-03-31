<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIMs Overview

AIM stands for AMD Inference Microservice. AIMs provide standardized, portable inference microservices for serving AI
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
| Docker  | A platform for developing, shipping, and running applications in containers                         |
| GPU     | A graphics processing unit. Essential hardware for running AI models                                |
| HF      | Hugging Face, a popular platform for sharing machine learning models and datasets                   |
| LLM     | Large Language Model                                                                                |
| Profile | A predefined AIM run configuration that can be optimized for specific models, compute, or use cases |
| ROCm    | Radeon Open Compute, AMD's open software platform for GPU computing                                 |
| YAML    | A human-readable data serialization format often used for configuration files                       |


# General-purpose AIM

This AIM allows to deploy any supported model with a general set of profiles.


## Available profiles

The following profiles are available for this model:

|Profile|GPU|Precision|Engine|GPU count|Metric|Type| Manual Only |
|-------|---|---------|------|---------|------|----|-------------|
|vllm-mi250x-fp16-tp1-latency|MI250X|fp16|vllm|1|latency|general|False|
|vllm-mi250x-fp16-tp1-throughput|MI250X|fp16|vllm|1|throughput|general|False|
|vllm-mi250x-fp16-tp2-latency|MI250X|fp16|vllm|2|latency|general|False|
|vllm-mi250x-fp16-tp2-throughput|MI250X|fp16|vllm|2|throughput|general|False|
|vllm-mi250x-fp16-tp4-latency|MI250X|fp16|vllm|4|latency|general|False|
|vllm-mi250x-fp16-tp4-throughput|MI250X|fp16|vllm|4|throughput|general|False|
|vllm-mi250x-fp16-tp8-latency|MI250X|fp16|vllm|8|latency|general|False|
|vllm-mi250x-fp16-tp8-throughput|MI250X|fp16|vllm|8|throughput|general|False|
|vllm-mi300x-fp16-tp1-latency|MI300X|fp16|vllm|1|latency|general|False|
|vllm-mi300x-fp16-tp1-throughput|MI300X|fp16|vllm|1|throughput|general|False|
|vllm-mi300x-fp16-tp2-latency|MI300X|fp16|vllm|2|latency|general|False|
|vllm-mi300x-fp16-tp2-throughput|MI300X|fp16|vllm|2|throughput|general|False|
|vllm-mi300x-fp16-tp4-latency|MI300X|fp16|vllm|4|latency|general|False|
|vllm-mi300x-fp16-tp4-throughput|MI300X|fp16|vllm|4|throughput|general|False|
|vllm-mi300x-fp16-tp8-latency|MI300X|fp16|vllm|8|latency|general|False|
|vllm-mi300x-fp16-tp8-throughput|MI300X|fp16|vllm|8|throughput|general|False|
|vllm-mi325x-fp16-tp1-latency|MI325X|fp16|vllm|1|latency|general|False|
|vllm-mi325x-fp16-tp1-throughput|MI325X|fp16|vllm|1|throughput|general|False|
|vllm-mi325x-fp16-tp2-latency|MI325X|fp16|vllm|2|latency|general|False|
|vllm-mi325x-fp16-tp2-throughput|MI325X|fp16|vllm|2|throughput|general|False|
|vllm-mi325x-fp16-tp4-latency|MI325X|fp16|vllm|4|latency|general|False|
|vllm-mi325x-fp16-tp4-throughput|MI325X|fp16|vllm|4|throughput|general|False|
|vllm-mi325x-fp16-tp8-latency|MI325X|fp16|vllm|8|latency|general|False|
|vllm-mi325x-fp16-tp8-throughput|MI325X|fp16|vllm|8|throughput|general|False|
|vllm-mi350x-fp16-tp1-latency|MI350X|fp16|vllm|1|latency|general|False|
|vllm-mi350x-fp16-tp1-throughput|MI350X|fp16|vllm|1|throughput|general|False|
|vllm-mi350x-fp16-tp2-latency|MI350X|fp16|vllm|2|latency|general|False|
|vllm-mi350x-fp16-tp2-throughput|MI350X|fp16|vllm|2|throughput|general|False|
|vllm-mi350x-fp16-tp4-latency|MI350X|fp16|vllm|4|latency|general|False|
|vllm-mi350x-fp16-tp4-throughput|MI350X|fp16|vllm|4|throughput|general|False|
|vllm-mi350x-fp16-tp8-latency|MI350X|fp16|vllm|8|latency|general|False|
|vllm-mi350x-fp16-tp8-throughput|MI350X|fp16|vllm|8|throughput|general|False|
|vllm-mi355x-fp16-tp1-latency|MI355X|fp16|vllm|1|latency|general|False|
|vllm-mi355x-fp16-tp1-throughput|MI355X|fp16|vllm|1|throughput|general|False|
|vllm-mi355x-fp16-tp2-latency|MI355X|fp16|vllm|2|latency|general|False|
|vllm-mi355x-fp16-tp2-throughput|MI355X|fp16|vllm|2|throughput|general|False|
|vllm-mi355x-fp16-tp4-latency|MI355X|fp16|vllm|4|latency|general|False|
|vllm-mi355x-fp16-tp4-throughput|MI355X|fp16|vllm|4|throughput|general|False|
|vllm-mi355x-fp16-tp8-latency|MI355X|fp16|vllm|8|latency|general|False|
|vllm-mi355x-fp16-tp8-throughput|MI355X|fp16|vllm|8|throughput|general|False|

The columns should be read as follows:
* **Profile**: Name of the deployment profile.
* **GPU**: Target GPU model for the profile.
* **Precision**: Numerical precision used for model inference. Most common precisions are `fp16` (half-precision floating point) and `fp8` (8-bit floating point).
* **Engine**: Inference engine used to run the model.
* **GPU count**: Number of GPUs utilized in the profile.
* **Metric**: Performance metric optimized the profile is optimized for. Common metrics are `latency` (time taken to generate a response) and `throughput` (number of requests handled per second).
* **Type**: Indicates whether the profile is `optimized`, `unoptimized`, or `general`.
  * `"optimized"`: Performance-tuned profiles with benchmarked configurations for specific model/hardware combinations
  * `"unoptimized"`: Basic profiles with default or minimal tuning, suitable as starting points for experimentation
  * `"general"`: Generic profiles applicable across multiple models, providing baseline configurations when model-specific profiles are unavailable
  * `"preview"`: Performance-tuned profiles which do not reach the same level of performance as "optimized" profiles, intended for early access to new configurations


# Terms of use


This AIM can be used in accordance with the following licenses: MIT.



