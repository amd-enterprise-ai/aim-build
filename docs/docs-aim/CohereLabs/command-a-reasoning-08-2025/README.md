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


# Model-specific AIM

This AIM allows to deploy CohereLabs/command-a-reasoning-08-2025 with a tailored set of profiles.

* Model name: CohereLabs/command-a-reasoning-08-2025
* Description: Cohere's Command A Reasoning is a multilingual text generation model optimized for conversational AI and reasoning tasks across 23+ languages.
* Capabilities:
  * text-generation
  * conversational
  * chat
  * multilingual


## Available profiles

The following profiles are available for this model:

|Profile|GPU|Precision|Engine|GPU count|Metric|Type|
|-------|---|---------|------|---------|------|----|
|vllm-mi300x-fp16-tp8-latency|MI300X|fp16|vllm|8|latency|unoptimized|
|vllm-mi300x-fp16-tp1-latency|MI300X|fp16|vllm|1|latency|unoptimized|
|vllm-mi250x-fp16-tp8-latency|MI250X|fp16|vllm|8|latency|unoptimized|
|vllm-mi250x-fp16-tp8-throughput|MI250X|fp16|vllm|8|throughput|unoptimized|
|vllm-mi300x-fp16-tp2-latency|MI300X|fp16|vllm|2|latency|preview|
|vllm-mi300x-fp16-tp2-throughput|MI300X|fp16|vllm|2|throughput|preview|
|vllm-mi300x-fp16-tp4-latency|MI300X|fp16|vllm|4|latency|preview|
|vllm-mi300x-fp16-tp4-throughput|MI300X|fp16|vllm|4|throughput|optimized|
|vllm-mi300x-fp16-tp8-latency|MI300X|fp16|vllm|8|latency|optimized|
|vllm-mi300x-fp16-tp8-throughput|MI300X|fp16|vllm|8|throughput|optimized|
|vllm-mi325x-fp16-tp2-latency|MI325X|fp16|vllm|2|latency|unoptimized|
|vllm-mi325x-fp16-tp2-throughput|MI325X|fp16|vllm|2|throughput|unoptimized|
|vllm-mi325x-fp16-tp4-latency|MI325X|fp16|vllm|4|latency|unoptimized|
|vllm-mi325x-fp16-tp4-throughput|MI325X|fp16|vllm|4|throughput|unoptimized|
|vllm-mi325x-fp16-tp8-latency|MI325X|fp16|vllm|8|latency|unoptimized|
|vllm-mi325x-fp16-tp8-throughput|MI325X|fp16|vllm|8|throughput|unoptimized|
|vllm-mi300x-fp16-tp1-latency|MI300X|fp16|vllm|1|latency|general|
|vllm-mi300x-fp16-tp1-throughput|MI300X|fp16|vllm|1|throughput|general|
|vllm-mi300x-fp16-tp2-latency|MI300X|fp16|vllm|2|latency|general|
|vllm-mi300x-fp16-tp2-throughput|MI300X|fp16|vllm|2|throughput|general|
|vllm-mi300x-fp16-tp4-latency|MI300X|fp16|vllm|4|latency|general|
|vllm-mi300x-fp16-tp4-throughput|MI300X|fp16|vllm|4|throughput|general|
|vllm-mi300x-fp16-tp8-latency|MI300X|fp16|vllm|8|latency|general|
|vllm-mi300x-fp16-tp8-throughput|MI300X|fp16|vllm|8|throughput|general|
|vllm-mi325x-fp16-tp1-latency|MI325X|fp16|vllm|1|latency|general|
|vllm-mi325x-fp16-tp1-throughput|MI325X|fp16|vllm|1|throughput|general|
|vllm-mi325x-fp16-tp2-latency|MI325X|fp16|vllm|2|latency|general|
|vllm-mi325x-fp16-tp2-throughput|MI325X|fp16|vllm|2|throughput|general|
|vllm-mi325x-fp16-tp4-latency|MI325X|fp16|vllm|4|latency|general|
|vllm-mi325x-fp16-tp4-throughput|MI325X|fp16|vllm|4|throughput|general|
|vllm-mi325x-fp16-tp8-latency|MI325X|fp16|vllm|8|latency|general|
|vllm-mi325x-fp16-tp8-throughput|MI325X|fp16|vllm|8|throughput|general|

The columns should be read as follows:
* **Profile**: Name of the deployment profile.
* **GPU**: Target GPU model for the profile.
* **Precision**: Numerical precision used for model inference. Most common precisions are `fp16` (half-precision floating point) and `fp8` (8-bit floating point).
* **Engine**: Inference engine used to run the model.
* **GPU count**: Number of GPUs utilized in the profile.
* **Metric**: Performance metric optimized the profile is optimized for. Common metrics are `latency` (time taken to generate a response) and `throughput` (number of requests handled per second).
* **Type**: Indicates whether the profile is `optimized`, `unoptimized`, or `general`.
  * `optimized` profiles are those that have been specifically tuned for the model and hardware combination to deliver the best performance.
  * `unoptimized` profiles are provided without specific tuning and may not deliver optimal performance.
  * `general` profiles are created without a specific model in mind and can be used for a variety of models.


# Terms of use


This AIM can be used in accordance with the following licenses: CC-BY-NC-4.0, MIT.



This model requires a Hugging Face authentication. See instructions on how to get a Hugging Face token [here](https://huggingface.co/docs/hub/en/security-tokens).
To run AIM with this model, set the `HF_TOKEN` environment variable with your Hugging Face token value.


