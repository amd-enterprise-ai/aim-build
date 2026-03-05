<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Metadata Overview

## Introduction

This document describes the metadata structure used in AIM (AMD Inference Microservice). Metadata files define the
properties, configuration, and deployment recommendations for AI models in the AIM catalog. Each model in the catalog
has its own `metadata.yaml` file that provides essential information about the model, its source, licensing, deployment
configurations, and other attributes.

The metadata is stored in YAML format and follows a structured schema that ensures consistency across all models in the
catalog. There are two types of metadata:

1. **Base Metadata** - For the generic AIM base image (`metadata/base/metadata.yaml`)
2. **Model Metadata** - For specific models (e.g., `metadata/meta-llama/Llama-3.1-8B-Instruct/metadata.yaml`)

## Directory Structure

The metadata directory is organized by model publisher and model name:

```
metadata/
├── base/
│   └── metadata.yaml                    # Base image metadata
├── CohereLabs/
│   └── command-a-reasoning-08-2025/
│       └── metadata.yaml
├── meta-llama/
│   ├── Llama-3.1-405B-Instruct/
│   │   └── metadata.yaml
│   ├── Llama-3.1-8B-Instruct/
│   │   └── metadata.yaml
│   └── ...
├── mistralai/
│   ├── Mistral-Small-3.2-24B-Instruct-2506/
│   │   └── metadata.yaml
│   └── ...
└── ...
```

## Metadata Schema

### Model Metadata Structure

Model metadata files contain two main sections: `com.amd.aim` (AMD-specific properties) and `org.opencontainers.image`
(OCI image labels).

#### Complete Example

```yaml
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

com:
  amd:
    aim:
      model:
        canonicalName: "meta-llama/Llama-3.1-8B-Instruct"
        tags:
          - "text-generation"
          - "chat"
          - "instruction"
        source: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
        variants:
          - "amd/Llama-3.1-8B-Instruct-FP8-KV"
          - "meta-llama/Llama-3.1-8B-Instruct"
        recommendedDeployments:
          - gpuModel: "MI300X"
            gpuCount: 1
            precision: "fp8"
            metric: "latency"
            description: "Optimized for latency on MI300X using fp8 precision"
          - gpuModel: "MI300X"
            gpuCount: 1
            precision: "fp8"
            metric: "throughput"
            description: "Optimized for throughput on MI300X using fp8 precision"
        publisher: "Meta"
      hfToken:
        required: true
      release:
        notes: ""
      description:
        full: "Meta Llama 3.1 8B model optimized for chat and instruction following. Built on transformer architecture with GQA and RLHF training."
      title: "Llama 3.1 8B Instruct"

org:
  opencontainers:
    image:
      vendor: "AMD"
      authors: ""
      licenses: "Apache-2.0"
      description: "Meta Llama 3.1 8B model optimized for chat and instruction following."
      documentation: ""
      source: "https://github.com/amd-enterprise-ai/aim-build"
```

### Field Descriptions

#### com.amd.aim Section

##### model (required)

Contains core model information.

- `canonicalName` (string, required): The canonical name of the model in the format `org/model-name`. This typically
matches with the Hugging Face model identifier.

- `tags` (array of strings, optional): Relevant model tags describing its capabilities and modalities. Tags include:
  - `text-generation`
  - `chat`
  - `instruction`
  - `code-generation`
  - `reasoning`

- `source` (string URI, optional): URL where the model can be found, typically a Hugging Face model page.

- `variants` (array of strings, optional): List of model variants available through the image, including quantized
variants.

- `publisher` (string, required): Name of the organization or individual that published the model (e.g., "Meta",
"Mistral AI", "OpenAI").

- `recommendedDeployments` (array of objects, optional): Array of recommended deployment configurations for
different hardware and optimization goals.

###### Recommended Deployment Object

Each deployment configuration can include:

- `gpuModel` (string, required): GPU model name. Supported values:
  - `MI100`, `MI210`, `MI250X`
  - `MI300A`, `MI300X`, `MI308X`, `MI325X`, `MI350X`, `MI355X`
  - `V620`, `V710`
  - `W6800`, `W6800X`, `W6900X`, `W7800`, `W7900`
  - `RX6800`, `RX6900`, `RX7900`, `RX9070`
  - `NONE` (for CPU-only deployments)

- `gpuCount` (integer, required): Number of GPUs required (0-8).

- `precision` (string, optional): Precision format for the deployment:
  - `fp4`, `fp8`, `fp16`, `fp32`, `bf16`
  - `int4`, `int8`

- `metric` (string, optional): Optimization metric:
  - `latency` - Optimized for low latency
  - `throughput` - Optimized for high throughput

- `description` (string, optional): Human-readable description of this deployment configuration.

- `profileId` (string, optional): Identifier for the specific profile to use (e.g., `vllm-mi325x-fp8-tp1-latency`).
This field is used when `manual_selection_only` is set to `true`, meaning that there is no optimized or preview profile
available for an AIM.

##### hfToken (optional)

Specifies whether a Hugging Face token is required to access the model.

- `required` (boolean, required): Set to `true` if the model requires authentication, `false` otherwise.

##### release (optional)

Release information for the model.

- `notes` (string, required): Release notes for this model version. Can be empty string.

##### description (optional)

Detailed description of the model.

- `full` (string, required): Full description of the model, its capabilities, and use cases.

##### title (required)

- `title` (string, required): Human readable title for an AIM (e.g., "Llama 3.1 8B Instruct").

#### org.opencontainers.image Section

This section follows the OCI image annotation specification and provides standard container image metadata.

##### image (required)

###### Values set in `metadata.yaml` files

These values are set explicitly in each model's `metadata.yaml` file:

- `authors` (string, required): Contact details of the people or organization responsible for the image. Can be empty
string.

- `description` (string, required): Brief description of the image/model. Maximum 160 characters.

- `documentation` (string, required): URL to documentation. Can be empty string.

- `licenses` (string, required): License information (e.g., "Apache-2.0", "MIT", "Apache-2.0, MIT"). Contains
comma-separated license names. Typically, contains 2 licenses: one for the model and one for the AIM image. When it
contains only one license, it applies to both the model and the AIM image. The value does not follow SPDX format at the
moment.

- `source` (string URI, required): Source repository URL (typically `https://github.com/amd-enterprise-ai/aim-build`).

- `vendor` (string, required): Vendor/organization name that created the image (typically "AMD").

###### Calculated values

These values are set by various processes during the build of an image:

- `created` (string): Image creation date (ISO 8601 string without microseconds).

- `ref.name` (string): Base OS name, typically `ubuntu`.

- `revision` (string): Specific source repository version tag/hash.

- `title` (string): Currently, equal to canonicalName for model-specific images and is set to `base` for base image.

- `version` (string): Image version label, changed if there are actual changes to the image.

### Base Metadata Structure

The base metadata (for the generic AIM image) has a simplified structure without model-specific fields:

```yaml
org:
  opencontainers:
    image:
      vendor: "AMD"
      authors: ""
      licenses: "MIT"
      description: "Generic image that can run any model in the AIM catalog. Model name should be specified using the environment variable AIM_MODEL_NAME."
      documentation: ""
      source: "https://github.com/amd-enterprise-ai/aim-build"

com:
  amd:
    aim:
      release:
        notes: ""
      description:
        full: "Generic image that can run any model in the AIM catalog. Model name should be specified using the environment variable AIM_MODEL_NAME."
      title: "AIM Base"
```

## Schema Validation

Metadata files are validated against JSON schemas located in the `schemas/` directory:

- `schemas/metadata_schema.json`: Schema for model-specific metadata files
- `schemas/base_metadata_schema.json`: Schema for the base metadata file

These schemas enforce:
- Required fields are present
- Field types are correct
- Enum values (like GPU models, precision types) are valid
- String length constraints (e.g., OCI description ≤ 160 characters)
- No additional properties beyond the schema

## Recommended Profiles

The `recommendedDeployments` field provides guidance on optimal deployment configurations for each model. These
recommendations are automatically generated from available profiles. It lists the best configuration for each GPU model
and metric (latency/throughput). Selection prioritizes: (1) profiles with `manual_selection_only=false`, (2) lowest
precision (lower is better: int4 > int8 > fp4 > fp8 > fp16 > bf16 > fp32), and (3) lowest GPU count.

### Profile ID Usage

The `profileId` field explicitly identifies which profile to use for deployment. It is included in recommended
deployments when:

- The best profile has `manual_selection_only=true` (requires explicit selection)
- No optimized or preview profile is available for a given AIM
- Explicit profile identification is needed for correct deployment

When `profileId` is present, the `precision` field is omitted since precision is determined by the referenced profile.
