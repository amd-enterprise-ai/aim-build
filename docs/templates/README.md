<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

{{ read_file('../overview.md') | skip_lines(6) }}

{% if aim_overview.is_general %}

# General-purpose AIM

This AIM allows to deploy any supported model with a general set of profiles.

{% else %}

# Model-specific AIM

This AIM allows to deploy {{ aim_overview.model_name }} with a tailored set of profiles.

* Model name: {{ aim_overview.model_name }}
* Description: {{ aim_overview.model_description }}
{% if aim_overview.model_tags %}
* Capabilities:
  {% for tag in aim_overview.model_tags %}
  * {{ tag }}
  {% endfor %}
{% endif %}

{% endif %}

## Available profiles

The following profiles are available for this model:

|Profile| Accelerator model |Precision|Engine|Accelerator count|Metric|Type|
|-------|-------------------|---------|------|---------|------|----|
{% for profile in profiles %}
|{{ profile.profile_handling.profile_name }}|{{ profile.metadata.accelerator_model.value }}|{{ profile.metadata.precision.value }}|{{ profile.metadata.engine.value }}|{{ profile.metadata.accelerator_count }}|{{ profile.metadata.metric.value }}|{{ profile.metadata.type.value }}|
{% endfor %}

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

## Getting started

See the [deployment guide](DEPLOYMENT.md) for Docker and Kubernetes instructions.

{% if terms_of_use.defined %}

# Terms of use

  {% if terms_of_use.licenses_defined %}

This AIM can be used in accordance with the following licenses: {{ terms_of_use.licenses }}.

  {% endif %}

  {% if terms_of_use.hf_token_defined %}

    {% if terms_of_use.hf_token %}
This model requires a Hugging Face authentication. See instructions on how to get a Hugging Face token [here](https://huggingface.co/docs/hub/en/security-tokens).
To run AIM with this model, set the `HF_TOKEN` environment variable with your Hugging Face token value.
    {% else %}
This model does not require a Hugging Face authentication.
    {% endif %}

  {% endif %}

{% endif %}
