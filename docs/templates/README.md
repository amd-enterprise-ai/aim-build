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

|Profile|GPU|Precision|Engine|GPU count|Metric|Type|
|-------|---|---------|------|---------|------|----|
{% for profile in profiles %}
|{{ profile.profile_name }}|{{ profile.gpu }}|{{ profile.precision }}|{{ profile.engine }}|{{ profile.gpu_count }}|{{ profile.metric }}|{{ profile.profile_type }}|
{% endfor %}

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
