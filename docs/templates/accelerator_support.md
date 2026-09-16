<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Supported Accelerator Models

AIMs support several accelerator models at varying support levels, determined by the types of profiles included in a
given AIM. There are the following profile types (from the most optimized to the least optimized):
  * `"optimized"`: Performance-tuned profiles with benchmarked configurations for specific model/hardware combinations
  * `"preview"`: Performance-tuned profiles that do not reach the same level of performance as `"optimized"` profiles, intended for early access to new configurations
  * `"unoptimized"`: Basic profiles with default or minimal tuning, suitable as starting points for experimentation
  * `"general"`: Generic profiles applicable across multiple models, providing baseline configurations when model-specific profiles are unavailable

If an AIM contains at least one optimized profile for a specific accelerator model, then the support level for that
accelerator model is also optimized. If there are no optimized profiles but at least one preview profile, then the
support level is preview. If there are no optimized or preview profiles but there are model-specific unoptimized
profiles, then the support level is unoptimized. Otherwise, the support level is general.

The supported accelerator models and their support levels for each AIM are based on the latest public release and are
summarized in the table below.

```{admonition} ROCm version
ROCm 7.0.2 is recommended. ROCm 7.2.x is currently not supported.
```

<div class="sd-tab-set docutils">

{% set ns = namespace(index=1) %}

{% for accelerator_family in accelerator_families %}

{% if ns.index == 1 %}
<input checked="checked" id="sd-tab-item-{{ ns.index }}" name="sd-tab-set-1" type="radio">
{% else %}
<input id="sd-tab-item-{{ ns.index }}" name="sd-tab-set-1" type="radio">
{% endif %}

<label class="sd-tab-label" for="sd-tab-item-{{ ns.index }}">
 {{ accelerator_family.representation }}</label><div class="sd-tab-content docutils">
<div class="pst-scrollable-table-container">

<table style="border-collapse:collapse">
<thead>
<tr>
<th style="border:1px solid #ccc;padding:10px">#</th>
<th style="border:1px solid #ccc;padding:10px">AIM</th>
{% for gpu in accelerator_family.accelerators %}<th style="border:1px solid #ccc;padding:10px">{{ gpu.value }}</th>
{% endfor %}
</tr>
</thead>
<tbody>
{% for aim in accelerator_family.entities %}
<tr>
<td style="border:1px solid #ccc;padding:10px">{{ loop.index }}</td>
<td style="border:1px solid #ccc;padding:10px"><a href="{{ aim.artefact_url }}">{{ aim.model_name }}</a></td>
{% for gpu in accelerator_family.accelerators %}{% set level = aim.accelerator_support[gpu.value] %}{% if level == "optimized" %}<td style="border:1px solid #ccc;padding:10px;background-color:#409353">{{ level }}</td>
{% else %}<td style="border:1px solid #ccc;padding:10px">{{ level }}</td>
{% endif %}{% endfor %}
</tr>
{% endfor %}
</tbody>
</table>


</div>
</div>

{% set ns.index = ns.index + 1 %}

{% endfor %}

</div>


The tables should be read as follows:
* The **AIM** column contains links to each AIM's Docker images publicly available on Docker Hub.
* Accelerator model columns (MI250X, MI300X, MI325X, MI350X, MI355X, ...) show the support level for that accelerator in
the given AIM.
