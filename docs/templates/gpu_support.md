<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Supported GPU Models

Every AIM supports several GPU models at varying support levels, determined by the types of profiles included in a
given AIM. There are the following profile types (from the most optimized to the least optimized):
  * `"optimized"`: Performance-tuned profiles with benchmarked configurations for specific model/hardware combinations
  * `"preview"`: Performance-tuned profiles that do not reach the same level of performance as `"optimized"` profiles, intended for early access to new configurations
  * `"unoptimized"`: Basic profiles with default or minimal tuning, suitable as starting points for experimentation
  * `"general"`: Generic profiles applicable across multiple models, providing baseline configurations when model-specific profiles are unavailable

If an AIM contains at least one optimized profile for a specific GPU model, then the support level for that GPU model
is also optimized. If there are no optimized profiles but at least one preview profile, then the support level is
preview. If there are no optimized or preview profiles but there are model-specific unoptimized profiles, then the
support level is unoptimized. Otherwise, the support level is general.

The supported GPU models and their support levels for each AIM are based on the latest public release and are summarized
in the table below.

<table style="border-collapse:collapse">
<thead>
<tr>
<th style="border:1px solid #ccc;padding:10px">#</th>
<th style="border:1px solid #ccc;padding:10px">AIM</th>
{% for gpu in gpus %}<th style="border:1px solid #ccc;padding:10px">{{ gpu }}</th>
{% endfor %}
</tr>
</thead>
<tbody>
{% for aim in gpu_support %}
<tr>
<td style="border:1px solid #ccc;padding:10px">{{ loop.index }}</td>
<td style="border:1px solid #ccc;padding:10px"><a href="{{ aim.artefact_url }}">{{ aim.model_name }}</a></td>
{% for gpu in gpus %}{% set level = aim.gpu_support[gpu] %}{% if level == "optimized" %}<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">{{ level }}</td>
{% else %}<td style="border:1px solid #ccc;padding:10px">{{ level }}</td>
{% endif %}{% endfor %}
</tr>
{% endfor %}
</tbody>
</table>

The table should be read as follows:
* The **AIM** column contains links to each AIM's Docker images publicly available on Docker Hub.
* GPU model columns (MI250X, MI300X, MI325X, MI350X, MI355X, ...) show the support level for that GPU in the given AIM.
