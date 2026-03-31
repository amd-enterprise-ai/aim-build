<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Available AIM containers

{% for organization in organizations %}

## {{ organization.representation }}

::::{grid} 1 1 1 1
:gutter: 3

  {% for aim in organization.aims %}
:::{grid-item-card}
[{{ aim.model_name }}]({{ aim.artefact_url }}) ({{ aim.suffix_representation }})
^^^
{{ aim.description }}

[Technical specification](../docs-aim/{{ aim.model_name }}/README.md#model-specific-aim)

+++

```bash
docker pull {{ aim.docker_info.registry_host }}/{{ aim.docker_info.registry_namespace }}/{{ aim.repository }}:{{ aim.tag }}
```

:::
  {% endfor %}
::::
{% endfor %}
