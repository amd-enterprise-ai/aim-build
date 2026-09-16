<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIMs Catalog

The AIM catalog includes ready-to-deploy inference microservices for popular open-weight models, from lightweight instruction-tuned models to large mixture-of-experts systems. The catalog is constantly expanded to add new models and hardware-optimized profiles. To deploy models currently not available in the catalog but available through Hugging Face and supported by vLLM, utilize the AIM base container together with a custom profile ([see here](../custom_profiles.md)).

AIMs are available for AMD Instinct™, Radeon™, and EPYC™. Browse the catalog below to find a container for your model and hardware. 

<div style="display: flex; flex-wrap: wrap; gap: 1em; align-items: end; margin-bottom: 1em;">
  <label style="display: flex; flex-direction: column; gap: 0.25em;">
    Search
    <input type="text" id="aim-catalog-search" placeholder="Filter by model, organization, or description..." style="width: 24em; max-width: 100%; padding: 0.5em; box-sizing: border-box;">
  </label>

  <label style="display: flex; flex-direction: column; gap: 0.25em;">
    Organization
    <select id="aim-organization-filter" style="min-width: 16em; padding: 0.5em;">
      <option value="">All organizations</option>
    </select>
  </label>
</div>

<div id="aim-catalog-tabs-marker"></div>

<script>
document.addEventListener("DOMContentLoaded", function () {
  var search = document.getElementById("aim-catalog-search");
  var organizationFilter = document.getElementById("aim-organization-filter");
  var tabSet = document.getElementById("aim-catalog-tabs-marker").nextElementSibling;

  while (tabSet && !tabSet.classList.contains("sd-tab-set")) {
    tabSet = tabSet.nextElementSibling;
  }

  if (!tabSet) {
    return;
  }

  var radios = Array.from(tabSet.children).filter(function (element) {
    return element.matches('input[type="radio"]');
  });
  var contents = Array.from(tabSet.children).filter(function (element) {
    return element.classList.contains("sd-tab-content");
  });

  function activeTable() {
    var index = radios.findIndex(function (radio) {
      return radio.checked;
    });
    return contents[index] ? contents[index].querySelector("table") : null;
  }

  function filterRows() {
    var table = activeTable();
    if (!table) {
      return;
    }

    var query = search.value.trim().toLowerCase();
    table.querySelectorAll("tbody tr").forEach(function (row) {
      var organization = row.cells[1].textContent.trim();
      var matchesSearch = !query || row.textContent.toLowerCase().includes(query);
      var matchesOrganization = !organizationFilter.value || organization === organizationFilter.value;
      row.style.display = matchesSearch && matchesOrganization ? "" : "none";
    });
  }

  function populateOrganizations() {
    var table = activeTable();
    var organizations = new Set();

    if (table) {
      table.querySelectorAll("tbody tr").forEach(function (row) {
        organizations.add(row.cells[1].textContent.trim());
      });
    }

    organizationFilter.replaceChildren(new Option("All organizations", ""));
    Array.from(organizations)
      .sort(function (left, right) {
        return left.localeCompare(right);
      })
      .forEach(function (organization) {
        organizationFilter.add(new Option(organization, organization));
      });

    filterRows();
  }

  search.addEventListener("input", filterRows);
  organizationFilter.addEventListener("change", filterRows);
  radios.forEach(function (radio) {
    radio.addEventListener("change", populateOrganizations);
  });

  populateOrganizations();
});
</script>

::::{tab-set}

{% for accelerator_family in accelerator_families %}

:::{tab-item} {{ accelerator_family.representation }}

| Model | Organization | Description | Resources |
|-------|--------------|-------------|-----------|
{% for organization in accelerator_family.entities %}
{% for aim in organization.aims %}
| [{{ aim.model_name }}]({{ aim.artefact_url }}) ({{ aim.suffix_representation }}) | {{ organization.representation }} | {{ aim.description }} | [Spec](../docs-aim/{{ aim.accelerator_family.value }}/{{ aim.model_name }}/README.md#model-specific-aim) [Deploy](../docs-aim/{{ aim.accelerator_family.value }}/{{ aim.model_name }}/DEPLOYMENT.md) |
{% endfor %}
{% endfor %}

:::

{% endfor %}

::::
