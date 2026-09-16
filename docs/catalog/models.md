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


:::{tab-item} Instinct

| Model | Organization | Description | Resources |
|-------|--------------|-------------|-----------|
| [CohereLabs/command-a-reasoning-08-2025](https://hub.docker.com/r/amdenterpriseai/aim-coherelabs-command-a-reasoning-08-2025/tags) (stable) | Cohere Labs | 111B parameter language model with configurable reasoning and tool use capabilities. | [Spec](../docs-aim/instinct/CohereLabs/command-a-reasoning-08-2025/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/CohereLabs/command-a-reasoning-08-2025/DEPLOYMENT.md) |
| [deepseek-ai/DeepSeek-R1](https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-r1/tags) (stable) | DeepSeek | 671B parameter MoE reasoning model with 37B active parameters. | [Spec](../docs-aim/instinct/deepseek-ai/DeepSeek-R1/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/deepseek-ai/DeepSeek-R1/DEPLOYMENT.md) |
| [deepseek-ai/DeepSeek-R1-0528](https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-r1-0528/tags) (stable) | DeepSeek | 671B parameter MoE reasoning model with 37B active parameters, updated version of DeepSeek-R1. | [Spec](../docs-aim/instinct/deepseek-ai/DeepSeek-R1-0528/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/deepseek-ai/DeepSeek-R1-0528/DEPLOYMENT.md) |
| [deepseek-ai/DeepSeek-V3.1](https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-v3-1/tags) (stable) | DeepSeek | 671B parameter MoE model with 37B active parameters supporting thinking and non-thinking modes. | [Spec](../docs-aim/instinct/deepseek-ai/DeepSeek-V3.1/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/deepseek-ai/DeepSeek-V3.1/DEPLOYMENT.md) |
| [deepseek-ai/DeepSeek-V3.1-Terminus](https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-v3-1-terminus/tags) (stable) | DeepSeek | 671B parameter MoE model with 37B active parameters, refined for language consistency and agent tasks. | [Spec](../docs-aim/instinct/deepseek-ai/DeepSeek-V3.1-Terminus/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/deepseek-ai/DeepSeek-V3.1-Terminus/DEPLOYMENT.md) |
| [google/gemma-3-1b-it](https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-3-1b-it/tags) (stable) | Google | Gemma 3 1B IT is a lightweight instruction-tuned model supporting text generation with a 32K context window. | [Spec](../docs-aim/instinct/google/gemma-3-1b-it/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/google/gemma-3-1b-it/DEPLOYMENT.md) |
| [google/gemma-3-27b-it](https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-3-27b-it/tags) (stable) | Google | Gemma 3 27B IT is a multimodal instruction-tuned model supporting text and image input with a 128K context window. | [Spec](../docs-aim/instinct/google/gemma-3-27b-it/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/google/gemma-3-27b-it/DEPLOYMENT.md) |
| [google/gemma-4-31B-it](https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-4-31b-it/tags) (preview) | Google | Gemma 4 31B IT is a multimodal instruction-tuned model with text and image input, 256K native context, and Gemma 4 reasoning + tool-call parsers. | [Spec](../docs-aim/instinct/google/gemma-4-31B-it/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/google/gemma-4-31B-it/DEPLOYMENT.md) |
| [google/medgemma-27b-it](https://hub.docker.com/r/amdenterpriseai/aim-google-medgemma-27b-it/tags) (stable) | Google | Gemma 3-based 27B multimodal model fine-tuned for medical text and image tasks (X-ray, dermatology, ophthalmology, pathology, radiology reports). | [Spec](../docs-aim/instinct/google/medgemma-27b-it/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/google/medgemma-27b-it/DEPLOYMENT.md) |
| [meta-llama/Llama-3.1-405B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-1-405b-instruct/tags) (stable) | Meta | Multilingual 405B parameter instruction-tuned language model for dialogue use cases. | [Spec](../docs-aim/instinct/meta-llama/Llama-3.1-405B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/meta-llama/Llama-3.1-405B-Instruct/DEPLOYMENT.md) |
| [meta-llama/Llama-3.1-8B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct/tags) (stable) | Meta | Multilingual 8B parameter instruction-tuned language model for dialogue use cases. | [Spec](../docs-aim/instinct/meta-llama/Llama-3.1-8B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/meta-llama/Llama-3.1-8B-Instruct/DEPLOYMENT.md) |
| [meta-llama/Llama-3.2-1B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-2-1b-instruct/tags) (stable) | Meta | Multilingual 1B parameter instruction-tuned language model for dialogue and on-device use cases. | [Spec](../docs-aim/instinct/meta-llama/Llama-3.2-1B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/meta-llama/Llama-3.2-1B-Instruct/DEPLOYMENT.md) |
| [meta-llama/Llama-3.2-3B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-2-3b-instruct/tags) (stable) | Meta | Multilingual 3B parameter instruction-tuned language model for dialogue and on-device use cases. | [Spec](../docs-aim/instinct/meta-llama/Llama-3.2-3B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/meta-llama/Llama-3.2-3B-Instruct/DEPLOYMENT.md) |
| [meta-llama/Llama-3.3-70B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-3-70b-instruct/tags) (stable) | Meta | Multilingual 70B parameter instruction-tuned language model for dialogue use cases. | [Spec](../docs-aim/instinct/meta-llama/Llama-3.3-70B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/meta-llama/Llama-3.3-70B-Instruct/DEPLOYMENT.md) |
| [MiniMaxAI/MiniMax-M2.5](https://hub.docker.com/r/amdenterpriseai/aim-minimaxai-minimax-m2-5/tags) (stable) | MiniMax | 228B parameter mixture-of-experts language model with reasoning, tool calling, and coding capabilities. | [Spec](../docs-aim/instinct/MiniMaxAI/MiniMax-M2.5/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/MiniMaxAI/MiniMax-M2.5/DEPLOYMENT.md) |
| [MiniMaxAI/MiniMax-M3](https://hub.docker.com/r/amdenterpriseai/aim-minimaxai-minimax-m3/tags) (stable) | MiniMax | Native multimodal MoE model with 428B parameters, 23B active parameters, a 1M-token context window, reasoning, tool use, and coding capabilities. | [Spec](../docs-aim/instinct/MiniMaxAI/MiniMax-M3/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/MiniMaxAI/MiniMax-M3/DEPLOYMENT.md) |
| [mistralai/Ministral-3-14B-Instruct-2512](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-ministral-3-14b-instruct-2512/tags) (stable) | Mistral AI | 14B parameter instruction-tuned language model with vision and function calling capabilities. | [Spec](../docs-aim/instinct/mistralai/Ministral-3-14B-Instruct-2512/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Ministral-3-14B-Instruct-2512/DEPLOYMENT.md) |
| [mistralai/Ministral-3-14B-Reasoning-2512](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-ministral-3-14b-reasoning-2512/tags) (stable) | Mistral AI | 14B parameter instruction-tuned language model with vision and function calling capabilities. | [Spec](../docs-aim/instinct/mistralai/Ministral-3-14B-Reasoning-2512/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Ministral-3-14B-Reasoning-2512/DEPLOYMENT.md) |
| [mistralai/Mistral-Large-3-675B-Instruct-2512](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-large-3-675b-instruct-2512/tags) (stable) | Mistral AI | 675B parameter granular MoE multimodal model with 41B active parameters and vision capabilities. | [Spec](../docs-aim/instinct/mistralai/Mistral-Large-3-675B-Instruct-2512/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Mistral-Large-3-675B-Instruct-2512/DEPLOYMENT.md) |
| [mistralai/Mistral-Small-24B-Instruct-2501](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-small-24b-instruct-2501/tags) (stable) | Mistral AI | 24B parameter instruction-tuned language model (Mistral Small 3) with native function calling. Text-only. | [Spec](../docs-aim/instinct/mistralai/Mistral-Small-24B-Instruct-2501/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Mistral-Small-24B-Instruct-2501/DEPLOYMENT.md) |
| [mistralai/Mistral-Small-3.2-24B-Instruct-2506](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-small-3-2-24b-instruct-2506/tags) (stable) | Mistral AI | 24B parameter instruction-tuned language model with vision and function calling capabilities. | [Spec](../docs-aim/instinct/mistralai/Mistral-Small-3.2-24B-Instruct-2506/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Mistral-Small-3.2-24B-Instruct-2506/DEPLOYMENT.md) |
| [mistralai/Mixtral-8x22B-Instruct-v0.1](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mixtral-8x22b-instruct-v0-1/tags) (stable) | Mistral AI | Sparse MoE language model with 141B total parameters across 8 experts and function calling support. | [Spec](../docs-aim/instinct/mistralai/Mixtral-8x22B-Instruct-v0.1/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Mixtral-8x22B-Instruct-v0.1/DEPLOYMENT.md) |
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1/tags) (stable) | Mistral AI | Sparse MoE language model with 47B total parameters across 8 experts. | [Spec](../docs-aim/instinct/mistralai/Mixtral-8x7B-Instruct-v0.1/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/mistralai/Mixtral-8x7B-Instruct-v0.1/DEPLOYMENT.md) |
| [openai/gpt-oss-120b](https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-120b/tags) (stable) | OpenAI | Open-weight 117B parameter MoE model with 5.1B active parameters and configurable reasoning. | [Spec](../docs-aim/instinct/openai/gpt-oss-120b/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/openai/gpt-oss-120b/DEPLOYMENT.md) |
| [openai/gpt-oss-20b](https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-20b/tags) (stable) | OpenAI | Open-weight 21B parameter MoE model with 3.6B active parameters for lower-latency use cases. | [Spec](../docs-aim/instinct/openai/gpt-oss-20b/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/openai/gpt-oss-20b/DEPLOYMENT.md) |
| [Qwen/Qwen3-235B-A22B](https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-235b-a22b/tags) (stable) | Qwen | 235B parameter MoE language model with 22B active parameters and dual thinking modes. | [Spec](../docs-aim/instinct/Qwen/Qwen3-235B-A22B/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/Qwen/Qwen3-235B-A22B/DEPLOYMENT.md) |
| [Qwen/Qwen3-32B](https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-32b/tags) (stable) | Qwen | 32.8B parameter dense language model with dual thinking modes and multilingual support. | [Spec](../docs-aim/instinct/Qwen/Qwen3-32B/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/Qwen/Qwen3-32B/DEPLOYMENT.md) |
| [Qwen/Qwen3-Coder-Next](https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-coder-next/tags) (stable) | Qwen | 80B parameter MoE coding agent model with 3B active parameters and hybrid attention architecture. | [Spec](../docs-aim/instinct/Qwen/Qwen3-Coder-Next/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/Qwen/Qwen3-Coder-Next/DEPLOYMENT.md) |
| [Qwen/Qwen3-VL-235B-A22B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-vl-235b-a22b-instruct/tags) (stable) | Qwen | 236B parameter MoE vision-language model with 22B active parameters and multimodal capabilities. | [Spec](../docs-aim/instinct/Qwen/Qwen3-VL-235B-A22B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/Qwen/Qwen3-VL-235B-A22B-Instruct/DEPLOYMENT.md) |
| [Qwen/Qwen3-VL-235B-A22B-Thinking](https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-vl-235b-a22b-thinking/tags) (stable) | Qwen | 236B parameter MoE vision-language model with reasoning-enhanced thinking capabilities. | [Spec](../docs-aim/instinct/Qwen/Qwen3-VL-235B-A22B-Thinking/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/Qwen/Qwen3-VL-235B-A22B-Thinking/DEPLOYMENT.md) |
| [zai-org/GLM-4.7](https://hub.docker.com/r/amdenterpriseai/aim-zai-org-glm-4-7/tags) (stable) | Z.ai | GLM-4.7 is a large language model with multi-turn conversation, tool use, and reasoning capabilities. | [Spec](../docs-aim/instinct/zai-org/GLM-4.7/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/zai-org/GLM-4.7/DEPLOYMENT.md) |
| [zai-org/GLM-5.2](https://hub.docker.com/r/amdenterpriseai/aim-zai-org-glm-5-2/tags) (stable) | Z.ai | GLM-5.2 is a large Mixture-of-Experts language model with multi-turn conversation, tool use, and reasoning capabilities. | [Spec](../docs-aim/instinct/zai-org/GLM-5.2/README.md#model-specific-aim) [Deploy](../docs-aim/instinct/zai-org/GLM-5.2/DEPLOYMENT.md) |

:::


:::{tab-item} EPYC

| Model | Organization | Description | Resources |
|-------|--------------|-------------|-----------|
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-deepseek-ai-deepseek-r1-distill-qwen-14b/tags) (preview) | DeepSeek | 14B parameter distilled reasoning model based on Qwen-2.5, optimized for CPU inference on AMD EPYC. | [Spec](../docs-aim/epyc/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/DEPLOYMENT.md) |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-deepseek-ai-deepseek-r1-distill-qwen-7b/tags) (stable) | DeepSeek | 7B parameter distilled reasoning model based on Qwen-2.5, optimized for CPU inference on AMD EPYC. | [Spec](../docs-aim/epyc/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/DEPLOYMENT.md) |
| [google/gemma-3-12b-it](https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-3-12b-it/tags) (stable) | Google | Gemma 3 12B Instruct multimodal model from Google DeepMind. | [Spec](../docs-aim/epyc/google/gemma-3-12b-it/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/google/gemma-3-12b-it/DEPLOYMENT.md) |
| [google/gemma-3-1b-it](https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-3-1b-it/tags) (stable) | Google | Gemma 3 1B Instruct multimodal model from Google DeepMind. | [Spec](../docs-aim/epyc/google/gemma-3-1b-it/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/google/gemma-3-1b-it/DEPLOYMENT.md) |
| [google/gemma-3-4b-it](https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-3-4b-it/tags) (stable) | Google | Gemma 3 4B Instruct multimodal model from Google DeepMind. | [Spec](../docs-aim/epyc/google/gemma-3-4b-it/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/google/gemma-3-4b-it/DEPLOYMENT.md) |
| [google/gemma-4-E4B-it](https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-4-e4b-it/tags) (stable) | Google | Gemma 4 E4B is Google's efficient instruction-tuned model with a ~4B effective-parameter MatFormer architecture for low-latency on-device and CPU inference. | [Spec](../docs-aim/epyc/google/gemma-4-E4B-it/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/google/gemma-4-E4B-it/DEPLOYMENT.md) |
| [meta-llama/Llama-3.1-8B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-meta-llama-llama-3-1-8b-instruct/tags) (stable) | Meta | Multilingual 8B parameter instruction-tuned language model for dialogue use cases. | [Spec](../docs-aim/epyc/meta-llama/Llama-3.1-8B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/meta-llama/Llama-3.1-8B-Instruct/DEPLOYMENT.md) |
| [meta-llama/Llama-3.2-1B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-meta-llama-llama-3-2-1b-instruct/tags) (stable) | Meta | Multilingual 1B parameter instruction-tuned language model for dialogue and on-device use cases. | [Spec](../docs-aim/epyc/meta-llama/Llama-3.2-1B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/meta-llama/Llama-3.2-1B-Instruct/DEPLOYMENT.md) |
| [meta-llama/Llama-3.2-3B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-meta-llama-llama-3-2-3b-instruct/tags) (stable) | Meta | Multilingual 3B parameter instruction-tuned language model for dialogue and on-device use cases. | [Spec](../docs-aim/epyc/meta-llama/Llama-3.2-3B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/meta-llama/Llama-3.2-3B-Instruct/DEPLOYMENT.md) |
| [Qwen/Qwen2.5-Coder-7B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen2-5-coder-7b-instruct/tags) (stable) | Qwen | 7B parameter coding and instruction-following model optimized for CPU inference on AMD EPYC. | [Spec](../docs-aim/epyc/Qwen/Qwen2.5-Coder-7B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen2.5-Coder-7B-Instruct/DEPLOYMENT.md) |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen2-5-vl-7b-instruct/tags) (preview) | Qwen | 7B parameter vision-language model optimized for CPU inference on AMD EPYC. | [Spec](../docs-aim/epyc/Qwen/Qwen2.5-VL-7B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen2.5-VL-7B-Instruct/DEPLOYMENT.md) |
| [Qwen/Qwen3-0.6B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-0-6b/tags) (stable) | Qwen | Reasoning-enhanced 0.6B parameter LLM with thinking/non-thinking mode switching, excelling in math, coding, and multi-turn conversations. | [Spec](../docs-aim/epyc/Qwen/Qwen3-0.6B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3-0.6B/DEPLOYMENT.md) |
| [Qwen/Qwen3-1.7B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-1-7b/tags) (stable) | Qwen | Reasoning-enhanced 1.7B parameter LLM with thinking/non-thinking mode switching, excelling in math, coding, and multi-turn conversations. | [Spec](../docs-aim/epyc/Qwen/Qwen3-1.7B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3-1.7B/DEPLOYMENT.md) |
| [Qwen/Qwen3-30B-A3B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-30b-a3b/tags) (stable) | Qwen | Mixture-of-experts 30B (3B active) LLM with thinking/non-thinking mode switching, advanced reasoning, agent capabilities, and 100+ language support. | [Spec](../docs-aim/epyc/Qwen/Qwen3-30B-A3B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3-30B-A3B/DEPLOYMENT.md) |
| [Qwen/Qwen3-4B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-4b/tags) (stable) | Qwen | Reasoning-enhanced 4B parameter LLM with thinking/non-thinking mode switching, excelling in math, coding, and multi-turn conversations. | [Spec](../docs-aim/epyc/Qwen/Qwen3-4B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3-4B/DEPLOYMENT.md) |
| [Qwen/Qwen3-8B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-8b/tags) (stable) | Qwen | Reasoning-enhanced 8B parameter LLM with thinking/non-thinking mode switching, excelling in math, coding, and multi-turn conversations. | [Spec](../docs-aim/epyc/Qwen/Qwen3-8B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3-8B/DEPLOYMENT.md) |
| [Qwen/Qwen3.5-4B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-5-4b/tags) (stable) | Qwen | Qwen3.5-4B is a 4B parameter LLM with thinking/non-thinking dual-mode reasoning, strong math and coding ability, and multilingual support. | [Spec](../docs-aim/epyc/Qwen/Qwen3.5-4B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3.5-4B/DEPLOYMENT.md) |
| [Qwen/Qwen3.5-9B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-5-9b/tags) (stable) | Qwen | Qwen3.5-9B is a 9B parameter LLM with thinking/non-thinking dual-mode reasoning, strong math and coding ability, and multilingual support. | [Spec](../docs-aim/epyc/Qwen/Qwen3.5-9B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3.5-9B/DEPLOYMENT.md) |
| [Qwen/Qwen3.6-35B-A3B](https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-6-35b-a3b/tags) (stable) | Qwen | Qwen3.6-35B-A3B is a Mixture-of-Experts LLM with 35B total parameters and ~3B active per token, balancing high quality with efficient inference. | [Spec](../docs-aim/epyc/Qwen/Qwen3.6-35B-A3B/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/Qwen/Qwen3.6-35B-A3B/DEPLOYMENT.md) |
| [ibm-granite/granite-3.3-2b-instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-ibm-granite-granite-3-3-2b-instruct/tags) (stable) | ibm-granite | 2B parameter instruction-tuned language model with 128K context length and improved reasoning capabilities. | [Spec](../docs-aim/epyc/ibm-granite/granite-3.3-2b-instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/ibm-granite/granite-3.3-2b-instruct/DEPLOYMENT.md) |
| [ibm-granite/granite-4.1-3b](https://hub.docker.com/r/amdenterpriseai/aim-epyc-ibm-granite-granite-4-1-3b/tags) (stable) | ibm-granite | 3B parameter long-context instruct model with 131K context length and enhanced capabilities for summarization, RAG, and code tasks. | [Spec](../docs-aim/epyc/ibm-granite/granite-4.1-3b/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/ibm-granite/granite-4.1-3b/DEPLOYMENT.md) |
| [ibm-granite/granite-4.1-8b](https://hub.docker.com/r/amdenterpriseai/aim-epyc-ibm-granite-granite-4-1-8b/tags) (stable) | ibm-granite | 8B parameter long-context instruct model with 131K context length and enhanced capabilities for summarization, RAG, and code tasks. | [Spec](../docs-aim/epyc/ibm-granite/granite-4.1-8b/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/ibm-granite/granite-4.1-8b/DEPLOYMENT.md) |
| [microsoft/Phi-3.5-mini-instruct](https://hub.docker.com/r/amdenterpriseai/aim-epyc-microsoft-phi-3-5-mini-instruct/tags) (stable) | microsoft | 3.8B parameter language model optimized for reasoning, math, and code tasks in memory-constrained environments. | [Spec](../docs-aim/epyc/microsoft/Phi-3.5-mini-instruct/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/microsoft/Phi-3.5-mini-instruct/DEPLOYMENT.md) |
| [unsloth/gpt-oss-20b-BF16](https://hub.docker.com/r/amdenterpriseai/aim-epyc-unsloth-gpt-oss-20b-bf16/tags) (stable) | unsloth | gpt-oss-20B is OpenAI's open-weight 20B Mixture-of-Experts model (BF16 conversion by Unsloth) with strong reasoning and tool-use capabilities. | [Spec](../docs-aim/epyc/unsloth/gpt-oss-20b-BF16/README.md#model-specific-aim) [Deploy](../docs-aim/epyc/unsloth/gpt-oss-20b-BF16/DEPLOYMENT.md) |

:::


:::{tab-item} Radeon

| Model | Organization | Description | Resources |
|-------|--------------|-------------|-----------|
| [google/gemma-3n-E4B-it](https://hub.docker.com/r/amdenterpriseai/aim-radeon-google-gemma-3n-e4b-it/tags) (preview) | Google | Gemma 3n E4B IT is a gated multimodal instruction-tuned model supporting text, image, video, and audio inputs. | [Spec](../docs-aim/radeon/google/gemma-3n-E4B-it/README.md#model-specific-aim) [Deploy](../docs-aim/radeon/google/gemma-3n-E4B-it/DEPLOYMENT.md) |
| [meta-llama/Llama-3.1-8B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-radeon-meta-llama-llama-3-1-8b-instruct/tags) (preview) | Meta | Multilingual 8B parameter instruction-tuned language model for dialogue use cases. | [Spec](../docs-aim/radeon/meta-llama/Llama-3.1-8B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/radeon/meta-llama/Llama-3.1-8B-Instruct/DEPLOYMENT.md) |
| [Qwen/Qwen3-VL-8B-Instruct](https://hub.docker.com/r/amdenterpriseai/aim-radeon-qwen-qwen3-vl-8b-instruct/tags) (preview) | Qwen | 8B parameter vision-language model with advanced multimodal reasoning. | [Spec](../docs-aim/radeon/Qwen/Qwen3-VL-8B-Instruct/README.md#model-specific-aim) [Deploy](../docs-aim/radeon/Qwen/Qwen3-VL-8B-Instruct/DEPLOYMENT.md) |
| [Qwen/Qwen3.5-9B](https://hub.docker.com/r/amdenterpriseai/aim-radeon-qwen-qwen3-5-9b/tags) (preview) | Qwen | 9B parameter hybrid language model with Gated DeltaNet and dual thinking modes. | [Spec](../docs-aim/radeon/Qwen/Qwen3.5-9B/README.md#model-specific-aim) [Deploy](../docs-aim/radeon/Qwen/Qwen3.5-9B/DEPLOYMENT.md) |
| [zai-org/GLM-4.7-Flash](https://hub.docker.com/r/amdenterpriseai/aim-radeon-zai-org-glm-4-7-flash/tags) (preview) | Z.ai | 30B-A3B MoE language model with balanced performance and efficiency for lightweight deployment. | [Spec](../docs-aim/radeon/zai-org/GLM-4.7-Flash/README.md#model-specific-aim) [Deploy](../docs-aim/radeon/zai-org/GLM-4.7-Flash/DEPLOYMENT.md) |

:::


::::
