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



<input checked="checked" id="sd-tab-item-1" name="sd-tab-set-1" type="radio">

<label class="sd-tab-label" for="sd-tab-item-1">
 Instinct</label><div class="sd-tab-content docutils">
<div class="pst-scrollable-table-container">

<table style="border-collapse:collapse">
<thead>
<tr>
<th style="border:1px solid #ccc;padding:10px">#</th>
<th style="border:1px solid #ccc;padding:10px">AIM</th>
<th style="border:1px solid #ccc;padding:10px">MI250X</th>
<th style="border:1px solid #ccc;padding:10px">MI300X</th>
<th style="border:1px solid #ccc;padding:10px">MI325X</th>
<th style="border:1px solid #ccc;padding:10px">MI350X</th>
<th style="border:1px solid #ccc;padding:10px">MI355X</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border:1px solid #ccc;padding:10px">1</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-coherelabs-command-a-reasoning-08-2025/tags">CohereLabs/command-a-reasoning-08-2025</a></td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">2</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-r1/tags">deepseek-ai/DeepSeek-R1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">3</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-r1-0528/tags">deepseek-ai/DeepSeek-R1-0528</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">4</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-v3-1/tags">deepseek-ai/DeepSeek-V3.1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">5</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-v3-1-terminus/tags">deepseek-ai/DeepSeek-V3.1-Terminus</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">6</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-3-1b-it/tags">google/gemma-3-1b-it</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">7</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-3-27b-it/tags">google/gemma-3-27b-it</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">8</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-4-31b-it/tags">google/gemma-4-31B-it</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">9</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-google-medgemma-27b-it/tags">google/medgemma-27b-it</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">10</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-1-405b-instruct/tags">meta-llama/Llama-3.1-405B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">11</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct/tags">meta-llama/Llama-3.1-8B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">12</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-2-1b-instruct/tags">meta-llama/Llama-3.2-1B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">13</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-2-3b-instruct/tags">meta-llama/Llama-3.2-3B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">14</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-3-70b-instruct/tags">meta-llama/Llama-3.3-70B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">15</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-minimaxai-minimax-m2-5/tags">MiniMaxAI/MiniMax-M2.5</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">16</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-minimaxai-minimax-m3/tags">MiniMaxAI/MiniMax-M3</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">17</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-ministral-3-14b-instruct-2512/tags">mistralai/Ministral-3-14B-Instruct-2512</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">18</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-ministral-3-14b-reasoning-2512/tags">mistralai/Ministral-3-14B-Reasoning-2512</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">19</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-large-3-675b-instruct-2512/tags">mistralai/Mistral-Large-3-675B-Instruct-2512</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">20</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-small-24b-instruct-2501/tags">mistralai/Mistral-Small-24B-Instruct-2501</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">21</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-small-3-2-24b-instruct-2506/tags">mistralai/Mistral-Small-3.2-24B-Instruct-2506</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">22</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mixtral-8x22b-instruct-v0-1/tags">mistralai/Mixtral-8x22B-Instruct-v0.1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">23</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1/tags">mistralai/Mixtral-8x7B-Instruct-v0.1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">24</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-120b/tags">openai/gpt-oss-120b</a></td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">25</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-20b/tags">openai/gpt-oss-20b</a></td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">26</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-235b-a22b/tags">Qwen/Qwen3-235B-A22B</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">27</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-32b/tags">Qwen/Qwen3-32B</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">28</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-coder-next/tags">Qwen/Qwen3-Coder-Next</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">29</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-vl-235b-a22b-instruct/tags">Qwen/Qwen3-VL-235B-A22B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">30</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-vl-235b-a22b-thinking/tags">Qwen/Qwen3-VL-235B-A22B-Thinking</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">31</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-zai-org-glm-4-7/tags">zai-org/GLM-4.7</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">32</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-zai-org-glm-5-2/tags">zai-org/GLM-5.2</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
</tr>
</tbody>
</table>


</div>
</div>



<input id="sd-tab-item-2" name="sd-tab-set-1" type="radio">

<label class="sd-tab-label" for="sd-tab-item-2">
 EPYC</label><div class="sd-tab-content docutils">
<div class="pst-scrollable-table-container">

<table style="border-collapse:collapse">
<thead>
<tr>
<th style="border:1px solid #ccc;padding:10px">#</th>
<th style="border:1px solid #ccc;padding:10px">AIM</th>
<th style="border:1px solid #ccc;padding:10px">EPYC_9965</th>
<th style="border:1px solid #ccc;padding:10px">EPYC_ZEN4</th>
<th style="border:1px solid #ccc;padding:10px">EPYC_ZEN5</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border:1px solid #ccc;padding:10px">1</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-deepseek-ai-deepseek-r1-distill-qwen-14b/tags">deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">2</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-deepseek-ai-deepseek-r1-distill-qwen-7b/tags">deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">3</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-3-12b-it/tags">google/gemma-3-12b-it</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">4</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-3-1b-it/tags">google/gemma-3-1b-it</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">5</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-3-4b-it/tags">google/gemma-3-4b-it</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">6</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-google-gemma-4-e4b-it/tags">google/gemma-4-E4B-it</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">7</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-ibm-granite-granite-3-3-2b-instruct/tags">ibm-granite/granite-3.3-2b-instruct</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">8</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-ibm-granite-granite-4-1-3b/tags">ibm-granite/granite-4.1-3b</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">9</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-ibm-granite-granite-4-1-8b/tags">ibm-granite/granite-4.1-8b</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">10</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-meta-llama-llama-3-1-8b-instruct/tags">meta-llama/Llama-3.1-8B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">11</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-meta-llama-llama-3-2-1b-instruct/tags">meta-llama/Llama-3.2-1B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">12</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-meta-llama-llama-3-2-3b-instruct/tags">meta-llama/Llama-3.2-3B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">13</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-microsoft-phi-3-5-mini-instruct/tags">microsoft/Phi-3.5-mini-instruct</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">14</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen2-5-coder-7b-instruct/tags">Qwen/Qwen2.5-Coder-7B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">15</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen2-5-vl-7b-instruct/tags">Qwen/Qwen2.5-VL-7B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">16</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-0-6b/tags">Qwen/Qwen3-0.6B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">17</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-1-7b/tags">Qwen/Qwen3-1.7B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">18</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-30b-a3b/tags">Qwen/Qwen3-30B-A3B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">19</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-4b/tags">Qwen/Qwen3-4B</a></td>
<td style="border:1px solid #ccc;padding:10px;background-color:#409353">optimized</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">20</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-8b/tags">Qwen/Qwen3-8B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">21</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-5-4b/tags">Qwen/Qwen3.5-4B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">22</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-5-9b/tags">Qwen/Qwen3.5-9B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">23</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-qwen-qwen3-6-35b-a3b/tags">Qwen/Qwen3.6-35B-A3B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">24</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-epyc-unsloth-gpt-oss-20b-bf16/tags">unsloth/gpt-oss-20b-BF16</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
</tr>
</tbody>
</table>


</div>
</div>



<input id="sd-tab-item-3" name="sd-tab-set-1" type="radio">

<label class="sd-tab-label" for="sd-tab-item-3">
 Radeon</label><div class="sd-tab-content docutils">
<div class="pst-scrollable-table-container">

<table style="border-collapse:collapse">
<thead>
<tr>
<th style="border:1px solid #ccc;padding:10px">#</th>
<th style="border:1px solid #ccc;padding:10px">AIM</th>
<th style="border:1px solid #ccc;padding:10px">R9700</th>
<th style="border:1px solid #ccc;padding:10px">W7900</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border:1px solid #ccc;padding:10px">1</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-radeon-google-gemma-3n-e4b-it/tags">google/gemma-3n-E4B-it</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">2</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-radeon-meta-llama-llama-3-1-8b-instruct/tags">meta-llama/Llama-3.1-8B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">3</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-radeon-qwen-qwen3-vl-8b-instruct/tags">Qwen/Qwen3-VL-8B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">4</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-radeon-qwen-qwen3-5-9b/tags">Qwen/Qwen3.5-9B</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">5</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-radeon-zai-org-glm-4-7-flash/tags">zai-org/GLM-4.7-Flash</a></td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
</tr>
</tbody>
</table>


</div>
</div>



</div>


The tables should be read as follows:
* The **AIM** column contains links to each AIM's Docker images publicly available on Docker Hub.
* Accelerator model columns (MI250X, MI300X, MI325X, MI350X, MI355X, ...) show the support level for that accelerator in
the given AIM.
