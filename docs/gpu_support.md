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
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">2</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-235b-a22b/tags">Qwen/Qwen3-235B-A22B</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">3</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-32b/tags">Qwen/Qwen3-32B</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">4</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-coder-next/tags">Qwen/Qwen3-Coder-Next</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">5</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-vl-235b-a22b-instruct/tags">Qwen/Qwen3-VL-235B-A22B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">6</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-qwen-qwen3-vl-235b-a22b-thinking/tags">Qwen/Qwen3-VL-235B-A22B-Thinking</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">7</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-r1/tags">deepseek-ai/DeepSeek-R1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">8</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-r1-0528/tags">deepseek-ai/DeepSeek-R1-0528</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">9</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-v3-1/tags">deepseek-ai/DeepSeek-V3.1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">10</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-deepseek-ai-deepseek-v3-1-terminus/tags">deepseek-ai/DeepSeek-V3.1-Terminus</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">11</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-google-gemma-3-27b-it/tags">google/gemma-3-27b-it</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">12</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-1-405b-instruct/tags">meta-llama/Llama-3.1-405B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">13</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct/tags">meta-llama/Llama-3.1-8B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">14</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-2-1b-instruct/tags">meta-llama/Llama-3.2-1B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">preview</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">15</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-2-3b-instruct/tags">meta-llama/Llama-3.2-3B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">16</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-meta-llama-llama-3-3-70b-instruct/tags">meta-llama/Llama-3.3-70B-Instruct</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">17</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-ministral-3-14b-instruct-2512/tags">mistralai/Ministral-3-14B-Instruct-2512</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">18</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-ministral-3-14b-reasoning-2512/tags">mistralai/Ministral-3-14B-Reasoning-2512</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">19</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-large-3-675b-instruct-2512/tags">mistralai/Mistral-Large-3-675B-Instruct-2512</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">20</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mistral-small-3-2-24b-instruct-2506/tags">mistralai/Mistral-Small-3.2-24B-Instruct-2506</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">21</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mixtral-8x22b-instruct-v0-1/tags">mistralai/Mixtral-8x22B-Instruct-v0.1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">22</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-mistralai-mixtral-8x7b-instruct-v0-1/tags">mistralai/Mixtral-8x7B-Instruct-v0.1</a></td>
<td style="border:1px solid #ccc;padding:10px">general</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">23</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-120b/tags">openai/gpt-oss-120b</a></td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
<tr>
<td style="border:1px solid #ccc;padding:10px">24</td>
<td style="border:1px solid #ccc;padding:10px"><a href="https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-20b/tags">openai/gpt-oss-20b</a></td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px">unoptimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
<td style="border:1px solid #ccc;padding:10px;background-color:#d4edda">optimized</td>
</tr>
</tbody>
</table>

The table should be read as follows:
* The **AIM** column contains links to each AIM's Docker images publicly available on Docker Hub.
* GPU model columns (MI250X, MI300X, MI325X, MI350X, MI355X, ...) show the support level for that GPU in the given AIM.
