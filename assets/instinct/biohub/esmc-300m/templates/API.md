<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Embed one protein sequence (default), or score it with a masked-token loss and perplexity (`mode` `masked_lm`). |
| `GET` | `/v1/models` | Report the served model. |
| `GET` | `/health` | Check whether the service is ready to accept requests. |

`/healthz` also answers with HTTP 200 once the model is loaded; see section 4.1.

### 4.3 Request body

Request bodies are flat JSON objects. Do not nest the fields under another key.

`/v1/inference` takes `sequence` and three optional fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sequence` | string | required | Protein sequence in one-letter amino-acid codes. Whitespace (including inside the sequence) is removed and lowercase is converted to uppercase. |
| `mode` | string | `embedding` | `embedding` returns the embedding of the sequence; `masked_lm` returns the masked-token loss and perplexity. Any other value is rejected with HTTP 400. |
| `mask_fraction` | number | `0.15` | `masked_lm` only: fraction of residues to mask. Must be greater than 0 and less than 1. Not used in `embedding` mode, but the range is still checked in every mode: a value outside it is rejected with 400. |
| `seed` | integer | `0` | `masked_lm` only: seed that selects which residues are masked. The same seed gives the same loss. Not used in `embedding` mode. |

`/v1/models` and `/health` take no request body.

A sequence can hold at most **2046 residues**: the 2048-token context window minus the start and end tokens. A longer
sequence, or an empty one, is rejected with HTTP 400 and the service does not truncate it.

**Which characters are accepted.** After whitespace removal and upper-casing, every character must be a one-letter
residue code in the tokenizer's vocabulary (the 20 standard amino acids plus the ambiguity and rare codes the
tokenizer lists, including `X` for an unknown residue). Any other character, such as a digit or a symbol (for example
`MKTV1RQE`), is rejected with HTTP 400 and a message that lists the first few unsupported characters; the sequence is
never silently turned into an unknown token.

### 4.4 Example request

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"},
    timeout=60,
)

embedding = response.json()["embedding"]
print(len(embedding))
```

Score a sequence with the masked-token objective:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "mode": "masked_lm", "mask_fraction": 0.15, "seed": 0}'
```

### 4.5 Response

`/v1/inference` in the default `embedding` mode returns the embedding, its length and the served model:

```json
{
  "embedding": [0.0123, -0.0456, "... 960 values in total ..."],
  "embedding_dim": 960,
  "model": "biohub/ESMC-300M"
}
```

The embedding is the mean of the final hidden states over the residue tokens. The start and end tokens are left out.

In `masked_lm` mode it returns the loss and perplexity over the masked residues instead:

```json
{
  "masked_positions": 10,
  "sequence_length": 65,
  "loss": 1.5559,
  "perplexity": 4.7396,
  "model": "biohub/ESMC-300M"
}
```

These `masked_lm` values are an example for the request in section 4.4 (`mask_fraction` 0.15, `seed` 0): other
hardware or library versions can differ. The example values are reproduced within a couple of percent; when you
check your deployment against them, allow a tolerance of 0.03 on the loss and 0.15 on the perplexity.

`/v1/models` lists the served model:

```json
{
  "object": "list",
  "data": [{"id": "biohub/ESMC-300M", "object": "model"}]
}
```

`/health` returns `{"status": "ok"}` with HTTP 200 once the model is loaded. Before that the port is closed (connection
refused), so there is no 503 body to read while the model loads; poll until the connection succeeds. `/health` answers
`{"status": "unavailable"}` with HTTP 503 only when the service is up but its BentoML health check fails.

### 4.6 What to know before you deploy

**The model is a protein language model.** It takes amino-acid sequences and nothing else. The embedding is a
per-sequence summary, not per-residue; use it for similarity search, clustering or as input to a downstream predictor.

**The perplexity depends on the mask.** A masked-token loss from the `masked_lm` mode varies with `mask_fraction` and
`seed`. Compare scores only when both are the same. A single seed gives a noisy estimate of the model's fit to a sequence:
average the loss over several seeds (for example 5 to 10) before comparing sequences.

**The service needs a GPU.** It fails at startup with an error when no GPU is visible (see the startup failure note
below), instead of falling back to CPU.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**The weights live under `AIM_CACHE_PATH`.** The service reads `biohub/ESMC-300M` (about 620 MB) from the Hugging Face
hub cache layout under `AIM_CACHE_PATH` and downloads it there on first start when absent. At startup it logs a
`Weights pre-flight` line (to the container's stderr, so it shows in `docker logs`) with the cache path, whether
weights are already cached and whether `HF_TOKEN` is set. The model is public and needs no `HF_TOKEN`. Mount a
persistent volume at `AIM_CACHE_PATH` to avoid downloading on every start.

**Startup failure.** If no GPU is visible, or the weights cannot be loaded (not cached and the download fails, or
`ESMC_HF_REPO` names a repo that does not exist), the service logs a critical `ESMC-300M startup failed: ...` message
and the BentoML worker exits. The message for a weights failure names `AIM_CACHE_PATH` and notes that `HF_TOKEN` is not
required; the no-GPU message is `No GPU found. ESMC requires a CUDA/ROCm GPU.` and names neither. The container does **not** exit: BentoML's
supervisor restarts the worker about every 10 seconds, so the message repeats, the container stays running and the
service never becomes ready (`/health` is unreachable and can hang until the client times out). Use a readiness probe
(for example `/readyz`) and watch the container logs to detect this; the container exit code and a restart policy will
not.

**One optional variable selects the Hub repo.** `ESMC_HF_REPO` (default `biohub/ESMC-300M`) overrides the repo the
service loads. Only the default is tested; other repos can break the 960-dimension and 2048-token assumptions. The
service does not read `AIM_MODEL_ID`, which the AIM runtime reserves for catalog model selection.

**Use is subject to the Biohub Acceptable Use Policy.** The weights are released under the MIT license, and the model
card prohibits uses banned by the Biohub Acceptable Use Policy.
