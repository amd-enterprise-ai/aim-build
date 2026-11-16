<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 Endpoints

This AIM serves one model endpoint and the shared AIM health and model-listing endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Return the SaProt 650M AF2 embedding of one structure-aware protein sequence. |
| `GET` | `/health` | Readiness. Forwards to the BentoML `/healthz` check of the same service and returns its status code. |
| `GET` | `/v1/models` | Lists the model the container serves. |
| `GET` | `/healthz` | BentoML built-in health check. The service also keeps BentoML's `/livez`, `/readyz`, `/metrics` and `/docs.json` (OpenAPI schema). |

There is no `/predict` endpoint. Requests to `/predict` are not served; use `/v1/inference`.

`GET /health` returns `{"status": "ok"}` with status `200` when the BentoML health check succeeds. It returns
`{"status": "unavailable"}` with the upstream status code if the check fails, and with `503` if the service cannot reach
its own health check. Use it as a readiness signal, not as a liveness signal.

`GET /v1/models` returns an OpenAI-style list with the single served model. `id` is the AIM name,
`metadata.dtype` is the precision the model runs in, and `metadata.accelerator` is the name PyTorch reports for the first
visible GPU (`null` if it cannot be read):

```json
{
  "object": "list",
  "data": [
    {
      "id": "westlake-repl/saprot-650m-af2",
      "object": "model",
      "metadata": {"dtype": "fp16", "accelerator": "<GPU name reported by PyTorch>"}
    }
  ]
}
```

### 4.3 Request body

The whole request is nested under a top-level `request` object:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sequence` | string | required | A structure-aware sequence in SaProt format. |

SaProt reads a sequence that interleaves one amino-acid letter and one 3Di structural token per residue, for example
`M#A#K#T#`. The `#` character is the placeholder for a residue with no structural information, so a plain
amino-acid sequence is sent as each residue followed by `#`. A sequence of `N` residues is therefore `2N` characters
long.

The model context is 1024 tokens. SaProt has one token per residue (amino-acid letter plus 3Di token), and the start
and end tokens take two positions, so the service accepts at most **1022 residues** (2044 characters). A longer
sequence is rejected with status `400` before the model runs. The service never truncates the input, so a `200` response
always embeds the whole sequence you sent. The service also rejects an empty or whitespace-only `sequence` and any `sequence` that is not strictly paired.
There is no batch form: send one request per sequence.

### 4.4 Example request

The examples assume that the service is reachable at `localhost:8000`; adjust the port to the one you published.

Check readiness and the served model:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the embedding request:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"request": {"sequence": "M#A#K#T#A#Y#I#A#K#Q#"}}'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"request": {"sequence": "M#A#K#T#A#Y#I#A#K#Q#"}},
    timeout=60,
)
response.raise_for_status()

result = response.json()
print(result["embedding_dim"], result["embedding"][:5])
```

A sequence over the limit returns `400` with a message that states the limit and the received length, for example for
a sequence of 1023 residues:

```text
'sequence' is too long: 1023 residues received, the maximum is 1022 residues (1024 tokens including start and end tokens). Shorten the sequence; it is not truncated.
```

The `400` body has one of two shapes. Errors raised by the service itself (this message, the empty-sequence
message, the format messages) are a JSON array, `[{"error": "..."}]`. Request-schema failures (missing `request`
wrapper or `sequence`, not valid JSON, `sequence` not a string) are an object,
`{"error": "1 validation error for Input", "detail": [...]}`. Clients must handle both. Residues are counted as half the
number of characters of the stripped `sequence`, rounded up. The service also keeps a defensive check of the
real tokenized length against the 1024-token maximum; the paired-format and residue limits above already prevent it
from triggering.

The `sequence` must be strictly paired: an uppercase amino-acid letter followed by a lowercase 3Di letter or `#`,
so its stripped length is even. The amino-acid letter must be one of `ACDEFGHIKLMNPQRSTVWY` and the 3Di letter one of
`acdefghiklmnpqrstvwy` or `#`, the model's real vocabulary (446 entries: 5 special tokens and 441 other tokens).
The letters `B J O U X Z` and `b j o u x z` are not in the vocabulary and return `400` naming the offending
characters, instead of being silently turned into an unknown token.
This includes `X` (unknown residue) and `U` (selenocysteine), which do occur in real sequences. The real SaProt
tokenizer maps any amino-acid and 3Di pair that is not in the vocabulary, such as `Xd` or `Ud`, to the `<unk>` token
without an error, and the model would then embed that position as unknown. The service does not allow this: it returns
`400`, and it does not replace or drop such residues for you. A plain amino-acid sequence, an odd length,
internal whitespace, uppercase 3Di letters, or tokens that start with `#` (such as `#x` or `##`) also return `400`
with a message naming the offending characters.

### 4.5 Sequences with structure tokens

Embeddings from SaProt are meant to be computed from a sequence with real 3Di tokens. An all-`#` sequence carries no
structural information, so use `#` only for residues that have no structure or a low-confidence one (the upstream
model card uses `#` for AlphaFold2 regions with pLDDT below 70).

An example with real 3Di letters, taken from the upstream model card
(<https://huggingface.co/westlake-repl/SaProt_650M_AF2>, the `seq` variable in its usage example), where `#` marks
low-pLDDT residues:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"request": {"sequence": "M#EvVpQpL#VyQdYaKv"}}'
```

This is 9 residues (18 characters): `M#`, `Ev`, `Vp`, `Qp`, `L#`, `Vy`, `Qd`, `Ya`, `Kv`.

When you have 3Di tokens for a structure, for example from Foldseek, put each residue's 3Di token after its amino-acid
letter in place of `#`. The service takes the sequence as given and does not compute structure tokens itself.
Foldseek prints the 3Di string in uppercase; lowercase it before interleaving (for example `Ma` for residue `M` with
3Di `A`), otherwise the request returns `400`.

### 4.6 Response

A successful response has this shape:

```json
{
  "embedding": [0.0123, -0.0456, "..."],
  "embedding_dim": 1280,
  "model": "westlake-repl/SaProt_650M_AF2"
}
```

| Key | Purpose |
|---|---|
| `embedding` | The last hidden state of the model, averaged over the residue positions (the start and end tokens are not part of the average), as 1280 floats. |
| `embedding_dim` | Length of `embedding`, `1280`. |
| `model` | The Hugging Face model ID the service loaded. |

Only this pooled, sequence-level vector is returned. The service does not return per-residue embeddings, which tasks
such as per-residue or variant-effect analysis need; that is not supported by this AIM.

The model runs in fp16.

### 4.7 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | The request does not match the schema, for example it lacks the `request` wrapper or the `sequence` field, the body is not valid JSON, `sequence` is empty, `sequence` is not strictly paired amino-acid and 3Di characters (odd length, plain amino-acid sequence), `sequence` contains a letter outside the model vocabulary (`B J O U X Z`, `b j o u x z`), or `sequence` has more than 1022 residues (the length message states the limit and the received count). | No. Correct the request. |
| `503` | Returned by `GET /health` when the service cannot reach its own health check. | Retry later. |
| `500` | Inference failed on the server. The response carries a generic message; details are in the container log only. | Not without a change to the request or the deployment. |

The server-side request timeout is 60 seconds.

### 4.8 What to know before you deploy

**The service runs on one accelerator.** The profiles, `bentoml-mi300x-fp16-tp1-throughput` and
`bentoml-mi355x-fp16-tp1-throughput`, each use one accelerator. On a host with several accelerators, restrict the
container to the accelerator it should use. To serve more sequences per second, run more replicas.
The image sets `AIM_METRIC=throughput`, so that automatic profile selection finds the throughput profiles.

**The service accepts connections only once it is ready.** At startup it loads the tokenizer and the model weights
and moves the model to the accelerator. Use `GET /health` (or `GET /readyz`) as the readiness probe.

**The weights are downloaded at the first start.** The checkpoint is fetched from Hugging Face. The service stores it under `AIM_CACHE_PATH` (default `/workspace/model-cache`). Mount a persistent
volume there, as described in section 2, so a restart does not download it again.

**The service needs an accelerator.** If none is visible to the container, the service does not fall back to the CPU.
It logs `RuntimeError: SaProt 650M AF2 requires a ROCm GPU ...` and `Application startup failed`, but the container does
not exit: BentoML retries the service initialisation about every 10 seconds, and `/health` stays unreachable. Use a
readiness probe on `/health` (not process exit) to detect this.

**This model does not need a Hugging Face token.** The weights are public and not gated.

**Benchmark metric.** `aim-runtime benchmark` reports `sequences_per_sec` = 1 / mean latency. Requests are sent
sequentially (concurrency 1), per length bucket, and the latency includes the HTTP round trip and tokenization.
Warm-up requests are excluded. If any timed request fails, the benchmark is reported as failed.

### 4.9 Model environment variables

| Variable | Profile value | Effect |
|---|---|---|
| `AIM_CACHE_PATH` | not set, which means `/workspace/model-cache` | Directory where the weights are cached. |

The service always loads `westlake-repl/SaProt_650M_AF2`. Do not set `AIM_MODEL_ID`: the runtime reserves it for
model selection, and together with `AIM_ID` it stops the container at startup (`Cannot set both AIM_ID and
AIM_MODEL_ID`), so it cannot be used to load other weights.
