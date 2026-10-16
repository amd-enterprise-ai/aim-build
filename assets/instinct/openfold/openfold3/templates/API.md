<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The prediction endpoint

This AIM serves one prediction endpoint:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Predict the structure of one or more biomolecular queries. |

### 4.3 Request body

The whole request is nested under a top-level `data` object:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `queries` | object | required | Map of query name to `{"chains": [...]}`. |
| `seeds` | list of integers | `[42]` | Explicit random seeds. One prediction runs per seed. |
| `num_diffusion_samples` | integer | `1` | Diffusion samples per seed. |
| `num_model_seeds` | integer | not set | Generate this many seeds instead of listing them. Cannot be combined with `seeds`. |
| `use_msa_server` | boolean | `true` | Build the MSA with the ColabFold MSA server. |
| `use_templates` | boolean | `true` | Fetch PDB structure templates. Needs `use_msa_server: true`. |
| `output_format` | string | `"mmcif"` | `"mmcif"` or `"pdb"`. The service does not validate this field: only `"pdb"` gives PDB, and every other value gives mmCIF. |
| `include_atom_confidences` | boolean | `false` | Add per-atom confidences to the response. |

Each entry of `chains` describes one molecule:

| Field | Type | Purpose |
|---|---|---|
| `molecule_type` | string | `protein`, `dna`, `rna` or `ligand`. |
| `chain_ids` | list of strings | Chain identifiers, for example `["A"]`. |
| `sequence` | string | The residue or atom sequence. |
| `main_msa` | string or list of strings | Optional precomputed MSA as a3m text. |
| `paired_msa` | string or list of strings | Optional precomputed paired MSA as a3m text. |

The service rejects these combinations with HTTP 400, before any computation starts:

* `main_msa` or `paired_msa` together with `use_msa_server: true`. The server MSA would overwrite the inline one.
* `seeds` together with `num_model_seeds`. One generates seeds, the other lists them.
* `use_templates: true` together with `use_msa_server: false`. Templates are fetched through the same server.

Because `use_templates` defaults to `true`, a request that sets `use_msa_server: false` must also set
`use_templates: false`.

### 4.4 Example request

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "queries": {
        "ubiquitin": {
          "chains": [
            {
              "molecule_type": "protein",
              "chain_ids": ["A"],
              "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            }
          ]
        }
      },
      "seeds": [42],
      "num_diffusion_samples": 1,
      "output_format": "mmcif"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "data": {
            "queries": {
                "ubiquitin": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                        }
                    ]
                }
            },
            "seeds": [42],
            "num_diffusion_samples": 1,
            "output_format": "mmcif",
        }
    },
    timeout=600,
)

print(response.json()["structures"][0]["content"][:200])
```

### 4.5 Request with a precomputed MSA

To avoid the MSA server, give the alignment in the request and switch off both the server and the templates:

```bash
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "queries": {
        "ubiquitin": {
          "chains": [
            {
              "molecule_type": "protein",
              "chain_ids": ["A"],
              "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
              "main_msa": ">query\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n"
            }
          ]
        }
      },
      "use_msa_server": false,
      "use_templates": false
    }
  }'
```

### 4.6 Response

A successful response has this shape:

```json
{
  "error": false,
  "structures": [
    {
      "record_id": "ubiquitin_seed_42_sample_1",
      "format": "mmcif",
      "content": "data_ubiquitin\n#\n_entry.id ubiquitin\n..."
    }
  ],
  "confidence": {
    "ubiquitin_seed_42_sample_1": { "...": "..." }
  },
  "timing": {
    "ubiquitin_seed_42_sample_1": { "runtime_s": 42.0 }
  }
}
```

| Key | Purpose |
|---|---|
| `error` | `false` on success, `true` on a prediction failure. |
| `structures` | One entry per sample. `content` holds the mmCIF or PDB text. |
| `confidence` | OF3 confidence metrics, keyed by sample identifier. |
| `timing` | `{"runtime_s": <float>}` per sample, keyed by sample identifier. Always present, `{}` when OF3 emits no timing. |
| `atom_confidence` | Present only with `include_atom_confidences: true`. Per-atom `plddt` and per-token `pae` / `pde` matrices, keyed by sample identifier. The matrix size increases as the square of the token count. |

When a prediction fails, the service answers HTTP 200 with `"error": true`, a `message` field, and empty `structures` and
`confidence`. `timing` is then `{}`, because the service records a runtime only for a sample that produced a structure.

### 4.7 Status codes

| Code | Meaning | Retry |
|---|---|---|
| `200` with `"error": false` | Success. | — |
| `200` with `"error": true` | The prediction failed. `message` holds the reason. | No. The same request fails again. |
| `400` | The request combines mutually exclusive inputs. See section 4.3. | No. Correct the request. |
| `429` | Too many requests in flight. | Yes. |
| `503` | The MSA server did not answer within its budget, or the request queued too long. The `Retry-After` header gives the interval. | Yes. |

### 4.8 What to know before you deploy

**The `tpN` in a profile name is the number of data-parallel workers, not tensor parallelism.** OpenFold3 has no
tensor parallelism. A profile named `bentoml-mi300x-fp16-tp4-latency` runs four workers, each pinned to one
accelerator, and each holding its own copy of the model. The deployment then serves up to four predictions at the
same time. It does not split one prediction across four accelerators, and one prediction is not faster with a larger
profile.

Two costs grow with the worker count: host memory, which is roughly N times the single-accelerator footprint, and
cold start, because each worker compiles on its own first request per sequence length.

**`AIM_ACCELERATOR_COUNT` must be an integer that matches the profile.** The service reads the value to size its
worker pool and rejects `auto`.

**The default MSA path calls a public third-party server.** With `use_msa_server: true`, which is the default, the
service calls the public ColabFold server at `api.colabfold.com`. That server limits the rate per caller address, and
every worker in the deployment calls it from the same address, so a larger deployment is throttled sooner. Throttling
gives HTTP 503. For sustained use, set the service to use a self-hosted ColabFold or MMseqs2 server, or give
precomputed MSAs in the request as shown in section 4.5.

**The first request for each sequence length is slow.** The service compiles kernels for each distinct sequence
length, which takes tens of seconds. Later requests with the same length run at the compiled speed.

**The checkpoint is about 2.2 GB.** It is downloaded at the first start and kept in `$AIM_CACHE_PATH`. Mount a
persistent volume there, as described in section 2.

**This model does not need a Hugging Face token.**

### 4.9 Request budget environment variables

| Variable | Default | Effect |
|---|---|---|
| `OPENFOLD3_REQUEST_TIMEOUT_SECONDS` | `600` | Whole-request cap. |
| `OPENFOLD3_MSA_TIMEOUT_SECONDS` | `300` | Cap on MSA-server work. |
| `OPENFOLD3_INFERENCE_RESERVE_SECONDS` | `120` | Budget held back from the MSA stage for accelerator work. |
| `OPENFOLD3_MAX_CONCURRENCY` | `0` | In-flight requests before the service answers 429. The value `0` means "derive from the accelerator count", which gives 3 per accelerator. |

Set a client timeout no shorter than `OPENFOLD3_REQUEST_TIMEOUT_SECONDS`, or the client gives up before the service
answers.
