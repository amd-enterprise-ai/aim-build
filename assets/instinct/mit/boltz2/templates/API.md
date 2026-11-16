<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Readiness probe built into BentoML. See section 4.1. |
| `GET` | `/health` | The same readiness answer on the path that all AIMs share. It forwards to `/healthz`. |
| `GET` | `/v1/models` | Lists the served model, `mit/boltz2`, in the OpenAI list format. |
| `POST` | `/v1/inference` | Predict the structure of one biomolecular complex. |

The service is ready when `/health` answers HTTP 200. See section 4.9 for the start-up time.

### 4.3 Request body

The request is `POST /v1/inference` with a JSON body. The whole request is nested under a top-level `data` object:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sequences` | list of objects | required | The molecules of the complex. See the entity format below. |
| `name` | string | `"prediction"` | Name of the prediction. The service replaces each character that is not a letter, a digit, `-` or `_` with `_`, and removes `_` from the start and the end. An empty result gives `prediction`. |
| `output_format` | string | `"mmcif"` | `"mmcif"` or `"pdb"`. Other values give HTTP 400. |
| `recycling_steps` | integer | `3` | Recycling steps of the trunk. Limit 10. |
| `sampling_steps` | integer | `200` | Diffusion sampling steps. Limit 500. |
| `diffusion_samples` | integer | `1` | Number of structures to sample. Limit 4. |
| `max_parallel_samples` | integer | `5` | Samples that run in one batch. Limit 5. |
| `step_scale` | number | not set | Diffusion step scale. Not set gives the model default. |
| `seed` | integer | not set | Seed for a repeatable prediction. Range 0 to 4294967295. See the section on the seed below. |
| `use_msa_server` | boolean | `false` | Build the MSA with an MSA server. See section 4.8. |
| `msa_pairing_strategy` | string | `"greedy"` | `"greedy"` or `"complete"`. Only used with an MSA. |
| `constraints` | list of objects | not set | Boltz constraints, in the Boltz input format. |
| `templates` | list of objects | not set | Structure templates. Their `cif` or `pdb` value is inline text. |
| `properties` | list of objects | not set | Boltz properties. The only property that the service serves is `affinity`. See section 4.11. |
| `affinity_mw_correction` | boolean | `false` | Add the molecular weight correction to the affinity value. Used only with an `affinity` property. |
| `sampling_steps_affinity` | integer | `200` | Diffusion sampling steps of the affinity pass. Limit 500. Used only with an `affinity` property. |
| `diffusion_samples_affinity` | integer | `5` | Diffusion samples of the affinity pass. Limit 5. Used only with an `affinity` property. |
| `method` | string | not set | Boltz method label, for example `x-ray diffraction` or `md`. Boltz accepts a fixed list of names, in any letter case. Other values give HTTP 400. |
| `version` | integer | `1` | Version of the Boltz input format. Only `1` is accepted. |

Each entry of `sequences` has one key, the molecule type, and an object as value. The format is the input format of
Boltz:

| Molecule type | Fields |
|---|---|
| `protein` | `id`, `sequence`, optional `msa`. |
| `dna`, `rna` | `id`, `sequence`. |
| `ligand` | `id`, and either `smiles` or `ccd` (one Chemical Component Dictionary code). `smiles` must be a string. Another type gives HTTP 400. |

`id` is a chain identifier, or a list of identifiers for identical chains.

A protein chain without `msa` runs without an alignment unless `use_msa_server` is `true`.

#### Inline content only

The request holds file content, not file paths. `msa` is either `"empty"` or the text of an A3M alignment that starts
with `>`. A template `cif` or `pdb` value is the mmCIF or PDB text. The service answers HTTP 400 for a value that is a
file path.

An A3M alignment must have at least one sequence line. Each sequence line holds upper-case residue letters and `-` for
gaps, and lower-case letters for inserts. The aligned length of each line (without the lower-case letters) must equal
the length of the protein `sequence`. The service answers HTTP 400 for another A3M.

#### Seed

With `seed`, the service seeds all random number generators for the request and uses deterministic algorithms for it.
The same request with the same `seed` gives the same result, within one service process. The result is not guaranteed to
be the same in another service process, with another image, or on another accelerator. Without `seed`, the prediction
is not seeded and the service does not use deterministic algorithms. The response has the key `seed` (`null` without a
seed). Deterministic algorithms can make a prediction slower.

The first request with a `seed` after the service started takes about 1 minute longer than the later ones, because the
service compiles the model once more for the deterministic setting. A later request with a `seed` can compile once
more for each new input size in this setting. Requests without a `seed` do not compile after the start-up.

#### Limits

The service refuses a request above these limits with HTTP 422, before any computation starts:

| Limit | Value |
|---|---|
| Tokens of the whole request | 2364 |
| `diffusion_samples` | 4 |
| `recycling_steps` | 10 |
| `sampling_steps` | 500 |
| `max_parallel_samples` | 5 |
| `sampling_steps_affinity` | 500 |
| `diffusion_samples_affinity` | 5 |
| Heavy atoms of the affinity binder | 128 |

A token is one residue or nucleotide of a polymer chain, counted for each chain copy, or one heavy atom of a ligand
given as `smiles`. A ligand given as a `ccd` code is not counted.

### 4.4 Example request

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "name": "trp_cage",
      "sequences": [
        {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}}
      ],
      "output_format": "mmcif"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={
        "data": {
            "name": "trp_cage",
            "sequences": [{"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}}],
            "output_format": "mmcif",
        }
    },
    timeout=1800,
)
response.raise_for_status()
result = response.json()
print(result["structures"][0]["content"][:200])
```

### 4.5 Request with a ligand

A `ligand` entry carries `smiles` or `ccd` instead of `sequence`. This request folds the protein together with toluene:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "sequences": [
        {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
        {"ligand": {"id": "L", "smiles": "Cc1ccccc1"}}
      ]
    }
  }'
```

### 4.6 Request with a precomputed MSA

To use an alignment that you have, give its text in `msa`. The alignment is inline A3M text. The service does not
contact an MSA server:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "sequences": [
        {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS", "msa": ">query\nNLYIQWLKDGGPSSGRPPPS\n"}}
      ]
    }
  }'
```

#### Constraints, templates and properties

These fields use the input format of Boltz. Each example below is a request body that you send to
`POST /v1/inference`, for example with `curl http://localhost:8000/v1/inference -H "Content-Type: application/json" -d @request.json`.

A `contact` constraint asks for a distance of at most `max_distance` angstrom between two tokens. A token is
`[chain id, residue number]`, and the residue number starts at 1:

```json
{
  "data": {
    "name": "trp_cage_contact",
    "sequences": [{"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}}],
    "constraints": [{"contact": {"token1": ["A", 1], "token2": ["A", 20], "max_distance": 8}}],
    "sampling_steps": 10,
    "recycling_steps": 1
  }
}
```

A template is the text of a PDB file (`pdb`) or an mmCIF file (`cif`), and `chain_id` names the chain of your
`sequences` that it describes. The text is inline. The service does not read a path. The file must list the chain
sequence: `SEQRES` records in a PDB file, `_entity_poly_seq` in an mmCIF file. A file from the PDB has them. This Python
code sends the file `template.pdb`:

```python
import requests

with open("template.pdb") as template_file:
    template = template_file.read()

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={
        "data": {
            "name": "peptide_with_template",
            "sequences": [{"protein": {"id": "A", "sequence": "NLYIQW"}}],
            "templates": [{"pdb": template, "chain_id": "A"}],
            "sampling_steps": 10,
            "recycling_steps": 1,
        }
    },
    timeout=1800,
)
response.raise_for_status()
```

A `properties` entry asks for the affinity of a ligand. This one names the ligand `L` as the binder. See section 4.11
for the rules of the request and for the response:

```json
{
  "data": {
    "name": "trp_cage_ligand",
    "sequences": [
      {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
      {"ligand": {"id": "L", "smiles": "CCO"}}
    ],
    "properties": [{"affinity": {"binder": "L"}}],
    "sampling_steps": 10,
    "recycling_steps": 1
  }
}
```

### 4.7 Response

A successful response has this shape:

```json
{
  "structures": [
    {"record_id": "trp_cage_model_0", "format": "mmcif", "content": "data_...\n#\n..."}
  ],
  "confidence": {
    "trp_cage_model_0": {"confidence_score": 0.9, "ptm": 0.8, "iptm": 0.0, "complex_plddt": 0.9, "...": "..."}
  },
  "timing": {"runtime_s": 3.2},
  "seed": null,
  "runtime": {"mode": "affinity", "optimizations": "on", "...": "..."}
}
```

| Key | Purpose |
|---|---|
| `structures` | One entry per sample, best confidence first. `record_id` is `<name>_model_<rank>`, where `<name>` is the request `name` after the clean-up in section 4.3 and rank 0 is the best sample. `content` holds the mmCIF or PDB text. |
| `confidence` | Boltz-2 confidence metrics of each sample, keyed by `record_id`. `complex_plddt` is on a 0 to 1 scale. |
| `timing` | `{"runtime_s": <float>}`. The prediction time in the service. With an affinity pass, it also has `affinity_s`, the time of that pass. `runtime_s` includes it. |
| `seed` | The `seed` of the request, or `null`. |
| `affinity` | Only for a request with an `affinity` property. See section 4.11. |
| `runtime` | The serving mode (`mode`), the performance setting that answered the request, and the state of the start-up compilation. For diagnosis. |

### 4.8 Status codes and the MSA server

| Code | Meaning | Retry |
|---|---|---|
| `200` | Success. | — |
| `400` | The request is not valid, for example an unknown `output_format`, a file path in `msa`, `use_msa_server` on a deployment without an MSA server, or an affinity request that Boltz does not accept. See section 4.11. | No. Correct the request. |
| `422` | The request is above a limit. See section 4.3. | No. Reduce the request. |
| `429` | The queue is full. Each accelerator takes 3 requests at a time: one runs and two wait. | Yes. The client must retry after a delay. |
| `500` | The prediction failed. The message is generic. The service log holds the detail. | Not with the same request. |
| `503` | The MSA server did not answer within its time budget, or the request waited so long in the queue that it can no longer finish. The `Retry-After` header gives the interval. | Yes. |
| `504` | The request passed `BOLTZ2_REQUEST_TIMEOUT_SECONDS`. The service answers before this in normal use (see section 4.9). | Not with the same request. |

Every error answer of `POST /v1/inference` has the JSON body `{"error": "<text>"}`, one object. A request to an unknown path gets the plain text answer `Not Found` from BentoML. The text of a `400` or `422` answer starts with `Invalid request.`. For an invalid request, the text has one short entry for each field, as `<field>: <message>`. It does not repeat the request. The text of a `500` answer is generic. The service log has the full detail of every error.

**With `use_msa_server: true`, the protein sequences of the request are sent to an MSA server.** The default is the
public ColabFold server at `https://api.colabfold.com`, which is a third-party service. Set
`BOLTZ2_MSA_SERVER_URL` to the address of an MSA server that you run, to keep sequences on your own network. Set it to
an empty value to turn the MSA server off: a request with `use_msa_server: true` then gets HTTP 400. A request with
`use_msa_server: false`, the default, never contacts an MSA server.

### 4.9 What to know before you deploy

**The model weights are downloaded at the first start.** The structure checkpoint (2.3 GB) and the molecule
library go to `$AIM_CACHE_PATH`. The library is 1.8 GB, and its 1.9 GB archive stays in the cache. This is 6.0 GB.
The default profile runs in affinity mode. This mode also downloads the affinity checkpoint (2.1 GB): 8.0 GB in all.
Mount a persistent volume there, so that later starts do not download again.

**The default profile runs in affinity mode, and this mode costs more than structure mode.** The service downloads a
second checkpoint and keeps a second model loaded. It uses more accelerator memory, and the start-up is longer. A request
without an `affinity` property pays nothing extra. A deployment that needs only structure prediction can run in structure
mode: see section 4.11.

To use weights that you staged in that directory, set `BOLTZ2_DOWNLOAD_ASSETS` to `false`. The start then fails if a
file is missing. If downloads are on, the directory is read-only and a file is missing, the start also fails once.
The message names the directory. Mount it writable, or stage the weights. This model does not need a Hugging Face
token.

**The service needs about 4 minutes to become ready when the cache holds the weights, and longer at a first start.**
The download time depends on the network. In structure mode, a measured cold start took about 6.5 minutes. Affinity
mode adds a 2.1 GB download. In affinity mode, a measured cold start took about 7 min. With `BOLTZ2_OPTIMIZATIONS=on`,
the default, the service compiles the model at start-up with test predictions, and `/health` and `/healthz` answer HTTP 200 only after that. The
first request after the service is ready runs at normal speed, with one exception: the first request above 2048 tokens
compiles the model once more, which takes about 5 seconds extra. Later requests run at normal speed. Set the
readiness probe to allow for the start-up time.

**`BOLTZ2_OPTIMIZATIONS` is the one performance setting.** `on`, the default, uses the optimized kernels and the
start-up compilation. `off` runs the upstream Boltz code without these, which is slower and has no start-up
compilation.

**Requests wait in a queue.** Each accelerator runs one prediction at a time. The service accepts 3 requests per
accelerator: one runs and two wait. A request over that gets HTTP 429 at once, and the client must retry it after a
delay. Send requests one after the other, or use more replicas.

**Each request has its own deadline.** The deadline counts from the arrival of the request and ends before
`BOLTZ2_REQUEST_TIMEOUT_SECONDS`. A request that waited so long in the queue that it can no longer finish gets HTTP
503 with `Retry-After`, and the service does not start it. This happens when the request waits longer than
`BOLTZ2_REQUEST_TIMEOUT_SECONDS` minus 30 seconds (kept to send the answer) minus `BOLTZ2_INFERENCE_RESERVE_SECONDS`:
more than 1170 seconds with the profile values (1800 and 600). With `use_msa_server: true`, the MSA stage has its own time
limit (`BOLTZ2_MSA_TIMEOUT_SECONDS`) inside the request deadline, and the time that remains is kept for the
prediction. Set a client timeout that is not shorter than `BOLTZ2_REQUEST_TIMEOUT_SECONDS`, or the client gives up
before the service answers.

**The prediction time grows with the size of the complex.** A request of a few hundred tokens takes seconds. A request
near the limit of 2364 tokens takes minutes.

### 4.10 Environment variables

| Variable | Default | Effect |
|---|---|---|
| `BOLTZ2_MODE` | `structure` (the profile sets `affinity`) | `structure` or `affinity`. See section 4.11. |
| `BOLTZ2_OPTIMIZATIONS` | `on` | `on` or `off`. See section 4.9. |
| `BOLTZ2_AFFINITY_COMPILE` | `off` | `on` or `off`. `on` also compiles the affinity model. See section 4.11. |
| `BOLTZ2_MSA_SERVER_URL` | `https://api.colabfold.com` | MSA server for `use_msa_server` requests. An empty value turns the MSA server off. |
| `BOLTZ2_REQUEST_TIMEOUT_SECONDS` | `1800` (set by the profile) | Whole-request cap. The request deadline is set inside it. |
| `BOLTZ2_MSA_TIMEOUT_SECONDS` | `300` | Cap on the MSA-server work of one request. |
| `BOLTZ2_INFERENCE_RESERVE_SECONDS` | `600` | Time of the request deadline that the MSA stage cannot use, and the least time that a waiting request must have left to be started. |
| `BOLTZ2_MAX_CONCURRENCY` | `3` (set by the profile) | Requests in the service before it answers 429. The value `0` means 3 per accelerator. |
| `BOLTZ2_ACCELERATOR` | `gpu` (set by the profile) | `gpu` or `cuda` stops the start-up with an error when the container sees no GPU. `auto` uses a GPU when there is one and the CPU if not. `cpu` uses the CPU. |
| `BOLTZ2_DOWNLOAD_ASSETS` | `true` | Download the weights at start-up when they are not in `$AIM_CACHE_PATH`. |
| `AIM_CACHE_PATH` | `/workspace/model-cache` | Directory of the model weights. |

The profile sets `BOLTZ2_MODE` (`affinity`), `BOLTZ2_REQUEST_TIMEOUT_SECONDS` (1800), `BOLTZ2_MAX_CONCURRENCY` (3) and `BOLTZ2_ACCELERATOR` (`gpu`). A value that you pass with `docker run -e` does not change them. The profile does not set the other variables in this table, so `docker run -e` changes them.

### 4.11 Affinity prediction

Boltz-2 can predict the binding affinity of one ligand to the rest of the complex. The profile sets `BOLTZ2_MODE` to
`affinity`: the service loads the structure model and the affinity model, and serves both kinds of request. A request
without an `affinity` property gets the same answer as in structure mode. `BOLTZ2_MODE` is not case sensitive. Any
other value than `structure` or `affinity` stops the start-up with an error that lists the valid values.

`docker run -e BOLTZ2_MODE=structure` has no effect, because the profile value wins. To run in structure mode, mount
a copy of the profile file with `BOLTZ2_MODE: structure` over
`/workspace/aim-runtime/profiles/mit/boltz2/bentoml-mi300x-bf16-tp1-latency.yaml` in the container (`docker run -v`). The service then loads
the structure model only, downloads 2.1 GB less, and answers a request with an `affinity` property with HTTP 400 and the text
`affinity is not served by this profile`.

The affinity model costs the following, compared with structure mode: 2.1 GB more weights, more start-up time for the
warm-up, and more accelerator memory (the model stays resident). A request without an `affinity` property pays nothing
extra at run time. A request with one adds the affinity pass, `affinity_s` in `timing`. This takes about 3 to 7 seconds
when the cache is warm, for the complexes that were measured (up to about 1500 tokens).

With `BOLTZ2_OPTIMIZATIONS=on`, the start-up warm-up runs one complex with a ligand through both models. It compiles the
structure model, and it runs the affinity model with the triangle kernels and a CUDA graph of its score model, so the
first request of each kind does not compile. With `BOLTZ2_OPTIMIZATIONS=off`, only this one complex runs, in eager
mode. In structure mode with `BOLTZ2_OPTIMIZATIONS=off`, there is no warm-up.

`BOLTZ2_AFFINITY_COMPILE=on` is an option for `BOLTZ2_MODE=affinity` with `BOLTZ2_OPTIMIZATIONS=on`. It is `off` by
default, and the profile does not set it, so `docker run -e BOLTZ2_AFFINITY_COMPILE=on` turns it on. The service
then also compiles the Transition and token-transformer modules of the affinity model with TorchInductor. The warm-up
runs 3 affinity sampling steps and a second affinity request at the crop size, so the start-up takes longer. Set the
readiness probe to allow for it. `runtime.affinity_compile_status` shows the compile status of the affinity model.
The compiles of the affinity model appear in `runtime.warmup` under the labels `affinity_<selector>`. In structure mode, or with
`BOLTZ2_OPTIMIZATIONS=off`, the service ignores the option and writes one log line. A value other than `on` or `off` stops the start-up.

#### The affinity request

Add one `affinity` entry to `properties`. `binder` is the `id` of one ligand in `sequences`:

```json
{
  "data": {
    "name": "trp_cage_aspirin",
    "sequences": [
      {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
      {"ligand": {"id": "B", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}}
    ],
    "properties": [{"affinity": {"binder": "B"}}],
    "affinity_mw_correction": false,
    "sampling_steps_affinity": 200,
    "diffusion_samples_affinity": 5
  }
}
```

The service predicts the structure first. It takes the best sample (rank 0) and runs the affinity model on it, as
`boltz predict` does. The affinity pass uses 5 recycling steps and one parallel sample. It crops the complex to 256
tokens and 2048 atoms around the binder, and it does not use templates. The settings `recycling_steps`,
`sampling_steps`, `diffusion_samples` and `step_scale` of the request apply to the structure pass. The `step_scale`
also applies to the affinity pass.

The binder must follow these rules:

* The request has one `affinity` entry, and `binder` is one `id` string.
* The binder is a ligand with one chain, one copy and one residue.
* The service refuses a ligand with a list of `id` values for more than one copy. It also refuses a `ccd` list of
  more than one code, and another ligand entry with the same `smiles` or `ccd`.
* The binder is a ligand, not a polymer chain. Boltz supports the affinity of small molecules to protein targets.
  For an RNA or DNA target, the values are not reliable.
* The `smiles` of the binder is a string that is not empty.
* A binder that is a `smiles` ligand has at most 128 heavy atoms in its largest fragment. Above that, the service
  answers HTTP 422.
* Boltz advises against more than 56 atoms, which was the maximum in training.

The service answers HTTP 400 for a request that breaks a rule, except the atom limit (HTTP 422). The text has the
reason, for example `properties.affinity.binder must be the id of a ligand in sequences` or
`the affinity binder must be a ligand with one copy`.

#### The affinity response

The key `affinity` holds the values that `boltz predict` writes to its `affinity_<id>.json` file:

```json
{
  "affinity": {
    "affinity_pred_value": 0.42,
    "affinity_probability_binary": 0.87,
    "affinity_pred_value1": 0.40,
    "affinity_probability_binary1": 0.85,
    "affinity_pred_value2": 0.44,
    "affinity_probability_binary2": 0.89
  },
  "timing": {"runtime_s": 41.7, "affinity_s": 22.5}
}
```

| Key | Meaning |
|---|---|
| `affinity_pred_value` | Predicted binding affinity, as the base-10 logarithm of the IC50 in micromolar. A lower value means a stronger binder. Use it to compare known binders, for example in hit-to-lead work, and not to compare non-binders. |
| `affinity_probability_binary` | Probability, from 0 to 1, that the ligand binds. Use it to separate binders from decoys, for example in a screen. |
| `affinity_pred_value1`, `affinity_probability_binary1`, `affinity_pred_value2`, `affinity_probability_binary2` | The values of the first and the second member of the model ensemble. The two main keys are the values of the ensemble. |

The values in the example are for illustration. If the complex cannot be featurized for the affinity model, the service
answers HTTP 400 and no structure is returned.
