<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Segment pancreas and pancreatic tumor in one CT volume. |
| `POST` | `/v1/models` | Report the served model and its 3-entry label map. |
| `GET` | `/healthz` | Check whether the service is ready to accept requests. |

`/v1/inference` and `/v1/models` accept `POST` only. `/healthz` accepts `GET`; see section 4.1.

### 4.3 Request body

`/v1/inference` takes three top-level fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | string | the served model | Model identifier. |
| `input` | object | required | Holds the input and output paths. |
| `parameters` | object | empty | Returned unchanged in the response metadata. |

`input` holds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_path` | string | required | Absolute path to the input NIfTI CT volume, **inside the container**. |
| `output_dir` | string | `$AIM_OUTPUT_PATH` (default `/workspace/output`) | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body.

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask rather than the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/ct-data:/workspace/data` for the input and `-v /path/to/output:/workspace/output` for the masks to the
`docker run` command of section 1.1. Without the output mount, the masks stay in the container and are lost when it is
removed. In the Kubernetes deployment of section 3, add both volumes to the existing `volumeMounts` and `volumes`
lists, and create the two claims like the cache claim in `pvc.yaml`:

```yaml
volumeMounts:
  - name: input-data
    mountPath: /workspace/data
  - name: output-data
    mountPath: /workspace/output
volumes:
  - name: input-data
    persistentVolumeClaim:
      claimName: minimal-aim-input-data
  - name: output-data
    persistentVolumeClaim:
      claimName: minimal-aim-output-data
```

The examples below use `/workspace/data` for input and `/workspace/output` for output, which are the defaults.
`AIM_INPUT_PATH` and `AIM_OUTPUT_PATH` change them; a path outside those roots is rejected. The file name is a
placeholder. Any NIfTI CT volume works; the examples use a case from the [Medical
Segmentation Decathlon](http://medicaldecathlon.com/) Task07_Pancreas set (CC-BY-SA 4.0), which is the bundle's own
training source.

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/abdomen_001.nii.gz",
      "output_dir": "/workspace/output"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"input": {"image_path": "/workspace/data/abdomen_001.nii.gz", "output_dir": "/workspace/output"}},
    timeout=1800,
)

print(response.json()["output"]["output_path"])
```

### 4.5 Response

```json
{
  "model": "monai/pancreas_ct_dints_segmentation",
  "output": {
    "output_path": "/workspace/output/abdomen_001/abdomen_001_trans.nii.gz"
  },
  "metadata": {
    "device": "cuda:0",
    "used_amp": true,
    "parameters": {},
    "optimizations": {
      "compile": false,
      "compile_mode": null,
      "dynamic": null,
      "autocast": true,
      "gpu_transforms": true
    }
  }
}
```

The mask is written to `<output_dir>/<stem>/<stem>_trans.nii.gz`, where `<stem>` is the input filename without its
extensions. Read the file from the volume mounted at `/workspace/output`, as described in section 4.4. The mask has
three values: `0` background, `1` pancreas, `2` pancreatic tumor. A later request for the same input writes this same
path and overwrites the earlier mask. To keep both, give each request its own `output_dir` under `/workspace/output`.

`/v1/models` answers with the served model and its 3-entry label map:

```json
{
  "object": "list",
  "data": [
    {
      "id": "monai/pancreas_ct_dints_segmentation",
      "object": "model",
      "owned_by": "MONAI",
      "channel_def": {
        "0": "background",
        "1": "pancreas",
        "2": "pancreatic tumor"
      }
    }
  ]
}
```

### 4.6 What to know before you deploy

**The model expects abdominal CT.** CT is Computed Tomography, an X-ray imaging method that gives a 3D volume in
NIfTI format (`.nii.gz`). Preprocessing clips intensity and resamples to the model's native spacing. Other modalities,
such as MRI, give meaningless output.

**DiNTS uses a searched architecture.** The network topology is determined by a Neural Architecture Search result
stored in `models/search_code_18590.pt` alongside the weights. Both files are downloaded from Hugging Face at startup
if absent.

**The service always listens on port 8000.** The image sets `BENTOML_PORT=8000`, and the service does not read
`AIM_PORT`, so the `AIM_PORT=8080` example in section 1.2 does not work for this AIM: nothing listens on port 8080.
Leave `AIM_PORT` unset and map the host port you want to 8000, for example `-p 8080:8000`. In Kubernetes, keep
`containerPort` and `targetPort` at 8000, as in section 3.

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `PANCREAS_DINTS_TIMEOUT_SECONDS` changes it.

**The weights are fetched at startup when absent.** They are downloaded from `MONAI/pancreas_ct_dints_segmentation` on
Hugging Face and kept in `$AIM_CACHE_PATH`. Mount a persistent volume there, as described in section 2, or they are
downloaded again on every start. No Hugging Face token is required.

### 4.7 Optimization environment variables

A profile can set these variables in its `env_vars`. A value that the profile sets overrides the code default below
and any `-e`. The `bentoml-mi300x-fp16-tp1-latency` profile sets `PANCREAS_DINTS_AUTOCAST` and
`PANCREAS_DINTS_GPU_TRANSFORMS` to `true`, and the example response in section 4.5 comes from that profile. To see
what another profile sets, run the `dry-run` command of section 6.3 with `AIM_PROFILE_ID` set, and read its
`env_vars`.

**Leave `PANCREAS_DINTS_COMPILE` off.** `torch.compile` does not handle the DiNTS network on ROCm: with compile on, the
first request fails with HTTP 500.

| Variable | Code default | Effect |
|---|---|---|
| `PANCREAS_DINTS_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `PANCREAS_DINTS_COMPILE_MODE` | `max-autotune` | Compile mode used when `PANCREAS_DINTS_COMPILE` is on. |
| `PANCREAS_DINTS_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `PANCREAS_DINTS_AUTOCAST` | `false` | Run inference in mixed precision. |
| `PANCREAS_DINTS_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `PANCREAS_DINTS_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `PANCREAS_DINTS_LOG_LEVEL` | `INFO` | Log level of the service. |

A variable that the profile does not set can be changed with `-e`, as in section 1.2. To change a variable that the
profile sets, write a custom profile with the changed `env_vars`, mount its directory at
`/workspace/aim-runtime/profiles/custom`, and set `AIM_PROFILE_ID` to its filename without the `.yaml` extension.
