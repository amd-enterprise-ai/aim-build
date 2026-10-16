<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Segment one CT volume (automatic or prompted). |
| `POST` | `/v1/models` | Report the served model and its full 133-entry label map. |
| `GET` | `/healthz` | Check whether the service is ready to accept requests. |

`/v1/inference` and `/v1/models` accept `POST` only. `/healthz` accepts `GET`; see section 4.1.

### 4.3 Request body

`/v1/inference` takes three top-level fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | string | the served model | Model identifier. |
| `input` | object | required | Holds the image path, optional prompts, and optional output directory. |
| `parameters` | object | empty | Returned unchanged in the response metadata. |

`input` holds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_path` | string | required | Absolute path to the input NIfTI volume, **inside the container**. |
| `label_prompt` | integer array | none | Class IDs to segment. Omit for automatic 117-class mode. |
| `points` | integer\[3\] array | none | Foreground/background click points in original-image xyz voxel coordinates. Requires `label_prompt` of length 1. |
| `point_labels` | integer array | none | Click type for each point: `1` = foreground, `0` = background, `-1` = ignore. Must match `points` length. |
| `output_dir` | string | `$AIM_OUTPUT_PATH` (default `/workspace/output`) | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body.

**VISTA-3D has three inference modes, selected by which prompt fields are present:**

| Mode | How to trigger | What the model segments |
|---|---|---|
| **Automatic** | omit all prompt fields | All 117 classes simultaneously (`everything_labels`). |
| **Class-prompt** | supply `label_prompt` only | The listed class IDs only (e.g. `[3]` for spleen). |
| **Point-prompt** | supply `points` + `point_labels` + `label_prompt` of length 1 | The single class in `label_prompt`, guided by the click points. |

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask rather than the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/ct-data:/workspace/data` to the `docker run` command of section 1.1, or mount the same directory in
the Kubernetes deployment of section 3. The examples below use `/workspace/data` for input and `/workspace/output` for
output, which are the defaults. `AIM_INPUT_PATH` and `AIM_OUTPUT_PATH` change them; a path outside those roots is
rejected. The file name is a placeholder. Any NIfTI CT volume works; the examples use a case from the [Medical
Segmentation Decathlon](http://medicaldecathlon.com/) Task09_Spleen set (CC-BY-SA 4.0), which is also what this AIM's
harness evaluates against.

**Automatic mode** — segments all 117 classes:

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

**Class-prompt mode** — segment spleen only (class ID 3):

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/abdomen_001.nii.gz",
      "label_prompt": [3],
      "output_dir": "/workspace/output"
    }
  }'
```

**Point-prompt mode** — segment spleen using two click points:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/abdomen_001.nii.gz",
      "label_prompt": [3],
      "points": [[138, 245, 18], [271, 343, 27]],
      "point_labels": [1, 0],
      "output_dir": "/workspace/output"
    }
  }'
```

The same request in Python (automatic mode):

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
  "model": "monai/vista3d",
  "output": {
    "output_path": "/workspace/output/abdomen_001/abdomen_001_trans.nii.gz"
  },
  "metadata": {
    "device": "cuda:0",
    "used_amp": false,
    "parameters": {},
    "optimizations": {
      "compile": false,
      "compile_mode": null,
      "dynamic": null,
      "autocast": false,
      "gpu_transforms": false
    }
  }
}
```

The mask is written to `<output_dir>/<stem>/<stem>_trans.nii.gz`, where `<stem>` is the input filename without its
extensions. Read the file from the mounted directory.

`/v1/models` answers with the served model and the 133-entry label map covering the background channel and 132
anatomical structures:

```json
{
  "object": "list",
  "data": [
    {
      "id": "monai/vista3d",
      "object": "model",
      "owned_by": "MONAI",
      "channel_def": {
        "0": "background",
        "1": "liver",
        "2": "spleen",
        "3": "pancreas",
        "...": "..."
      }
    }
  ]
}
```

The full 133-entry `channel_def` is served at runtime. Fetch it with a `/v1/models` call.

### 4.6 What to know before you deploy

**The model expects abdominal CT volumes.** CT is Computed Tomography, an X-ray imaging method that gives a 3D
volume in NIfTI format (`.nii.gz`). Preprocessing resamples to isotropic spacing and reorients to RAS. Other
modalities, such as MRI, give meaningless output.

**Point coordinates are in original-image voxel space.** The `points` field takes `[x, y, z]` triples in the frame
of the input volume before any preprocessing. `point_labels` maps `1` = foreground click, `0` = background click,
`-1` = ignore.

**Point-prompt mode requires exactly one class in `label_prompt`.** Multi-class point-prompt is not supported — this
is a bundle-level constraint enforced by `Vista3dInferer`.

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `VISTA3D_TIMEOUT_SECONDS` changes it.

**The weights are fetched at startup when absent.** They are downloaded from `MONAI/vista3d` on Hugging Face and kept
in `$AIM_CACHE_PATH`. Mount a persistent volume there, as described in section 2, or they are downloaded again on
every start. A valid `HF_TOKEN` is required — the weights are governed by the NVIDIA OneWay Noncommercial License and
require accepting the license at `https://huggingface.co/MONAI/vista3d` before the token is issued.

### 4.7 Optimization environment variables

These are off by default. Switch them on to increase the startup time and decrease the inference time.

| Variable | Default | Effect |
|---|---|---|
| `VISTA3D_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `VISTA3D_COMPILE_MODE` | `max-autotune` | Compile mode used when `VISTA3D_COMPILE` is on. |
| `VISTA3D_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `VISTA3D_AUTOCAST` | `false` | Run inference in mixed precision. |
| `VISTA3D_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `VISTA3D_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `VISTA3D_LOG_LEVEL` | `INFO` | Log level of the service. |
