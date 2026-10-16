<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Segment the spleen in one CT volume, optionally guided by click points. |
| `POST` | `/v1/models` | Report the served model and its 2-entry label map. |
| `GET` | `/healthz` | Check whether the service is ready to accept requests. |

`/v1/inference` and `/v1/models` accept `POST` only. `/healthz` accepts `GET`; see section 4.1.

### 4.3 Request body

`/v1/inference` takes three top-level fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | string | the served model | Model identifier. |
| `input` | object | required | Holds the image path, optional click points, and optional output directory. |
| `parameters` | object | empty | Returned unchanged in the response metadata. |

`input` holds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_path` | string | required | Absolute path to the input NIfTI CT volume, **inside the container**. |
| `foreground_clicks` | integer\[3\] array | `[]` | Voxel coordinates `[x, y, z]` of foreground (spleen) click points in the original image frame. |
| `background_clicks` | integer\[3\] array | `[]` | Voxel coordinates `[x, y, z]` of background click points in the original image frame. |
| `output_dir` | string | `$AIM_OUTPUT_PATH` (default `/workspace/output`) | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body.

**The service has two modes, selected by whether click points are provided:**

| Mode | How to trigger | What the model does |
|---|---|---|
| **Automatic** | omit click fields (or pass empty arrays) | Segments the spleen without guidance, using zero-filled guidance channels. |
| **Interactive** | supply one or more foreground or background clicks | Refines the segmentation using DeepEdit's interactive correction path. |

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask rather than the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/ct-data:/workspace/data` to the `docker run` command of section 1.1, or mount the same directory in
the Kubernetes deployment of section 3. The examples below use `/workspace/data` for input and `/workspace/output` for
output, which are the defaults. `AIM_INPUT_PATH` and `AIM_OUTPUT_PATH` change them; a path outside those roots is
rejected. The file name is a placeholder. Any NIfTI CT volume works; the examples use a case from the [Medical
Segmentation Decathlon](http://medicaldecathlon.com/) Task09_Spleen set (CC-BY-SA 4.0), the same source
`monai.apps.DecathlonDataset` pulls from.

**Automatic mode** — segments the spleen without guidance:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/spleen_001.nii.gz",
      "output_dir": "/workspace/output"
    }
  }'
```

**Interactive mode** — segments with two foreground clicks and one background click:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/spleen_001.nii.gz",
      "foreground_clicks": [[20, 50, 30], [40, 60, 35]],
      "background_clicks": [[10, 10, 10]],
      "output_dir": "/workspace/output"
    }
  }'
```

The same request in Python (automatic mode):

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"input": {"image_path": "/workspace/data/spleen_001.nii.gz", "output_dir": "/workspace/output"}},
    timeout=1800,
)

print(response.json()["output"]["output_path"])
```

### 4.5 Response

```json
{
  "model": "monai/spleen_deepedit_annotation",
  "output": {
    "output_path": "/workspace/output/spleen_001/spleen_001_trans.nii.gz"
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
extensions. Read the file from the mounted directory. The mask has two values: `0` background, `1` spleen.

`/v1/models` answers with the served model and its 2-entry label map:

```json
{
  "object": "list",
  "data": [
    {
      "id": "monai/spleen_deepedit_annotation",
      "object": "model",
      "owned_by": "MONAI",
      "channel_def": {
        "0": "background",
        "1": "spleen"
      }
    }
  ]
}
```

### 4.6 What to know before you deploy

**The model expects abdominal CT.** CT is Computed Tomography, an X-ray imaging method that gives a 3D volume in
NIfTI format (`.nii.gz`). Preprocessing resamples and reorients to RAS. Other modalities, such as MRI, give
meaningless output.

**Click coordinates are in original-image voxel space.** Each `[x, y, z]` triple is in the frame of the input
volume before any preprocessing. The `AddGuidanceFromPointsDeepEditd` transform rescales them into the model's
internal 128³ frame.

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `SPLEEN_DEEPEDIT_TIMEOUT_SECONDS` changes it.

**The weights are fetched at startup when absent.** They are downloaded from `MONAI/spleen_deepedit_annotation` on
Hugging Face and kept in `$AIM_CACHE_PATH`. Mount a persistent volume there, as described in section 2, or they are
downloaded again on every start. No Hugging Face token is required.

### 4.7 Optimization environment variables

These are off by default. Switch them on to increase the startup time and decrease the inference time.

| Variable | Default | Effect |
|---|---|---|
| `SPLEEN_DEEPEDIT_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `SPLEEN_DEEPEDIT_COMPILE_MODE` | `max-autotune` | Compile mode used when `SPLEEN_DEEPEDIT_COMPILE` is on. |
| `SPLEEN_DEEPEDIT_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `SPLEEN_DEEPEDIT_AUTOCAST` | `false` | Run inference in mixed precision. |
| `SPLEEN_DEEPEDIT_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `SPLEEN_DEEPEDIT_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `SPLEEN_DEEPEDIT_LOG_LEVEL` | `INFO` | Log level of the service. |
