<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Segment one brain MRI volume. |
| `POST` | `/v1/models` | Report the served model and its label map. |
| `GET` | `/healthz` | Check whether the service is ready to accept requests. |

`/v1/inference` and `/v1/models` accept `POST` only. `/healthz` accepts `GET`; see section 4.1.

### 4.3 Request body

`/v1/inference` takes three top-level fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | string | the served model | Model identifier. |
| `input` | object | required | Holds the input and output paths. See below. |
| `parameters` | object | empty | Returned unchanged in the response metadata. |

`input` holds:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image_path` | string | required | Path of the input NIfTI volume, **inside the container**. |
| `output_dir` | string | `$AIM_OUTPUT_PATH` (default `/workspace/output`) | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body. `/healthz` needs no request body.

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask and not the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/mri-data:/workspace/data` to the `docker run` command of section 1.1, or mount the same directory in
the Kubernetes deployment of section 3. The examples below use `/workspace/data` for input and `/workspace/output` for
output, which are the defaults. `AIM_INPUT_PATH` and `AIM_OUTPUT_PATH` change them; a path outside those roots is
rejected.

The file name in the examples stands for a volume of your own. The `mni305` part of it is a reminder: the service
accepts only a volume that is already registered to the MNI305 atlas, as section 4.6 describes.

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/workspace/data/subject_001_t1_mni305.nii.gz",
      "output_dir": "/workspace/output"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"input": {"image_path": "/workspace/data/subject_001_t1_mni305.nii.gz", "output_dir": "/workspace/output"}},
    timeout=1800,
)

print(response.json()["output"]["output_path"])
```

### 4.5 Response

```json
{
  "model": "monai/wholeBrainSeg_Large_UNEST_segmentation",
  "output": {
    "output_path": "/workspace/output/subject_001_t1_mni305/subject_001_t1_mni305_trans.nii.gz"
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

`/v1/models` answers with the served model and the label map. The map has 133 entries: the background and 132 brain
regions.

`/healthz` returns HTTP 200 when the service is ready. It does not return the model details or optimization settings.

### 4.6 What to know before you deploy

**The input must already be registered to the MNI305 atlas.** The service does no resampling, no reorientation and
no resizing. It normalizes intensity and nothing else. Register your volume with NiftyReg, ANTs or an equivalent
tool before you send it. If the volume is not registered, the mask is wrong.

**The model expects a T1-weighted brain MRI.** Other modalities give meaningless output.

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `BRAINSEG_TIMEOUT_SECONDS` changes it.

**The weights are fetched at startup.** They are not baked into the image. They are downloaded from
`MONAI/wholeBrainSeg_Large_UNEST_segmentation` on Hugging Face and kept in `$AIM_CACHE_PATH`. Mount a persistent
volume there, as described in section 2, or they are downloaded again on every start.

**This model does not need a Hugging Face token.**

### 4.7 Optimization environment variables

These are off by default. Switch them on to increase the startup time and decrease the inference time.

| Variable | Default | Effect |
|---|---|---|
| `BRAINSEG_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `BRAINSEG_COMPILE_MODE` | `max-autotune` | Compile mode used when `BRAINSEG_COMPILE` is on. |
| `BRAINSEG_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `BRAINSEG_AUTOCAST` | `false` | Run inference in mixed precision. |
| `BRAINSEG_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `BRAINSEG_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `BRAINSEG_LOG_LEVEL` | `INFO` | Log level of the service. |
