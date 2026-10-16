<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Segment one CT volume. |
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
| `output_dir` | string | the bundle `eval` directory | Directory for the segmentation mask, **inside the container**. |

`/v1/models` takes an empty body. `/healthz` needs no request body.

**The service reads and writes files, not request bodies.** `image_path` must exist in the container filesystem, and
the response returns the path of the mask and not the mask itself. Mount your data into the container, as shown in
section 4.4.

### 4.4 Example request

The service reads and writes the paths in the request inside the container, so the volumes must be reachable there.
Add `-v /path/to/ct-data:/data` to the `docker run` command of section 1.1, or mount the same directory in the
Kubernetes deployment of section 3. The examples below use `/data` for that mount.

Call the endpoint:

```bash
curl http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_path": "/data/abdomen_001.nii.gz",
      "output_dir": "/data/out"
    }
  }'
```

The same request in Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/inference",
    json={"input": {"image_path": "/data/abdomen_001.nii.gz", "output_dir": "/data/out"}},
    timeout=1800,
)

print(response.json()["output"]["output_path"])
```

### 4.5 Response

```json
{
  "model": "MONAI/swin_unetr_btcv_segmentation",
  "output": {
    "output_path": "/data/out/abdomen_001/abdomen_001_trans.nii.gz"
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

`/v1/models` answers with the served model and the 14-entry label map, one entry for the background and one for each
of the 13 segmented organs:

```json
{
  "object": "list",
  "data": [
    {
      "id": "MONAI/swin_unetr_btcv_segmentation",
      "object": "model",
      "owned_by": "MONAI",
      "channel_def": {
        "0": "background",
        "1": "spleen",
        "2": "Right Kidney",
        "3": "Left Kidney",
        "4": "Gallbladder",
        "5": "Esophagus",
        "6": "Liver",
        "7": "Stomach",
        "8": "Aorta",
        "9": "IVC",
        "10": "Portal and Splenic Veins",
        "11": "Pancreas",
        "12": "Right adrenal gland",
        "13": "Left adrenal gland"
      }
    }
  ]
}
```

`/healthz` returns HTTP 200 when the service is ready. It does not return the model details or optimization settings.

### 4.6 What to know before you deploy

**The model expects abdominal CT.** CT is Computed Tomography, an X-ray imaging method that gives a 3D volume.
Preprocessing resamples to 1.5 × 1.5 × 2.0 mm, reorients to RAS, and clips intensity to the −175 to 250 Hounsfield
range. Other modalities, such as MRI, give meaningless output.

**Requests are served one at a time.** The service holds a lock around inference, so a second request waits for the
first to finish. The profile uses one accelerator.

**The request timeout is 1800 seconds.** Set a client timeout that is not shorter, or the client stops before the
service replies. `SWINUNETR_TIMEOUT_SECONDS` changes it.

**The weights ship in the image.** They are downloaded from Hugging Face at startup only if they are missing from the
bundle directory.

**This model does not need a Hugging Face token.**

### 4.7 Optimization environment variables

These are off by default. Switch them on to increase the startup time and decrease the inference time.

| Variable | Default | Effect |
|---|---|---|
| `SWINUNETR_COMPILE` | `false` | Compile the network with `torch.compile`. |
| `SWINUNETR_COMPILE_MODE` | `max-autotune` | Compile mode used when `SWINUNETR_COMPILE` is on. |
| `SWINUNETR_DYNAMIC` | `false` | Allow dynamic shapes in the compiled network. |
| `SWINUNETR_AUTOCAST` | `false` | Run inference in mixed precision. |
| `SWINUNETR_GPU_TRANSFORMS` | `false` | Run the preprocessing transforms on the accelerator. |
| `SWINUNETR_TIMEOUT_SECONDS` | `1800` | Whole-request cap. |
| `SWINUNETR_LOG_LEVEL` | `INFO` | Log level of the service. |
