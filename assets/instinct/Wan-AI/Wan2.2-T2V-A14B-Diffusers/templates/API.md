<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The endpoints

This AIM serves the vLLM-Omni diffusion API for text-to-video generation. Generation is asynchronous: submit a job,
poll its status, then download the clip.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/videos` | Submit a text-to-video generation job. Returns a job id. |
| `GET` | `/v1/videos/{id}` | Report the status of a job (`queued`, `completed`, `failed`). |
| `GET` | `/v1/videos/{id}/content` | Download the generated video once the job is `completed`. |
| `GET` | `/v1/models` | Report the served model. Used for the health check in section 4.1. |

This AIM does **not** serve the OpenAI `/v1/completions` or `/v1/chat/completions` endpoints.

### 4.3 Request body

`POST /v1/videos` takes `multipart/form-data` fields (send them as form fields, not JSON):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `prompt` | string | required | Text description of the video to generate. |
| `negative_prompt` | string | not set | Text to steer generation away from. The model runs classifier-free guidance, so a negative prompt is supported. |
| `size` | string | model default | Frame resolution as `WIDTHxHEIGHT`. The model targets 480P and 720P; the AIM is benchmarked at `854x480` and `1280x720`. |
| `seconds` | integer | model default | Clip length in seconds. |
| `fps` | integer | model default | Frames per second. |
| `num_inference_steps` | integer | model default | Number of denoising steps. Fewer steps are faster and lower quality; this dominates wall-clock time. |
| `seed` | integer | not set | Random seed for reproducible output. |

### 4.4 Example request

Submit a job, poll until it completes, then download the clip:

```bash
# Submit the job and capture its id
JOB=$(curl -s http://localhost:8000/v1/videos \
  -F prompt="a red panda surfing a wave, cinematic" \
  -F size=832x480 -F seconds=5 -F fps=16 -F num_inference_steps=18 \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

# Poll until the status is "completed" (or "failed")
curl -s http://localhost:8000/v1/videos/$JOB

# Download the finished clip
curl -s http://localhost:8000/v1/videos/$JOB/content -o out.mp4
```

The same flow in Python:

```python
import time
import requests

base = "http://localhost:8000/v1/videos"

job = requests.post(
    base,
    files={
        "prompt": (None, "a red panda surfing a wave, cinematic"),
        "size": (None, "832x480"),
        "seconds": (None, "5"),
        "fps": (None, "16"),
        "num_inference_steps": (None, "18"),
    },
    timeout=300,
).json()["id"]

while True:
    status = requests.get(f"{base}/{job}", timeout=30).json()["status"]
    if status in ("completed", "failed"):
        break
    time.sleep(2)

if status == "completed":
    content = requests.get(f"{base}/{job}/content", timeout=120).content
    with open("out.mp4", "wb") as fh:
        fh.write(content)
```

### 4.5 Response

`POST /v1/videos` answers with a job id and its initial status:

```json
{
  "id": "vid-abc123",
  "status": "queued"
}
```

`GET /v1/videos/{id}` reports the current job state:

```json
{
  "id": "vid-abc123",
  "status": "completed"
}
```

| `status` | Meaning |
|---|---|
| `queued` / `in_progress` | The job is still generating. Keep polling. |
| `completed` | The clip is ready. Download it from `/v1/videos/{id}/content`. |
| `failed` | Generation failed. The `error` field holds the reason. |

`GET /v1/videos/{id}/content` streams the generated video (an MP4) as the response body.

### 4.6 What to know before you deploy

**Multi-GPU profiles need shared memory.** A profile named `...-uspN` with N greater than one shards the latent
sequence across N accelerators with Unified Sequence Parallelism, and the workers exchange data over shared memory.
Run the container with `--ipc=host` (or `--shm-size=32g`) and set `AIM_ACCELERATOR_COUNT` to N, or the workers fail to
start. The [README](README.md) lists the available profiles and their accelerator counts.

**`num_inference_steps` dominates latency.** Each step runs the full diffusion transformer, and the model runs two
passes per step for classifier-free guidance. Lower the step count for faster, lower-quality drafts and raise it for
final renders.

**The checkpoint is large and downloaded at first start.** It is kept in `$AIM_CACHE_PATH`, which defaults to
`/workspace/model-cache`. Mount a persistent volume there, as described in section 2, so that later starts skip the
download.

**Submission can be slow.** Some builds generate the clip synchronously on `POST /v1/videos` rather than returning a
job id immediately. Set a client timeout long enough for a short generation, not just an enqueue round-trip.

### 4.7 Validating and benchmarking the deployment

The image ships a harness that the `aim-runtime` CLI drives. Against the container started in section 1.1 (here named
`<CONTAINER>`), run the checks to confirm health and a smoke generation, or to benchmark throughput and latency:

```bash
# List the checks this image exposes
docker exec <CONTAINER> aim-runtime list-checks

# Validate against the running service
docker exec <CONTAINER> aim-runtime validate \
  --service-url http://localhost:8000 --scope runtime --scope offline

# Benchmark against the running service
docker exec <CONTAINER> aim-runtime benchmark \
  --service-url http://localhost:8000 \
  --timeout-seconds 1800
```

The 1800-second timeout applies to the whole benchmark. Increase it if the run needs more time.

`validate` reports `health` and `smoke_video`; `benchmark` reports `throughput_qps` and latency percentiles for the
chosen recipe.

`aim-runtime evaluate` is not available for this AIM yet: the image does not ship a video-quality evaluation, so the
command returns a placeholder result.
