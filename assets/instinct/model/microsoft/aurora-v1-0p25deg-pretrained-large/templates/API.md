<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

### 4.2 The forecast endpoint

This AIM serves one forecast endpoint and two read-only endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/inference` | Run a forecast from an initial condition sent as a NetCDF file. The forecast is written to a NetCDF file. |
| `GET` | `/health` | HTTP 200 once the checkpoint is loaded. Forwards to `/healthz`. |
| `GET` | `/v1/models` | The model this container serves, as `data[0].id`. |

The forecast is written to disk. A full 0.25° roll-out is large, and base64 inflates it by about a third. Set
`return_data_base64` to also return the bytes in the response.

### 4.3 Request body

| Field | Type | Default | Purpose |
|---|---|---|---|
| `input_netcdf_path` | string | `""` | Initial condition, named relative to `AIM_INPUT_PATH`. Preferred: one initial condition is about 345 MB. |
| `input_netcdf_b64` | string | `""` | The same file sent inline, base64-encoded. Small inputs only. |
| `forecast_steps` | integer | `1` | Roll-out steps. One step is 6 hours. Values below 1 are treated as 1. |
| `output_format` | string | `"netcdf"` | Only `"netcdf"`. |
| `output_dir` | string | `""` | Subdirectory of `AIM_OUTPUT_PATH`. Empty uses the output root. |
| `return_data_base64` | boolean | `false` | Also return the NetCDF bytes inline. |
| `conditioning_netcdf_b64` | string | `""` | Ignored. This checkpoint takes no conditioning field. |
| `num_samples` | integer | `1` | Ignored. This checkpoint is deterministic. |

One of `input_netcdf_path` or `input_netcdf_b64` is required. Sending both is rejected.

The file must be written by `aurora.Batch.to_netcdf()`, which prefixes the variables with `surf_`, `static_` and
`atmos_`. A plain ERA5 NetCDF does not read as an Aurora batch and is rejected. The file needs two history steps
6 hours apart, surface `2t`, `10u`, `10v`, `msl`, static `lsm`, `z`, `slt`, and upper-air `z`, `u`, `v`, `t`, `q` on
the 13 standard ERA5 pressure levels (50 to 1000 hPa), on the ERA5 0.25° grid (721×1440). The model crops the grid to
720×1440.

The service rejects these before the model runs:

* `input_netcdf_path` and `input_netcdf_b64` together.
* Neither initial-condition field.
* `output_format` set to anything other than `netcdf`.
* `input_netcdf_path` when `AIM_INPUT_PATH` is unset, when the path leaves that directory, or when it names no file there.
* `output_dir` that resolves outside `AIM_OUTPUT_PATH`.

An input that lacks a variable, or has the wrong number of history steps or pressure levels, fails with an error that
names what is missing.

### 4.4 Example request

The image holds ten days of ERA5 in `/workspace/inputs`, 1 to 10 June 2022, as `<YYYY-MM-DD>_input.nc`;
`aurora_input.nc` is the first day. Mount your own files over that directory to use them:

```bash
curl -X POST http://localhost:8000/v1/inference \
  -H 'Content-Type: application/json' \
  -d '{"input_netcdf_path": "aurora_input.nc", "forecast_steps": 4}'
```

`forecast_steps: 4` is a 24-hour forecast. The file contains lead times 0, 6, 12, 18 and 24 hours. Lead 0 is the
initial condition.

### 4.5 Response

A successful response has this shape:

```json
{
  "output_path": "/tmp/weather/output/forecast_1735689600000.nc",
  "output_format": "netcdf",
  "dims": {"time": 1, "lead_time": 5, "lat": 720, "lon": 1440, "level": 13},
  "variables": ["2t", "10u", "10v", "msl", "z", "u", "v", "t", "q"],
  "all_finite": true,
  "optimizations": {"autocast": false, "compile": true}
}
```

| Key | Purpose |
|---|---|
| `output_path` | Absolute path of the NetCDF file inside the container. |
| `output_format` | `"netcdf"`. |
| `dims` | Size of each axis. `lead_time` is `forecast_steps + 1`, because lead 0 is kept. |
| `variables` | Surface `2t`, `10u`, `10v`, `msl` and upper-air `z`, `u`, `v`, `t`, `q`. |
| `all_finite` | `true` when every value in the file is finite. |
| `optimizations` | Whether reduced precision (`AURORALG_AUTOCAST`) and graph compilation (`AURORALG_COMPILE`) were on. |
| `data_base64` | Present only with `return_data_base64: true`. The NetCDF bytes. |

Surface variables have dimensions `(time, lead_time, lat, lon)`. Upper-air variables have
`(time, lead_time, level, lat, lon)`. `time` holds one entry per initial condition in the input file. `lead_time` is
the horizon in hours.

### 4.6 What to know before you deploy

**Each roll-out step is 6 hours.** A horizon of N days takes 4N steps.

**The checkpoint is not in the image.** `aurora-0.25-pretrained.ckpt`, about 4.7 GiB, is downloaded from Hugging Face
at the first start and kept in `$AIM_CACHE_PATH`. Mount a persistent volume there, as described in section 2, or mount
a directory holding the checkpoint and set `AURORALG_WEIGHTS_DIR` to it. The weights are
[MIT](https://huggingface.co/microsoft/aurora), Copyright (c) Microsoft Corporation.

**The example inputs are Copernicus data.** The ERA5 files in `/workspace/inputs` contain modified Copernicus Climate
Change Service information 2022, from the WeatherBench 2 copy of ERA5, under the
[Licence to Use Copernicus Products](https://apps.ecmwf.int/datasets/licences/copernicus/). The full attribution is in
`/workspace/NOTICE.md`.

**Forecasts are written inside the container.** Each request writes one file to `AIM_OUTPUT_PATH`, which defaults to
`/tmp/weather/output`. Mount a volume there to keep them. Nothing prunes that directory.

**The first request is slower.** The shipped profile turns on graph compilation, which runs on the first request.

**A request may run for 30 minutes.** The service allows 1800 seconds per request. Set the client timeout at least
that high.

**This model does not need a Hugging Face token.**

**The shipped profile (MI300X) is of type `optimized`.** It is selected automatically.

### 4.7 Runtime settings

| Variable | Default | Effect |
|---|---|---|
| `AIM_INPUT_PATH` | `/workspace/inputs` | Directory `input_netcdf_path` resolves against. Empty refuses that field. |
| `AIM_OUTPUT_PATH` | `/tmp/weather/output` | Where forecasts are written. |
| `AIM_CACHE_PATH` | `/workspace/model-cache` | Where the downloaded checkpoint is kept. |
| `AIM_PORT` | `8000` | Service port. |
| `AURORALG_AUTOCAST` | off | Reduced precision (bf16). |
| `AURORALG_COMPILE` | on in the shipped profile | Graph compilation. Slower first request, faster after. |
| `AURORALG_WEIGHTS_DIR` | unset | Load the checkpoint from this directory instead of downloading it. For air-gapped hosts. |
| `AURORALG_WARMUP_ON_STARTUP` | on | Load the checkpoint during startup rather than on the first request. |
| `AURORALG_CHECKPOINT` | `aurora-0.25-pretrained.ckpt` | Checkpoint filename. |

The shipped profile also sets `TORCH_BLAS_PREFER_HIPBLASLT=1` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, which keep the
model at fp32.
