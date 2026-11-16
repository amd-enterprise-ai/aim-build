<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Aurora harness

How the Aurora checks are laid out, and how to run them by hand.

## Where the code sits

| directory | holds |
|---|---|
| `assets/instinct/engine/aurora` | the BentoML service and the model code |
| `assets/instinct/base/aurora` | the AIM runtime and this harness |
| `assets/instinct/model/microsoft/aurora-v1-0p25deg-pretrained-large` | profiles and metadata, no code |

## Input data

The image holds ten days of ERA5 (1 to 10 June 2022) in `/workspace/inputs`,
one file per day: `<YYYY-MM-DD>_input.nc`. `aurora_input.nc` is the first day.

The build downloads them from the public WeatherBench 2 bucket, no API key needed:

- `data/fetch_weatherbench2.py` downloads the days.
- `data/make_aurora_input.py` converts them to the format Aurora reads.

Pick other days with the `ERA5_START` and `ERA5_DAYS` build args (1959 to 2023-01-10).

The data is modified Copernicus Climate Change Service information. Its
attribution and licence are in `/workspace/NOTICE.md` in the image, copied from
`assets/instinct/base/aurora/NOTICE.md`.

## Running it by hand

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  -e AIM_OUTPUT_PATH=/workspace/outputs \
  <aurora-image> aim-runtime serve

docker exec <container> aim-runtime validate --service-url http://localhost:8000
```

To use your own files, mount them over `/workspace/inputs`.

## What the checks cover

| check | what fails it |
|---|---|
| `profile_facts` | the input file is missing |
| `accelerator_visible` | torch cannot see the GPU |
| `weights_resolvable` | never fails; reports where the checkpoint comes from |
| `service_health` | `/health` does not answer 200 |
| `predict_smoke` | one inference does not return a usable response |
| `forecast_sanity` | missing dims or variables, or non-finite values |
| `grid_matches` | response dims are not the grid the profile declares |
| `bad_input` | garbage input comes back as a forecast, or the service dies |
| `concurrent_requests` | answers get swapped, files collide, or health drops |
| `request_fields` | does not gate; a coverage table of honoured request fields |
| `lead_zero` | lead 0 is not the initial condition handed back |

Per-model values (dims, grid, variables, ignored fields, unchanged dims,
concurrent requests) live in this harness, in `default_facts()` in `harness.py`.
The input is `aurora_input.nc` unless `AIM_DATA_PATH` names another file.
