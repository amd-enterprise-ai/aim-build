<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ERA5 snapshot for weather AIM CI

Download ERA5, convert it to the file the Aurora AIM reads. Raw ERA5 is not
usable directly.

## Setup API key

Go here and get your API key:
https://cds.climate.copernicus.eu/how-to-api

Then, store it at `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <your-key>
```

## Download & prepare a batch
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python3 fetch_era5.py --start 2024-06-01 --days 10 --outdir era5
python3 make_aurora_input.py --source-dir era5 --outdir out
```

One input file per day, at `out/<date>/aurora_input.nc`.
