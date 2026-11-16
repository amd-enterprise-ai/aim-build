<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Notices for the Aurora AIM

## AIM code

The serving code, harness and build scripts are © Advanced Micro Devices, Inc.,
under the MIT licence in `/workspace/LICENSE`.

## Input data in `/workspace/inputs`

ERA5 reanalysis, 1 to 10 June 2022, 0.25 degree, 00 and 06 UTC, plus the ERA5
land-sea mask, surface geopotential and soil type.

- Creator: European Centre for Medium-Range Weather Forecasts (ECMWF), for the
  Copernicus Climate Change Service (C3S).
- Source: the WeatherBench 2 copy of ERA5 published by Google Research,
  `gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`,
  itself downloaded from the Copernicus Climate Data Store.
- Licence: the Licence to Use Copernicus Products
  (<https://apps.ecmwf.int/datasets/licences/copernicus/>), which allows
  redistribution with attribution. Copernicus data is now also released under
  CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>).
- Changes: variables and the 13 pressure levels selected, latitude ordered north
  to south, missing values set to 0, soil type rounded to whole classes, values
  cast to float32 and written to NetCDF in the layout `aurora.Batch` reads.
- Attribution: Contains modified Copernicus Climate Change Service information
  2022. Neither the European Commission nor ECMWF is responsible for any use
  that may be made of the Copernicus information or data it contains.
- Citation: ERA5 data obtained from the WeatherBench 2 archive
  (`gs://weatherbench2/datasets/era5/`), originally from the Copernicus Climate
  Change Service (C3S) Climate Data Store: Hersbach, H. et al. (2023): ERA5
  hourly data on pressure levels from 1940 to present, DOI: 10.24381/cds.bd0915c6;
  and ERA5 hourly data on single levels from 1940 to present,
  DOI: 10.24381/cds.adbb2d47.
- Reference: Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly
  Journal of the Royal Meteorological Society, 146, 1999–2049.
  <https://doi.org/10.1002/qj.3803>

## Model weights (not in this image)

`aurora-0.25-pretrained.ckpt`, Aurora, by Microsoft Research.

- Licence: MIT, Copyright (c) Microsoft Corporation
  (<https://huggingface.co/microsoft/aurora>).
- The image does not contain the weights. The container downloads them from
  Hugging Face at startup.
- Code: the `microsoft-aurora` Python package, MIT, Copyright (c) Microsoft
  Corporation (<https://github.com/microsoft/aurora>). Its licence is in its
  `*.dist-info` directory.
- Reference: Bodnar, C. et al. (2025). A foundation model for the Earth system.
  Nature. <https://doi.org/10.1038/s41586-025-09005-y>

## Third-party software

Each Python package in the image keeps its licence in its `*.dist-info`
directory under `site-packages`.
