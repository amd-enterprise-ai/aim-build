<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

**Checkpoint.** The full-size 0.25° pretrained checkpoint, `aurora-0.25-pretrained.ckpt` (1.26 B parameters). It is
deterministic and autoregressive: it advances the atmosphere in 6-hour steps and keeps every step in the output, on the
ERA5 0.25° grid with 13 pressure levels from 50 to 1000 hPa.

**Weights.** The weights are [MIT](https://huggingface.co/microsoft/aurora), Copyright (c) Microsoft Corporation, from
[microsoft/aurora](https://github.com/microsoft/aurora). They are not in the image; the container downloads them from
Hugging Face at startup. If you use Aurora in published work, cite Bodnar, C. et al. (2025),
[A foundation model for the Earth system](https://doi.org/10.1038/s41586-025-09005-y), Nature.

**Example inputs.** The files in `/workspace/inputs` contain modified Copernicus Climate Change Service information
2022, from the WeatherBench 2 copy of ERA5, under the
[Licence to Use Copernicus Products](https://apps.ecmwf.int/datasets/licences/copernicus/). Neither the European
Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
ERA5 data obtained from the WeatherBench 2 archive (`gs://weatherbench2/datasets/era5/`), originally from the Copernicus
Climate Change Service (C3S) Climate Data Store: Hersbach, H. et al. (2023): ERA5 hourly data on pressure levels from
1940 to present, DOI: [10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6); and ERA5 hourly data on single
levels from 1940 to present, DOI: [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47).

Full attribution for the data and the weights is in `/workspace/NOTICE.md` inside the image.
