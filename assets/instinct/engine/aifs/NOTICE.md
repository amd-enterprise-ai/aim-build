<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Notices for the AIFS AIMs

## AIM code

The serving code, harness and build scripts are © Advanced Micro Devices, Inc.,
under the MIT licence in `/workspace/LICENSE`.

## Input data in `/workspace/inputs`

ECMWF Open Data, IFS 0.25 degree, step-0 fields from the `oper` and `wave`
streams every 6 hours from 2026-10-01T00 to 2026-10-10T18 UTC. Under
`members/pf01` and `members/pf02`: perturbed ensemble members 1 and 2 from the
`enfo` and `waef` streams at 2026-10-01T18, 2026-10-02T00, 2026-10-02T18 and
2026-10-03T00 UTC, with the fixed fields (`lsm`, `z`, `slor`, `sdor`) from `oper`.

- Creator: European Centre for Medium-Range Weather Forecasts (ECMWF).
- Source: ECMWF Open Data (<https://www.ecmwf.int/en/forecasts/datasets/open-data>),
  accessed from its copies on Google Cloud
  (`https://storage.googleapis.com/ecmwf-open-data`) and Amazon Web Services
  (`s3://ecmwf-forecasts`).
- Licence: CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>).
- Changes: the GRIB2 messages AIFS reads were taken from the published files and
  joined into one file per time. The messages themselves are unchanged. Each
  directory's `manifest.json` lists the source files and the SHA-256 of every
  file written.
- Attribution: ECMWF IFS forecast data (Open Data, 0.25 degree), © European
  Centre for Medium-Range Weather Forecasts (ECMWF), 2026, accessed via Google
  Cloud and Amazon Web Services. Licensed under CC-BY-4.0. Modified: selected
  GRIB messages joined into one file per time.

## Regrid matrix in `/opt/aifs/regrid-db`

The linear 0.25 degree to N320 interpolation matrix, unchanged, from the
earthkit-regrid matrix database that ECMWF hosts at
<https://sites.ecmwf.int/repository/earthkit/regrid/db/1/>. © ECMWF. It is
published for use with earthkit-regrid, which is licensed Apache-2.0; ECMWF
states no separate licence for the matrix files.

## Model weights (not in this image)

- AIFS Single 2.0: `aifs-single-mse-2.0.ckpt` and `lsm.grib`,
  <https://huggingface.co/ecmwf/aifs-single-2.0>.
- AIFS ENS 2.0: `aifs-ens-crps-2.0.ckpt` and `lsm.grib`,
  <https://huggingface.co/ecmwf/aifs-ens-2.0>.

Both are © ECMWF, licensed CC-BY-4.0. The image does not contain them. The
container downloads them, unchanged, from Hugging Face at startup.

The input preparation in `/workspace/model/src/data_utils.py` is adapted from
ECMWF's notebooks `run_AIFS_v2.0.ipynb` and `run_AIFS_ENS_v2.0.ipynb` from the
same repositories, licensed Apache-2.0, and changed to run as a service.

References:

- Lang, S. et al. (2024). AIFS – ECMWF's data-driven forecasting system.
  arXiv:2406.01465. <https://arxiv.org/abs/2406.01465>
- Lang, S. et al. (2024). AIFS-CRPS: Ensemble forecasting using a model trained
  with a loss function based on the Continuous Ranked Probability Score.
  arXiv:2412.15832. <https://arxiv.org/abs/2412.15832>
- Lang, S., Leutbecher, M. and Maciel, P. (2025). A multi-scale loss formulation
  for learning a probabilistic model with proper score optimisation.
  arXiv:2506.10868. <https://arxiv.org/abs/2506.10868>

## Third-party software

- flash-attention, built from <https://github.com/ROCm/flash-attention> at the
  commit in `/opt/aifs/flash-attention.commit`: BSD-3-Clause. Its licence and
  authors list are in `/opt/aifs/licenses/`, with the licence of the Composable
  Kernel library it is built with.
- anemoi-inference, anemoi-models, anemoi-graphs, earthkit and ecCodes, by
  ECMWF: Apache-2.0.
- Each Python package in the image keeps its licence in its `*.dist-info`
  directory under `site-packages`.
