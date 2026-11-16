# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""NetCDF output helper.

``save_netcdf`` writes a forecast to a NetCDF file and reports its structure for
the ``/v1/inference`` response.

numpy and xarray are imported lazily so importing this module stays cheap.
"""

import os

# The 13 standard ERA5 pressure levels (hPa).
DEFAULT_PRESSURE_LEVELS = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)


def save_netcdf(out_path, data_vars, coords):
    """Write a forecast to NetCDF and return its structure.

    ``data_vars`` is ``{name: (dims_tuple, ndarray)}`` and ``coords`` is
    ``{dim_name: 1-D array}``, the same shapes ``xarray.Dataset`` takes. Returns
    ``{"dims": {...}, "variables": [...], "all_finite": bool}``.

    The file is written under a temporary name and renamed, so a reader never
    sees a half-written forecast.
    """
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    partial = "%s.partial" % out_path
    ds.to_netcdf(partial)
    os.replace(partial, out_path)
    all_finite = all(bool(np.isfinite(np.asarray(var.values)).all()) for var in ds.data_vars.values())
    dims = {str(d): int(ds.sizes[d]) for d in ds.dims}
    return {"dims": dims, "variables": list(ds.data_vars), "all_finite": all_finite}
