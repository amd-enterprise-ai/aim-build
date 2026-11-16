# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Copy the earthkit-regrid 0.25 deg to N320 matrix into a local matrix DB.

Pass the output directory as ``matrix_source`` to ``earthkit.regrid.interpolate``
to regrid without network access.
"""

import os
import shutil
import sys

from earthkit.regrid import db

IN_GRID = {"grid": [0.25, 0.25]}
OUT_GRID = {"grid": "N320"}


def main(outdir):
    entry = db.SYS_DB.find_entry(IN_GRID, OUT_GRID, "linear")
    if entry is None:
        raise SystemExit("no 0.25 deg -> N320 linear matrix in the earthkit-regrid DB")
    os.makedirs(outdir, exist_ok=True)
    path = db.SYS_DB.copy_matrix_file(entry, outdir, exist_ok=True)
    shutil.copy(db.SYS_DB.index_file_path(), os.path.join(outdir, db._INDEX_FILENAME))

    local = db.MatrixDb.from_path(outdir)
    z, shape = local.find(IN_GRID, OUT_GRID, "linear")
    if z is None or tuple(shape) != (542080,) or z.shape != (542080, 721 * 1440):
        raise SystemExit("local matrix DB check failed: shape=%r matrix=%r" % (shape, getattr(z, "shape", None)))
    print("matrix %s (%d bytes, nnz %d)" % (path, os.path.getsize(path), z.nnz))


if __name__ == "__main__":
    main(sys.argv[1])
