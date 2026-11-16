<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# On vs off OpenStructure scoring

Scores two Boltz-2 evaluate runs against each other and against the deposited
structures. OpenStructure stays in its own container. The Boltz-2 image does
not copy this directory, and CI does not collect it.

Container: `registry.scicore.unibas.ch/schwede/openstructure@sha256:9f2669ffde16c45939fc6e8e6dd4f907697301c081edf3d97d464c034a421d19`, pinned by digest so that the scores can be repeated. `--image` selects another container.

## Reference structures

The targets come from a public performance table. The reference structures are the deposited structures of those
targets. They were cleaned for the OpenFold3 AIM, and this tool uses the same files. The scorer reads
`<label>.cif` from the `--pdb` directory.

## Input

Each run directory holds one mmCIF per target, named `<label>.cif`. If the
directory itself has no mmCIF files, the tool reads `<label>.cif` from its
`structures/` subdirectory. The PDB directory uses the same file names.

## Run

```bash
python3 assets/instinct/mit/boltz2/image/tools/accuracy_on_off/score.py \
  --on <evaluate output, BOLTZ2_OPTIMIZATIONS=on> \
  --off <evaluate output, BOLTZ2_OPTIMIZATIONS=off> \
  --pdb <directory of deposited label.cif files> \
  --out <output directory>
```

The tool copies the mmCIF files into `--out` and then runs OpenStructure.
Docker does not follow a symlink to a file outside the directory it mounts.

It writes `<out>/ost_scores.csv` and one JSON per comparison under
`<out>/ost_json/`. Rows are `on_vs_off`, `on_vs_pdb`, and `off_vs_pdb`.
Columns are all-atom lDDT, backbone lDDT, TM-score, RMSD, oligomeric GDT,
QS-score, DockQ, and fnat. `--image` selects a different container.
