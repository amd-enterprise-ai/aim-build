<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# CI inputs of the GROMACS job AIM

Inputs of the run-to-completion check of the job-profile CI.

| Case | Input | Expected |
|------|-------|----------|
| `small` | `small/run.tpr` (how it was made: below) with `small.env.yaml` | exit 0, `gromacs.log` plus `md.*` files and `DONE` in `AIM_OUTPUT_PATH` |
| `broken` | `broken/run.tpr`, a text file that is not a run input | exit 1 (mdrun rejects it) and `gromacs.log` in `AIM_OUTPUT_PATH` |

`small.env.yaml` sets `AIM_ENGINE_ARGS` to `{"nsteps": 500}`. CI passes the profile's `engine_args` as
`AIM_ENGINE_ARGS` and merges a case's value over them key by key
([docs/engine-args-override.md](../../../../../../../docs/engine-args-override.md)), so the run does 500 steps and keeps
the profile's `nstlist` and `ntmpi`.

## How `small/run.tpr` was made

`small/run.tpr` is an SPC water box (55,028 bytes, 2.8 nm cube, OPLS-AA, `gen-vel = no` so no velocities are
stored). The input is encoded as a base64 gzip tar of at most 190,000 characters, so `run.tpr` must stay at or below about
140 KB (it does not compress well). It was built inside the published image (`grompp` needs no GPU) and checked with the
image's own entrypoint on one MI300X: the `small` case exits 0 with `gromacs.log`, `md.log`, `md.edr`, `md.cpt`, `md.gro`
and `DONE`, and the `broken` case exits 1 with `gromacs.log`. To rebuild it, for example after a GROMACS version change
(the `.tpr` format is tied to the version that wrote it), run:

```bash
mkdir -p /tmp/gmx-small && cd /tmp/gmx-small
cat > topol.top <<'TOP'
#include "oplsaa.ff/forcefield.itp"
#include "oplsaa.ff/spc.itp"
[ system ]
Water
[ molecules ]
TOP
cat > md.mdp <<'MDP'
integrator            = md
dt                    = 0.002
nsteps                = 20000     ; CI overrides this through AIM_ENGINE_ARGS
cutoff-scheme         = Verlet
coulombtype           = PME
rcoulomb              = 0.9
rvdw                  = 0.9
tcoupl                = V-rescale
tc-grps               = System
tau-t                 = 0.1
ref-t                 = 300
constraints           = h-bonds
nstcalcenergy         = 100
nstenergy             = 500
nstlog                = 500
gen-vel               = no        ; no velocities stored: smaller .tpr
MDP
docker run --rm -v "$PWD":/w -w /w --entrypoint /bin/sh <published-image> -c '
  gmx solvate -cs spc216.gro -box 2.8 2.8 2.8 -o conf.gro -p topol.top &&
  gmx grompp -f md.mdp -c conf.gro -p topol.top -maxwarn 1 -o run.tpr'
ls -l run.tpr            # must be at or below ~140000 bytes; if larger, use -box 2.6 in the solvate step
```

Then verify on a GPU node (single GPU, same settings the CI applies; the profile's `engine_args` with `nsteps` 500):

```bash
mkdir -p in out && cp run.tpr in/
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -e AIM_INPUT_PATH=/in -e AIM_OUTPUT_PATH=/out -e AIM_ACCELERATOR_COUNT=1 -e AIM_ACCELERATOR_MODEL=MI300X \
  -e AIM_ENGINE_ARGS='{"nsteps": 500, "nstlist": 200, "ntmpi": 1}' \
  -v "$PWD/in":/in:ro -v "$PWD/out":/out <published-image>
echo "exit=$?"; ls out      # expect exit=0, gromacs.log, md.log, md.edr, md.cpt, DONE
```

Run the same with the `broken/` input as `/in` and confirm a non-zero exit below 128 with `out/gromacs.log` present
(a status of 128 or more counts as "killed by a signal" and fails the case, so a crash of mdrun on the corrupt file
would need a different broken input, for example a missing `run.tpr`).
