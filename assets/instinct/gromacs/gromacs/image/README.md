<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# GROMACS AIM - Long-Running Job Container

[GROMACS](https://www.gromacs.org/) is a molecular dynamics package. This AIM runs `gmx mdrun` with AMD ROCm/HIP GPU
offload (non-bonded, PME and update on the GPU; bonded forces on the CPU by default) as a job container: it has no HTTP API, reads a run-input
file, writes results, streams its log to stdout and exits.

GROMACS is built from the pinned commit `924b96a7ac69a2e84cf2b73374f82d1f2eeab68a` of
`https://gitlab.com/gromacs/gromacs.git`, the tip of the `4947-hip-feature-enablement` branch (verified with
`git ls-remote` on 2026-10-20). This is a HIP dev-branch commit, not a release. Its CMake version is 2026.4 with
the fork string `2026-HIP-ENABLEMENT`.

The base image is `rocm/dev-ubuntu-24.04:7.2.4` (newest ROCm 7.2.x tag), resolved on 2026-10-20 to digest
`sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054`. apt package versions are not pinned individually: they come from the pinned base image's Ubuntu
repositories at build time. The exact versions installed are recorded in every image in
`/opt/aim-build-packages.txt` (runtime stage) and `/opt/gromacs-build-packages.txt` (build-stage tools).

Changing the base image requires a rebuild and an MI300X validation run before the pins can be trusted.

## Supported accelerators

This AIM targets the AMD Instinct MI300X (`gfx942`) only. The HIP offload target list in the Dockerfile
(`CMAKE_HIP_ARCHITECTURES`) is `gfx942`, and only MI300X profiles are shipped. Other accelerators (for example
MI250X `gfx90a` or MI355X `gfx950`) are not built for and have not been validated. The CPU side is built with
`-DGMX_SIMD=AVX2_256`.

## Usage

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -e AIM_INPUT_PATH=/input \
  -e AIM_OUTPUT_PATH=/output \
  -v /host/input:/input:ro \
  -v /host/output:/output \
  <image>
```

## Inputs

`$AIM_INPUT_PATH` must contain a GROMACS run-input file, `run.tpr` by default (`tpr` setting for another name).
Create it with `gmx grompp -f md.mdp -c system.gro -p topol.top -o run.tpr`. The input is only read.

## Outputs

Written to `$AIM_OUTPUT_PATH`:

| File          | When                | Description                                                       |
|---------------|---------------------|-------------------------------------------------------------------|
| `gromacs.log` | always, live        | Everything the run prints (also on stdout), incl. startup errors   |
| `md.log`      | success             | GROMACS run log (performance)                                      |
| `md.edr`      | success             | Energy file                                                        |
| `md.cpt`      | success             | Final checkpoint                                                   |
| `md.xtc`      | success, if the `.mdp` writes a trajectory | Compressed trajectory                    |
| `DONE`        | success, written last | Completion marker                                                |

"Finished" versus "still being written": GROMACS writes its `md.*` files in place and cannot do otherwise, so the
AIM runs it in `$AIM_OUTPUT_PATH/.work` and moves the files into `$AIM_OUTPUT_PATH` (one atomic rename each) only
when the run exits 0, then writes `DONE`. Any `md.*` file in `$AIM_OUTPUT_PATH` is therefore complete, and `DONE`
says the set is complete. `gromacs.log` is the exception: it is appended to live, so use `DONE` or the exit status
as the completion signal. After a failed or interrupted run the partial files stay in `.work/`.

## Settings

Only these environment variables are read; any other `AIM_*` variable is logged as a warning and ignored.

| Variable                | Description                                                                        | Default              |
|-------------------------|------------------------------------------------------------------------------------|----------------------|
| `AIM_INPUT_PATH`        | Input directory (required)                                                         | -                    |
| `AIM_OUTPUT_PATH`       | Output directory (required)                                                        | -                    |
| `AIM_ACCELERATOR_COUNT` | GPUs to use (1-9). Overrides device detection                                      | detected             |
| `AIM_ACCELERATOR_MODEL` | Informational, logged only                                                         | -                    |
| `AIM_ENGINE_ARGS`       | JSON object of tunables below (same grammar as the other AIMs)                     | `{}`                 |

GPUs are detected by counting `/dev/dri/renderD*` when `/dev/kfd` exists; when `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` is set, the number of devices listed caps that count. The run needs at least `ntmpi` GPUs and
fails with status 2 otherwise (also when `AIM_ACCELERATOR_COUNT` exceeds the visible devices, or no GPU is found).

`AIM_ENGINE_ARGS` is a JSON object, for example `-e 'AIM_ENGINE_ARGS={"ntomp": 24, "nstlist": 150}'`. Keys may be
written with `-` or `_`. Space-separated flag strings are not accepted, and unknown keys are rejected (status 2).

| Key       | Description                                         | Default                      |
|-----------|-----------------------------------------------------|------------------------------|
| `ntmpi`   | GPUs / ranks (`>1` uses `gmx_mpi` via `mpirun`)     | detected / `AIM_ACCELERATOR_COUNT` |
| `ntomp`   | OpenMP threads per rank; default caps the total at 24 and leaves one hardware thread free (HIP runtime thread) | `min(24, usable threads - 1)`, divided by `ntmpi` (at least 1) |
| `nstlist` | Neighbor-list update interval                       | `200`                        |
| `nsteps`  | Steps to run (`-2` = value in the `.tpr`)           | `-2`                         |
| `gpu-id`  | GPU id string, e.g. `0123`                          | the first `ntmpi` GPUs       |
| `tpr`     | Input file name inside `$AIM_INPUT_PATH`            | `run.tpr`                    |
| `pme_tune` | `true` lets mdrun auto-tune the PME grid; `false` passes `-notunepme` (best for short runs, long production runs may gain from tuning) | `false` |
| `bonded_cpu` | `true` computes bonded interactions on the CPU (`-bonded cpu`), `false` on the GPU (`-bonded gpu`) | `true` |

The same tunables exist as command-line flags (`--ntomp 24`), which win over `AIM_ENGINE_ARGS`.

### Profiles

The profiles `job-mi300x-fp32-tp{1,2,4}-throughput` describe the supported configurations. This image does not
contain the aim-runtime and never reads profile files. A platform that selects a profile passes its `engine_args`
as the `AIM_ENGINE_ARGS` JSON object (the keys are exactly the ones above, and a unit test checks that) and sets
`AIM_ACCELERATOR_COUNT` from the profile's `accelerator_count`; the profile `env_vars` are plain environment
variables for the container (they equal the defaults the AIM already applies). Running the container without a
profile uses the defaults above.

## Exit status

| Status | Meaning                                                                          |
|--------|----------------------------------------------------------------------------------|
| `0`    | mdrun finished, results published, `DONE` written                                |
| `1`    | Input missing, or mdrun failed (status in `gromacs.log`; mdrun's own code if non-zero) |
| `2`    | Invalid settings (bad `AIM_ENGINE_ARGS`, too few GPUs)                           |
| `127`  | GROMACS or `mpirun` could not be started                                         |
| `128+N`| Stopped by signal N (`143` for SIGTERM / `docker stop`, `130` for SIGINT)         |

## Interrupt and resume

`SIGTERM` and `SIGINT` are forwarded to the whole mdrun/mpirun process group. mdrun stops at the next step, writes a
checkpoint into `.work/` and the container exits `128+N`; a child that ignores the signal is killed after 60 s.
Results of an interrupted run are not published (no `DONE`).

Resume: start the container again with the same `$AIM_OUTPUT_PATH`. When `.work/md.cpt` exists, the runner passes it to
mdrun with `-cpi` and mdrun continues from the checkpoint, appending to `.work/md.*`; the run then completes as usual. The runner passes `-cpi` explicitly because without it `mdrun -deffnm md` starts a new run from step 0 even when `md.cpt` exists (tested with GROMACS 2026.4 on CPU after SIGTERM and SIGKILL, thread-MPI binary only). Resume also needs `md.log` and `md.edr` from the earlier run to still be present in `.work`, and `-nsteps` counts total steps from step 0, so a resumed run does the remaining steps. mdrun writes a checkpoint every 15 minutes, so
a hard kill loses at most that. Without a surviving `.work/` directory the run starts from step 0. A directory that
already contains a finished run is overwritten by the next successful run, which also removes the `md.*` files of the earlier run that it did not produce (for example an old `md.xtc`). There is no wall-clock limit; bound a run
with `nsteps`.

## Filesystem requirements

mdrun appends to `md.log`/`md.edr`, seeks in `md.xtc`/`md.cpt` and renames the checkpoint. `$AIM_OUTPUT_PATH` must be
a POSIX filesystem (local disk, NFS, Lustre/GPFS); object-store or FUSE mounts without append, seek and atomic
rename are not supported. Publishing renames files from `.work/` into the same directory tree, so keep
`$AIM_OUTPUT_PATH` on one filesystem (do not mount a separate volume at `.work`). The input directory can be
read-only.

## Multi-GPU runs

With `ntmpi > 1` the AIM starts `mpirun --bind-to none -np <ntmpi> gmx_mpi mdrun` (without `--bind-to none` OpenMPI
pins each rank to one core and throughput drops noticeably). No multi-GPU profile ships: this path has not been run on
this image, so treat it as untested.

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -e AIM_INPUT_PATH=/input -e AIM_OUTPUT_PATH=/output -e AIM_ACCELERATOR_COUNT=4 \
  -v /host/input:/input:ro -v /host/output:/output <image>
```

## Benchmark inputs

The image ships no input. Benchmark systems (`adh_dodec`, `cellulose_nve`, `stmv`) can be downloaded from
[AMD InfinityHub GROMACS](https://www.amd.com/en/developer/infinity-hub/gromacs.html) and turned into a `run.tpr`.

## CI inputs

The job-profile CI runs the published image twice, on the inputs in `image/tests/ci-input/` (see the README there):
`small/run.tpr`, a tiny valid system that must exit 0 and leave `gromacs.log` plus `md.*` files in `$AIM_OUTPUT_PATH`,
and `broken/run.tpr`, a file that is not a run input, which must exit non-zero (`1`) with `gromacs.log` present.

The CI sets `AIM_INPUT_PATH`, `AIM_OUTPUT_PATH` (under `/artifacts`), `AIM_ACCELERATOR_COUNT`,
`AIM_ACCELERATOR_MODEL`, the profile's `env_vars` and the profile's `engine_args` as `AIM_ENGINE_ARGS`. A case's
`<case>.env.yaml` can set `AIM_ENGINE_ARGS` itself: it overrides the profile **key by key** and keeps the keys it does
not set (see [docs/engine-args-override.md](../../../../../docs/engine-args-override.md)). `small.env.yaml` uses this to
set `nsteps` to 500, so the run finishes in seconds instead of running the whole `.tpr` (the profile has `nsteps: -2`),
while `nstlist` and `ntmpi` stay as in the profile. Inside the image, the precedence is command-line flag >
`AIM_ENGINE_ARGS` > default.
