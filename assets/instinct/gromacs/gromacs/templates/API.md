<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

<!-- This file is named API.md by convention, but it documents a job container: this AIM serves no HTTP API and opens no port. -->

### 4.1 Overview

The container is a long-running job. It runs one `gmx mdrun` on a GROMACS run-input file
(`.tpr`) with the non-bonded, PME and update work on the GPU and the bonded work on the CPU (default, see `bonded_cpu`), reads `AIM_INPUT_PATH`, writes `AIM_OUTPUT_PATH`,
streams the run log to the container's stdout and exits. The image entrypoint is `python3 /app/src/entrypoint.py`. Start
it with no arguments and configure it with the environment (section 4.3); the optional command-line arguments `--input`,
`--output` and the tunable flags of section 4.3 take precedence over the environment. There is no wall-clock limit: bound
a run with `nsteps`.
The GROMACS build is version 2026.4 (fork string `2026-HIP-ENABLEMENT`), a HIP enablement development commit (`924b96a7ac69a2e84cf2b73374f82d1f2eeab68a`), not a release; which GPU offload combinations work depends on this version. The AIM is force-field agnostic: the force field, topology and all simulation parameters come from the `.tpr` you supply.

### 4.2 Input

`$AIM_INPUT_PATH` is required and is only read, never written, so it can be mounted read-only. It must be an existing
directory containing the run-input file.

| File | Required | Content |
|---|---|---|
| `run.tpr` | yes | The GROMACS run-input file; create it with `gmx grompp`. The `tpr` setting (section 4.3) names another file inside the directory. |

If the directory or the file is missing the container exits with status `1` before starting mdrun. There is no
`settings.json`: settings come from the environment and the command line only. Weights are not input; the `.tpr` file
holds the system, and no `AIM_CACHE_PATH` is read (it is not a supported variable; if it is set, it is logged as a warning and ignored).

### 4.3 Settings

Each tunable is a key of the JSON object in `AIM_ENGINE_ARGS` or a command-line flag. The command line wins over
`AIM_ENGINE_ARGS`, and `AIM_ENGINE_ARGS` wins over the default. An unknown key, a bad
command-line argument and an `AIM_ENGINE_ARGS` value that is not a JSON object (for example a flag string) are errors
(exit status `2`). An unknown `AIM_*` environment variable is logged as a warning and ignored, so a launcher may set
variables for every AIM. Keys may be written with `-` or `_`; integer values may be given as JSON integers or numeric strings.

| `AIM_ENGINE_ARGS` key | CLI flag | Default | Meaning |
|---|---|---|---|
| `ntmpi` | `--ntmpi` | `AIM_ACCELERATOR_COUNT`, else detected GPUs | GPUs / MPI ranks, 1 to 9. |
| `ntomp` | `--ntomp` | `min(24, usable hardware threads - 1)`, divided by `ntmpi` (at least 1) | OpenMP threads per rank, at least 1. The default caps the total at 24 and always leaves one hardware thread free for the HIP runtime thread, because one OpenMP thread per hardware thread starves it. Set a value to override it. |
| `nstlist` | `--nstlist` | `200` | Neighbor-list update interval. |
| `nsteps` | `--nsteps` | `-2` | Steps to run. A negative value, including `-1`, passes no `-nsteps` to mdrun, so the `.tpr` value applies; unlike `gmx mdrun -nsteps -1`, a `-1` here does not mean "run without a step limit". |
| `gpu-id` | `--gpu-id` | the first `ntmpi` GPUs (`0`, `01`, ...) | GPU id string, for example `0123`. |
| `tpr` | `--tpr` | `run.tpr` | Input file name inside `$AIM_INPUT_PATH`. |
| `pme_tune` | `--pme-tune` / `--no-pme-tune` | `false` | Boolean (`true`/`false`). `false` passes `-notunepme` to mdrun; `true` lets mdrun auto-tune the PME grid. See the note below. |
| `bonded_cpu` | `--bonded-cpu` / `--no-bonded-cpu` | `true` | Boolean. `true` passes `-bonded cpu` (bonded interactions on the CPU), `false` passes `-bonded gpu`. The default is `true`. It only moves where the bonded forces are computed; it does not change the simulation setup. |

Notes on `pme_tune`, `bonded_cpu` and `nstlist`:

- `pme_tune` defaults to `false` (`-notunepme`) because PME auto-tuning costs time on short runs. For long production
  runs set `pme_tune` to `true`.
- `bonded_cpu` defaults to `true`; on other systems or GPUs set it to `false` to compute the bonded forces on the GPU.
- `nstlist` (default `200`) is a simulation-setup parameter (it changes the pair-list buffer that GROMACS derives from
  the Verlet buffer tolerance) and is not changed silently.

Other `AIM_*` variables (any other variable starting with `AIM_` is logged as a warning and ignored, as in section 4.3):

| Variable | Meaning |
|---|---|
| `AIM_INPUT_PATH` | Required (or `--input`). The input directory. |
| `AIM_OUTPUT_PATH` | Required (or `--output`). The output directory; it is created if missing. |
| `AIM_ACCELERATOR_COUNT` | Number of GPUs to use, 1 to 9; overrides device detection. |
| `AIM_ACCELERATOR_MODEL` | Informational, only written to the log. |
| `AIM_ENGINE_ARGS` | JSON object with the tunables above, for example `{"ntomp": 24, "nstlist": 150}`. Default `{}`. |

GPUs are detected by counting `/dev/dri/renderD*` when `/dev/kfd` exists; when `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` is set, the number of devices listed caps that count. The run fails with status `2` if no GPU is
found and `ntmpi` is not set, if `ntmpi` exceeds the available GPUs, or if `AIM_ACCELERATOR_COUNT` exceeds the detected
devices (when at least one is detected). The image ships both a thread-MPI build (`gmx`) and an OpenMPI build (`gmx_mpi`), built from the same commit. With `ntmpi` of 1 the AIM runs the thread-MPI `gmx` with `-ntmpi 1`. With more
than one it runs `mpirun --bind-to none -np <ntmpi> gmx_mpi mdrun`, for example with `-e AIM_ACCELERATOR_COUNT=4`. Only single-GPU profiles ship: the multi-GPU path has not been run on this image, so
treat it as untested.

#### CPU FFT and licence build

This image is built without FFTW (GPL): the CPU FFT library is the bundled `fftpack` (`-DGMX_FFT_LIBRARY=fftpack`), and the
bundled Colvars is disabled (`-DGMX_USE_COLVARS=NONE`). GPU PME uses the bundled VkFFT and is not affected. The AIM always
runs with `-pme gpu`, `-nb gpu` and `-update gpu` (and `-bonded cpu` by default, `-bonded gpu` with `bonded_cpu` false); it never sets `-pme cpu` or `-pmefft cpu`, so the CPU FFT
is not on the default path. Consequences:

- A `.tpr` whose system GROMACS cannot run with PME on the GPU (for example LJ-PME, or some free-energy setups with perturbed
  charges) fails at start-up, because `-pme gpu` is forced; there is no automatic fall back to the CPU PME.
- Other setups that GROMACS does not support with `-pme gpu` or `-update gpu` also fail at start-up with an mdrun error, for example
  a system without PME electrostatics (reaction-field or Martini), a `pme-order` other than 4, an integrator other than `md`
  (steep, cg, l-bfgs, md-vv, bd, nm), Nosé-Hoover or MTTK coupling, and SHAKE constraints. The list is not exhaustive: the
  mdrun start-up error is authoritative.
- Running `gmx mdrun` yourself with `-pme cpu` or `-pmefft cpu` uses fftpack, which is much slower than FFTW.
- A `.mdp` with `colvars = yes` errors at `gmx grompp` or `gmx mdrun`.

The job needs `--ipc=host` (section 4.9).

### 4.4 Output

All results are written to `$AIM_OUTPUT_PATH`, which is created if it does not exist.

| Path | When | Content |
|---|---|---|
| `DONE` | success, written last | Completion marker (contains `0`). |
| `gromacs.log` | live, from the start of the run | mdrun output, the command line and the AIM's own notes; also streamed to stdout. |
| `md.log` | success | mdrun log (contains the performance summary). |
| `md.edr` | success | Energy file. |
| `md.cpt` | success | Final checkpoint (full state at the last step). |
| `md.gro` | success | Final structure (mdrun's end-of-run configuration, named `md.gro` because mdrun runs with `-deffnm md`). |
| `md.xtc` | success, only if the `.mdp` writes one | Compressed trajectory. |
| `.work/` | during the run | Resume state and intermediate files. Not a result. |

There is no `summary.json`: the completion signals are the `DONE` marker and the exit status (section 4.6), and the
performance summary is in `md.log`. The final structure is `md.gro` and the final full state (coordinates and velocities) is in `md.cpt`.

* mdrun runs with `-deffnm` in the `$AIM_OUTPUT_PATH/.work` directory. Only when mdrun exits with status 0 are the
  regular files in `.work` moved into `$AIM_OUTPUT_PATH` (one `os.replace` each), after which `DONE` is written. A reader
  can treat any moved file as complete and `DONE` as "the set is complete". On failure or interruption nothing is moved
  and no `DONE` is written; the partial files stay in `.work`.
* `gromacs.log` is the one file appended in place while the job runs, so use `DONE` or the exit status as the
  completion signal. It is created once the settings are valid and before the input is checked, so a missing input
  directory or `.tpr` (exit `1`) is recorded in it as well as on stderr. Invalid settings (exit `2`, nothing started)
  do not create it; those errors appear only on the container's stderr. Output of earlier runs in the same directory
  stays in the file.
* mdrun lines are written to both stdout and `gromacs.log`. The AIM's own log messages go to stderr.

### 4.5 Filesystem needs

mdrun appends to `md.log` and `md.edr`, seeks in `md.xtc` and `md.cpt`, and renames the checkpoint, so
`$AIM_OUTPUT_PATH` must be a POSIX filesystem (local disk, NFS, Lustre, GPFS). Object-store or FUSE mounts without
append, seek and atomic rename are not supported. Publishing renames files from `.work` into the output directory, so
keep `$AIM_OUTPUT_PATH` and its `.work` subdirectory on one filesystem and do not mount a separate volume at `.work`.
The input directory can be read-only.

### 4.6 Exit status

| Code | Meaning |
|---|---|
| `0` | mdrun exited 0; results published and `DONE` written. |
| `1` | Input directory or `.tpr` missing; mdrun failed with status 1; no usable GROMACS binary was found; or an unexpected error in the runner. |
| `2` | Invalid settings: bad `AIM_ENGINE_ARGS`, bad command-line argument, missing `AIM_INPUT_PATH` or `AIM_OUTPUT_PATH`, invalid or insufficient GPU count. Nothing was started. |
| `127` | A GROMACS or `mpirun` executable could not be started (the operating system reported it as not found). |
| `128+n` | Stopped by signal `n`: `143` for SIGTERM, `130` for SIGINT; also used when mdrun itself was killed by signal `n`. |
| other | mdrun's own non-zero exit status is passed through unchanged. |

The cause of a failure is in `gromacs.log` (the last 30 lines of mdrun output are repeated there) and on stderr.

### 4.7 Resume

Resume is supported at checkpoint granularity. Start the container again with the **same `$AIM_OUTPUT_PATH`** (and the
same settings). mdrun runs in `$AIM_OUTPUT_PATH/.work` with `-deffnm md`. When `.work/md.cpt` exists, the runner passes it to mdrun
with `-cpi`, so the run continues from that checkpoint. The runner passes `-cpi` explicitly because without it `mdrun -deffnm md` starts a new run from step 0 even when `md.cpt` exists (tested with GROMACS 2026.4 on CPU after SIGTERM and SIGKILL, thread-MPI binary only). Resume also needs `md.log` and `md.edr` from the earlier run to still be present in `.work`, and `-nsteps` counts total steps from step 0, so a resumed run does the remaining steps.

* GROMACS writes a checkpoint every 15 minutes by default, so a hard kill loses at most that much.
* Without a surviving `.work` directory the run starts from step 0.
* Starting the container removes any existing `DONE` marker first. Re-running after a successful run therefore starts
  from step 0 (the successful run removed `.work`), recomputes the simulation and, on success, overwrites the published
  files and writes `DONE` again. If that re-run fails, the older published files remain in place but without `DONE`.
* A successful run that reuses an output directory also removes the `md.*` files of the earlier run that it did not
  produce itself (for example an old `md.xtc` when the new run writes no trajectory), after the new set is in place
  and before `DONE` is written, so the directory never mixes two simulations. A resumed run is not affected: it
  publishes its own complete set. `gromacs.log` and files that are not `md.*` are never removed. For a new run that
  must keep earlier results, use a fresh directory.
* There is no wall-clock limit; bound a run with `nsteps`.

### 4.8 Interruption

On SIGTERM or SIGINT the job forwards the signal to the whole mdrun/mpirun process group (a signal that arrives before
mdrun has started is forwarded as soon as it starts), mdrun stops at the next step and writes a checkpoint into `.work`,
the results are not published (no `DONE`), and the container exits `128+n` for the first signal `n` received, even if
mdrun itself exited with 0. If the process group has not exited 60 seconds after the first forwarded signal, it is
killed with SIGKILL. The same handling applies whichever scheduler delivers the signal. `docker stop` sends SIGTERM, so
set its timeout above 60 seconds (`docker stop -t 90`) if the checkpoint matters; likewise set the Kubernetes
`terminationGracePeriodSeconds` (30 seconds by default) above 60. SIGKILL cannot be caught, so only the last periodic
checkpoint remains.

### 4.9 Running the job

First run (the number of steps is bounded by `nsteps`, here set through `AIM_ENGINE_ARGS`):

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AIM_INPUT_PATH=/input -e AIM_OUTPUT_PATH=/output \
  -e 'AIM_ENGINE_ARGS={"ntomp": 24, "nstlist": 150, "nsteps": 10000}' \
  -v /host/input:/input:ro -v /host/output:/output \
  <image>
```

Follow progress with `docker logs -f <container>`; the same text is written to `/host/output/gromacs.log`. On success
the results are in `/host/output` together with the `DONE` marker.

Resume: run the same command again against the same `/host/output`. Check the result with `cat /host/output/DONE`
(it contains `0`) and the exit status with `echo $?`.
