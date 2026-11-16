<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice job deployment guide

This guide explains how to run the cytokineking/freebindcraft AIM as a long-running job container on an
MI300X system. A job AIM has no HTTP API: it reads its inputs from a mounted directory,
writes its results to a mounted directory, streams its log to stdout, and exits with a status code.

## Prerequisites

* AMD Instinct GPU with ROCm support, and Docker configured for it
* A host directory with the job input and a writable host directory for the results
  (a local or POSIX filesystem; see section 4 for the filesystem requirements)

## Running the job

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -e AIM_INPUT_PATH=/input \
  -e AIM_OUTPUT_PATH=/output \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  amdenterpriseai/aim-instinct-model-cytokineking-freebindcraft:7.2.0-preview
```

The number of GPUs used, how tunables are passed, and any workload-specific container flags are described in section 4,
together with the input, output, exit codes, and how to resume or stop a run.

## Status and termination

* Follow progress with `docker logs -f <container>`. The log file and result files written to `$AIM_OUTPUT_PATH` are
  described in section 4.
* Exit status `0` means the run finished and its results are in the output directory. A non-zero status means failure
  or interruption; section 4 lists the exit statuses of this job and how a stopped run is resumed.

## Other schedulers

The image entrypoint runs the job, so neither recipe sets a command. Pass the settings and tunables of section 4.3 as
environment variables, and use the volumes and the exit status as in the sections above.

### Kubernetes Job

One GPU, input and output on persistent volumes:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: aim-instinct-model-cytokineking-freebindcraft
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      # Keep this above the time the job needs to stop (section 4.8).
      terminationGracePeriodSeconds: 90
      containers:
        - name: job
          image: amdenterpriseai/aim-instinct-model-cytokineking-freebindcraft:7.2.0-preview
          env:
            - {name: AIM_INPUT_PATH, value: /input}
            - {name: AIM_OUTPUT_PATH, value: /output}
          resources:
            limits:
              amd.com/gpu: 1
          volumeMounts:
            - {name: input, mountPath: /input, readOnly: true}
            - {name: output, mountPath: /output}
      volumes:
        - name: input
          persistentVolumeClaim: {claimName: job-input}
        - name: output
          persistentVolumeClaim: {claimName: job-output}
```

The pod has no `command:`: the image starts the run. Kubernetes sends `SIGTERM` when the pod is deleted and waits
`terminationGracePeriodSeconds` (30 seconds by default) before `SIGKILL`; section 4.8 states how long the job needs to
stop. To resume after an interruption, start the job again against the same output volume (section 4.7).

### Apptainer and Slurm

```bash
apptainer run --rocm --bind /path/to/input:/input:ro --bind /path/to/output:/output \
  --env AIM_INPUT_PATH=/input --env AIM_OUTPUT_PATH=/output \
  docker://amdenterpriseai/aim-instinct-model-cytokineking-freebindcraft:7.2.0-preview
```

Wrap the command in an `sbatch` script that requests the GPU with `--gres=gpu:1` and bounds the run with `--time`.
Slurm sends `SIGTERM` when the allocation ends; section 4.8 states what the job does then.

## 4. Job reference

<!-- This file is named API.md by convention, but it documents a job container: this AIM serves no HTTP API and opens no port. -->

### 4.1 Overview

The container is a long-running job. Start it with **no arguments**: it reads
`AIM_INPUT_PATH`, designs binders, writes `AIM_OUTPUT_PATH` and exits. The only accepted arguments are `-h` and `--help`
(print usage, exit `0`); any other argument is rejected with exit status `2`. A run takes minutes to many hours, so
always bound it with `AIM_FBC_TIME_LIMIT` and/or `AIM_FBC_MAX_TRAJECTORIES`.

### 4.2 Input

`AIM_INPUT_PATH` is required and is only read, never written.

| File | Required | Content |
|---|---|---|
| `target.pdb` | yes | The target structure. It must be non-empty and have ATOM records with C-alpha atoms. Every chain in `AIM_FBC_TARGET_CHAINS`, and every chain letter named in `AIM_FBC_HOTSPOTS`, must be present. |
| `settings.json` | no | A JSON object. Its keys are the setting names in section 4.3 without the `AIM_FBC_` prefix, in lower case. |

Weights are not input and are not in the image. Mount a directory that contains `params/params_model_1_multimer_v3.npz`
(the whole AlphaFold2 `alphafold_params_2022-12-06.tar` extracted into `params/`, CC BY 4.0: all the files in the
archive, including the five multimer parameter files `params_model_1_multimer_v3.npz` to `params_model_5_multimer_v3.npz`
and the monomer models; do not pick single files) and set `AIM_CACHE_PATH` to it. The job checks only for
`params_model_1_multimer_v3.npz`; if that file is missing it exits with status `2` and a message that names the expected path.

`FBC_AF2_CACHE_PATH` is not read by the job. It is used only by the GPU smoke test shipped with this asset
(`image/tests/smoke/test_job_gpu_freebindcraft.py`): the operator sets it to the host directory that holds
`params/params_model_*_multimer_v3.npz`, and the test mounts it read-only at `/af2` with `AIM_CACHE_PATH=/af2`. The test
checks that the variable is set and that the directory holds `params/params_model_1_multimer_v3.npz` before it starts
Docker, and fails with a message that names the problem otherwise.

#### Preparing the target

Always trim `target.pdb` to the smallest region that contains the binding site: a smaller target speeds up the design
and lowers the GPU memory needed (upstream BindCraft README, "Always try to trim the input target PDB to the smallest
size possible"). Before the run:

* Remove waters, ligands and other HETATM records, and keep one clean conformation (no alternate locations, one model).
* Keep only the domain or chains the binder should bind, with their original residue numbering, because `AIM_FBC_HOTSPOTS`
  refers to those numbers.
* Hotspots are optional. A small patch of several target residues narrows the search; with none, the design tool picks
  the binding site itself (upstream: "AF2 is very good at detecting good binding sites if no hotspot is specified").
  The job logs how many of the hotspot residue numbers it finds in `target.pdb` and lists any it does not find; a number
  that is missing does not stop the run, so check that line in the log.

### 4.3 Settings

Each setting is an `AIM_FBC_*` environment variable or a key in `settings.json`. The environment wins over the file, and
the file wins over the default. An unknown `settings.json` key, an unknown `AIM_*` variable, and the deprecated
`AIM_GPU_COUNT` and `AIM_GPU_MODEL` are errors (exit status `2`). A non-empty environment value overrides the file; an
empty one is ignored.

| Environment variable | `settings.json` key | Default | Meaning |
|---|---|---|---|
| `AIM_FBC_TARGET_CHAINS` | `target_chains` | `A` | Chain(s) of `target.pdb` to bind, for example `A` or `A,B` (single-character chain ids). |
| `AIM_FBC_HOTSPOTS` | `hotspots` | empty | Hotspot residues, comma separated; each item is a residue number or range, optionally with a chain letter in front, or a chain letter alone: `56`, `1,2-10`, `A1-10,B1-20` or `A`. Empty means no preference, and the design tool picks the site. A bare chain letter targets the whole chain (`A`, or `A,B1-10` mixed with residue items); it can be useful for a small target, but for a large chain it is no better than leaving the hotspots empty. Chain letters must exist in `target.pdb`. Syntax after the upstream BindCraft README (`target_hotspot_residues`). |
| `AIM_FBC_BINDER_NAME` | `binder_name` | `binder` | Prefix of the design names (letters, digits, `_`, `.`, `-`; up to 64 characters). |
| `AIM_FBC_MIN_LENGTH` | `min_length` | `80` | Minimum binder length, 31 to 300. The default `80` to `150` range covers the usual size of a designed binder; very short binders (peptide range) are a different regime and may be rejected by the default filters (section 4.4). |
| `AIM_FBC_MAX_LENGTH` | `max_length` | `150` | Maximum binder length, 31 to 300, not below `min_length`. |
| `AIM_FBC_NUM_DESIGNS` | `num_designs` | `200` | Number of accepted designs at which the run stops and ranks them (1 or more). A default run stops after 200 accepted designs, so it runs longer than with fewer. Upstream recommends at least 100 accepted designs, from which the top 5 to 20 are taken for experiments; screen more when high-affinity binders are required. |
| `AIM_FBC_MAX_TRAJECTORIES` | `max_trajectories` | no cap | Stop after this many relaxed trajectories (1 or more; `0` or empty means no cap). |
| `AIM_FBC_TIME_LIMIT` | `time_limit` | no limit | Wall-clock cap: seconds, or a number with `s`, `m`, `h` or `d` (for example `90m`, `2h`). `0`, `none`, `off`, `unlimited` or empty means no limit. |
| `AIM_FBC_MAX_RESTARTS` | `max_restarts` | `20` | Restarts of the design process after it crashes by itself (0 or more; `0` disables restarts). |
| `AIM_FBC_PREFLIGHT_ONLY` | `preflight_only` | `0` | Check-only mode for CI and diagnostics (`1` or `0`). The job validates the settings, `target.pdb` and the GPU, writes `bindcraft.log` and `summary.json` (`status` `preflight_ok`, no designs) under `AIM_OUTPUT_PATH` and exits `0`. It needs no AlphaFold2 parameters and creates no `.work/` directory. Invalid settings, input or a missing GPU still exit `2` or `3`. |

Other `AIM_*` variables:

| Variable | Meaning |
|---|---|
| `AIM_INPUT_PATH` | Required. The input directory. |
| `AIM_OUTPUT_PATH` | Required. The output directory; it is created if missing. |
| `AIM_CACHE_PATH` | Optional. Directory with operator-supplied AlphaFold2 parameters (section 4.2). |
| `AIM_ACCELERATOR_COUNT` | Overrides the detected GPU count (an integer, 0 or more). |
| `AIM_ACCELERATOR_MODEL` | Logged only. |
| `AIM_ENGINE_ARGS` | Accepted and ignored, with a warning in the log: the profiles of this AIM have no engine arguments, so nothing is applied. |
| `AIM_ID`, `AIM_MODEL_ID`, `AIM_PROFILE_ID`, `AIM_ENGINE`, `AIM_PRECISION`, `AIM_METRIC`, `AIM_ACCELERATOR_TYPE`, `AIM_ACCELERATOR_FAMILY`, `AIM_LOG_LEVEL`, `AIM_LOG_LEVEL_ROOT` | Accepted and ignored. |

The resolved settings are logged and written to the `settings` field of `summary.json`. Example `settings.json`:

```json
{
  "target_chains": "A",
  "hotspots": "A56",
  "min_length": 80,
  "max_length": 150,
  "num_designs": 200,
  "time_limit": "12h"
}
```

The CI smoke test uses a short binder (31 to 40 residues) and one design only to keep the run short; it is not a
recommended design setup.

The job needs one AMD GPU. It counts `/dev/dri/renderD*` nodes when `/dev/kfd` exists, and exits with status `3` if there
are none; extra GPUs are not used. Set `AIM_ACCELERATOR_COUNT` only where the nodes cannot be counted.

### 4.4 Output

Everything is written under `AIM_OUTPUT_PATH`.

| Path | Content |
|---|---|
| `summary.json` | Machine-readable result, written at the start, about every 30 seconds while the run is in progress, and at the end. `finished: true` means the run is over. |
| `bindcraft.log` | Full log: the job's `[aim-fbc ...]` lines and the design tool's output, appended as it happens and also streamed to stdout. |
| `Accepted/*.pdb` | Accepted designs: designs that pass all the filters of upstream `settings_filters/default_filters.json`, which this job uses unchanged, on the AlphaFold2 re-prediction. The job runs upstream with `--no-pyrosetta`, so the filters are of two kinds, listed below. The full list is in that file; the failure counts per filter are in `failure_csv.csv`. |
| `Accepted/Ranked/*.pdb` | Accepted designs ranked by `Average_i_pTM`; exists once `num_designs` accepted designs exist. |
| `trajectory_stats.csv`, `mpnn_design_stats.csv`, `failure_csv.csv`, `rejected_mpnn_full_stats.csv`, `final_design_stats.csv` | Score tables and filter-failure counters of the design tool. `final_design_stats.csv` exists once ranking has happened. |
| `.work/` | Resume state and intermediate files. Not a result. |

Filters computed from the structures. On the average over the five models (the first two models also individually):
pLDDT above 0.8, pTM above 0.55, ipTM above 0.5, i_pAE below 0.35, shape complementarity above 0.6, dSASA above 1, at
least 7 interface residues, surface hydrophobicity below 0.35, binder loop content below 90 %, binder RMSD below 3.5 and
hotspot RMSD below 6 angstrom.

Filters that pass by construction. Without PyRosetta, upstream (`functions/pr_alternative_utils.py`, the version this image
pins) does not compute four interface properties but substitutes fixed values chosen to satisfy the default thresholds:
the number of interface hydrogen bonds (at least 3), the number of unsatisfied interface hydrogen bonds (at most 4), the
binder energy score (below 0) and the interface dG (below 0). The four filters still run and are counted in
`failure_csv.csv`, but they never reject a design. An accepted design is therefore not evidence that it has at least 3
interface hydrogen bonds, at most 4 unsatisfied ones, a negative binder energy or a negative interface dG, and the values
of those four columns in the score tables are the placeholders, not measurements. Check these properties with a separate
tool (for example PyRosetta) before relying on them.

Result files are published under a temporary dot-name and renamed, so a reader never sees a half-written file.
`bindcraft.log` is the one file appended in place.

The upstream design animations (an HTML5 video of each trajectory) are off, because the image does not ship ffmpeg. This
changes nothing in the designs, the score tables or the plots; only the optional animation files are not produced.

`summary.json` fields:

| Field | Meaning |
|---|---|
| `schema_version` | `1`. |
| `status` | `running`, `completed`, `max_trajectories_reached`, `time_limit`, `interrupted`, `stopped_early` or `failed`. |
| `finished` | `true` when the run is over. |
| `exit_code` | The exit status of the container; `null` while running. |
| `started_at`, `finished_at` | UTC timestamps; `finished_at` is `null` while running. |
| `wall_time_s` | Wall-clock seconds of this invocation. |
| `time_to_first_trajectory_s` | Seconds from the start of the design process to the end of the first trajectory's hallucination, JAX start-up and compilation included; `null` before that. |
| `trajectories` | Rows of `trajectory_stats.csv`. |
| `trajectories_relaxed` | Relaxed trajectory structures. |
| `trajectories_rejected_early` | Trajectories rejected before relaxation, from `failure_csv.csv`. |
| `per_trajectory_times_s`, `mean_trajectory_time_s` | Time of each scored trajectory and their mean. |
| `mpnn_designs_scored`, `mpnn_designs_rejected` | Rows of `mpnn_design_stats.csv` and `rejected_mpnn_full_stats.csv`. |
| `accepted_designs`, `ranked_designs`, `requested_designs` | Accepted and ranked design counts, and `num_designs`. |
| `resumed` | `true` when this invocation continued earlier work. |
| `restarts` | Restarts after crashes. |
| `crash_causes` | One entry per crash: `attempt`, `exit_code`, `last_line`, `new_result_rows`, `elapsed_s`. |
| `accelerators` | Detected GPU count; `null` if the run failed before detection. |
| `settings` | The resolved settings. |
| `notes` | Human-readable remarks, for example the reason for a failure. |

The profile metric is `throughput`: accepted designs per GPU-hour over a fixed wall-clock budget (`AIM_FBC_TIME_LIMIT`),
that is `accepted_designs` divided by `wall_time_s`, scaled to hours. `summary.json` reports the time to first trajectory
separately as startup cost (`time_to_first_trajectory_s`) and the per-trajectory time as the mean over scored
trajectories (`mean_trajectory_time_s`).

ipTM (the `Average_i_pTM` ranking column) is a good yes/no predictor of whether a binder binds the target, not of how
strongly (upstream BindCraft README). Ranking by it does not rank affinity; test the top designs experimentally.

### 4.5 Filesystem needs

`AIM_OUTPUT_PATH` must be a POSIX filesystem with create and rename within a directory, append, rewrite in place
(the design tool rewrites its CSVs), and many small files (`.work/`). An object-store mount without rename or append does
not work. The output directory must be writable by the container user (root by default). The state folder holds many small files; its size
grows with the number of trajectories and accepted designs.

### 4.6 Exit status

| Code | Meaning |
|---|---|
| `0` | Finished. Either `num_designs` designs were accepted (`completed`), or the `max_trajectories` cap was reached (`max_trajectories_reached`; possibly fewer designs than requested, so read `accepted_designs`). Also `-h` and `--help`, and a check-only run (`AIM_FBC_PREFLIGHT_ONLY=1`) whose checks passed (`preflight_ok`). |
| `1` | The run failed: the design process failed, reported no GPU, scored no trajectory, or kept crashing (more crashes than `AIM_FBC_MAX_RESTARTS`, or three in a row with no new result), or the work directory could not be prepared. Partial results are kept. If the design process itself exits with another non-zero status, that status is passed through. |
| `2` | Invalid input or settings, a refused resume (section 4.7), unexpected arguments, or `AIM_INPUT_PATH` or `AIM_OUTPUT_PATH` unset or unusable. Nothing was started. `summary.json` is written when `AIM_OUTPUT_PATH` is usable. |
| `3` | No AMD GPU found. |
| `4` | The design tool stopped early with fewer designs than requested and no cap reached (`stopped_early`, its low-acceptance-rate guard). |
| `124` | `AIM_FBC_TIME_LIMIT` was reached (`time_limit`). Not a finished run, but the partial results are usable. |
| `128+n` | Stopped by signal `n`: `143` for SIGTERM, `130` for SIGINT, `129` for SIGHUP. If the design process alone is killed by a signal (for example `137` for SIGKILL from the out-of-memory killer), the job treats it as a crash and may restart it (section 4.7). If the job itself is SIGKILLed, no final `summary.json` is written. |

### 4.7 Resume

Resume is supported at trajectory granularity. Start the container again with the **same `AIM_OUTPUT_PATH`** and the same
`target.pdb`, chains and hotspots. The design tool re-reads its CSVs and result folders under `.work/`, so earlier
trajectories and accepted designs count towards the new run's targets, and `summary.json` reports `resumed: true`.

* A different target, chain set or hotspot set against an output directory that already holds work is refused with
  exit status `2`.
* `wall_time_s` and `AIM_FBC_TIME_LIMIT` apply to each invocation separately.
* Raise `AIM_FBC_NUM_DESIGNS` or `AIM_FBC_MAX_TRAJECTORIES` to extend a finished run.
* Use an empty output directory for a new run.

Within one run, when the design process crashes by itself the job restarts it and it resumes the same way, at most
`AIM_FBC_MAX_RESTARTS` times; the time limit applies across restarts. Three crashes in a row without a new result give
up. A signal, the time limit and invalid input are never retried.

### 4.8 Interruption

On SIGTERM, SIGINT or SIGHUP the job forwards the signal to the whole process group of the design process (SIGINT as
SIGINT, the others as SIGTERM), publishes the results, writes `summary.json` with status `interrupted`, and exits
`128+n`. If the design process has not stopped 20 seconds after the request, it is killed with SIGKILL. The same
20-second grace applies after `AIM_FBC_TIME_LIMIT` is reached (SIGTERM, then SIGKILL). There is no checkpoint on signal:
the trajectory in flight is lost and the finished ones are kept. The same handling applies whichever scheduler delivers
the signal. The default `docker stop` timeout (10 seconds) is shorter than
this grace, so use `docker stop -t 30`; the Kubernetes default of 30 seconds is enough for the publish step. SIGKILL cannot be caught, so only the last periodic publish remains.

### 4.9 Running the job

`/host/af2` is the directory that holds `params/params_model_*_multimer_v3.npz` (section 4.2); the job exits with status `2`
without it.

First run (one design, bounded):

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -e AIM_INPUT_PATH=/input -e AIM_OUTPUT_PATH=/output -e AIM_CACHE_PATH=/af2 \
  -e AIM_FBC_NUM_DESIGNS=1 -e AIM_FBC_MAX_TRAJECTORIES=1 -e AIM_FBC_TIME_LIMIT=2h \
  -v /host/input:/input:ro -v /host/output:/output \
  -v /host/af2:/af2:ro \
  <image>
```

With a `settings.json` in `/host/input` and a time limit from the environment:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -e AIM_INPUT_PATH=/input -e AIM_OUTPUT_PATH=/output -e AIM_CACHE_PATH=/af2 -e AIM_FBC_TIME_LIMIT=12h \
  -v /host/input:/input:ro -v /host/output:/output \
  -v /host/af2:/af2:ro \
  <image>
```

Resume: run the same command again against the same `/host/output`. Check the result with
`cat /host/output/summary.json` and the exit status with `echo $?`.

On a scheduler with a time bound (for example Slurm `--time`), set the bound above `AIM_FBC_TIME_LIMIT`.