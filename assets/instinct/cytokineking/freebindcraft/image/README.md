<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# FreeBindCraft AIM: Long-running job container

[FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) designs miniprotein
binders against a target protein structure: AlphaFold2 hallucination (ColabDesign),
ProteinMPNN sequence design, AlphaFold2 re-prediction and filtering. This image runs
it in `--no-pyrosetta` mode (OpenMM relaxation, sc-rs shape complementarity, FreeSASA),
so it is fully open source, on AMD Instinct GPUs (JAX 0.8.2 / ROCm 7.2.2).

This is a job, not a service:
there is no HTTP endpoint and no port. Running the image with **no arguments** starts the
run; it reads its input, writes its output and exits. A run takes minutes to many hours.
A single trajectory alone can take many minutes on one GPU, so always set a bound
(`AIM_FBC_TIME_LIMIT` and/or `AIM_FBC_MAX_TRAJECTORIES`, below).

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

`/host/af2` is the directory that holds `params/params_model_*_multimer_v3.npz` (see below); the job exits with status 2
without it.

Under Kubernetes use a Pod with no `command:` and the same paths mounted (including the AlphaFold2 directory); under
Slurm use `apptainer run` or `docker run` as above. The image is the same everywhere.

## Inputs (`AIM_INPUT_PATH`, read-only)

| File | Required | Content |
|---|---|---|
| `target.pdb` | yes | The target protein structure: a real, folded structure with ATOM records (a PDB or an AlphaFold model). Every requested target chain must be present. |
| `settings.json` | no | Run settings as a JSON object; keys are the setting names in the table below without the `AIM_FBC_` prefix, lower case. |

The directory is never written to. Weights never come from `AIM_INPUT_PATH`: the
The AlphaFold2 parameters (CC BY 4.0, about 5.6 GB) are not baked into the image: mount a directory that holds
`params/params_model_*_multimer_v3.npz` (from `https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`)
and set `AIM_CACHE_PATH` to it, e.g. `-v /data/af2:/cache:ro -e AIM_CACHE_PATH=/cache`. Without it the run exits 2
with a message naming the missing file.

## Settings

Settings are `AIM_*` environment variables and/or keys in `settings.json`. The environment
wins over the file, the file over the default. **A setting the job does not know is an
error** (exit 2): an unknown key in `settings.json` and any unknown `AIM_*` variable are
rejected, as are the deprecated `AIM_GPU_COUNT` / `AIM_GPU_MODEL`. The one exception is `AIM_ENGINE_ARGS`, which a platform
may set from a profile's `engine_args`: the profiles of this AIM declare none, so it is ignored with a warning in the log.

| Environment variable | `settings.json` key | Default | Meaning |
|---|---|---|---|
| `AIM_FBC_TARGET_CHAINS` | `target_chains` | `A` | Chain(s) of `target.pdb` to bind, e.g. `A` or `A,B`. |
| `AIM_FBC_HOTSPOTS` | `hotspots` | none | Hotspot residues to target, chain letter plus numbers, e.g. `A56` or `A1,B20-25`. Empty: no preference. |
| `AIM_FBC_BINDER_NAME` | `binder_name` | `binder` | Prefix of the design names. |
| `AIM_FBC_MIN_LENGTH` / `AIM_FBC_MAX_LENGTH` | `min_length` / `max_length` | `80` / `150` | Binder length range, 31 or more (upstream miniprotein mode). |
| `AIM_FBC_NUM_DESIGNS` | `num_designs` | `200` | Number of accepted designs after which the run stops and ranks them. |
| `AIM_FBC_MAX_TRAJECTORIES` | `max_trajectories` | no cap | Stop after this many relaxed trajectories. Written into a copy of upstream's advanced-settings JSON (upstream has no CLI flag). Upstream counts only trajectories that reached relaxation, not early rejects. |
| `AIM_FBC_TIME_LIMIT` | `time_limit` | no limit | Wall-clock cap, enforced by the wrapper (upstream has none): seconds or `90s`, `30m`, `2h`, `1d`. |
| `AIM_FBC_PREFLIGHT_ONLY` | `preflight_only` | `0` | Check-only mode (CI and diagnostics): validate the settings, `target.pdb` and GPU visibility, write `bindcraft.log` and `summary.json` (`status` `preflight_ok`) under `AIM_OUTPUT_PATH`, exit `0`. No AlphaFold2 parameters are needed, no `.work/` is created and nothing is designed. Bad settings, input or no GPU still exit `2` or `3`. |
| `AIM_FBC_MAX_RESTARTS` | `max_restarts` | `20` | Upstream's AlphaFold2 step occasionally crashes on a NaN (seed and length dependent, nondeterministic). When `bindcraft.py` exits non-zero on its own, the wrapper restarts it, and upstream resumes from the CSVs and directories already written. At most this many restarts (`0` disables); the time limit applies across restarts. Three crashes in a row with no new trajectory result give up early. Never retried: our own SIGTERM/SIGINT/SIGHUP, the time limit, invalid settings or input. |
| `AIM_INPUT_PATH` / `AIM_OUTPUT_PATH` | | required | Directories described here. |
| `AIM_CACHE_PATH` | | none | Optional AlphaFold2 parameter location (see above). |
| `AIM_ACCELERATOR_COUNT` | | detected | Overrides the detected GPU count. |

The upstream design animations (an HTML5 video per trajectory) are switched off: the image does not ship ffmpeg. Designs, score
tables and plots are unchanged; only the optional animation files are not produced.
| `AIM_ACCELERATOR_MODEL` | | none | Logged only. |

The platform variables `AIM_ID`, `AIM_MODEL_ID`, `AIM_PROFILE_ID`, `AIM_ENGINE`,
`AIM_PRECISION`, `AIM_METRIC`, `AIM_ACCELERATOR_TYPE`, `AIM_ACCELERATOR_FAMILY`,
`AIM_LOG_LEVEL` and `AIM_LOG_LEVEL_ROOT` are accepted and ignored.

## Hardware

The job needs one AMD GPU. It counts the device nodes it was given (`/dev/dri/renderD*`
when `/dev/kfd` exists) rather than reading a visible-devices variable, and fails with exit
status 3 and a clear message if there is none. `AIM_ACCELERATOR_COUNT` overrides detection
for platforms where the nodes cannot be counted. One GPU is used; extra GPUs are not, and no RDMA or interconnect is needed (the profiles carry no `hardware_requirements` block: it is not part of the profile schema). The count is
of device nodes, so a platform that exposes every node (for example `docker run --device=/dev/dri` on a
shared host) shows all of them; the GPU ROCm actually uses is the one the scheduler allocated
(for example via `ROCR_VISIBLE_DEVICES` set by the scheduler or `docker run -e`; the wrapper itself does not set or pin it).

## Outputs (`AIM_OUTPUT_PATH`)

| Path | Content |
|---|---|
| `summary.json` | Machine-readable result, always written (see below). `finished: true` means the run is over. |
| `bindcraft.log` | The full log: the wrapper's `[aim-fbc ...]` lines and the upstream output, appended live. Also streamed to stdout. Written for failed runs and for runs that failed before starting. |
| `Accepted/*.pdb` | Accepted designs (best model per design). |
| `Accepted/Ranked/*.pdb` | The accepted designs ranked by `Average_i_pTM`; written once `AIM_FBC_NUM_DESIGNS` accepted designs exist. |
| `trajectory_stats.csv`, `mpnn_design_stats.csv`, `failure_csv.csv`, `final_design_stats.csv`, `rejected_mpnn_full_stats.csv` | Upstream's score tables and filter-failure counters. `final_design_stats.csv` exists once ranking happened. |
| `.work/` | Resume state and all intermediate files of the upstream tool (not a result). |

Result files are published while the run is in progress, about every 30 seconds, and once
more at the end. Each one is written under a temporary dot-name and renamed, so a reader
of the directory never sees a half-written result file. `bindcraft.log` is the one file
that is appended in place; `summary.json` says whether the run is finished.

`summary.json` fields: `status` (`running`, `completed`, `max_trajectories_reached`,
`time_limit`, `interrupted`, `stopped_early`, `failed`), `finished`, `exit_code`,
`wall_time_s`, `time_to_first_trajectory_s` (from the start of the upstream process to the
end of the first trajectory's hallucination, whether or not it then passed upstream's confidence checks; JAX
start-up and compilation included),
`trajectories` (rows of `trajectory_stats.csv`), `trajectories_relaxed`,
`trajectories_rejected_early`, `per_trajectory_times_s`, `mean_trajectory_time_s`,
`mpnn_designs_scored` (rows of `mpnn_design_stats.csv`), `mpnn_designs_rejected` (rows of
`rejected_mpnn_full_stats.csv`: sequences that failed the filters before or after scoring), `accepted_designs`, `ranked_designs`, `requested_designs`,
`resumed`, `restarts` (number of restarts after crashes), `crash_causes` (per crash: `attempt`, `exit_code`, `last_line`, `new_result_rows`, `elapsed_s`), `accelerators`, `settings`, `notes`.

## Headline metric

The profiles declare `metric: throughput`: accepted designs per GPU-hour, measured over a fixed
wall-clock budget set with `AIM_FBC_TIME_LIMIT` (`accepted_designs` divided by `wall_time_s` in
`summary.json`, scaled to hours). `time_to_first_trajectory_s` (JAX start-up and compilation included)
is reported separately as the startup cost, not as the profile metric. The per-trajectory time is
`mean_trajectory_time_s`, the mean over the scored trajectories.

## Exit status

| Code | Meaning |
|---|---|
| `0` | Finished: the requested number of designs was accepted (`completed`), or the `max_trajectories` cap was reached (`max_trajectories_reached`, possibly with fewer designs than requested; read `accepted_designs`). Also a check-only run whose checks passed (`preflight_ok`). |
| `1` | The run failed (upstream error, upstream found no GPU, or a crashing upstream exhausted `AIM_FBC_MAX_RESTARTS` or crashed three times in a row without progress; partial results are kept). |
| `2` | Invalid input or settings; nothing was started. |
| `3` | No AMD GPU found. |
| `4` | Upstream stopped early with fewer designs than requested and no cap hit (its low-acceptance-rate guard). |
| `124` | The `AIM_FBC_TIME_LIMIT` wall-clock cap was reached. Not a finished run: the partial results published so far are usable. |
| `128+n` | Killed by signal `n`: `143` for SIGTERM, `130` for SIGINT, `137` for SIGKILL. |

## What a capped or interrupted run leaves

Everything that was finished stays usable: the CSVs, the accepted designs found so far,
`summary.json` (with `status` `time_limit` or `interrupted`, the trajectories scored,
accepted designs, wall time, time to first trajectory and the per-trajectory times) and
the log. `Accepted/Ranked` is only produced when the requested number of designs was
reached; for a capped run rank `Accepted/*.pdb` from `mpnn_design_stats.csv`.

## Interrupt

On SIGTERM, SIGINT or SIGHUP the wrapper forwards the signal to the whole process group of
the upstream run, publishes the results, writes `summary.json` and exits `128+signal`
(143 for SIGTERM). If the run has not stopped 20 seconds after the request it is killed.
Upstream has no checkpoint-on-signal: the trajectory that was in flight is lost, finished
ones are kept. Kubernetes sends SIGTERM and waits `terminationGracePeriodSeconds`
(default 30 seconds), which is enough for the publish step. SIGKILL (for example an
out-of-memory kill) cannot be caught: the last periodic publish remains.

## Resume

Yes, at trajectory granularity. Start the container again with the **same
`AIM_OUTPUT_PATH`** and the same `target.pdb`, chains and hotspots: upstream re-reads its
CSVs and result folders under `.work/` and carries on, so earlier trajectories and
accepted designs count toward the new run's targets, and `summary.json` reports
`resumed: true`. A different target, chain set or hotspot set against an output
directory that already holds work is refused (exit 2) rather than mixed. `wall_time_s`
and the time limit apply to each invocation separately. Raise `AIM_FBC_NUM_DESIGNS` or
`AIM_FBC_MAX_TRAJECTORIES` to extend a finished run. Use an empty output directory for a
new run.

## Filesystem needs

`AIM_OUTPUT_PATH` must be a POSIX filesystem that supports: create/rename within a
directory (every published file), append and flush (`bindcraft.log`), rewrite in place
(upstream rewrites its CSVs), and many small files (the folder `.work/` is the resume state). An object-store mount without
rename or append will not work. The state folder holds many small files; its size
grows with the number of trajectories and accepted designs. Files are written as the container user
(root by default). `AIM_INPUT_PATH` can be read-only.

## Smoke input

`tests/references/1PGB_A.pdb` is chain A of PDB 1PGB (the 56-residue B1 domain of
streptococcal protein G), a real folded target (strip the leading `#` license comment lines before
using it as `target.pdb`). A recommended first run:
`AIM_FBC_MIN_LENGTH=31 AIM_FBC_MAX_LENGTH=40 AIM_FBC_NUM_DESIGNS=1 AIM_FBC_MAX_TRAJECTORIES=1
AIM_FBC_TIME_LIMIT=2h`.

## CI inputs

`tests/ci-input/` holds the two inputs of the run-to-completion check for job AIMs; `tests/` is excluded
from the image by `.dockerignore`, so they are not part of the released image. CI stages each case into
`AIM_INPUT_PATH`, sets `AIM_OUTPUT_PATH` under `/artifacts`, the accelerator variables and the case's `.env.yaml`, and
starts the image's own entrypoint on a GPU runner.

| Case | Input | `.env.yaml` | Expected |
|---|---|---|---|
| `small` | `target.pdb` (1PGB chain A) and `settings.json` (31-45 residue binder, one design) | `AIM_FBC_PREFLIGHT_ONLY: "1"` | exit 0, `bindcraft.log` and `summary.json` |
| `broken` | a `target.pdb` that is not a structure | `AIM_FBC_PREFLIGHT_ONLY: "1"` | exit 2 (invalid input), `bindcraft.log` and `summary.json` |

The CI runner has no AlphaFold2 parameters and cannot spend hours designing, so the `small` case runs in check-only mode
(`AIM_FBC_PREFLIGHT_ONLY=1`, above): the image validates its settings, target and GPU visibility, writes the log and
summary and exits 0 without designing. The whole input is a few tens of KB, far below the CI limit of 190,000 base64
characters per case. The profiles declare `engine_args: {}`, so CI sets no `AIM_ENGINE_ARGS`; if an operator or a case
sets one, the job logs a warning and ignores it (a case value would override a profile value key by key, per
`docs/engine-args-override.md`, but there is no key this image reads). The job reads no Hugging Face token.
