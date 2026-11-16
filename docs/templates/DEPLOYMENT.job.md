<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice job deployment guide

This guide explains how to run the {{ aim_deployment.model_name }} AIM as a long-running job container on an
{{ aim_deployment.accelerator_model }} system. A job AIM has no HTTP API: it reads its inputs from a mounted directory,
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
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
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
  name: {{ aim_deployment.repository }}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      # Keep this above the time the job needs to stop (section 4.8).
      terminationGracePeriodSeconds: 90
      containers:
        - name: job
          image: {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
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
  docker://{{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
```

Wrap the command in an `sbatch` script that requests the GPU with `--gres=gpu:1` and bounds the run with `--time`.
Slurm sends `SIGTERM` when the allocation ends; section 4.8 states what the job does then.

## 4. Job reference

{# skip_lines drops the file's copyright header, which pre-commit requires in assets/. #}
{{ read_file('templates/API.md') | skip_lines(6) | trim }}
