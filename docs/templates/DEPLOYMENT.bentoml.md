<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AMD Inference Microservice deployment guide

This guide gives step-by-step instructions to deploy the AMD Inference Microservice (AIM) container for the
{{ aim_deployment.model_name }} model in different environments. Follow these instructions to run the model on AMD
accelerators. This guide assumes a {{ aim_deployment.accelerator_model }} accelerator on the target system.

This AIM has an API of its own, shaped by what the model takes and returns. See
[Testing your deployment](#4-testing-your-deployment) for the endpoints and payloads of this model.

## Prerequisites

{% if aim_deployment.accelerator_family.value == "instinct" %}
* AMD Instinct™ GPU with ROCm support (e.g., MI300X, MI325X)
{% elif aim_deployment.accelerator_family.value == "radeon" %}
* AMD Radeon™ Pro GPU with ROCm support (e.g., W7900, R9700)
{% elif aim_deployment.accelerator_family.value == "epyc" %}
* AMD EPYC™ CPU (e.g., EPYC_ZEN4, EPYC_ZEN5, EPYC_9965)
{% endif %}
{% if aim_deployment.accelerator_type.value == "gpu" %}
* Docker installed and configured with GPU support
{% elif aim_deployment.accelerator_type.value == "cpu" %}
* Docker installed and configured
{% endif %}
* A host directory for the model cache, if the model downloads its weights at first start
{% if aim_deployment.hf_token %}
* Access to model repositories (Hugging Face account with appropriate permissions for gated models)
{% endif %}

## 1. Docker deployment

### 1.1 Running the container

Pin the profile explicitly, so that the selection cannot depend on the hardware the runtime detects. Set
`AIM_PROFILE_ID` to one of the profiles listed in the [README](README.md), and set `AIM_ACCELERATOR_COUNT` to the
accelerator count of that profile.

```bash
mkdir -p /path/to/model-cache

docker run \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
  -e AIM_ID={{ aim_deployment.model_name }} \
{% if aim_deployment.profile_name_to_override_automatic_selection %}
  -e AIM_PROFILE_ID={{ aim_deployment.profile_name_to_override_automatic_selection }} \
{% else %}
  -e AIM_PROFILE_ID=<PROFILE_ID> \
{% endif %}
  -e AIM_ACCELERATOR_COUNT={{ aim_deployment.accelerator_count }} \
  -v /path/to/model-cache:/workspace/model-cache \
{% if aim_deployment.accelerator_type.value == "gpu" %}
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
{% endif %}
  -p 8000:8000 \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
```
{% if aim_deployment.hf_token %}

Where <YOUR_HUGGINGFACE_TOKEN> is your Hugging Face access token (required for gated models).
{% endif %}

`AIM_ACCELERATOR_COUNT` must agree with the `accelerator_count` of the selected profile. Set an integer, not `auto`,
because some AIMs read the value directly.

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_ID={{ aim_deployment.model_name }} \
{% if aim_deployment.profile_name_to_override_automatic_selection %}
  -e AIM_PROFILE_ID={{ aim_deployment.profile_name_to_override_automatic_selection }} \
{% else %}
  -e AIM_PROFILE_ID=<PROFILE_ID> \
{% endif %}
  -e AIM_ACCELERATOR_COUNT={{ aim_deployment.accelerator_count }} \
  -e AIM_CACHE_PATH=/workspace/model-cache \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_PORT=8080 \
  -v /path/to/model-cache:/workspace/model-cache \
{% if aim_deployment.accelerator_type.value == "gpu" %}
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
{% endif %}
  -p 8080:8080 \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
```

## 2. Model caching for production

This AIM has no separate model download step. If the model downloads its weights, it does so at service
startup and keeps them in `$AIM_CACHE_PATH`, which defaults to `/workspace/model-cache`.

Mount a persistent volume at that path, as shown in section 1.1. Without the mount, the container downloads the
weights again at each start.

If the cache must be at a different path, set `AIM_CACHE_PATH` and mount the volume at the same path:

```bash
docker run \
  -e AIM_ID={{ aim_deployment.model_name }} \
{% if aim_deployment.profile_name_to_override_automatic_selection %}
  -e AIM_PROFILE_ID={{ aim_deployment.profile_name_to_override_automatic_selection }} \
{% else %}
  -e AIM_PROFILE_ID=<PROFILE_ID> \
{% endif %}
  -e AIM_ACCELERATOR_COUNT={{ aim_deployment.accelerator_count }} \
  -e AIM_CACHE_PATH=/mnt/aim-cache \
  -v /path/to/model-cache:/mnt/aim-cache \
{% if aim_deployment.accelerator_type.value == "gpu" %}
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
{% endif %}
  -p 8000:8000 \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
```

The first start is slower than later starts, because it downloads the weights. Size the volume for the model
checkpoint, and keep it writable.

## 3. Kubernetes deployment

### 3.1 Deployment

It is possible to deploy AIM with Kubernetes. This section gives a sample Kubernetes deployment manifest.

{% if aim_deployment.hf_token %}
Create a secret for the Hugging Face token used to download the model:

```bash
kubectl create secret generic hf-token \
    --from-literal="hf-token=<YOUR_HUGGINGFACE_TOKEN>" \
    -n <YOUR_K8S_NAMESPACE>
```

{% endif %}
Create `deployment.yaml` with the following content:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minimal-aim-deployment
  labels:
    app: minimal-aim-deployment
spec:
  progressDeadlineSeconds: 3600
  replicas: 1
  selector:
    matchLabels:
      app: minimal-aim-deployment
  template:
    metadata:
      labels:
        app: minimal-aim-deployment
    spec:
      containers:
        - name: minimal-aim-deployment
          image: {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
          imagePullPolicy: Always
          env:
            - name: AIM_ID
              value: "{{ aim_deployment.model_name }}"
            - name: AIM_PROFILE_ID
{% if aim_deployment.profile_name_to_override_automatic_selection %}
              value: "{{ aim_deployment.profile_name_to_override_automatic_selection }}"
{% else %}
              value: "<PROFILE_ID>"
{% endif %}
            - name: AIM_ACCELERATOR_COUNT
              value: "{{ aim_deployment.accelerator_count }}"
            - name: AIM_CACHE_PATH
              value: "/workspace/model-cache"
            - name: AIM_LOG_LEVEL_ROOT
              value: "INFO"
            - name: AIM_LOG_LEVEL
              value: "INFO"
            - name: AIM_PORT
              value: "8000"
            {% if aim_deployment.hf_token %}
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: hf-token
            {% endif %}
          ports:
            - name: http
              containerPort: 8000
          resources:
            requests:
              memory: "64Gi"
              {% if aim_deployment.accelerator_type.value == "gpu" %}
              cpu: "8"
              amd.com/gpu: "{{ aim_deployment.accelerator_count }}"
              {% elif aim_deployment.accelerator_type.value == "cpu" %}
              cpu: "{{ aim_deployment.accelerator_count }}"
              {% endif %}
            limits:
              memory: "64Gi"
              {% if aim_deployment.accelerator_type.value == "gpu" %}
              cpu: "8"
              amd.com/gpu: "{{ aim_deployment.accelerator_count }}"
              {% elif aim_deployment.accelerator_type.value == "cpu" %}
              cpu: "{{ aim_deployment.accelerator_count }}"
              {% endif %}
          startupProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 30
          volumeMounts:
            - name: model-cache
              mountPath: /workspace/model-cache
            - name: ephemeral-storage
              mountPath: /tmp
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: minimal-aim-model-cache
        - name: ephemeral-storage
          emptyDir:
            sizeLimit: 256Gi
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
```

The startup probe gives the start up to one hour: 360 checks at 10-second intervals. That covers a first start that
downloads weights, as described in section 2. The pod stays out of the service endpoints until this probe succeeds,
so no request reaches a model that is still loading. A later start reads the cache and succeeds much sooner.

Kubernetes runs the liveness probe only after the startup probe succeeds. The liveness probe then accepts 30 failed
checks, which is five minutes, before it restarts the container. This tolerates a model that does more work after the
health endpoint answers.

The `memory` and `cpu` values above are a starting point. Raise them if the profile uses more than one accelerator,
because this AIM starts one worker process per accelerator, and each worker holds its own copy of
the model.

Create the cache volume claim in `pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minimal-aim-model-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

Create a service configuration in the `service.yaml` file:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: minimal-aim-deployment
  labels:
    app: minimal-aim-deployment
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 80
      targetPort: 8000
  selector:
    app: minimal-aim-deployment
```

Customize the deployment further to fit your needs. Apply it to the Kubernetes cluster with `kubectl`:

```bash
kubectl apply -f . -n <YOUR_K8S_NAMESPACE>
```

## 4. Testing your deployment

To test the deployment, execute an API call. First find the pod, then set up port forwarding.

```bash
kubectl port-forward service/minimal-aim-deployment 8000:80 -n <YOUR_K8S_NAMESPACE>
```

After port forwarding starts, the API is available locally at `http://localhost:8000/`.

### 4.1 Health check

```bash
curl http://localhost:8000/healthz
```

The service answers with HTTP 200 when it is ready to accept requests. The startup probe and the liveness probe in
section 3.1 use it.

`/healthz` answers a `GET`. An AIM can also give an endpoint that reports the state in more detail. That endpoint
takes a `POST`, and the section below names it if this AIM has one.

{# skip_lines drops the file's copyright header, which pre-commit requires in assets/. #}
{{ read_file('templates/API.md') | skip_lines(6) | trim }}

## 5. Advanced deployment scenarios

### 5.1 Selecting a different profile

Each profile of this AIM is pinned with `AIM_PROFILE_ID`. The profile identifier is the filename of the profile
without the `.yaml` extension. The [README](README.md) lists the available profiles and their accelerator counts.

```bash
docker run \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
  -e AIM_ID={{ aim_deployment.model_name }} \
  -e AIM_PROFILE_ID=<PROFILE_ID> \
  -e AIM_ACCELERATOR_COUNT=<ACCELERATOR_COUNT_OF_THAT_PROFILE> \
  -v /path/to/model-cache:/workspace/model-cache \
{% if aim_deployment.accelerator_type.value == "gpu" %}
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
{% endif %}
  -p 8000:8000 \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
```

`AIM_ACCELERATOR_COUNT` must match the `accelerator_count` of the selected profile. If the two disagree, the
container stops at startup with a profile selection error.

## 6. Monitoring and troubleshooting

### 6.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }} \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }} \
  <subcommand> --help
```

### 6.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
{% if aim_deployment.hf_token %}
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
{% endif %}
  -e AIM_ID={{ aim_deployment.model_name }} \
{% if aim_deployment.profile_name_to_override_automatic_selection %}
  -e AIM_PROFILE_ID={{ aim_deployment.profile_name_to_override_automatic_selection }} \
{% else %}
  -e AIM_PROFILE_ID=<PROFILE_ID> \
{% endif %}
  -e AIM_ACCELERATOR_COUNT={{ aim_deployment.accelerator_count }} \
  -v /path/to/model-cache:/workspace/model-cache \
{% if aim_deployment.accelerator_type.value == "gpu" %}
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
{% endif %}
  -p 8000:8000 \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }}
```

### 6.3 Checking profile selection results

It is possible to check which profile AIM selects from the given environment variables.

```bash
docker run \
  -e AIM_ID={{ aim_deployment.model_name }} \
  -e AIM_ACCELERATOR_COUNT={{ aim_deployment.accelerator_count }} \
  -e AIM_PRECISION=fp16 \
{% if aim_deployment.accelerator_family.value == "instinct" %}
  -e AIM_ACCELERATOR_MODEL=MI300X \
{% elif aim_deployment.accelerator_family.value == "radeon" %}
  -e AIM_ACCELERATOR_MODEL=W7900 \
{% elif aim_deployment.accelerator_family.value == "epyc" %}
  -e AIM_ACCELERATOR_MODEL=EPYC_9965 \
{% endif %}
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }} \
  dry-run
```

### 6.4 List available profiles

```bash
docker run \
  {{ aim_deployment.docker_info.registry_namespace }}/{{ aim_deployment.repository }}:{{ aim_deployment.tag }} \
  list-profiles
```

### 6.5 Common problems

* **The container restarts during the first start.** The manifest in section 3.1 gives the start one hour. If the
  model needs more, raise `failureThreshold` on the startup probe. Mount a persistent cache, so that later starts
  skip the download.
* **The container restarts after it answers on `/healthz`.** The liveness probe accepts five minutes of failed
  checks. If the model does more work after the health endpoint answers, and that work takes longer, raise
  `failureThreshold` on the liveness probe.
* **The container stops with a profile selection error.** `AIM_PROFILE_ID` and `AIM_ACCELERATOR_COUNT` must agree.
* **`/healthz` does not answer yet.** The service binds the port only after the model is loaded. Read the container
  logs to see how far startup got.

## 7. Security considerations

{% if aim_deployment.hf_token %}
* Never include HF_TOKEN in Dockerfiles or commit it to version control
{% endif %}
* Use Kubernetes secrets or environment variables for sensitive credentials
* Implement appropriate network policies to restrict access to your deployment
* The model cache holds downloaded model weights. Restrict write access to it
