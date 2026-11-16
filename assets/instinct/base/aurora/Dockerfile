# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Base image for the Aurora weather models: the AIM runtime and the weather
# harness on top of the engine image at assets/instinct/engine/aurora/.

# ARGs before FROM are used in the FROM instruction
# these must be overridden via --build-arg
ARG PARENT_REGISTRY_PREFIX=dummy_parent_registry_prefix
ARG PARENT_REPOSITORY=dummy_parent_repository
ARG PARENT_TAG=dummy_parent_tag

# Downloads and converts the ERA5 input files; only the results are kept.
FROM ${PARENT_REGISTRY_PREFIX}/${PARENT_REPOSITORY}:${PARENT_TAG} AS era5-inputs

# The bucket holds 1959-01-01 to 2023-01-10.
ARG ERA5_START=2022-06-01
ARG ERA5_DAYS=10
ENV PYTHONUNBUFFERED=1

COPY assets/instinct/base/aurora/src/data/ /opt/era5/

RUN pip install --no-cache-dir -r /opt/era5/requirements.txt && \
    python3 /opt/era5/fetch_weatherbench2.py \
        --start "${ERA5_START}" --days "${ERA5_DAYS}" --outdir /tmp/era5 && \
    python3 /opt/era5/make_aurora_input.py \
        --source-dir /tmp/era5 --outdir /opt/era5/inputs --verify-with-aurora && \
    ln -s "${ERA5_START}_input.nc" /opt/era5/inputs/aurora_input.nc && \
    rm -rf /tmp/era5

FROM ${PARENT_REGISTRY_PREFIX}/${PARENT_REPOSITORY}:${PARENT_TAG}

# Re-declare ARGs after FROM to use them in subsequent instructions
ARG PARENT_REGISTRY_PREFIX
ARG PARENT_REPOSITORY
ARG PARENT_TAG

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace/aim-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"
ENV AIM_CACHE_PATH="/workspace/model-cache"
ENV HF_HOME="${AIM_CACHE_PATH}"
ENV HF_DATASETS_CACHE="${AIM_CACHE_PATH}"
ENV HF_HUB_DISABLE_TELEMETRY="1"
ENV AIM_UPSTREAM_IMAGE_REF=${PARENT_REGISTRY_PREFIX}/${PARENT_REPOSITORY}:${PARENT_TAG}
ENV AIM_ALLOW_GENERAL_PROFILE_FALLBACK="true"
ENV AIM_ACCELERATOR_FAMILY="instinct"
ENV AIM_ACCELERATOR_TYPE="gpu"
ENV AIM_CONFIG_PATH="/workspace/aim-runtime/config"

# AIMRuntime.__init__ resolves AIM_ENGINE against engines.yaml before profile
# selection, so it has to be set here rather than read from the profile.
ENV AIM_ENGINE=bentoml

RUN echo "AIM Upstream Image: ${AIM_UPSTREAM_IMAGE_REF}"

# Create workspace directory
WORKDIR /workspace

# Copy AIM Runtime source code (only runtime packages, not developer tooling)
COPY src/aim_runtime/ /workspace/aim-runtime/src/aim_runtime/
COPY src/aim_common/ /workspace/aim-runtime/src/aim_common/
COPY requirements/requirements.txt /workspace/aim-runtime/
COPY requirements/${AIM_ACCELERATOR_FAMILY}-requirements.txt /workspace/aim-runtime/
COPY src/entrypoint.py /workspace/entrypoint.py
RUN chmod +x /workspace/entrypoint.py

ARG AIM_BASE_CONFIG_DIR=assets/instinct/base/aurora/config

COPY ${AIM_BASE_CONFIG_DIR} /workspace/aim-runtime/config

# Install runtime dependencies
RUN cd /workspace/aim-runtime && \
    pip3 install -r requirements.txt && \
    pip3 install -r ${AIM_ACCELERATOR_FAMILY}-requirements.txt

# Extra dependencies shipped in the config directory copied above.
RUN if [ -f "${AIM_CONFIG_PATH}/requirements-extra.txt" ]; then \
        echo "Installing base extra requirements from ${AIM_CONFIG_PATH}/requirements-extra.txt"; \
        pip3 install --no-cache-dir -r "${AIM_CONFIG_PATH}/requirements-extra.txt"; \
    fi

# Ten days of ERA5 input, one file per day. Modified Copernicus data; its
# attribution is in /workspace/NOTICE.md.
COPY --from=era5-inputs /opt/era5/inputs/ /workspace/inputs/
ENV AIM_INPUT_PATH=/workspace/inputs

# The harness. Discovery imports /workspace/model/src/harness.py, which imports
# weather_forecast_harness and the engine's config module from the same
# directory.
COPY assets/instinct/base/aurora/src/ /workspace/model/src/

# Create cache directories
RUN mkdir -p /workspace/model-cache

# Copy License files (placed late to preserve build cache for expensive preceding layers)
COPY LICENSE /workspace/LICENSE
# Attribution and licences for the ERA5 inputs and the model weights.
COPY assets/instinct/base/aurora/NOTICE.md /workspace/NOTICE.md

ENTRYPOINT ["./entrypoint.py"]
