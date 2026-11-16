# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# GROMACS AIM — long-running job container
#
# Layer 0: rocm/dev-ubuntu-24.04:7.2.4  (Ubuntu 24.04 + ROCm 7.2.4 runtime stack)
#           Pinned 2026-10-20: tag 7.2.4 resolved on Docker Hub to digest
#           sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
#           (newest 7.2.x tag; the digest is recorded here, the tag is what config.yaml and ARG use).
#           Vendor fallback — no specialized aim-base available for GROMACS yet
#           (see config.yaml: specialized: false).
# Layer 1: This image — GROMACS binaries built from source (HIP/ROCm,
#           pinned commit on branch 4947-hip-feature-enablement; HIP dev-branch commit, not a
#           release; CMake reports GROMACS 2026.4, fork string 2026-HIP-ENABLEMENT) + entrypoint.
#           No BentoML, no HTTP server, no port.
#
# Licence variant V1: GPL FFTW is not built against or shipped. The CPU FFT is the bundled fftpack
# (GMX_FFT_LIBRARY=fftpack; GPU PME uses the bundled MIT VkFFT, unaffected) and the bundled Colvars is
# disabled (GMX_USE_COLVARS=NONE). A build-time check fails the build if gmx/libgromacs link FFTW.
#
# Two GROMACS builds are produced:
#   /opt/gromacs-tmpi/bin/gmx     — thread-MPI build (-DGMX_MPI=OFF -DGMX_THREAD_MPI=ON)
#                                   Used for ntmpi=1 (single rank); no system MPI needed
#   /opt/gromacs-mpi/bin/gmx_mpi  — OpenMPI build (-DGMX_MPI=ON)
#                                   Used for ntmpi>1 via `mpirun --bind-to none`
#
# mpirun --bind-to none is required: without it OpenMPI pins each rank to 1 core
# and all OpenMP threads fight for it (throughput drops noticeably).
#
# Contract:
#   - reads $AIM_INPUT_PATH (directory containing a .tpr file)
#   - runs gmx mdrun
#   - writes all output to $AIM_OUTPUT_PATH
#   - exits 0 on success / non-zero on failure
#   - log streamed to stdout and $AIM_OUTPUT_PATH/gromacs.log, written even on failure
#   - 128+N exit status when stopped by signal N

# ---- Stage 0: Build GROMACS thread-MPI build (gmx, no system MPI) ----
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4
FROM ${BASE_IMAGE} AS gromacs-tmpi-builder

# Install build dependencies for GROMACS + HIP
# rocprim/rocthrust/hipcub are not part of rocm/dev-ubuntu-24.04; GROMACS' HIP build needs their CMake packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libhwloc-dev \
        rocprim-dev \
        rocthrust-dev \
        hipcub-dev \
        git \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make /opt/rocm/lib visible to the linker (needed for hipDeviceGetStreamPriorityRange)
RUN echo '/opt/rocm/lib' > /etc/ld.so.conf.d/rocm.conf && ldconfig

# Fetch GROMACS at an immutable commit (HIP feature branch 4947-hip-feature-enablement as of
# 2026-10-16). A moving branch name would let the two builds differ; pin the SHA instead.
ARG GROMACS_REPO=https://gitlab.com/gromacs/gromacs.git
ARG GROMACS_REF=924b96a7ac69a2e84cf2b73374f82d1f2eeab68a
RUN git init /src/gromacs && cd /src/gromacs \
    && git remote add origin "${GROMACS_REPO}" \
    && git fetch --depth=1 origin "${GROMACS_REF}" \
    && git checkout --detach FETCH_HEAD

# Build thread-MPI GROMACS (single-GPU, no system MPI dependency)
# -DGMX_MPI=OFF -DGMX_THREAD_MPI=ON: thread-MPI only (produces "gmx" binary)
# -DCMAKE_INSTALL_PREFIX=/opt/gromacs-tmpi
RUN mkdir -p /src/gromacs/build-tmpi && cd /src/gromacs/build-tmpi && \
    cmake .. \
        -G Ninja \
        -DGMX_GPU=HIP \
        -DGMX_MPI=OFF \
        -DGMX_THREAD_MPI=ON \
        -DGMX_DOUBLE=OFF \
        -DGMX_FFT_LIBRARY=fftpack \
        -DGMX_USE_COLVARS=NONE \
        -DGMX_SIMD=AVX2_256 \
        -DGMX_GPU_NB_DISABLE_CLUSTER_PAIR_SPLIT=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/gromacs-tmpi \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_HIP_ARCHITECTURES="gfx942" \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
    && ninja -j$(nproc) install \
    && rm -rf /src/gromacs/build-tmpi \
    && ( bins=$(find /opt/gromacs-tmpi -type f \( -name gmx -o -name gmx_mpi -o -name 'libgromacs*.so*' \)) \
         && test -n "$bins" && ! ldd $bins | grep -i fftw ) \
    && dpkg-query -W -f='${Package}=${Version}\n' > /opt/gromacs-tmpi/build-packages.txt

# ---- Stage 1: Build GROMACS OpenMPI build (gmx_mpi, multi-rank) ----
FROM ${BASE_IMAGE} AS gromacs-mpi-builder

# Install build dependencies including OpenMPI
# rocprim/rocthrust/hipcub are not part of rocm/dev-ubuntu-24.04; GROMACS' HIP build needs their CMake packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libhwloc-dev \
        rocprim-dev \
        rocthrust-dev \
        hipcub-dev \
        git \
        wget \
        ca-certificates \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Make /opt/rocm/lib visible to the linker (needed for hipDeviceGetStreamPriorityRange)
RUN echo '/opt/rocm/lib' > /etc/ld.so.conf.d/rocm.conf && ldconfig

# Same pinned commit as the thread-MPI stage (same ARG default), so both binaries match
ARG GROMACS_REPO=https://gitlab.com/gromacs/gromacs.git
ARG GROMACS_REF=924b96a7ac69a2e84cf2b73374f82d1f2eeab68a
RUN git init /src/gromacs && cd /src/gromacs \
    && git remote add origin "${GROMACS_REPO}" \
    && git fetch --depth=1 origin "${GROMACS_REF}" \
    && git checkout --detach FETCH_HEAD

# Build OpenMPI GROMACS (multi-rank, produces "gmx_mpi" binary)
# -DGMX_MPI=ON: real MPI (requires mpirun at runtime)
# -DCMAKE_INSTALL_PREFIX=/opt/gromacs-mpi
RUN mkdir -p /src/gromacs/build-mpi && cd /src/gromacs/build-mpi && \
    cmake .. \
        -G Ninja \
        -DGMX_GPU=HIP \
        -DGMX_MPI=ON \
        -DGMX_THREAD_MPI=OFF \
        -DGMX_DOUBLE=OFF \
        -DGMX_FFT_LIBRARY=fftpack \
        -DGMX_USE_COLVARS=NONE \
        -DGMX_SIMD=AVX2_256 \
        -DGMX_GPU_NB_DISABLE_CLUSTER_PAIR_SPLIT=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/gromacs-mpi \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_HIP_ARCHITECTURES="gfx942" \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
    && ninja -j$(nproc) install \
    && rm -rf /src/gromacs/build-mpi \
    && ( bins=$(find /opt/gromacs-mpi -type f \( -name gmx -o -name gmx_mpi -o -name 'libgromacs*.so*' \)) \
         && test -n "$bins" && ! ldd $bins | grep -i fftw ) \
    && dpkg-query -W -f='${Package}=${Version}\n' > /opt/gromacs-mpi/build-packages.txt

# ---- Stage 2: AIM Layer-1 image (vendor base — Ubuntu 24.04 + ROCm) ----
# No aim-runtime needed (no BentoML, no harness, no serve — container runs and exits)
FROM ${BASE_IMAGE}

# Install runtime deps: OpenMPI (for mpirun at runtime), hwloc (no FFTW: licence variant V1)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhwloc15 \
        ca-certificates \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Make /opt/rocm/lib visible to the dynamic linker in the runtime stage
RUN echo '/opt/rocm/lib' > /etc/ld.so.conf.d/rocm.conf && ldconfig

ENV LD_LIBRARY_PATH=/opt/rocm/lib

# Exact apt package versions (from the pinned base image's Ubuntu repositories at build time).
# build-packages.txt of each builder is copied with its install prefix.
RUN dpkg-query -W -f='${Package}=${Version}\n' > /opt/aim-build-packages.txt

# No Python dependencies: entrypoint.py / runner.py / utils.py use only the standard library.

# ---- GROMACS binary installation ----
# Thread-MPI build: gmx (no system MPI dependency)
COPY --from=gromacs-tmpi-builder /opt/gromacs-tmpi /opt/gromacs-tmpi
# OpenMPI build: gmx_mpi (for mpirun multi-rank runs)
COPY --from=gromacs-mpi-builder /opt/gromacs-mpi /opt/gromacs-mpi

# Set PATH: mpi build first (gmx_mpi), then tmpi (gmx)
RUN cat /opt/gromacs-tmpi/build-packages.txt /opt/gromacs-mpi/build-packages.txt \
    | sort -u > /opt/gromacs-build-packages.txt

ENV PATH="/opt/gromacs-mpi/bin:/opt/gromacs-tmpi/bin:${PATH}"

# ---- Application source (entrypoint) ----
RUN mkdir -p /app/src
COPY src/entrypoint.py /app/src/entrypoint.py
COPY src/runner.py     /app/src/runner.py
COPY src/utils.py      /app/src/utils.py

# Add /app/src to sys.path
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /app/src > {}/gromacs-aim.pth'

WORKDIR /app

# No HTTP server, no port, just an entrypoint that runs and exits
ENTRYPOINT ["python3", "/app/src/entrypoint.py"]
