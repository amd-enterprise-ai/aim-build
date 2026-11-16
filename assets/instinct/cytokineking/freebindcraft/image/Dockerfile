# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# FreeBindCraft AIM: long-running job container (protein binder design,
# no-PyRosetta mode).
#   Layer 0: rocm/dev-ubuntu-22.04:7.2.2 (vendor base, see config.yaml)
#   Layer 1: this image: Miniforge3, BindCraft conda env, JAX ROCm wheels,
#            ColabDesign (pinned + patched), FreeBindCraft (pinned) and the job
#            entrypoint. The AlphaFold2 parameters are mounted at run time.
# No BentoML, no HTTP server, no port. Running the image with no arguments
# starts the design run: it reads $AIM_INPUT_PATH/target.pdb (and an optional
# settings.json), writes results and logs to $AIM_OUTPUT_PATH, and exits.
#
# FreeBindCraft requires:
#   - JAX with ROCm support (via repo.radeon.com AMD wheels)
#   - OpenMM for structure relaxation (--no-pyrosetta mode)
#   - ColabDesign (JAX-based AF2 hallucination)
#   - sc-rs and FASPR pre-compiled Linux binaries (bundled in repo)
#   - AlphaFold2 model weights: NOT baked; mounted at run time, AIM_CACHE_PATH=<dir containing params/>
#
# JAX ROCm wheels from repo.radeon.com:
#   ROCm 7.2.2: jax 0.8.2 wheels (jax_rocm7_plugin, jax_rocm7_pjrt)
#   https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/
#
# NOTE: FreeBindCraft's install script pins jax==0.6.0, but ROCm 7.2.2 only
# provides jax==0.8.2 ROCm wheels. We use jax 0.8.2 with ROCm 7.2.2. ColabDesign
# does NOT track JAX API removals: its HEAD (pinned below) still calls
# jax.lib.xla_bridge.get_backend() in colabdesign/shared/utils.py:clear_mem(),
# which JAX 0.8 removed, so every design crashed with AttributeError.
# The image therefore pins ColabDesign and patches that
# one call at build time (see "ColabDesign JAX 0.8 compatibility patch"
# below). If strict jax 0.6.0 compatibility is required, switch the base
# image to ROCm 6.3 and use the rocm6.3 wheel channel.
#
# FIX (base/wheel ROCm mismatch): the base image previously stayed on
# rocm/dev-ubuntu-22.04:6.3.3 — i.e. ROCm 6.3.3 userspace libraries
# (rocBLAS, MIOpen, hipBLAS, the ROCm runtime) — while the JAX plugin/pjrt
# wheels above are built against ROCm 7.2.2. That ABI/userspace mismatch is
# the confirmed cause of "Unable to initialize backend 'rocm' ...
# rocm_plugin_extension not found" seen at run time. The base image is now rocm/dev-ubuntu-22.04:7.2.2, matching
# the JAX wheel channel's ROCm version (7.2.2) exactly.
#
# Python 3.12 used: cp312 JAX ROCm 7.2.2 wheels available at repo.radeon.com.

ARG BASE_IMAGE=rocm/dev-ubuntu-22.04:7.2.2
FROM ${BASE_IMAGE}

# ── System packages ────────────────────────────────────────────────────────────
# The JAX ROCm PJRT plugin (jax_plugins.xla_rocm7) dlopens several runtime
# libraries at import time that rocm/dev-ubuntu-22.04 doesn't pull in by
# default — confirmed one at a time (librccl.so.1; then
# libMIOpen.so.1 after adding rccl; then libdw.so.1,
# a plain Ubuntu elfutils lib, not ROCm-specific, after adding the ROCm set).
# Verified via `ldd xla_rocm_plugin.so | grep "not found"` directly on the
# built image that this is the complete set — zero unresolved deps remain,
# and `python -c "import jax"` gets past plugin loading entirely (the only
# remaining error is "No AMD GPUs were found", which is the GPU-detection
# step failing because that check was run with no GPU devices attached, not
# a library problem).
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        git \
        build-essential \
        libssl-dev \
        ca-certificates \
        ocl-icd-libopencl1 \
        libdw1 \
        rocm-hip-libraries \
        miopen-hip \
    && rm -rf /var/lib/apt/lists/*

# ── Miniforge3 (conda) ────────────────────────────────────────────────────────
ARG MINIFORGE_VERSION=24.11.3-0
RUN wget -qO /tmp/Miniforge3.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3.sh -b -p /miniforge3 \
    && rm /tmp/Miniforge3.sh

ENV PATH="/miniforge3/bin:${PATH}"
ENV CONDA_DIR="/miniforge3"

# ── Conda environment: Python 3.12 + core scientific stack ────────────────────
# Python 3.12 used: cp312 JAX ROCm 7.2.2 wheels are available at repo.radeon.com
# and aim-build's instinct-requirements.txt is compiled for Python 3.12, so the
# unit-test validation step (which installs that lockfile inside the container)
# requires Python >=3.12. flax<0.10.0 remains a hard constraint.
# PINNED VERSIONS (provenance): every version below is the one found in the image built from the
# head of PR #1435 (83d5407d5) on a Crusoe MI355X node (conda list plus the modules actually
# imported, 2026-11-06); the packages the explicit list pulls in transitively (about 250) are
# solved by conda at build time and are not pinned here.
# numpy: conda installs numpy 1.26.4 here (the old numpy<2.0.0 constraint), but the pip step
# below (--force-reinstall --ignore-installed) replaces it with numpy 2.5.4, which jax 0.8.2 and
# ml_dtypes 0.6.0 need (numpy>=2.0). The numpy that runs is therefore the pip one, pinned there;
# the conda pin only keeps the conda solve identical. `pip freeze` and `pip check` inside the image
# still show the stale conda metadata (numpy 1.26.4, jax and jaxlib 0.7.1, scipy 1.17.1); `conda list`
# and the imported modules show the versions that run.
# matplotlib<3.9: ColabDesign's animation code (plot_pseudo_3D) calls
# matplotlib.cm.get_cmap, removed in matplotlib 3.9, which crashed every design
# right after the first hallucination (AttributeError).
# ffmpeg is NOT installed: it would be a GPL-licensed build (licence review). The only caller is the optional
# design-animation video, which the entrypoint switches off (save_design_animations=false); designs and scores
# are unaffected.
# py3dmol and fsspec are in upstream's own conda package list.
RUN conda create -n BindCraft python=3.12.16 -y && \
    conda run -n BindCraft conda install -y -c conda-forge \
        "numpy==1.26.4" \
        "pandas==2.3.3" \
        "matplotlib==3.8.4" \
        "biopython==1.88" \
        "scipy==1.17.1" \
        "pdbfixer==1.12" \
        "py3dmol==2.5.4" \
        "dssp==4.6.1" \
        "fsspec==2026.9.0" \
        "openmm==8.6.1" \
        "seaborn==0.13.2" \
        "tqdm==4.70.1" \
        "chex==0.1.92" \
        "dm-haiku==0.0.17" \
        "flax==0.9.0" \
        "dm-tree==0.1.9" \
        "joblib==1.6.0" \
        "ml-collections==1.0.0" \
        "immutabledict==4.3.1" \
        "optax==0.2.8" \
        "libgfortran5==16.2.0" \
    && conda clean -afy

ENV PATH="/miniforge3/envs/BindCraft/bin:${PATH}"
ENV CONDA_DEFAULT_ENV="BindCraft"

# ── JAX ROCm wheels (repo.radeon.com ROCm 7.2.2) ─────────────────────────────
# Install order matters: pjrt → plugin → jaxlib → jax.
# ROCm 7.2.2 wheel index at repo.radeon.com provides jax 0.8.2.
# numpy, scipy, ml_dtypes and opt_einsum are jax 0.8.2's dependencies, pinned to the versions of the
# validated image (see PINNED VERSIONS above) so that pip does not float them.
ARG ROCM_WHL_BASE=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2
RUN /miniforge3/envs/BindCraft/bin/pip install --no-cache-dir \
        --force-reinstall --ignore-installed \
        "${ROCM_WHL_BASE}/jax_rocm7_pjrt-0.8.2+rocm7.2.2-py3-none-manylinux_2_28_x86_64.whl" \
        "${ROCM_WHL_BASE}/jax_rocm7_plugin-0.8.2+rocm7.2.2-cp312-cp312-manylinux_2_28_x86_64.whl" \
        "${ROCM_WHL_BASE}/jaxlib-0.8.2+rocm7.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
        "jax==0.8.2" \
        "numpy==2.5.4" \
        "scipy==1.18.1" \
        "ml_dtypes==0.6.0" \
        "opt_einsum==3.4.0"

# ── JAX GPU canary — verify the JAX ROCm plugin is importable ────────────────
# Imports the ROCm PJRT plugin module (jax_plugins.xla_rocm7) and exits non-zero if the import fails, so
# a broken plugin (for example "rocm_plugin_extension not found") fails the build instead of the job.
# Actual GPU enumeration occurs at runtime (not available in CI buildkitd
# without --device=/dev/kfd, so we skip device enumeration here).
RUN /miniforge3/envs/BindCraft/bin/python - <<'PYEOF'
import sys
try:
    import jax
    print('JAX version:', jax.__version__)
    from jax._src.lib import xla_client
    print('XLA client version:', xla_client._version)
    import jax_plugins.xla_rocm7
    print('JAX ROCm plugin module:', jax_plugins.xla_rocm7.__name__)
except Exception as exc:
    sys.exit('JAX ROCm canary failed: %s: %s' % (type(exc).__name__, exc))
print('JAX ROCm canary: plugin importable')
PYEOF

# ── ColabDesign (JAX-based AlphaFold2 hallucination) ─────────────────────────
# Install the pinned commit from git; --no-deps avoids overriding the pinned
# jax/jaxlib. The commit is the one validated in this image; the JAX 0.8
# compatibility patch is applied to it later (see "ColabDesign JAX 0.8
# compatibility patch") so that this expensive layer stays cached.
ARG COLABDESIGN_COMMIT=e31a56fe1d9b4de25c8697f3a28b75892941cc72
RUN /miniforge3/envs/BindCraft/bin/pip install --no-cache-dir --no-deps \
        "git+https://github.com/sokrypton/ColabDesign.git@${COLABDESIGN_COMMIT}"

# ── FreeSASA (SASA computation in --no-pyrosetta mode) ───────────────────────
RUN /miniforge3/envs/BindCraft/bin/pip install --no-cache-dir freesasa==2.2.1

# ── FreeBindCraft (pinned commit) ─────────────────────────────────────────────
# Pinned so the image is reproducible; the commit is the one the job was
# validated against. Fetching a single commit works on GitHub (shallow, no history).
ARG FREEBINDCRAFT_COMMIT=8b8d4c4627da06f084f46b7007450048fe3ca22f
WORKDIR /app
RUN git init -q /app && \
    git -C /app fetch --depth=1 https://github.com/cytokineking/FreeBindCraft.git "${FREEBINDCRAFT_COMMIT}" && \
    git -C /app checkout -q FETCH_HEAD && \
    rm -f /app/functions/DAlphaBall.gcc /app/functions/dssp && \
    chmod +x /app/functions/sc /app/functions/FASPR

# ── Licence-driven removals (FreeBindCraft checkout above) ───────────────────
# DAlphaBall.gcc (unprovenanced Rosetta-lineage binary, PyRosetta-only) is deleted: the --no-pyrosetta
# run never executes it. The bundled UPX-packed functions/dssp (unknown version and licence) is deleted
# and replaced by conda-forge `dssp` (mkdssp, BSD-2-Clause, installed above); entrypoint.py sets the
# advanced setting dssp_path to it. FASPR, dun2010bbdep.bin (identical to the MIT FASPR file) and sc-rs stay.
# conda-forge mkdssp 4.x cannot read PDB files with free-text REMARK records (the bundled dssp 2.0.4 ignored
# them), so `mkdssp` is a thin wrapper (src/mkdssp_shim.sh) in front of the real binary (mkdssp.real) that drops
# such records before calling it. The smoke test fails the build if Biopython cannot parse the output for a
# plain PDL1.pdb and for the same file with REMARK records, or if the two results differ.
COPY src/mkdssp_shim.sh /tmp/mkdssp_shim.sh
RUN test ! -e /app/functions/DAlphaBall.gcc && test ! -e /app/functions/dssp \
    && mv /miniforge3/envs/BindCraft/bin/mkdssp /miniforge3/envs/BindCraft/bin/mkdssp.real \
    && install -m 0755 /tmp/mkdssp_shim.sh /miniforge3/envs/BindCraft/bin/mkdssp && rm -f /tmp/mkdssp_shim.sh \
    && sed '1i REMARK   Copyright (c) test, free-text remark' /app/example/PDL1.pdb > /tmp/PDL1_remark.pdb \
    && /miniforge3/envs/BindCraft/bin/python - <<'PYEOF'
import os, shutil, sys
from Bio.PDB import PDBParser, DSSP
mk = shutil.which("mkdssp")
if not mk:
    sys.exit("mkdssp not on PATH")
res = {}
for pdb in ("/app/example/PDL1.pdb", "/tmp/PDL1_remark.pdb"):
    model = PDBParser(QUIET=True).get_structure("t", pdb)[0]
    d = DSSP(model, pdb, dssp=mk)
    res[pdb] = "".join(d[k][2] for k in d.keys())
    print("mkdssp", mk, pdb, "residues", len(res[pdb]), "ss", res[pdb][:40])
os.remove("/tmp/PDL1_remark.pdb")
plain, remark = res.values()
if len(plain) < 50 or "E" not in plain:
    sys.exit("DSSP output from mkdssp was not parsed into a plausible secondary structure")
if plain != remark:
    sys.exit("mkdssp wrapper: PDB with REMARK records gives a different result")
PYEOF

# ── AlphaFold2 parameters are NOT baked in (CC BY 4.0, about 5.6 GB) ─────────
# Mount a directory holding params/params_model_*_multimer_v3.npz and set AIM_CACHE_PATH to it
# (fbc_preflight.resolve_af_params_dir). Nothing in the build reads the weights.

# ── ColabDesign JAX 0.8 compatibility patch (kept late to preserve layer cache) ─
# The patch targets the commit pinned above (COLABDESIGN_COMMIT).
# colabdesign/shared/utils.py:clear_mem() calls jax.lib.xla_bridge.get_backend(),
# removed in JAX 0.8; the supported replacement is jax.extend.backend.get_backend()
# (jax.extend must be imported explicitly). The script fails the build if the
# pattern is absent beforehand or still present afterwards.
RUN /miniforge3/envs/BindCraft/bin/python - <<'PYEOF'
import colabdesign.shared.utils as u, pathlib, re, sys
p = pathlib.Path(u.__file__)
s = p.read_text()
old = "jax.lib.xla_bridge.get_backend()"
if old not in s:
    sys.exit("patch pattern not found in %s" % p)
s = s.replace(old, "jax.extend.backend.get_backend()")
s, n = re.subn(r"^import jax$", "import jax\nimport jax.extend", s, count=1, flags=re.M)
if n != 1:
    sys.exit("'import jax' line not found in %s" % p)
p.write_text(s)
PYEOF

RUN test "$(grep -c 'jax.lib.xla_bridge' /miniforge3/envs/BindCraft/lib/python3.12/site-packages/colabdesign/shared/utils.py)" = 0 \
    && test "$(grep -c 'jax.extend.backend.get_backend()' /miniforge3/envs/BindCraft/lib/python3.12/site-packages/colabdesign/shared/utils.py)" = 1

# Canary: import the patched module and call clear_mem() on the CPU backend
# (no GPU in the build); if the call cannot run there, the import alone must work.
RUN JAX_PLATFORMS=cpu /miniforge3/envs/BindCraft/bin/python - <<'PYEOF'
import colabdesign.shared.utils as u
print('import ok')
try:
    u.clear_mem()
    print('clear_mem ok')
except AttributeError:
    raise  # a removed-API regression must fail the build
except Exception as e:
    print('clear_mem skipped (no usable backend in build):', type(e).__name__)
PYEOF

# ── Job entrypoint─────────────────────────────────────────────────
# Standalone Python wrapper around /app/bindcraft.py. It is stdlib-only and is
# not importable from the BindCraft env's site-packages on purpose.
COPY src/ /opt/aim/src/

# ── Environment variables ─────────────────────────────────────────────────────
# Unbuffered so the wrapper can stream the run's output line by line.
ENV PYTHONUNBUFFERED=1
ENV BINDCRAFT_HOME=/app
# JAX platform selection: use ROCm backend on AMD Instinct GPUs.
ENV JAX_PLATFORMS=rocm

# matplotlib keeps get_cmap (matplotlib < 3.9) for ColabDesign's plotting code; ffmpeg must stay absent
# (licence review: animations are off, so nothing needs it).
RUN /miniforge3/envs/BindCraft/bin/python -c "import matplotlib.cm as c, sys; sys.exit(0 if hasattr(c, 'get_cmap') else 1)" \
    && ! command -v ffmpeg > /dev/null \
    && ! conda list -n BindCraft --json | grep -q '"name": "ffmpeg"'

# Import gate: the whole upstream import chain (JAX, ColabDesign, OpenMM, ...)
# must load on the CPU backend; no GPU is needed at build time.
RUN JAX_PLATFORMS=cpu /miniforge3/envs/BindCraft/bin/python /app/bindcraft.py --help > /dev/null

WORKDIR /app

# No HTTP server, no port. A bare `docker run <image>` starts the work.
ENTRYPOINT ["/miniforge3/envs/BindCraft/bin/python", "/opt/aim/src/entrypoint.py"]
