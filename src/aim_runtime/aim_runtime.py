# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Runtime

This module contains the AIMRuntime class which orchestrates the complete
AIM runtime workflow including profile selection and command generation.
"""

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import shlex
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aim_runtime.utils import read_yaml

from .command_generator import CommandGenerator
from .config import AIMConfig
from .cpu_detector import EpycDetector
from .engine_config import load_engine_config
from .engines import build_engine
from .gpu_detector import get_gfx_arch
from .model_cache_resolver import ModelCacheResolver
from .model_storage import StorageBackendRegistry
from .object_model import AcceleratorType
from .profile_selector import ProfileSelector
from .supervisor import EngineSupervisor

logger = logging.getLogger(__name__)


def _extract_models_from_profile(profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract required models from a profile template.

    Args:
        profile_data: Parsed YAML profile data

    Returns:
        List of dictionaries with model information. Each dict has 'name' and 'source' keys.
        Uses model_id field for the actual model to load.
        Model sources are prefixed with 'hf://'.
    """
    models: List[Dict[str, Any]] = []

    # Use model_id field (general profiles won't have this)
    if "model_id" not in profile_data or not profile_data["model_id"]:
        # No model specified (e.g., general profile), return empty list
        return models

    model_name = profile_data["model_id"]

    # Prefix with hf:// for the source
    model_source = f"hf://{model_name}"

    # Create model info dict
    model_info = {"name": model_name, "source": model_source}

    models.append(model_info)

    return models


def _add_storage_estimates(models: List[Dict[str, Any]], storage_registry: StorageBackendRegistry) -> None:
    """
    Add storage size estimates to model information dictionaries (in-place).

    Args:
        models: List of model dictionaries with 'source' keys
        storage_registry: Storage backend registry for estimating storage needs
    """
    for model_info in models:
        if "source" in model_info:
            size_gb = storage_registry.estimate_storage(model_info["source"])
            model_info["size_gb"] = size_gb


def _private_temp_dir(name: str) -> Path:
    """Return a reusable temp directory only the current user can write to.

    The predictable path in a shared /tmp could be pre-created by another user to plant
    native modules, so falls back to a fresh random directory if it isn't private. A
    directory that was ever open to others is not trusted, as it may already hold such files.
    """
    path = Path(tempfile.gettempdir()) / name
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        st = path.lstat()
        if not stat.S_ISLNK(st.st_mode) and st.st_uid == os.getuid() and not st.st_mode & 0o077:
            return path
    except OSError:
        pass
    logger.warning(f"{path} is not a private directory, using a fresh temporary directory")
    return Path(tempfile.mkdtemp(prefix=f"{name}-"))


def _aiter_version() -> str:
    """Return the installed AITER version from package metadata, or "" if unavailable."""
    try:
        return importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        pass
    # Scanning all distributions is slow, so only done for builds under another name.
    for dist in importlib.metadata.packages_distributions().get("aiter", []):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return ""


def _install_aiter_prebuilt_kernels(gpu_model: str) -> None:
    """Make pre-built AITER kernels visible to AITER.

    Copies them into AITER's package JIT directory, or, if that is read-only or
    AITER_JIT_DIR is set, symlinks the package modules and any pre-built
    kernels into AITER_JIT_DIR. See docker/aiter-jit-prebuilt/README.md.

    Args:
        gpu_model: GPU model from profile.metadata.gpu (e.g., "MI300X")
    """
    logger.info("Installing pre-built AITER kernels for %s", gpu_model)
    gfx_arch = get_gfx_arch(gpu_model)
    prebuilt_dir = Path(f"/workspace/aiter-jit-prebuilt/{gfx_arch}")
    prebuilt_dirs: List[Path] = []
    if not gfx_arch:
        logger.warning("Could not resolve a GFX architecture for GPU model %s", gpu_model)
    elif not prebuilt_dir.is_dir():
        logger.warning("Pre-built AITER kernel directory not found at %s", prebuilt_dir)
    else:
        prebuilt_dirs = [prebuilt_dir]

    # Importing aiter on a read-only install copies its JIT dir into $HOME.
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        logger.warning("AITER package not available; skipping pre-built kernel installation")
        return

    jit_dir = Path(list(spec.submodule_search_locations)[0]) / "jit"
    if not jit_dir.is_dir():
        logger.warning(f"AITER JIT directory not found at {jit_dir}")
        return

    if "AITER_JIT_DIR" not in os.environ and os.access(jit_dir, os.W_OK):
        _install_modules(prebuilt_dirs, jit_dir, shutil.copy2)
        return

    try:
        if os.environ.get("AITER_JIT_DIR"):
            target_dir = Path(os.environ["AITER_JIT_DIR"])
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Runtime JIT builds are reused across restarts. The upstream image (AITER build, toolchain,
            # install paths) and AITER version decide whether they are compatible.
            build_key = f"{os.environ.get('AIM_UPSTREAM_IMAGE_REF', '')}\n{_aiter_version()}"
            build_id = hashlib.sha256(build_key.encode()).hexdigest()[:12]
            target_dir = _private_temp_dir(f"aiter-jit-{build_id}-{gfx_arch or 'none'}-{os.getuid()}")
    except OSError as e:
        logger.warning(f"No writable AITER JIT directory, skipping kernel installation: {e}")
        return
    os.environ["AITER_JIT_DIR"] = str(target_dir)
    # AITER's codegen cache defaults to ~/.aiter, which is shared across images on Apptainer/Slurm.
    # A subdirectory, because AITER_REBUILD=1 wipes $AITER_ROOT_DIR/build.
    os.environ.setdefault("AITER_ROOT_DIR", str(target_dir / "root"))
    logger.info(f"Using AITER_JIT_DIR={target_dir}")
    _install_modules([jit_dir, *prebuilt_dirs], target_dir, os.symlink)


def _install_modules(source_dirs: List[Path], target_dir: Path, install: Callable[[Path, Path], object]) -> None:
    """Install .so files from source_dirs into target_dir, earlier sources and existing files winning.

    Broken symlinks (e.g. left by an image with AITER at another path) are removed first, as AITER
    would otherwise write freshly compiled modules through them.
    """
    for link in target_dir.glob("*.so"):
        if link.is_symlink() and not link.exists():
            with contextlib.suppress(OSError):
                link.unlink()

    installed = existing = 0
    for source_dir in source_dirs:
        for so_file in source_dir.glob("*.so"):
            dest = target_dir / so_file.name
            if dest.exists():
                existing += 1
                continue
            tmp = target_dir / f".{so_file.name}.{os.getpid()}.tmp"
            try:
                install(so_file, tmp)
                os.replace(tmp, dest)
                installed += 1
            except OSError as e:
                tmp.unlink(missing_ok=True)
                logger.warning(f"Failed to install AITER module {so_file.name}: {e}")

    logger.info(f"AITER modules in {target_dir}: {installed} installed, {existing} already present")


class AIMRuntime:
    """Main orchestrator for AIM runtime operations."""

    def __init__(self, config: AIMConfig):
        """Initialize the AIM runtime with configuration."""
        self.config = config
        self.profile_selector = ProfileSelector(config)
        engine_config = load_engine_config(config.engine, config.config_path)
        self.engine = build_engine(config, engine_config)
        self.command_generator = CommandGenerator(config, self.engine)
        self.storage_registry = StorageBackendRegistry()

    @staticmethod
    def normalize_model_source(model_id: str) -> str:
        """
        Normalize a model ID to include appropriate protocol prefix.

        If the model ID already has a protocol (e.g. hf://), returns it as-is.
        Otherwise, adds the hf:// prefix for HuggingFace models.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B")

        Returns:
            Normalized model source with protocol prefix

        Examples:
            normalize_model_source("meta-llama/Llama-3.1-8B") -> "hf://meta-llama/Llama-3.1-8B"
            normalize_model_source("hf://org/model") -> "hf://org/model"
        """
        if model_id.startswith("hf://"):
            return model_id
        else:
            return f"hf://{model_id}"

    @staticmethod
    def drop_protocol(model_id: str) -> str:
        if model_id.startswith("hf://"):
            return model_id[5:]
        return model_id

    def download_to_cache(
        self,
        model_id: Optional[str] = None,
        use_hf_cache: bool = False,
    ) -> str:
        """
        Download model to cache directory.

        Downloads the model to the cache directory specified by AIM_CACHE_PATH environment variable.
        By default, downloads directly to the local directory (local-dir mode). Use use_hf_cache
        to download using HuggingFace's default cache structure instead.

        If model_id is not provided, uses the current configuration to determine the model.

        Args:
            model_id: Explicit model id to download (e.g. hf://org/model).
                      Overrides profile selection.
            use_hf_cache: Use HuggingFace's default cache directory structure instead of
                          downloading directly to local directory.

        Returns:
            Path: The path where the model was downloaded

        Raises:
            ValueError: Configuration error
            FileNotFoundError: If profile or model not found
            RuntimeError: If download fails
        """
        logger.info("AIM Runtime download-to-cache mode...")
        logger.debug(f"Log levels - Root: {self.config.log_level_root}, AIM: {self.config.log_level}")
        logger.info(f"Using cache directory: {self.config.cache_path}")
        download_filter = None

        if model_id:
            # Use explicit model id (with protocol)
            model_source = self.normalize_model_source(model_id)
            logger.info(f"Using explicit model id: {model_source}")
        else:
            # Select the profile
            logger.info("Selecting profile...")
            profile = self.profile_selector.find_profile()
            logger.info(f"Selected profile: {profile.profile_handling.path}")
            download_filter = profile.download_filter

            # Read and parse profile to extract model
            profile_data = read_yaml(Path(profile.profile_handling.path))

            # Get model_id from profile
            if "model_id" in profile_data and profile_data["model_id"]:
                model_id_from_profile = profile_data["model_id"]
                logger.info(f"Using model: {model_id_from_profile}")
                model_source = self.normalize_model_source(model_id_from_profile)
            else:
                raise ValueError(
                    "The 'model_id' field is missing from the selected profile. "
                    "This field is required to identify which model to download. Please ensure your profile "
                    "YAML includes a valid 'model_id' entry."
                )

        # Estimate storage before downloading
        logger.info("Estimating storage requirements...")
        size_gb = self.storage_registry.estimate_storage(model_source)
        if size_gb:
            logger.info(f"Estimated model size: {size_gb} GB")
        else:
            logger.warning("Could not estimate model size")

        # Download the model
        if use_hf_cache:
            logger.info("Downloading model using HuggingFace cache structure")
            logger.info(f"Cache directory: {self.config.cache_path}")
        else:
            logger.info(f"Downloading model to local directory: {self.config.cache_path}")

        downloaded_path = self.storage_registry.download(
            model_source,
            self.config.cache_path,
            use_hf_cache=use_hf_cache,
            download_filter=download_filter,
        )

        logger.info(f"✓ Model successfully downloaded to: {downloaded_path}")
        return downloaded_path

    def serve(self) -> None:
        """
        Select profile and execute the inference server (replaces current process).

        This method performs the complete serve workflow:
        1. Select the appropriate profile
        2. Determine actual model and auto-download if needed
        3. Generate execution parameters
        4. Set environment variables
        5. Execute the inference server command (via os.execv)

        Raises:
            FileNotFoundError: If the executable cannot be found
        """
        self.config.log_debug_info()

        # Step 1: Select the profile
        logger.info("Selecting profile...")
        profile = self.profile_selector.find_profile()
        logger.info(f"Selected profile: {profile.profile_handling.path}")

        # Step 2: Determine the actual model
        # Resolve model with fallback chain (same logic as CommandGenerator)
        model_id = profile.model_id or self.config.model_id or self.config.aim_id

        if not model_id:
            raise ValueError("Model not specified in profile or configuration")

        if profile.download_filter is not None and not ModelCacheResolver(
            self.config.cache_path
        ).check_local_dir_exists(model_id, profile.download_filter):
            logger.info("Downloading model with profile download filter...")
            downloaded_path = self.storage_registry.download(
                self.normalize_model_source(model_id),
                self.config.cache_path,
                download_filter=profile.download_filter,
            )
            logger.info(f"Model available at: {downloaded_path}")

        # Install pre-built AITER kernels for the target GPU architecture
        # (only for engines that use them, e.g. vLLM).
        if self.engine.requires_aiter_kernels and profile.metadata:
            _install_aiter_prebuilt_kernels(profile.metadata.accelerator_model)

        # Step 3: Generate execution parameters
        logger.info("Generating execution parameters...")
        command_list, env_vars = self.command_generator.generate_execution_params(profile)

        # Step 3b: Override CPU env vars based on detected cores
        if self.config.accelerator_type == AcceleratorType.CPU and hasattr(self.profile_selector, "detected_cpu_cores"):
            EpycDetector.override_cpu_env_vars(
                env_vars,
                self.profile_selector.detected_cpu_cores,
                cpuset_bind=self.profile_selector.cpuset_bind,
            )

        # Step 4: Set environment variables
        logger.info("--- Setting Environment Variables ---")
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            logger.info(f"Set env var: {key}={value}")
        logger.info("-------------------------------------")

        # Step 5: Log command
        logger.info("--- Execution Command ---")
        logger.info(f"Command: {shlex.join(command_list)}")
        logger.info("-------------------------")

        # Step 6: Resolve the executable path and execute
        executable_path = shutil.which(command_list[0])
        if not executable_path:
            raise FileNotFoundError(f"Could not find executable: {command_list[0]}")

        # Step 7: Run the engine.
        # The supervisor is the default so engine failures become container exit
        # codes. The environment switch keeps the previous execv behavior available.
        watcher = None
        needs_adapter_watcher = self.engine.needs_adapter_watcher(profile)
        if needs_adapter_watcher:
            from .engines.adapter_watcher import AdapterWatcher

            watcher = AdapterWatcher(
                source=self.config.adapter_source,
                max_rank=self.config.adapter_max_rank,
                base_url=f"http://localhost:{self.config.port}",
                refresh_interval=self.config.adapter_refresh_interval,
            )

        if needs_adapter_watcher and not self.config.serve_supervisor:
            logger.warning("AIM_SERVE_SUPERVISOR=false is ignored because dynamic adapters require supervision")

        if needs_adapter_watcher or self.config.serve_supervisor:
            self._serve_supervised(command_list, watcher=watcher)
        else:
            os.execv(executable_path, command_list)

    def _serve_supervised(self, command_list: List[str], *, watcher=None) -> None:
        """Run the engine as a child and drive the dynamic-mode adapter watcher.

        The runtime stays PID 1: it forwards SIGTERM/SIGINT to the engine,
        polls the adapter source via the watcher, and exits with the engine's
        return code so Kubernetes restart semantics are preserved.
        """
        supervisor = EngineSupervisor(
            command_list,
            watcher=watcher,
            on_ready=self._wait_for_engine_ready,
        )
        returncode = supervisor.run()
        logger.info(f"Engine exited with code {returncode}; runtime shutting down.")
        # subprocess reports signal death as a negative code (e.g. -15 for
        # SIGTERM). Translate to the conventional 128 + signal so exit status
        # matches the old os.execv path and k8s restart/alerting stays correct
        # (a raw sys.exit(-15) would exit 241, not 143).
        sys.exit(128 - returncode if returncode < 0 else returncode)

    def _wait_for_engine_ready(self, proc, timeout_seconds: int = 600) -> None:
        """Block until the dynamic adapter engine is ready or the probe expires."""
        import time

        import requests

        url = f"http://localhost:{self.config.port}/v1/models"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    logger.info("Engine ready; adapter watcher starting.")
                    return
            except requests.RequestException:
                pass
            time.sleep(2.0)
        logger.warning("Engine not ready after %ss; starting watcher anyway.", timeout_seconds)

    def dry_run(self) -> List[Dict[str, Any]]:
        """
        Get the selected profile as a serializable dictionary.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with 'filename', 'profile', 'models', and 'script' keys.
                             Each model includes storage estimation information.
                             Returns empty list if no compatible profile is found.
        """
        self.config.log_debug_info()

        try:
            # Step 1: Select the profile (same as dry_run)
            logger.info("Selecting profile...")
            profile = self.profile_selector.find_profile()

            # Step 2: Build list with dict containing filename and profile data
            filename = profile.profile_handling.filename
            path = profile.profile_handling.path

            # Read and parse profile contents as YAML
            profile_data = read_yaml(Path(profile.profile_handling.path))

            # Override CPU env vars in the profile data for dry-run output so the
            # rendered profile reflects the container's cpus (cpuset), matching serve.
            script_env_override: Optional[Dict[str, Any]] = None
            if self.config.accelerator_type == AcceleratorType.CPU and hasattr(
                self.profile_selector, "detected_cpu_cores"
            ):
                if "env_vars" in profile_data and profile_data["env_vars"]:
                    EpycDetector.override_cpu_env_vars(
                        profile_data["env_vars"],
                        self.profile_selector.detected_cpu_cores,
                        cpuset_bind=self.profile_selector.cpuset_bind,
                    )
                    # Reuse the same cpuset-adjusted env vars when rendering the
                    # script so the script block stays consistent with the profile.
                    script_env_override = profile_data["env_vars"]

            # Step 3: Extract required models from profile
            models = _extract_models_from_profile(profile_data)

            # Step 4: Fallback to config if no models in profile (e.g., general profiles)
            if not models:
                fallback_model_id = self.config.model_id or self.config.aim_id
                if fallback_model_id:
                    models = [{"name": self.drop_protocol(fallback_model_id), "source": fallback_model_id}]

            # Step 5: Normalize model sources using class method
            for model_info in models:
                if "source" in model_info:
                    model_info["source"] = self.normalize_model_source(model_info["name"])

            script_path = self.command_generator.generate_command_script(profile, env_vars=script_env_override)
            with open(script_path, "r", encoding="utf-8") as f:
                output_script = f.read()
            # Step 6: Add storage estimates to models
            _add_storage_estimates(models, self.storage_registry)

            logger.info(f"Selected profile: {profile.profile_handling.path}")
            return [
                {
                    "filename": filename,
                    "path": path,
                    "profile": profile_data,
                    "models": models,
                    "script": output_script,
                }
            ]

        except Exception as e:
            # If no profile found or any error occurs, return empty list
            # This includes ProfileNotFound, FileNotFoundError, yaml.YAMLError, etc.
            logger.warning(f"No compatible profile found or error occurred: {e}")
            return []
