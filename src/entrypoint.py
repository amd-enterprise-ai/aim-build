#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
AIM Runtime Entrypoint

Lean CLI interface that delegates to business logic in aim_runtime package.
This module handles Click command definitions and command logic.
"""

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, NoReturn

import click

from aim_runtime.logging_config import configure_logging
from aim_runtime.utils import dump_yaml

if TYPE_CHECKING:
    from aim_runtime.harness import HarnessConfig, ModelHarness

root_log_level = os.environ.get("AIM_LOG_LEVEL_ROOT", "WARNING")
configure_logging(
    root_log_level=root_log_level,
    aim_log_level=os.environ.get("AIM_LOG_LEVEL", "INFO"),
)
os.environ["VLLM_LOGGING_LEVEL"] = root_log_level

from aim_runtime.accelerator_detector import AcceleratorDetector  # noqa: E402
from aim_runtime.aim_runtime import AIMRuntime  # noqa: E402
from aim_runtime.config import AIMConfig  # noqa: E402
from aim_runtime.object_model import AcceleratorFamily, AcceleratorModel, AcceleratorType  # noqa: E402
from aim_runtime.profile_selector import ProfileCompatibilityState, ProfileSelector  # noqa: E402

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Create logger at module level
logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """AIM Runtime - Profile selection and command generation."""
    # If no subcommand is provided, default to serve
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@cli.command()
def serve():
    """Select profile and execute the inference server (default)."""
    try:
        # Load configuration from environment variables
        config = AIMConfig.from_environment()

        # Configure logging based on the config
        configure_logging(root_log_level=config.log_level_root, aim_log_level=config.log_level)

        logger.debug("AIM Runtime starting...")
        logger.debug(f"Log levels - Root: {config.log_level_root}, AIM: {config.log_level}")

        # Create runtime and execute serve
        runtime = AIMRuntime(config)
        runtime.serve()

    except ValueError as e:
        # Configure basic logging in case config loading failed
        configure_logging(root_log_level="WARNING", aim_log_level="WARNING")
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


@cli.command(name="dry-run")
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"], case_sensitive=False),
    default="yaml",
    help="Output format for dry-run results",
)
def dry_run(format):
    """Perform profile selection and display the selected profile without execution."""
    try:
        # Load configuration from environment variables
        config = AIMConfig.from_environment()

        # Configure logging based on the config
        configure_logging(root_log_level=config.log_level_root, aim_log_level=config.log_level)

        logger.debug(f"AIM Runtime dry-run mode (format: {format})...")
        logger.debug(f"Log levels - Root: {config.log_level_root}, AIM: {config.log_level}")

        # Create runtime and perform dry-run
        runtime = AIMRuntime(config)

        profiles_dict = runtime.dry_run()

        if format == "json":
            # Return all compatible profiles as JSON
            print(json.dumps(profiles_dict, indent=2))
        else:
            # Display the selected profile as YAML
            print(dump_yaml(profiles_dict))

    except ValueError as e:
        # Configure basic logging in case config loading failed
        configure_logging(root_log_level="WARNING", aim_log_level="WARNING")
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


@cli.command(name="download-to-cache")
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Explicit model id to download (e.g. hf://org/model). Overrides profile selection.",
)
@click.option(
    "--use-hf-cache",
    is_flag=True,
    default=False,
    help="Use HuggingFace's default cache directory structure instead of downloading directly to local directory.",
)
def download_to_cache(model_id, use_hf_cache):
    """Download the model to cache.

    Downloads the model to the cache directory specified by AIM_CACHE_PATH environment variable.
    By default, downloads directly to the local directory (local-dir mode). Use --use-hf-cache
    to download using HuggingFace's default cache structure instead.

    If --model-id is not provided, uses the current configuration to determine the model.

    Examples:
      aim-runtime download-to-cache
      aim-runtime download-to-cache --model-id hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0
      aim-runtime download-to-cache --use-hf-cache
    """
    try:
        # Load configuration from environment variables
        config = AIMConfig.from_environment(model_id)

        # Configure logging based on the config
        configure_logging(root_log_level=config.log_level_root, aim_log_level=config.log_level)

        # Create runtime
        runtime = AIMRuntime(config)

        # Download the model
        # Custom model name from CLI takes precedence over env var
        downloaded_path = runtime.download_to_cache(model_id=model_id, use_hf_cache=use_hf_cache)

        print(f"\nModel downloaded to: {downloaded_path}")

    except ValueError as e:
        # Configure basic logging in case config loading failed
        configure_logging(root_log_level="WARNING", aim_log_level="INFO")
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


@cli.command(name="list-profiles")
@click.option(
    "--state",
    type=click.Choice(
        [state.value for state in ProfileCompatibilityState] + ["all"],
        case_sensitive=False,
    ),
    default="all",
    help="Show only profiles in specific compatibility state (default: all)",
)
@click.option(
    "--format",
    type=click.Choice(["text", "table", "json", "yaml"], case_sensitive=False),
    default="table",
    help="Output format: text, table, json, or yaml (default: table)",
)
@click.option(
    "--skip-compatibility-check",
    is_flag=True,
    help="Skip GPU detection and compatibility checks; list all profiles without categorization",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def list_profiles(state, format, skip_compatibility_check, verbose):
    """List and categorize profiles by compatibility with current configuration.

    Examples:
      aim-runtime list-profiles
      aim-runtime list-profiles --state compatible
      aim-runtime list-profiles --format json
      aim-runtime list-profiles --format yaml
      aim-runtime list-profiles --state gpu_mismatch --format table --verbose
      aim-runtime list-profiles --skip-compatibility-check --format json
    """
    try:
        # Load configuration from environment variables
        config = AIMConfig.from_environment()

        # Configure logging (verbose flag overrides config)
        configure_logging(
            root_log_level="DEBUG" if verbose else config.log_level_root,
            aim_log_level="DEBUG" if verbose else config.log_level,
        )

        # Create profile selector
        selector = ProfileSelector(config)

        if skip_compatibility_check and state != "all":
            logger.warning("Ignoring --state filter: not applicable with --skip-compatibility-check")

        if format in ("json", "yaml"):
            # Machine-readable output
            if skip_compatibility_check:
                serialized = selector.serialize_all_profiles()
            else:
                categorized = selector.get_categorized_profiles()
                if state != "all":
                    state_key = ProfileCompatibilityState(state)
                    categorized = {state_key: categorized[state_key]}
                serialized = selector.serialize_profiles(categorized)

            if format == "json":
                print(json.dumps(serialized, indent=2))
            else:
                print(dump_yaml(serialized))
        else:
            # Human-readable output
            if skip_compatibility_check:
                output = selector.format_all_profiles_report(format_type=format)
            else:
                categorized = selector.get_categorized_profiles()
                if state != "all":
                    state_key = ProfileCompatibilityState(state)
                    categorized = {state_key: categorized[state_key]}

                if format == "table":
                    output = selector.format_table_report(categorized)
                else:
                    output = selector.format_text_report(categorized)

            print(output)

    except ValueError as e:
        configure_logging(root_log_level="WARNING", aim_log_level="WARNING")
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


#: How long a crashed server can pass for one that is still loading. The async
#: benchmark path allows 7200s for startup, so without this a crash wastes the
#: whole two hours.
READINESS_POLL_SLICE_SECONDS = 30


def _raise_if_server_exited(server_process: subprocess.Popen | None) -> None:
    """Fail fast when a server the CLI started has already exited."""
    if server_process is None:
        return
    returncode = server_process.poll()
    if returncode is not None:
        raise RuntimeError(f"Model server exited with code {returncode} before becoming ready")


def _wait_for_harness_readiness(
    harness,
    service_url: str,
    timeout_seconds: int,
    server_process: subprocess.Popen | None = None,
) -> float:
    """Block until ``harness.health_check`` reports the service is ready.

    Delegates to the harness so each engine uses its own readiness endpoint
    (e.g. BentoML ``/healthz`` vs the OpenAI-compatible ``/v1/models`` probe
    used by the default :class:`ModelHarness.health_check`).

    ``health_check`` blocks for however long it is given and returns only
    yes/no, so there is no way to notice a dead server from inside it. When we
    started the server we therefore call it in short slices and check the
    process between them. With no process to watch it gets the whole budget in
    one call.

    Returns elapsed wall-clock seconds so callers can preserve the startup
    duration for reporting (the harness re-probes on an already-live service and
    would otherwise report ~0).
    """
    start = time.monotonic()
    if server_process is None:
        if not harness.health_check(service_url, timeout_seconds=timeout_seconds):
            raise RuntimeError(f"Service not ready at {service_url} after {timeout_seconds}s")
        return round(time.monotonic() - start, 3)

    deadline = start + timeout_seconds
    while True:
        _raise_if_server_exited(server_process)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Service not ready at {service_url} after {timeout_seconds}s")
        slice_seconds = max(1, int(min(READINESS_POLL_SLICE_SECONDS, remaining)))
        if harness.health_check(service_url, timeout_seconds=slice_seconds):
            return round(time.monotonic() - start, 3)


def _start_server_in_background(config: AIMConfig) -> subprocess.Popen:
    runtime = AIMRuntime(config)
    logger.info("Selecting profile for the model server...")
    profile = runtime.profile_selector.find_profile()
    logger.info(f"Selected profile: {profile.profile_handling.path}")

    command_list, env_vars = runtime.command_generator.generate_execution_params(profile)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_vars.items()})

    logger.info(f"Starting model server: {shlex.join(command_list)}")
    return subprocess.Popen(command_list, env=env)


def _stop_server(server_process) -> None:
    """Stop a background model server started by the CLI, if any.

    A server that already exited is logged with its exit code, which tells a
    crash apart from an unreachable service. That is all ``validate`` gets: it
    has already spent its full timeout polling a dead port to get here.
    """
    if not server_process:
        return
    if (returncode := server_process.poll()) is not None:
        logger.error("Model server had already exited with code %s", returncode)
        return

    logger.info("Stopping model server...")
    try:
        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Model server did not exit after SIGINT; sending SIGTERM.")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Model server did not exit cleanly; killing it.")
            server_process.kill()
    except OSError:
        pass  # Process already exited between poll() and send_signal()


@contextmanager
def _managed_service(
    service_url: str | None,
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    """Yield (service_url, server_process) to run against, starting a server if we must.

    Same decision as ``benchmark``: an explicit ``--service-url`` means someone
    else owns a running service, anything else means the CLI starts one for the
    duration of the run and tears it down afterwards.

    This helper only manages lifecycle. Callers that need startup waiting must
    explicitly invoke ``_wait_for_harness_readiness`` with their own timeout.
    """
    if service_url:
        yield service_url, None
        return

    aim_config = AIMConfig.from_environment()
    server_process = _start_server_in_background(aim_config)
    try:
        yield f"http://localhost:{aim_config.port}", server_process
    finally:
        _stop_server(server_process)


def _benchmark_via_harness(
    *,
    service_url: str | None,
    timeout_seconds: int,
    output_dir: str,
    startup_timeout: int,
    config_file: str | None = None,
) -> None:
    """Run the benchmark through the discovered ModelHarness and exit.

    This is the single benchmark flow for every image. It discovers the active
    harness — a custom one for specialized images, otherwise the shipped
    :class:`VLLMHarness` — and delegates the run to it, so the CLI drives
    standard and specialized harnesses identically. The CLI owns process
    lifecycle (starting a local server when no service URL is given) and writes
    the ``HarnessResult`` to ``harness_benchmark_results.json``.

    That name is deliberately not ``benchmark_results.json``: the vLLM suite
    already writes a file by that name in a different schema
    (``overall_success`` / ``benchmark_configs``), which is what
    ``ci/benchmarking/parse_benchmark_results.py`` reads.
    """
    from aim_runtime.harness import HarnessConfig
    from aim_runtime.harness.discovery import discover_harness, has_custom_harness

    custom = has_custom_harness()
    resolved_profile = _resolve_profile_dict(None)
    harness = discover_harness(profile=resolved_profile)

    # ``--config`` means different things per path: a per-run overrides mapping
    # for specialized harnesses, or the benchmark-suite file that the standard
    # vLLM path forwards straight to AIMBenchmark.
    if custom:
        extra = _load_config_yaml(config_file)
    else:
        extra = {"config_file": config_file} if config_file else {}

    # A harness that writes its own artifacts needs the destination, but a
    # ``--config`` entry of the same name still wins.
    extra.setdefault("output_dir", output_dir)

    server_process = None
    try:
        # Service lifecycle: when no URL is given, the CLI starts the server the
        # same way for every image. The command generator is engine-aware
        # (vLLM, vLLM-Omni, BentoML, ...), so the only thing that varies is the
        # readiness endpoint — and that is handled by the harness's own
        # ``health_check`` (e.g. ``/v1/models`` vs ``/healthz``).
        if not service_url:
            aim_config = AIMConfig.from_environment()
            server_process = _start_server_in_background(aim_config)
            service_url = f"http://localhost:{aim_config.port}"
            logger.info("Waiting for service readiness at %s ...", service_url)
            _wait_for_harness_readiness(harness, service_url, startup_timeout, server_process)

        config = HarnessConfig(
            profile=resolved_profile,
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            output_format="json",
            extra=extra,
        )

        result = harness.benchmark(config)

        _write_results_json(result, output_dir, "harness_benchmark_results.json")

        _report_harness_result(result, "json")
        sys.exit(0 if result.success else 1)
    finally:
        _stop_server(server_process)


@cli.command(name="benchmark")
@click.option(
    "--service-url",
    type=str,
    required=False,
    help="AIM service URL including port (e.g. http://localhost:8000).",
)
@click.option(
    "--timeout-seconds",
    type=int,
    default=30,
    show_default=True,
    help="Timeout in seconds for service requests.",
)
@click.option(
    "--config",
    "config_file",
    type=str,
    default=None,
    help="Path to benchmark config YAML (defaults to built-in config).",
)
@click.option(
    "--output-dir",
    type=str,
    default=".",
    show_default=True,
    help="Directory to write benchmark results.",
)
@click.option(
    "--startup-timeout",
    type=int,
    default=120,
    show_default=True,
    help="Seconds to wait for the server to become ready.",
)
def benchmark(service_url, timeout_seconds, config_file, output_dir, startup_timeout):
    """Run the benchmark suite against a running AIM service.

    The benchmark is always delegated to the discovered ModelHarness — a custom
    harness for specialized images, otherwise the shipped ``VLLMHarness`` — so
    the CLI drives standard and specialized harnesses through the same path.
    """
    try:
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
        )

        _benchmark_via_harness(
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
            startup_timeout=startup_timeout,
            config_file=config_file,
        )
    except click.ClickException:
        raise  # Let Click render usage errors (e.g. a bad --config path) itself.
    except Exception:
        logger.exception("Benchmarking failed")
        sys.exit(1)


@cli.command(name="validate")
@click.option("--profile", default=None, help="Profile name. Defaults to auto-detected profile.")
@click.option(
    "--service-url",
    type=str,
    required=False,
    help=(
        "Service URL including port (e.g. http://localhost:8000). "
        "When omitted, the CLI starts the model server itself and stops it afterwards."
    ),
)
@click.option(
    "--scope",
    multiple=True,
    default=("runtime", "offline"),
    type=click.Choice(["runtime", "offline"]),
    help="Check scope(s). Repeat for multiple: --scope runtime --scope offline",
)
@click.option(
    "--config",
    "config_file",
    type=str,
    default=None,
    help="Path to config YAML with per-run overrides.",
)
@click.option(
    "--startup-timeout",
    type=int,
    default=3600,
    show_default=True,
    help="Seconds to wait for the service to become ready before validating.",
)
@click.option(
    "--timeout",
    default=300,
    show_default=True,
    help="Timeout in seconds for individual validation requests/checks.",
)
@click.option(
    "--output-format",
    default="json",
    type=click.Choice(["json", "table", "ci"]),
    help="Output format.",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to write validation results. If set, writes validate_results.json.",
)
def validate(profile, service_url, scope, config_file, startup_timeout, timeout, output_format, output_dir):
    """Validate that the model service is healthy and producing correct output.

    With ``--service-url`` the checks run against that service. Without it, the
    CLI starts the model server for the profile being validated, runs the checks
    once it serves a model, and shuts it down again. ``--startup-timeout`` is
    the readiness budget for the service to come up. ``--timeout`` controls the
    per-request timeout of each check; warmup keeps its own
    ``max_warmup_time`` budget (``--config``).
    """
    try:
        _configure_harness_logging()

        from aim_runtime.harness import STARTUP_READY_TIME_SECONDS_KEY, CheckScope

        # Resolving the profile first also pins AIM_PROFILE_ID, so a server the
        # CLI starts below runs the very profile being validated.
        harness, config = _harness_session(
            profile,
            service_url=service_url,
            timeout_seconds=timeout,
            output_format=output_format,
            check_scopes={CheckScope(s) for s in scope},
            extra=_load_config_yaml(config_file),
        )

        with ExitStack() as stack:
            # Only the offline checks need a live service; runtime-only
            # validation reads the profile and never touches the network.
            if CheckScope.OFFLINE in config.check_scopes:
                config.service_url, server_process = stack.enter_context(_managed_service(service_url))
                ready_time = _wait_for_harness_readiness(harness, config.service_url, startup_timeout, server_process)
                # Inject the pre-measured startup time so the harness doesn't re-probe
                # an already-live service and report ~0 for ready_time_seconds.
                config.extra.setdefault(STARTUP_READY_TIME_SECONDS_KEY, ready_time)
            result = harness.validate(config)
            if output_dir:
                _write_results_json(result, output_dir, "validate_results.json")
            _report_and_exit(result, output_format)
    except click.ClickException:
        raise  # Let Click render usage errors (e.g. a bad --config path) itself.
    except Exception:
        logger.exception("Validation failed")
        sys.exit(1)


@cli.command(name="evaluate")
@click.option("--profile", default=None, help="Profile name. Defaults to auto-detected profile.")
@click.option(
    "--service-url",
    type=str,
    required=False,
    help="Service URL including port (e.g. http://localhost:8000). Defaults to localhost:{profile.port}.",
)
@click.option(
    "--config",
    "config_file",
    type=str,
    default=None,
    help="Path to evaluation config YAML with per-run overrides (data_dir, limit, etc.).",
)
@click.option("--timeout", default=1800, show_default=True, help="Timeout in seconds.")
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to write evaluation results. If set, writes evaluate_results.json.",
)
@click.option(
    "--startup-timeout",
    type=int,
    default=120,
    show_default=True,
    help="Seconds to wait for the service to become ready before evaluating.",
)
@click.option(
    "--output-format",
    default="json",
    type=click.Choice(["json", "table", "ci"]),
    help="Output format.",
)
def evaluate(profile, service_url, config_file, timeout, output_dir, startup_timeout, output_format):
    """Run accuracy/quality evaluation against the model service."""
    try:
        _configure_harness_logging()

        extra = _load_config_yaml(config_file)
        # The harness writes the results envelope, the per-task CSV and the raw
        # backend document itself, so it needs the destination; a ``--config``
        # entry of the same name still wins.
        extra.setdefault("output_dir", output_dir)

        harness, config = _harness_session(
            profile,
            service_url=service_url,
            timeout_seconds=timeout,
            output_format=output_format,
            extra=extra,
        )

        # evaluate never starts the service itself; it only waits for the one
        # already serving the image.
        if not service_url:
            resolved_url = config.resolve_service_url()
            logger.info("Waiting for service readiness at %s ...", resolved_url)
            _wait_for_harness_readiness(harness, resolved_url, startup_timeout)

        result = harness.evaluate(config)
        if output_dir:
            _write_results_json(result, output_dir, "evaluate_results.json")
        _report_and_exit(result, output_format)
    except click.ClickException:
        raise  # Let Click render usage errors (e.g. a bad --config path) itself.
    except Exception:
        logger.exception("Evaluation failed")
        sys.exit(1)


@cli.command(name="list-checks")
@click.option("--profile", default=None, help="Profile name. Defaults to auto-detected profile.")
@click.option(
    "--output-format",
    default="table",
    type=click.Choice(["json", "table"]),
    help="Output format.",
)
def list_checks(profile, output_format):
    """List available checks and their result types for this image."""
    try:
        _configure_harness_logging()

        harness, _ = _harness_session(profile, output_format=output_format)
        checks = harness.list_checks()

        if output_format == "json":
            print(
                json.dumps(
                    [
                        {
                            "name": c.name,
                            "result_type": c.result_type.value,
                            "scope": c.scope.value,
                            "description": c.description,
                        }
                        for c in checks
                    ],
                    indent=2,
                )
            )
        else:
            width = max((len(c.name) for c in checks), default=4) + 2
            header = f"{'Name':<{width}} {'Type':<14} {'Scope':<10} Description"
            print(header)
            print("-" * len(header))
            for c in checks:
                print(f"{c.name:<{width}} {c.result_type.value:<14} {c.scope.value:<10} {c.description}")

    except Exception:
        logger.exception("list-checks failed")
        sys.exit(1)


# --------------------------------------------------------------------- #
# Harness helpers
# --------------------------------------------------------------------- #


def _configure_harness_logging() -> None:
    """Apply the env-driven log levels shared by all harness commands."""
    configure_logging(
        root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
        aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
    )


def _harness_session(
    profile: str | None,
    *,
    service_url: str | None = None,
    timeout_seconds: int = 300,
    output_format: str = "json",
    check_scopes: set | None = None,
    extra: dict | None = None,
) -> "tuple[ModelHarness, HarnessConfig]":
    """Resolve the profile, discover its harness, and build the harness config.

    Every harness command needs this same trio. The profile has to be resolved
    before discovery so multi-engine images can dispatch on the engine it names.
    """
    from aim_runtime.harness import HarnessConfig
    from aim_runtime.harness.discovery import discover_harness

    resolved_profile = _resolve_profile_dict(profile)
    harness = discover_harness(profile=resolved_profile)
    config = HarnessConfig(
        profile=resolved_profile,
        service_url=service_url,
        timeout_seconds=timeout_seconds,
        output_format=output_format,
        extra=extra or {},
    )
    if check_scopes is not None:
        config.check_scopes = check_scopes
    return harness, config


def _load_config_yaml(path: str | None) -> dict:
    """Load a YAML config file and return its contents as a dict.

    Returns an empty dict when *path* is None (no ``--config`` flag).
    Raises :class:`click.BadParameter` if a path is given but the file
    is missing or contains non-dict YAML.
    """
    if not path:
        return {}
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise click.BadParameter(f"Config file not found: {path}", param_hint="'--config'")
    with open(config_path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise click.BadParameter(
            f"Config file must contain a YAML mapping, got {type(data).__name__}", param_hint="'--config'"
        )
    return data


def _resolve_profile_dict(profile_name: str | None) -> dict:
    """Resolve a profile name to its full dict representation.

    When *profile_name* is given it overrides ``AIM_PROFILE_ID`` for this
    resolution.  When it is None the runtime's auto-selection logic picks
    the best match.  The returned dict is the profile content suitable for
    passing to :class:`HarnessConfig`.

    Whichever way it was resolved, the result is pinned back into
    ``AIM_PROFILE_ID`` so that a model server the CLI starts afterwards runs
    the profile that was actually resolved here, instead of re-running
    auto-selection and possibly landing somewhere else.
    """
    try:
        if profile_name:
            os.environ["AIM_PROFILE_ID"] = profile_name

        config = AIMConfig.from_environment()
        runtime = AIMRuntime(config)
        selected = runtime.profile_selector.find_profile()
        os.environ["AIM_PROFILE_ID"] = selected.profile_id

        profile_dict = {
            "aim_id": selected.aim_id,
            "model_id": selected.model_id,
            "profile_id": selected.profile_id,
            "engine": selected.metadata.engine.value,
            "engine_args": selected.engine_args or {},
            "env_vars": selected.env_vars or {},
            "metadata": selected.metadata.to_dict(),
            "port": int(os.getenv("AIM_PORT", "8000")),
        }
        return profile_dict
    except Exception:
        logger.debug("Profile resolution failed — returning minimal dict", exc_info=True)
        return {"profile_id": profile_name} if profile_name else {}


def _write_results_json(result, output_dir: str, filename: str) -> Path:
    """Write a harness result to ``output_dir/filename`` and return the path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / filename
    results_file.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Results written to %s", results_file)
    return results_file


def _report_and_exit(result, output_format: str) -> NoReturn:
    """Print a harness result and exit with a status reflecting its verdict."""
    _report_harness_result(result, output_format)
    sys.exit(0 if result.success else 1)


def _report_harness_result(result, output_format: str) -> None:
    """Print a :class:`HarnessResult` in the requested format."""
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif output_format == "ci":
        print(json.dumps(result.to_dict()))
    else:
        print(result.summary)
        for check in result.checks:
            mark = "SKIP" if check.skipped else ("PASS" if check.success else "FAIL")
            print(f"  [{mark}] {check.name}: {check.detail or check.value}")
            for warning in check.warnings:
                print(f"         warning: {warning}")
        for key, value in result.metrics.items():
            if value is not None:
                print(f"  {key}: {value}")


@cli.command(name="detect-hardware")
@click.option(
    "--type",
    "accelerator_type",
    type=click.Choice(["gpu", "cpu", "all"], case_sensitive=False),
    default="all",
    help="Which accelerator types to detect (default: all)",
)
@click.option(
    "--format",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    default="json",
    help="Output format (default: json)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show full detection details (GPUInfo, CPUInfo)")
def detect_hardware(accelerator_type, format, verbose):
    """Detect hardware accelerators and report identifiers.

    Runs GPU and/or CPU detectors and prints the results as a list of dicts.
    Use --verbose for full detection details.

    Examples:
      aim-runtime detect-hardware
      aim-runtime detect-hardware --type gpu
      aim-runtime detect-hardware --type cpu --format yaml
      aim-runtime detect-hardware --verbose
    """
    try:
        # Suppress info/warning logs unless --verbose to keep output clean for piping
        default_aim_level = "INFO" if verbose else "ERROR"
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", default_aim_level),
        )

        # Determine which accelerator types to detect
        if accelerator_type == "all":
            types_to_detect = [AcceleratorType.GPU, AcceleratorType.CPU]
        else:
            types_to_detect = [AcceleratorType(accelerator_type)]

        detector = AcceleratorDetector()
        output: list = []

        for acc_type in types_to_detect:
            if acc_type == AcceleratorType.CPU:
                # Prefer specific EPYC detection, but fall back to generic CPU family
                # so Ryzen/Intel still report as CPU.
                cpu_result = None
                for family in (AcceleratorFamily.EPYC, AcceleratorFamily.CPU):
                    candidate = detector.detect(accelerator_type=acc_type, accelerator_family=family)
                    cpu_result = candidate
                    if candidate.accelerator_model not in (None, AcceleratorModel.CPU):
                        break
                result = cpu_result
            else:
                result = detector.detect(accelerator_type=acc_type)
            if verbose:
                output.append(result.to_detail_dict())
            else:
                output.extend(result.to_label_dicts())

        if format == "json":
            print(json.dumps(output, indent=2))
        else:
            print(dump_yaml(output))

    except Exception:
        logger.exception("Hardware detection failed")
        sys.exit(1)


def main():
    """Main entrypoint for AIM runtime."""
    cli()


if __name__ == "__main__":
    main()
