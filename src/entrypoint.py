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
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click

from aim_runtime.logging_config import configure_logging
from aim_runtime.utils import dump_yaml

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


def _wait_for_service(service_url: str, timeout_seconds: int, poll_interval: float = 2.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            with urlopen(Request(f"{service_url}/v1/models"), timeout=5) as response:
                if response.status == 200:
                    return
                last_error = f"status {response.status}"
        except HTTPError as exc:
            last_error = f"status {exc.code}"
        except URLError as exc:
            last_error = str(exc)

        time.sleep(poll_interval)

    raise RuntimeError(
        f"Service not ready at {service_url} after {timeout_seconds}s" + (f": {last_error}" if last_error else "")
    )


def _wait_for_harness_readiness(harness, service_url: str, timeout_seconds: int) -> None:
    """Block until ``harness.health_check`` reports the service is ready.

    Delegates to the harness so specialized engines (e.g. BentoML ``/healthz``)
    use their own readiness endpoint instead of the OpenAI-compatible
    ``/v1/models`` probe that :func:`_wait_for_service` hardcodes.
    """
    if not harness.health_check(service_url, timeout_seconds=timeout_seconds):
        raise RuntimeError(f"Service not ready at {service_url} after {timeout_seconds}s")


def _start_server_in_background(config: AIMConfig) -> subprocess.Popen:
    runtime = AIMRuntime(config)
    logger.info("Selecting profile for benchmark server...")
    profile = runtime.profile_selector.find_profile()
    logger.info(f"Selected profile: {profile.profile_handling.path}")

    command_list, env_vars = runtime.command_generator.generate_execution_params(profile)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_vars.items()})

    logger.info(f"Starting benchmark server: {shlex.join(command_list)}")
    return subprocess.Popen(command_list, env=env)


def _benchmark_via_harness(
    *,
    service_url: str | None,
    timeout_seconds: int,
    output_dir: str,
    startup_timeout: int,
    config_overrides: dict | None = None,
) -> None:
    """Run the benchmark through the discovered ModelHarness and exit."""
    from aim_runtime.harness import HarnessConfig
    from aim_runtime.harness.discovery import discover_harness

    resolved_profile = _resolve_profile_dict(None)
    harness = discover_harness(profile=resolved_profile)

    config = HarnessConfig(
        profile=resolved_profile,
        service_url=service_url,
        timeout_seconds=timeout_seconds,
        output_format="json",
        extra=config_overrides or {},
    )

    svc_url = config.resolve_service_url()
    if not service_url:
        logger.info("Waiting for service readiness at %s ...", svc_url)
        _wait_for_harness_readiness(harness, svc_url, startup_timeout)

    result = harness.benchmark(config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "benchmark_results.json"
    results_file.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Benchmark results written to %s", results_file)

    _report_harness_result(result, "json")
    sys.exit(0 if result.success else 1)


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

    When a custom ModelHarness is found (specialized images), the benchmark is
    delegated to the harness.  Otherwise, the legacy AIMBenchmark runner is used.
    """
    server_process = None
    try:
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
        )

        from aim_runtime.harness.discovery import has_custom_harness

        if has_custom_harness():
            _benchmark_via_harness(
                service_url=service_url,
                timeout_seconds=timeout_seconds,
                output_dir=output_dir,
                startup_timeout=startup_timeout,
                config_overrides=_load_config_yaml(config_file),
            )
            return

        # ---- Legacy AIMBenchmark path (vLLM / standard images) ----

        # If no service URL is provided, start the server and use the local address
        if not service_url:
            config = AIMConfig.from_environment()
            server_process = _start_server_in_background(config)

            service_url = f"http://localhost:{config.port}"

            logger.info(f"Waiting for server readiness at {service_url}...")
            _wait_for_service(service_url, startup_timeout)

        from aim_runtime.benchmarking import AIMBenchmark

        benchmark_runner = AIMBenchmark(
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            config_file=config_file,
        )

        results = benchmark_runner.run_benchmark_suite()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        benchmark_runner.export_results(results, output_dir=str(output_path))

        if results.get("overall_success"):
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        sys.exit(1)
    finally:
        if server_process:
            logger.info("Stopping benchmark server...")
            if server_process.poll() is None:
                try:
                    server_process.send_signal(signal.SIGINT)
                    server_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logger.warning("Benchmark server did not exit after SIGINT; sending SIGTERM.")
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning("Benchmark server did not exit cleanly; killing it.")
                        server_process.kill()
                except OSError:
                    pass  # Process already exited between poll() and send_signal()


@cli.command(name="validate")
@click.option("--profile", default=None, help="Profile name. Defaults to auto-detected profile.")
@click.option(
    "--service-url",
    type=str,
    required=False,
    help="Service URL including port (e.g. http://localhost:8000). Defaults to localhost:{profile.port}.",
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
@click.option("--timeout", default=300, show_default=True, help="Timeout in seconds.")
@click.option(
    "--output-format",
    default="json",
    type=click.Choice(["json", "table", "ci"]),
    help="Output format.",
)
def validate(profile, service_url, scope, config_file, timeout, output_format):
    """Validate that the model service is healthy and producing correct output."""
    try:
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
        )

        from aim_runtime.harness import CheckScope, HarnessConfig
        from aim_runtime.harness.discovery import discover_harness, has_custom_harness

        if not has_custom_harness():
            logger.error("No custom harness found — validate requires a ModelHarness implementation.")
            sys.exit(1)

        resolved_profile = _resolve_profile_dict(profile)
        harness = discover_harness(profile=resolved_profile)
        config = HarnessConfig(
            profile=resolved_profile,
            service_url=service_url,
            timeout_seconds=timeout,
            output_format=output_format,
            check_scopes={CheckScope(s) for s in scope},
            extra=_load_config_yaml(config_file),
        )
        result = harness.validate(config)
        _report_harness_result(result, output_format)
        sys.exit(0 if result.success else 1)

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
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
        )

        from aim_runtime.harness import HarnessConfig
        from aim_runtime.harness.discovery import discover_harness, has_custom_harness

        if not has_custom_harness():
            logger.error("No custom harness found — evaluate requires a ModelHarness implementation.")
            sys.exit(1)

        resolved_profile = _resolve_profile_dict(profile)
        harness = discover_harness(profile=resolved_profile)
        config = HarnessConfig(
            profile=resolved_profile,
            service_url=service_url,
            timeout_seconds=timeout,
            output_format=output_format,
            extra=_load_config_yaml(config_file),
        )

        svc_url = config.resolve_service_url()
        if not service_url:
            logger.info("Waiting for service readiness at %s ...", svc_url)
            _wait_for_harness_readiness(harness, svc_url, startup_timeout)

        result = harness.evaluate(config)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            results_file = out / "evaluate_results.json"
            results_file.write_text(json.dumps(result.to_dict(), indent=2))
            logger.info("Evaluation results written to %s", results_file)

        _report_harness_result(result, output_format)
        sys.exit(0 if result.success else 1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
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
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
        )

        from aim_runtime.harness.discovery import discover_harness, has_custom_harness

        if not has_custom_harness():
            logger.error("No custom harness found — list-checks requires a ModelHarness implementation.")
            sys.exit(1)

        resolved_profile = _resolve_profile_dict(profile)
        harness = discover_harness(profile=resolved_profile)
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
            header = f"{'Name':<22} {'Type':<14} {'Scope':<10} Description"
            print(header)
            print("-" * len(header))
            for c in checks:
                print(f"{c.name:<22} {c.result_type.value:<14} {c.scope.value:<10} {c.description}")

    except Exception as e:
        logger.error(f"list-checks failed: {e}")
        sys.exit(1)


# --------------------------------------------------------------------- #
# Harness helpers
# --------------------------------------------------------------------- #


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
    """
    try:
        if profile_name:
            os.environ["AIM_PROFILE_ID"] = profile_name

        config = AIMConfig.from_environment()
        runtime = AIMRuntime(config)
        selected = runtime.profile_selector.find_profile()

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


def _report_harness_result(result, output_format: str) -> None:
    """Print a :class:`HarnessResult` in the requested format."""
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif output_format == "ci":
        print(json.dumps(result.to_dict()))
    else:
        print(result.summary)
        for check in result.checks:
            mark = "PASS" if check.success else "FAIL"
            print(f"  [{mark}] {check.name}: {check.detail or check.value}")


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
