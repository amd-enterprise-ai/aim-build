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
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click
import yaml

from logging_config import configure_logging

root_log_level = os.environ.get("AIM_LOG_LEVEL_ROOT", "WARNING")
configure_logging(
    root_log_level=root_log_level,
    aim_log_level=os.environ.get("AIM_LOG_LEVEL", "INFO"),
)
os.environ["VLLM_LOGGING_LEVEL"] = root_log_level

from aim_runtime.aim_runtime import AIMRuntime  # noqa: E402
from aim_runtime.config import AIMConfig  # noqa: E402
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
            print(yaml.safe_dump(profiles_dict, sort_keys=False))

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
                print(yaml.safe_dump(serialized, sort_keys=False))
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


def _start_server_in_background(config: AIMConfig) -> subprocess.Popen:
    runtime = AIMRuntime(config)
    logger.info("Selecting profile for benchmark server...")
    profile = runtime.profile_selector.find_profile()
    logger.info(f"Selected profile: {profile.profile_handling.path}")

    command_list, env_vars = runtime.command_generator.generate_execution_params(profile)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_vars.items()})

    logger.info(f"Starting benchmark server: {' '.join(command_list)}")
    return subprocess.Popen(command_list, env=env)


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
    """Run the benchmark suite against a running AIM service."""
    server_process = None
    try:
        configure_logging(
            root_log_level=os.getenv("AIM_LOG_LEVEL_ROOT", "WARNING"),
            aim_log_level=os.getenv("AIM_LOG_LEVEL", "INFO"),
        )

        # If no service URL is provided, start the server and use the local address
        if not service_url:
            config = AIMConfig.from_environment()
            server_process = _start_server_in_background(config)

            service_url = f"http://localhost:{config.port}"

            logger.info(f"Waiting for server readiness at {service_url}...")
            _wait_for_service(service_url, startup_timeout)

        # Lazy import: avoid loading benchmarking dependencies for non-benchmark commands
        from aim_runtime.benchmarking import AIMBenchmark

        # Initialize benchmark runner
        benchmark_runner = AIMBenchmark(
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            config_file=config_file,
        )

        # Run benchmark suite
        results = benchmark_runner.run_benchmark_suite()

        # Export results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        benchmark_runner.export_results(results, output_dir=str(output_path))

        # Exit with appropriate code
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


def main():
    """Main entrypoint for AIM runtime."""
    cli()


if __name__ == "__main__":
    main()
