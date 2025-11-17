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
import sys
from pathlib import Path

import click

from aim_runtime.aim_runtime import AIMRuntime
from aim_runtime.config import AIMConfig
from aim_runtime.logging_config import configure_logging
from aim_runtime.profile_selector import ProfileCompatibilityState, ProfileSelector

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

        if format == "json":
            # Return all compatible profiles as JSON
            profiles_dict = runtime.dry_run_json()
            print(json.dumps(profiles_dict, indent=2))
        else:
            # Display the selected profile as YAML
            yaml_output = runtime.dry_run()
            print(yaml_output)

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


@cli.command(name="download-to-cache")
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Explicit model id to download (e.g. hf://org/model or s3://path/to/model). Overrides profile selection.",
)
@click.option(
    "--use-hf-cache",
    is_flag=True,
    default=False,
    help="Use HuggingFace's default cache directory structure instead of downloading directly to local directory.",
)
@click.option(
    "--custom-model-name",
    type=str,
    default=None,
    help="Custom directory name for S3 downloads (e.g. 'org/model' or 'custom-name'). If not provided, auto-detects from S3 URI. Ignored for HuggingFace downloads. Can also be set via AIM_CUSTOM_MODEL_NAME env var.",
)
def download_to_cache(model_id, use_hf_cache, custom_model_name):
    """Download the model to cache.

    Downloads the model to the cache directory specified by AIM_CACHE_PATH environment variable.
    By default, downloads directly to the local directory (local-dir mode). Use --use-hf-cache
    to download using HuggingFace's default cache structure instead.

    If --model-id is not provided, uses the current configuration to determine the model.

    For S3 downloads, use --custom-model-name to specify a custom directory name.

    Examples:
      aim-runtime download-to-cache
      aim-runtime download-to-cache --model-id hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0
      aim-runtime download-to-cache --use-hf-cache
      aim-runtime download-to-cache --model-id s3://bucket/path/to/model --custom-model-name org/custom-model
    """
    try:
        # Load configuration from environment variables
        config = AIMConfig.from_environment()

        # Configure logging based on the config
        configure_logging(root_log_level=config.log_level_root, aim_log_level=config.log_level)

        # Create runtime
        runtime = AIMRuntime(config)

        # Download the model
        # Custom model name from CLI takes precedence over env var
        final_custom_model_name = custom_model_name or config.custom_model_name
        downloaded_path = runtime.download_to_cache(
            model_id=model_id, use_hf_cache=use_hf_cache, custom_model_name=final_custom_model_name
        )

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
    type=click.Choice(["text", "table"], case_sensitive=False),
    default="table",
    help="Output format: text (grouped by state) or table (all profiles in table) (default: table)",
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
      aim-runtime list-profiles --format table
      aim-runtime list-profiles --state gpu_mismatch --format table --verbose
      aim-runtime list-profiles --skip-compatibility-check --format table
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

        if skip_compatibility_check:
            # Skip GPU detection and compatibility checks - list all profiles
            output = selector.format_all_profiles_report(format_type=format)
        else:
            # Get categorized profiles with compatibility checks
            categorized = selector.get_categorized_profiles()

            # Filter by state if specified
            if state != "all":
                state_key = ProfileCompatibilityState(state)
                categorized = {state_key: categorized[state_key]}

            # Output results
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


def main():
    """Main entrypoint for AIM runtime."""
    cli()


if __name__ == "__main__":
    main()
