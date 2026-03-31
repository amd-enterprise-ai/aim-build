# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
OCI metadata label generation for AIM container images.

Generates OCI-standard labels from metadata YAML files and environment variables.
Used by both CI (via .github/actions/build-image/) and manual builds (via Makefile).

Usage:
    python -m aim_utils.generate_labels <yaml_path>

Environment variables:
    LABELS_SERVER_URL      - Source control server URL (e.g. https://github.com)
    LABELS_REPOSITORY      - Repository path (e.g. silogen/aim-build)
    LABELS_VERSION_NUMBER  - Image version string
    LABELS_IS_BASE         - "true" for base images, "false" for model images
    LABELS_ORG             - Model organization (required when IS_BASE=false)
    LABELS_MODEL_NAME      - Model name (required when IS_BASE=false)
    LABELS_SHA             - Git commit SHA
    LABELS_TIMESTAMP       - Build timestamp (ISO 8601)
    LABELS_UPDATED_AT      - Fallback timestamp if LABELS_TIMESTAMP is not set
    LABELS_IS_DOCKER_FORMAT - "true" to output --label flags for docker CLI
"""

import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def get_all_key_values(yaml_path: Path) -> Dict[str, str]:
    """Recursively extract all key-value pairs from a YAML file, using dot-separated keys."""

    def to_string(value: Any) -> str:
        if isinstance(value, list):
            if value:
                if isinstance(value[0], dict):
                    return str(value)
            return ", ".join(map(str, value))

        return str(value)

    try:
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        if not yaml_data or not isinstance(yaml_data, dict):
            logger.warning(f"File {yaml_path} does not contain a valid YAML dictionary")
            return {}

        key_values = {}

        def traverse(data: Any, prefix: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict) and value:
                        traverse(value, new_prefix)
                    else:
                        key_values[new_prefix] = to_string(value)

        traverse(yaml_data)
        return key_values

    except Exception as e:
        logger.error(f"Error processing {yaml_path}: {str(e)}")
        raise e


def date_time_now() -> str:
    """Return current time as an ISO 8601 string (seconds precision, timezone-aware)."""
    now_aware = datetime.datetime.now().astimezone()
    iso_string = now_aware.isoformat(timespec="seconds")
    return iso_string


def generate_labels(yaml_path: str) -> Dict[str, str]:
    """Build the full label dict from a metadata YAML file and LABELS_* env vars."""
    server_url = os.getenv("LABELS_SERVER_URL", "")
    repository = os.getenv("LABELS_REPOSITORY", "")
    version_number = os.getenv("LABELS_VERSION_NUMBER", "")
    is_base = os.getenv("LABELS_IS_BASE", "false").lower() == "true"
    org = os.getenv("LABELS_ORG", "")
    model_name = os.getenv("LABELS_MODEL_NAME", "")
    sha = os.getenv("LABELS_SHA", "")
    timestamp = os.getenv("LABELS_TIMESTAMP")
    updated_at = os.getenv("LABELS_UPDATED_AT")

    if not is_base and not model_name:
        raise ValueError("LABELS_MODEL_NAME is required when LABELS_IS_BASE is false")

    title = "base" if is_base else f"{org}/{model_name}"

    created = timestamp or updated_at
    if created is None:
        created = date_time_now()

    result = {
        "org.opencontainers.image.source": f"{server_url}/{repository}",
        "org.opencontainers.image.version": version_number,
        "org.opencontainers.image.title": title,
        "org.opencontainers.image.revision": sha,
        "org.opencontainers.image.created": created,
    }

    all_key_values = get_all_key_values(Path(yaml_path))

    for key, value in all_key_values.items():
        result.setdefault(key, value)

    return dict(sorted(result.items()))


def format_labels(labels: Dict[str, str], docker_format: bool = False) -> str:
    """Format label dict as either newline-separated key=value or --label CLI flags."""
    if docker_format:
        string_result = [f'--label "{key}={value}"' for key, value in labels.items()]
        return " ".join(string_result)
    else:
        string_result = [f"{key}={value}" for key, value in labels.items()]
        return "\n".join(string_result)


def main() -> None:
    """CLI entrypoint: generate and print OCI labels from a metadata YAML file."""
    if len(sys.argv) < 2:
        logger.error("Usage: python -m aim_utils.generate_labels <yaml_path>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    if not yaml_path:
        raise ValueError("YAML_PATH argument is required")

    is_docker_format = os.getenv("LABELS_IS_DOCKER_FORMAT", "false").lower() == "true"

    labels = generate_labels(yaml_path)
    print(format_labels(labels, docker_format=is_docker_format))


if __name__ == "__main__":
    main()
