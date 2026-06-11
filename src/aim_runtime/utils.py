# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary."""
    if not path or not path.exists():
        error_message = f"YAML file does not exist: {path}"
        logger.error(error_message)
        raise ValueError(error_message)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        error_message = f"YAML file must contain a top-level mapping: {path} (got {type(data).__name__})"
        logger.error(error_message)
        raise ValueError(error_message)

    logger.debug(f"Read YAML from {path}")
    return data


def dump_yaml(data: Any, **kwargs) -> str:
    """Serialize a Python object to a YAML string."""
    return yaml.safe_dump(data, sort_keys=False, **kwargs)
