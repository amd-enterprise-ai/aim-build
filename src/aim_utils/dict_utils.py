# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_value(dict_value: Dict[str, Any], key_path: str) -> Optional[Any]:
    """Retrieve a value from a nested dictionary using dot notation"""
    keys = key_path.split(".")
    current = dict_value
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def set_value(
    dict_value: Dict[str, Any], key_path: str, new_value: Any, add_if_missing: bool = False
) -> Dict[str, Any]:
    keys = key_path.split(".")
    current = dict_value
    for i, k in enumerate(keys):
        # If we're at the last key, update its value
        if i == len(keys) - 1:
            if k in current:
                old_value = current[k]
                current[k] = new_value
                logger.debug(f"Updated {key_path} from '{old_value}' to '{new_value}'")
            else:
                keys_str = ".".join(keys)
                if add_if_missing:
                    current[k] = new_value
                    logger.debug(f"Key '{keys_str}' was added")
                else:
                    logger.debug(f"Key '{keys_str}' was not found")
                break
        else:
            # Navigate to the next level
            if k in current:
                current = current[k]
            else:
                if add_if_missing:
                    current[k] = {}
                    current = current[k]
                    logger.debug(f"Key '{k}' was added")
                else:
                    logger.debug(f"Key '{key_path}' doesn't exist")
                    break

    return dict_value


def delete_key(dict_value: Dict[str, Any], key: str) -> Dict[str, Any]:
    """
    Remove a specific key from a specified metadata.yaml file.

    Args:
        dict_value: Dictionary with hierarchical keys
        key: Dot notation path to the key (e.g., "org.opencontainers.image.vendor")
    """
    keys = key.split(".")
    current = dict_value
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            if isinstance(current, dict) and k in current:
                del current[k]
                logger.debug(f"Removed key '{key}'")
            else:
                logger.debug(f"Key '{key}' not found")
        else:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                logger.debug(f"Key '{key}' not found")
                break

    return dict_value
