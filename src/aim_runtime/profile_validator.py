# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from typing import Any

from jsonschema import validate
from referencing import Registry, Resource


class ProfileValidator:
    """
    Validates AIM profile YAML files against profile schema.

    Loads and registers a hierarchy of schemas:
        general -> model profile

    The schema for vLLM engine is used by the general schema to validate engine arguments.
    """

    def __init__(self, schema_search_path: str) -> None:
        """
        Initialize the ProfileValidator.

        Loads the required JSON schema files from the given directory and registers them
        for reference resolution during validation.

        Args:
            schema_search_path (str): Path to the directory containing the schema files.

        """
        self.schema_search_path = Path(schema_search_path)

        with open(self.schema_search_path / "general_profile_schema.json", "r") as f:
            self.general_schema_contents = json.load(f)
        with open(self.schema_search_path / "vllm_engine_schema.json", "r") as f:
            self.vllm_schema_contents = json.load(f)
        with open(self.schema_search_path / "model_profile_schema.json", "r") as f:
            self.model_schema_contents = json.load(f)

        general_profile_schema_resource = Resource.from_contents(self.general_schema_contents)
        vllm_resource = Resource.from_contents(self.vllm_schema_contents)
        model_profile_schema_resource = Resource.from_contents(self.model_schema_contents)

        self.registry = Registry().with_resources(
            [
                ("general_profile_schema.json", general_profile_schema_resource),
                ("vllm_engine_schema.json", vllm_resource),
                ("model_profile_schema.json", model_profile_schema_resource),
            ]
        )

    def validate(self, profile_data: dict[str, Any], is_general_profile: bool = False) -> None:
        """
        Validate already-loaded profile data against the appropriate schema.
        Args:
            profile_data (dict): Already loaded YAML profile data.
            is_general_profile (bool): If True, validate against the general profile schema.
                                        Otherwise, validate against the model profile schema.

        Raises:
            jsonschema.ValidationError: If the profile does not comply with the schema.
        """
        if is_general_profile:
            validate(instance=profile_data, schema=self.general_schema_contents, registry=self.registry)
        else:
            validate(instance=profile_data, schema=self.model_schema_contents, registry=self.registry)
