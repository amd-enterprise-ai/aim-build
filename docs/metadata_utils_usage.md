<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Metadata Documentation

## Overview

This document describes the usage and management of metadata for AIM (AMD Inference Microservice). See
[Metadata Overview](metadata_overview.md) for the description of the fields and their meanings.

## Usage

### Accessing Metadata in Python

Metadata can be managed using the `metadata_utils.py` module. It contains a bunch of methods to perform various
specific operations on metadata files, such as initializing, validating, and extracting specific field values.

Supported commands:

| Command                     | Purpose                                             |
| --------------------------- | --------------------------------------------------- |
| init                        | Initialize metadata files with default values       |
| delete                      | Delete all metadata files                           |
| delete-key                  | Delete specific keys from metadata                  |
| update-value                | Update values in metadata                           |
| copy-value                  | Copy values between metadata fields                 |
| rename-key                  | Rename keys in metadata                             |
| add-recommended-deployments | Add or update recommended deployment configurations |
| validate                    | Validate metadata files against schemas             |
| list-keys                   | List all keys in metadata files                     |

### Command Examples

General pattern for command execution:

```bash
python -m aim_utils.metadata_utils COMMAND --canonical_name=<Name in 'org/model' format>
```

If `--canonical_name` is specified, the command will target only the metadata file for that specific model. Otherwise, it will
operate on all metadata files in the given `METADATA_PATH`.

**Initialize metadata files:**

```bash
python -m aim_utils.metadata_utils init
```

Options:

| Name              | Purpose                 | Default  |
| ----------------- | ----------------------- | -------- |
| `--metadata_path` | Path to metadata folder | metadata |
| `--profiles_path` | Path to profiles folder | profiles |

**Delete all metadata files:**

```bash
python -m aim_utils.metadata_utils delete
```

Options:

| Name              | Purpose                 | Default  |
| ----------------- | ----------------------- | -------- |
| `--metadata_path` | Path to metadata folder | metadata |

**Delete a specific key from metadata:**

```bash
python -m aim_utils.metadata_utils delete-key "com.amd.aim.model.tags" --canonical_name="meta-llama/Llama-3.1-8B-Instruct"
```

Arguments and options:

| Name               | Purpose                                | Default  |
| ------------------ | -------------------------------------- | -------- |
| `key`              | Key name to delete                     |          |
| `--metadata_path`  | Path to metadata folder                | metadata |
| `--canonical_name` | If specified, apply only to that model | None     |

**Update a value in metadata:**

```bash
python -m aim_utils.metadata_utils update-value "org.opencontainers.image.vendor" "AMD" --canonical_name="meta-llama/Llama-3.1-8B-Instruct"
```

Arguments and options:

| Name               | Purpose                                | Default  |
| ------------------ | -------------------------------------- | -------- |
| `key`              | Key name to update                     |          |
| `new_value`        | New value to set for the key           |          |
| `--metadata_path`  | Path to metadata folder                | metadata |
| `--canonical_name` | If specified, apply only to that model | None     |
| `--add_if_missing` | Add the key if it does not exist       | False    |

**Copy a value between fields:**

```bash
python -m aim_utils.metadata_utils copy-value "com.amd.aim.model.canonicalName" "org.opencontainers.image.title" --prefix="AIM:" --separator=" "
```

Arguments and options:

| Name               | Purpose                                                       | Default        |
| ------------------ |---------------------------------------------------------------| -------------- |
| `source_key`       | Key name to take a value from                                 |                |
| `target_key`       | Key name to set a value to                                    |                |
| `--metadata_path`  | Path to metadata folder                                       | metadata       |
| `--canonical_name` | If specified, apply only to that model                        | None           |
| `--add_if_missing` | Add the key if it does not exist                              | False          |
| `--prefix`         | An addition before the resulting value                        | None           |
| `--postfix`        | An addition after the resulting value                         | None           |
| `--separator`      | Filler value between prefix, postfix, and the resulting value | <EMPTY STRING> |

**Rename a key:**

```bash
python -m aim_utils.metadata_utils rename-key "com.amd.aim.oldKey" "com.amd.aim.newKey"
```

Arguments and options:

| Name               | Purpose                                | Default  |
| ------------------ | -------------------------------------- | -------- |
| `source_key`       | Old key name                           |          |
| `target_key`       | New key name                           |          |
| `--metadata_path`  | Path to metadata folder                | metadata |
| `--canonical_name` | If specified, apply only to that model | None     |

**Add recommended deployments:**

```bash
python -m aim_utils.metadata_utils add-recommended-deployments --canonical_name="meta-llama/Llama-3.1-8B-Instruct"
```

Options:

| Name               | Purpose                                | Default  |
| ------------------ | -------------------------------------- | -------- |
| `--metadata_path`  | Path to metadata folder                | metadata |
| `--canonical_name` | If specified, apply only to that model | None     |

**Validate metadata files:**

```bash
python -m aim_utils.metadata_utils validate
```

Options:

| Name               | Purpose                                | Default  |
| ------------------ | -------------------------------------- | -------- |
| `--metadata_path`  | Path to metadata folder                | metadata |
| `--canonical_name` | If specified, apply only to that model | None     |

**List all keys in metadata:**

```bash
python -m aim_utils.metadata_utils list-keys
```

Options:

| Name               | Purpose                                | Default  |
| ------------------ | -------------------------------------- | -------- |
| `--metadata_path`  | Path to metadata folder                | metadata |
| `--canonical_name` | If specified, apply only to that model | None     |

### Extracting Specific Fields

Common metadata extraction patterns:

```python
# Get canonical name
from pathlib import Path
from aim_utils.metadata_utils import get_value
from aim_utils.yaml_utils import read_yaml

data = read_yaml(Path("metadata/org/model/metadata.yaml"))
canonical_name = get_value(data, "com.amd.aim.model.canonicalName")
```

## Adding a New Model

To add metadata for a new model, follow these steps:

1. **Create the directory structure**:

The name of the model will be taken from profiles folder. The following command will create the necessary directory
structure and put a default metadata.yaml file in it.

```bash
python -m aim_utils.metadata_utils init
```

2. **Populate the metadata** following the schema and examples above. Metadata files are populated manually, except the
   `recommendedDeployments` part. You can generate it using:

   ```bash
   python -m aim_utils.metadata_utils add-recommended-deployments --canonical_name=<Name in 'org/model' format>
   ```

3. **Validate the metadata**:
   ```bash
   python -m aim_utils.metadata_utils validate --canonical_name=<Name in 'org/model' format>
   ```

## Best Practices

1. **Keep descriptions concise**: `org.opencontainers.image.description` has a 160-character limit. `com.amd.aim.description.full` has no length constraint and can contain comprehensive technical details.

2. **Specify accurate GPU requirements**: Use the `recommendedDeployments` section to guide users on optimal hardware configurations.

3. **Include all variants**: List all available model variants, including quantized versions.

4. **Set HF token correctly**: Ensure the `hfToken.required` field accurately reflects whether the model requires authentication.

5. **Validate before committing**: Always validate metadata against the schema before committing changes.

6. **Keep licenses accurate**: Ensure the license information matches the original model's license.

## Schema Files

The complete JSON schemas can be found in:

- `/schemas/metadata_schema.json`
- `/schemas/base_metadata_schema.json`

These files provide specification for the metadata structure and validation rules.
