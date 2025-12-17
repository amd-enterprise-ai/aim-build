#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Sort non-top-level keys in profile YAML files alphabetically."""

import argparse
import sys
from io import StringIO
from pathlib import Path

import yaml


def sort_yaml_file(file_path: Path) -> bool:
    """
    Sort non-top-level keys in a YAML file alphabetically.

    Args:
        file_path: Path to the YAML file

    Returns:
        True if file was modified, False otherwise
    """
    # Read the original file to preserve comments and structure
    with open(file_path, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Parse YAML
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return False

    # Check which sections need sorting
    modified = False

    for section in ["metadata", "engine_args", "env_vars"]:
        if section in data and isinstance(data[section], dict):
            original_keys = list(data[section].keys())
            sorted_keys = sorted(original_keys)

            if original_keys != sorted_keys:
                modified = True
                # Create new ordered dict with sorted keys
                sorted_section = {k: data[section][k] for k in sorted_keys}
                data[section] = sorted_section

    if modified:
        # Write back to file preserving format
        output = StringIO()
        yaml.dump(
            data,
            output,
            Dumper=yaml.SafeDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=float("inf"),
        )

        new_content = output.getvalue()

        # Preserve the copyright header from original file
        lines = original_content.split("\n")
        header_lines = []
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                header_lines.append(line)
            else:
                break

        # Combine header with new content
        if header_lines:
            header = "\n".join(header_lines).rstrip() + "\n\n"
            # Remove any header from new_content if it exists
            new_lines = new_content.split("\n")
            content_start = 0
            for i, line in enumerate(new_lines):
                if line and not line.startswith("#"):
                    content_start = i
                    break
            new_content = "\n".join(new_lines[content_start:])
            final_content = header + new_content
        else:
            final_content = new_content

        # Write back to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"Sorted keys in: {file_path}")

    return modified


def main():
    """Sort all YAML files in the profiles directory."""
    parser = argparse.ArgumentParser(
        description="Sort non-top-level keys in profile YAML files",
        epilog="Without arguments, sorts all YAML files in the profiles directory.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Absolute paths to specific profile files to sort (used by pre-commit)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    profiles_dir = repo_root / "profiles"

    if not profiles_dir.exists():
        print(f"Error: profiles directory not found at {profiles_dir}")
        return 1

    # Handle files from pre-commit or command line
    if args.files:
        modified_count = 0
        error_count = 0

        for file_arg in args.files:
            file_path = Path(file_arg)

            # Require absolute path
            if not file_path.is_absolute():
                print(f"Error: Please provide an absolute path. Got: {file_arg}")
                return 1

            if not file_path.exists():
                print(f"Error: file not found at {file_path}")
                return 1

            try:
                if sort_yaml_file(file_path):
                    modified_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                error_count += 1

        # For pre-commit: return 1 if files were modified (so they get staged)
        return 1 if modified_count > 0 else 0

    # Find all YAML files in profiles directory
    yaml_files = list(profiles_dir.rglob("*.yaml"))

    if not yaml_files:
        print("No YAML files found in profiles directory")
        return 0

    print(f"Processing {len(yaml_files)} YAML files...")

    modified_count = 0
    error_count = 0

    for yaml_file in sorted(yaml_files):
        try:
            if sort_yaml_file(yaml_file):
                modified_count += 1
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            error_count += 1

    print("\n✅ Processing complete:")
    print(f"   - {len(yaml_files)} files processed")
    print(f"   - {modified_count} files modified")
    print(f"   - {error_count} errors")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
