#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT


import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

# Directories to always skip
# .git is not in .gitignore since it's git's own directory
# requirements dir typically contains dependency files that don't need copyright headers
EXCLUDED_DIRS = {".git", "requirements"}

# Specific filenames to exclude (case-insensitive)
# These are files that typically don't need copyright headers
EXCLUDED_FILENAMES = {
    "license",
    "license.txt",
    "license.md",
    "copying",
    "copying.txt",
}

# File extensions to exclude from copyright check

# File extensions to exclude from copyright check
EXCLUDED_EXTENSIONS = {
    ".bin",
    ".cache",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".json",
    ".log",
    ".map",
    ".mjs",
    ".mod",
    ".pdf",
    ".png",
    ".pyc",
    ".sum",
    ".svg",
    ".tar",
    ".tmp",
    ".tsv",
    ".webp",
    ".xml",
    ".zip",
}

# Comment styles grouped by type
COMMENT_STYLES_GROUPS = {
    # Hash/pound comments
    ("# ", ""): [
        ".bash",
        ".cfg",
        ".dockerfile",
        ".dockerignore",
        ".env",
        ".fish",
        ".flake8",
        ".gitattributes",
        ".gitignore",
        ".ini",
        ".jinja",
        ".jsonl",
        ".lock",
        ".mdc",
        ".prettierignore",
        ".prettierrc",
        ".puml",
        ".py",
        ".resource",
        ".robot",
        ".sh",
        ".shellcheckrc",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
        ".zsh",
        "dockerfile",
        "makefile",
    ],
    # Double slash comments
    ("// ", ""): [
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
    ],
    # HTML/XML style comments
    ("<!--\n", "\n-->"): [".html", ".htm", ".md", ".markdown"],
    # CSS style comments
    ("/*\n", "\n*/"): [".css", ".scss", ".sass", ".less"],
    # SQL comments
    ("-- ", ""): [".sql"],
}

# Default comment style for unknown file types
DEFAULT_COMMENT_STYLE = ("# ", "")

COPYRIGHT_PATTERNS = [
    re.compile(
        r"Copyright\s*©\s*Advanced\s+Micro\s+Devices,\s*Inc\.,\s*or\s+its\s+affiliates\.",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"Copyright\s*\(c\)\s*Advanced\s+Micro\s+Devices,\s*Inc\.,\s*or\s+its\s+affiliates\.",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"SPDX-License-Identifier:\s*MIT",
        re.MULTILINE | re.IGNORECASE,
    ),
]

COPYRIGHT_TEXT = "Copyright © Advanced Micro Devices, Inc., or its affiliates.\n\nSPDX-License-Identifier: MIT"


def find_git_root():
    git_root = Path.cwd()
    while git_root != git_root.parent and not (git_root / ".git").exists():
        git_root = git_root.parent
    return git_root


def get_gitignore_patterns():
    patterns = []
    git_root = find_git_root()

    for gitignore_path in git_root.rglob(".gitignore"):
        try:
            gitignore_dir = gitignore_path.parent
            relative_dir = gitignore_dir.relative_to(git_root)

            with open(gitignore_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if relative_dir != Path("."):
                            # Prefix patterns with the relative path of the .gitignore file
                            if line.startswith("/"):
                                pattern = str(relative_dir / line[1:])
                            else:
                                pattern = str(relative_dir / line)
                            patterns.append(pattern)
                        else:
                            patterns.append(line)
        except Exception:  #
            pass
    return patterns


def is_gitignored(file_path, patterns):
    try:
        git_root = find_git_root()
        relative_path = file_path.relative_to(git_root)
    except ValueError:
        return False

    relative_path_str = str(relative_path)

    # check folders and files matches
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern - check if any part of the path matches
            dir_pattern = pattern[:-1]
            # Check each part of the path against the pattern (handles wildcards)
            for part in relative_path.parts:
                if fnmatch.fnmatch(part, dir_pattern):
                    return True
            # Also check exact matches and prefix matches
            if any(part == dir_pattern for part in relative_path.parts):
                return True
            if relative_path_str.startswith(dir_pattern + "/"):
                return True
        else:
            if fnmatch.fnmatch(relative_path_str, pattern):
                return True
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
            if relative_path_str.startswith(pattern + "/"):
                return True
    return False


def has_copyright(file_path):
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read(1000)
        has_copyright_line = COPYRIGHT_PATTERNS[0].search(content) or COPYRIGHT_PATTERNS[1].search(content)
        has_license_line = COPYRIGHT_PATTERNS[2].search(content)
        return bool(has_copyright_line and has_license_line)
    except Exception:
        return False


def should_skip_directory(file_path):
    """Check if file is in an excluded directory."""
    return any(excluded_dir in file_path.parts for excluded_dir in EXCLUDED_DIRS)


def should_skip_filename(file_path):
    """Check if filename should be excluded."""
    filename_lower = file_path.name.lower()
    return filename_lower in EXCLUDED_FILENAMES


def check_files(file_paths):
    gitignore_patterns = get_gitignore_patterns()
    missing = []

    for file_path in file_paths:
        # Skip .git directory explicitly (it's not in .gitignore since it's git's own directory)
        if should_skip_directory(file_path):
            continue

        # Skip excluded filenames
        if should_skip_filename(file_path):
            continue

        if (
            file_path.is_file()
            and not is_gitignored(file_path, gitignore_patterns)
            and file_path.suffix not in EXCLUDED_EXTENSIONS
            and get_comment_style(file_path)  # Only check files that support comments
            and not has_copyright(file_path)
        ):
            missing.append(file_path)

    return missing


def get_comment_style(file_path):
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()

    # Search through comment style groups
    for (comment_start, comment_end), extensions in COMMENT_STYLES_GROUPS.items():
        if suffix in extensions or name in extensions:
            return (comment_start, comment_end)

    # Use default comment style with warning
    print(
        f"⚠️  Warning: Unknown file type '{file_path}', using default hash comments. Consider adding to COMMENT_STYLES_GROUPS."
    )
    return DEFAULT_COMMENT_STYLE


def create_copyright_header(comment_start, comment_end):
    if comment_end:  # Multi-line comment style
        return f"{comment_start}{COPYRIGHT_TEXT}{comment_end}\n"
    else:  # Single-line comment style
        lines = COPYRIGHT_TEXT.split("\n")
        header_lines = []
        for line in lines:
            if line.strip():
                header_lines.append(f"{comment_start}{line}")
            else:
                header_lines.append(comment_start.rstrip())
        return "\n".join(header_lines) + "\n"


def add_copyright_to_file(file_path):
    comment_style = get_comment_style(file_path)
    if not comment_style:
        return False
    comment_start, comment_end = comment_style

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        copyright_header = create_copyright_header(comment_start, comment_end)

        # Handle shebang lines
        if content.startswith("#!"):
            lines = content.split("\n", 1)
            if len(lines) > 1:
                new_content = lines[0] + "\n\n" + copyright_header + lines[1]
            else:
                new_content = lines[0] + "\n\n" + copyright_header
        else:
            if content.strip():
                new_content = copyright_header + "\n" + content
            else:
                new_content = copyright_header

        # Ensure file ends with newline
        if not new_content.endswith("\n"):
            new_content += "\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True
    except Exception:
        return False


def add_copyright_to_files(files):
    success_count = 0
    for file_path in files:
        if add_copyright_to_file(file_path):
            success_count += 1
            print(f"✅ Added copyright to {file_path}")
        else:
            print(f"❌ Failed to add copyright to {file_path}")

    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for AMD copyright boilerplate")
    parser.add_argument("--fix", action="store_true", help="Automatically add missing copyright headers")
    parser.add_argument("files", nargs="*", help="Files to check (if not provided, checks all files)")

    args = parser.parse_args()

    # Auto-enable --fix in non-CI environments
    is_ci = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"
    if not args.fix and not is_ci:
        args.fix = True

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = list(Path.cwd().rglob("*"))
        if not args.fix:
            print("Checking for AMD copyright boilerplate...")

    missing = check_files(files)

    if missing:
        if args.fix:
            print(f"Adding copyright headers to {len(missing)} files...")
            success_count = add_copyright_to_files(missing)
            print(f"\n✅ Successfully added copyright headers to {success_count}/{len(missing)} files!")
            sys.exit(0 if success_count == len(missing) else 1)
        else:
            print(f"\n❌ Found {len(missing)} files without AMD copyright:")
            for f in sorted(missing):
                print(f"  - {f}")
            print("\nExpected format:")
            print("Copyright © Advanced Micro Devices, Inc., or its affiliates.")
            print("")
            print("SPDX-License-Identifier: MIT")
            print("\nRun with --fix to automatically add missing headers")
            sys.exit(1)
    elif not args.files:
        print("✅ All files have proper AMD copyright!")

    sys.exit(0)
