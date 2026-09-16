# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Lint checks for published assets that the Pydantic schemas can't express.

The schema validators (`profile_utils`, `metadata_utils`, `config_utils`) check
that a file's *structure* is right. These checks look at its *content*: leftover
scaffolding, internal notes that shouldn't be published, and dependency
specifiers that make a build unreproducible.

Each check reports the section of docs/aim_conventions.md that explains the rule.
"""

import io
import logging
import re
import sys
import tokenize
import tomllib
from pathlib import Path
from typing import Iterable, List, Pattern, Sequence, Tuple

import click
import yaml

from .asset_utils import assets_root_option

logger = logging.getLogger(__name__)

CONVENTIONS_DOC = "docs/aim_conventions.md"

# Scaffolding markers left behind when a generated AIM is not filled in. These
# reach runtime as literal strings — an unsubstituted __AIM_ID__ fails profile
# lookup with ProfileNotFound only once the image is deployed.
PLACEHOLDER_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"__[A-Z][A-Z0-9_]*__"),
    re.compile(r"\[AGENT:"),
    re.compile(r"<(?:TODO|FIXME|PLACEHOLDER)>", re.IGNORECASE),
)

# Narration from the agent or human that generated an asset. Harmless to the
# runtime, but assets/ is a public repository and these leak internal process.
INTERNAL_COMMENT_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"\bDECISION\b"),
    re.compile(r"\bController override\b", re.IGNORECASE),
    re.compile(r"\bAGENT NOTE\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Z0-9]+-\d{2,}\b"),
)

# A requirement line is acceptable only if it pins an exact version or an
# immutable VCS revision. Anything looser resolves differently on a later build,
# so the image stops matching the one CI validated without a repo change.
EXACT_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]+\])?\s*==\s*[^\s,*]+(?:\s*;.*)?$")
GIT_SHA_RE = re.compile(r"(?:^|\s@\s|git\+\S+@)([0-9a-fA-F]{40})(?:#|\s|$)")

# pip options that carry no requirement to pin. Editable installs are handled as
# requirements because their VCS revision must still be immutable.
NON_REQUIREMENT_PREFIXES = ("#", "-r ", "--requirement ", "-c ", "--constraint ", "--")


def _iter_files(files: Sequence[Path], assets_root: str, glob: str) -> List[Path]:
    """Resolve the file set: explicit paths from pre-commit, else a full scan."""
    if files:
        return [f for f in files if f.is_file()]
    return sorted(p for p in Path(assets_root).glob(glob) if p.is_file())


def _yaml_comment_start(line: str) -> int:
    """Return the first YAML comment marker outside a quoted scalar."""
    quote = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote == '"':
            if char == "\\":
                index += 2
                continue
            if char == '"':
                quote = None
        elif quote == "'":
            if char == "'" and index + 1 < len(line) and line[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return index
        index += 1
    return -1


def _scan_lines(path: Path, patterns: Iterable[Pattern[str]], comments_only: bool = False) -> List[str]:
    """Return "path:line: text" for every line matching any pattern."""
    hits = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        haystack = line
        if comments_only:
            comment_start = _yaml_comment_start(line)
            if comment_start == -1:
                continue
            haystack = line[comment_start:]
        if any(pattern.search(haystack) for pattern in patterns):
            hits.append(f"{path}:{number}: {line.strip()}")
    return hits


def _scan_python_comments(path: Path, patterns: Iterable[Pattern[str]]) -> List[str]:
    """Return "path:line: text" for every '#' comment token matching any pattern.

    Uses the tokenizer rather than the YAML comment-boundary scanner: Python's
    comment and string grammar (triple-quoted strings, same-line trailing
    comments with no preceding whitespace) doesn't match YAML's, so reusing
    that scanner both misses and mis-flags real cases.
    """
    hits = []
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            if any(pattern.search(token.string) for pattern in patterns):
                number = token.start[0]
                line = lines[number - 1] if number - 1 < len(lines) else token.string
                hits.append(f"{path}:{number}: {line.strip()}")
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        logger.warning(f"⚠️ Could not tokenize {path} to scan for internal comments: {exc}")
    return hits


def _report(hits: List[str], summary: str, section: str, blocking: bool = True) -> int:
    if not hits:
        logger.info(f"✅ {summary}: no findings")
        return 0
    level = logger.error if blocking else logger.warning
    mark = "❌" if blocking else "⚠️"
    level(f"{mark} {summary}: {len(hits)} finding(s)")
    for hit in hits:
        level(f"  {hit}")
    level(f"  See {CONVENTIONS_DOC}#{section}")
    return len(hits)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command(name="check-placeholders")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@assets_root_option
def check_placeholders_command(files: Sequence[Path], assets_root: str = "assets") -> None:
    """Fail on unsubstituted scaffolding markers in assets."""
    targets = _iter_files(files, assets_root, "**/*")
    hits: List[str] = []
    for path in targets:
        if path.suffix.lower() not in {".yaml", ".yml", ".py", ".toml", ".txt", ".md", ".json"}:
            continue
        hits.extend(_scan_lines(path, PLACEHOLDER_PATTERNS))
    sys.exit(1 if _report(hits, "Template placeholders", "public-repository-hygiene") else 0)


@cli.command(name="check-internal-comments")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@assets_root_option
def check_internal_comments_command(files: Sequence[Path], assets_root: str = "assets") -> None:
    """Fail on internal narration in asset YAML and Python source comments."""
    targets = [p for p in _iter_files(files, assets_root, "**/*") if p.suffix.lower() in {".yaml", ".yml", ".py"}]
    hits: List[str] = []
    for path in targets:
        if path.suffix.lower() == ".py":
            hits.extend(_scan_python_comments(path, INTERNAL_COMMENT_PATTERNS))
        else:
            hits.extend(_scan_lines(path, INTERNAL_COMMENT_PATTERNS, comments_only=True))
    sys.exit(1 if _report(hits, "Internal comments", "public-repository-hygiene") else 0)


def is_requirement_pinned(line: str) -> bool:
    """Return whether a requirement uses an exact version or immutable git SHA."""
    requirement = line.strip()
    if requirement.startswith(("-e ", "--editable ")):
        requirement = requirement.split(maxsplit=1)[1]
    return bool(EXACT_VERSION_RE.fullmatch(requirement) or GIT_SHA_RE.search(requirement))


def find_unpinned_requirements(path: Path) -> List[str]:
    """Return "path:line: text" for every requirement without an immutable pin."""
    unpinned = []
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(NON_REQUIREMENT_PREFIXES):
            continue
        if not is_requirement_pinned(line):
            unpinned.append(f"{path}:{number}: {line}")
    return unpinned


def find_unpinned_pyproject_dependencies(path: Path) -> List[str]:
    """Return unpinned entries from ``project.dependencies`` in a pyproject."""
    with path.open("rb") as f:
        dependencies = tomllib.load(f).get("project", {}).get("dependencies", [])
    return [f"{path}: project.dependencies: {dep}" for dep in dependencies if not is_requirement_pinned(dep)]


def find_canonical_name_mismatch(path: Path, assets_root: str = "assets") -> List[str]:
    """Return a finding when metadata canonicalName differs from its asset path."""
    try:
        relative = path.parent.relative_to(Path(assets_root))
    except ValueError:
        return []
    if len(relative.parts) != 3 or relative.parts[1] in {"base", "engines"}:
        return []

    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = metadata.get("com", {}).get("amd", {}).get("aim", {}).get("model", {}).get("canonicalName")
    expected = f"{relative.parts[1]}/{relative.parts[2]}"
    if actual != expected:
        return [f"{path}: canonicalName is {actual!r}; expected {expected!r}"]
    return []


@cli.command(name="check-canonical-name")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@assets_root_option
def check_canonical_name_command(files: Sequence[Path], assets_root: str = "assets") -> None:
    """Fail when metadata canonicalName does not match its asset path."""
    targets = [p for p in _iter_files(files, assets_root, "*/*/*/metadata.yaml") if p.name == "metadata.yaml"]
    hits = [hit for path in targets for hit in find_canonical_name_mismatch(path, assets_root)]
    sys.exit(1 if _report(hits, "Canonical names", "enforced-automatically") else 0)


@cli.command(name="check-requirements-pinned")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@assets_root_option
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero on findings. Off by default while the catalog is being pinned.",
)
def check_requirements_pinned_command(files: Sequence[Path], assets_root: str = "assets", strict: bool = False) -> None:
    """Report dependencies in specialized AIM images that are not pinned to an exact version."""
    targets = [
        p for p in _iter_files(files, assets_root, "**/image/*") if p.name in {"requirements.txt", "pyproject.toml"}
    ]
    hits: List[str] = []
    for path in targets:
        if path.name == "pyproject.toml":
            hits.extend(find_unpinned_pyproject_dependencies(path))
        else:
            hits.extend(find_unpinned_requirements(path))
    count = _report(hits, "Unpinned requirements", "dependency-pinning", blocking=strict)
    sys.exit(1 if (strict and count) else 0)


# Secret key names that must never appear in profile env_vars. env_vars are
# applied as hard overrides (os.environ[key] = value) before the service
# launches, so setting HF_TOKEN='' clobbers any pod-injected secret and
# crash-loops the service with a misleading error.
SECRET_ENV_VAR_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"\bHF_TOKEN\b"),
    re.compile(r"\b\w+_TOKEN\b"),
    re.compile(r"\b\w+_API_KEY\b"),
    re.compile(r"\b\w+_SECRET\b"),
)

# A ROCm torch canary line is unsafe once it prints without also calling
# sys.exit() — see check_rocm_canary_command for the line-matching logic.
_ROCM_SYSEXIT_RE = re.compile(r"sys\.exit")


@cli.command(name="check-profile-env-vars-no-secrets")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@assets_root_option
def check_profile_env_vars_no_secrets_command(files: Sequence[Path], assets_root: str = "assets") -> None:
    """Fail when a profile env_vars block contains a secret key name."""
    targets = [
        p for p in _iter_files(files, assets_root, "**/profiles/*.yaml") if p.suffix.lower() in {".yaml", ".yml"}
    ]
    hits: List[str] = []
    for path in targets:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        env_vars = doc.get("env_vars") or {}
        for key in env_vars:
            if any(pat.search(key) for pat in SECRET_ENV_VAR_PATTERNS):
                hits.append(
                    f"{path}: env_vars.{key} — secret key names belong in metadata.yaml hfToken, not profile env_vars"
                )
    sys.exit(1 if _report(hits, "Secret keys in profile env_vars", "specialized-aims") else 0)


@cli.command(name="check-rocm-canary")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@assets_root_option
def check_rocm_canary_command(files: Sequence[Path], assets_root: str = "assets") -> None:
    """Fail when a Dockerfile ROCm torch canary uses bare print without sys.exit()."""
    targets = [p for p in _iter_files(files, assets_root, "**/image/Dockerfile") if p.name == "Dockerfile"]
    hits: List[str] = []
    for path in targets:
        content = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(content.splitlines(), start=1):
            if "torch" in line and "hip" in line.lower() and "print" in line:
                if not _ROCM_SYSEXIT_RE.search(line):
                    hits.append(f"{path}:{number}: {line.strip()}")
    sys.exit(1 if _report(hits, "ROCm canary missing sys.exit()", "dockerfile") else 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
