# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Checks that run before the design run starts: input, hardware, resume guard.

Pure standard library. Every failure raises :class:`PreflightError`, which the
entrypoint reports and maps to an exit status after the log is already open.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping

from fbc_settings import Settings

TARGET_NAME = "target.pdb"
MIN_ACCELERATORS = 1

# AF2 parameter files the multimer design needs; used to decide whether an
# operator-supplied AIM_CACHE_PATH holds usable weights.
_AF2_MARKER = Path("params") / "params_model_1_multimer_v3.npz"


class PreflightError(RuntimeError):
    """A precondition failed; ``status`` is the process exit status to use."""

    def __init__(self, message: str, status: int = 2) -> None:
        super().__init__(message)
        self.status = status


def target_chains(pdb_path: Path) -> set[str]:
    """Chain ids that carry at least one ATOM record with a C-alpha."""
    chains: set[str] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and len(line) > 21:
                chains.add(line[21])
    return chains


def target_residues(pdb_path: Path) -> dict[str, set[int]]:
    """Residue numbers per chain id for ATOM records with a C-alpha."""
    residues: dict[str, set[int]] = {}
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and len(line) > 26:
                try:
                    number = int(line[22:26])
                except ValueError:
                    continue
                residues.setdefault(line[21], set()).add(number)
    return residues


def hotspot_report(pdb_path: Path, settings: Settings) -> str | None:
    """One log line saying how many hotspot residue numbers are present in the target.

    Informational only: it never fails the run. A chain letter alone is not a residue
    and is not counted; a bare number or range is looked up in the target chains.
    """
    wanted: list[tuple[str, int]] = []  # (chain or "", residue number)
    for item in filter(None, settings.hotspots.split(",")):
        chain = item[0] if item[0].isalpha() else ""
        body = item[1:] if chain else item
        if not body:
            continue
        low, _, high = body.partition("-")
        first, last = int(low), int(high or low)
        wanted.extend((chain, number) for number in range(first, last + 1))
    if not wanted:
        return None
    residues = target_residues(pdb_path)
    target_ids = set(settings.target_chains.split(","))
    missing = [
        f"{chain}{number}"
        for chain, number in wanted
        if not any(number in residues.get(c, set()) for c in ([chain] if chain else sorted(target_ids)))
    ]
    found = len(wanted) - len(missing)
    line = f"hotspots: {found} of {len(wanted)} residue numbers found in {TARGET_NAME}"
    if missing:
        shown = ", ".join(missing[:10]) + (", ..." if len(missing) > 10 else "")
        line += f"; not found: {shown}"
    return line


def validate_target(input_dir: Path, settings: Settings) -> Path:
    """Return the target PDB path or raise when it is missing or unusable."""
    if not input_dir.is_dir():
        raise PreflightError(f"AIM_INPUT_PATH is not a directory: {input_dir}")
    target = input_dir / TARGET_NAME
    if not target.is_file():
        raise PreflightError(f"{TARGET_NAME} not found in AIM_INPUT_PATH ({input_dir})")
    if target.stat().st_size == 0:
        raise PreflightError(f"{target} is empty")
    present = target_chains(target)
    if not present:
        raise PreflightError(f"{target} has no protein ATOM records (no C-alpha atoms found)")
    wanted = set(settings.target_chains.split(","))
    missing = sorted(wanted - present)
    if missing:
        raise PreflightError(
            f"target chain(s) {','.join(missing)} not in {TARGET_NAME} (chains present: {','.join(sorted(present))})"
        )
    for item in filter(None, settings.hotspots.split(",")):
        chain = item[0]
        if chain.isalpha() and chain not in present:
            raise PreflightError(
                f"hotspot {item!r} names chain {chain}, not in {TARGET_NAME} (chains present: {','.join(sorted(present))})"
            )
    return target


def detect_accelerators(env: Mapping[str, str], dev_root: Path = Path("/dev")) -> int:
    """Number of AMD GPUs: render nodes under /dev/dri when /dev/kfd exists.

    ``AIM_ACCELERATOR_COUNT`` overrides detection. The platform hands
    the container only the devices it was given, so counting nodes is the count.
    """
    override = env.get("AIM_ACCELERATOR_COUNT", "").strip()
    if override:
        try:
            count = int(override)
        except ValueError:
            raise PreflightError(f"AIM_ACCELERATOR_COUNT must be an integer, got {override!r}") from None
        if count < 0:
            raise PreflightError(f"AIM_ACCELERATOR_COUNT must not be negative, got {count}")
        return count
    if not (dev_root / "kfd").exists():
        return 0
    return len(list((dev_root / "dri").glob("renderD*")))


def require_accelerators(env: Mapping[str, str], dev_root: Path = Path("/dev")) -> int:
    count = detect_accelerators(env, dev_root)
    if count < MIN_ACCELERATORS:
        raise PreflightError(
            f"FreeBindCraft needs {MIN_ACCELERATORS} AMD GPU, found {count}. Pass --device=/dev/kfd --device=/dev/dri "
            "(or request a GPU from the scheduler); set AIM_ACCELERATOR_COUNT only to override detection.",
            status=3,
        )
    return count


def resolve_af_params_dir(env: Mapping[str, str], image_dir: Path = Path("/app")) -> str:
    """Directory that contains ``params/`` with the AlphaFold2 weights.

    The image does not ship the weights. ``AIM_CACHE_PATH`` must point to a
    directory holding ``params/params_model_1_multimer_v3.npz`` (mount it
    read-only). A copy under ``image_dir`` (a derived image that bakes them) is
    still honoured. Weights never come from ``AIM_INPUT_PATH``.
    """
    cache = env.get("AIM_CACHE_PATH", "").strip()
    if cache and (Path(cache) / _AF2_MARKER).is_file():
        return cache
    if (image_dir / _AF2_MARKER).is_file():
        return str(image_dir)
    where = f"AIM_CACHE_PATH={cache}" if cache else "AIM_CACHE_PATH (not set)"
    raise PreflightError(
        f"AlphaFold2 parameters not found: {where} must contain {_AF2_MARKER.as_posix()}. "
        "The image does not bundle them: mount a directory that holds params/params_model_*_multimer_v3.npz "
        "(from https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar, CC BY 4.0) and set AIM_CACHE_PATH to it."
    )


def input_fingerprint(target: Path, settings: Settings) -> str:
    """Identity of what the work directory was started for (resume guard)."""
    digest = hashlib.sha256()
    digest.update(target.read_bytes())
    digest.update(re.sub(r"\s+", "", settings.target_chains).encode())
    digest.update(b"|" + settings.hotspots.encode())
    return digest.hexdigest()


def check_resume_guard(work_dir: Path, fingerprint: str) -> bool:
    """Record or compare the fingerprint; return True when resuming prior work.

    A work directory that was started for a different target or hotspot set must
    not be continued: its trajectories would be mixed into the new run's results.
    """
    marker = work_dir / "input.sha256"
    resumed = False
    if marker.exists():
        previous = marker.read_text(encoding="utf-8").strip()
        if previous != fingerprint:
            raise PreflightError(
                f"{work_dir} holds work for a different target/chains/hotspots. "
                "Use an empty AIM_OUTPUT_PATH for a new run, or restore the original input to resume."
            )
        resumed = True
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_name(marker.name + ".tmp")
        tmp.write_text(fingerprint + "\n", encoding="utf-8")
        tmp.replace(marker)
    return resumed
