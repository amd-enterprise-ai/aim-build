# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Score two Boltz-2 runs with OpenStructure.

The Boltz-2 image does not contain OpenStructure. This runs
``ost compare-structures`` from the OpenStructure container and writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

IMAGE = "registry.scicore.unibas.ch/schwede/openstructure@sha256:9f2669ffde16c45939fc6e8e6dd4f907697301c081edf3d97d464c034a421d19"

# Pairs are (model arm, reference arm, row name). "pdb" is the deposited structure.
PAIRS = (
    ("on", "off", "on_vs_off"),
    ("on", "pdb", "on_vs_pdb"),
    ("off", "pdb", "off_vs_pdb"),
)

COLUMNS = (
    "lddt",
    "bb_lddt",
    "tm_score",
    "rmsd",
    "oligo_gdtts",
    "oligo_gdtha",
    "qs_global",
    "qs_best",
    "dockq",
    "fnat",
)


def structure_map(directory: Path) -> dict[str, Path]:
    """``<dir>/<label>.cif``, or ``<dir>/structures/<label>.cif`` when the top is empty."""
    files = sorted(path for path in directory.glob("*.cif") if path.is_file())
    if not files:
        nested = directory / "structures"
        if nested.is_dir():
            files = sorted(path for path in nested.glob("*.cif") if path.is_file())
    return {path.stem: path for path in files}


def _pick(data: dict, name: str):
    if name in data and data[name] is not None:
        return data[name]
    rigid = data.get("rigid_scores") or {}
    if isinstance(rigid, dict) and name in rigid and rigid[name] is not None:
        return rigid[name]
    return ""


def _compare(image: str, work: Path, model: Path, reference: Path, out_json: Path) -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "ost",
        "-v",
        f"{work}:/work",
        image,
        "compare-structures",
        "-m",
        f"/work/{model.relative_to(work)}",
        "-r",
        f"/work/{reference.relative_to(work)}",
        "-o",
        f"/work/{out_json.relative_to(work)}",
        "-mf",
        "mmcif",
        "-rf",
        "mmcif",
        "--lddt",
        "--bb-lddt",
        "--tm-score",
        "--rigid-scores",
        "--qs-score",
        "--dockq",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return {"error": detail[-800:]}
    return json.loads(out_json.read_text())


def score(on_dir: Path, off_dir: Path, pdb_dir: Path, out_dir: Path, image: str = IMAGE) -> list[dict]:
    """Copy the mmCIF files into ``out_dir`` and score on vs off, on vs PDB, off vs PDB.

    Docker does not follow a symlink to a file outside the bind mount, so the
    copies are real files.
    """
    arms = {
        "on": structure_map(on_dir),
        "off": structure_map(off_dir),
        "pdb": structure_map(pdb_dir),
    }
    labels = sorted(set(arms["on"]) | set(arms["off"]))
    if not labels:
        raise SystemExit(f"no mmCIF files in {on_dir} or {off_dir}")

    stage = out_dir / "stage"
    json_dir = out_dir / "ost_json"
    staged: dict[tuple[str, str], Path] = {}
    for arm, found in arms.items():
        for label, src in found.items():
            dest = stage / arm / f"{label}.cif"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            staged[(arm, label)] = dest

    rows = []
    for label in labels:
        for model_arm, ref_arm, pair in PAIRS:
            row = {"pdb": label, "pair": pair}
            model = staged.get((model_arm, label))
            reference = staged.get((ref_arm, label))
            if model is None or reference is None:
                row["error"] = "missing structure"
                print(f"{label} {pair}: missing", flush=True)
                rows.append(row)
                continue
            data = _compare(image, out_dir, model, reference, json_dir / f"{label}_{pair}.json")
            if "error" in data and "lddt" not in data:
                row["error"] = data["error"]
                print(f"{label} {pair}: FAIL {row['error'][:200]}", flush=True)
            else:
                for name in COLUMNS:
                    row[name] = _pick(data, name)
                row["error"] = ""
                print(
                    f"{label} {pair}: lddt={row['lddt']} tm={row['tm_score']} rmsd={row['rmsd']}",
                    flush=True,
                )
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ost_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb", "pair", *COLUMNS, "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Boltz-2 on vs off with OpenStructure.")
    parser.add_argument("--on", type=Path, required=True, help="mmCIF files from BOLTZ2_OPTIMIZATIONS=on")
    parser.add_argument("--off", type=Path, required=True, help="mmCIF files from BOLTZ2_OPTIMIZATIONS=off")
    parser.add_argument("--pdb", type=Path, required=True, help="deposited structures, <label>.cif")
    parser.add_argument("--out", type=Path, required=True, help="directory for the CSV and JSON")
    parser.add_argument("--image", default=IMAGE, help="OpenStructure container (default: the pinned digest)")
    args = parser.parse_args()
    rows = score(args.on, args.off, args.pdb, args.out, args.image)
    failed = [row for row in rows if row.get("error")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
