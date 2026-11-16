# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""summary.json synthesis and atomic publishing of results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import fbc_summary as summary
import pytest


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("0 hours, 1 minutes, 5 seconds", 65),
        ("2 hours, 0 minutes, 0 seconds", 7200),
        ("1 hours, 59 minutes, 59 seconds", 7199),
        ("", None),
        ("garbage", None),
    ],
)
def test_parse_time_text(text: str, seconds) -> None:
    assert summary.parse_time_text(text) == seconds


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _design_dir(tmp_path: Path) -> Path:
    d = tmp_path / "design"
    _write_csv(
        d / "trajectory_stats.csv",
        ["Design", "TrajectoryTime"],
        [["a", "0 hours, 2 minutes, 0 seconds"], ["b", "0 hours, 4 minutes, 0 seconds"]],
    )
    _write_csv(d / "mpnn_design_stats.csv", ["Design"], [["a_mpnn1"], ["a_mpnn2"], ["b_mpnn1"]])
    _write_csv(d / "rejected_mpnn_full_stats.csv", ["Design", "Sequence"], [["a_mpnn3", "MKT"], ["b_mpnn2", "MKV"]])
    _write_csv(d / "failure_csv.csv", ["Trajectory_Clashes", "Trajectory_Contacts", "pLDDT"], [[3, 2, 9]])
    (d / "Accepted/Ranked").mkdir(parents=True)
    (d / "Accepted/a_mpnn1.pdb").write_text("x")
    (d / "Accepted/Ranked/1_a_mpnn1.pdb").write_text("x")
    (d / "Trajectory/Relaxed").mkdir(parents=True)
    (d / "Trajectory/Relaxed/a.pdb").write_text("x")
    return d


def _summary(d: Path, **kw):
    base = dict(
        design_dir=d,
        settings={"num_designs": 2},
        status="running",
        finished=False,
        exit_code=None,
        started_at="2026-10-14T00:00:00Z",
        finished_at=None,
        wall_time_s=12.345,
        first_trajectory_s=7.89,
        accelerators=1,
        resumed=False,
    )
    base.update(kw)
    return summary.build_summary(**base)


def test_summary_is_synthesized_from_the_upstream_csvs(tmp_path: Path) -> None:
    s = _summary(_design_dir(tmp_path))
    assert s["trajectories"] == 2
    assert s["per_trajectory_times_s"] == [120, 240] and s["mean_trajectory_time_s"] == 180.0
    assert s["trajectories_rejected_early"] == 5  # only the Trajectory_* counters
    assert (s["mpnn_designs_scored"], s["mpnn_designs_rejected"]) == (3, 2)
    assert (s["accepted_designs"], s["ranked_designs"], s["trajectories_relaxed"], s["requested_designs"]) == (
        1,
        1,
        1,
        2,
    )
    assert s["wall_time_s"] == 12.3 and s["time_to_first_trajectory_s"] == 7.9
    json.dumps(s)  # serializable


def test_summary_of_an_empty_work_dir_is_all_zero(tmp_path: Path) -> None:
    s = _summary(tmp_path / "nothing", first_trajectory_s=None)
    assert s["trajectories"] == 0 and s["accepted_designs"] == 0 and s["per_trajectory_times_s"] == []
    assert s["mean_trajectory_time_s"] is None and s["time_to_first_trajectory_s"] is None


def test_torn_csv_does_not_raise(tmp_path: Path) -> None:
    d = _design_dir(tmp_path)
    (d / "trajectory_stats.csv").write_bytes(b'Design,TrajectoryTime\n\xff\xfe"unterminated')
    assert _summary(d)["trajectories"] >= 0


def test_publish_copies_results_atomically_and_mirrors_removals(tmp_path: Path) -> None:
    d = _design_dir(tmp_path)
    out = tmp_path / "out"
    summary.publish_results(d, out)
    assert (out / "trajectory_stats.csv").is_file() and (out / "failure_csv.csv").is_file()
    assert (out / "Accepted/a_mpnn1.pdb").is_file() and (out / "Accepted/Ranked/1_a_mpnn1.pdb").is_file()
    assert not [p for p in out.rglob("*") if p.name.endswith(".tmp")], "no temporary names are left behind"
    (d / "Accepted/Ranked/1_a_mpnn1.pdb").unlink()  # upstream clears Ranked when it re-ranks
    summary.publish_results(d, out)
    assert not (out / "Accepted/Ranked/1_a_mpnn1.pdb").exists()


def test_unchanged_files_are_not_recopied(tmp_path: Path) -> None:
    src, dst = tmp_path / "a", tmp_path / "out" / "a"
    src.write_text("hello")
    assert summary.atomic_copy(src, dst) is True
    assert summary.atomic_copy(src, dst) is False


def test_failed_copy_cleans_up_and_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst = tmp_path / "a", tmp_path / "out" / "a"
    src.write_text("hello")

    def broken(_src, tmp):
        Path(tmp).write_text("partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(summary.shutil, "copyfile", broken)
    with pytest.raises(OSError):
        summary.atomic_copy(src, dst)
    assert not dst.exists()
    assert not list(dst.parent.glob(".*.tmp")), "the temporary file is removed before the error propagates"


def test_publish_results_propagates_a_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _design_dir(tmp_path)

    def broken(_src, _tmp):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(summary.shutil, "copyfile", broken)
    with pytest.raises(OSError):
        summary.publish_results(d, tmp_path / "out")


def test_write_summary_is_atomic(tmp_path: Path) -> None:
    summary.write_summary(tmp_path, {"status": "x"})
    assert json.loads((tmp_path / "summary.json").read_text()) == {"status": "x"}
    assert not list(tmp_path.glob(".*.tmp"))
