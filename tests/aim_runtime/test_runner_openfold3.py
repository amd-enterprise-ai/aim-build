# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the OpenFold3 runner inline-MSA shim and output parser.

These tests do **not** require torch or OpenFold3 — they exercise the pure
stdlib helpers ``_materialize_inline_msas`` / ``_write_msa_file`` and the
``_parse_output_dir`` output-directory parser, loaded directly from the asset
path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

OF3_RUNNER_PATH = Path(__file__).resolve().parents[2] / "assets/instinct/openfold/openfold3/image/src/runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_of3_runner_under_test", OF3_RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_runner_file_exists():
    assert OF3_RUNNER_PATH.is_file()


def test_main_msa_str(tmp_path):
    msa_dir = tmp_path / "msas"
    chain = {"sequence": "AAA", "main_msa": ">q\nAAA\n"}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert "main_msa" not in chain
    paths = chain["main_msa_file_paths"]
    assert len(paths) == 1
    assert Path(paths[0]).read_text() == ">q\nAAA\n"


def test_main_msa_list_is_concatenated_in_order(tmp_path):
    """A list[str] is concatenated into ONE recognized file, in order.

    OF3 only parses MSA files whose basename is a recognized key, so multiple
    arbitrarily-named files cannot survive its filter; the wrapper merges the
    alignments into a single recognized file instead (OF3 concatenates a
    chain's MSA files anyway).
    """
    msa_dir = tmp_path / "msas"
    chain = {"main_msa": [">a\nA\n", ">b\nB\n", ">c\nC\n"]}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    paths = chain["main_msa_file_paths"]
    assert len(paths) == 1
    assert Path(paths[0]).read_text() == ">a\nA\n>b\nB\n>c\nC\n"


def test_main_msa_uses_recognized_basename(tmp_path):
    """Main MSA is written with an OF3-recognized basename (else OF3 skips it)."""
    msa_dir = tmp_path / "msas"
    chain = {"main_msa": ">q\nAAA\n"}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert Path(chain["main_msa_file_paths"][0]).name == "colabfold_main.a3m"


def test_paired_msa_uses_recognized_basename(tmp_path):
    """Paired MSA is written with an OF3-recognized basename."""
    msa_dir = tmp_path / "msas"
    chain = {"paired_msa": ">q\nAAA\n"}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert Path(chain["paired_msa_file_paths"][0]).name == "colabfold_paired.a3m"


def test_each_chain_gets_its_own_subdir(tmp_path):
    """Each chain's MSAs live in a distinct directory.

    OF3 derives a chain's MSA representative id from the file's PARENT directory
    name, so a shared directory would collapse multiple chains onto one rep id.
    """
    msa_dir = tmp_path / "msas"
    queries = {"q": {"chains": [{"main_msa": ">q\nA\n"}, {"main_msa": ">q\nB\n"}]}}

    runner._materialize_inline_msas(queries, msa_dir)

    d0 = Path(queries["q"]["chains"][0]["main_msa_file_paths"][0]).parent
    d1 = Path(queries["q"]["chains"][1]["main_msa_file_paths"][0]).parent
    assert d0 != d1


def test_paired_msa(tmp_path):
    msa_dir = tmp_path / "msas"
    chain = {"paired_msa": ">q\nAAA\n"}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert "paired_msa" not in chain
    assert len(chain["paired_msa_file_paths"]) == 1
    assert Path(chain["paired_msa_file_paths"][0]).read_text() == ">q\nAAA\n"


def test_explicit_paths_win(tmp_path):
    msa_dir = tmp_path / "msas"
    chain = {
        "main_msa": ">q\nAAA\n",
        "main_msa_file_paths": ["/server/side/path.a3m"],
    }
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert "main_msa" not in chain
    assert chain["main_msa_file_paths"] == ["/server/side/path.a3m"]
    assert not msa_dir.exists()


def test_no_op_when_no_inline(tmp_path):
    msa_dir = tmp_path / "msas"
    chain = {"sequence": "AAA"}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert chain == {"sequence": "AAA"}
    assert not msa_dir.exists()


def test_empty_content_is_no_op_and_removes_key(tmp_path):
    msa_dir = tmp_path / "msas"
    chain = {"main_msa": "", "paired_msa": []}
    queries = {"q": {"chains": [chain]}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert "main_msa" not in chain
    assert "paired_msa" not in chain
    assert "main_msa_file_paths" not in chain
    assert "paired_msa_file_paths" not in chain
    assert not msa_dir.exists()


def test_q_name_path_traversal_is_contained(tmp_path):
    msa_dir = tmp_path / "msas"
    queries = {
        "/abs/path/evil": {"chains": [{"main_msa": ">q\nAAA\n"}]},
        "../../escape": {"chains": [{"main_msa": ">q\nBBB\n"}]},
    }

    runner._materialize_inline_msas(queries, msa_dir)

    written = [p for p in msa_dir.rglob("*.a3m")]
    assert len(written) == 2
    for p in written:
        resolved = p.resolve()
        assert resolved.is_relative_to(msa_dir.resolve())


def test_colliding_query_basenames_do_not_overwrite(tmp_path):
    msa_dir = tmp_path / "msas"
    queries = {
        "a/x": {"chains": [{"main_msa": ">A\nAAA\n"}]},
        "x": {"chains": [{"main_msa": ">B\nBBB\n"}]},
    }

    runner._materialize_inline_msas(queries, msa_dir)

    p1 = queries["a/x"]["chains"][0]["main_msa_file_paths"]
    p2 = queries["x"]["chains"][0]["main_msa_file_paths"]
    assert p1 != p2
    assert Path(p1[0]).read_text() == ">A\nAAA\n"
    assert Path(p2[0]).read_text() == ">B\nBBB\n"
    assert len(list(msa_dir.rglob("*.a3m"))) == 2


def test_non_dict_chain_is_skipped(tmp_path):
    msa_dir = tmp_path / "msas"
    queries = {"q": {"chains": ["not_a_dict", None, {"main_msa": ">q\nAAA\n"}]}}

    runner._materialize_inline_msas(queries, msa_dir)

    good_chain = queries["q"]["chains"][2]
    assert len(good_chain["main_msa_file_paths"]) == 1


def test_non_list_chains_is_noop(tmp_path):
    msa_dir = tmp_path / "msas"
    queries = {"q": {"chains": {"0": {"main_msa": ">q\nAAA\n"}}}}

    runner._materialize_inline_msas(queries, msa_dir)

    assert not msa_dir.exists()


@pytest.mark.parametrize("bad_content", [{"a": "b"}, [">a\nA\n", 42]])
def test_non_str_content_raises(tmp_path, bad_content):
    msa_dir = tmp_path / "msas"
    queries = {"q": {"chains": [{"main_msa": bad_content}]}}

    with pytest.raises(TypeError):
        runner._materialize_inline_msas(queries, msa_dir)


def test_multiple_queries_unique_stems(tmp_path):
    msa_dir = tmp_path / "msas"
    queries = {
        "q1": {"chains": [{"main_msa": ">1\nA\n"}, {"main_msa": ">2\nB\n"}]},
        "q2": {"chains": [{"paired_msa": ">3\nC\n"}]},
    }

    runner._materialize_inline_msas(queries, msa_dir)

    all_paths = []
    for query in queries.values():
        for chain in query["chains"]:
            all_paths.extend(chain.get("main_msa_file_paths", []))
            all_paths.extend(chain.get("paired_msa_file_paths", []))

    assert len(all_paths) == 3
    assert len(set(all_paths)) == 3
    assert all(Path(p).is_file() for p in all_paths)


def test_list_with_empty_entry_raises_value_error(tmp_path):
    """A truthy list containing an empty string must raise ValueError, not write zero-byte files."""
    msa_dir = tmp_path / "msas"
    queries = {"q": {"chains": [{"main_msa": ["", "x"]}]}}

    with pytest.raises(ValueError, match="inline MSA content is empty"):
        runner._materialize_inline_msas(queries, msa_dir)


def test_whitespace_only_scalar_raises_value_error(tmp_path):
    """A whitespace-only scalar string must raise ValueError, not write a zero-byte file."""
    msa_dir = tmp_path / "msas"
    queries = {"q": {"chains": [{"main_msa": "   "}]}}

    with pytest.raises(ValueError, match="inline MSA content is empty"):
        runner._materialize_inline_msas(queries, msa_dir)


# --------------------------------------------------------------------------- #
# find_request_conflicts  (rejected as HTTP 400 by the service's model_validator)
# --------------------------------------------------------------------------- #

_INLINE_QUERIES = {"q": {"chains": [{"sequence": "AAA", "main_msa": ">q\nAAA\n"}]}}
_PLAIN_QUERIES = {"q": {"chains": [{"sequence": "AAA"}]}}


def _conflicts(queries, *, use_msa_server, use_templates, num_model_seeds, seeds_explicit):
    return runner.find_request_conflicts(
        queries,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
        num_model_seeds=num_model_seeds,
        seeds_explicit=seeds_explicit,
    )


def test_conflict_inline_msa_with_server():
    c = _conflicts(
        _INLINE_QUERIES, use_msa_server=True, use_templates=False, num_model_seeds=None, seeds_explicit=False
    )
    assert any("cannot be combined with use_msa_server" in m for m in c)


def test_no_conflict_inline_msa_without_server():
    c = _conflicts(
        _INLINE_QUERIES, use_msa_server=False, use_templates=False, num_model_seeds=None, seeds_explicit=False
    )
    assert c == []


def test_no_conflict_server_without_inline_msa():
    c = _conflicts(_PLAIN_QUERIES, use_msa_server=True, use_templates=False, num_model_seeds=None, seeds_explicit=False)
    assert c == []


def test_empty_string_msa_is_not_inline():
    """main_msa="" is a no-op (no MSA), so it must NOT count as an inline MSA conflict."""
    queries = {"q": {"chains": [{"main_msa": ""}]}}
    c = _conflicts(queries, use_msa_server=True, use_templates=False, num_model_seeds=None, seeds_explicit=False)
    assert c == []


def test_conflict_seeds_and_num_model_seeds():
    c = _conflicts(_PLAIN_QUERIES, use_msa_server=False, use_templates=False, num_model_seeds=2, seeds_explicit=True)
    assert any("seeds and num_model_seeds cannot both be set" in m for m in c)


def test_no_conflict_num_model_seeds_without_explicit_seeds():
    c = _conflicts(_PLAIN_QUERIES, use_msa_server=False, use_templates=False, num_model_seeds=2, seeds_explicit=False)
    assert c == []


def test_conflict_templates_without_server():
    c = _conflicts(_PLAIN_QUERIES, use_msa_server=False, use_templates=True, num_model_seeds=None, seeds_explicit=False)
    assert any("use_templates=True requires use_msa_server=True" in m for m in c)


def test_no_conflict_templates_with_server():
    c = _conflicts(_PLAIN_QUERIES, use_msa_server=True, use_templates=True, num_model_seeds=None, seeds_explicit=False)
    assert c == []


def _make_output_dir(tmp_path, with_timing=True, with_atom=True):
    """Build a minimal OF3 output tree; return (output_dir, sample_id)."""
    out = tmp_path / "output"
    query_id = "q0"
    sample = f"{query_id}_seed_42_sample_0"
    seed_dir = out / query_id / "seed_42"
    seed_dir.mkdir(parents=True)

    (seed_dir / f"{sample}_model.pdb").write_text("PDB\n")
    (seed_dir / f"{sample}_confidences_aggregated.json").write_text(json.dumps({"ranking_score": 0.9}))

    if with_atom:
        (seed_dir / f"{sample}_confidences.json").write_text(
            json.dumps({"plddt": [1.0, 2.0], "pae": [[0, 1], [1, 0]], "pde": [[0, 0.5], [0.5, 0]]})
        )
    if with_timing:
        (seed_dir / "timing.json").write_text(json.dumps({"runtime_s": 12.5}))

    return out, sample


def test_parse_output_dir_atom_off_by_default(tmp_path):
    out, sample = _make_output_dir(tmp_path)

    result = runner._parse_output_dir(out)

    assert "atom_confidence" not in result
    assert len(result["structures"]) == 1
    assert result["confidence"][sample] == {"ranking_score": 0.9}
    assert result["timing"] == {sample: {"runtime_s": 12.5}}


def test_parse_output_dir_atom_on(tmp_path):
    out, sample = _make_output_dir(tmp_path)

    result = runner._parse_output_dir(out, include_atom_confidences=True)

    atom = result["atom_confidence"][sample]
    assert atom["plddt"] == [1.0, 2.0]
    assert atom["pae"] == [[0, 1], [1, 0]]
    assert atom["pde"] == [[0, 0.5], [0.5, 0]]


@pytest.mark.parametrize("flag", [False, True])
def test_parse_output_dir_timing_surfaces(tmp_path, flag):
    out, sample = _make_output_dir(tmp_path)

    result = runner._parse_output_dir(out, include_atom_confidences=flag)

    assert result["timing"] == {sample: {"runtime_s": 12.5}}


def test_parse_output_dir_missing_atom_file_is_graceful(tmp_path):
    out, _ = _make_output_dir(tmp_path, with_atom=False)

    result = runner._parse_output_dir(out, include_atom_confidences=True)

    assert result["atom_confidence"] == {}


def test_parse_output_dir_missing_timing_is_graceful(tmp_path):
    out, _ = _make_output_dir(tmp_path, with_timing=False)

    result = runner._parse_output_dir(out)

    assert result["timing"] == {}


def test_parse_output_dir_multi_sample(tmp_path):
    out, first = _make_output_dir(tmp_path)
    seed_dir = out / "q0" / "seed_42"
    second = "q0_seed_42_sample_1"
    (seed_dir / f"{second}_model.pdb").write_text("PDB\n")
    (seed_dir / f"{second}_confidences_aggregated.json").write_text(json.dumps({"ranking_score": 0.8}))
    (seed_dir / f"{second}_confidences.json").write_text(json.dumps({"plddt": [3.0]}))

    result = runner._parse_output_dir(out, include_atom_confidences=True)

    assert len(result["structures"]) == 2
    assert set(result["atom_confidence"]) == {first, second}
    assert result["atom_confidence"][second]["plddt"] == [3.0]
    assert result["timing"] == {first: {"runtime_s": 12.5}, second: {"runtime_s": 12.5}}
