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

import contextlib
import importlib.util
import json
import subprocess
import sys
import types
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


# --- _download_lock ---------------------------------------------------------


def test_download_lock_creates_a_lock_file(tmp_path):
    cache = tmp_path / "of3"
    with runner._download_lock(cache):
        pass
    assert (cache / ".parameters.lock").is_file()


def test_download_lock_releases_on_normal_exit(tmp_path):
    """The lock must not outlive the block.

    Nothing unlocks explicitly — closing the file is what releases it — so this
    is the guard against a refactor that keeps the descriptor alive.
    """
    with runner._download_lock(tmp_path):
        pass

    assert _lock_is_free(tmp_path)


def _fail_while_holding_lock(cache) -> None:
    """Raise from inside the lock, as a failing download would."""
    with runner._download_lock(cache):
        raise RuntimeError("download blew up")


def test_download_lock_releases_on_exception(tmp_path):
    """A failed download must not wedge every other worker."""
    with pytest.raises(RuntimeError):
        _fail_while_holding_lock(tmp_path)

    # Re-acquiring from another process proves the lock was released, not just
    # that this process could re-enter it (flock is per-fd, so a same-process
    # re-acquire would succeed even on a leaked lock).
    assert _lock_is_free(tmp_path)


def test_download_lock_excludes_another_process(tmp_path):
    with runner._download_lock(tmp_path):
        assert not _lock_is_free(tmp_path)


def test_download_lock_yields_unlocked_when_the_lock_file_cannot_be_created(tmp_path):
    """A cache that cannot hold a lock file — e.g. a read-only mount — must still start.

    Blocked with a non-directory parent rather than a read-only directory: root
    bypasses permission bits, and the in-image test run is root.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    cache = blocker / "cache"

    with runner._download_lock(cache):
        pass

    assert not (cache / ".parameters.lock").exists()


def _lock_is_free(cache) -> bool:
    """Whether a separate process can take the cache's lock right now."""
    script = (
        "import fcntl,sys\n"
        f"f = open({str(cache / '.parameters.lock')!r}, 'w')\n"
        "try:\n"
        "    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    return subprocess.run([sys.executable, "-c", script], timeout=30).returncode == 0


# --- per-request template scratch -------------------------------------------


def test_scratch_dir_lives_under_the_request_work_dir(tmp_path):
    args = runner._scratch_runner_args(tmp_path)

    templates = args["template_preprocessor_settings"]["output_directory"]

    assert templates.is_relative_to(tmp_path)


def test_msa_scratch_is_left_to_openfold3():
    """AIM must not override OF3's MSA workspace.

    Two-request isolation is OpenFold3's job (per-run
    ``msa-{user}-{utc}-{token_hex(4)}`` with ``exist_ok=False``). This only
    locks the AIM contract: scratch args do not set ``msa_computation_settings``.
    """
    assert "msa_computation_settings" not in runner._scratch_runner_args(Path("/work"))


def test_scratch_dirs_are_not_the_shared_of3_tmpdir(tmp_path):
    """OF3 defaults the template tree to /tmp/of3-of-<user>/, shared by every worker."""
    args = runner._scratch_runner_args(tmp_path)

    for settings in args.values():
        for path in settings.values():
            assert not any(part.startswith("of3-of-") for part in path.parts)


def test_two_requests_get_different_scratch_dirs(tmp_path):
    """Isolation is per request, so a later request cannot inherit stale state."""
    first = runner._scratch_runner_args(tmp_path / "req-1")
    second = runner._scratch_runner_args(tmp_path / "req-2")

    assert (
        first["template_preprocessor_settings"]["output_directory"]
        != second["template_preprocessor_settings"]["output_directory"]
    )


def test_scratch_stays_out_of_the_inline_msa_and_output_dirs(tmp_path):
    """work_path/msas holds caller-supplied alignments and work_path/output the returned results."""
    reserved = (tmp_path / "msas", tmp_path / "output")
    args = runner._scratch_runner_args(tmp_path)

    for settings in args.values():
        for path in settings.values():
            for other in reserved:
                assert not path.is_relative_to(other)
                assert path != other


def test_scratch_args_do_not_disturb_the_rocm_args(tmp_path):
    """The scratch keys must merge alongside the tuned settings, not replace them."""
    merged = runner._get_rocm_runner_args()
    merged.update(runner._scratch_runner_args(tmp_path))

    assert "model_update" in merged
    assert set(runner._scratch_runner_args(tmp_path)) == {
        "template_preprocessor_settings",
    }


# --- MSA server identification ----------------------------------------------


def test_requests_to_the_msa_server_identify_this_aim(monkeypatch):
    """OF3 sends a bare "openfold"; the shared public server asks callers to say who they are."""
    monkeypatch.delenv("OPENFOLD3_MSA_USER_AGENT", raising=False)

    agent = runner._msa_server_args()["server_user_agent"]

    assert agent == runner.DEFAULT_MSA_USER_AGENT
    assert "aim" in agent.lower()


def test_the_user_agent_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_MSA_USER_AGENT", "acme/1.0 ops@acme.example")

    assert runner._msa_server_args()["server_user_agent"] == "acme/1.0 ops@acme.example"


def test_an_empty_user_agent_falls_back_rather_than_going_anonymous(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_MSA_USER_AGENT", "")

    assert runner._msa_server_args()["server_user_agent"] == runner.DEFAULT_MSA_USER_AGENT


def test_the_user_agent_is_the_only_msa_override():
    """AIM only overrides the user agent; OpenFold3 owns the MSA scratch path."""
    assert set(runner._msa_server_args()) == {"server_user_agent"}


# --- MSA deadline plumbing ---------------------------------------------------
#
# runner._colabfold_client() imports openfold3.core.data.tools.colabfold_msa_server
# lazily, and whether that import succeeds depends on where the suite runs:
# openfold3 is absent on a dev machine and present in the built image, which
# also runs these tests. So the ambient state cannot be asserted against in
# either direction. The fixture below installs a fake module chain, pinning the
# import to succeed so the real logic (hasattr checks, cause/context walking)
# is what gets exercised.


@pytest.fixture
def install_fake_colabfold_client(monkeypatch):
    """Register a fake ``openfold3.core.data.tools.colabfold_msa_server`` chain.

    Every parent package ``from openfold3.core.data.tools import
    colabfold_msa_server`` touches is registered in ``sys.modules`` (and linked
    via attributes, matching what the real import machinery would set), so the
    import in ``runner._colabfold_client()`` resolves to the fake leaf module
    instead of raising ImportError.
    """

    def _install(*, with_msa_deadline: bool = True, with_msa_server_timeout: bool = True):
        calls: list[float | None] = []

        class MsaServerTimeout(Exception):
            pass

        @contextlib.contextmanager
        def msa_deadline(deadline):
            calls.append(deadline)
            yield

        client = types.ModuleType("openfold3.core.data.tools.colabfold_msa_server")
        if with_msa_deadline:
            client.msa_deadline = msa_deadline
        if with_msa_server_timeout:
            client.MsaServerTimeout = MsaServerTimeout

        of3 = types.ModuleType("openfold3")
        core = types.ModuleType("openfold3.core")
        data = types.ModuleType("openfold3.core.data")
        tools = types.ModuleType("openfold3.core.data.tools")
        of3.core = core
        core.data = data
        data.tools = tools
        tools.colabfold_msa_server = client

        for name, module in (
            ("openfold3", of3),
            ("openfold3.core", core),
            ("openfold3.core.data", data),
            ("openfold3.core.data.tools", tools),
            ("openfold3.core.data.tools.colabfold_msa_server", client),
        ):
            monkeypatch.setitem(sys.modules, name, module)

        return client, calls, MsaServerTimeout

    return _install


def test_msa_deadline_hook_available_requires_the_deadline_context_manager(install_fake_colabfold_client):
    """A hasattr check on only MsaServerTimeout must not pass this."""
    install_fake_colabfold_client(with_msa_deadline=False, with_msa_server_timeout=True)

    assert runner.msa_deadline_hook_available() is False


def test_msa_deadline_hook_available_requires_the_timeout_exception(install_fake_colabfold_client):
    """A hasattr check on only msa_deadline must not pass this."""
    install_fake_colabfold_client(with_msa_deadline=True, with_msa_server_timeout=False)

    assert runner.msa_deadline_hook_available() is False


def test_msa_deadline_hook_available_when_both_symbols_are_present(install_fake_colabfold_client):
    install_fake_colabfold_client(with_msa_deadline=True, with_msa_server_timeout=True)

    assert runner.msa_deadline_hook_available() is True


def test_is_msa_server_timeout_true_for_a_direct_instance(install_fake_colabfold_client):
    _, _, MsaServerTimeout = install_fake_colabfold_client()

    assert runner.is_msa_server_timeout(MsaServerTimeout("timed out")) is True


def test_is_msa_server_timeout_true_through_a_cause_chain(install_fake_colabfold_client):
    """The failure surfaces from inside Lightning's predict loop, wrapped."""
    _, _, MsaServerTimeout = install_fake_colabfold_client()

    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = MsaServerTimeout("timed out")

    assert runner.is_msa_server_timeout(wrapper) is True


def test_is_msa_server_timeout_true_through_a_context_chain(install_fake_colabfold_client):
    """An implicit `raise` inside an `except` block chains via __context__, not __cause__."""
    _, _, MsaServerTimeout = install_fake_colabfold_client()

    wrapper = RuntimeError("wrapped")
    wrapper.__context__ = MsaServerTimeout("timed out")

    assert runner.is_msa_server_timeout(wrapper) is True


def test_an_unrelated_failure_is_not_mistaken_for_a_throttled_msa_server(install_fake_colabfold_client):
    """Misreporting it as transient would tell callers to retry a real bug forever."""
    install_fake_colabfold_client()

    assert runner.is_msa_server_timeout(RuntimeError("checkpoint corrupt")) is False


def test_is_msa_server_timeout_terminates_on_a_looping_cause_chain(install_fake_colabfold_client):
    """The id()-based seen-set guard must stop a self-referential chain rather than hang."""
    install_fake_colabfold_client()

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a

    assert runner.is_msa_server_timeout(a) is False


def test_bounded_msa_server_enters_the_clients_deadline_context(install_fake_colabfold_client):
    """A real deadline must reach the fake client's contextmanager, with the same value."""
    _, calls, _ = install_fake_colabfold_client()

    with runner.bounded_msa_server(1234.5):
        pass

    assert calls == [1234.5]


def test_a_prediction_without_a_deadline_is_left_unbounded(install_fake_colabfold_client):
    """Inline-MSA requests never reach the server, so the client must not be touched at all."""
    _, calls, _ = install_fake_colabfold_client()

    with runner.bounded_msa_server(None):
        pass

    assert calls == []
