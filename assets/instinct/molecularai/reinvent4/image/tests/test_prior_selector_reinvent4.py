# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Prior selector of the REINVENT4 service: registry, staging, lazy download, 400/503.

service.py needs bentoml/torch, so helpers are extracted from the AST and run
against a stub namespace; the network is mocked.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parents[1] / "src/service.py").read_text(encoding="utf-8")
FUNCS = {
    "_cache_root", "_priors_dir", "_prior_path", "_md5", "_sha256", "_builtin_priors", "_staged_priors",
    "_prior_file", "_verify_staged", "_allowed_names", "_default_prior_name", "_ensure_prior",
    "_resolve_prior", "_expected_checksum", "_snapshot_prior", "_preflight_priors", "_list_priors", "_verify_builtin", "_configure_logging", "_cache_key", "_remember_verified"
}  # fmt: skip
CONSTS = {
    "PRIOR_FILENAME", "PRIOR_URL", "PRIOR_MD5", "PRIOR_SIZE", "DEFAULT_PRIOR", "PRIOR_ENV", "EXTRA_PRIORS_ENV",
    "_PRIOR_NAME_RE", "_SHA256_RE", "_DOWNLOAD_TIMEOUT_S", "_DOWNLOAD_LOCK", "_VERIFIED_BUILTINS", "_RACY_WINDOW_NS",
}  # fmt: skip


class _Err(Exception):
    pass


class _Resp:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, _n: int):
        yield self.data


def _ns(downloads: dict[str, bytes] | None = None):
    """Exec the selector helpers; ``downloads`` maps URL -> payload for the mocked network."""
    body = []
    for n in ast.parse(SRC).body:
        if isinstance(n, ast.FunctionDef) and n.name in FUNCS:
            body.append(n)
        elif isinstance(n, ast.ClassDef) and n.name in {"UnknownPriorError", "PriorIntegrityError"}:
            body.append(n)
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in CONSTS:
            body.append(n)
    calls: list[str] = []

    @contextlib.contextmanager
    def fake_stream(_m, url, **_kw):
        calls.append(url)
        if downloads is None or url not in downloads:
            raise _Err("network down")
        yield _Resp(downloads[url])

    class _Httpx:
        HTTPError = _Err
        stream = staticmethod(fake_stream)

    ns = {
        "os": os, "time": __import__("time"), "Path": Path, "hashlib": hashlib, "tempfile": tempfile, "json": json, "re": re,
        "threading": threading, "logging": logging, "sys": __import__("sys"), "logger": logging.getLogger("t"), "httpx": _Httpx,
    }  # fmt: skip
    exec(compile(ast.Module(body, []), "service", "exec"), ns)  # noqa: S102
    ns["calls"] = calls
    return ns


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    monkeypatch.delenv("REINVENT4_PRIOR", raising=False)
    monkeypatch.delenv("REINVENT4_EXTRA_PRIORS", raising=False)
    return tmp_path / "reinvent4" / "priors"


def _stage(cache: Path, name: str, data: bytes = b"ckpt") -> str:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{name}.prior").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _with_lazy_builtin(ns, data: bytes) -> str:
    """Add a second built-in prior (mocked download) to the registry."""
    url = "https://example.invalid/other.prior"
    base = ns["_builtin_priors"]()
    base["other"] = {"filename": "other.prior", "url": url, "md5": hashlib.md5(data).hexdigest(), "size": len(data)}
    ns["_builtin_priors"] = lambda: base
    return url


def _pin_builtin(ns, data: bytes) -> dict:
    """Pin the default built-in prior to ``data`` (the real pin is a 23 MB file)."""
    entry = ns["_builtin_priors"]()["reinvent_pubchem"]
    entry.update(md5=hashlib.md5(data).hexdigest(), size=len(data))
    ns["_builtin_priors"] = lambda: {"reinvent_pubchem": entry}
    return entry


def test_default_unchanged(cache: Path) -> None:
    ns = _ns()
    assert ns["_default_prior_name"]() == "reinvent_pubchem"
    assert ns["_allowed_names"]() == ["reinvent_pubchem"]
    assert ns["_prior_path"]() == cache.resolve() / "reinvent.prior"
    _pin_builtin(ns, b"x")
    (cache).mkdir(parents=True)
    (cache / "reinvent.prior").write_bytes(b"x")
    assert ns["_resolve_prior"](None) == ns["_prior_path"]()
    assert ns["_resolve_prior"]("reinvent_pubchem") == ns["_prior_path"]()


def test_unknown_name_lists_allowed(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": sha}))
    ns = _ns()
    with pytest.raises(ns["UnknownPriorError"]) as ei:
        ns["_resolve_prior"]("nope")
    assert "reinvent_pubchem" in str(ei.value) and "mine" in str(ei.value)


@pytest.mark.parametrize("bad", ["../x", "a/b", "/etc/passwd", "..", "a..b/c", ".hidden", "", "x" * 65, "a b", "a\\b"])
def test_invalid_names_and_paths_rejected(cache: Path, bad: str) -> None:
    ns = _ns()
    with pytest.raises(ns["UnknownPriorError"]):
        ns["_resolve_prior"](bad)


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", ".hidden", "x y"])
def test_invalid_staged_declaration_refused(cache: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({bad: "0" * 64}))
    with pytest.raises(RuntimeError):
        _ns()["_staged_priors"]()


@pytest.mark.parametrize(
    "raw", ["not json", "[]", json.dumps({"reinvent_pubchem": "0" * 64}), json.dumps({"ok": "short"})]
)
def test_bad_extra_priors_config_refused(cache: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", raw)
    with pytest.raises(RuntimeError):
        _ns()["_staged_priors"]()


def test_prior_file_cannot_escape_cache(cache: Path) -> None:
    ns = _ns()
    with pytest.raises(RuntimeError):
        ns["_prior_file"]("../escape.prior")
    cache.mkdir(parents=True)
    (cache.parent / "outside.prior").write_bytes(b"x")
    (cache / "link.prior").symlink_to(cache.parent / "outside.prior")
    with pytest.raises(RuntimeError):
        ns["_prior_file"]("link.prior")


def test_staged_sha256_match_and_mismatch(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": sha.upper()}))
    ns = _ns()
    _pin_builtin(ns, b"d")
    (cache / "reinvent.prior").write_bytes(b"d")
    assert ns["_resolve_prior"]("mine") == (cache / "mine.prior").resolve()
    ns["_preflight_priors"]()  # passes
    (cache / "mine.prior").write_bytes(b"tampered")  # swapped after startup
    with pytest.raises(RuntimeError) as ei:
        ns["_resolve_prior"]("mine")
    msg = str(ei.value)
    assert "mine.prior" in msg and sha in msg and hashlib.sha256(b"tampered").hexdigest() in msg
    with pytest.raises(RuntimeError, match="mine.prior"):
        ns["_preflight_priors"]()


def test_staged_missing_file_refuses_startup(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"ghost": "a" * 64}))
    with pytest.raises(RuntimeError, match="ghost.prior"):
        _ns()["_preflight_priors"]()


def test_undeclared_file_in_cache_is_never_selectable(cache: Path) -> None:
    _stage(cache, "sneaky")
    ns = _ns()
    with pytest.raises(ns["UnknownPriorError"]):
        ns["_resolve_prior"]("sneaky")


def test_env_default_selection(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": sha}))
    monkeypatch.setenv("REINVENT4_PRIOR", "mine")
    ns = _ns()
    assert ns["_default_prior_name"]() == "mine"
    assert ns["_resolve_prior"](None) == (cache / "mine.prior").resolve()
    # explicit request overrides the env default
    _pin_builtin(ns, b"d")
    (cache / "reinvent.prior").write_bytes(b"d")
    assert ns["_resolve_prior"]("reinvent_pubchem") == ns["_prior_path"]()
    ns["_preflight_priors"]()


def test_unknown_env_default_stops_startup(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REINVENT4_PRIOR", "nope")
    ns = _ns()
    with pytest.raises(RuntimeError, match="REINVENT4_PRIOR.*Allowed priors: reinvent_pubchem"):
        ns["_preflight_priors"]()


def test_lazy_download_atomic_and_cached(cache: Path) -> None:
    data = b"other-weights"
    ns = _ns()
    url = _with_lazy_builtin(ns, data)
    ns2 = _ns({url: data})
    ns2["_builtin_priors"] = ns["_builtin_priors"]
    path = ns2["_resolve_prior"]("other")
    assert path.read_bytes() == data and ns2["calls"] == [url]
    assert [p.name for p in cache.iterdir()] == ["other.prior"]  # no .part left
    ns2["_resolve_prior"]("other")
    assert len(ns2["calls"]) == 1


@pytest.mark.parametrize("mode", ["md5", "size", "network"])
def test_lazy_download_failure_names_path_url_and_leaves_nothing(cache: Path, mode: str) -> None:
    data = b"other-weights"
    ns = _ns()
    url = _with_lazy_builtin(ns, data)
    entry = ns["_builtin_priors"]()["other"]
    served = {url: data} if mode != "network" else {}
    if mode == "md5":
        entry["md5"] = "0" * 32
    if mode == "size":
        entry["size"] = 1
    ns2 = _ns(served)
    ns2["_builtin_priors"] = ns["_builtin_priors"]
    with pytest.raises(RuntimeError) as ei:
        ns2["_resolve_prior"]("other")
    assert url in str(ei.value) and str(cache.resolve() / "other.prior") in str(ei.value)
    assert list(cache.iterdir()) == []


def test_models_listing(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": sha}))
    ns = _ns()
    assert ns["_list_priors"]() == [
        {"name": "reinvent_pubchem", "source": "built-in", "checksum": "md5:" + ns["PRIOR_MD5"], "present": False},
        {"name": "mine", "source": "staged", "checksum": "sha256:" + sha, "present": True},
    ]
    (cache / "reinvent.prior").write_bytes(b"d")
    assert ns["_list_priors"]()[0]["present"] is True
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", "garbage")
    assert [p["name"] for p in ns["_list_priors"]()] == ["reinvent_pubchem"]  # never raises


def test_registry_default_is_pinned_zenodo(cache: Path) -> None:
    e = _ns()["_builtin_priors"]()["reinvent_pubchem"]
    assert e["url"].startswith("https://zenodo.org/records/20701824/files/") and len(e["md5"]) == 32


def test_wiring_400_503_and_no_path_input() -> None:
    tree = ast.parse(SRC)
    infer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "inference")
    dump = ast.dump(infer)
    assert "BadInput" in dump and "ServiceUnavailable" in dump and "_resolve_prior" in dump
    # integrity failure is a permanent server-side fault: 500, never 503 (download failure only)
    handlers = {ast.unparse(h.type): ast.dump(h) for t in ast.walk(infer) if isinstance(t, ast.Try) for h in t.handlers}
    assert "InternalServerError" in handlers["PriorIntegrityError"]
    assert "ServiceUnavailable" not in handlers["PriorIntegrityError"]
    assert "ServiceUnavailable" in handlers["RuntimeError"]
    # the only client input reaching a prior is the NAME field, resolved via the registry
    req = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SamplingRequest")
    fields = [n.target.id for n in req.body if isinstance(n, ast.AnnAssign)]
    assert "prior" in fields and not {"model_file", "path", "prior_path", "file"} & set(fields)
    api_routes = [
        kw.value.value
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "REINVENT4Service"
        for f in c.body
        if isinstance(f, ast.FunctionDef)
        for d in f.decorator_list
        if isinstance(d, ast.Call)
        for kw in d.keywords
        if kw.arg == "route"
    ]
    assert api_routes == ["/v1/inference"]
    # the resident sampler is only ever fed the resolved path (plus the validated name), never the
    # request's prior name; it loads a private verified snapshot of that path
    assert "prior_path=prior_path" in SRC and "data.prior" not in SRC.split("_resolve_prior(data.prior)")[1]
    assert "_snapshot_prior(job.prior_name, prior_path," in SRC


def test_builtin_prior_reverified_and_cached(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    cache.mkdir(parents=True)
    f = cache / entry["filename"]
    f.write_bytes(data)
    hashed: list[Path] = []
    real_md5 = ns["_md5"]
    ns["_md5"] = lambda p: (hashed.append(p), real_md5(p))[1]

    class _Later:  # observe the file long after its last change, so its key is conclusive
        time_ns = staticmethod(lambda: f.stat().st_ctime_ns + 10 * ns["_RACY_WINDOW_NS"])

    ns["time"] = _Later
    assert ns["_resolve_prior"]("reinvent_pubchem") == f
    ns["_resolve_prior"]("reinvent_pubchem")
    assert len(hashed) == 1  # second call served from the (path, size, mtime, inode, ctime) cache
    f.write_bytes(b"tampered-weights")  # size change
    with pytest.raises(ns["PriorIntegrityError"], match="size|md5"):
        ns["_resolve_prior"]("reinvent_pubchem")


def test_builtin_same_size_tamper_detected(cache: Path) -> None:
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    cache.mkdir(parents=True)
    f = cache / entry["filename"]
    f.write_bytes(b"x" * len(data))
    with pytest.raises(ns["PriorIntegrityError"], match="md5 mismatch"):
        ns["_resolve_prior"]("reinvent_pubchem")


def test_staged_mismatch_is_integrity_error(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": digest}))
    ns = _ns()
    ns["_resolve_prior"]("mine")
    (cache / "mine.prior").write_bytes(b"swapped")
    with pytest.raises(ns["PriorIntegrityError"], match="sha256 mismatch"):
        ns["_resolve_prior"]("mine")


def test_info_lines_reach_stderr(cache: Path, capfd: pytest.CaptureFixture[str]) -> None:
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    lg = logging.getLogger("reinvent4-stderr-test")
    lg.handlers.clear()
    ns["logger"] = lg
    ns["_configure_logging"](lg)
    ns["_configure_logging"](lg)  # idempotent
    assert len(lg.handlers) == 1 and lg.level == logging.INFO
    cache.mkdir(parents=True)
    (cache / entry["filename"]).write_bytes(data)
    ns["_ensure_prior"](entry)
    err = capfd.readouterr().err
    assert str(cache / entry["filename"]) in err and "verified" in err and "[INFO]" in err
    lg.handlers.clear()


def test_downloaded_prior_is_private_to_the_service_user(cache: Path) -> None:
    data = b"downloaded-weights"
    ns = _ns({"http://x/p": data})
    entry = _pin_builtin(ns, data)
    entry["url"] = "http://x/p"
    dest = ns["_ensure_prior"](entry)
    assert dest.stat().st_mode & 0o777 == 0o600
    ns["_ensure_prior"](entry)  # re-verify path still fine


def test_same_size_tamper_with_restored_mtime_is_detected(cache: Path) -> None:
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    cache.mkdir(parents=True)
    f = cache / entry["filename"]
    f.write_bytes(data)
    ns["_resolve_prior"]("reinvent_pubchem")
    st = f.stat()
    # replace the file (new inode) with same-size content and restore the mtime
    f.unlink()
    f.write_bytes(b"X" * len(data))
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ns["PriorIntegrityError"], match="md5"):
        ns["_resolve_prior"]("reinvent_pubchem")
    # in-place rewrite + restored mtime: ctime still moves
    f.write_bytes(data)
    ns["_resolve_prior"]("reinvent_pubchem")
    st = f.stat()
    f.write_bytes(b"Y" * len(data))
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ns["PriorIntegrityError"], match="md5"):
        ns["_resolve_prior"]("reinvent_pubchem")


def test_fresh_file_is_not_trusted_by_cache_key_but_settled_file_is(cache: Path) -> None:
    """A key is only conclusive once ctime is older than the clock granularity (racy-file rule).

    Coarse-timestamp / inode-recycling filesystems (tmpfs, overlayfs on CI) can give a same-size
    rewrite with a restored mtime the very same (ino, mtime, ctime) key; so a just-written file
    must be re-hashed on every call, and only a settled one may be skipped.
    """
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    cache.mkdir(parents=True)
    f = cache / entry["filename"]
    f.write_bytes(data)
    hashed: list[Path] = []
    real_md5 = ns["_md5"]
    ns["_md5"] = lambda p: (hashed.append(p), real_md5(p))[1]
    ns["_resolve_prior"]("reinvent_pubchem")
    ns["_resolve_prior"]("reinvent_pubchem")
    assert len(hashed) == 2 and not ns["_VERIFIED_BUILTINS"]  # fresh: hashed every time

    class _Later:  # the same file, observed well after its last change
        @staticmethod
        def time_ns() -> int:
            return os.stat(f).st_ctime_ns + 10 * ns["_RACY_WINDOW_NS"]

    ns["time"] = _Later
    ns["_resolve_prior"]("reinvent_pubchem")
    ns["_resolve_prior"]("reinvent_pubchem")
    assert len(hashed) == 3 and len(ns["_VERIFIED_BUILTINS"]) == 1  # settled: hashed once, then cached


def test_tamper_detected_on_coarse_ctime_filesystem(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (CI runner): ctime reported at 1 s granularity + recycled inode + restored mtime."""
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    cache.mkdir(parents=True)
    f = cache / entry["filename"]
    f.write_bytes(data)
    orig = Path.stat

    class _Coarse:
        def __init__(self, st: os.stat_result) -> None:
            self._st = st
            self.st_ctime_ns = st.st_ctime_ns // 10**9 * 10**9

        def __getattr__(self, name: str):
            return getattr(self._st, name)

    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: _Coarse(orig(self, *a, **k)))
    ns["_resolve_prior"]("reinvent_pubchem")
    st = orig(f)
    f.write_bytes(b"Z" * len(data))  # same size, same inode
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ns["PriorIntegrityError"], match="md5"):
        ns["_resolve_prior"]("reinvent_pubchem")


def test_snapshot_is_private_verified_copy_and_ignores_later_swap(cache: Path, tmp_path: Path) -> None:
    data = b"builtin-weights"
    ns = _ns()
    entry = _pin_builtin(ns, data)
    cache.mkdir(parents=True)
    src = cache / entry["filename"]
    src.write_bytes(data)
    dest_dir = tmp_path / "req"
    dest_dir.mkdir(mode=0o700)
    snap = ns["_snapshot_prior"]("reinvent_pubchem", ns["_resolve_prior"]("reinvent_pubchem"), dest_dir)
    assert snap.parent == dest_dir and snap != src
    assert snap.stat().st_mode & 0o777 == 0o600
    src.write_bytes(b"evil-pickle-xx")  # swapped after verification: the snapshot is unaffected
    assert snap.read_bytes() == data


def test_snapshot_rejects_a_file_swapped_before_the_copy(
    cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": digest}))
    ns = _ns()
    src = ns["_resolve_prior"]("mine")  # verified ...
    src.write_bytes(b"swapped")  # ... then replaced before the copy is taken
    dest_dir = tmp_path / "req"
    dest_dir.mkdir(mode=0o700)
    with pytest.raises(ns["PriorIntegrityError"], match="mismatch"):
        ns["_snapshot_prior"]("mine", src, dest_dir)
    assert list(dest_dir.iterdir()) == []  # the bad copy is removed


def test_snapshot_refuses_a_symlink(cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    digest = _stage(cache, "mine")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"mine": digest}))
    ns = _ns()
    link = cache / "link.prior"
    link.symlink_to(cache / "mine.prior")
    dest_dir = tmp_path / "req"
    dest_dir.mkdir(mode=0o700)
    with pytest.raises(OSError):
        ns["_snapshot_prior"]("mine", link, dest_dir)


def test_staged_name_cannot_take_a_builtin_prior_filename(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Staged 'reinvent' maps to reinvent.prior, the file of the built-in reinvent_pubchem."""
    digest = _stage(cache, "reinvent")
    monkeypatch.setenv("REINVENT4_EXTRA_PRIORS", json.dumps({"reinvent": digest}))
    ns = _ns()
    with pytest.raises(RuntimeError, match="reinvent.prior"):
        ns["_staged_priors"]()
