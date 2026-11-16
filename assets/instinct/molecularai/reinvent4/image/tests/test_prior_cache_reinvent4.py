# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Prior cache path + pre-flight download of the REINVENT4 service.

service.py needs bentoml/torch, so the pure helpers are extracted from the AST
and executed against a stub namespace.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import logging
import os
import tempfile
import threading
from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parents[1] / "src/service.py").read_text(encoding="utf-8")
WANTED = {
    "_cache_root", "_priors_dir", "_prior_path", "_prior_file", "_md5", "_sha256", "_builtin_priors",
    "_ensure_prior", "_verify_builtin", "_cache_key", "_remember_verified",
}  # fmt: skip
CONSTS = {
    "PRIOR_FILENAME",
    "PRIOR_URL",
    "PRIOR_MD5",
    "PRIOR_SIZE",
    "DEFAULT_PRIOR",
    "_DOWNLOAD_TIMEOUT_S",
    "_DOWNLOAD_LOCK",
    "_VERIFIED_BUILTINS",
    "_RACY_WINDOW_NS",
}


class _Resp:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, _n: int):
        yield self.data


class _HttpxErr(Exception):
    pass


def _load(data: bytes | None, *, md5: str | None = None, size: int | None = None, fail: bool = False):
    tree = ast.parse(SRC)
    body = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "PriorIntegrityError":
            body.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in WANTED:
            body.append(n)
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in CONSTS:
            body.append(n)
    calls: list[str] = []

    @contextlib.contextmanager
    def fake_stream(method, url, **_kw):
        calls.append(url)
        if fail:
            raise _HttpxErr("boom")
        yield _Resp(data or b"")

    class _Httpx:
        HTTPError = _HttpxErr
        stream = staticmethod(fake_stream)

    ns = {
        "os": os, "time": __import__("time"), "Path": Path, "hashlib": hashlib, "tempfile": tempfile,
        "logger": logging.getLogger("t"), "httpx": _Httpx, "threading": threading,
    }  # fmt: skip
    exec(compile(ast.Module(body, []), "service", "exec"), ns)  # noqa: S102
    if data is not None:
        ns["PRIOR_MD5"] = md5 or hashlib.md5(data).hexdigest()
        ns["PRIOR_SIZE"] = size if size is not None else len(data)
    return ns, calls


def test_path_resolution_default_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ns, _ = _load(None)
    monkeypatch.delenv("AIM_CACHE_PATH", raising=False)
    assert ns["_prior_path"]() == Path("/workspace/model-cache/reinvent4/priors/reinvent.prior")
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    assert ns["_prior_path"]() == tmp_path.resolve() / "reinvent4/priors/reinvent.prior"


def test_downloads_when_missing_and_is_atomic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    ns, calls = _load(b"weights")
    dest = ns["_ensure_prior"]()
    assert dest.read_bytes() == b"weights" and calls == [ns["PRIOR_URL"]]
    assert [p.name for p in dest.parent.iterdir()] == ["reinvent.prior"]  # temp file renamed away
    ns["_ensure_prior"]()
    assert len(calls) == 1  # present: no second download


@pytest.mark.parametrize("kw", [{"md5": "0" * 32}, {"size": 1}, {"fail": True}])
def test_failure_names_path_and_url_and_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kw) -> None:
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    ns, _ = _load(b"weights", **kw)
    with pytest.raises(RuntimeError) as ei:
        ns["_ensure_prior"]()
    msg = str(ei.value)
    assert str(tmp_path.resolve() / "reinvent4/priors/reinvent.prior") in msg and ns["PRIOR_URL"] in msg
    assert list((tmp_path / "reinvent4/priors").iterdir()) == []


def test_pinned_constants_and_startup_wiring() -> None:
    assert "https://zenodo.org/records/20701824/files/reinvent_pubchem.prior" in SRC
    assert 'PRIOR_MD5 = "75052d5e93b2470e9c6684d369704f5f"' in SRC
    assert "PRIOR_PATH" not in SRC and "/workspace/reinvent4/priors" not in SRC
    assert "AIM_INPUT_PATH" not in SRC
