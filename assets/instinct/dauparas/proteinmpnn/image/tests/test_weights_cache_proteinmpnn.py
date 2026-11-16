# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Weights live under AIM_CACHE_PATH and are fetched/verified at startup.

Extracts the weight helpers from service.py by AST (bentoml/torch are not installed); network mocked.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
DOCKERFILE = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
FUNCS = {
    "_cache_root",
    "_sha256",
    "_check_weight",
    "_ensure_weight",
    "_preflight_weights",
    "_fail_startup",
    "_configure_logging",
}
ASSIGNS = {
    "ALLOWED_MODEL_NAMES",
    "WEIGHTS_SOURCE_COMMIT",
    "WEIGHTS_URL_TEMPLATE",
    "WEIGHT_SIZE",
    "WEIGHT_SHA256",
    "_DOWNLOAD_TIMEOUT_S",
    "_VERIFIED_WEIGHTS",
}


class _Resp:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, n: int):
        yield self.data


class _Httpx:
    class HTTPError(Exception):
        pass

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.urls: list[str] = []

    def stream(self, method, url, **kw):
        self.urls.append(url)
        return _Resp(self.payloads[url.rsplit("/", 1)[1][:-3]])


def _load(tmp_path: Path, payloads: dict[str, bytes], size: int):
    tree = ast.parse(SERVICE_SRC)
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in FUNCS:
            n.returns = None
            body.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            t = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(t, ast.Name) and t.id in ASSIGNS:
                if isinstance(n, ast.AnnAssign):
                    n.annotation = ast.Constant("x")
                body.append(n)
    httpx = _Httpx(payloads)
    import tempfile

    ns = {
        "os": os,
        "Path": Path,
        "hashlib": hashlib,
        "tempfile": tempfile,
        "httpx": httpx,
        "logger": logging.getLogger("pmpnn-w"),
        "logging": logging,
        "threading": threading,
        "sys": __import__("sys"),
        "annotations": None,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "service.py", "exec"), ns
    )  # noqa: S102
    ns["WEIGHTS_DIR"] = tmp_path / "proteinmpnn" / "vanilla_model_weights"
    ns["_DOWNLOAD_LOCK"] = threading.Lock()
    ns["WEIGHT_SIZE"] = size
    ns["WEIGHT_SHA256"] = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    ns["ALLOWED_MODEL_NAMES"] = tuple(payloads)
    return ns, httpx


@pytest.fixture
def payloads() -> dict[str, bytes]:
    return {n: n[-3:].encode() * 3 for n in ("v_48_002", "v_48_010", "v_48_020", "v_48_030")}


def test_cache_root_default_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payloads) -> None:
    ns, _ = _load(tmp_path, payloads, 9)
    monkeypatch.delenv("AIM_CACHE_PATH", raising=False)
    assert ns["_cache_root"]() == Path("/workspace/model-cache").resolve()
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    assert ns["_cache_root"]() == tmp_path.resolve()


def test_weights_dir_is_under_cache_not_proteinmpnn_path() -> None:
    line = next(ln for ln in SERVICE_SRC.splitlines() if ln.startswith("WEIGHTS_DIR ="))
    assert "_cache_root()" in line and "proteinmpnn" in line and "PROTEINMPNN_PATH" not in line
    assert "WEIGHTS_DIR" in SERVICE_SRC and "path_to_model_weights=str(WEIGHTS_DIR)" in SERVICE_SRC


def test_pinned_constants() -> None:
    assert "905b0081f3aa3404d7ac22174bc25710f9995ff8" in SERVICE_SRC
    assert len(re.findall(r'"v_48_0\d\d": "[0-9a-f]{64}"', SERVICE_SRC)) == 4


def test_preflight_downloads_missing_and_renames_atomically(tmp_path: Path, payloads, caplog) -> None:
    ns, httpx = _load(tmp_path, payloads, 9)
    with caplog.at_level(logging.INFO):
        ns["_preflight_weights"]()
    wd = ns["WEIGHTS_DIR"]
    assert sorted(p.name for p in wd.iterdir()) == sorted(f"{n}.pt" for n in payloads)  # no .part left
    assert (wd / "v_48_002.pt").read_bytes() == payloads["v_48_002"]
    assert len(httpx.urls) == 4 and all("905b0081" in u for u in httpx.urls)
    assert str(wd) in caplog.text


def test_existing_good_files_not_redownloaded(tmp_path: Path, payloads) -> None:
    ns, httpx = _load(tmp_path, payloads, 9)
    wd = ns["WEIGHTS_DIR"]
    wd.mkdir(parents=True)
    for n, d in payloads.items():
        (wd / f"{n}.pt").write_bytes(d)
    ns["_preflight_weights"]()
    ns["_preflight_weights"]()
    assert httpx.urls == []
    assert len(ns["_VERIFIED_WEIGHTS"]) == 4


def test_checksum_mismatch_stops_startup(tmp_path: Path, payloads, monkeypatch) -> None:
    ns, httpx = _load(tmp_path, payloads, 9)
    httpx.payloads["v_48_010"] = b"X" * 9  # right size, wrong content
    with pytest.raises(RuntimeError) as ei:
        ns["_preflight_weights"]()
    msg = str(ei.value)
    assert "v_48_010.pt" in msg and "sha256" in msg
    # Parse the URL out of the message and compare scheme/host/path, not a substring.
    url = urlparse(re.search(r"https://[^\s)]+?\.pt", msg).group(0))
    assert (url.scheme, url.hostname) == ("https", "raw.githubusercontent.com")
    assert url.path.endswith("/v_48_010.pt")
    assert not list(ns["WEIGHTS_DIR"].glob("*v_48_010*"))  # temp file cleaned, nothing renamed
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    try:
        ns["_preflight_weights"]()
    except RuntimeError as exc:
        ns["_fail_startup"](str(exc))
    assert exits == [1]


def test_corrupt_existing_file_stops_startup(tmp_path: Path, payloads) -> None:
    ns, httpx = _load(tmp_path, payloads, 9)
    wd = ns["WEIGHTS_DIR"]
    wd.mkdir(parents=True)
    (wd / "v_48_002.pt").write_bytes(b"short")
    with pytest.raises(RuntimeError, match="size 5"):
        ns["_preflight_weights"]()
    assert httpx.urls == []


def test_dockerfile_does_not_ship_weights() -> None:
    assert "PROTEINMPNN_REF=905b0081f3aa3404d7ac22174bc25710f9995ff8" in DOCKERFILE
    # Clone, checkout and prune must be ONE RUN, or the weights and .git stay in an earlier layer.
    run = next(b for b in re.split(r"\n\s*\n", DOCKERFILE) if "git clone ${PROTEINMPNN_REPO}" in b)
    run = re.sub(r"\s+", " ", re.sub(r"\\\n\s*", "", run))
    assert run.index("checkout ${PROTEINMPNN_REF}") < run.index("-exec rm -rf")
    # Allowlist prune: everything but the run script, its utils and the LICENSE is removed
    # (that drops all three *_model_weights directories and .git), and no .pt may remain.
    assert "! -name protein_mpnn_run.py ! -name protein_mpnn_utils.py ! -name LICENSE" in run
    assert "-mindepth 1 -maxdepth 1" in run
    assert "*.pt" in run


def test_preflight_message_reaches_stderr(tmp_path: Path, payloads, capfd) -> None:
    ns, _ = _load(tmp_path, payloads, 9)
    lg = logging.getLogger("pmpnn-stderr-test")
    lg.handlers.clear()
    ns["logger"] = lg
    ns["_configure_logging"](lg)
    ns["_configure_logging"](lg)  # idempotent
    assert len(lg.handlers) == 1
    ns["_preflight_weights"]()
    err = capfd.readouterr().err
    assert "ProteinMPNN weights directory:" in err and str(ns["WEIGHTS_DIR"]) in err
    assert "downloading from" in err
    lg.handlers.clear()
