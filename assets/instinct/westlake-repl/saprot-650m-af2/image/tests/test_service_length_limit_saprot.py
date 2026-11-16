# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Length limit of the SaProt service: over-long input is rejected, never truncated.

Never imports service.py (bentoml and torch are not installed here); the pure
length-check function is extracted by AST and executed with a stub InvalidArgument.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable

import pytest

SRC = (Path(__file__).resolve().parents[1] / "src/service.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


class _InvalidArgument(Exception):
    """Stand-in for bentoml.exceptions.InvalidArgument (HTTP 400)."""


def _namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"InvalidArgument": _InvalidArgument}
    wanted = {"MAX_TOKENS", "MAX_RESIDUES", "AA_LETTERS", "STRUCT_LETTERS"}
    for node in TREE.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in wanted:
            exec(compile(ast.Module([node], []), "service.py", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_check_sequence_length",
            "_check_sequence_format",
            "_check_token_count",
        }:
            exec(compile(ast.Module([node], []), "service.py", "exec"), ns)
    return ns


@pytest.fixture
def check() -> Callable[[str], None]:
    return _namespace()["_check_sequence_length"]


def _seq(n_residues: int) -> str:
    return "A#" * n_residues


def test_limit_is_1022_residues() -> None:
    ns = _namespace()
    assert ns["MAX_TOKENS"] == 1024
    assert ns["MAX_RESIDUES"] == 1022


def test_exactly_1022_residues_accepted(check: Callable[[str], None]) -> None:
    check(_seq(1022))


def test_1023_residues_rejected_with_message(check: Callable[[str], None]) -> None:
    with pytest.raises(_InvalidArgument) as exc:
        check(_seq(1023))
    msg = str(exc.value)
    assert "1023" in msg and "1022" in msg


def test_structure_tokens_count_as_part_of_the_residue(check: Callable[[str], None]) -> None:
    # 1022 residues is 2044 characters: the structure token is not a separate residue.
    assert len(_seq(1022)) == 2044
    check("Ma" * 1022)
    with pytest.raises(_InvalidArgument):
        check("Ma" * 1023)


@pytest.fixture
def fmt() -> Callable[[str], None]:
    return _namespace()["_check_sequence_format"]


def test_paired_1022_accepted_and_1023_rejected(fmt: Callable[[str], None], check: Callable[[str], None]) -> None:
    for n, ok in ((1022, True), (1023, False)):
        seq = _seq(n)
        fmt(seq)  # well-formed either way
        if ok:
            check(seq)
        else:
            with pytest.raises(_InvalidArgument, match="1022"):
                check(seq)


def test_plain_amino_acid_sequence_rejected(fmt: Callable[[str], None]) -> None:
    with pytest.raises(_InvalidArgument, match="structure-aware"):
        fmt("MAKT")
    with pytest.raises(_InvalidArgument, match="structure-aware"):
        fmt("MAKTLV" * 100)


def test_odd_length_rejected(fmt: Callable[[str], None]) -> None:
    with pytest.raises(_InvalidArgument, match="even"):
        fmt("M#A")


def test_internal_whitespace_and_bad_characters_rejected(fmt: Callable[[str], None]) -> None:
    for bad in ("M# A#", "m#A#", "M1A#", "MaA"):
        with pytest.raises(_InvalidArgument):
            fmt(bad)


def test_valid_pairs_accepted(fmt: Callable[[str], None]) -> None:
    fmt("M#A#K#T#")
    fmt(" Mp Ad".replace(" ", "") + "\n")


def test_token_count_over_max_rejected(tokens: Callable[[int], None]) -> None:
    tokens(1024)
    with pytest.raises(_InvalidArgument, match="1025"):
        tokens(1025)


@pytest.fixture
def tokens() -> Callable[[int], None]:
    return _namespace()["_check_token_count"]


def test_format_and_token_checks_precede_model_and_map_to_400() -> None:
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "SaProtEmbeddingService")
    infer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "inference")
    calls = [
        (n.lineno, n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(infer)
        if isinstance(n, ast.Call)
    ]
    line = {name: ln for ln, name in calls}
    assert line["_check_sequence_format"] < line["_tokenizer"]
    assert line["_check_token_count"] < line["_model"]
    # Tokenizer failures are caught separately and raised as InvalidArgument, not InternalServerError.
    tries = [n for n in ast.walk(infer) if isinstance(n, ast.Try)]
    tok_try = next(
        t
        for t in tries
        if any(
            isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_tokenizer" for b in t.body for c in ast.walk(b)
        )
    )
    raised = [
        n.exc.func.id
        for h in tok_try.handlers
        for n in ast.walk(h)
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
    ]
    assert raised == ["InvalidArgument"]


def test_surrounding_whitespace_not_counted(check: Callable[[str], None]) -> None:
    check("  " + _seq(1022) + "\n")


def test_check_runs_before_tokenizer_and_model() -> None:
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "SaProtEmbeddingService")
    infer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "inference")
    calls = [
        (n.lineno, n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(infer)
        if isinstance(n, ast.Call)
    ]
    check_line = next(line for line, name in calls if name == "_check_sequence_length")
    assert check_line < next(line for line, name in calls if name == "_tokenizer")
    assert check_line < next(line for line, name in calls if name == "_model")


def test_tokenizer_never_truncates() -> None:
    """No truncation=True / max_length anywhere: over-long input must not be silently cut."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.keyword):
            if node.arg == "truncation":
                assert isinstance(node.value, ast.Constant) and node.value.value is False
            assert node.arg != "max_length"


AA = "ACDEFGHIKLMNPQRSTVWY"
STRUCT = "acdefghiklmnpqrstvwy#"
API_MD = (Path(__file__).resolve().parents[2] / "templates/API.md").read_text(encoding="utf-8")


def test_full_real_vocabulary_accepted(fmt: Callable[[str], None]) -> None:
    for a in AA:
        for s in STRUCT:
            fmt(a + s)
    fmt("".join(a + s for a in AA for s in STRUCT))


@pytest.mark.parametrize("bad", list("BJOUXZ"))
def test_non_vocab_amino_acid_rejected_naming_characters(fmt: Callable[[str], None], bad: str) -> None:
    with pytest.raises(_InvalidArgument) as exc:
        fmt("M#" + bad + "a" + "K#")
    assert repr(bad + "a") in str(exc.value)


@pytest.mark.parametrize("bad", list("bjouxz"))
def test_non_vocab_3di_rejected_naming_characters(fmt: Callable[[str], None], bad: str) -> None:
    with pytest.raises(_InvalidArgument) as exc:
        fmt("M#K" + bad)
    assert repr("K" + bad) in str(exc.value)


def test_unk_collapsing_inputs_rejected(fmt: Callable[[str], None]) -> None:
    for bad in ("B#J#O#U#X#Z#", "Xa", "MbMjMoMuMxMz", "M#X#K#", "MzKz"):
        with pytest.raises(_InvalidArgument):
            fmt(bad)


def test_hash_first_tokens_rejected(fmt: Callable[[str], None]) -> None:
    for bad in ("#a#d", "##", "M#K#####"):
        with pytest.raises(_InvalidArgument):
            fmt(bad)


def test_uppercase_3di_message_names_lowercase(fmt: Callable[[str], None]) -> None:
    with pytest.raises(_InvalidArgument) as exc:
        fmt("MDKVQP")  # foldseek-style uppercase 3Di after each residue
    msg = str(exc.value)
    assert "lowercase" in msg and "Foldseek" in msg
    assert "append '#'" not in msg


def test_api_md_documents_validation_rules() -> None:
    assert "446" in API_MD and "441 tokens" not in API_MD
    assert "ACDEFGHIKLMNPQRSTVWY" in API_MD and "acdefghiklmnpqrstvwy" in API_MD
    assert "`B J O U X Z`" in API_MD
    assert "`#x`" in API_MD and "`##`" in API_MD
    assert "lowercase it before interleaving" in API_MD
    assert '[{"error": "..."}]' in API_MD and "validation error for Input" in API_MD
    assert "non-zero exit" not in API_MD
    assert "readiness probe" in API_MD
