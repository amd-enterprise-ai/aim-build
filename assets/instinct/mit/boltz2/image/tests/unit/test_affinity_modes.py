# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The mode switch (BOLTZ2_MODE), the affinity request checks and the weights each mode needs."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import runner
import service as service_mod
import torch
import warmup
from bentoml.exceptions import InvalidArgument

MINIMAL = {"sequences": [{"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}}]}
COMPLEX = {
    "sequences": [
        {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
        {"ligand": {"id": "B", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}},
    ],
    "properties": [{"affinity": {"binder": "B"}}],
}


def _request(**overrides):
    return service_mod.Boltz2Request(**{**COMPLEX, **overrides})


def test_the_default_mode_is_structure():
    assert service_mod.ServiceConfig().mode == "structure"


@pytest.mark.parametrize("mode", ["structure", "affinity", " Affinity "])
def test_a_known_mode_is_accepted(mode):
    assert service_mod.ServiceConfig(BOLTZ2_MODE=mode).mode == mode.strip().lower()


@pytest.mark.parametrize("mode", ["affinty", "both", "affinity-only", ""])
def test_an_unknown_mode_stops_the_start_up_and_lists_the_valid_values(mode):
    with pytest.raises(ValueError, match=r"BOLTZ2_MODE must be one of \['structure', 'affinity'\]"):
        service_mod.ServiceConfig(BOLTZ2_MODE=mode)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mode", "files", "present"),
    [
        ("structure", ["boltz2_conf.ckpt"], True),
        ("structure", ["boltz2_aff.ckpt"], False),
        ("affinity", ["boltz2_conf.ckpt"], False),
        ("affinity", ["boltz2_conf.ckpt", "boltz2_aff.ckpt"], True),
    ],
)
def test_assets_present_asks_only_for_the_checkpoints_of_the_mode(tmp_path, mode, files, present):
    (tmp_path / "mols").mkdir()
    for name in files:
        (tmp_path / name).write_text("x")
    assert runner.assets_present(tmp_path, mode) is present


def _fake_downloads(monkeypatch, tmp_path):
    """Replace upstream's URLs and ``urlretrieve``; return the list of fetched URLs."""
    import sys
    import tarfile
    import types
    import urllib.request

    main = types.ModuleType("boltz.main")
    main.MOL_URL = "https://example.test/mols.tar"
    main.BOLTZ2_URL_WITH_FALLBACK = ["https://example.test/conf.ckpt"]
    main.BOLTZ2_AFFINITY_URL_WITH_FALLBACK = ["https://example.test/aff.ckpt"]
    monkeypatch.setitem(sys.modules, "boltz", types.ModuleType("boltz"))
    monkeypatch.setitem(sys.modules, "boltz.main", main)

    source = tmp_path / "source"
    (source / "mols").mkdir(parents=True)
    (source / "mols" / "ALA.pkl").write_text("x")
    fetched = []

    def urlretrieve(url, filename):
        fetched.append(url)
        if url.endswith("mols.tar"):
            with tarfile.open(filename, "w") as tar:
                tar.add(source / "mols", arcname="mols")
        else:
            open(filename, "w").write("x")

    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)
    return fetched


@pytest.mark.parametrize(
    ("mode", "checkpoints"),
    [
        ("structure", ["boltz2_conf.ckpt"]),
        ("affinity", ["boltz2_conf.ckpt", "boltz2_aff.ckpt"]),
    ],
)
def test_each_mode_downloads_only_the_checkpoints_it_loads(tmp_path, monkeypatch, mode, checkpoints):
    fetched = _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    runner.ensure_boltz2_assets(cache, mode)
    assert sorted(p.name for p in cache.glob("*.ckpt")) == sorted(checkpoints)
    assert (cache / "mols").is_dir()
    assert len(fetched) == 1 + len(checkpoints)
    assert runner.assets_present(cache, mode)


def test_a_staged_cache_without_the_mols_archive_fetches_no_archive(tmp_path, monkeypatch):
    fetched = _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    (cache / "mols").mkdir(parents=True)
    runner.ensure_boltz2_assets(cache, "structure")
    assert fetched == ["https://example.test/conf.ckpt"]
    assert not (cache / "mols.tar").exists()


def _staged_archive(cache, add_members):
    import io
    import tarfile

    cache.mkdir()
    with tarfile.open(cache / "mols.tar", "w") as tar:
        info = tarfile.TarInfo("mols/ALA.pkl")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
        add_members(tar)


def test_the_mols_archive_extracts_its_regular_members(tmp_path, monkeypatch):
    _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    _staged_archive(cache, lambda tar: None)
    runner.ensure_boltz2_assets(cache, "structure")
    assert (cache / "mols" / "ALA.pkl").read_bytes() == b"x"
    assert not list(cache.glob(".mols-*"))


def _traversal(tar):
    import io
    import tarfile

    info = tarfile.TarInfo("../outside.txt")
    info.size = 1
    tar.addfile(info, io.BytesIO(b"x"))


def _outside_symlink(tar):
    import tarfile

    info = tarfile.TarInfo("mols/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../outside.txt"
    tar.addfile(info)


@pytest.mark.parametrize("add_members", [_traversal, _outside_symlink])
def test_a_mols_archive_member_outside_the_cache_is_refused(tmp_path, monkeypatch, add_members):
    import tarfile

    _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    _staged_archive(cache, add_members)
    with pytest.raises(tarfile.FilterError):
        runner.ensure_boltz2_assets(cache, "structure")
    assert not (tmp_path / "outside.txt").exists()
    assert not (cache / "mols").exists()
    assert not list(cache.glob(".mols-*"))

    _staged_archive(tmp_path / "cache2", lambda tar: None)
    (cache / "mols.tar").write_bytes((tmp_path / "cache2" / "mols.tar").read_bytes())
    runner.ensure_boltz2_assets(cache, "structure")
    assert (cache / "mols" / "ALA.pkl").read_bytes() == b"x"
    assert not list(cache.glob(".mols-*"))


def test_a_stale_staging_directory_is_removed_by_the_next_extraction(tmp_path, monkeypatch):
    _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    _staged_archive(cache, lambda tar: None)
    (cache / ".mols-stale").mkdir()
    (cache / ".mols-stale" / "part").write_text("x")
    runner.ensure_boltz2_assets(cache, "structure")
    assert not list(cache.glob(".mols-*"))


def test_an_archive_without_a_mols_directory_is_refused(tmp_path, monkeypatch):
    import io
    import tarfile

    _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    with tarfile.open(cache / "mols.tar", "w") as tar:
        info = tarfile.TarInfo("other/ALA.pkl")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError, match="no 'mols/' directory"):
        runner.ensure_boltz2_assets(cache, "structure")
    assert not (cache / "mols").exists()
    assert not list(cache.glob(".mols-*"))


def test_an_absolute_mols_archive_member_stays_inside_the_cache(tmp_path, monkeypatch):
    import io
    import tarfile

    def absolute(tar):
        info = tarfile.TarInfo(str(tmp_path / "outside.txt"))
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    _staged_archive(cache, absolute)
    runner.ensure_boltz2_assets(cache, "structure")
    assert not (tmp_path / "outside.txt").exists()


def test_a_second_download_run_fetches_nothing(tmp_path, monkeypatch):
    fetched = _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    runner.ensure_boltz2_assets(cache, "affinity")
    fetched.clear()
    runner.ensure_boltz2_assets(cache, "affinity")
    assert fetched == []


def test_a_structure_cache_gets_the_affinity_checkpoint_when_the_mode_changes(tmp_path, monkeypatch):
    fetched = _fake_downloads(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    runner.ensure_boltz2_assets(cache, "structure")
    fetched.clear()
    runner.ensure_boltz2_assets(cache, "affinity")
    assert fetched == ["https://example.test/aff.ckpt"]


# --------------------------------------------------------------------------- #
# Request checks
# --------------------------------------------------------------------------- #


def test_structure_mode_refuses_an_affinity_property(monkeypatch):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "structure")
    with pytest.raises(InvalidArgument, match="affinity is not served by this profile"):
        service_mod.Boltz2Prediction.inner._check_request(_request())  # noqa: SLF001


def test_structure_mode_accepts_a_request_without_an_affinity_property(monkeypatch):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "structure")
    service_mod.Boltz2Prediction.inner._check_request(service_mod.Boltz2Request(**MINIMAL))  # noqa: SLF001


def test_affinity_mode_accepts_one_binder_ligand(monkeypatch):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "affinity")
    service_mod.Boltz2Prediction.inner._check_request(_request())  # noqa: SLF001


@pytest.mark.parametrize(
    "overrides",
    [
        {"properties": [{"affinity": {"binder": "A"}}]},
        {"properties": [{"affinity": {"binder": "Z"}}]},
        {"properties": [{"affinity": {"binder": ["B"]}}]},
        {"properties": [{"affinity": {}}]},
        {"properties": [{"affinity": {"binder": "B"}}, {"affinity": {"binder": "B"}}]},
        {
            "sequences": [
                {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
                {"ligand": {"id": ["B", "C"], "smiles": "CCO"}},
            ]
        },
        {
            "sequences": [
                {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
                {"ligand": {"id": "B", "ccd": ["ATP", "MG"]}},
            ]
        },
        {
            "sequences": [
                {"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}},
                {"ligand": {"id": "B", "smiles": "CCO"}},
                {"ligand": {"id": "C", "smiles": "CCO"}},
            ]
        },
        {"properties": [{"Affinity": {"binder": "B"}}]},
    ],
)
def test_affinity_mode_refuses_a_request_without_exactly_one_binder_ligand(monkeypatch, overrides):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "affinity")
    with pytest.raises(InvalidArgument):
        service_mod.Boltz2Prediction.inner._check_request(_request(**overrides))  # noqa: SLF001


def test_a_binder_with_a_one_element_id_list_is_accepted():
    request = _request(sequences=[COMPLEX["sequences"][0], {"ligand": {"id": ["B"], "smiles": "CCO"}}])
    service_mod.check_affinity_request(request)


@pytest.mark.parametrize("smiles", ["", " ", "[H][H]"])
@pytest.mark.parametrize("rdkit", [True, False])
def test_a_binder_smiles_without_a_heavy_atom_is_refused(monkeypatch, smiles, rdkit):
    if rdkit:
        pytest.importorskip("rdkit")
    else:
        monkeypatch.setitem(sys.modules, "rdkit", None)
    request = _request(sequences=[COMPLEX["sequences"][0], {"ligand": {"id": "B", "smiles": smiles}}])
    with pytest.raises(InvalidArgument):
        service_mod.check_affinity_request(request)


@pytest.mark.parametrize("key", ["Affinity", "AFFINITY"])
def test_a_misspelled_affinity_key_gets_a_message_that_names_the_key(key):
    request = _request(properties=[{key: {"binder": "B"}}])
    with pytest.raises(InvalidArgument, match="written 'affinity'"):
        service_mod.check_affinity_request(request)


def test_an_empty_smiles_is_refused_through_the_request_check(monkeypatch):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "affinity")
    request = _request(sequences=[COMPLEX["sequences"][0], {"ligand": {"id": "B", "smiles": ""}}])
    with pytest.raises(InvalidArgument):
        service_mod.Boltz2Prediction.inner._check_request(request)  # noqa: SLF001


def test_a_ring_closure_across_a_dot_does_not_lower_the_atom_count():
    pytest.importorskip("rdkit")
    assert service_mod.largest_fragment_atoms("C1.C1") == 2


def test_the_atom_limit_counts_the_largest_fragment_of_a_salt():
    smiles = "C" * service_mod.MAX_AFFINITY_LIGAND_ATOMS + ".[Na+].[Cl-]"
    request = _request(sequences=[COMPLEX["sequences"][0], {"ligand": {"id": "B", "smiles": smiles}}])
    service_mod.check_affinity_request(request)


def test_structure_mode_refuses_the_affinity_key_in_any_case(monkeypatch):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "structure")
    with pytest.raises(InvalidArgument, match="affinity is not served by this profile"):
        service_mod.Boltz2Prediction.inner._check_request(  # noqa: SLF001
            _request(properties=[{"Affinity": {"binder": "B"}}])
        )


def test_affinity_mode_accepts_a_binder_given_by_one_ccd_code(monkeypatch):
    monkeypatch.setattr(service_mod.CONFIG, "mode", "affinity")
    request = _request(sequences=[COMPLEX["sequences"][0], {"ligand": {"id": "B", "ccd": "ATP"}}])
    service_mod.Boltz2Prediction.inner._check_request(request)  # noqa: SLF001


def test_a_binder_above_the_ligand_size_limit_is_a_422():
    smiles = "C" * (service_mod.MAX_AFFINITY_LIGAND_ATOMS + 1)
    request = _request(sequences=[COMPLEX["sequences"][0], {"ligand": {"id": "B", "smiles": smiles}}])
    with pytest.raises(service_mod.UnprocessableRequest):
        service_mod.check_affinity_request(request)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_steps_affinity": service_mod.MAX_SAMPLING_STEPS + 1},
        {"diffusion_samples_affinity": service_mod.MAX_DIFFUSION_SAMPLES_AFFINITY + 1},
    ],
)
def test_the_affinity_settings_above_a_limit_are_a_422(overrides):
    with pytest.raises(service_mod.UnprocessableRequest):
        service_mod.check_limits(_request(**overrides))


def test_the_affinity_defaults_are_those_of_boltz_predict():
    request = _request()
    assert (request.affinity_mw_correction, request.sampling_steps_affinity, request.diffusion_samples_affinity) == (
        False,
        200,
        5,
    )


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class _FakeContext:
    def __init__(self):
        self.response = type("Resp", (), {"status_code": 200, "headers": {}})()
        self.request = type("Req", (), {"scope": {}})()


def test_the_answer_carries_the_affinity_and_the_mode(tmp_path, monkeypatch):
    service = service_mod
    monkeypatch.setattr(service.CONFIG, "cache", tmp_path)
    monkeypatch.setattr(service.CONFIG, "download_assets", False)
    monkeypatch.setattr(service, "assets_present", lambda cache, *args: True)
    monkeypatch.setattr(service, "load_boltz2_model", lambda cache, **kwargs: type("FakeModel", (), {})())
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: type("FakeAffinity", (), {})())
    monkeypatch.setattr(service.CONFIG, "mode", "affinity")
    values = {"affinity_pred_value": 0.5, "affinity_probability_binary": 0.9}
    seen = {}

    def _run(**kwargs):
        seen.update(kwargs)
        return {
            "error": False,
            "structures": [{"record_id": "A", "format": "pdb", "content": "x"}],
            "confidence": {},
            "timing": {"runtime_s": 0.0, "affinity_s": 0.0},
            "affinity": values,
        }

    monkeypatch.setattr(service, "run_boltz2_prediction", _run)
    answer = service.Boltz2Prediction.inner().inference(_request(), _FakeContext())
    assert answer["affinity"] == values
    assert answer["runtime"]["mode"] == "affinity"
    assert seen["sampling_steps_affinity"] == 200


def test_a_result_with_an_error_is_a_400(tmp_path, monkeypatch):
    service = service_mod
    monkeypatch.setattr(service.CONFIG, "cache", tmp_path)
    monkeypatch.setattr(service.CONFIG, "download_assets", False)
    monkeypatch.setattr(service, "assets_present", lambda cache, *args: True)
    monkeypatch.setattr(service, "load_boltz2_model", lambda cache, **kwargs: type("FakeModel", (), {})())
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: type("FakeAffinity", (), {})())
    monkeypatch.setattr(service.CONFIG, "mode", "affinity")
    monkeypatch.setattr(
        service, "run_boltz2_prediction", lambda **kwargs: {"error": True, "message": "no affinity", "structures": []}
    )
    with pytest.raises(InvalidArgument, match="no affinity"):
        service.Boltz2Prediction.inner().inference(_request(), _FakeContext())


def test_the_affinity_file_of_the_binder_record_is_read(tmp_path):
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "affinity_rec.json").write_text('{"affinity_pred_value": 1.5}')
    assert runner._parse_affinity(tmp_path, ["rec"]) == {"affinity_pred_value": 1.5}  # noqa: SLF001
    assert runner._parse_affinity(tmp_path, ["other"]) == {}  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #


def test_the_affinity_warmup_request_is_a_valid_request():
    request = warmup.affinity_warmup_request()
    service_mod.check_affinity_request(
        service_mod.Boltz2Request(sequences=request["sequences"], properties=request["properties"])
    )


# --------------------------------------------------------------------------- #
# Triangle kernels in the affinity model
# --------------------------------------------------------------------------- #


class _Triangles(torch.nn.Module):
    """Stand-in for a module that gets ``use_kernels`` from its caller; records the value it sees."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, use_kernels=False):
        self.seen.append(use_kernels)
        return x


class _AffinityModel(torch.nn.Module):
    structure_module = None

    def __init__(self, use_kernels):
        super().__init__()
        self.use_kernels = use_kernels
        self.pairformer_module = _Triangles()
        self.affinity_module1 = _Triangles()
        self.affinity_module2 = _Triangles()


@pytest.fixture
def fake_boltz(monkeypatch):
    events = []
    fake_triangle = types.ModuleType("rocm_triangle")
    fake_triangle.prepare_model = lambda model: events.append("pack")
    fake_triangle.install = lambda mode: events.append(("install", mode))
    monkeypatch.setitem(sys.modules, "rocm_triangle", fake_triangle)

    fake_main = types.ModuleType("boltz.main")
    for name in ("Boltz2DiffusionParams", "BoltzSteeringParams", "MSAModuleArgs", "PairformerArgsV2"):
        setattr(fake_main, name, type(name, (), {"__init__": lambda self, *a, **k: None, "step_scale": 1.5}))
    fake_boltz2 = types.ModuleType("boltz.model.models.boltz2")

    class _Boltz2:
        @staticmethod
        def load_from_checkpoint(*args, **kwargs):
            events.append(("model", kwargs["use_kernels"]))
            return _AffinityModel(kwargs["use_kernels"])

    fake_boltz2.Boltz2 = _Boltz2
    for name in ("boltz", "boltz.model", "boltz.model.models"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "boltz.main", fake_main)
    monkeypatch.setitem(sys.modules, "boltz.model.models.boltz2", fake_boltz2)
    monkeypatch.setattr(runner, "asdict", lambda obj: {}, raising=False)
    return events


def test_the_affinity_trunk_takes_the_kernels_and_the_affinity_module_stays_eager(fake_boltz):
    model = runner.load_affinity_model(Path("/nonexistent"), optimizations=True)
    assert fake_boltz == [("install", "both"), ("model", True), "pack"]
    model.pairformer_module(1, use_kernels=model.use_kernels)
    model.affinity_module1(1, use_kernels=model.use_kernels)
    model.affinity_module2(1, use_kernels=model.use_kernels)
    assert model.pairformer_module.seen == [True]
    assert model.affinity_module1.seen == [False]
    assert model.affinity_module2.seen == [False]


def test_the_affinity_loader_installs_the_kernels_before_it_builds_the_model(fake_boltz):
    runner.load_affinity_model(Path("/nonexistent"), optimizations=True)
    assert fake_boltz.index(("install", "both")) < fake_boltz.index(("model", True))


def test_the_affinity_model_without_optimizations_has_no_kernels_and_no_wrapper(fake_boltz):
    model = runner.load_affinity_model(Path("/nonexistent"), optimizations=False)
    assert fake_boltz == [("model", False)]
    model.affinity_module1(1, use_kernels=True)
    assert model.affinity_module1.seen == [True]


# --------------------------------------------------------------------------- #
# BOLTZ2_AFFINITY_COMPILE
# --------------------------------------------------------------------------- #


def test_the_affinity_compile_switch_is_off_by_default():
    config = service_mod.ServiceConfig(BOLTZ2_MODE="affinity")
    assert config.affinity_compile == "off"
    assert config.affinity_compiled is False


@pytest.mark.parametrize("value", ["on", "ON", " On "])
def test_the_affinity_compile_switch_accepts_on_in_any_case(value):
    assert service_mod.ServiceConfig(BOLTZ2_MODE="affinity", BOLTZ2_AFFINITY_COMPILE=value).affinity_compiled is True


@pytest.mark.parametrize("value", ["1", "true", "yes", ""])
def test_a_wrong_affinity_compile_value_stops_the_start_up(value):
    with pytest.raises(ValueError, match="BOLTZ2_AFFINITY_COMPILE must be 'on' or 'off'"):
        service_mod.ServiceConfig(BOLTZ2_AFFINITY_COMPILE=value)


@pytest.mark.parametrize(
    ("mode", "optimizations", "active"),
    [("affinity", "on", True), ("affinity", "off", False), ("structure", "on", False), ("structure", "off", False)],
)
def test_the_affinity_compile_switch_acts_only_in_affinity_mode_with_optimizations(mode, optimizations, active):
    config = service_mod.ServiceConfig(
        BOLTZ2_MODE=mode, BOLTZ2_OPTIMIZATIONS=optimizations, BOLTZ2_AFFINITY_COMPILE="on"
    )
    assert config.affinity_compiled is active


def _compile_with_status(monkeypatch, events, callbacks, source="dynamo-callback"):
    """Stub ``install_compilation`` that attaches one status per selector with the given compile callbacks."""
    import compilation

    def install(model, config):
        events.append(("compile", config.transition, config.token_transformer))
        for label, count in callbacks.items():
            status = compilation.CompileStatus(mode=config.transition, compile_callbacks=count)
            status.compile_event_source = source
            compilation._status_map(model)[label] = status  # noqa: SLF001

    monkeypatch.setattr(compilation, "install_compilation", install)


def test_the_affinity_loader_does_not_compile_or_count_compiles_by_default(fake_boltz, monkeypatch):
    import compilation
    import score_graph

    monkeypatch.setattr(compilation, "install_compilation", lambda *a, **k: fake_boltz.append("compile"))
    monkeypatch.setattr(
        score_graph,
        "install_score_graph",
        lambda model, compile_count=None: fake_boltz.append(("graph", compile_count)),
    )
    runner.load_affinity_model(Path("/nonexistent"), optimizations=True, device=torch.device("cpu"))
    assert fake_boltz == [("install", "both"), ("model", True), "pack", ("graph", None)]


def test_the_affinity_loader_compiles_before_the_graph_and_counts_the_compiles(fake_boltz, monkeypatch):
    import score_graph

    _compile_with_status(monkeypatch, fake_boltz, {"transition": 2, "token_transformer": 3})
    monkeypatch.setattr(
        score_graph,
        "install_score_graph",
        lambda model, compile_count=None: fake_boltz.append(("graph", compile_count())),
    )
    runner.load_affinity_model(
        Path("/nonexistent"), optimizations=True, device=torch.device("cpu"), compile_affinity=True
    )
    assert fake_boltz == [
        ("install", "both"),
        ("model", True),
        "pack",
        ("compile", "inductor", "inductor"),
        ("graph", 5),
    ]


def test_the_affinity_compile_needs_the_optimizations(fake_boltz, monkeypatch):
    _compile_with_status(monkeypatch, fake_boltz, {"transition": 0})
    runner.load_affinity_model(
        Path("/nonexistent"), optimizations=False, device=torch.device("cpu"), compile_affinity=True
    )
    assert fake_boltz == [("model", False)]


def test_no_score_graph_without_compile_events(fake_boltz, monkeypatch, caplog):
    import score_graph

    _compile_with_status(monkeypatch, fake_boltz, {"transition": 0}, source="unavailable")
    monkeypatch.setattr(score_graph, "install_score_graph", lambda *a, **k: fake_boltz.append("graph"))
    with caplog.at_level("WARNING", logger="runner"):
        runner.load_affinity_model(
            Path("/nonexistent"), optimizations=True, device=torch.device("cpu"), compile_affinity=True
        )
    assert "graph" not in fake_boltz
    assert caplog.text.count("runs without a CUDA graph") == 1


def test_the_default_affinity_warmup_request_keeps_the_warmup_sampling_steps():
    assert warmup.affinity_warmup_request()["sampling_steps_affinity"] == warmup.WARMUP_SAMPLING_STEPS


def test_the_crop_warmup_request_is_a_valid_request_with_a_protein_longer_than_the_crop():
    request = warmup.affinity_warmup_request(
        warmup.AFFINITY_CROP_WARMUP_TOKENS,
        warmup.AFFINITY_CROP_WARMUP_SMILES,
        name="warmup_affinity_crop",
        diffusion_samples_affinity=1,
    )
    service_mod.check_affinity_request(
        service_mod.Boltz2Request(sequences=request["sequences"], properties=request["properties"])
    )
    assert len(request["sequences"][0]["protein"]["sequence"]) > 200
    assert request["diffusion_samples_affinity"] == 1
    assert "diffusion_samples_affinity" not in warmup.affinity_warmup_request()
