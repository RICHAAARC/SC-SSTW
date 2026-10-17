"""CPU-only discriminating tests; never load a model or a saved research tensor."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

pytestmark = pytest.mark.unit

from main.tube_state import local_joint_staged_native_state_v1 as m
from runtime.wan import local_joint_staged_native_state_v1 as runtime
from scripts.build_local_joint_staged_native_state_v1_notebook import build, SOURCE_FILES, ROOT, STEM

torch.set_num_threads(1)


def test_local_write_actual_cap_and_payload_invariance():
    base = torch.randn(m.LATENT_SHAPE, generator=torch.Generator().manual_seed(431))*3
    marked, receipt = m.write_state(base, "watermark")
    assert 0.0198 < receipt["actual_fp32_l2"] <= 0.02
    assert receipt["support_coordinates"] == 135168
    assert receipt["outside_support_max"] == receipt["payload_channel_max"] == 0
    assert torch.equal(base[:, :4], marked[:, :4])
    for r in range(4):
        book = m.codebook("watermark")[r]
        assert torch.allclose(book @ book.T, torch.eye(22, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(book.mean(dim=1), torch.zeros(22, dtype=torch.float64), atol=1e-12)


def test_blind_reader_follows_observation_permutation_not_index_or_truth():
    marked, _ = m.write_state(torch.zeros(m.LATENT_SHAPE), "watermark")
    permutation = list(reversed(range(22)))
    shuffled = marked.clone()
    for p, original in enumerate(permutation):
        shuffled[:, :, 1+2*p:3+2*p] = marked[:, :, 1+2*original:3+2*original]
    raw = m.read_state(shuffled, "watermark")
    assert [d["label"] for d in raw["decisions"]] == permutation
    assert m.evaluate_state(raw, permutation)["exact_count"] == 22
    assert m.evaluate_state(raw, list(range(22)))["exact_count"] == 0
    assert list(inspect.signature(m.read_state).parameters) == ["received", "key"]
    wrong = m.read_state(shuffled, "watermark-wrong")
    assert [d["label"] for d in wrong["decisions"]] != permutation
    empty = m.read_state(torch.zeros(m.LATENT_SHAPE), "watermark")
    assert all(d["status"] == "UNRESOLVED" and d["label"] is None for d in empty["decisions"])
    assert m.support_mapping()[0]["nominal_video_half_open"] == [1, 9]
    assert m.support_mapping()[-1]["nominal_video_half_open"] == [169, 177]


def test_invalid_and_fp32_lost_additions_fail_without_adjustment():
    with pytest.raises(ValueError, match="finite"):
        m.write_state(torch.full(m.LATENT_SHAPE, float("nan")), "watermark")
    with pytest.raises(ValueError, match="rounding"):
        m.write_state(torch.full(m.LATENT_SHAPE, 1e8), "watermark")
    with pytest.raises(m.ConstructionError, match="cap") as failure:
        m.write_state(torch.full(m.LATENT_SHAPE, 768.0), "watermark")
    assert failure.value.receipt["actual_fp32_l2"] > 0.02


class StubBackend:
    """Identity feature roundtrip with broadcast synthetic video; no model/codec."""
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.feature = None

    def note(self, name):
        self.calls.append(name)
        if self.fail == name:
            raise RuntimeError(f"synthetic failure: {name}")

    def load_terminal(self, path):
        self.note("source")
        return torch.zeros(m.LATENT_SHAPE)

    def load(self):
        self.note("load")
        return dict(device="cpu", synthetic=True)

    def decode(self, value):
        self.note("decode")
        self.feature = value
        return torch.tensor(0.5+float(value.abs().sum())*1e-5).expand(m.VIDEO_SHAPE)

    def encode(self, value):
        self.note("encode")
        return self.feature.clone()

    def save_mp4(self, rgb, path):
        self.note("save")
        self.rgb = rgb
        return dict(synthetic=True)

    def probe(self, path):
        self.note("probe")
        return dict(width=512, height=320, synthetic=True)

    def read_mp4(self, path, probe):
        self.note("read")
        return self.rgb


def test_stub_budget_raw_before_truth_and_no_repeat(tmp_path, monkeypatch):
    backend = StubBackend()
    original = m.evaluate_state
    def checked_evaluate(raw, targets):
        assert len(list((tmp_path/"blind").rglob("*.json"))) == 24
        return original(raw, targets)
    monkeypatch.setattr(m, "evaluate_state", checked_evaluate)
    result = runtime.run("synthetic-only", tmp_path, backend=backend)
    assert result["status"] == "EXECUTION_COMPLETE"
    expected = dict(terminal_load=1, construct=1, vae_load=1, ordinary_decode=2,
                    posterior_encode=4, mp4_save=2, ffprobe=2, mp4_decode=2,
                    state_read=12, payload_fft_read=12, paired_quality=2)
    assert {kind:v["returned"] for kind,v in result["counts"].items()} == expected
    assert len(result["evaluation"]) == 12
    assert result["evaluation"]["PAYLOAD_STATE/mp4/CORRECT"]["state"]["exact_count"] == 22
    assert result["quality"]["float_rgb"]["rmse"] > 0
    assert len(result["quality"]["mp4"]["temporal"]) == 180
    count = len(backend.calls)
    again = runtime.run("another-path-is-not-used", tmp_path, backend=backend)
    assert again["status"] == "EXISTING_ATTEMPT_NOT_REEXECUTED"
    assert len(backend.calls) == count


@pytest.mark.parametrize("failure,read_count", [("source", 0), ("load", 4), ("decode", 4), ("probe", 8)])
def test_fixed_denominators_survive_failure(tmp_path, failure, read_count):
    backend = StubBackend(fail=failure)
    result = runtime.run("synthetic-only", tmp_path, backend=backend)
    assert result["status"] == "INCOMPLETE"
    assert result["fixed_denominator"]["state_positions"] == 264
    assert len(result["evaluation"]) == 12
    assert sum(len(row["state"]["rows"]) for row in result["evaluation"].values()) == 264
    assert sum(len(row["payload"]["rows"]) for row in result["evaluation"].values()) == 384
    assert result["counts"]["state_read"]["returned"] == read_count
    runtime.run("synthetic-only", tmp_path, backend=backend)
    assert backend.calls.count(failure) == (2 if failure in ("decode", "probe") else 1)


def test_existing_incomplete_attempt_is_read_only(tmp_path):
    (tmp_path/"attempt.json").write_text("{}")
    ledger = {"vae": {"status": "STARTED"}}
    (tmp_path/"ledger.json").write_text(json.dumps(ledger))
    backend = StubBackend()
    result = runtime.run("unused", tmp_path, backend=backend)
    assert result["status"] == "EXISTING_ATTEMPT_WITHOUT_CHECKPOINT"
    assert backend.calls == []
    assert json.loads((tmp_path/"ledger.json").read_text()) == ledger


def test_bundle_actual_notebook_and_editable_no_git_cli(tmp_path):
    notebook_path, archive_path = build()
    notebook = json.loads(notebook_path.read_text())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="code"]
    assert codes[0] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for text in codes:
        ast.parse(text)
    assert "is_relative_to" in codes[1] and "S_ISLNK" in codes[1]
    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == sorted(SOURCE_FILES)
        for name in SOURCE_FILES:
            assert archive.read(name) == (ROOT/name).read_bytes()
        archive.extractall(tmp_path)
    command = [sys.executable, "-m", "experiments.wan_state_clock.local_joint_staged_native_state_v1_run", "--help"]
    check = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert check.returncode == 0, check.stderr
    # Execute the real CLI with an absent source: no model calls, fixed failure report.
    config = dict(source_terminal=str(tmp_path/"missing.pt"), output=str(tmp_path/"failure-report"), model_id="never-loaded")
    config_path = tmp_path/"missing-source-config.json"
    config_path.write_text(json.dumps(config))
    check = subprocess.run(command[:-1]+["--config", str(config_path)], cwd=tmp_path, text=True, capture_output=True)
    assert check.returncode == 1, check.stdout+check.stderr
    report = json.loads((tmp_path/"failure-report/result.json").read_text())
    assert report["calls"]["source"]["status"] == "FAILED"
    assert report["counts"]["vae_load"]["started"] == 0


# Review repair checks are selected with -k review; the original ten are reused.
@pytest.fixture(scope="module")
def review_notebook():
    path, _ = build()
    return json.loads(path.read_text())


def test_review_markerless_records_never_restart_or_overwrite(tmp_path, monkeypatch):
    cases = [name+suffix for name in ("attempt.json", "result.json", "checkpoint.json", "ledger.json", "evaluation.json")
             for suffix in ("", ".tmp")]
    cases += ["blind/row.json.tmp", "features/PAYLOAD_ONLY/terminal.pt", "media/partial.mp4", "quality/float_rgb.json.tmp"]
    def forbidden(*args, **kwargs):
        raise AssertionError("must not create Journal for previous scientific work")
    monkeypatch.setattr(runtime, "Journal", forbidden)
    for i, name in enumerate(cases):
        folder = tmp_path/str(i)
        artifact = folder/name
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"unreadable prior data is still spent work")
        backend = StubBackend()
        result = runtime.run("unused", folder, backend=backend)
        assert result["status"].startswith("EXISTING_ATTEMPT")
        assert backend.calls == []
        assert artifact.read_bytes() == b"unreadable prior data is still spent work"
        assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()) == [name]
    setup = tmp_path/"setup-only"
    setup.mkdir()
    (setup/"setup.log").write_text("imports only")
    (setup/"companion_source.zip").write_bytes(b"source only")
    (setup/"features").mkdir()
    assert runtime.execution_artifacts(setup) == []


@pytest.mark.parametrize("missing_kind", ["state", "payload"])
def test_review_partial_pair_keeps_other_valid_measurement(missing_kind):
    import copy
    evaluated = {}
    for arm in m.ARMS:
        for layer in m.LAYERS:
            for key in m.KEYS:
                marked = arm=="PAYLOAD_STATE"
                evaluated[f"{arm}/{layer}/{key}"] = dict(
                    state=dict(status="EVALUATED", exact_count=22 if marked else 0,
                               rows=[dict(true_minus_max_wrong=1 if marked else -1, label=i if marked else None)
                                     for i in range(22)]),
                    payload=dict(status="EVALUATED", bit_errors=1 if marked else 0,
                                 rows=[dict(correct=not(marked and i==3)) for i in range(32)]))
    evaluated["PAYLOAD_STATE/mp4/CORRECT"][missing_kind] = dict(status="FAILED", error="synthetic")
    prior = copy.deepcopy(evaluated)
    pair = runtime.paired_readout_comparison(evaluated)["mp4/CORRECT"]
    assert pair["status"] == "PARTIAL"
    assert pair[missing_kind+"_status"] == "NOT_SCORED"
    assert pair["state_denominator"] == len(pair["state_rows"]) == 22
    assert pair["payload_denominator"] == len(pair["payload_rows"]) == 32
    if missing_kind == "state":
        assert pair["payload_status"] == "EVALUATED"
        assert pair["lost_correct_bits"] == [3]
        assert pair["gained_correct_bits"] == []
    else:
        assert pair["state_status"] == "EVALUATED"
        assert pair["marked_exact_states"] == 22
        assert pair["state_rows"][7]["marked_label"] == 7
    assert evaluated == prior


def test_review_real_notebook_existing_offline_read_only(tmp_path, monkeypatch, capsys, review_notebook):
    import urllib.request
    def forbidden(*args, **kwargs):
        raise AssertionError("offline result reopening must not touch network or subprocesses")
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    codes = ["".join(c["source"]) for c in review_notebook["cells"] if c["cell_type"]=="code"]
    cases = [("result.json", {"status": "SAVED_TEST_RESULT"}),
             ("ledger.json", {"vae": {"status": "STARTED"}}),
             ("evaluation.json.tmp", {}), ("checkpoint.json.tmp", {"status": "UNFINISHED_TEST"}),
             ("features/PAYLOAD_ONLY/terminal.pt.tmp", None)]
    for i, (name, data) in enumerate(cases):
        folder = tmp_path/str(i)
        path = folder/name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data) if data is not None else "not read as a tensor")
        namespace = {}
        for cell in codes[1:]:
            localized = cell.replace("/content/drive/MyDrive/Video-WM/B-Staged-Native-State-V1/fixed-clean-pair-v1", str(folder))
            exec(compile(localized, "actual-notebook-cell", "exec"), namespace)
        assert "SOURCE_ROOT" not in namespace
        assert len([p for p in folder.rglob("*") if p.is_file()]) == 1
        assert namespace["EXISTING_ARTIFACTS"] == [name]
    shown = capsys.readouterr().out
    assert "SAVED_TEST_RESULT" in shown and "STARTED" in shown and "UNFINISHED_TEST" in shown


def _review_function(notebook, name, namespace):
    code = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"]=="code" and f"def {name}(" in "".join(cell["source"]))
    node = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-notebook-helper", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("case", ["already_usable", "missing", "known_conflict", "unknown", "repeated_conflict"])
def test_review_targeted_import_probe_only_repairs_observed_fault(tmp_path, review_notebook, case):
    log = tmp_path/"setup.log"
    calls, probes = [], []
    conflict = "ImportError: cannot import name 'FqnToConfig' from 'torchao.quantization'"
    failures = {"missing": "ModuleNotFoundError: No module named 'huggingface_hub'",
                "known_conflict": conflict, "repeated_conflict": conflict, "unknown": "unrelated import failure"}
    def command(args, label, check=True):
        calls.append(args)
        if args[1] == "-c":
            probes.append(args[2])
            if case != "already_usable" and (len(probes)==1 or case=="repeated_conflict"):
                with log.open("a") as stream:
                    stream.write(failures[case]+"\n")
                return 1
        return 0
    namespace = dict(LOG=log, command=command, sys=sys)
    ensure = _review_function(review_notebook, "ensure_imports", namespace)
    probe = "import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan"
    if case in ("unknown", "repeated_conflict"):
        with pytest.raises(RuntimeError):
            ensure(probe, ["diffusers", "accelerate", "safetensors", "huggingface-hub"], "VAE")
    else:
        ensure(probe, ["diffusers", "accelerate", "safetensors", "huggingface-hub"], "VAE")
    repairs = [args[3:] for args in calls if args[1:3]==["-m", "pip"]]
    expected = [] if case in ("already_usable", "unknown") else (
        [["install", "huggingface-hub"]] if case=="missing" else [["uninstall", "-y", "torchao"]])
    assert repairs == expected
    assert all("WanPipeline" not in value and "AutoencoderKLWan" in value for value in probes)


def test_review_vae_cache_priority_and_single_selected_load(tmp_path, monkeypatch):
    from types import SimpleNamespace
    def cached(path):
        (path/"vae").mkdir(parents=True, exist_ok=True)
        (path/"vae/config.json").write_text("{}")
        (path/"vae/diffusion_pytorch_model.safetensors").write_bytes(b"stub only")
    for label, candidate in (("available", "available"), ("root", "."), ("child", "older-location")):
        base = tmp_path/label
        cached(base/candidate)
        if label=="available":
            cached(base)
            cached(base/"another-child")
        selected = runtime.select_vae_source(runtime.DEFAULT_MODEL_ID, base)
        assert Path(selected["location"]) == base/candidate
        assert selected["origin"] == "shared_drive_cache"
    selected = runtime.select_vae_source(runtime.DEFAULT_MODEL_ID, tmp_path/"absent")
    assert selected["location"] == runtime.DEFAULT_MODEL_ID and selected["weights_may_download"]
    selected = runtime.select_vae_source(str(tmp_path/"root"))
    assert selected["origin"] == "explicit_local"
    monkeypatch.setattr(runtime, "select_vae_source", lambda model_id: selected)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    class FakeModel:
        def to(self, device): return self
        def eval(self): return self
        def requires_grad_(self, value):
            assert value is False
            return self
    def load(location, **kwargs):
        calls.append((location, kwargs))
        return FakeModel()
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(AutoencoderKLWan=SimpleNamespace(from_pretrained=load)))
    receipt = runtime.WanBackend(runtime.DEFAULT_MODEL_ID).load()
    assert len(calls)==1 and calls[0][0]==selected["location"]
    assert calls[0][1]["subfolder"]=="vae" and receipt["model_selection"]==selected
    def fail(location, **kwargs):
        calls.append((location, kwargs))
        raise RuntimeError("synthetic unreadable weights")
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(AutoencoderKLWan=SimpleNamespace(from_pretrained=fail)))
    with pytest.raises(RuntimeError) as caught:
        runtime.WanBackend(runtime.DEFAULT_MODEL_ID).load()
    assert len(calls)==2 and caught.value.receipt["model_selection"]==selected


def test_review_media_preparation_only_when_missing(monkeypatch, review_notebook):
    import shutil
    calls = []
    ready = True
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/"+name if ready else None)
    def command(args, label, check=True):
        nonlocal ready
        calls.append(args)
        if "install" in args:
            ready = True
        return 0
    ensure = _review_function(review_notebook, "ensure_media_tools", dict(command=command))
    ensure()
    assert calls == []
    ready = False
    ensure()
    assert calls == [["apt-get", "-qq", "update"], ["apt-get", "-qq", "install", "-y", "ffmpeg"]]


def test_review_rebuilt_notebook_schema_and_package_sync(review_notebook):
    import nbformat
    nbformat.validate(nbformat.from_dict(review_notebook))
    assert review_notebook["metadata"]["accelerator"]=="GPU"
    codes = ["".join(c["source"]) for c in review_notebook["cells"] if c["cell_type"]=="code"]
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert codes[1].index("OUTPUT =") < codes[1].index("urlretrieve")
    for code in codes:
        ast.parse(code)
    with zipfile.ZipFile(ROOT/"notebooks"/f"{STEM}_portable_source.zip") as package:
        assert sorted(package.namelist())==sorted(SOURCE_FILES)
        for name in SOURCE_FILES:
            assert package.read(name)==(ROOT/name).read_bytes()
    # Only narrow B helpers are embedded, with no A worktree/build imports.
    assert "from experiments.paper_results_v1" not in (ROOT/"scripts"/f"build_{STEM}_notebook.py").read_text()
