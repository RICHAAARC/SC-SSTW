"""Synthetic CPU and static package checks only; no research tensor or model."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from main.tube_state import local_joint_positive_projection_margin_v1 as m
from main.tube_state.grow_video_reference import read_latent_bits
from runtime.wan import local_joint_positive_projection_margin_v1 as runtime
from runtime.wan.local_joint_positive_projection_margin_quality_v1 import distortion
from scripts.build_local_joint_positive_projection_margin_v1_notebook import build, ROOT, SOURCE_FILES, STEM

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


def test_geometry_and_coupled_projection_kkt():
    public = m.public_codebook("watermark")
    assert set(public) == {"key", "domain", "directions", "selected_basis", "permutations", "masks"}
    for roi in public["directions"]:
        flat = roi.reshape(54, -1)
        assert torch.allclose(flat@flat.T, torch.eye(54, dtype=torch.float64), atol=1e-12)
    host = torch.randn(m.LATENT_SHAPE, generator=torch.Generator().manual_seed(47))*0.2
    raw = m.read_local(host, "watermark", public)
    bits = [i % 2 for i in range(32)]
    candidate, receipt = m.write_projection(host, raw, bits, public)
    actual = m.read_local(candidate, "watermark", public)
    evaluation = m.evaluate_local(actual, list(m.STATES), bits)
    assert evaluation["exact_states"] == 22 and evaluation["local_decoded_correct"] == 704
    assert evaluation["payload"]["bit_errors"] == 0
    assert evaluation["min_state_gap"] > 0 and evaluation["min_payload_margin"] > 0
    assert receipt["actual_fp32"]["global_delta"]["l2"] > 0.02
    assert receipt["actual_fp32"]["outside_support"]["linf"] == 0
    assert receipt["actual_fp32"]["legacy_channels_0_3"]["linf"] == 0
    # Independent primal/dual/stationarity checks, not a second solver.
    for k in m.STATES:
        a = torch.tensor(raw["pooled_scores"][k][:22], dtype=torch.float64)
        correction = torch.tensor(receipt["rows"][k]["state_score_correction"], dtype=torch.float64)
        projected = a+correction
        dual = -correction.clone()
        dual[k] = 0
        assert bool((dual >= -1e-14).all())
        assert abs(float(correction[k]-dual.sum())) < 1e-12
        for j in m.STATES:
            if j != k:
                slack = projected[k]-projected[j]-m.GAMMA_S
                assert float(slack) > -1e-12
                assert abs(float(dual[j]*slack)) < 1e-12
    actual_squared = receipt["actual_fp32"]["global_delta"]["l2"]**2
    assert abs(actual_squared-receipt["continuous_distance_squared_display"])/actual_squared < 1e-6


def test_receiver_decodes_inferred_state_not_window_index_or_known_message():
    public = m.public_codebook("watermark")
    zero = torch.zeros(m.LATENT_SHAPE)
    bits = [int(i % 3 == 0) for i in range(32)]
    candidate, _ = m.write_projection(zero, m.read_local(zero, "watermark", public), bits, public)
    permutation = list(reversed(m.STATES))
    moved = candidate.clone()
    for pos, original in enumerate(permutation):
        moved[:, :, 1+2*pos:3+2*pos] = candidate[:, :, 1+2*original:3+2*original]
    raw = m.read_local(moved, "watermark", public)
    assert list(inspect.signature(m.read_local).parameters) == ["received", "key", "public"]
    assert [r["label"] for r in raw["states"]] == permutation
    assert [r["decoded"] for r in raw["payload"]] == bits
    evaluated = m.evaluate_local(raw, permutation, bits)
    assert evaluated["local_decoded_correct"] == 704
    wrong = m.read_local(moved, "watermark-wrong", m.public_codebook("watermark-wrong"))
    assert [r["label"] for r in wrong["states"]] != permutation
    tied = m.read_local(zero, "watermark", public)
    assert all(r["decoded"] is None and r["support_count"] == 0 for r in tied["payload"])
    missing = m.evaluate_local(dict(status="FAILED"), list(m.STATES), bits)
    assert len(missing["local_decoded_bits"]) == len(missing["chips"]) == 704
    assert sum(len(s["competitors"]) for s in missing["states"]) == missing["state_competitor_denominator"] == 462
    assert all(r["status"] == "FAILED" for r in missing["local_decoded_bits"])
    assert missing["exact_states"] is None and missing["unresolved_states"] is None
    assert missing["payload"]["bit_errors"] is None and missing["payload"]["unavailable_bits"] == 32


def test_shared_legacy_ties_and_prequantized_bytes(monkeypatch, tmp_path):
    # Each time has constant FFT-real sign, alternating from negative: exact ties.
    value = torch.zeros(m.LATENT_SHAPE)
    value[0, :, :, 0, 0] = torch.tensor([-1.0, 1.0]*23)
    raw = read_latent_bits(value, "watermark")
    assert raw["decoded_bits"] == [0]*32
    assert all(v["tie"] and v["first_vote"] == 0 and v["ones"] == v["zeros"] for v in raw["votes"])
    from runtime.wan.io import encode_rgb8
    pixels = torch.arange(12, dtype=torch.uint8).reshape(1, 2, 2, 3)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    encode_rgb8(pixels, tmp_path/"synthetic.mp4", fps=8, crf=18)
    assert len(calls) == 1 and calls[0][1]["input"] == pixels.numpy().tobytes()
    with pytest.raises(ValueError, match="uint8"):
        encode_rgb8(pixels.float(), tmp_path/"not-called.mp4", fps=8, crf=18)


def test_quality_preclip_temporal_and_outside_nominal_support():
    base = torch.zeros((3, 320, 512, 3))
    changed = base.clone()
    changed[0] = 2.0  # preclip may exceed 1; entire special first frame is outside.
    changed[1, 64:128, 96:160] = 1.0
    row = distortion(base, changed, meaning="synthetic paired increment")
    assert row["frame_denominator"] == 3 and row["adjacent_denominator"] == 2
    assert row["frames"][0]["outside_nominal_support"]["rms"] == 2
    assert row["frames"][1]["outside_nominal_support"]["linf"] == 0
    assert row["adjacent_residual"]["linf"] == 2
    assert row["quality_pass"] is None


class StubBackend:
    device = "cpu"
    blocks = {}
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []
    def note(self, name):
        self.calls.append(name)
        if self.fail == name:
            self.fail = None  # one failed operation; not a retry of that operation.
            raise RuntimeError("synthetic failure "+name)
    def load_terminal(self, path):
        self.note("source")
        return torch.zeros(m.LATENT_SHAPE)
    def load(self):
        self.note("load")
        return dict(synthetic=True, device="cpu")
    def decode(self, feature):
        self.note("decode")
        self.feature = feature
        rgb = torch.tensor(0.5).expand(m.VIDEO_SHAPE)
        return dict(preclip=rgb, float_rgb=rgb)
    def encode(self, rgb):
        self.note("encode")
        return self.feature.clone()
    def quantize(self, rgb):
        self.note("quantize")
        return torch.tensor(128, dtype=torch.uint8).expand(m.VIDEO_SHAPE)
    def save_mp4(self, pixels, path):
        self.note("save")
        self.pixels = pixels
        return dict(synthetic=True)
    def probe(self, path):
        self.note("probe")
        return dict(width=512, height=320)
    def read_mp4(self, path, probe):
        self.note("read")
        return self.pixels


@pytest.fixture
def cheap_storage(monkeypatch):
    # Dispatch test never writes synthetic 1.66GiB RGB; quality tested separately.
    monkeypatch.setattr(torch, "save", lambda value, path: Path(path).write_bytes(b"synthetic fixture only"))
    monkeypatch.setattr(runtime, "distortion", lambda *args, **kwargs: dict(status="SCORED", frame_denominator=181, adjacent_denominator=180, synthetic=True))


@pytest.mark.parametrize("force_shortfall", [False, True])
def test_fixed_dispatch_blind_before_truth_no_repeat(tmp_path, monkeypatch, cheap_storage, force_shortfall):
    if force_shortfall:
        monkeypatch.setattr(m, "write_projection", lambda base, *args: (base.clone(), dict(status="CONSTRUCTED", synthetic_shortfall=True)))
    original = runtime.evaluate_all
    def checked(raw):
        assert len(list((tmp_path/"blind").rglob("*.json"))) == 32
        return original(raw)
    monkeypatch.setattr(runtime, "evaluate_all", checked)
    backend = StubBackend()
    result = runtime.run("synthetic-only", tmp_path, backend=backend)
    assert result["status"] == "EXECUTION_COMPLETE", result.get("fatal_error")
    expected = dict(terminal_load=1, construct=1, public_codebook=2, vae_load=1,
                    ordinary_decode=2, posterior_encode=6, rgb8_quantize=2, mp4_save=2,
                    ffprobe=2, mp4_decode=2, joint_read=16, payload_fft_read=16,
                    processing_distortion=6, paired_quality=4)
    assert {k:v["returned"] for k,v in result["counts"].items()} == expected
    assert sum(v["returned"] for v in result["counts"].values()) == 63
    assert sum(len(r["local"]["chips"]) for r in result["evaluation"].values()) == 11264
    assert sum(len(r["local"]["local_decoded_bits"]) for r in result["evaluation"].values()) == 11264
    local = result["evaluation"]["PROJECTION_MARGIN/mp4/CORRECT"]["local"]
    assert local["exact_states"] == (0 if force_shortfall else 22)
    count = len(backend.calls)
    assert runtime.run("must-not-load", tmp_path, backend=backend)["status"] == "EXISTING_ATTEMPT_NOT_REEXECUTED"
    assert len(backend.calls) == count


@pytest.mark.parametrize("failure", ["source", "encode"])
def test_failure_denominators_and_independent_remaining_calls(tmp_path, cheap_storage, failure):
    backend = StubBackend(failure)
    result = runtime.run("synthetic-only", tmp_path, backend=backend)
    assert result["status"] == "INCOMPLETE"
    assert result["fixed_denominator"]["local_payload_chips"] == 11264
    assert len(result["evaluation"]) == 16
    assert sum(len(row["local"]["states"]) for row in result["evaluation"].values()) == 352
    assert sum(len(row["legacy"]["rows"]) for row in result["evaluation"].values()) == 512
    if failure == "source":
        assert backend.calls == ["source"]
        assert result["counts"]["vae_load"]["started"] == 0
        assert len(result["quality"]["paired/mp4"]["frames"]) == 181
        assert len(result["quality"]["paired/mp4"]["adjacent"]) == 180
    else:
        assert result["counts"]["posterior_encode"]["started"] == 6
        assert result["counts"]["posterior_encode"]["returned"] == 5
        assert result["evaluation"]["PAYLOAD_ONLY/float_rgb/CORRECT"]["local"]["status"] == "NOT_ENTERED"
        assert result["evaluation"]["PROJECTION_MARGIN/mp4/CORRECT"]["local"]["exact_states"] == 22


def test_notebook_bundle_and_markerless_read_only(tmp_path, monkeypatch):
    import nbformat
    notebook_path, archive_path = build()
    notebook = json.loads(notebook_path.read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert codes[0] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for code in codes:
        ast.parse(code)
    assert "is_relative_to" in codes[1] and "S_ISLNK" in codes[1]
    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == sorted(SOURCE_FILES)
        for name in SOURCE_FILES:
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            assert archive.read(name) == (ROOT/name).read_bytes()
        archive.extractall(tmp_path/"editable-source")
    check = subprocess.run([sys.executable, "-m", "experiments.wan_state_clock."+STEM+"_run", "--help"],
                           cwd=tmp_path/"editable-source", capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    for residue in ("construction.json.tmp", "media/partial.mp4", "blind/incomplete.json.tmp"):
        output = tmp_path/residue.replace("/", "_")
        marker = output/residue
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"incomplete saved scientific work")
        backend = StubBackend()
        assert runtime.run("never-read", output, backend=backend)["status"] == "EXISTING_ATTEMPT_WITHOUT_CHECKPOINT"
        assert backend.calls == []
        assert marker.read_bytes() == b"incomplete saved scientific work"
    scope = {}
    output_path = str(output)
    cell = codes[1].replace("/content/drive/MyDrive/Video-WM/B-Positive-Projection-Margin-V1/fixed-clean-pair-v1", output_path)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *args: pytest.fail("existing work must skip downloads"))
    exec(compile(cell, "existing-notebook-cell", "exec"), scope)
    assert scope["EXISTING_ARTIFACTS"]
    assert "SOURCE_ROOT" not in scope
