"""Card54 targeted synthetic CPU/static checks; no saved research data or model."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import typing
import zipfile

import numpy as np
import pytest
import torch

from main.tube_state import b_segmented_payload_v1 as payload
from main.tube_state import local_segmented_terminal_sync_v1 as state
from main.tube_state import local_joint_positive_projection_margin_v1 as old_state
from runtime.wan import b_segmented_payload_v1 as native
from runtime.wan import local_segmented_terminal_sync_v1 as runtime
from runtime.wan.local_segmented_terminal_sync_evaluation_v1 import bit_row
from scripts.build_local_segmented_terminal_sync_v1_notebook import build, ROOT, SOURCE_FILES

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


def test_actual_state_only_book_writer_support_and_blind_ties():
    public = state.public_codebook("watermark")
    assert set(public) == {"key", "domain", "directions", "selected_basis"}
    assert torch.equal(public["directions"], old_state.public_codebook("watermark")["directions"][:, :22])
    for directions in public["directions"]:
        a = directions.reshape(22, -1)
        assert torch.allclose(a @ a.T, torch.eye(22, dtype=torch.float64), atol=1e-12)
    assert list(inspect.signature(state.write_state_only).parameters) == ["source", "host_state", "public"]
    assert list(inspect.signature(state.observe_state).parameters) == ["received", "key", "public"]
    z = torch.zeros(state.LATENT_SHAPE)
    tied = state.observe_state(z, "watermark", public)
    assert np.shape(tied["roi_scores"]) == (22, 4, 22)
    assert all(r["status"] == "UNRESOLVED" and r["tied_candidates"] == list(range(22)) for r in tied["states"])
    candidate, receipt = state.write_state_only(z, tied, public)
    observed = state.observe_state(candidate, "watermark", public)
    assert [r["label"] for r in observed["states"]] == list(range(22))
    assert receipt["payload_directions"] == receipt["payload_corrections"] == 0
    assert receipt["fp32_materializations"] == 1 and receipt["norm_cap"] is None
    delta = state.delta_statistics(z, candidate)
    assert len(delta["times"]) == 46
    assert delta["outside_support"]["linf"] == delta["legacy_channels_0_3"]["linf"] == 0
    assert delta["times"][0]["linf"] == delta["times"][45]["linf"] == 0
    moved = candidate.clone()
    for k in range(22):
        moved[:, :, 1+2*k:3+2*k] = candidate[:, :, 43-2*k:45-2*k]
    assert [r["label"] for r in state.observe_state(moved, "watermark", public)["states"]] == list(reversed(range(22)))
    assert bit_row(None, 0, "UNRESOLVED")["status"] == "UNRESOLVED"
    assert bit_row(None, 0, "READ")["status"] == "MISSING"
    assert bit_row(None, 0, "INTERRUPTED")["status"] == "INTERRUPTED"


def test_temporal_target_full_cell_geometry_and_ordered_zero_votes():
    bits = [0]*8 + [1]*8 + [i % 2 for i in range(8)] + [1-i % 2 for i in range(8)]
    target, mask = payload.build_target(torch.zeros(state.LATENT_SHAPE), "watermark", bits)
    assert int(mask.sum()) == 44160 and not mask[:, 4:].any()
    h, w = payload.original.coordinates("watermark")[0]
    for t in range(46):
        g = 0 if t == 0 else 3 if t == 45 else (t-1)//11
        assert target[0, :4, t, h, w].tolist() == [(2*bits[8*g]-1)*.5]*4
    plan = payload.segment_plan(list(range(181)))
    assert [len(g["cells"]) for g in plan["segments"]] == [11]*4
    crop = payload.segment_plan(list(range(37, 126)))
    assert [len(g["cells"]) for g in crop["segments"]] == [2, 11, 9, 0]
    deleted = payload.segment_plan([i for i in range(181) if not 45 <= i <= 88])
    assert [len(g["cells"]) for g in deleted["segments"]] == [11, 0, 11, 11]
    assert deleted["operation"]["synthetic_output_indices"] == list(range(45, 89))
    assert typing.get_type_hints(payload.decode_visible_span)["return"] == dict[str, typing.Any]
    signed = np.ones((4, 44, 240), dtype=np.int8)
    signed[:2] = -1  # channel-major exact ties encounter zero first.
    zeros = np.zeros_like(signed)
    zeros[0, 0, 0] = 1
    raw = payload.decode_signed_votes(plan, signed, zeros)
    assert raw["decoded_bits"] == [0]*32
    assert all(b["count"] == 1320 and b["tie"] for g in raw["segments"] for b in g["bits"])
    assert raw["segments"][0]["bits"][0]["zero_coefficients"] == 1
    assert payload.decode_signed_votes(plan, -signed, np.zeros_like(signed))["decoded_bits"] == [1]*32
    missing = payload.decode_signed_votes(deleted, signed, np.zeros_like(signed))
    assert missing["decoded_bits"][8:16] == [None]*8
    with pytest.raises(ValueError, match="strict >0"):
        payload.decode_signed_votes(plan, -signed, zeros)
    # Real FFT on synthetic zeros only: no source tensor is read.
    fft = payload.read_segments(torch.zeros(state.LATENT_SHAPE), "watermark", list(range(181)))
    assert fft["decoded_bits"] == [0]*32
    assert np.all(fft["signed_votes"] == -1) and np.all(fft["zero_coefficient_mask"] == 1)


def test_native_fifty_steps_dispatches_twenty_five_segmented_controls(monkeypatch):
    events, records, controls = [], [], []
    sentinel = torch.ones((1, 1))
    monkeypatch.setattr(native.layout, "build_target", lambda z, key, bits: (sentinel, torch.ones_like(z, dtype=torch.bool)))
    def local(clean, target, mask):
        assert target is sentinel and mask.all()
        controls.append(clean.clone())
        return torch.zeros_like(clean), {"synthetic": True}
    monkeypatch.setattr(native.payload, "local_delta", local)
    class Scheduler:
        config = SimpleNamespace(prediction_type="flow_prediction", thresholding=False, lower_order_final=True)
        predict_x0 = True
        step_index = None
        sigmas = torch.linspace(1, 0, 51)
        timesteps = torch.arange(50, 0, -1)
        def step(self, v, timestep, z, return_dict):
            self.step_index = (self.step_index or 0)+1
            return (z-v*.001,)
    def transformer(**kw):
        return (torch.full_like(kw["hidden_states"], kw["encoder_hidden_states"]),)
    out, receipt = native.run_trajectory(SimpleNamespace(transformer=transformer), torch.zeros((1, 1)), Scheduler(),
        2., 1., torch.float32, "watermark", [0]*32, lambda *x: events.append(x), records.append)
    assert torch.isfinite(out).all() and len(records) == 50 and len(controls) == 25
    assert [r["index"] for r in records if r["enabled"]] == list(range(25, 50))
    for name, count in (("transformer_conditional", 50), ("transformer_unconditional", 50), ("native_step", 50), ("payload_gradient", 25)):
        assert events.count((name, False)) == events.count((name, True)) == count
    assert receipt["net_bits"] == 32 and receipt["actual_eta"] == 5520


def install_dispatch_fixture(monkeypatch, tmp_path, failure=None):
    output, truth = tmp_path/"output", tmp_path/"writer.json"
    truth.write_text(json.dumps({"fixed_method_semantics": {"bits": [0]*32}}))
    events = []
    original_read = Path.read_text
    def read(path, *a, **kw):
        if path == truth:
            assert (output/"blind_completion.json").is_file()
            for arm in state.ARMS:
                for layer in state.LAYERS:
                    for key in state.KEYS:
                        for name in ("state.json", "payload.json"):
                            assert (output/"blind"/arm/layer/key/name).is_file()
            events.append("truth")
        return original_read(path, *a, **kw)
    monkeypatch.setattr(Path, "read_text", read)
    def entered():
        ledger = json.loads(original_read(output/"ledger.json"))
        active = [k for k, v in ledger.items() if v["status"] == "STARTED"]
        assert len(active) == 1 and (output/"attempt.json").exists()
        name = active[0]
        events.append(name)
        assert "truth" not in events
        if name == failure:
            raise KeyboardInterrupt("synthetic interruption") if name.endswith("/payload") else RuntimeError("synthetic failure")
        return name
    z = torch.zeros(state.LATENT_SHAPE)
    pixels = torch.zeros((), dtype=torch.uint8).expand(state.VIDEO_SHAPE)
    class Backend:
        device = "cpu"
        blocks = {"encoder": 0, "decoder": 0}
        def load_terminal(self, path): entered(); return z
        def load_native(self, path): entered(); return pixels
        def load(self): entered(); return {"synthetic": True}
        def decode(self, value): entered(); self.blocks["decoder"] += 1; return {"float_rgb": pixels}
        def quantize(self, value): entered(); return pixels
        def save_mp4(self, value, path): entered(); return {"synthetic": True}
        def probe(self, path): entered(); return {"synthetic": True}
        def read_mp4(self, path, probe): entered(); return pixels
        def encode(self, value): entered(); self.blocks["encoder"] += 1; return z
    def book(key): entered(); return {}
    def observe(*args):
        entered()
        raw = state.unavailable_state("UNRESOLVED")
        raw.update(status="READ", pooled_scores=[[0.]*22 for _ in range(22)], roi_scores=[[[0.]*22 for _ in range(4)] for _ in range(22)])
        return raw
    def construct(*args): entered(); return z, {"status": "CONSTRUCTED", "synthetic": True}
    def delta(*args): entered(); return {"times": [{"latent_time": i} for i in range(46)]}
    def read_payload(*args):
        entered()
        assert (output/"state_completion.json").exists()
        assert len(list((output/"blind").rglob("state.json"))) == 8
        assert len(list((output/"blind").rglob("clean_payload_plan.json"))) == 8
        plan = payload.segment_plan(list(range(181)))
        signed = -np.ones((4, 44, 240), dtype=np.int8)
        return payload.decode_signed_votes(plan, signed, np.zeros_like(signed))
    def quality(*args, **kw): entered(); return runtime.shared.unavailable_quality("SYNTHETIC_DISPATCH_ONLY", kw["meaning"])
    monkeypatch.setattr(runtime.state, "public_codebook", book)
    monkeypatch.setattr(runtime.state, "observe_state", observe)
    monkeypatch.setattr(runtime.state, "write_state_only", construct)
    monkeypatch.setattr(runtime.state, "delta_statistics", delta)
    monkeypatch.setattr(runtime.payload, "read_segments", read_payload)
    monkeypatch.setattr(runtime, "distortion", quality)
    monkeypatch.setattr(runtime.shared, "validate_rgb", lambda value, **kw: value)
    monkeypatch.setattr(torch, "save", lambda value, path: Path(path).write_bytes(b"synthetic dispatch placeholder"))
    return output, truth, events, Backend()


@pytest.mark.parametrize("failure", [None, "source/P", "construct", "S/decode", "P/encode_mp4", "latent/P_to_S", "P/terminal/CORRECT/payload"])
def test_clean_runner_fixed_slots_blind_persistence_failure_and_readonly(monkeypatch, tmp_path, failure):
    output, truth, events, backend = install_dispatch_fixture(monkeypatch, tmp_path, failure)
    result = runtime.run("synthetic-terminal", "synthetic-native", output, truth, backend=backend)
    assert len(result["calls"]) == 35 and events[-1] == "truth"
    assert result["status"] == ("EXECUTION_COMPLETE" if failure is None else "INCOMPLETE")
    assert len(result["evaluation"]) == 8
    assert sum(v["state"]["denominator"] for v in result["evaluation"].values()) == 176
    assert sum(v["payload"]["denominator"] for v in result["evaluation"].values()) == 256
    assert sum(len(v["state_competitors"]) for v in result["evaluation"].values()) == 3696
    assert len(result["latent_difference"]["times"]) == 46
    for layer in ("rgb8", "mp4"):
        q = json.loads((output/"quality"/(layer+".json")).read_text())
        assert len(q["frames"]) == 181 and len(q["adjacent"]) == 180
    if failure:
        expected = "INTERRUPTED" if failure.endswith("/payload") else "FAILED"
        assert result["calls"][failure]["status"] == expected
        if expected == "INTERRUPTED":
            rows = result["evaluation"]["P/terminal/CORRECT"]["payload"]["rows"]
            assert all(r["status"] == expected for r in rows)
    else:
        assert all(r["status"] == "RETURNED" for r in result["calls"].values())
        assert result["counts"]["ordinary_decode"]["started"] == 1
        assert result["counts"]["posterior_encode"]["started"] == 2
        assert all(r["status"] == "UNRESOLVED" for r in result["evaluation"]["P/terminal/CORRECT"]["state"]["rows"])
        assert len(list((output/"blind").rglob("votes.npz"))) == 8
    before = list(events)
    def forbidden(*a, **kw): raise AssertionError("backend constructed on read-only reopen")
    monkeypatch.setattr(runtime, "WanBackend", forbidden)
    reopened = runtime.run("not-readable", "not-readable", output, "no-truth")
    assert reopened["status"] == "EXISTING_ATTEMPT_NOT_REEXECUTED" and events == before
    # Even unreadable saved JSON cannot authorize another attempt.
    (output/"result.json").write_text("{")
    (output/"checkpoint.json").write_text("{")
    assert runtime.run("bad", "bad", output, "bad")["status"] == "EXISTING_ATTEMPT_WITHOUT_CHECKPOINT"


def test_quality_maximum_exact_ties_and_missing_rows():
    q = dict(status="SCORED", frame_denominator=3, adjacent_denominator=2,
        frames=[dict(frame=i, whole=dict(rms=1, linf=2), outside_nominal_support=dict(rms=0, linf=0)) for i in range(3)],
        adjacent=[dict(frame_before=i, frame_after=i+1, whole=dict(rms=0, linf=0), outside_union_nominal_support=dict(rms=0, linf=0)) for i in range(2)])
    peaks = runtime.with_quality_peaks(q)["peaks"]
    assert peaks["frames"]["whole"]["rms"]["locations"] == [{"frame": i} for i in range(3)]
    q["frames"].pop()
    assert runtime.with_quality_peaks(q)["peaks"]["frames"]["whole"]["rms"]["status"] == "UNAVAILABLE"


def test_notebook_source_sync_safe_loader_and_editable_no_git_cli(tmp_path):
    nb_path, archive = build()
    nb = json.loads(nb_path.read_text())
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes[0] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for code in codes: ast.parse(code)
    assert "https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/" in codes[1]
    assert "is_relative_to" in codes[1] and "stat.S_ISLNK" in codes[1]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(SOURCE_FILES)
        for name in SOURCE_FILES: assert bundle.read(name) == (ROOT/name).read_bytes()
        bundle.extractall(tmp_path/"editable")
    source = tmp_path/"editable"
    assert not (source/".git").exists()
    command = subprocess.run([sys.executable, "-m", "experiments.wan_state_clock.local_segmented_terminal_sync_v1_run", "--help"],
        cwd=source, capture_output=True, text=True, check=True)
    assert "--config" in command.stdout and "--mode" not in command.stdout
    subprocess.run([sys.executable, "-c", "import runtime.wan.b_segmented_payload_v1"], cwd=source, check=True)
    for name in ("main/tube_state/b_segmented_payload_v1.py", "main/tube_state/local_segmented_terminal_sync_v1.py"):
        for node in ast.walk(ast.parse((source/name).read_text())):
            if isinstance(node, ast.ImportFrom): assert (node.module or "").split(".")[0] not in ("runtime", "experiments", "governance")
