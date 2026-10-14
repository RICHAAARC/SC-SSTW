"""CPU/stub checks for the fixed residual-clock and NO_SYNC handoff."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import residual_clock_v1 as residual
from runtime.wan import framewise_autoencoder_kl as framewise

pytestmark = pytest.mark.unit


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fixture(tmp_path):
    checkpoint = tmp_path / residual.CHECKPOINT_RUN_ID
    layer = tmp_path / residual.LAYER_RUN_ID
    (checkpoint / "run_state").mkdir(parents=True)
    (checkpoint / "attack_run").mkdir()
    (checkpoint / "confirmation_receiver").mkdir()
    (layer / "run_state").mkdir(parents=True)
    config = {
        "keys": {"K0": "fixed-key"}, "payload_bits": [index % 2 for index in range(32)],
        "attacks": [{"attack_id": "delete_g0", "positions": [0, 10, 42, 44, 59, 64, 65, 83, 84, 89, 91, 100, 102, 116, 131, 144, 160, 169]}],
        "models": {"wan": {"local_snapshot_path": "/wan", "device": "cpu"},
                   "framewise": {"local_snapshot_path": "/framewise", "batch_frames": 8, "device": "cpu"}},
    }
    atomic(checkpoint / "effective_confirmation_config.json", config)
    atomic(checkpoint / "run_state/run_state.json", {"artifacts": []})
    atomic(checkpoint / "attack_run/attack_run_state.json", {"artifacts": []})
    atomic(checkpoint / "confirmation_receiver/evaluation_report.json", {"receiver_rows": []})
    atomic(layer / "run_state/evaluation_report.json", {"rows": []})
    runtime = tmp_path / "runtime.json"; atomic(runtime, config)
    return residual.initialize(checkpoint, layer, tmp_path / "out", runtime)


def fake_payload(frames, mapping):
    support = min(44, (frames - 1) // 4)
    signs = np.ones((4, support, 240), dtype=np.int8)
    votes, bits = [], []
    for bit in range(32):
        channel, slot = divmod(bit, 8)
        values = signs[channel, :, slot::8].reshape(-1)
        ones = int((values > 0).sum()); zeros = int(values.size - ones)
        votes.append({"ones": ones, "zeros": zeros, "count": int(values.size)}); bits.append(1)
    return {"status": "READ", "R": support, "output_frames": frames, "decoded_bits": bits,
            "votes": votes, "bit_rows": [], "signed_votes": signs, "zero_mask": signs == 0,
            "original_reader_match": True, "truth_inputs": False}


def evaluated_payload(truth, errors, *, status="READ", support=40):
    bits = list(truth)
    for index in range(errors): bits[index] = 1 - bits[index]
    votes = [{"ones": 20, "zeros": 10, "count": 30} for _ in bits]
    return {"status": status, "R": support, "decoded_bits": bits, "votes": votes}


def make_inputs(store):
    for case in residual.CASES:
        row = residual._row(store, "RESIDUAL_PRE", case)
        path = store.output / "stub" / case / "candidate.rgb8"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
        row["units"]["residual_rgb"].update(status="AVAILABLE", path=str(path))
        sync = residual._row(store, "DELETE_G0_SYNC_VALUE", case)
        path = store.output / "stub" / case / "received.rgb8"; path.write_bytes(b"x")
        sync["units"]["received_order_rgb"].update(status="AVAILABLE", path=str(path))
    store.save()


def test_fixed_integer_formula_has_unit_scale_and_clips():
    native = np.array([0, 100, 250], dtype=np.uint8)
    recon = np.array([10, 150, 5], dtype=np.uint8)
    m05 = np.array([30, 100, 20], dtype=np.uint8)
    assert residual.residual_rgb8(native, recon, m05).tolist() == [20, 50, 255]
    with pytest.raises(ValueError): residual.residual_rgb8(native.astype(np.int16), recon, m05)


def test_saved_blind_map_mismatch_is_retained_as_posthoc_diagnostic(tmp_path):
    store = fixture(tmp_path)
    truth = json.loads(Path(store.data["source_config"]).read_text())["payload_bits"]
    rows = []
    for case, path_metrics in (
        ("confirm_01", {"status": "EVALUATED_POSTHOC", "minimum_support_absolute_error_mean": 1.25,
                        "minimum_support_absolute_error_max": 3}),
        ("confirm_02", None),
    ):
        result = evaluated_payload(truth, 2, status="EVALUATED", support=44)
        rows.append({
            "case_id": case, "arm": "PAYLOAD_FRAMEWISE_M05", "attack_id": "delete_g0",
            "key_label": "K0", "receiver": "CENTERED_U", **result,
            "operation": {"status": "SUPPORTED", "received_index_map": [0] * 181},
            "path_metrics": path_metrics,
            "selector_record": {"path": "/saved/selector.json"},
        })
    atomic(store.data["receiver_report"], {"receiver_rows": rows})
    mismatch = residual._saved_blind(store, "confirm_01")
    assert mismatch["status"] == "EVALUATED" and len(mismatch["decoded_bits"]) == 32
    assert mismatch["blind_map_equal_oracle"] is False
    assert mismatch["path_error"]["minimum_support_absolute_error_mean"] == 1.25
    assert mismatch["path_error"]["evidence_origin"] == "SAVED_RECEIVER_POSTHOC_PATH_METRICS"
    saved = store.output / "sync/confirm_01/saved_blind.json"; atomic(saved, mismatch)
    sync = residual._row(store, "DELETE_G0_SYNC_VALUE", "confirm_01")
    sync["units"]["saved_blind"].update(status="REUSED", path=str(saved))
    sync["units"]["payload_no_sync"].update(evaluated_payload(truth, 3))
    store.save()
    pair = next(item for item in residual.report(store)["delete_g0_paired_sync_effect"]["case_rows"]
                if item["case_id"] == "confirm_01")
    assert pair["status"] == "AVAILABLE" and pair["no_sync_minus_saved_blind_bit_errors"] == 1
    unknown = residual._saved_blind(store, "confirm_02")
    assert unknown["status"] == "EVALUATED" and unknown["blind_map_equal_oracle"] is False
    assert unknown["path_error"]["status"] == "UNKNOWN"
    assert "mean_absolute" not in unknown["path_error"]


def test_wan_fixed_eight_encodes_and_complete_rerun_skip(tmp_path, monkeypatch):
    store = fixture(tmp_path); make_inputs(store)
    monkeypatch.setattr(residual, "_load_rgb_file", lambda path, shape: torch.zeros(1, 1, 1, 3, dtype=torch.uint8))
    loads, encodes = [], []
    def encode(_vae, _rgb, *, encode_observer):
        encode_observer("STARTED", None); encodes.append(1); encode_observer("RETURNED", None)
        frames = 181 if len(encodes) % 2 else 161
        return torch.zeros(1, 16, (frames - 1) // 4 + 1, 40, 64)
    # _wan_units is candidate then no_sync for every case.
    sequence = iter([181, 161] * 4)
    def ordered_encode(_vae, _rgb, *, encode_observer):
        encode_observer("STARTED", None); encodes.append(1); encode_observer("RETURNED", None)
        frames = next(sequence); return torch.zeros(1, 16, (frames - 1) // 4 + 1, 40, 64)
    residual.run_wan(store, load_vae=lambda: loads.append(1) or object(), encode=ordered_encode,
                     payload_read=lambda latent, key, frames, mapping: fake_payload(frames, mapping))
    current = residual.Store(store.output)
    assert len(loads) == 1 and len(encodes) == 8
    assert Counter(item["status"] for item in current.data["wan_encode_clips"]) == {"COMPLETE": 8}
    assert all(residual._row(current, "RESIDUAL_PRE", case)["units"]["payload_candidate"]["R"] == 44 for case in residual.CASES)
    assert all(residual._row(current, "DELETE_G0_SYNC_VALUE", case)["units"]["payload_no_sync"]["R"] == 40 for case in residual.CASES)
    residual.run_wan(current, load_vae=lambda: pytest.fail("complete rerun loaded Wan"),
                     encode=lambda *args, **kwargs: pytest.fail("complete rerun encoded"),
                     payload_read=lambda *args: pytest.fail("complete rerun reread"))
    assert len(residual.Store(store.output).data["wan_encode_clips"]) == 8


def test_corrupt_durable_wan_latent_fails_one_unit_and_continues(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    bad = residual._wan_latent_path(store, "candidate", "confirm_01")
    good = residual._wan_latent_path(store, "candidate", "confirm_02")
    for path in (bad, good): path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"saved")
    calls = []
    def load(path, shape):
        calls.append(str(path))
        if Path(path) == bad: raise ValueError("corrupt cached Wan latent")
        return torch.zeros(shape)
    monkeypatch.setattr(residual, "_load_torch", load)
    residual.run_wan(store, load_vae=lambda: pytest.fail("corrupt resume should not load Wan"),
                     payload_read=lambda latent, key, frames, mapping: fake_payload(frames, mapping))
    current = residual.Store(store.output)
    failed = residual._row(current, "RESIDUAL_PRE", "confirm_01")["units"]["payload_candidate"]
    continued = residual._row(current, "RESIDUAL_PRE", "confirm_02")["units"]["payload_candidate"]
    assert failed["status"] == "FAILED" and "corrupt cached Wan latent" in failed["reason"]
    assert continued["status"] == "READ"
    bad_calls = calls.count(str(bad))
    residual.run_wan(current, load_vae=lambda: pytest.fail("terminal resume failure should not load Wan"),
                     payload_read=lambda *args: pytest.fail("completed durable read should not repeat"))
    assert calls.count(str(bad)) == bad_calls


def test_framewise_reuses_native_and_records_92_physical_batches(tmp_path, monkeypatch):
    store = fixture(tmp_path); make_inputs(store)
    monkeypatch.setattr(residual, "_load_rgb_file", lambda path, shape: torch.zeros(1, 1, 1, 3, dtype=torch.uint8))
    native = np.zeros((181, 4, 40, 64), dtype=np.float32)
    monkeypatch.setattr(residual, "_saved_native_latent", lambda store, case: (native, Path("/saved") / case))
    monkeypatch.setattr(residual, "_save_clock", lambda store, kind, case, latent, origin: {
        "status": "SCORED", "selector_status": "ESTIMATED", "path": list(range(181)),
        "best_score": 1.0, "runner_up_score": 0.0, "score_gap": 1.0, "latent_origin": origin,
    })
    class Backend:
        def encode(self, rgb, *, batch_observer):
            for start in range(0, 181, 8):
                stop = min(start + 8, 181); batch_observer("STARTED", start, stop, None); batch_observer("RETURNED", start, stop, None)
            return native.copy()
        def close(self): pass
    residual.run_clock(store, load_backend=lambda: Backend())
    current = residual.Store(store.output)
    assert Counter(item["status"] for item in current.data["framewise_encode_clips"]) == {"COMPLETE": 4}
    assert Counter(item["status"] for item in current.data["framewise_physical_batches"]) == {"COMPLETE": 92}
    assert all(residual._row(current, "RESIDUAL_PRE", case)["units"]["clock_native"]["latent_origin"] == "REUSED_SAVED_SHARED_FRAMEWISE_LATENT" for case in residual.CASES)


def test_corrupt_durable_clock_latent_fails_one_unit_and_continues(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    bad = residual._framewise_path(store, "candidate", "confirm_01")
    good = residual._framewise_path(store, "candidate", "confirm_02")
    for path in (bad, good): path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"saved")
    latent = np.zeros((181, 4, 40, 64), dtype=np.float32); calls = []
    def load(path):
        calls.append(str(path))
        if Path(path) == bad: raise ValueError("corrupt cached candidate latent")
        return latent
    monkeypatch.setattr(residual, "_load_framewise_array", load)
    monkeypatch.setattr(residual, "_saved_native_latent", lambda store, case: (latent, Path("/saved") / case))
    monkeypatch.setattr(residual, "_save_clock", lambda store, kind, case, value, origin: {
        "status": "SCORED", "selector_status": "ESTIMATED", "path": list(range(181)), "latent_origin": origin,
    })
    residual.run_clock(store, load_backend=lambda: pytest.fail("durable clock resume should not load backend"))
    current = residual.Store(store.output)
    failed = residual._row(current, "RESIDUAL_PRE", "confirm_01")["units"]["clock_candidate"]
    continued = residual._row(current, "RESIDUAL_PRE", "confirm_02")["units"]["clock_candidate"]
    assert failed["status"] == "FAILED" and "corrupt cached candidate latent" in failed["reason"]
    assert continued["status"] == "SCORED"
    bad_calls = calls.count(str(bad))
    residual.run_clock(current, load_backend=lambda: pytest.fail("terminal clock failure should not load backend"))
    assert calls.count(str(bad)) == bad_calls


def test_saved_native_clock_read_failure_is_per_unit(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    latent = np.zeros((181, 4, 40, 64), dtype=np.float32)
    for case in residual.CASES:
        residual._row(store, "RESIDUAL_PRE", case)["units"]["clock_candidate"].update(
            status="FAILED", reason="candidate excluded from this stub")
    store.save(); seen = []
    monkeypatch.setattr(residual, "_saved_native_latent", lambda store, case: (latent, Path("/saved") / case))
    def save_clock(store, kind, case, value, origin):
        seen.append((kind, case))
        if case == "confirm_01": raise ValueError("saved native score failed")
        return {"status": "SCORED", "selector_status": "ESTIMATED", "path": list(range(181)), "latent_origin": origin}
    monkeypatch.setattr(residual, "_save_clock", save_clock)
    residual.run_clock(store, load_backend=lambda: pytest.fail("saved native reuse should not load backend"))
    current = residual.Store(store.output)
    failed = residual._row(current, "RESIDUAL_PRE", "confirm_01")["units"]["clock_native"]
    continued = residual._row(current, "RESIDUAL_PRE", "confirm_02")["units"]["clock_native"]
    assert failed["status"] == "FAILED" and "saved native score failed" in failed["reason"]
    assert continued["status"] == "SCORED" and ("native", "confirm_03") in seen


def test_corrupt_native_fallback_latent_fails_one_unit_and_continues(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    latent = np.zeros((181, 4, 40, 64), dtype=np.float32)
    for case in residual.CASES:
        residual._row(store, "RESIDUAL_PRE", case)["units"]["clock_candidate"].update(
            status="FAILED", reason="candidate excluded from this stub")
    bad = residual._framewise_path(store, "native_fallback", "confirm_01")
    good = residual._framewise_path(store, "native_fallback", "confirm_02")
    for path in (bad, good): path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"saved")
    store.save()
    monkeypatch.setattr(residual, "_saved_native_latent", lambda store, case: (_ for _ in ()).throw(FileNotFoundError("saved native absent")))
    def load(path):
        if Path(path) == bad: raise ValueError("corrupt native fallback")
        return latent
    monkeypatch.setattr(residual, "_load_framewise_array", load)
    monkeypatch.setattr(residual, "_save_clock", lambda store, kind, case, value, origin: {
        "status": "SCORED", "selector_status": "ESTIMATED", "path": list(range(181)), "latent_origin": origin,
    })
    residual.run_clock(store, load_backend=lambda: pytest.fail("durable fallback resume should not load backend"))
    current = residual.Store(store.output)
    failed = residual._row(current, "RESIDUAL_PRE", "confirm_01")["units"]["clock_native"]
    continued = residual._row(current, "RESIDUAL_PRE", "confirm_02")["units"]["clock_native"]
    assert failed["status"] == "FAILED" and "corrupt native fallback" in failed["reason"]
    assert continued["status"] == "SCORED"


def test_framewise_interruption_retains_batches_and_resumes_only_unfinished_clip(tmp_path, monkeypatch):
    store = fixture(tmp_path); make_inputs(store)
    monkeypatch.setattr(residual, "_load_rgb_file", lambda path, shape: torch.zeros(1, 1, 1, 3, dtype=torch.uint8))
    native = np.zeros((181, 4, 40, 64), dtype=np.float32)
    monkeypatch.setattr(residual, "_saved_native_latent", lambda store, case: (native, Path("/saved") / case))
    monkeypatch.setattr(residual, "_save_clock", lambda store, kind, case, latent, origin: {
        "status": "SCORED", "selector_status": "ESTIMATED", "path": list(range(181)),
        "best_score": 1.0, "runner_up_score": 0.0, "score_gap": 1.0, "latent_origin": origin,
    })
    class Interrupted:
        def encode(self, rgb, *, batch_observer):
            batch_observer("STARTED", 0, 8, None); batch_observer("RETURNED", 0, 8, None)
            exc = KeyboardInterrupt("stub stop")
            batch_observer("STARTED", 8, 16, None); batch_observer("FAILED", 8, 16, exc)
            raise exc
        def close(self): pass
    with pytest.raises(KeyboardInterrupt): residual.run_clock(store, load_backend=lambda: Interrupted())
    interrupted = residual.Store(store.output)
    assert Counter(item["status"] for item in interrupted.data["framewise_physical_batches"]) == {"COMPLETE": 1, "INTERRUPTED": 1}
    assert interrupted.data["framewise_encode_clips"][0]["status"] == "INTERRUPTED"
    class Complete:
        def encode(self, rgb, *, batch_observer):
            for start in range(0, 181, 8):
                stop = min(start + 8, 181); batch_observer("STARTED", start, stop, None); batch_observer("RETURNED", start, stop, None)
            return native.copy()
        def close(self): pass
    residual.run_clock(interrupted, load_backend=lambda: Complete())
    resumed = residual.Store(store.output)
    assert Counter(item["status"] for item in resumed.data["framewise_encode_clips"]) == {"INTERRUPTED": 1, "COMPLETE": 4}
    assert Counter(item["status"] for item in resumed.data["framewise_physical_batches"]) == {"COMPLETE": 93, "INTERRUPTED": 1}
    assert len(resumed.data["framewise_loads"]) == 2


def test_new_process_retires_only_stale_running_attempts(tmp_path):
    store = fixture(tmp_path)
    row = residual._row(store, "RESIDUAL_PRE", "confirm_01")
    row["units"]["clock_candidate"].update(status="RUNNING", reason=None)
    store.data["phases"]["clock"] = "RUNNING"
    store.data["framewise_encode_clips"] = [
        {"clip_id": "candidate/confirm_01", "status": "RUNNING"},
        {"clip_id": "candidate/confirm_02", "status": "COMPLETE"},
    ]
    store.data["framewise_physical_batches"] = [
        {"batch_id": "candidate/confirm_01/000-008", "status": "RUNNING"},
        {"batch_id": "candidate/confirm_02/000-008", "status": "COMPLETE"},
    ]
    store.data["unit_attempts"] = {"framewise/candidate/confirm_01": [
        {"attempt": 1, "status": "RUNNING"}, {"attempt": 2, "status": "COMPLETE"},
    ]}
    store.save()
    resumed = residual.Store(store.output)
    assert resumed.data["phases"]["clock"] == "INTERRUPTED"
    current = residual._row(resumed, "RESIDUAL_PRE", "confirm_01")
    assert current["units"]["clock_candidate"]["status"] == "INTERRUPTED"
    assert [item["status"] for item in resumed.data["framewise_encode_clips"]] == ["INTERRUPTED", "COMPLETE"]
    assert [item["status"] for item in resumed.data["framewise_physical_batches"]] == ["INTERRUPTED", "COMPLETE"]
    assert [item["status"] for item in resumed.data["unit_attempts"]["framewise/candidate/confirm_01"]] == ["INTERRUPTED", "COMPLETE"]


def test_sync_pair_delta_keeps_four_pair_denominator(tmp_path):
    store = fixture(tmp_path)
    truth = json.loads(Path(store.data["source_config"]).read_text())["payload_bits"]
    deltas = {"confirm_01": (3, 1), "confirm_02": (1, 2), "confirm_07": (2, 2), "confirm_03": (4, 2)}
    for case, (no_sync_errors, blind_errors) in deltas.items():
        row = residual._row(store, "DELETE_G0_SYNC_VALUE", case)
        row["units"]["payload_no_sync"].update(evaluated_payload(truth, no_sync_errors))
        row["units"]["saved_blind"].update(evaluated_payload(truth, blind_errors, status="EVALUATED", support=44))
    # One unavailable pair stays in the denominator and makes the complete total unavailable.
    residual._row(store, "DELETE_G0_SYNC_VALUE", "confirm_03")["units"]["payload_no_sync"].update(
        status="FAILED", reason="stub missing pair")
    store.save(); partial = residual.report(store)["delete_g0_paired_sync_effect"]
    assert partial["planned_pairs"] == 4 and partial["available_pairs"] == 3 and partial["unavailable_pairs"] == 1
    assert partial["no_sync_minus_saved_blind_bit_errors_total"] is None
    assert next(row for row in partial["case_rows"] if row["case_id"] == "confirm_03")["status"] == "UNAVAILABLE"
    row = residual._row(store, "DELETE_G0_SYNC_VALUE", "confirm_03")
    row["units"]["payload_no_sync"].update(evaluated_payload(truth, 4))
    store.save(); complete = residual.report(store)["delete_g0_paired_sync_effect"]
    assert complete["available_pairs"] == 4
    assert complete["no_sync_minus_saved_blind_bit_errors_total"] == sum(a - b for a, b in deltas.values())
    assert "support length" in complete["attribution_limit"]


def test_framewise_batch_observer_is_optional_and_reports_failure():
    class Distribution:
        def __init__(self, value): self.value = value
        def mode(self): return self.value
    class Encoded:
        def __init__(self, value): self.latent_dist = Distribution(value)
    class VAE:
        config = type("Config", (), {"scaling_factor": framewise.SCALED_LATENT_FACTOR})()
        def __init__(self, fail=False): self.parameter = torch.nn.Parameter(torch.zeros(())); self.fail = fail
        def parameters(self): return iter((self.parameter,))
        def encode(self, value):
            if self.fail: raise RuntimeError("batch failed")
            return Encoded(torch.zeros(len(value), 4, 2, 2))
    rgb = torch.zeros(3, 16, 16, 3)
    assert framewise.encode_rgb_frames(VAE(), rgb, batch_frames=2).shape == (3, 4, 2, 2)
    events = []
    with pytest.raises(RuntimeError):
        framewise.encode_rgb_frames(VAE(fail=True), rgb, batch_frames=2,
                                    batch_observer=lambda *args: events.append(args))
    assert [event[0] for event in events] == ["STARTED", "FAILED"]
    assert events[-1][1:3] == (0, 2) and isinstance(events[-1][3], RuntimeError)


def test_notebook_zip_and_portable_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/paper_results_v1_residual_clock_v1_colab.ipynb").read_text())
    assert "".join(notebook["cells"][0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    joined = json.dumps(notebook)
    assert "force_remount" not in joined and "strict" not in joined.lower()
    assert "4 clips / 92 physical batches" in joined and "12 payload rows" in joined
    assert "including restored support length" in joined
    assert "At most four unique old MP4s" in joined and "cumulative actual attempts can exceed four" in joined
    assert "eligible regardless of whether its map equals the offline oracle" in joined
    assert "missing path-error evidence is reported as unknown" in joined
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"])); assert item["outputs"] == [] and item["execution_count"] is None
    archive_path = root / "notebooks/paper_results_v1_residual_clock_v1_companion.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist(): assert archive.read(name) == (root / name).read_bytes()
        assert "experiments/paper_results_v1/native_adapters.py" in archive.namelist()
        archive.extractall(tmp_path / "portable")
    result = subprocess.run([sys.executable, "-B", str(tmp_path / "portable/experiments/paper_results_v1/residual_clock_v1_cli.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    isolated = subprocess.run([
        sys.executable, "-I", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import experiments.paper_results_v1.real_backends; "
        "from experiments.paper_results_v1.residual_clock_v1 import run_clock; "
        "\ntry: run_clock(None, load_backend=lambda: None)\nexcept AttributeError: pass\nelse: raise AssertionError('lazy clock path did not execute')",
        str(tmp_path / "portable"),
    ], cwd=tmp_path, capture_output=True, text=True)
    assert isolated.returncode == 0, isolated.stderr
