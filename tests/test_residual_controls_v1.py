"""Targeted CPU/stub checks for the post-MP4 control follow-up."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat
import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import residual_controls_v1 as controls
from experiments.paper_results_v1 import build_residual_controls_v1_colab as builder


pytestmark = pytest.mark.unit


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rgb(path, frames, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.full((frames, 1, 1, 3), value, np.uint8).tofile(path)
    return str(path)


def operation(received_frames=181, mapping=None):
    mapping = [min(index, received_frames - 1) for index in range(181)] if mapping is None else list(mapping)
    return {"status": "SUPPORTED", "output_frames": 181, "wan_support": 44,
            "received_index_map": mapping,
            "source_coordinate_map": [[{"source_index": index, "weight": 1.0}] for index in range(181)]}


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "RGB_SHAPE", [181, 1, 1, 3])
    monkeypatch.setattr(controls, "DELETE_SHAPE", [163, 1, 1, 3])
    source = tmp_path / "joint"; checkpoint = tmp_path / "checkpoint"
    source_state = {"schema": "paper-results-v1-residual-joint-v1", "key": "watermark", "sources": {}, "rows": []}
    artifacts = []
    truth = [index % 2 for index in range(32)]
    deleted = set(controls.DELETE_POSITIONS)
    keep = [index for index in range(181) if index not in deleted]
    for case_index, case in enumerate(controls.CASES):
        candidate = write_rgb(source / "run_state/rgb" / case / "candidate.rgb8", 181, 10 + case_index)
        source_state["sources"][case] = {"path": candidate, "status": "AVAILABLE", "shape": [181, 1, 1, 3]}
        native_pre = write_rgb(checkpoint / "run_state/artifacts" / case / "PAYLOAD_NATIVE/pre.rgb8", 181, 8)
        native_post = write_rgb(checkpoint / "run_state/artifacts" / case / "PAYLOAD_NATIVE/post.rgb8", 181, 9)
        off_post = write_rgb(checkpoint / "run_state/artifacts" / case / "OFF_NATIVE/post.rgb8", 181, 3)
        artifacts += [
            {"artifact_id": f"{case}/PAYLOAD_NATIVE/PRE", "status": "AVAILABLE", "path": native_pre, "shape": [181, 1, 1, 3]},
            {"artifact_id": f"{case}/PAYLOAD_NATIVE/POST", "status": "AVAILABLE", "path": native_post, "shape": [181, 1, 1, 3]},
            {"artifact_id": f"{case}/OFF_NATIVE/POST", "status": "AVAILABLE", "path": off_post, "shape": [181, 1, 1, 3],
             "mp4": {"path": str(checkpoint / "run_state/artifacts" / case / "OFF_NATIVE/source.mp4")}},
        ]
        for group in controls.GROUPS:
            frames = 181 if group == "CLEAN_MP4" else 163
            received = write_rgb(source / "run_state/media" / case / (group.lower() + ".rgb8"), frames, 11 + case_index)
            framewise = source / "run_state/latents/framewise" / group.lower() / (case + ".npy")
            framewise.parent.mkdir(parents=True, exist_ok=True)
            np.save(framewise, np.full((frames, 4, 40, 64), case_index, np.float32), allow_pickle=False)
            selector = source / "run_state/clock" / group.lower() / case / "blind_selection.json"
            write_json(selector, {"operation": operation(frames)})
            wan = source / "run_state/latents/wan" / group.lower() / (case + ".pt")
            wan.parent.mkdir(parents=True, exist_ok=True); torch.save(torch.zeros(1, 16, 46, 40, 64), wan)
            source_state["rows"].append({
                "slot_id": f"{group.lower()}/{case}", "group": group, "case_id": case, "status": "READ",
                "received_frames": frames, "missing_bits": 0,
                "media": {"status": "AVAILABLE", "received_path": received},
                "clock": {"status": "CLOCK_READY", "latent_path": str(framewise), "selector_record": str(selector),
                          "selector_status": "ESTIMATED", "output_frames": 181, "wan_support": 44},
                "payload": {"status": "READ", "decoded_bits": truth, "read_record": str(source / "read.json"),
                            "R": 44, "votes": [], "output_frames": 181, "original_reader_match": True},
            })
    write_json(source / "run_state/residual_joint_state.json", source_state)
    write_json(checkpoint / "run_state/run_state.json", {"artifacts": artifacts})
    config = tmp_path / "config.json"
    write_json(config, {"codec": controls.CODEC, "keys": {"K0": "watermark", "K1": "watermark-wrong"},
                        "payload_bits": truth,
                        "models": {"wan": {"local_snapshot_path": "/models/wan", "device": "cuda"},
                                   "framewise": {"local_snapshot_path": "/models/framewise", "device": "cuda", "batch_frames": 8}}})
    store = controls.initialize(source, checkpoint, tmp_path / "output", config)
    return store, keep


def test_fixed_plan_and_post_mp4_quality(tmp_path, monkeypatch):
    store, _ = fixture(tmp_path, monkeypatch)
    assert len(store.data["rows"]) == 24 and store.data["fixed_denominator"]["bits"] == 768
    assert len(store.data["quality_rows"]) == 24
    assert store.data["planned_budget"]["new_wan_encode_clips_upper_bound"] == 16
    assert "new_wan_encode_clips_lower_bound" not in store.data["planned_budget"]
    controls.run_quality(store)
    state = controls.Store(store.output).data
    assert {row["status"] for row in state["quality_rows"]} == {"EVALUATED"}
    row = state["quality_rows"][0]
    assert row["full_clip"]["frames"] == 181
    record = json.loads(Path(row["record"]).read_text())
    assert len(record["per_frame"]) == 181 and len(record["temporal"]["rows"]) == 180
    assert record["quality_gate"] is None
    assert set(record["peaks"]) == {
        "frame_rmse", "frame_max_pixel_abs", "transition_rmse", "transition_max_pixel_delta_abs",
    }
    for peak in record["peaks"].values():
        assert peak["neighborhood"]
        assert Path(peak["visuals"]["frame_triptych"]).is_file()
        assert Path(peak["visuals"]["crop_triptych"]).is_file()
        assert "fixed 0..255" in peak["visuals"]["display_difference"]


class FakeFramewise:
    def encode(self, rgb, batch_observer):
        for start in range(0, len(rgb), 8):
            stop = min(len(rgb), start + 8); batch_observer("STARTED", start, stop, None); batch_observer("RETURNED", start, stop, None)
        return np.zeros((len(rgb), 4, 40, 64), np.float32)

    def close(self):
        pass


def test_key_isolation_operation_reuse_and_resume(tmp_path, monkeypatch):
    store, keep = fixture(tmp_path, monkeypatch)
    def fake_load(path, shape):
        return torch.zeros(tuple(shape), dtype=torch.uint8)
    def fake_edit(rgb, spec):
        return rgb[keep], {"status": "APPLIED", "positions": list(controls.DELETE_POSITIONS)}
    controls.run_media(store, load_rgb=fake_load, apply_edit=fake_edit)
    seen_keys = []
    def fake_select(latent, key):
        seen_keys.append(key)
        return {"signed_projection": np.zeros((1, 1)), "rho": np.zeros((1, 1))}, {
            "estimate": {"status": "ESTIMATED", "score_gap": 1.0}, "operation": operation(len(latent)), "reason": None}
    controls.run_clock(store, load_backend=lambda: FakeFramewise(), load_rgb=fake_load, select_clock=fake_select)
    assert seen_keys.count("watermark-wrong") == 8 and seen_keys.count("watermark") == 8
    reloaded = controls.Store(store.output)
    changed = controls._row(reloaded, "WM_K1", "DELETE_G0", "confirm_01")
    record = json.loads(Path(changed["clock"]["selector_record"]).read_text())
    record["operation"]["received_index_map"][0] = 1
    write_json(changed["clock"]["selector_record"], record)
    missing = Path(reloaded.data["source_joint_run"]) / "run_state/latents/wan/clean_mp4/confirm_02.pt"
    missing.unlink()
    invalid = Path(reloaded.data["source_joint_run"]) / "run_state/latents/wan/clean_mp4/confirm_07.pt"
    invalid.write_bytes(b"not-a-torch-tensor")
    encodes = []
    def fake_encode(vae, rgb, observer):
        encodes.append(tuple(rgb.shape)); observer("RETURNED", None)
        return torch.zeros(1, 16, 46, 40, 64)
    def fake_read(latent, key, frames, source_map):
        assert key in ("watermark", "watermark-wrong")
        return {"status": "READ", "decoded_bits": reloaded.data["payload_bits"], "signed_votes": np.zeros((1,)),
                "zero_mask": np.zeros((1,), bool), "votes": [], "R": 44, "output_frames": frames,
                "original_reader_match": True}
    controls.run_wan(reloaded, load_vae=lambda: object(), encode=fake_encode, payload_read=fake_read, load_rgb=fake_load)
    final = controls.Store(store.output).data
    assert len(encodes) == 11  # eight OFF + changed operation + missing and invalid K0 cached latents
    assert len(final["wan_reuses"]) == 5
    assert {row["status"] for row in final["wan_cache_repairs"]} == {"MISSING_REENCODE", "INVALID_REENCODE"}
    assert len(final["payload_read_calls"]) == 16
    summary = json.loads((store.output / "summary.json").read_text())
    assert summary["status"] == "IN_PROGRESS"  # quality is intentionally pending in this test
    assert summary["arms"]["WM_K0_REUSED"]["evaluated_rows"] == 8
    assert summary["arms"]["WM_K0_REUSED"]["exact_payload_recoveries"] == 8
    assert summary["arms"]["WM_K1"]["false_target_exact_matches"] == 8
    assert summary["arms"]["OFF_K0"]["false_target_exact_matches"] == 8
    assert summary["controls_are_presence_threshold_free"] is True
    csv_text = (store.output / "control_rows.csv").read_text()
    assert "exact_payload_recovery" in csv_text and "false_target_exact_match" in csv_text
    assert "matches_known_probe_bits" not in csv_text
    assert not (store.output / "control_rows.csv.tmp").exists()
    report = json.loads((store.output / "evaluation_report.json").read_text())
    k0 = next(row for row in report["rows"] if row["arm"] == "WM_K0_REUSED")
    control = next(row for row in report["rows"] if row["arm"] == "WM_K1")
    for name in ("received_frames", "selector_status", "score_gap", "output_frames",
                 "wan_support", "R", "original_reader_match", "clock_path_error", "clock_truth_exact"):
        assert name in k0 and name in control
    assert k0["exact_payload_recovery"] is True and "false_target_exact_match" not in k0
    assert control["false_target_exact_match"] is True and "exact_payload_recovery" not in control


def test_unsupported_clock_keeps_slot_and_does_not_force_wan_encode(tmp_path, monkeypatch):
    store, keep = fixture(tmp_path, monkeypatch)
    load_rgb = lambda path, shape: torch.zeros(tuple(shape), dtype=torch.uint8)
    controls.run_media(store, load_rgb=load_rgb,
                       apply_edit=lambda rgb, spec: (rgb[keep], {"status": "APPLIED"}))
    selections = 0
    def fake_select(latent, key):
        nonlocal selections
        selections += 1
        selected = operation(len(latent)) if selections != 9 else {"status": "UNSUPPORTED"}
        return {"signed_projection": np.zeros((1, 1)), "rho": np.zeros((1, 1))}, {
            "estimate": {"status": "ESTIMATED", "score_gap": 1.0}, "operation": selected,
            "reason": None if selections != 9 else "unsupported visible span",
        }
    controls.run_clock(store, load_backend=lambda: FakeFramewise(), load_rgb=load_rgb,
                       select_clock=fake_select)
    encoded = []
    def fake_encode(vae, rgb, observer):
        encoded.append(tuple(rgb.shape)); observer("RETURNED", None)
        return torch.zeros(1, 16, 46, 40, 64)
    def fake_read(latent, key, frames, source_map):
        return {"status": "READ", "decoded_bits": store.data["payload_bits"],
                "signed_votes": np.zeros((1,)), "zero_mask": np.zeros((1,), bool),
                "votes": [], "R": 44, "output_frames": frames, "original_reader_match": True}
    controls.run_wan(controls.Store(store.output), load_vae=lambda: object(), encode=fake_encode,
                     payload_read=fake_read, load_rgb=load_rgb)
    final = controls.Store(store.output).data
    skipped = controls._row(controls.Store(store.output), "OFF_K0", "CLEAN_MP4", "confirm_01")
    assert skipped["status"] == "UNSUPPORTED" and skipped["payload"]["status"] == "PENDING"
    assert len(final["rows"]) == 24 and len(encoded) == 7
    assert len(final["payload_read_calls"]) == 15


def test_failed_off_media_retains_fixed_rows(tmp_path, monkeypatch):
    store, keep = fixture(tmp_path, monkeypatch)
    Path(controls._row(store, "OFF_K0", "CLEAN_MP4", "confirm_01")["media"]["source_artifact"]["path"]).unlink()
    controls.run_media(store, load_rgb=lambda path, shape: torch.zeros(tuple(shape), dtype=torch.uint8),
                       apply_edit=lambda rgb, spec: (rgb[keep], {"status": "APPLIED"}))
    state = controls.Store(store.output).data
    assert len(state["rows"]) == 24
    assert controls._row(controls.Store(store.output), "OFF_K0", "CLEAN_MP4", "confirm_01")["status"] == "FAILED"
    assert controls._row(controls.Store(store.output), "OFF_K0", "DELETE_G0", "confirm_01")["status"] == "FAILED"


def test_initialize_materializes_slots_before_missing_sources(tmp_path, monkeypatch):
    original, _ = fixture(tmp_path, monkeypatch)
    source_run = Path(original.data["source_joint_run"])
    checkpoint_run = Path(original.data["checkpoint_run"])
    config = Path(original.data["effective_config"])
    source_state = json.loads((source_run / "run_state/residual_joint_state.json").read_text())
    source_state["rows"] = [row for row in source_state["rows"]
                            if not (row["group"] == "CLEAN_MP4" and row["case_id"] == "confirm_01")]
    write_json(source_run / "run_state/residual_joint_state.json", source_state)
    checkpoint = json.loads((checkpoint_run / "run_state/run_state.json").read_text())
    checkpoint["artifacts"] = [row for row in checkpoint["artifacts"]
                               if row["artifact_id"] != "confirm_02/OFF_NATIVE/POST"]
    write_json(checkpoint_run / "run_state/run_state.json", checkpoint)
    repaired = controls.initialize(source_run, checkpoint_run, tmp_path / "missing-output", config)
    assert len(repaired.data["rows"]) == 24 and len(repaired.data["quality_rows"]) == 24
    assert controls._row(repaired, "WM_K0_REUSED", "CLEAN_MP4", "confirm_01")["status"] == "FAILED"
    assert controls._row(repaired, "WM_K1", "CLEAN_MP4", "confirm_01")["media"]["status"] == "MISSING"
    assert controls._row(repaired, "OFF_K0", "DELETE_G0", "confirm_02")["status"] == "FAILED"
    assert (repaired.output / "residual_controls_state.json").is_file()

    (source_run / "run_state/residual_joint_state.json").write_text("{unreadable", encoding="utf-8")
    global_failure = controls.initialize(source_run, checkpoint_run, tmp_path / "unreadable-output", config)
    assert len(global_failure.data["rows"]) == 24 and len(global_failure.data["quality_rows"]) == 24
    assert all(row["status"] == "FAILED" for row in global_failure.data["rows"] if row["arm"].startswith("WM_"))
    assert (global_failure.output / "handoff_summary.json").exists() is False
    assert (global_failure.output / "summary.json").is_file()


def test_resume_retires_stale_attempt_after_atomic_cache_and_skips_encode(tmp_path, monkeypatch):
    store, keep = fixture(tmp_path, monkeypatch)
    load_rgb = lambda path, shape: torch.zeros(tuple(shape), dtype=torch.uint8)
    controls.run_media(store, load_rgb=load_rgb,
                       apply_edit=lambda rgb, spec: (rgb[keep], {"status": "APPLIED"}))
    target = controls._row(store, "OFF_K0", "CLEAN_MP4", "confirm_01")
    for row in store.data["rows"]:
        if row["arm"] in controls.NEW_ARMS and row is not target:
            row["media"].update(status="FAILED", reason="test isolation")
            row.update(status="FAILED", reason="test isolation")
    latent_path = controls._control_framewise_path(store, target)
    controls._atomic_npy(latent_path, np.zeros((181, 4, 40, 64), np.float32))
    store.data["unit_attempts"]["clock-encode/" + target["slot_id"]] = [
        {"attempt": 1, "status": "RUNNING", "started_at": 1.0}
    ]
    store.save()
    resumed = controls.Store(store.output)
    stale = resumed.data["unit_attempts"]["clock-encode/" + target["slot_id"]][0]
    assert stale["status"] == "INTERRUPTED" and stale["finished_at"] >= stale["started_at"]
    controls.run_clock(resumed, load_backend=lambda: pytest.fail("durable cache should skip backend"),
                       load_rgb=load_rgb,
                       select_clock=lambda latent, key: (
                           {"signed_projection": np.zeros((1, 1)), "rho": np.zeros((1, 1))},
                           {"estimate": {"status": "ESTIMATED", "score_gap": 1.0},
                            "operation": operation(len(latent)), "reason": None}))
    assert controls._row(controls.Store(store.output), "OFF_K0", "CLEAN_MP4", "confirm_01")["clock"]["status"] == "CLOCK_READY"


def test_call_ledgers_close_failures_immediately(tmp_path, monkeypatch):
    store, keep = fixture(tmp_path, monkeypatch)
    load_rgb = lambda path, shape: torch.zeros(tuple(shape), dtype=torch.uint8)
    controls.run_media(store, load_rgb=load_rgb,
                       apply_edit=lambda rgb, spec: (rgb[keep], {"status": "APPLIED"}))
    encode_target = controls._row(store, "OFF_K0", "CLEAN_MP4", "confirm_01")
    for row in store.data["rows"]:
        if row["arm"] in controls.NEW_ARMS and row is not encode_target:
            row["media"].update(status="FAILED", reason="test isolation")
            row.update(status="FAILED", reason="test isolation")
    class BrokenFramewise:
        def encode(self, rgb, batch_observer):
            batch_observer("STARTED", 0, 8, None)
            raise RuntimeError("framewise exploded")
        def close(self): pass
    controls.run_clock(store, load_backend=lambda: BrokenFramewise(), load_rgb=load_rgb)
    state = controls.Store(store.output).data
    assert state["framewise_encode_clips"][-1]["status"] == "FAILED"
    assert state["framewise_physical_batches"][-1]["status"] == "FAILED"
    assert state["unit_attempts"]["clock-encode/" + encode_target["slot_id"]][-1]["status"] == "FAILED"

    selector_store, keep = fixture(tmp_path / "selector", monkeypatch)
    controls.run_media(selector_store, load_rgb=load_rgb,
                       apply_edit=lambda rgb, spec: (rgb[keep], {"status": "APPLIED"}))
    selector_target = controls._row(selector_store, "WM_K1", "CLEAN_MP4", "confirm_01")
    for row in selector_store.data["rows"]:
        if row["arm"] in controls.NEW_ARMS and row is not selector_target:
            row["media"].update(status="FAILED", reason="test isolation")
            row.update(status="FAILED", reason="test isolation")
    controls.run_clock(selector_store, select_clock=lambda latent, key: (_ for _ in ()).throw(RuntimeError("selector exploded")))
    selector_state = controls.Store(selector_store.output).data
    assert selector_state["clock_calls"][-1]["status"] == "FAILED"
    assert "selector exploded" in selector_state["clock_calls"][-1]["reason"]

    read_store, keep = fixture(tmp_path / "reader", monkeypatch)
    controls.run_media(read_store, load_rgb=load_rgb,
                       apply_edit=lambda rgb, spec: (rgb[keep], {"status": "APPLIED"}))
    read_target = controls._row(read_store, "WM_K1", "CLEAN_MP4", "confirm_01")
    for row in read_store.data["rows"]:
        if row["arm"] in controls.NEW_ARMS and row is not read_target:
            row["media"].update(status="FAILED", reason="test isolation")
            row.update(status="FAILED", reason="test isolation")
    controls.run_clock(read_store, select_clock=lambda latent, key: (
        {"signed_projection": np.zeros((1, 1)), "rho": np.zeros((1, 1))},
        {"estimate": {"status": "ESTIMATED", "score_gap": 1.0},
         "operation": operation(len(latent)), "reason": None}))
    controls.run_wan(read_store, payload_read=lambda *args: (_ for _ in ()).throw(RuntimeError("reader exploded")),
                     load_rgb=load_rgb)
    read_state = controls.Store(read_store.output).data
    assert read_state["payload_read_calls"][-1]["status"] == "FAILED"
    assert "reader exploded" in read_state["payload_read_calls"][-1]["reason"]


def test_truth_is_not_joined_before_wan_complete(tmp_path, monkeypatch):
    store, _ = fixture(tmp_path, monkeypatch)
    report = json.loads((store.output / "evaluation_report.json").read_text())
    assert report["summary"]["truth_joined"] is False
    assert all(row["truth_join"] == "NOT_JOINED" for row in report["rows"])
    assert all(row["bit_errors_against_known_probe"] is None for row in report["rows"])
    assert report["summary"]["arms"]["WM_K0_REUSED"]["exact_payload_recoveries"] is None
    assert report["summary"]["arms"]["WM_K1"]["false_target_exact_matches"] is None
    assert len(report["appendix"]["post_mp4_quality_pairs"]) == 24


def test_truth_stays_unjoined_when_clock_is_interrupted_after_wan_complete(tmp_path, monkeypatch):
    store, _ = fixture(tmp_path, monkeypatch)
    store.data["phases"]["clock"] = "INTERRUPTED"
    store.data["phases"]["wan"] = "COMPLETE"
    store.save(); controls.report(store)
    report = json.loads((store.output / "evaluation_report.json").read_text())
    assert report["summary"]["truth_joined"] is False
    assert report["summary"]["pending_model_units"]
    assert all(row["truth_join"] == "NOT_JOINED" for row in report["rows"])
    assert all(row["bit_errors_against_known_probe"] is None for row in report["rows"])
    assert report["summary"]["arms"]["WM_K0_REUSED"]["exact_payload_recoveries"] is None


def test_clock_exact_requires_zero_weighted_error_and_full_frame_count(tmp_path, monkeypatch):
    store, _ = fixture(tmp_path, monkeypatch)
    for row in store.data["rows"]:
        if row["arm"] in controls.NEW_ARMS:
            row["media"].update(status="FAILED", reason="test terminal")
            row["clock"].update(status="FAILED", reason="test terminal")
            row["payload"].update(status="FAILED", reason="test terminal")
            row.update(status="FAILED", reason="test terminal")
    store.data["phases"]["clock"] = "COMPLETE"
    store.data["phases"]["wan"] = "COMPLETE"
    store.save()
    monkeypatch.setattr(controls, "_posthoc_path_metrics", lambda row: {
        "status": "EVALUATED_POSTHOC", "frames": row["received_frames"],
        "minimum_support_absolute_error_sum": 0, "weighted_absolute_error_sum": 7.0,
    })
    controls.report(store)
    report = json.loads((store.output / "evaluation_report.json").read_text())
    assert report["summary"]["truth_joined"] is True
    clean = next(row for row in report["rows"]
                 if row["arm"] == "WM_K0_REUSED" and row["group"] == "CLEAN_MP4")
    assert clean["received_frames"] == 181
    assert clean["clock_path_error"]["minimum_support_absolute_error_sum"] == 0
    assert clean["clock_path_error"]["weighted_absolute_error_sum"] == 7.0
    assert clean["clock_truth_exact"] is False


def test_notebook_and_portable_companion(tmp_path):
    result = builder.build()
    notebook = nbformat.read(result["notebook"], as_version=4)
    assert "drive.mount('/content/drive')" in notebook.cells[0].source
    assert notebook.cells[0].source.strip().count("\n") == 1
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "24 rows / 768" in text and "OFF_K1" not in text
    assert '"--phase", "quality"' in text and '"--phase", "media"' in text and '"--phase", phase' in text
    assert "8–16" not in text and "at most 16" in text
    assert "SOURCE_SHA" not in text and "base64" not in text.lower() and "strict" not in text.lower()
    assert "uuid.uuid4().hex[:10]" in text and "source_probe = subprocess.run" in text
    assert "load_list_history(attempts_path" in text and ".unreadable-" in text
    assert "if not required.is_file():\n    archive" not in text
    import ast
    setup_tree = ast.parse(notebook.cells[2].source)
    history_function = next(node for node in setup_tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == "load_list_history")
    namespace = {"Path": Path, "json": json, "os": __import__("os"), "time": __import__("time")}
    exec(compile(ast.Module(body=[history_function], type_ignores=[]), "<history-helper>", "exec"), namespace)
    damaged = tmp_path / "attempts.json"; damaged.write_text("{bad", encoding="utf-8")
    recovered = namespace["load_list_history"](damaged, "attempts")
    assert recovered[0]["status"] == "UNREADABLE_PRESERVED"
    assert not damaged.exists() and len(list(tmp_path.glob("attempts.json.unreadable-*"))) == 1
    import zipfile
    with zipfile.ZipFile(builder.ZIP) as archive:
        names = set(archive.namelist())
        with tempfile.TemporaryDirectory() as directory:
            archive.extractall(directory)
            for name in names:
                if name.endswith(".py"): compile((Path(directory) / name).read_text(), name, "exec")
            child = subprocess.run([sys.executable, "-m", "experiments.paper_results_v1.residual_controls_v1_cli", "--help"],
                                   cwd=directory, capture_output=True, text=True)
            assert child.returncode == 0, child.stderr
    assert "experiments/paper_results_v1/residual_controls_v1.py" in names
