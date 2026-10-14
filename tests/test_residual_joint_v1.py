"""CPU/stub checks for the fixed saved-residual joint handoff."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import nbformat
import pytest
import torch

from experiments.paper_results_v1 import residual_joint_v1 as joint
from experiments.paper_results_v1 import build_residual_joint_v1_colab as builder


pytestmark = pytest.mark.unit


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(joint, "RGB_SHAPE", [181, 1, 1, 3])
    monkeypatch.setattr(joint, "DELETE_SHAPE", [163, 1, 1, 3])
    source = tmp_path / "source"
    truth = [index % 2 for index in range(32)]
    write_json(source / "run_state/residual_clock_state.json", {
        "schema": "paper-results-v1-residual-clock-v1", "key": "fixed-k0",
    })
    write_json(source / "effective_residual_clock_config.json", {"payload_bits": truth})
    for case in joint.CASES:
        path = source / "run_state/rgb" / case / "residual_clock_pre.rgb8"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(181 * 3))
        write_json(source / "run_state/quality" / case / "residual_vs_native.json", {
            "peak_pixel": {"frame": 2},
            "temporal_peak_pixel": {"from_frame": 3, "to_frame": 4},
        })
    config = tmp_path / "config.json"
    write_json(config, {
        "codec": joint.CODEC, "attacks": [joint.delete_spec()],
        "keys": {"K0": "fixed-k0"}, "payload_bits": truth,
        "models": {
            "wan": {"local_snapshot_path": "/models/wan", "device": "cuda"},
            "framewise": {"local_snapshot_path": "/models/framewise", "device": "cuda", "batch_frames": 8},
        },
    })
    return source, config


def fixture(tmp_path, monkeypatch):
    source, config = fixture_inputs(tmp_path, monkeypatch)
    return joint.initialize(source, tmp_path / "output", config)


def test_fixed_plan_budget_and_truth_boundary(tmp_path, monkeypatch):
    store = fixture(tmp_path, monkeypatch)
    assert [row["slot_id"] for row in store.data["rows"]] == joint._slot_ids()
    assert store.data["fixed_denominator"]["joint_rows"] == 8
    assert store.data["planned_budget"]["framewise_physical_batches"] == 176
    assert store.data["planned_budget"]["wan_encode_clips_upper_bound"] == 8
    assert store.data["method"]["no_sync_control"].startswith("not included")
    assert joint.delete_spec()["positions"] == joint.DELETE_POSITIONS
    store.data["payload_read_calls"].append({"slot_id": "clean_mp4/confirm_01", "status": "RUNNING"})
    store.save()
    reopened = joint.Store(store.output)
    assert reopened.data["payload_read_calls"][-1]["status"] == "INTERRUPTED"
    assert "payload_read_calls/0" in reopened.data["resume_recoveries"][-1]["retired_running"]


def test_initialize_requires_saved_payload_truth_and_retains_missing_case(tmp_path, monkeypatch):
    source, config = fixture_inputs(tmp_path, monkeypatch)
    candidate = source / "run_state/rgb/confirm_02/residual_clock_pre.rgb8"
    candidate.unlink()
    store = joint.initialize(source, tmp_path / "retained-output", config)
    clean = joint._row(store, "CLEAN_MP4", "confirm_02")
    attacked = joint._row(store, "DELETE_G0", "confirm_02")
    assert clean["status"] == "FAILED" and "FileNotFoundError" in clean["reason"]
    assert attacked["status"] == "BLOCKED"
    assert len(store.data["rows"]) == 8

    changed = json.loads(config.read_text())
    changed["payload_bits"][0] = 1 - changed["payload_bits"][0]
    mismatch = tmp_path / "mismatch.json"
    write_json(mismatch, changed)
    with pytest.raises(ValueError, match="payload_bits differ"):
        joint.initialize(source, tmp_path / "mismatch-output", mismatch)


def test_fresh_kernel_environment_sees_wan_work_after_clock_latent_resume(tmp_path, monkeypatch):
    store = fixture(tmp_path, monkeypatch)
    row = joint._row(store, "CLEAN_MP4", "confirm_01")
    row["media"].update(status="AVAILABLE", received_path="/saved/received.rgb8")
    row["clock"].update(status="INTERRUPTED", reason="prior kernel ended")
    row["status"] = "INTERRUPTED"
    latent = joint._clock_latent_path(store, row)
    latent.parent.mkdir(parents=True, exist_ok=True)
    latent.write_bytes(b"durable-framewise-latent")
    units = joint.pending_model_units(store)
    assert "framewise/clean_mp4/confirm_01" not in units
    assert "wan-after-clock/clean_mp4/confirm_01" in units


def test_stub_pipeline_keeps_eight_rows_and_actual_attempts(tmp_path, monkeypatch):
    store = fixture(tmp_path, monkeypatch)

    monkeypatch.setattr(joint, "_native_source_path", lambda store, case: (Path("/saved/native.rgb8"), Path("/saved/state.json")))
    monkeypatch.setattr(joint, "_read_rgb_frames", lambda path, indices: {
        index: np.zeros((320, 512, 3), np.uint8) for index in set(indices)
    })
    joint.run_quality(store)
    assert store.data["quality_inspection"]["status"] == "COMPLETE"
    assert Path(store.data["quality_inspection"]["contact_sheet"]).is_file()

    def load_rgb(path, shape):
        return torch.zeros(tuple(shape), dtype=torch.uint8)

    def roundtrip(source, output, *, media_output):
        output.mkdir(parents=True); media_output.mkdir(parents=True)
        received = output / "received.rgb8"; received.write_bytes(bytes(181 * 3))
        mp4 = media_output / "published.mp4"; mp4.write_bytes(b"stub")
        return source.clone(), {"mp4": {"path": str(mp4)}, "received_rgb": {"path": str(received)}, "codec": joint.CODEC}

    def apply_edit(source, spec):
        assert spec == joint.delete_spec()
        return np.zeros((163, 1, 1, 3), np.uint8), {"attack_id": "delete_g0", "truth_is_posthoc_only": True}

    joint.run_media(store, roundtrip=roundtrip, apply_edit=apply_edit, load_rgb=load_rgb)

    class Framewise:
        def encode(self, rgb, *, batch_observer):
            for start in range(0, len(rgb), 8):
                stop = min(start + 8, len(rgb))
                batch_observer("STARTED", start, stop, None)
                batch_observer("RETURNED", start, stop, None)
            return np.zeros((len(rgb), 4, 40, 64), np.float32)
        def close(self): pass

    seen = []
    def select_clock(latent, key):
        assert key == "fixed-k0"
        frames = len(latent)
        path = list(range(181)) if frames == 181 else [index for index in range(181) if index not in joint.DELETE_POSITIONS]
        if len(seen) == 0:
            path[-1] = 179
        seen.append((frames, key))
        operation = {
            "status": "SUPPORTED", "received_index_map": [0], "source_coordinate_map": [0],
            "output_frames": 1, "wan_support": 1, "truth_inputs": False,
        }
        selection = {
            "status": "SUPPORTED", "reason": None, "operation": operation,
            "estimate": {"status": "ESTIMATED", "reason": None, "path": path,
                         "best_path": path, "runner_up_path": None, "score_gap": 1.0},
        }
        evidence = {"signed_projection": np.zeros((frames, 181)), "rho": np.ones((frames, 181))}
        return evidence, selection

    monkeypatch.setattr(joint, "_atomic_npy", lambda path, value: None)
    joint.run_clock(store, load_backend=Framewise, load_rgb=load_rgb, select_clock=select_clock)
    assert seen == [(181, "fixed-k0")] * 4 + [(163, "fixed-k0")] * 4
    intermediate = json.loads((store.output / "evaluation_report.json").read_text())
    assert intermediate["summary"]["status"] == "IN_PROGRESS"
    assert intermediate["summary"]["truth_join_status"] == "NOT_JOINED_BEFORE_WAN_TERMINAL"
    assert all(row["path_metrics"]["status"] == "NOT_JOINED_BEFORE_WAN_TERMINAL"
               for row in intermediate["rows"])
    state_after_clock = json.loads((store.output / "residual_joint_state.json").read_text())
    assert all("posthoc_path_metrics" not in row for row in state_after_clock["rows"])

    def encode(vae, rgb, observer):
        observer("STARTED", None)
        value = torch.zeros((1, 16, 1, 40, 64), dtype=torch.float32)
        observer("RETURNED", None)
        return value

    truth = [index % 2 for index in range(32)]
    def payload_read(latent, key, output_frames, source_map):
        assert (key, output_frames, source_map) == ("fixed-k0", 1, [0])
        return {
            "status": "READ", "R": 1, "output_frames": 1, "decoded_bits": list(truth),
            "votes": [{"ones": bit, "zeros": 1 - bit, "count": 1} for bit in truth],
            "signed_votes": np.zeros((4, 1, 240), np.int8),
            "zero_mask": np.ones((4, 1, 240), bool), "bit_rows": [], "time_bit_rows": [],
            "original_reader_match": True, "truth_inputs": False,
        }

    joint.run_wan(store, load_vae=lambda: object(), encode=encode, payload_read=payload_read, load_rgb=load_rgb)
    summary = joint.report(joint.Store(store.output))
    assert summary["status"] == "COMPLETE"
    assert summary["groups"]["CLEAN_MP4"]["payload_exact_recoveries"] == 4
    assert summary["groups"]["CLEAN_MP4"]["clock_exact_recoveries"] == 3
    assert summary["groups"]["CLEAN_MP4"]["joint_exact_recoveries"] == 3
    assert summary["groups"]["CLEAN_MP4"]["exact_recoveries"] == 3
    assert summary["groups"]["DELETE_G0"]["exact_recoveries"] == 4
    assert summary["actual_calls"]["codec_encode_and_readback"] == {"COMPLETE": 4}
    assert summary["actual_calls"]["peak_quality"] == {"COMPLETE": 1}
    assert summary["actual_calls"]["cpu_delete_edits"] == {"COMPLETE": 4}
    assert summary["actual_calls"]["framewise_physical_batches"] == {"COMPLETE": 176}
    assert summary["actual_calls"]["wan_encode_clips"] == {"COMPLETE": 8}
    assert summary["actual_calls"]["payload_reads"] == {"COMPLETE": 8}
    report = json.loads((store.output / "evaluation_report.json").read_text())
    assert len(report["rows"]) == 8
    wrong = next(row for row in report["rows"] if row["slot_id"] == "clean_mp4/confirm_01")
    assert wrong["payload_exact"] is True and wrong["clock_exact"] is False and wrong["joint_exact"] is False
    assert all(row["truth_join"] == "POSTHOC_COMPLETE_AFTER_WAN_TERMINAL" for row in report["rows"])
    final_state = json.loads((store.output / "residual_joint_state.json").read_text())
    assert all("posthoc_path_metrics" not in row for row in final_state["rows"])


@pytest.mark.parametrize("invalid", ("nonfinite_latent", "nonbinary_bits"))
def test_wan_rejects_nonfinite_latent_and_nonbinary_bits(tmp_path, monkeypatch, invalid):
    store = fixture(tmp_path, monkeypatch)
    row = joint._row(store, "CLEAN_MP4", "confirm_01")
    received = tmp_path / "received.rgb8"
    received.write_bytes(bytes(181 * 3))
    selector = tmp_path / "selector.json"
    write_json(selector, {
        "estimate": {"status": "ESTIMATED", "path": list(range(181))},
        "operation": {"status": "SUPPORTED", "received_index_map": [0],
                      "source_coordinate_map": [0], "output_frames": 1},
    })
    row["media"].update(status="AVAILABLE", received_path=str(received))
    row["clock"].update(status="CLOCK_READY", selector_record=str(selector))
    row["status"] = "CLOCK_READY"
    store.save()

    payload_called = []
    def encode(vae, rgb, observer):
        observer("RETURNED", None)
        fill = float("nan") if invalid == "nonfinite_latent" else 0.0
        return torch.full((1, 16, 1, 40, 64), fill, dtype=torch.float32)

    def payload_read(latent, key, output_frames, source_map):
        payload_called.append(True)
        return {"status": "READ", "decoded_bits": [2] * 32}

    joint.run_wan(
        store, load_vae=lambda: object(), encode=encode, payload_read=payload_read,
        load_rgb=lambda path, shape: torch.zeros(tuple(shape), dtype=torch.uint8),
    )
    failed = joint._row(store, "CLEAN_MP4", "confirm_01")
    assert failed["payload"]["status"] == "FAILED"
    if invalid == "nonfinite_latent":
        assert not payload_called
        assert "nonfinite" in failed["payload"]["reason"]
    else:
        assert payload_called == [True]
        assert "binary integer bits" in failed["payload"]["reason"]
        assert store.data["payload_read_calls"][-1]["status"] == "FAILED"


def test_builder_has_direct_runall_and_portable_entrypoints():
    builder.build()
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    text = "\n".join("".join(cell["source"]) for cell in code)
    assert '"--phase", "quality"' in text and '"--phase", "media"' in text and '"--phase", phase' in text
    assert "if PENDING_MODEL_UNITS:" in text
    assert "peak_quality_contact_sheet" not in text  # display follows the state-recorded path, not a hidden locator
    assert "SOURCE_SHA" not in text and "dirty" not in text and "strict" not in text
    import zipfile
    with zipfile.ZipFile(builder.ZIP) as archive:
        names = set(archive.namelist())
        with tempfile.TemporaryDirectory() as directory:
            archive.extractall(directory)
            for name in names:
                if name.endswith(".py"):
                    compile((Path(directory) / name).read_text(encoding="utf-8"), name, "exec")
            child = subprocess.run(
                [sys.executable, "-m", "experiments.paper_results_v1.residual_joint_v1_cli", "--help"],
                cwd=directory, capture_output=True, text=True,
            )
            assert child.returncode == 0, child.stderr
    assert "experiments/paper_results_v1/residual_joint_v1_cli.py" in names
    assert "experiments/paper_results_v1/native_adapters.py" in names
    assert "runtime/wan/video_trajectory_temporal_edit_receiver_v1.py" in names
