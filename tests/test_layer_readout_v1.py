"""CPU-only persistence, semantic reuse, and notebook checks for layer readout."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import layer_readout_v1 as layer


def fake_result(*_args):
    signs = np.ones((4, 44, 240), dtype=np.int8)
    signs[0, 0, 0] = 0
    zeros = signs == 0
    votes = []
    bits = []
    for bit in range(32):
        channel, slot = divmod(bit, 8)
        values = signs[channel, :, slot::8].reshape(-1)
        ones = int((values > 0).sum())
        nonpositive = int(values.size - ones)
        votes.append({"ones": ones, "zeros": nonpositive, "count": int(values.size)})
        bits.append(int(ones > nonpositive))
    return {
        "status": "READ", "R": 44, "output_frames": 181, "decoded_bits": bits,
        "votes": votes, "bit_rows": [], "signed_votes": signs, "zero_mask": zeros,
        "original_reader_match": True, "truth_inputs": False,
    }


def fixture(tmp_path, source_name=layer.SOURCE_RUN_ID):
    source = tmp_path / source_name
    (source / "run_state").mkdir(parents=True)
    config = {
        "keys": {"K0": "fixed-key"}, "payload_bits": [1] * 32,
        "models": {"wan": {"local_snapshot_path": "/unused", "device": "cpu"}},
    }
    layer.atomic_json(source / "effective_confirmation_config.json", config)
    layer.atomic_json(source / "run_state/run_state.json", {"artifacts": []})
    runtime_config = tmp_path / "runtime-config.json"
    layer.atomic_json(runtime_config, config)
    return layer.initialize(source, tmp_path / "out", runtime_config)


def complete_reuse(store):
    layer.reuse(
        store,
        load_terminal=lambda row: object(),
        payload_read=lambda *args: fake_result(),
        post_result=lambda row: (fake_result(), {"kind": "TEST_REUSED_POST"}),
    )


def test_encode_commit_resume_and_complete_rerun_skip(tmp_path):
    store = fixture(tmp_path)
    complete_reuse(store)
    loads = []
    encodes = []
    reads = []

    def payload(*args):
        reads.append(1)
        if len(reads) == 12:
            raise KeyboardInterrupt("after twelfth latent commit")
        return fake_result()

    with pytest.raises(KeyboardInterrupt):
        layer.read(
            store, load_vae=lambda: loads.append(1) or object(),
            encode=lambda vae, rgb, encode_observer: (encode_observer("STARTED", None), encodes.append(1), encode_observer("RETURNED", None), torch.zeros(1, 16, 46, 40, 64))[-1],
            payload_read=payload, probe_rgb=lambda row: True,
            load_rgb=lambda row: torch.zeros(1, 1, 1, 3, dtype=torch.uint8),
        )
    assert len(loads) == 1 and len(encodes) == 12
    interrupted = layer.Store(store.output)
    final_row = interrupted.data["rows"][-2]  # confirm_03/M05_PRE precedes M05_POST
    assert final_row["status"] == "ENCODED"
    assert Path(final_row["latent_path"]).is_file()
    # Simulate loss of the row update after the atomic latent rename. The next
    # process still discovers the deterministic durable path and must not encode.
    final_row.pop("latent_path")
    final_row["status"] = "INTERRUPTED"
    interrupted.save()
    interrupted = layer.Store(store.output)
    assert layer.pending_model_rows(interrupted) == []

    layer.read(
        interrupted,
        load_vae=lambda: pytest.fail("durable latent caused a second VAE load"),
        encode=lambda *args: pytest.fail("durable latent caused a repeated encode"),
        payload_read=lambda *args: fake_result(), probe_rgb=lambda row: True,
        load_rgb=lambda row: pytest.fail("durable latent caused RGB reread"),
    )
    completed = layer.Store(store.output)
    summary = layer.report(completed)
    assert summary["status"] == "COMPLETE"
    assert summary["slot_attempts"] == {"attempted": 12, "statuses": {"COMPLETE": 12}}
    assert summary["actual_wan_encode_calls"] == {"attempted": 12, "statuses": {"COMPLETE": 12}}
    assert summary["actual_vae_loads"] == {"attempted": 1, "statuses": {"COMPLETE": 1}}
    assert all(row["status"] == "EVALUATED" for row in completed.data["rows"])
    with np.load(completed.data["rows"][0]["vote_sidecar"], allow_pickle=False) as arrays:
        assert arrays["signed_votes"].shape == (4, 44, 240)
        assert arrays["zero_mask"][0, 0, 0] and arrays["signed_votes"][0, 0, 0] == 0

    layer.read(
        completed,
        load_vae=lambda: pytest.fail("complete rerun loaded VAE"),
        encode=lambda *args: pytest.fail("complete rerun encoded"),
        payload_read=lambda *args: pytest.fail("complete rerun reread payload"),
        probe_rgb=lambda row: pytest.fail("complete rerun probed RGB"),
        load_rgb=lambda row: pytest.fail("complete rerun loaded RGB"),
    )
    assert layer.report(layer.Store(store.output))["actual_wan_encode_calls"]["attempted"] == 12
    damaged = layer.Store(store.output)
    lost = Path(damaged.data["rows"][0]["vote_sidecar"])
    lost.unlink()
    damaged_summary = layer.report(damaged)
    assert len(damaged.data["rows"]) == 20
    assert damaged_summary["status_counts"] == {"MISSING": 1, "EVALUATED": 19}
    assert damaged_summary["actual_wan_encode_calls"]["attempted"] == 12
    damaged_row = damaged.data["rows"][0]
    assert damaged_row["missing_bits"] == 32 and damaged_row["previous_evaluation"]["exact_recovery"] is True
    assert damaged_summary["exact_rows"] == 19


def test_missing_pre_is_retained_without_fallback_or_model_load(tmp_path):
    store = fixture(tmp_path)
    complete_reuse(store)
    pending = next(row for row in store.data["rows"] if row["layer"] in layer.ENCODE_LAYERS)
    for row in store.data["rows"]:
        if row["layer"] in layer.ENCODE_LAYERS and row is not pending:
            layer._save_read(store, row, fake_result(), {"kind": "TEST_PREEXISTING"})
    layer.read(
        store,
        load_vae=lambda: pytest.fail("missing-only run loaded VAE"),
        encode=lambda *args: pytest.fail("missing input fell back to encode"),
        payload_read=lambda *args: fake_result(),
        probe_rgb=lambda row: (_ for _ in ()).throw(FileNotFoundError("saved PRE absent")),
        load_rgb=lambda row: pytest.fail("missing input was loaded"),
    )
    current = layer.Store(store.output)
    target = next(row for row in current.data["rows"] if row["slot_id"] == pending["slot_id"])
    assert target["status"] == "MISSING"
    assert len(current.data["rows"]) == 20
    summary = layer.report(current)
    assert summary["status_counts"] == {"EVALUATED": 19, "MISSING": 1}
    assert summary["actual_vae_loads"]["attempted"] == 0
    assert summary["actual_codec_or_media_calls"] == 0
    history_count = len(target["resume_history"])
    layer.read(
        current,
        load_vae=lambda: pytest.fail("retained MISSING row loaded VAE on rerun"),
        encode=lambda *args, **kwargs: pytest.fail("retained MISSING row encoded on rerun"),
        payload_read=lambda *args: pytest.fail("retained MISSING row was reread"),
        probe_rgb=lambda row: pytest.fail("retained MISSING row was reprobed"),
        load_rgb=lambda row: pytest.fail("retained MISSING row was reloaded"),
    )
    retained = next(row for row in layer.Store(store.output).data["rows"] if row["slot_id"] == pending["slot_id"])
    assert retained["status"] == "MISSING" and len(retained["resume_history"]) == history_count


def test_source_directory_may_be_relocated_and_renamed(tmp_path):
    store = fixture(tmp_path, source_name="renamed-source-run-copy")
    source = Path(store.data["source_run"])
    run_file = source / "run_state/artifacts/example.bin"
    receiver_file = source / "confirmation_receiver/physical/example.npz"
    run_file.parent.mkdir(parents=True); run_file.write_bytes(b"rgb")
    receiver_file.parent.mkdir(parents=True); receiver_file.write_bytes(b"votes")
    old_root = "/content/drive/MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1/" + layer.SOURCE_RUN_ID
    assert store.resolve(old_root + "/run_state/artifacts/example.bin") == run_file
    assert store.resolve(old_root + "/confirmation_receiver/physical/example.npz") == receiver_file


@pytest.mark.parametrize("mutation", ["key", "R", "map"])
def test_post_reuse_rejects_wrong_fixed_method_fields(tmp_path, mutation):
    store = fixture(tmp_path)
    folder = Path(store.data["source_run"]) / "confirmation_receiver"
    folder.mkdir()
    sidecar = folder / "votes.npz"
    values = fake_result()
    layer._atomic_npz(sidecar, signed_votes=values["signed_votes"], zero_mask=values["zero_mask"])
    physical = folder / "physical.json"
    layer.atomic_json(physical, {
        "R": 44, "output_frames": 181, "truth_inputs": False,
        "original_reader_match": True, "decoded_bits": values["decoded_bits"],
        "votes": values["votes"], "bit_rows": [], "vote_sidecar": {"path": str(sidecar)},
    })
    detail = folder / "detail.json"
    source_row = {
        "slot_id": "saved", "case_id": "confirm_01", "arm": "PAYLOAD_FRAMEWISE_M05",
        "attack_id": "full", "key_label": "K0", "receiver": "CENTERED_U",
        "status": "EVALUATED", "truth_inputs": False, "R": 44,
        "operation": {"wan_support": 44, "output_frames": 181,
                      "source_coordinate_map": list(layer.IDENTITY_MAP),
                      "received_index_map": list(layer.IDENTITY_MAP)},
        "detail_record": {"path": str(detail)}, "decoded_bits": values["decoded_bits"],
        "votes": values["votes"], "original_reader_match": True,
        "physical_read_record": {"path": str(physical)},
    }
    layer.atomic_json(detail, {"slot_id": source_row["slot_id"], "source_coordinate_map": list(layer.IDENTITY_MAP),
                               "physical_read_record": str(physical)})
    if mutation == "key": source_row["key_label"] = "K1"
    elif mutation == "R": source_row["R"] = 43
    else: source_row["operation"]["received_index_map"][10] = 9
    layer.atomic_json(folder / "evaluation_report.json", {"receiver_rows": [source_row]})
    row = next(row for row in store.data["rows"] if row["slot_id"] == "confirm_01/M05_POST")
    with pytest.raises((ValueError, FileNotFoundError)):
        layer._post_result(store, row)


def test_notebook_companion_and_portable_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/paper_results_v1_layer_readout_v1_colab.ipynb").read_text())
    assert "".join(notebook["cells"][0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert "force_remount" not in json.dumps(notebook)
    setup = "".join(next(item for item in notebook["cells"] if item["id"] == "initialize-reuse")["source"])
    assert setup.index("CPU_IMPORT_PROBE") < setup.index('[*CLI, "--phase", "reuse"]')
    assert "from experiments" not in setup
    assert 'json.loads((RUN_OUTPUT / "summary.json").read_text())["pending_model_rows"]' in setup
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
            assert item["outputs"] == [] and item["execution_count"] is None
    with zipfile.ZipFile(root / "notebooks/paper_results_v1_layer_readout_v1_companion.zip") as archive:
        for name in archive.namelist():
            assert archive.read(name) == (root / name).read_bytes()
        archive.extractall(tmp_path / "portable")
    result = subprocess.run(
        [sys.executable, "-B", str(tmp_path / "portable/experiments/paper_results_v1/layer_readout_v1_cli.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    # A fresh parent cwd has no repository on sys.path. The extracted child CLI
    # still initializes the fixed plan and publishes pending work in summary JSON.
    fresh = tmp_path / "fresh-parent"; fresh.mkdir()
    child_output = fresh / "child-output"
    source = Path(fixture(tmp_path / "cli-fixture").data["source_run"])
    child_config = source.parent / "runtime-config.json"
    child = subprocess.run(
        [sys.executable, "-B", str(tmp_path / "portable/experiments/paper_results_v1/layer_readout_v1_cli.py"),
         "--output", str(child_output), "--source-run", str(source), "--config", str(child_config), "--phase", "init"],
        cwd=fresh, capture_output=True, text=True,
    )
    assert child.returncode == 0, child.stderr
    child_summary = json.loads((child_output / "summary.json").read_text())
    assert len(child_summary["pending_model_rows"]) == 12
    assert all(slot.endswith(("NATIVE_PRE", "RECON_PRE", "M05_PRE")) for slot in child_summary["pending_model_rows"])

    environment = "".join(next(item for item in notebook["cells"] if item["id"] == "environment")["source"])
    events = []
    scope = {
        "PENDING_MODEL_ROWS": [],
        "logged": lambda *args, **kwargs: pytest.fail("finished rerun entered Wan dependency work"),
        "record": lambda *args, **kwargs: events.append((args, kwargs)),
    }
    exec(compile(environment, "environment", "exec"), scope)
    assert scope["environment_status"] == "SKIPPED_NO_PENDING_ENCODE"
    assert events == [(('environment', 'SKIPPED_NO_PENDING_ENCODE'), {'pending_model_rows': 0})]
