"""CPU/fake/static checks; no model, GPU, MP4 codec, or Drive execution."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split_receiver
from main.tube_state import projection_margin as carrier
from runtime.wan import media_channel_retention as retention
from runtime.wan.vae import quantize_rgb8_no_codec
from experiments.wan_state_clock import media_channel_retention_run as run
from scripts import build_media_channel_retention_notebook as builder

pytestmark = pytest.mark.unit
SOURCE_TOKEN = "1" * 40
FIXTURE = Path(__file__).parent / "fixtures" / "media_channel_retention_eval_p2_off.json"


def _measurement(value=0.0):
    return {
        "status": "SCORED",
        "partition": "TOTAL",
        "score": value,
        "matched_score": value,
        "state_innovation_mean": abs(value),
        "q_by_window": [[value, -value] for _ in range(11)],
        "windows": [{"window": index, "valid": True, "q": [value, -value]} for index in range(11)],
        "failure": None,
    }


def _score_row(name, matched, q, innovation):
    total = name == "TOTAL"
    return {
        "status": "SCORED",
        "path": fixed_key.REFERENCE_PATHS["IDENTITY"],
        "score": matched - 0.05 * innovation,
        "matched_score": matched,
        "state_innovation_mean": innovation,
        "innovation_by_window": [innovation] * 11,
        "matched_windows": list(range(11)),
        "matched_components": 7040 if total else 3520,
        "full_blocks": 1760 if total else 880,
        "partial_blocks": 0,
        "windows": [
            {
                "window": index, "phase": 0, "selected": [4 * index + offset for offset in range(4)],
                "group_count": 4, "valid": True,
                "observed_components": 640 if total else 320,
                "support_kind": "FULL", "signed_evidence": matched,
                "q": list(q), "q_norm": 0.0,
            }
            for index in range(11)
        ],
    }


def test_manifest_fixed_denominators_partition_axes_and_no_deployment_receiver():
    config = run.load(run.MANIFEST)
    run.validate_manifest(config)
    assert config["call_plan"] == {"vae_encode": 20}
    assert config["fixed_denominator"] == {
        "trajectories": 10,
        "layer_partition_slots": 120,
        "marked_off_increment_slots": 96,
        "adjacent_increment_change_slots": 72,
    }
    assert retention.partition_contract() == {
        "A": {"support_count": 80, "q_axis_0_count": 40, "q_axis_1_count": 40},
        "B": {"support_count": 80, "q_axis_0_count": 40, "q_axis_1_count": 40},
    }
    source = Path(run.__file__).read_text()
    assert "fixed_key.read(" not in source
    assert "split_receiver.read(" not in source
    tree = ast.parse(source)
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "read_mp4" not in called_names and "encode_rgb" not in called_names
    assert Path(split_receiver.__file__) in run._source_files()


def test_real_audited_fixture_field_paths_are_consumed_without_invented_hashes():
    fixture = json.loads(FIXTURE.read_text())
    references = run._old_artifact_references(fixture["arm"])
    assert fixture["source_commit"] == "8aff4025fd0f8dd656091bf83af1311263f4d99c"
    assert references == {
        "terminal_fingerprint": "bfa52f6bd0e3381689647e12321aa92e772eaf59bff07f88bb52f1d5278d4c59",
        "terminal_path": "OFF_terminal.pt",
        "full_mp4_sha256": "030da23e7f23ac46a1731d5d4962241479a164f1482b21589a9bbe35fc1f2d02",
        "full_mp4_path": "received_videos/OFF/FULL.mp4",
        "full_mp4_frames": 181,
        "full_g0_sha256": "eda26060c9fabc6a2f40de42d455e44d5c92a1aad055df30d26de4fdea5b4034",
        "full_g0_path": "observations/OFF/FULL/g0.pt",
        "full_g0_status": "COMPLETE",
        "full_g0_frames_used": 181,
        "full_g0_tail_discarded": 0,
    }
    assert fixture["arm"]["views"]["FULL"]["observations"]["0"]["frames_used"] == 181
    assert run._recorded_metadata_audit("OFF", references)["status"] == "VERIFIED"


def test_total_and_partitions_are_scored_independently_not_reconstructed(monkeypatch):
    calls = []

    def total_score(observations, book, path):
        calls.append("TOTAL")
        return _score_row("TOTAL", 0.3, (0.2, 0.4), 0.91)

    class Partition:
        def __init__(self, observations, book, indices, name):
            self.name = name

        def score_path(self, path):
            calls.append(self.name)
            return (
                _score_row("A", 0.1, (0.0, 0.2), 0.5)
                if self.name == "A"
                else _score_row("B", 0.5, (0.4, 0.6), 0.7)
            )

    monkeypatch.setattr(fixed_key, "score_path", total_score)
    monkeypatch.setattr(split_receiver, "PartitionEvidence", Partition)
    result = retention.score_identity_layer(
        torch.zeros(1), {}, fixed_key.REFERENCE_PATHS["IDENTITY"], [1],
    )
    rows = result["partitions"]
    assert calls == ["TOTAL", "A", "B"]
    assert rows["TOTAL"]["matched_score"] == pytest.approx(
        (rows["A"]["matched_score"] + rows["B"]["matched_score"]) / 2,
    )
    assert rows["TOTAL"]["q_by_window"][0] == pytest.approx([
        (rows["A"]["q_by_window"][0][axis] + rows["B"]["q_by_window"][0][axis]) / 2
        for axis in (0, 1)
    ])
    assert rows["TOTAL"]["state_innovation_mean"] == 0.91
    assert rows["TOTAL"]["state_innovation_mean"] != pytest.approx(
        (rows["A"]["state_innovation_mean"] + rows["B"]["state_innovation_mean"]) / 2,
    )


def test_real_fixed_shape_identity_matches_nominal_and_equal_partition_support():
    tensor = torch.zeros(carrier.SHAPE, dtype=torch.float16)
    values = torch.linspace(-0.2, 0.2, 44, dtype=torch.float16)
    tensor[0, 0, 1:45, 0, 0] = values
    tensor[0, 3, 1:45, 7, 11] = values.flip(0) * 0.7
    book = fixed_key.codebook(b"WanProjection-first-validation-key-v1")
    rows = retention.score_identity_layer(
        tensor, book, fixed_key.REFERENCE_PATHS["IDENTITY"], list(carrier.SHAPE),
    )["partitions"]
    nominal = fixed_key.nominal_record(tensor, book, include_state=False)
    assert rows["TOTAL"]["matched_score"] == pytest.approx(nominal["clipped_score"], abs=1e-15)
    assert rows["TOTAL"]["matched_score"] == pytest.approx(
        (rows["A"]["matched_score"] + rows["B"]["matched_score"]) / 2, abs=1e-15,
    )
    for window in range(11):
        for axis in (0, 1):
            assert rows["TOTAL"]["q_by_window"][window][axis] == pytest.approx(
                (
                    rows["A"]["q_by_window"][window][axis]
                    + rows["B"]["q_by_window"][window][axis]
                ) / 2,
                abs=1e-15,
            )
    assert rows["TOTAL"]["state_innovation_mean"] != pytest.approx(
        (rows["A"]["state_innovation_mean"] + rows["B"]["state_innovation_mean"]) / 2,
        abs=1e-12,
    )


def test_rgb8_roundtrip_uses_the_shared_np_rint_raster():
    rgb = torch.tensor(
        [[[[0.0, 0.1, 0.5], [0.501, 0.9, 1.0]]]], dtype=torch.float32,
    )
    first = quantize_rgb8_no_codec(rgb)
    roundtrip = first.float().div(255.0)
    second = quantize_rgb8_no_codec(roundtrip)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_drive_hash_read_error_isolated_and_next_file_still_audited(monkeypatch, tmp_path):
    bad = tmp_path / "bad.pt"
    good = tmp_path / "good.pt"
    bad.write_bytes(b"bad")
    good.write_bytes(b"good")
    real_sha = run.sha256

    def flaky(path):
        if Path(path) == bad:
            raise OSError("Drive read interrupted")
        return real_sha(path)

    monkeypatch.setattr(run, "sha256", flaky)
    bad_row = run._file_audit(bad, "0" * 64)
    good_row = run._file_audit(good, real_sha(good))
    assert bad_row["status"] == "FAILED"
    assert bad_row["historical_hash_verification"] == "FILE_HASH_READ_FAILED"
    assert good_row["status"] == "VERIFIED"


def test_case_stub_reuses_one_lazy_vae_and_attempts_exactly_ten_encodes(monkeypatch, tmp_path):
    config = copy.deepcopy(json.loads(run.MANIFEST.read_text()))
    config["expected_shapes"] = {"latent": [1], "decoded_rgb": [2, 1, 1, 3]}
    old = {
        "cases": {
            "eval_p2_s2": {
                "source_commit": config["input_source_commit"],
                "videos": {
                    arm: {
                        "terminal_fingerprint": "fp",
                        "terminal_path": f"{arm}_terminal.pt",
                        "views": {"FULL": {
                            "sha256": "mp4", "path": f"received_videos/{arm}/FULL.mp4", "frames": 181,
                            "observations": {"0": {
                                "sha256": "g0", "path": f"observations/{arm}/FULL/g0.pt",
                                "status": "COMPLETE", "frames_used": 181, "tail_discarded": 0,
                            }},
                        }},
                    }
                    for arm in config["arms"]
                },
            }
        }
    }
    latent = torch.zeros(1)
    rgb = torch.linspace(0, 1, 6).reshape(2, 1, 1, 3)
    calls = {"load": 0, "encode": 0}

    monkeypatch.setattr(run, "load", lambda path: config)
    monkeypatch.setattr(run, "validate_manifest", lambda value: None)
    monkeypatch.setattr(run, "_root_input_audit", lambda *args: ({"status": "VERIFIED"}, old))
    monkeypatch.setattr(
        run, "_terminal_audit",
        lambda *args: ({"status": "VERIFIED", "actual_sha256": "terminal"}, latent),
    )

    def file_audit(path, *args, **kwargs):
        return {
            "status": "PRESENT" if "decoded_rgb" in str(path) else "VERIFIED",
            "actual_sha256": "current", "historical_hash_verification": (
                "UNAVAILABLE_NO_RECORDED_HASH" if "decoded_rgb" in str(path) else "MATCH"
            ),
        }

    monkeypatch.setattr(run, "_file_audit", file_audit)
    monkeypatch.setattr(
        run.torch, "load", lambda path, **kwargs: rgb if "decoded_rgb" in str(path) else latent,
    )
    monkeypatch.setattr(fixed_key, "codebook", lambda key: {})

    def score(layer_row, tensor, supplied_config, book):
        layer_row.update(
            status="SCORED",
            measurements={name: _measurement(0.1) | {"partition": name} for name in retention.PARTITIONS},
            failure=None,
        )

    monkeypatch.setattr(run, "_score_layer", score)

    def load_vae(protocol):
        calls["load"] += 1
        return object()

    def encode(vae, pixels):
        calls["encode"] += 1
        return latent

    monkeypatch.setattr(run, "load_frozen_vae", load_vae)
    monkeypatch.setattr(run.fixed_key_core, "load_protocol", lambda: {})
    monkeypatch.setattr(run, "reencode_rgb24_readback", encode)
    monkeypatch.setattr(run, "release", lambda *args: None)
    result = run.run_case("eval_p2_s2", "config.json", tmp_path / "out")
    assert result["status"] == "EXECUTION_COMPLETE"
    assert calls == {"load": 1, "encode": 10}
    assert result["call_accounting"]["vae_encode"] == {
        "expected": 10, "attempted": 10, "completed": 10, "status": "EXACT",
    }
    assert all(row["status"] == "SCORED" for row in result["trajectories"].values())


def test_vae_load_failure_does_not_count_encode_attempts_and_keeps_all_slots(monkeypatch, tmp_path):
    config = copy.deepcopy(json.loads(run.MANIFEST.read_text()))
    config["expected_shapes"] = {"latent": [1], "decoded_rgb": [2, 1, 1, 3]}
    arm_record = lambda arm: {
        "terminal_fingerprint": "fp", "terminal_path": f"{arm}_terminal.pt",
        "views": {"FULL": {
            "sha256": "mp4", "path": f"received_videos/{arm}/FULL.mp4", "frames": 181,
            "observations": {"0": {
                "sha256": "g0", "path": f"observations/{arm}/FULL/g0.pt",
                "status": "COMPLETE", "frames_used": 181, "tail_discarded": 0,
            }},
        }},
    }
    old = {"cases": {"eval_p2_s2": {
        "source_commit": config["input_source_commit"],
        "videos": {arm: arm_record(arm) for arm in config["arms"]},
    }}}
    latent = torch.zeros(1)
    rgb = torch.zeros(2, 1, 1, 3)
    loads = []

    monkeypatch.setattr(run, "load", lambda path: config)
    monkeypatch.setattr(run, "validate_manifest", lambda value: None)
    monkeypatch.setattr(run, "_root_input_audit", lambda *args: ({"status": "VERIFIED"}, old))
    monkeypatch.setattr(
        run, "_terminal_audit",
        lambda *args: ({"status": "VERIFIED", "actual_sha256": "terminal"}, latent),
    )
    monkeypatch.setattr(run, "_file_audit", lambda path, *args, **kwargs: {
        "status": "PRESENT" if "decoded_rgb" in str(path) else "VERIFIED",
        "actual_sha256": "current", "historical_hash_verification": "MATCH",
    })
    monkeypatch.setattr(
        run.torch, "load", lambda path, **kwargs: rgb if "decoded_rgb" in str(path) else latent,
    )
    monkeypatch.setattr(fixed_key, "codebook", lambda key: {})
    monkeypatch.setattr(run, "_score_layer", lambda row, *args: row.update(
        status="SCORED",
        measurements={name: _measurement(0.1) | {"partition": name} for name in retention.PARTITIONS},
        failure=None,
    ))

    def fail_load(protocol):
        loads.append(True)
        raise RuntimeError("stub VAE load failure")

    monkeypatch.setattr(run, "load_frozen_vae", fail_load)
    monkeypatch.setattr(run.fixed_key_core, "load_protocol", lambda: {})
    monkeypatch.setattr(
        run, "reencode_rgb24_readback",
        lambda *args: (_ for _ in ()).throw(AssertionError("encode must not be called")),
    )
    monkeypatch.setattr(run, "release", lambda *args: None)
    case = run.run_case("eval_p2_s2", "config.json", tmp_path / "load-failure")
    assert len(loads) == 1
    assert case["call_accounting"]["vae_load"] == {"attempted": 1, "completed": 0}
    assert case["call_accounting"]["vae_encode"] == {
        "expected": 10, "attempted": 0, "completed": 0, "status": "MISMATCH",
    }
    master = {"cases": {"eval_p2_s2": case, "eval_p3_s3": copy.deepcopy(case)}}
    run._attach_pair_comparisons(master, config)
    accounting = run._slot_accounting(master, config)
    assert {name: row["retained"] for name, row in accounting.items()} == {
        "layer_partition_slots": 120,
        "marked_off_increment_slots": 96,
        "adjacent_increment_change_slots": 72,
    }


def test_output_resolve_guards_input_tree_and_existing_results(tmp_path):
    config = run.load(run.MANIFEST)
    input_root = Path(config["input_root"])
    with pytest.raises(ValueError, match="fixed input root"):
        run.run_all(run.MANIFEST, input_root)
    with pytest.raises(ValueError, match="fixed input root"):
        run.run_case("eval_p2_s2", run.MANIFEST, input_root / "child")

    notebook_output = tmp_path / "notebook-output"
    notebook_output.mkdir()
    (notebook_output / "setup_receipt.json").write_text("{}")
    assert run._prepare_output(notebook_output, input_root, "result.json") == notebook_output.resolve()
    (notebook_output / "result.json").write_text("{}")
    with pytest.raises(FileExistsError, match="existing result"):
        run.run_all(run.MANIFEST, notebook_output)

    case_output = tmp_path / "case-output"
    case_output.mkdir()
    (case_output / "case_result.json").write_text("{}")
    with pytest.raises(FileExistsError, match="existing result"):
        run.run_case("eval_p2_s2", run.MANIFEST, case_output)


def test_all_failure_slots_keep_the_fixed_denominators():
    config = run.load(run.MANIFEST)
    result = {"cases": {case_id: run._empty_case(config, case_id) for case_id in config["cases"]}}
    run._attach_pair_comparisons(result, config)
    accounting = run._slot_accounting(result, config)
    assert len(result["marked_off_increments"]) == 96
    assert len(result["adjacent_increment_changes"]) == 72
    assert len(result["terminal_to_rgb8_combined_increment_changes"]) == 24
    assert accounting == {
        "layer_partition_slots": {
            "expected": 120, "retained": 120, "complete": 0,
            "failed_or_undefined": 120, "status": "RETAINED",
        },
        "marked_off_increment_slots": {
            "expected": 96, "retained": 96, "complete": 0,
            "failed_or_undefined": 96, "status": "RETAINED",
        },
        "adjacent_increment_change_slots": {
            "expected": 72, "retained": 72, "complete": 0,
            "failed_or_undefined": 72, "status": "RETAINED",
        },
    }


def test_numeric_markers_and_sign_changes_are_retained_without_ratios():
    before = _measurement(-0.2)
    after = _measurement(0.3)
    comparison = retention.compare_measurements(after, before, 1e-12)
    score = comparison["scalars"]["score"]
    assert score["delta"] == pytest.approx(0.5)
    assert score["delta_marker"] == "POSITIVE"
    assert score["endpoint_sign_change"] is True
    assert retention.numeric_marker(-1e-13, 1e-12) == "ZERO_OR_NEAR_ZERO"
    assert "ratio" not in json.dumps(comparison).lower()


def test_notebook_builder_is_pinned_run_all_and_retains_nonzero_result(tmp_path):
    path = builder.build(SOURCE_TOKEN, tmp_path / "diagnostic.ipynb")
    notebook = json.loads(path.read_text())
    assert "".join(notebook["cells"][0]["source"]) == (
        "from google.colab import drive\ndrive.mount('/content/drive')"
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert SOURCE_TOKEN in text
    assert "torch==2.11.0" in text and "https://download.pytorch.org/whl/cu128" in text
    assert "diffusers==0.40.0" in text
    assert "'transformers'" in text and "transformers<" not in text
    assert "check=False" in text and "runner produced no retained result.json" in text
    assert "--mode" not in text and "MODE =" not in text
    assert "get_device_name(0)" in text and "A100" not in text
    assert notebook["metadata"]["source_commit"] == SOURCE_TOKEN
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
