"""CPU/fake checks for the fixed H0/H1 writer and experiment accounting."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from main.tube_state import fixed_key, fixed_key_split_receiver, projection_margin as carrier
from runtime.wan import fixed_key_control, keyed_presence_t40_t49 as writer, payload_control, trajectory
from experiments.wan_state_clock import keyed_presence_t40_t49_run as run
from scripts import build_keyed_presence_t40_t49_notebook as notebook_builder

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


class Scheduler:
    def __init__(self):
        self.config = SimpleNamespace(prediction_type="flow_prediction", thresholding=False, lower_order_final=True)
        self.predict_x0 = True
        self.timesteps = torch.arange(50, 0, -1)
        self.sigmas = torch.linspace(1, 0, 51)
        self.step_index = None
        self.history = []

    def step(self, velocity, timestep, z, return_dict=False):
        if self.step_index is None: self.step_index = 0
        self.history.append(float(z.mean()))
        self.step_index += 1
        return (z - .01 * velocity,)


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states, **kwargs):
        return (hidden_states * .1 + self.weight * .01,)


def test_fixed_manifest_and_denominators():
    config = run.load(run.MANIFEST)
    run.validate_manifest(config)
    assert [x["seed"] for x in config["cases"][:9]] == list(range(2026102401, 2026102410))
    assert [x["seed"] for x in config["cases"][9:]] == list(range(2026102301, 2026102305))
    assert config["receiver"]["primary"] == "C2_STATE_CONFIRM"
    assert config["receiver"]["view"] == "FULL"
    assert config["fixed_calls"] == dict(generation=13, transformer=1620, scheduler_step=810,
        zero_shadow_step=88, unit_response_probe_step=88, clean_leaf_backward=88,
        vae_decode=29, mp4_save=29, mp4_read=29, vae_encode=156)
    assert sum(len(writer.ARMS[x]) for x in writer.ARMS) == 22


def test_live_control_histories_and_last_tanh(monkeypatch):
    pipe = SimpleNamespace(scheduler=Scheduler(), transformer=Transformer())
    monkeypatch.setattr(writer, "prepare_generation", lambda *a, **k:
        (pipe, torch.zeros(carrier.SHAPE), torch.tensor(1.), torch.tensor(-1.), torch.float32))
    directions = []
    def direction(clean, book, count):
        index = len(directions)
        directions.append(clean.clone())
        count("clean_leaf_backward", False); count("clean_leaf_backward", True)
        raw = torch.ones_like(clean)
        return raw, {"objective": fixed_key_control.DEFAULT_OBJECTIVE, "temperature": 1}
    monkeypatch.setattr(fixed_key_control, "clean_direction", direction)
    calls = []
    result = writer.generate_terminals({"generation": {"guidance_scale": 5.0}}, b"fixed-key", tuple(writer.ARMS),
        lambda kind, done: calls.append((kind, done)))
    assert all(row["status"] == "GENERATED" for row in result["arms"].values())
    for kind, n in (("transformer", 180), ("scheduler_step", 90),
                    ("zero_shadow_step", 22), ("unit_response_probe_step", 22), ("clean_leaf_backward", 22)):
        assert calls.count((kind, True)) == n
    assert [r["index"] for r in result["arms"]["T40_49"]["control_steps"]] == list(range(40, 50))
    assert [r["index"] for r in result["arms"]["T49_ONLY"]["control_steps"]] == [49]
    for arm in writer.ARMS:
        for step in result["arms"][arm]["control_steps"]:
            assert step["actual_D"]["support_rms"] == pytest.approx(step["target_D_support_rms"], rel=2e-5)
    assert result["arms"]["T40_48"]["control_steps"][0]["controlled_next_fingerprint"] != result["arms"]["T40_49"]["control_steps"][0]["controlled_next_fingerprint"]
    assert result["arms"]["T40_49"]["control_steps"][-1]["sigma"] > 0
    assert all(g["objective"] == fixed_key_control.DEFAULT_OBJECTIVE for arm in writer.ARMS for g in result["arms"][arm]["clean_gradients"])
    assert all(result["arms"][arm]["terminal_minus_off"]["status"] == "REPORTED" for arm in writer.ARMS)
    assert result["arms"]["OFF"]["terminal_minus_off"]["measures"]["support_rms"] == 0


def test_nonfinite_native_response_is_rejected(monkeypatch):
    snapshot = Scheduler(); snapshot.step_index = 49
    z = torch.zeros(1, 16, 45, 40, 64)
    monkeypatch.setattr(payload_control.trajectory, "zero_step", lambda *a, **k: (torch.full_like(z, float("nan")), snapshot))
    with pytest.raises(FloatingPointError, match="nonfinite actual native response"):
        payload_control.controlled_step(snapshot, z, z, z, torch.ones_like(z), 1.0, writer.R_STAR, 49,
            lambda kind, done: None)


def test_full_only_calibration_and_invalid_reporting():
    config = run.load(run.MANIFEST)
    cases = {c["id"]: run.empty_case(c, config) for c in config["cases"]}
    def binding(receiver):
        return {"spec_sha256": "fixed-spec", "key_id": fixed_key.key_identifier(config["key_utf8"].encode()),
                "view": "FULL", "receiver_protocol_id": fixed_key.PROTOCOL_ID if receiver == "ORIGINAL" else fixed_key_split_receiver.PROTOCOL_ID}
    for c in config["cases"]:
        if c["role"] == "calibration_off":
            for receiver in run.RECEIVERS:
                cases[c["id"]]["videos"]["OFF"]["receivers"][receiver].update(status="SCORED", statistic=.1, **binding(receiver))
    cal = run.freeze_calibration(cases, config, "fixed-spec")
    assert all(row["status"] == "FROZEN" and row["threshold"] == pytest.approx(.100001) for row in cal.values())
    for c in config["cases"]:
        if c["role"] != "evaluation": continue
        for arm, item in cases[c["id"]]["videos"].items():
            item["receivers"]["C2_STATE_CONFIRM"].update(status="SCORED", statistic=.2 if arm == "T40_49" else .0, **binding("C2_STATE_CONFIRM"))
            run.attach_decisions(item, cal, case_role="evaluation", arm=arm)
    summary = run.summarize_target(cases, config, cal)
    assert summary["target_pass"] and summary["TPR"] == 1 and summary["FPR"] == 0
    cases["s2"]["videos"]["OFF"]["receivers"]["C2_STATE_CONFIRM"].update(status="INVALID", statistic=None)
    run.attach_decisions(cases["s2"]["videos"]["OFF"], cal, case_role="evaluation", arm="OFF")
    summary = run.summarize_target(cases, config, cal)
    assert not summary["target_pass"] and summary["FPR"] is None
    assert summary["INVALID_OFF"] == 1 and summary["FPR_identification_range"] == [0, .25]
    cases["c09"]["videos"]["OFF"]["receivers"]["C2_STATE_CONFIRM"]["status"] = "INVALID"
    assert run.freeze_calibration(cases, config, "fixed-spec")["C2_STATE_CONFIRM"]["status"] == "UNCALIBRATED"


def test_parent_runs_nine_calibrators_before_any_evaluation_and_retains_failures(monkeypatch, tmp_path):
    attempts = []
    def fail(command, log):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        attempts.append((case_id, stage))
        if case_id == "c01" and stage == "generate":
            root = Path(command[command.index("--output") + 1]); root.mkdir(parents=True)
            run.dump(root / "progress.json", {"status": "GENERATION_RUNNING",
                "progress": {"stage": "prefix40"}, "actual_calls": {"transformer_attempted": 7}})
        raise OSError("planned child launch failure")
    monkeypatch.setattr(run, "_child", fail)
    result = run.run_all(run.MANIFEST, tmp_path / "run")
    assert len(attempts) == 18
    assert all(cid.startswith("c") for cid, _ in attempts[:18])
    assert result["calibration"]["C2_STATE_CONFIRM"]["status"] == "UNCALIBRATED"
    assert result["target_summary"]["INVALID_OFF"] == 4
    assert result["target_summary"]["FPR"] is None
    assert result["fixed_denominator"]["physical_videos"] == 29
    assert len([v for c in result["cases"].values() for v in c["videos"].values()]) == 29
    assert (tmp_path / "run" / "calibration.json").exists()
    assert result["cases"]["c01"]["actual_calls"]["transformer_attempted"] == 7
    assert result["cases"]["c01"]["last_child_progress"] == {"stage": "prefix40"}
    assert sum(len(parts) for source in result["raw_layer_changes"].values() for arm in source.values() for parts in arm.values()) == 180
    assert result["fixed_denominator"]["evaluation_raw_adjacent_changes"] == 180
    assert all(result["cases"][c]["status"] == "NOT_RUN_UNCALIBRATED" for c in ("s0", "s1", "s2", "s3"))
    assert all(v["receivers"]["C2_STATE_CONFIRM"]["decision"]["status"] == "UNCALIBRATED"
               for c in ("s0", "s1", "s2", "s3") for v in result["cases"][c]["videos"].values())


def test_parent_freezes_full_c2_before_eval_and_only_target_controls_pass(monkeypatch, tmp_path):
    config = run.load(run.MANIFEST)
    case_map = {c["id"]: c for c in config["cases"]}
    order = []
    spec_sha = run.sha(run.MANIFEST)
    key_id = fixed_key.key_identifier(config["key_utf8"].encode())
    root = tmp_path / "success"
    def child(command, log):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        path = Path(command[command.index("--output") + 1]); path.mkdir(parents=True, exist_ok=True)
        case = case_map[case_id]
        order.append((case_id, stage))
        record = run.empty_case(case, config)
        record["status"] = "GENERATION_COMPLETE" if stage == "generate" else "EXECUTION_COMPLETE"
        if case_id == "c01":
            record["actual_calls"]["transformer_attempted"] = 1
            run.dump(path / "progress.json", {"status": record["status"], "progress": {"stage": stage},
                "actual_calls": {"transformer_attempted": 7 if stage == "generate" else 2}})
        if case["role"] == "evaluation":
            assert run.load(root / "calibration.json")["C2_STATE_CONFIRM"]["status"] == "FROZEN"
        if stage == "media":
            for arm, video in record["videos"].items():
                for receiver in run.RECEIVERS:
                    video["receivers"][receiver].update(status="SCORED", statistic=.2 if arm == "T40_49" else .0,
                        spec_sha256=spec_sha, key_id=key_id, view="FULL",
                        receiver_protocol_id=fixed_key.PROTOCOL_ID if receiver == "ORIGINAL" else fixed_key_split_receiver.PROTOCOL_ID)
                if arm == "T49_ONLY":
                    video["receivers"]["ORIGINAL"].update(status="INVALID", statistic=None)
                if case_id == "c09":
                    video["receivers"]["ORIGINAL"].update(status="INVALID", statistic=None)
        run.dump(path / ("generation.json" if stage == "generate" else "result.json"), record)
        return 0
    monkeypatch.setattr(run, "_child", child)
    result = run.run_all(run.MANIFEST, root)
    assert all(case_id.startswith("c") for case_id, _ in order[:18])
    assert all(case_id.startswith("s") for case_id, _ in order[18:])
    assert result["calibration"]["C2_STATE_CONFIRM"]["status"] == "FROZEN"
    assert result["calibration"]["ORIGINAL"]["status"] == "UNCALIBRATED"
    assert result["target_summary"]["target_pass"]
    assert result["target_summary"]["TPR"] == 1 and result["target_summary"]["FPR"] == 0
    assert all(result["cases"][c]["videos"]["T49_ONLY"]["receivers"]["ORIGINAL"]["decision"]["status"] == "INVALID" for c in ("s0", "s1", "s2", "s3"))
    assert result["cases"]["c01"]["actual_calls"]["transformer_attempted"] == 7


def test_corrupt_child_json_and_valid_progress_keep_slots_and_finish(monkeypatch, tmp_path):
    attempted = []
    def child(command, log):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        attempted.append((case_id, stage))
        path = Path(command[command.index("--output") + 1]); path.mkdir(parents=True, exist_ok=True)
        if case_id == "c01" and stage == "generate":
            (path / "generation.json").write_text("{")
            run.dump(path / "progress.json", {"status": "GENERATION_RUNNING", "progress": {"stage": "prefix40"},
                "actual_calls": {"transformer_attempted": 7}})
            return -9
        return 1
    monkeypatch.setattr(run, "_child", child)
    result = run.run_all(run.MANIFEST, tmp_path / "corrupt")
    assert len(attempted) == 18
    assert result["status"] == "WITH_RETAINED_FAILURES"
    assert result["cases"]["c01"]["actual_calls"]["transformer_attempted"] == 7
    assert any("child result unreadable" in f["error"] for f in result["failures"])
    assert result["target_summary"]["FPR"] is None
    assert len([v for c in result["cases"].values() for v in c["videos"].values()]) == 29


def test_media_worker_saves_raster_and_observation_hashes_without_changing_blind_input(monkeypatch, tmp_path):
    config = run.load(run.MANIFEST)
    case = next(c for c in config["cases"] if c["id"] == "s0")
    root = tmp_path / "s0"; root.mkdir()
    record = run.empty_case(case, config)
    record["config"] = run.case_config(config, case)
    for arm, item in record["videos"].items():
        item["status"] = "GENERATED"
        path = root / "terminals" / f"{arm}.pt"
        run._artifact(torch.zeros(1, 16, 45, 40, 64), path, item, "terminal")
    run.dump(root / "generation.json", record)
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
        def clear_cache(self): pass
    monkeypatch.setattr(run, "load_frozen_vae", lambda config: VAE())
    monkeypatch.setattr(run, "decode_normalized_latent", lambda vae, terminal: torch.zeros(181, 2, 2, 3))
    monkeypatch.setattr(run, "reencode_rgb24_readback", lambda vae, pixels: torch.zeros(1, 16, 45, 40, 64))
    monkeypatch.setattr(run, "score_layer", lambda layer, tensor, book: layer.update(status="SCORED"))
    def save(rgb, path, fps, crf):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fake mp4")
    monkeypatch.setattr(run, "encode_rgb", save)
    monkeypatch.setattr(run, "read_mp4", lambda path: torch.zeros(181, 2, 2, 3))
    def phases(pixels, vae, count, on_phase):
        obs, rows = {}, {}
        for g in range(4):
            count("vae_encode", False); count("vae_encode", True)
            latent = torch.zeros(1, 16, 45, 40, 64)
            rows[str(g)] = {"status": "COMPLETE"}
            on_phase(g, latent, rows[str(g)])
            obs[g] = latent.numpy()
        return obs, rows
    monkeypatch.setattr(run, "encode_four_phases", phases)
    received = []
    def blind(observations, key, spec_sha):
        received.append((tuple(observations), key, spec_sha))
        return {name: {"status": "SCORED", "statistic": 0.0, "decision": None} for name in run.RECEIVERS}, {}, {}
    monkeypatch.setattr(run, "score_blind", blind)
    result = run.media_case("s0", run.MANIFEST, root)
    assert result["status"] == "EXECUTION_COMPLETE"
    assert len(received) == 5 and all(x[0] == (0, 1, 2, 3) for x in received)
    assert result["actual_calls"]["vae_encode_completed"] == 30
    assert result["actual_calls"]["mp4_read_completed"] == 5
    for item in result["videos"].values():
        assert item["artifacts"]["decoded_float_rgb"]["sha256"]
        assert item["artifacts"]["rgb8_no_codec"]["sha256"]
        assert item["artifacts"]["received_rgb"]["sha256"]
        assert all(item["phases"][str(g)]["sha256"] for g in range(4))

    calibration_case = next(c for c in config["cases"] if c["id"] == "c01")
    calibration_root = tmp_path / "c01"; calibration_root.mkdir()
    calibration_record = run.empty_case(calibration_case, config)
    calibration_record["config"] = run.case_config(config, calibration_case)
    off = calibration_record["videos"]["OFF"]
    off["status"] = "GENERATED"
    run._artifact(torch.zeros(1, 16, 45, 40, 64), calibration_root / "terminals" / "OFF.pt", off, "terminal")
    run.dump(calibration_root / "generation.json", calibration_record)
    def phase_save_failed_but_observation_retained(pixels, vae, count, on_phase):
        obs, rows = phases(pixels, vae, count, on_phase)
        rows["2"] = {"status": "FAILED", "error": "persist failed after in-memory encode"}
        return obs, rows
    monkeypatch.setattr(run, "encode_four_phases", phase_save_failed_but_observation_retained)
    failed = run.media_case("c01", run.MANIFEST, calibration_root)
    assert len(received) == 5  # The blind scorer never sees the incomplete persisted view.
    assert failed["videos"]["OFF"]["receivers"]["C2_STATE_CONFIRM"]["status"] == "INVALID"
    assert run.freeze_calibration({"c01": failed, **{
        c["id"]: run.empty_case(c, config) for c in config["cases"] if c["role"] == "calibration_off" and c["id"] != "c01"}},
        config, "spec")["C2_STATE_CONFIRM"]["status"] == "UNCALIBRATED"


def test_builder_requires_source_sha_and_emits_single_run_all_path(tmp_path):
    with pytest.raises(ValueError):
        notebook_builder.build("not-published", tmp_path / "bad.ipynb")
    path = notebook_builder.build("1" * 40, tmp_path / "draft_structure_only.ipynb")
    notebook = json.loads(path.read_text())
    assert notebook["metadata"]["source_commit"] == "1" * 40
    assert "".join(notebook["cells"][0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "keyed_presence_t40_t49_run" in source
    assert "--config" in source and "--output" in source
    assert "checkout', '--detach', SOURCE_SHA" in source
    assert "GPU whitelist" not in source
