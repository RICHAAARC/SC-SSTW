"""CPU/fake checks for the integrated four-bit writer/receiver candidate."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.tube_state import payload_codec, projection_margin, state_clock
from runtime.wan import payload_control, trajectory
from experiments.wan_state_clock import integrated_payload_run as run

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


def identity_path():
    yield {"g": 0, "scale": [1, 1], "offset": 0, "boundary": 11, "delta": 0}


def marked_latent(book, payload):
    z = np.zeros(projection_margin.SHAPE, dtype=np.float32)
    values = book["directions"] * book["codes"][payload, :, None]
    projection_margin.put_blocks(z, values.reshape(11, 160, 1024).astype(np.float32))
    return z


def test_four_bit_rm_roundtrip_distance_and_erasure_bound():
    assert payload_codec.RM_MIN_DISTANCE == 4
    assert len({tuple(payload_codec.rm_encode(value)) for value in range(16)}) == 16
    for value in range(16):
        word = payload_codec.rm_encode(value).tolist()
        assert payload_codec.rm_decode(word)["payload"] == value
        damaged = word.copy()
        damaged[2] ^= 1
        assert payload_codec.rm_decode(damaged)["payload"] == value
        crop_visible = word[:6] + [None, None]
        decoded = payload_codec.rm_decode(crop_visible)
        assert decoded["payload"] == value
        assert decoded["erasures"] == 2
        assert decoded["status"] == "BOUNDED_RECOVERY"
    outside = payload_codec.rm_decode([None, None, None, None, 0, 0, 0, 0])
    assert outside["payload"] is None


def test_mean_rival_two_code_loss_is_exact_legacy_formula():
    projections = torch.linspace(-2, 2, 31, dtype=torch.float64)
    codes = torch.where(torch.arange(62).reshape(2, 31) % 3 == 0, 1.0, -1.0)
    for message in (0, 1):
        legacy = -(torch.tanh(projections) * (codes[message] - codes[1 - message])).mean()
        actual = payload_codec.competitive_tanh_loss(projections, codes, message)
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)


def test_explicit_pilot_loss_has_nonzero_gradient_and_fixed_scale():
    book = payload_codec.codebook(b"pilot-test")
    z = torch.zeros(projection_margin.SHAPE, dtype=torch.float64, requires_grad=True)
    total, parts = payload_codec.payload_sync_loss(z, book, 10)
    gradient, = torch.autograd.grad(total, z)
    blocks = payload_codec.torch_blocks(gradient)
    pilot = torch.as_tensor(book["pilot_mask"])
    data = torch.as_tensor(book["data_mask"])
    assert payload_codec.PILOT_LOSS_WEIGHT == 0.25
    assert float(blocks[pilot].square().sum()) > 0
    assert float(blocks[data].square().sum()) > 0
    assert torch.count_nonzero(gradient[:, :, (0, 45)]) == 0
    assert float(parts["pilot_loss"]) == pytest.approx(0.0)


def test_blind_receiver_recovers_nonbinary_payload_and_real_crop_erasures(monkeypatch):
    book = payload_codec.codebook(b"receiver-test")
    full = marked_latent(book, 10)
    monkeypatch.setattr(state_clock, "clock_paths", identity_path)
    detected = payload_codec.read({0: full}, book)
    assert tuple(inspect.signature(payload_codec.read).parameters) == ("observations", "book")
    assert detected["best"]["payload"] == 10
    assert detected["decoder"]["payload"] == 10
    assert detected["decoder"]["status"] == "EXACT"
    cropped = full[:, :, :33].copy()
    crop_detection = payload_codec.read({0: cropped}, book)
    assert crop_detection["decoder"]["payload"] == 10
    assert crop_detection["decoder"]["erasures"] == 2
    assert crop_detection["decoder"]["status"] == "BOUNDED_RECOVERY"
    assert None in crop_detection["hard_data_bits"]


def fake_detection(payload=5, score=0.8, missing_bits=()):
    word = payload_codec.rm_encode(payload).tolist()
    evidence = [(-1.0 if bit else 1.0) for bit in word]
    for index in missing_bits:
        evidence[index] = None
        word[index] = None
    rows = {str(value): {"payload": value, "score": score - abs(value - payload) * 0.01,
                         "matched_data_supports": 960, "matched_pilot_supports": 480} for value in range(16)}
    return {"status": "SCORED", "best": rows[str(payload)], "best_by_payload": rows,
            "top_payload_unique": True, "existence_statistic": score,
            "hard_data_bits": word, "hard_window_evidence": evidence,
            "decoder": payload_codec.rm_decode(word)}


def test_crop_aggregate_runs_ecc_and_fixed_calibration_never_shrinks_denominator():
    detections = {view: fake_detection(5, 0.6 + index * 0.01) for index, view in enumerate(payload_codec.FIXED_VIEWS)}
    detections["CROP0_129"] = fake_detection(5, 0.7, missing_bits=(6, 7))
    detections["CROP4_129"] = fake_detection(5, 0.71, missing_bits=(0, 7))
    detections["CROP8_129"] = fake_detection(5, 0.72, missing_bits=(0, 1))
    aggregate = payload_codec.aggregate_crop_views(detections)
    assert aggregate["decoder"]["payload"] == 5
    assert aggregate["status"] == "SCORED"
    source = payload_codec.source_max_statistic(detections, aggregate)
    assert source["status"] == "SCORED"
    broken = dict(detections)
    broken["DELETE90"] = {"status": "INVALID"}
    invalid = payload_codec.source_max_statistic(broken, aggregate)
    assert invalid["status"] == "INVALID"
    calibration = payload_codec.freeze_calibration([source, invalid])
    assert calibration["status"] == "UNCALIBRATED"


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(prediction_type="flow_prediction", thresholding=False, lower_order_final=True)
        self.predict_x0 = True
        self.timesteps = torch.arange(50, 0, -1)
        self.sigmas = torch.linspace(1, 0, 51)
        self.step_index = None

    def step(self, velocity, timestep, sample, return_dict=False):
        if self.step_index is None:
            self.step_index = 0
        self.step_index += 1
        return (sample - 0.01 * velocity,)


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states, **kwargs):
        return (hidden_states * 0.1 + self.weight * 0.01,)


def test_real_unipc_none_cursor_initializes_then_prefix_is_strict():
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=1.0,
    )
    scheduler.set_timesteps(50)
    scheduler.set_begin_index(0)
    assert scheduler.step_index is None and scheduler.begin_index == 0
    pipe = SimpleNamespace(scheduler=scheduler, transformer=FakeTransformer())
    counts = []
    z, snapshot = trajectory.prefix44(
        pipe, torch.zeros(1, 1, 2, 2, 2), torch.tensor(1.0), torch.tensor(-1.0), torch.float32, 5.0,
        lambda kind, completed: counts.append((kind, completed)),
    )
    assert snapshot.step_index == 44
    assert z.shape == (1, 1, 2, 2, 2)
    assert counts.count(("transformer", True)) == 88
    with pytest.raises(ValueError, match="cursor"):
        trajectory.native_step(snapshot, z, torch.zeros_like(z), 43, lambda *args: None)


def test_same_history_unit_probe_budget_and_source_history_are_exact():
    scheduler = FakeScheduler()
    scheduler.step_index = 46
    z = torch.linspace(-1, 1, 46).reshape(1, 1, 46, 1, 1)
    velocity = torch.full_like(z, 0.2)
    sigma = float(scheduler.sigmas[46])
    zero_next, _ = trajectory.zero_step(scheduler, z, velocity, 46, lambda *args: None)
    raw = torch.cos(z * 3.0)
    source_history = trajectory.fingerprint(vars(scheduler))
    target = 0.042943312697648145
    unit, epsilon, probe = payload_control.prepare_direction(
        scheduler, z, velocity, zero_next, raw, target, 46, lambda *args: None,
    )
    legacy_support_rms = float(raw[:, :, 1:45].double().square().mean().sqrt())
    assert trajectory.measures(raw)["support_rms"] == legacy_support_rms
    manual_probe, _ = trajectory.zero_step(
        scheduler, z, velocity - (raw / legacy_support_rms) / sigma, 46, lambda *args: None,
    )
    manual_epsilon = target / trajectory.measures(manual_probe - zero_next)["support_rms"]
    assert epsilon == pytest.approx(manual_epsilon, rel=0, abs=0)
    after, _, row, _ = payload_control.controlled_step(
        scheduler, z, velocity, zero_next, unit, epsilon, target, 46, lambda *args: None,
    )
    assert row["actual_D"]["support_rms"] == pytest.approx(target, rel=2e-5)
    assert trajectory.fingerprint(vars(scheduler)) == source_history
    assert not torch.equal(after, zero_next)


def test_generate_calibration_case_is_fresh_and_uses_no_saved_cache(monkeypatch, tmp_path, capsys):
    pipe = SimpleNamespace(scheduler=FakeScheduler(), transformer=FakeTransformer())
    monkeypatch.setattr(run, "prepare_generation", lambda config, load_vae=False: (
        pipe, torch.zeros(1, 1, 2, 2, 2), torch.tensor(1.0), torch.tensor(-1.0), torch.float32,
    ))
    result = run.generate_case("cal_off_p0_s1", run.MANIFEST, tmp_path / "fresh")
    assert result["status"] == "GENERATION_COMPLETE", result["failures"]
    assert result["fresh_generation"]["source_cache_inputs"] == []
    assert result["fresh_generation"]["initial_noise"] == "NEW"
    assert result["actual_calls"]["transformer_completed"] == 100
    assert result["actual_calls"]["scheduler_step_completed"] == 50
    assert (tmp_path / "fresh" / "OFF_terminal.pt").exists()
    progress = run.load(tmp_path / "fresh" / "progress.json")
    assert progress["actual_calls"]["transformer_completed"] == 100
    assert progress["actual_calls"]["scheduler_step_completed"] == 50
    console = capsys.readouterr().out
    assert "stage=generate arm=OFF" in console
    assert "model_step_pairs_completed=50" in console


def complete_view(detection=None):
    return {
        "status": "SCORED",
        "observations": {str(phase): {"status": "COMPLETE"} for phase in run.PHASES},
        "detection": detection or fake_detection(5, 0.4),
    }


def fake_child_runner(tmp_path, *, partial_calibration=False):
    config = run.load(run.MANIFEST)
    by_id = {case["id"]: case for case in config["cases"]}

    def execute(command, log_path):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        case_root = Path(command[command.index("--output") + 1])
        case_root.mkdir(parents=True, exist_ok=True)
        case = by_id[case_id]
        record = run.empty_case(case)
        record.update(actual_calls={}, failures=[], config=run.case_config(config, case))
        if stage == "generate":
            record["status"] = "GENERATION_COMPLETE"
            run.dump(case_root / "generation.json", record)
        else:
            record["status"] = "EXECUTION_COMPLETE"
            for item in record["videos"].values():
                item["status"] = "MEDIA_COMPLETE"
                item["views"] = {view: complete_view() for view in run.VIEWS}
                if partial_calibration and case["role"] == "calibration_off":
                    item["views"]["DELETE90"]["status"] = "PARTIAL_OR_FAILED"
                    item["views"]["DELETE90"]["observations"]["0"]["status"] = "FAILED"
            run.dump(case_root / "result.json", record)
        log_path.write_text(f"progress fake case={case_id} stage={stage}\n", encoding="utf-8")
        print(f"progress fake case={case_id} stage={stage}", flush=True)
        return 0

    return execute


def test_run_all_predeclares_every_failure_row(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_run_child", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    result = run.run_all(run.MANIFEST, tmp_path / "failed")
    assert result["fixed_denominator"] == {
        "fresh_cases": 4, "generated_arms": 8, "saved_attack_views": 56,
        "receiver_encodes": 224, "calibration_sources": 2, "evaluation_sources": 2,
    }
    assert result["calibration"]["status"] == "UNCALIBRATED"
    assert sum(len(item["views"]) for case in result["cases"].values() for item in case["videos"].values()) == 56
    assert sum(len(view["observations"]) for case in result["cases"].values() for item in case["videos"].values() for view in item["views"].values()) == 224
    assert len(result["failures"]) == 8
    assert all("stage_logs" in case for case in result["cases"].values())


def test_protocol_eligibility_rejects_partial_raw_ranking_and_prevents_freeze():
    view = complete_view(fake_detection(5, 0.9))
    view["status"] = "PARTIAL_OR_FAILED"
    view["observations"]["0"]["status"] = "FAILED"
    adapted = run.eligible_detection(view)
    assert adapted["status"] == "INVALID"
    assert adapted["raw_detection_status"] == "SCORED"
    item = {"views": {name: complete_view() for name in run.VIEWS}}
    item["views"]["FULL"] = view
    aggregate, source = run._arm_source_record(item)
    assert aggregate["status"] == "SCORED"  # crops remain internally complete
    assert source["status"] == "INVALID"
    calibration = payload_codec.freeze_calibration([source, source])
    assert calibration["status"] == "UNCALIBRATED"
    run._attach_decisions(item, {"status": "FROZEN", "threshold": 0.1}, truth=5, calibration_sample=False)
    assert item["views"]["FULL"]["decision"] == {"status": "INVALID", "payload": None}
    assert item["decision"]["status"] == "INVALID"


def test_two_stage_orchestration_preserves_exits_calibration_decisions_and_completes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run, "_run_child", fake_child_runner(tmp_path))
    result = run.run_all(run.MANIFEST, tmp_path / "complete")
    assert result["status"] == "EXECUTION_COMPLETE"
    assert result["calibration"]["status"] == "FROZEN"
    assert result["failures"] == []
    for case in result["cases"].values():
        assert case["generate_exit_code"] == 0 and case["media_exit_code"] == 0
        assert set(case["stage_logs"]) == {"generate", "media"}
    for case_id in ("cal_off_p0_s1", "cal_off_p1_s1"):
        item = result["cases"][case_id]["videos"]["OFF"]
        assert item["decision_role"] == "THRESHOLD_CONSTRUCTION_SAMPLE_NOT_HELDOUT_FPR"
        assert item["decision"]["status"] == "REJECTED"
        assert all(view["decision"]["status"] == "REJECTED" for view in item["views"].values())
    console = capsys.readouterr().out
    assert "case=cal_off_p0_s1 stage=generate" in console
    assert "progress fake" in console


def test_uncalibrated_still_runs_eval_but_never_returns_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_run_child", fake_child_runner(tmp_path, partial_calibration=True))
    result = run.run_all(run.MANIFEST, tmp_path / "uncalibrated")
    assert result["calibration"]["status"] == "UNCALIBRATED"
    assert result["status"] == "WITH_RETAINED_FAILURES"
    for case_id in ("eval_p2_s2_m5", "eval_p3_s3_ma"):
        case = result["cases"][case_id]
        assert case["generate_exit_code"] == 0 and case["media_exit_code"] == 0
        for item in case["videos"].values():
            assert item["decision"]["status"] == "UNCALIBRATED"
            assert item["decision"]["payload"] is None
            assert all(view["decision"]["status"] == "UNCALIBRATED" and view["decision"]["payload"] is None for view in item["views"].values())


def test_one_spawn_exception_is_retained_without_aborting_later_children(monkeypatch, tmp_path):
    successful = fake_child_runner(tmp_path)
    calls = 0

    def one_failure(command, log_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("one launch failure")
        return successful(command, log_path)

    monkeypatch.setattr(run, "_run_child", one_failure)
    result = run.run_all(run.MANIFEST, tmp_path / "spawn")
    assert calls == 8
    assert result["failures"][0]["kind"] == "CHILD_SPAWN_OR_TEE_FAILURE"
    first = result["cases"]["cal_off_p0_s1"]
    assert first["generate_exit_code"] is None and first["media_exit_code"] == 0
    assert len(first["videos"]["OFF"]["views"]) == 7


def test_media_attack_chain_has_seven_saved_views_and_four_phases(monkeypatch, tmp_path):
    case = run.load(run.MANIFEST)["cases"][0]
    root = tmp_path / "media"
    root.mkdir()
    generation = run.empty_case(case)
    generation.update(status="GENERATION_COMPLETE", config=run.case_config(run.load(run.MANIFEST), case), failures=[], actual_calls={})
    run.dump(root / "generation.json", generation)
    torch.save(torch.zeros(1), root / "OFF_terminal.pt")
    book = payload_codec.codebook(b"media-test")
    np.savez(root / "payload_codebook.npz", **book)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    cache = {}
    monkeypatch.setattr(run, "load_frozen_vae", lambda config: VAE())
    rgb = torch.linspace(0, 1, 181).reshape(181, 1, 1, 1).expand(181, 4, 4, 3).clone()
    monkeypatch.setattr(run, "decode_normalized_latent", lambda *args: rgb.clone())

    def save_video(value, path, *args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        cache[str(path)] = value.clone()

    monkeypatch.setattr(run, "encode_rgb", save_video)
    monkeypatch.setattr(run, "read_mp4", lambda path: cache[str(path)].clone())
    monkeypatch.setattr(run, "reencode_rgb24_readback", lambda vae, value: torch.zeros(1, 1, 2, 2, 2))
    monkeypatch.setattr(run.payload_codec, "read", lambda observations, codebook: fake_detection(5, 0.4))
    result = run.media_case(case["id"], run.MANIFEST, root)
    assert result["status"] == "EXECUTION_COMPLETE", result["failures"]
    assert result["actual_calls"]["mp4_save_completed"] == 7
    assert result["actual_calls"]["mp4_read_completed"] == 8
    assert result["actual_calls"]["vae_encode_completed"] == 28
    assert all(len(view["observations"]) == 4 for view in result["videos"]["OFF"]["views"].values())
