"""CPU/fake fixed-arm and history checks; no model, GPU, Drive or real MP4."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.wan_state_clock import rgb_dct_multistep_comparison_run as trial
from main.tube_state import rgb_dct_group_consistency as receiver
from runtime.wan import rgb_dct_multistep_backend as backend_module
from runtime.wan import rgb_dct_multistep_adapter as adapter
from runtime.wan import trajectory
from scripts import build_rgb_dct_multistep_comparison_notebook as builder


pytestmark = pytest.mark.unit


def _measure(rms: float) -> dict:
    return dict(support_rms=rms, global_rms=rms, peak_abs=rms)


class FakeBackend:
    def __init__(self, store, *, fail_check=False, zero_multi44=False,
                 fail_multi_lift=False):
        self.store = store
        self.off_terminal = torch.tensor([0.0])
        self.terminals = {}
        self.phase = "INITIAL"
        self.events = []
        self.encodes = 0
        self.fail_check = fail_check
        self.zero_multi44 = zero_multi44
        self.fail_multi_lift = fail_multi_lift

    def prepare_off(self, artifact_dir):
        self.events.append("OFF")
        self.store.count("generation", False)
        for _ in range(100):
            self.store.count("transformer", False)
            self.store.count("transformer", True)
        for _ in range(50):
            self.store.count("scheduler_step", False)
            self.store.count("scheduler_step", True)
        self.store.count("generation", True)
        self.phase = "TRANSFORMER"
        return dict(initial_noise_fingerprint="fake-fixed-noise", history_indices=[44, 46, 49])

    def to_vae_phase(self, name):
        assert self.phase == "TRANSFORMER"
        self.phase = "VAE"
        self.events.append(name)
        return dict(name=name, transformer_resident="cpu", vae_resident="cuda")

    def to_transformer_phase(self, name):
        assert self.phase == "VAE"
        self.phase = "TRANSFORMER"
        self.events.append(name)
        return dict(name=name, identity_verified=True)

    def verify_cfg(self, index, *, multi=False):
        assert self.phase == "TRANSFORMER"
        self.events.append(f"VERIFY_{index}_{multi}")
        for _ in range(2):
            self.store.count("transformer_validation", False)
            self.store.count("transformer_validation", True)
        if self.fail_check and index == 44:
            raise RuntimeError("fake same-input CFG mismatch")
        return dict(index=index, multi=multi, numerical_match=True)

    def decode(self, latent):
        assert self.phase == "VAE"
        self.store.count("vae_decode", False)
        self.store.count("vae_decode", True)
        return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)

    def encode(self, rgb):
        assert self.phase == "VAE"
        self.store.count("vae_encode", False)
        self.store.count("vae_encode", True)
        self.encodes += 1
        return np.array([float(self.encodes % 2)], dtype=np.float32)

    def clean_proxy(self, index):
        assert index in (44, 46)
        self.events.append(f"OFF_PROXY_{index}")
        return torch.tensor([float(index)])

    def multi_clean_proxy(self):
        assert "MULTI_T46_LIVE" in self.events
        self.events.append("MULTI_PROXY_46")
        if self.fail_multi_lift:
            raise RuntimeError("fake MULTI T46 VAE lift failure")
        return torch.tensor([46.0])

    def _control(self, index, target):
        self.store.count("shadow_step", False)
        self.store.count("shadow_step", True)
        self.store.count("unit_response_probe_step", False)
        self.store.count("unit_response_probe_step", True)
        self.store.count("scheduler_step", False)
        self.store.count("scheduler_step", True)
        return dict(index=index, target_D_support_rms=target, actual_D=_measure(target),
                    raw_direction_fingerprint=f"unique_{index}_{len(self.events)}")

    def control_from_off(self, index, masked, target):
        assert self.phase == "TRANSFORMER" and index in (46, 49)
        self.events.append(f"CONTROL_OFF_{index}")
        row = self._control(index, target)
        return "READY", torch.tensor([float(index)]), index, row

    def continue_normal(self, z, scheduler, start, stop):
        for index in range(start, stop):
            for _ in range(2):
                self.store.count("transformer", False)
                self.store.count("transformer", True)
            self.store.count("scheduler_step", False)
            self.store.count("scheduler_step", True)
            self.events.append(f"FREE_{index}")
        return torch.tensor([float(stop)])

    def multi44_to_46(self, masked, target):
        assert self.phase == "TRANSFORMER" and target == trial.R_STAR / 2
        self.events.append("MULTI_CONTROL_44")
        if self.zero_multi44:
            return "ZERO_LIFT_DIRECTION", dict(index=44, target_D_support_rms=target)
        row = self._control(44, target)
        self.store.count("scheduler_step", False)
        self.store.count("scheduler_step", True)
        for _ in range(4):
            self.store.count("transformer", False)
            self.store.count("transformer", True)
        self.events.append("MULTI_T46_LIVE")
        return "READY", row

    def control_multi46(self, masked, target):
        assert self.phase == "TRANSFORMER" and "MULTI_PROXY_46" in self.events
        self.events.append("MULTI_CONTROL_46")
        row = self._control(46, target)
        return "READY", torch.tensor([47.0]), 46, row

    def net_from_off(self, terminal):
        return _measure(float(terminal[0]))

    def resources(self):
        return dict(fake_cpu=True, phase=self.phase, events=self.events)

    def release(self):
        self.phase = "RELEASED"


def _run_fake(tmp_path, monkeypatch, *, invalid=None, fail_check=False,
              zero_multi44=False, fail_multi_lift=False, off_c=10):
    config = trial.load_config()
    output = tmp_path / "fixed"
    output.mkdir()
    monkeypatch.setattr(trial, "apply_carrier",
                        lambda rgb, key, polarity: (rgb, {"fake": True}))
    monkeypatch.setattr(trial, "lift_direction",
                        lambda raw: (np.array([1.0], dtype=np.float32), {"status": "READY"}))
    monkeypatch.setattr(trial, "_paired_mp4_quality",
                        lambda off, marked: dict(compared_frames=181, fake=True))
    workers = []
    backends = []
    reads = []

    def encode_fn(rgb, path, fps, crf):
        assert fps == 8 and crf == 18
        path.write_bytes((str(path) + " fake MP4").encode())

    def score_fn(path, key):
        case_id, arm = path.parents[2].name, path.parent.name
        reads.append((case_id, arm))
        assert key == config["key_utf8"].encode() and path.is_file()
        if (case_id, arm) == invalid:
            return dict(status="INVALID", reason="FAKE_READ_INVALID", frames_used=0), None
        c = off_c if arm == "OFF" else 27
        q = [1.0] * c + [-1.0] * (30 - c)
        return (dict(status="SCORED", score=float(c) / 100, group_scores=q,
                     positive_groups=c, frames_used=181, spec_sha256=receiver.SPEC_SHA256,
                     key_id=config["receiver_key_id"]),
                np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE))

    def worker_fn(output, config, case_id):
        workers.append(case_id)
        store = trial.Store.open(output / "result.json")
        backend = FakeBackend(store, fail_check=fail_check and case_id == trial.CASE_IDS[0],
                              zero_multi44=zero_multi44 and case_id == trial.CASE_IDS[0],
                              fail_multi_lift=fail_multi_lift and case_id == trial.CASE_IDS[0])
        backends.append(backend)
        trial.run_case(store, case_id, config, backend,
                       encode_fn=encode_fn, score_fn=score_fn)
        return None

    trial._supervise(output, config, "a" * 40, worker_fn=worker_fn)
    return trial.Store.open(output / "result.json"), workers, backends, reads


def test_protocol_roster_budget_and_eight_preserved_slots(tmp_path):
    config = trial.load_config()
    assert trial._sha(trial.PROTOCOL_PATH) == "24cda99b931eeb726b98c929e2dfeadfd44b0a17e4a171655cfe5f995e4e651f"
    assert config["receiver_spec_sha256"] == receiver.SPEC_SHA256
    assert config["call_plan_max"]["transformer_total"] == 244
    assert config["call_plan_max"]["native_scheduler_total"] == 138
    store = trial.Store(tmp_path / "result.json",
                        trial.initial_result(config, tmp_path, "a" * 40))
    store.save()
    assert list(store.data["cases"]) == list(trial.CASE_IDS)
    assert all(list(case["slots"]) == list(trial.ARMS) for case in store.data["cases"].values())
    assert store.data["pending_media_slots"] == 8 and store.data["scored_frames"] == 0


def test_adapter_scores_from_one_read_and_keeps_invalid_separate(monkeypatch, tmp_path):
    key = b"WanProjection-first-validation-key-v1"
    reads = []
    monkeypatch.setattr(adapter, "read_mp4",
                        lambda path: (reads.append(path) or np.broadcast_to(
                            np.float32(0.5), trial.RGB_SHAPE)))
    scored, rgb = adapter.score_mp4_once(tmp_path / "FULL.mp4", key)
    assert len(reads) == 1 and rgb is not None
    assert scored["status"] == "SCORED" and scored["positive_groups"] == 0
    assert scored["decision"] is None
    monkeypatch.setattr(adapter, "read_mp4", lambda path: np.empty((180, 320, 512, 3)))
    invalid, rgb = adapter.score_mp4_once(tmp_path / "INVALID.mp4", key)
    assert invalid["status"] == "INVALID" and rgb is None and invalid["decision"] is None


def test_paired_mp4_quality_uses_full_decoded_frames():
    off = np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)
    marked = np.broadcast_to(np.float32(0.6), trial.RGB_SHAPE)
    quality = trial._paired_mp4_quality(off, marked)
    assert quality["compared_frames"] == 181
    assert quality["rgb_mae"] == pytest.approx(0.1, abs=1e-7)
    assert quality["rgb_rmse"] == pytest.approx(0.1, abs=1e-7)
    assert quality["temporal_difference_rmse"] == pytest.approx(0, abs=1e-12)


def test_complete_fake_run_checks_phases_histories_and_exact_caps(tmp_path, monkeypatch):
    store, workers, backends, reads = _run_fake(tmp_path, monkeypatch)
    assert workers == list(trial.CASE_IDS) and len(reads) == 8
    assert store.data["status"] == "FIXED_MULTISTEP_COMPARISON_COMPLETE"
    assert store.data["scored_media_slots"] == 8 and store.data["scored_frames"] == 1448
    assert store.data["native_scheduler_calls"]["attempted"] == 138
    assert store.data["transformer_calls"]["attempted"] == 244
    for kind, cap in trial.PLAN.items():
        assert store.data["calls"][kind] == {"attempted": cap, "completed": cap}
    for case_id, backend in zip(trial.CASE_IDS, backends, strict=True):
        assert backend.events.index("OFF_PROXY_46") < backend.events.index("CONTROL_OFF_46")
        assert backend.events.index("MULTI_T46_LIVE") < backend.events.index("MULTI_PROXY_46")
        assert backend.events.index("MULTI_PROXY_46") < backend.events.index("MULTI_CONTROL_46")
        assert sum(event.startswith("VERIFY_") for event in backend.events) == 3
        case = store.case(case_id)
        assert [row["index"] for row in case["controls"]["MULTI44_46"]] == [44, 46]
        assert case["arm_metrics"]["MULTI44_46"]["sum_support_rms"] == pytest.approx(trial.R_STAR)
        assert case["arm_metrics"]["MULTI44_46"]["sum_support_rms_squared"] == pytest.approx(trial.R_STAR ** 2 / 2)
        assert case["slots"]["MULTI44_46"]["quality_vs_off"]["compared_frames"] == 181
        assert all(case["slots"][arm]["positive_groups"] == 27 for arm in trial.ARMS[1:])


def test_valid_off_false_positive_does_not_stop_other_arms(tmp_path, monkeypatch):
    store, workers, _, reads = _run_fake(tmp_path, monkeypatch, off_c=24)
    assert workers == list(trial.CASE_IDS) and len(reads) == 8
    assert store.data["status"] == "FIXED_MULTISTEP_COMPARISON_NEGATIVE"
    errors = store.data["reporting"]["observed_misclassifications"]
    assert [(row["case_id"], row["arm"]) for row in errors] == [
        (trial.CASE_IDS[0], "OFF"), (trial.CASE_IDS[1], "OFF")]
    assert all(store.case(case)["slots"]["OFF"]["c_margin"] == 0
               for case in trial.CASE_IDS)


def test_invalid_off_and_multi44_method_negative_keep_later_source(tmp_path, monkeypatch):
    store, workers, backends, reads = _run_fake(
        tmp_path, monkeypatch, invalid=(trial.CASE_IDS[0], "OFF"), zero_multi44=True)
    assert workers == list(trial.CASE_IDS)
    first = store.case(trial.CASE_IDS[0])
    assert first["slots"]["OFF"]["status"] == "INVALID"
    assert first["slots"]["OFF"]["decision"] is None
    assert first["slots"]["MULTI44_46"]["status"] == "ZERO_LIFT_DIRECTION"
    assert first["slots"]["SINGLE49"]["status"] == "SCORED"
    assert first["slots"]["SINGLE46"]["status"] == "SCORED"
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert store.data["fixed_denominator"]["mp4_score_slots"] == 8
    assert store.data["status"] == "INCOMPLETE"
    assert (trial.CASE_IDS[0], "MULTI44_46") not in reads


def test_restoration_mismatch_is_engineering_invalid_not_h0(tmp_path, monkeypatch):
    store, workers, _, reads = _run_fake(tmp_path, monkeypatch, fail_check=True)
    assert workers == list(trial.CASE_IDS)
    first = store.case(trial.CASE_IDS[0])
    assert first["slots"]["OFF"]["status"] == "SCORED"
    assert all(first["slots"][arm]["status"] == "NOT_RUN_ENGINEERING_INVALID"
               and first["slots"][arm]["decision"] is None for arm in trial.ARMS[1:])
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert store.data["status"] == "INCOMPLETE" and len(reads) == 5


def test_multi_t46_vae_failure_preserves_single_media_and_next_source(tmp_path, monkeypatch):
    store, workers, backends, reads = _run_fake(tmp_path, monkeypatch,
                                                fail_multi_lift=True)
    assert workers == list(trial.CASE_IDS)
    first = store.case(trial.CASE_IDS[0])
    assert first["slots"]["SINGLE49"]["status"] == "SCORED"
    assert first["slots"]["SINGLE46"]["status"] == "SCORED"
    assert first["slots"]["MULTI44_46"]["status"] == "ENGINEERING_INVALID"
    assert first["slots"]["MULTI44_46"]["decision"] is None
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert len(reads) == 7 and store.data["scored_media_slots"] == 7
    assert store.data["status"] == "INCOMPLETE"


def test_history_move_and_shadow_copy_do_not_pollute_source():
    class FakeScheduler:
        def __init__(self):
            self.timesteps = torch.arange(50, dtype=torch.float32)
            self.sigmas = torch.linspace(1, 0, 51)
            self.step_index = 44
            self.model_outputs = [torch.tensor([1.0]), torch.tensor([2.0])]
            self.nested = {"history": [torch.tensor([3.0])]}
        def step(self, v, timestep, z, return_dict=False):
            self.step_index += 1
            self.model_outputs.append(v.clone())
            return (z - v * 0.1,)
    scheduler = FakeScheduler()
    before = trajectory.fingerprint(vars(scheduler))
    backend_module._move_scheduler(scheduler, torch.device("cpu"))
    assert trajectory.fingerprint(vars(scheduler)) == before
    counts = []
    result, changed = trajectory.zero_step(scheduler, torch.tensor([2.0]),
                                          torch.tensor([1.0]), 44,
                                          lambda kind, completed: counts.append((kind, completed)),
                                          "shadow_step")
    assert result.item() == pytest.approx(1.9)
    assert trajectory.fingerprint(vars(scheduler)) == before
    assert changed.step_index == 45 and counts == [("shadow_step", False), ("shadow_step", True)]


def test_notebook_builder_static_binding(tmp_path):
    path = builder.build("a" * 40, tmp_path / "fixed.ipynb")
    notebook = json.loads(path.read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]
    assert code[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert notebook["metadata"]["source_commit"] == "a" * 40
    assert "dev/rgb-dct-multistep-comparison-v1" in "\n".join(code)
    assert "rgb_dct_multistep_comparison_run" in "\n".join(code)
    assert "SETUP_SLOTS_PATH" in code[1]
    for source in code:
        compile(source, "notebook-cell", "exec")
