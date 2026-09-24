"""CPU/fake checks: no Wan weights, GPU, Drive or real MP4 generation."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state import rgb_dct_presence as baseline
from runtime.wan import rgb_dct_group_consistency_adapter as adapter
from experiments.wan_state_clock import rgb_dct_group_consistency_run as trial


pytestmark = pytest.mark.unit


def test_fixed_protocol_receiver_and_group_reduction():
    config = trial.load_config()
    assert trial._sha(trial.PROTOCOL_PATH) == "b11422eb59f8d2b80cffae239d92a64e14b6efd00bf372a694bf9f55a8aeb8d1"
    assert config["receiver_spec_sha256"] == receiver.SPEC_SHA256 != baseline.SPEC_SHA256
    assert tuple(inspect.signature(adapter.score_mp4).parameters) == ("path", "key")
    assert tuple(inspect.signature(receiver.score_rgb).parameters) == ("rgb", "key")
    assert [case["id"] for case in config["cases"]] == list(trial.CASE_IDS)
    assert [case["role"] for case in config["cases"]] == ["REF_OFF"] * 2 + ["EVAL"] * 2
    spatial, _, temporal = baseline.key_codes(config["key_utf8"].encode())
    features = temporal[:, None] * spatial[None, :] * np.ones((181, 160))
    positive = receiver._score_features(features, spatial, temporal)
    negative = receiver._score_features(-features, spatial, temporal)
    zero = receiver._score_features(np.zeros_like(features), spatial, temporal)
    assert (positive["positive_groups"], positive["decision"]) == (30, "H1")
    assert (negative["positive_groups"], negative["decision"]) == (0, "H0")
    assert (zero["positive_groups"], zero["decision"]) == (0, "H0")
    assert abs(positive["score"] - positive["score_reconstructed"]) <= 1e-10
    assert receiver.decide(23) == "H0" and receiver.decide(24) == "H1"
    with pytest.raises(ValueError):
        receiver.decide(24.0)


def test_single_read_adapter_and_invalid_is_not_h0(monkeypatch, tmp_path):
    calls = []
    def read_once(path):
        calls.append(path)
        return np.broadcast_to(np.float32(0.5), (181, 320, 512, 3))
    monkeypatch.setattr(adapter, "read_mp4", read_once)
    key = b"WanProjection-first-validation-key-v1"
    scored = adapter.score_mp4(tmp_path / "FULL.mp4", key)
    assert len(calls) == 1
    assert scored["status"] == "SCORED" and scored["positive_groups"] == 0
    assert scored["decision"] == "H0"
    monkeypatch.setattr(adapter, "read_mp4", lambda path: np.empty((180, 320, 512, 3)))
    invalid = adapter.score_mp4(tmp_path / "bad.mp4", key)
    assert invalid["status"] == "INVALID" and invalid["decision"] is None


def _fake_run(tmp_path, monkeypatch, counts, invalid=None):
    config = trial.load_config()
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(trial, "apply_carrier", lambda x, key, polarity: (x, {"fake": True}))
    monkeypatch.setattr(trial, "lift_direction",
                        lambda raw: (np.ones((1,), dtype=np.float32), {"status": "READY"}))
    workers = []
    scored = []

    def encode_fn(rgb, path, fps, crf):
        assert fps == 8 and crf == 18
        path.write_bytes(b"fake MP4")

    def score_fn(path, key):
        case_id, arm = path.parents[2].name, path.parent.name
        scored.append((case_id, arm))
        assert key == config["key_utf8"].encode() and path.is_file()
        if (case_id, arm) == invalid:
            return dict(status="INVALID", reason="FAKE_DECODE_FAILURE", frames_used=0)
        c = counts[(case_id, arm)]
        q = [1.0] * c + [-1.0] * (30 - c)
        return dict(status="SCORED", score=0.01 * c, group_scores=q,
                    positive_groups=c, decision=receiver.decide(c), frames_used=181,
                    spec_sha256=receiver.SPEC_SHA256, key_id=config["receiver_key_id"])

    class FakeBackend:
        def __init__(self, store):
            self.store = store
            self.encodes = 0
        def generate_prefix(self, artifact_dir):
            self.store.count("generation", False)
            for _ in range(100):
                self.store.count("transformer", False)
                self.store.count("transformer", True)
            for _ in range(50):
                self.store.count("scheduler_step", False)
                self.store.count("scheduler_step", True)
            self.store.count("generation", True)
            return SimpleNamespace(z0="OFF", metadata={"fake": True})
        def load_vae(self):
            return {"fake": True}
        def decode(self, terminal):
            self.store.count("vae_decode", False)
            self.store.count("vae_decode", True)
            return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)
        def encode(self, rgb):
            self.store.count("vae_encode", False)
            self.store.count("vae_encode", True)
            self.encodes += 1
            return np.array([float(self.encodes == 1)], dtype=np.float32)
        def controlled_terminal(self, prefix, masked, target):
            assert target == 0.042943312697648145
            self.store.count("unit_response_probe_step", False)
            self.store.count("unit_response_probe_step", True)
            self.store.count("scheduler_step", False)
            self.store.count("scheduler_step", True)
            return "READY", "NATIVE", {"fake": True}
        def resources(self):
            return {"fake_cpu": True}
        def release(self):
            pass

    def worker_fn(output, config, case_id):
        workers.append(case_id)
        store = trial.Store.open(output / "result.json")
        trial.run_case(store, case_id, config, FakeBackend(store),
                       encode_fn=encode_fn, score_fn=score_fn)
        return None
    trial._supervise(output, config, "a" * 40, worker_fn=worker_fn)
    return trial.Store.open(output / "result.json"), workers, scored


def test_fixed_six_slots_continue_after_ref_failure_and_report_misclassification(tmp_path, monkeypatch):
    c = {(case, "OFF"): 10 for case in trial.CASE_IDS}
    c.update({(case, "NATIVE_H1"): 28 for case in trial.EVAL_IDS})
    c[(trial.EVAL_IDS[0], "OFF")] = 24
    store, workers, scored = _fake_run(tmp_path, monkeypatch, c,
                                       invalid=(trial.REF_IDS[0], "OFF"))
    assert workers == list(trial.CASE_IDS) and len(scored) == 6
    assert store.data["fixed_denominator"] == {"sources": 4, "mp4_score_slots": 6, "frames": 1086}
    assert store.data["status"] == "INCOMPLETE"
    assert store.data["scored_media_slots"] == 5 and store.data["invalid_media_slots"] == 1
    assert store.case(trial.REF_IDS[0])["slots"]["OFF"]["decision"] is None
    errors = store.data["reporting"]["observed_misclassifications"]
    assert [(e["case_id"], e["arm"]) for e in errors] == [(trial.EVAL_IDS[0], "OFF")]
    assert store.data["calls"]["generation"]["attempted"] == 4
    assert store.data["native_scheduler_calls"]["attempted"] == 204


def test_complete_six_slots_threshold_and_call_caps(tmp_path, monkeypatch):
    c = {(case, "OFF"): 23 for case in trial.CASE_IDS}
    c.update({(case, "NATIVE_H1"): 24 for case in trial.EVAL_IDS})
    store, workers, _ = _fake_run(tmp_path, monkeypatch, c)
    assert workers == list(trial.CASE_IDS)
    assert store.data["status"] == "FUNCTIONAL_BLIND_PRESENCE_OBSERVED"
    assert store.data["scored_frames"] == 1086
    for kind, cap in trial.PLAN.items():
        assert store.data["calls"][kind] == {"attempted": cap, "completed": cap}
    assert store.data["reporting"]["observed_misclassifications"] == []
    assert json.loads(store.path.read_text())["status"] == store.data["status"]
