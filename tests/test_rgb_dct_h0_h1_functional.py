"""CPU/fake checks only: no Wan weights, codec, Drive, or GPU."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import nbformat
import pytest

from main.tube_state import rgb_dct_presence as receiver
from experiments.wan_state_clock import rgb_dct_h0_h1_functional_run as trial
from scripts import build_rgb_dct_h0_h1_functional_notebook as builder


pytestmark = pytest.mark.unit


def test_fixed_config_denominator_threshold_and_blind_signatures(tmp_path):
    config = trial.load_config()
    output = tmp_path / "fresh"
    output.mkdir()
    store = trial.Store(output / "result.json",
                        trial.initial_result(config, output, "a" * 40))
    store.save()
    saved = json.loads(store.path.read_text())
    assert trial._sha(trial.PROTOCOL_PATH) == config["protocol_sha256"]
    assert config["adopted_draft_sha256"] == "b3b287958e6a0df291d031c4fa935022b8d0b1dc2cd43dca8c01ca189fc8df7e"
    assert saved["fixed_denominator"] == {"sources": 4, "mp4_score_slots": 6, "frames": 1086}
    assert [row["status"] for case in saved["cases"].values()
            for row in case["slots"].values()] == ["PENDING"] * 6
    assert [list(saved["cases"][case]["slots"]) for case in trial.CASE_IDS] == [
        ["OFF"], ["OFF"], ["OFF", "NATIVE_H1"], ["OFF", "NATIVE_H1"]]
    assert saved["attempted_media_slots"] == 0
    assert config["call_plan_max"]["native_scheduler_total"] == 204
    assert config["call_plan_max"]["backward"] == 0
    assert tuple(inspect.signature(trial.decide).parameters) == ("score", "tau")
    assert tuple(inspect.signature(trial.calibration_threshold).parameters) == ("scores",)
    assert tuple(inspect.signature(trial.score_mp4).parameters) == ("path", "key")
    assert trial.calibration_threshold((0.1, 0.2)) == pytest.approx(0.200001)
    assert trial.decide(0.2, 0.2) == "H0"
    assert trial.decide(0.200001, 0.2) == "H1"
    with pytest.raises(ValueError):
        trial.calibration_threshold((0.1, float("nan")))
    with pytest.raises(ValueError):
        trial.decide(float("nan"), 0.2)


def _fake_supervision(tmp_path, monkeypatch, scores, *, invalid=None, zero=None,
                      nan_cal=False):
    config = trial.load_config()
    output = tmp_path / "fixed"
    output.mkdir()
    monkeypatch.setattr(trial, "apply_carrier",
                        lambda x0, key, polarity: (x0, {"clipped_fraction": 0.0}))
    monkeypatch.setattr(trial, "lift_direction",
                        lambda raw: (np.array([1.0], dtype=np.float32),
                                     {"status": "ZERO_LIFT_DIRECTION" if zero == "lift" else "READY"}))
    seen_workers = []
    seen_scores = []

    def encode_fn(rgb, path, fps, crf):
        assert fps == 8 and crf == 18
        path.write_bytes((str(path) + " fixed fake MP4").encode())

    def score_fn(path, key):
        case_id, arm = path.parents[2].name, path.parent.name
        assert key == config["key_utf8"].encode()
        assert path.is_file()
        seen_scores.append((case_id, arm))
        if (case_id, arm) == invalid:
            return dict(status="DECODE_INVALID", reason="FAKE_READ_FAILURE",
                        frames_used=0, spec_sha256=receiver.SPEC_SHA256,
                        key_id=config["receiver_key_id"])
        return dict(status="SCORED", score=scores[(case_id, arm)], frames_used=181,
                    spec_sha256=receiver.SPEC_SHA256,
                    key_id=config["receiver_key_id"])

    class FakeBackend:
        def __init__(self, store):
            self.store = store
            self.encodes = 0

        def generate_prefix(self, artifact_dir):
            if self.store.active_case in trial.EVAL_IDS:
                assert trial.read_frozen_calibration(self.store.path)["status"] == "FROZEN"
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
            value = (float("nan") if nan_cal and
                     self.store.active_case in trial.CAL_IDS else 0.5)
            return np.broadcast_to(
                np.array(value, dtype=np.float32).reshape(1, 1, 1, 1),
                trial.RGB_SHAPE)

        def encode(self, rgb):
            self.store.count("vae_encode", False)
            self.store.count("vae_encode", True)
            self.encodes += 1
            return np.array([1.0 if self.encodes == 1 else 0.0], dtype=np.float32)

        def controlled_terminal(self, prefix, masked, target):
            assert target == 0.042943312697648145
            self.store.count("unit_response_probe_step", False)
            self.store.count("unit_response_probe_step", True)
            if zero == "response":
                return "ZERO_NATIVE_RESPONSE", None, {"fake_zero": True}
            self.store.count("scheduler_step", False)
            self.store.count("scheduler_step", True)
            return "READY", "NATIVE", {"fake": True}

        def resources(self):
            return {"fake_cpu": True}

        def release(self):
            pass

    def worker_fn(output, config, case_id):
        seen_workers.append(case_id)
        store = trial.Store.open(output / "result.json")
        trial.run_case(store, case_id, config, FakeBackend(store),
                       encode_fn=encode_fn, score_fn=score_fn)
        return None

    trial._supervise(output, config, "a" * 40, worker_fn=worker_fn)
    return trial.Store.open(output / "result.json"), seen_workers, seen_scores


def test_calibration_failure_retains_six_slots_and_skips_all_evaluation(tmp_path, monkeypatch):
    scores = {(trial.CAL_IDS[0], "OFF"): 0.1,
              (trial.CAL_IDS[1], "OFF"): 0.2}
    store, workers, seen = _fake_supervision(
        tmp_path, monkeypatch, scores, invalid=(trial.CAL_IDS[1], "OFF"))
    assert workers == list(trial.CAL_IDS)
    assert seen == [(case, "OFF") for case in trial.CAL_IDS]
    assert store.data["status"] == "INCOMPLETE_UNCALIBRATED"
    assert store.data["calibration"] is None
    assert not (store.path.parent / "calibration.json").exists()
    assert store.case(trial.CAL_IDS[0])["slots"]["OFF"]["status"] == "SCORED"
    assert store.case(trial.CAL_IDS[1])["slots"]["OFF"]["status"] == "DECODE_INVALID"
    assert [slot["status"] for case_id in trial.EVAL_IDS
            for slot in store.case(case_id)["slots"].values()] == [
                "NOT_RUN_UNCALIBRATED"] * 4
    assert store.data["fixed_denominator"]["mp4_score_slots"] == 6
    assert store.data["attempted_media_slots"] == 2
    assert store.data["calls"]["generation"]["attempted"] == 2
    assert store.data["calls"]["mp4_save"]["attempted"] == 2


def test_nan_calibration_rgb_cannot_encode_score_or_freeze(tmp_path, monkeypatch):
    scores = {(trial.CAL_IDS[0], "OFF"): 0.1,
              (trial.CAL_IDS[1], "OFF"): 0.2}
    store, workers, seen = _fake_supervision(
        tmp_path, monkeypatch, scores, nan_cal=True)
    assert workers == list(trial.CAL_IDS)
    assert seen == []  # The fake receiver was never called.
    assert store.data["calibration"] is None
    assert not (store.path.parent / "calibration.json").exists()
    assert all(store.case(case)["slots"]["OFF"]["status"] == "ENGINEERING_INVALID"
               for case in trial.CAL_IDS)
    assert all("finite RGB" in store.case(case)["slots"]["OFF"]["reason"]
               for case in trial.CAL_IDS)
    assert [slot["status"] for case_id in trial.EVAL_IDS
            for slot in store.case(case_id)["slots"].values()] == [
                "NOT_RUN_UNCALIBRATED"] * 4
    assert store.data["status"] == "INCOMPLETE_UNCALIBRATED"
    assert store.data["attempted_media_slots"] == 0
    assert store.data["calls"]["mp4_save"]["attempted"] == 0
    assert store.data["calls"]["score"]["attempted"] == 0
    assert not list(store.path.parent.rglob("*.mp4"))


def test_full_fake_blind_presence_freezes_before_eval_and_uses_full_budget(tmp_path, monkeypatch):
    scores = {
        (trial.CAL_IDS[0], "OFF"): 0.1,
        (trial.CAL_IDS[1], "OFF"): 0.2,
        (trial.EVAL_IDS[0], "OFF"): 0.15,
        (trial.EVAL_IDS[0], "NATIVE_H1"): 0.31,
        (trial.EVAL_IDS[1], "OFF"): 0.18,
        (trial.EVAL_IDS[1], "NATIVE_H1"): 0.22,
    }
    store, workers, seen = _fake_supervision(tmp_path, monkeypatch, scores)
    assert workers == list(trial.CASE_IDS)
    assert seen == [(case, arm) for case in trial.CAL_IDS for arm in ("OFF",)] + [
        (case, arm) for case in trial.EVAL_IDS for arm in ("OFF", "NATIVE_H1")]
    frozen = trial.read_frozen_calibration(store.path)
    assert frozen["tau"] == pytest.approx(0.200001)
    assert [row["case_id"] for row in frozen["inputs"]] == list(trial.CAL_IDS)
    assert all(row["media_sha256"] for row in frozen["inputs"])
    assert store.data["status"] == "FUNCTIONAL_BLIND_PRESENCE_OBSERVED"
    assert store.data["scored_media_slots"] == 6
    assert store.data["scored_frames"] == 1086
    assert store.data["attempted_media_slots"] == 6
    assert store.data["reporting"]["joined_after_decision_persistence"]
    assert all(row == {"OFF": "H0", "NATIVE_H1": "H1"}
               for row in store.data["reporting"]["decisions"].values())
    for kind, cap in trial.PLAN.items():
        assert store.data["calls"][kind]["attempted"] == cap
        assert store.data["calls"][kind]["completed"] == cap
    assert store.data["native_scheduler_calls"]["attempted"] == 204
    assert [store.case(case)["calls"]["vae_encode"]["attempted"]
            for case in trial.CASE_IDS] == [0, 0, 2, 2]
    assert all(not (store.path.parent / case / "received_videos" / "PIXEL_PLUS").exists()
               for case in trial.EVAL_IDS)


def test_complete_misclassification_is_fixed_negative_without_eval_gate(tmp_path, monkeypatch):
    scores = {
        (trial.CAL_IDS[0], "OFF"): 0.1,
        (trial.CAL_IDS[1], "OFF"): 0.2,
        (trial.EVAL_IDS[0], "OFF"): 0.3,
        (trial.EVAL_IDS[0], "NATIVE_H1"): 0.1,
        (trial.EVAL_IDS[1], "OFF"): 0.1,
        (trial.EVAL_IDS[1], "NATIVE_H1"): 0.25,
    }
    store, workers, seen = _fake_supervision(tmp_path, monkeypatch, scores)
    assert workers == list(trial.CASE_IDS)
    assert len(seen) == 6  # Both NATIVE_H1 attempts despite OFF/threshold outcomes.
    assert store.data["status"] == "FIXED_FUNCTIONAL_NEGATIVE"
    assert store.data["reporting"]["decisions"][trial.EVAL_IDS[0]] == {
        "OFF": "H1", "NATIVE_H1": "H0"}


@pytest.mark.parametrize("zero,expected", [
    ("lift", "ZERO_LIFT_DIRECTION"),
    ("response", "ZERO_NATIVE_RESPONSE"),
])
def test_finite_zero_native_paths_are_method_negatives_with_retained_slots(
        tmp_path, monkeypatch, zero, expected):
    scores = {(case, "OFF"): value for case, value in zip(
        trial.CASE_IDS, (0.1, 0.2, 0.15, 0.18))}
    store, workers, seen = _fake_supervision(tmp_path, monkeypatch, scores, zero=zero)
    assert workers == list(trial.CASE_IDS)
    assert len(seen) == 4
    assert all(store.case(case)["slots"]["NATIVE_H1"]["status"] == expected
               for case in trial.EVAL_IDS)
    assert all(store.case(case)["status"] == "METHOD_NEGATIVE"
               for case in trial.EVAL_IDS)
    assert store.data["status"] == "INCOMPLETE"
    assert store.data["attempted_media_slots"] == 4
    assert store.data["fixed_denominator"]["mp4_score_slots"] == 6


def test_invalid_evaluation_media_is_incomplete_not_correct_rejection(tmp_path, monkeypatch):
    scores = {
        (trial.CAL_IDS[0], "OFF"): 0.1,
        (trial.CAL_IDS[1], "OFF"): 0.2,
        (trial.EVAL_IDS[0], "OFF"): 0.15,
        (trial.EVAL_IDS[0], "NATIVE_H1"): 0.31,
        (trial.EVAL_IDS[1], "OFF"): 0.18,
        (trial.EVAL_IDS[1], "NATIVE_H1"): 0.22,
    }
    store, workers, _ = _fake_supervision(
        tmp_path, monkeypatch, scores, invalid=(trial.EVAL_IDS[0], "OFF"))
    assert workers == list(trial.CASE_IDS)
    assert store.data["status"] == "INCOMPLETE"
    assert store.case(trial.EVAL_IDS[0])["slots"]["NATIVE_H1"]["status"] == "SCORED"
    assert store.case(trial.EVAL_IDS[0])["slots"]["OFF"]["decision"] is None
    assert store.data["scored_media_slots"] == 5


def test_builder_produces_only_temporary_static_immutable_notebook(tmp_path):
    path = builder.build("a" * 40, tmp_path / "static.ipynb")
    notebook = json.loads(path.read_text())
    nbformat.validate(nbformat.read(path, as_version=4))
    assert "".join(notebook["cells"][0]["source"]) == (
        "from google.colab import drive\ndrive.mount('/content/drive')")
    cells = ["".join(cell["source"]) for cell in notebook["cells"]]
    assert "dev/rgb-dct-h0-h1-functional-v1" in cells[4]
    assert "rgb_dct_h0_h1_functional_run" in cells[6]
    assert "RGB-DCT-H0-H1-Functional-V1" in cells[2]
    assert notebook["metadata"]["source_commit"] == "a" * 40
    assert "diffusers" in cells[3]
