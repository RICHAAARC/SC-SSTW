"""Directed CPU/fake checks; no Wan model, MP4 codec, Drive, or GPU run."""
from __future__ import annotations

import inspect
import ast
import importlib.metadata
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from main.tube_state import rgb_dct_presence as receiver
from main.tube_state import rgb_dct_t49_carrier as carrier
from runtime.wan import rgb_dct_presence_adapter
from experiments.wan_state_clock import rgb_dct_t49_vae_lift_run as trial
from scripts import build_rgb_dct_t49_vae_lift_notebook as builder


pytestmark = pytest.mark.unit


def _store(tmp_path):
    config = trial.load_config()
    output = tmp_path / "fresh"
    output.mkdir()
    store = trial.Store(output / "result.json", trial.initial_result(config, output, "a" * 40))
    store.save()
    return config, store


def test_protocol_config_and_analytic_nonclipped_dct_delta():
    config = trial.load_config()
    assert trial._sha(trial.PROTOCOL_PATH) == config["protocol_sha256"]
    assert config["receiver_spec_sha256"] == receiver.SPEC_SHA256
    assert tuple(x["id"] for x in config["cases"]) == trial.CASE_IDS
    assert config["call_plan_max"]["native_scheduler_total"] == 104
    assert config["call_plan_max"]["transformer"] == 200
    assert config["fixed_denominator"] == {"sources": 2, "mp4_score_slots": 8, "frames": 1448}
    assert config["call_plan_max"]["backward"] == 0
    key = config["key_utf8"].encode()
    spatial, _, temporal = receiver.key_codes(key)
    kernel = receiver.dct_difference_kernel()
    for t, b in ((0, 0), (180, 159)):
        patch = carrier.carrier_patch(key, t, b)
        assert np.max(np.abs(patch)) < 0.5
        baseline = np.full((32, 32), 0.5)
        off = float(np.sum(baseline * kernel))
        positive = float(np.sum((baseline + patch) * kernel))
        negative = float(np.sum((baseline - patch) * kernel))
        delta = 0.2 * temporal[t] * spatial[b]
        assert positive - off == pytest.approx(delta, abs=1e-13)
        assert negative - off == pytest.approx(-delta, abs=1e-13)


def test_finite_zero_lift_and_zero_unit_response_are_method_negatives():
    raw = np.zeros(carrier.LATENT_SHAPE, dtype=np.float32)
    raw[:, :, 0] = 1
    raw[:, :, 45] = -1
    masked, metrics = carrier.lift_direction(raw)
    assert metrics["status"] == "ZERO_LIFT_DIRECTION"
    assert metrics["raw_global_rms"] > 0 and metrics["masked_support_rms"] == 0
    assert np.count_nonzero(masked) == 0
    assert carrier.classify_unit_response(0.0) == "ZERO_NATIVE_RESPONSE"
    with pytest.raises(ValueError):
        carrier.classify_unit_response(float("nan"))
    with pytest.raises(ValueError):
        carrier.lift_direction(np.full(carrier.LATENT_SHAPE, np.nan, dtype=np.float32))


def test_eight_slots_are_persisted_before_any_call_and_budget_is_bounded(tmp_path):
    config, store = _store(tmp_path)
    persisted = json.loads(store.path.read_text())
    assert persisted["fixed_denominator"]["mp4_score_slots"] == 8
    assert [row["status"] for case in persisted["cases"].values()
            for row in case["slots"].values()] == ["PENDING"] * 8
    assert persisted["attempted_media_slots"] == 0
    for _ in range(config["call_plan_max"]["generation"]):
        store.count("generation", False)
        store.count("generation", True)
    with pytest.raises(RuntimeError, match="cap"):
        store.count("generation", False)
    assert store.data["calls"]["generation"]["completed"] == 2


def test_mp4_scoring_sees_only_path_and_key_and_records_score_change(tmp_path):
    config, store = _store(tmp_path)
    key = config["key_utf8"].encode()
    assert tuple(inspect.signature(rgb_dct_presence_adapter.score_mp4).parameters) == ("path", "key")
    assert trial._save_and_score.__kwdefaults__["score_fn"] is rgb_dct_presence_adapter.score_mp4
    rgb = np.broadcast_to(np.array(0, dtype=np.float32).reshape(1, 1, 1, 1),
                          (181, 320, 512, 3))
    seen = []

    def fake_encode(frames, path, fps, crf):
        assert fps == 8 and crf == 18
        path.write_bytes(b"one fake MP4 byte stream")

    def fake_score(path, passed_key):
        seen.append((path, passed_key))
        return dict(status="SCORED", score=0.125, frames_used=181,
                    spec_sha256=receiver.SPEC_SHA256,
                    key_id=config["receiver_key_id"])

    trial._save_and_score(store, trial.CASE_IDS[0], "OFF", rgb, key, config,
                          encode_fn=fake_encode, score_fn=fake_score)
    slot = store.case(trial.CASE_IDS[0])["slots"]["OFF"]
    assert slot["status"] == "SCORED" and slot["pre_encode_score"] == 0.0
    assert slot["score"] == 0.125 and slot["score_change_after_codec"] == 0.125
    assert seen == [(Path(slot["path"]), key)]
    assert store.data["calls"]["mp4_save"] == {"attempted": 1, "completed": 1}
    assert store.data["calls"]["mp4_read"] == {"attempted": 1, "completed": 1}
    assert store.data["calls"]["score"] == {"attempted": 1, "completed": 1}


def test_two_fixed_sources_keep_negative_stops_and_call_ceiling(tmp_path, monkeypatch):
    config, store = _store(tmp_path)
    monkeypatch.setattr(trial, "apply_carrier", lambda x0, key, polarity: (x0, {"clipped_fraction": 0.0}))
    scores = {"OFF": 0.0, "PIXEL_PLUS": 0.2, "PIXEL_MINUS": -0.2}

    def fake_media(store, case_id, arm, rgb, key, config, **kwargs):
        for kind in ("mp4_save", "mp4_read", "score"):
            store.count(kind, False)
            store.count(kind, True)
        store.case(case_id)["slots"][arm].update(
            status="SCORED", score=scores[arm], frames_used=181,
            pre_encode_score=scores[arm], score_change_after_codec=0.0,
        )
        store.save()

    monkeypatch.setattr(trial, "_save_and_score", fake_media)

    class FakeBackend:
        def __init__(self, zero_lift):
            self.zero_lift = zero_lift
            self.encodes = 0
            self.native_calls = 0

        def generate_prefix(self, artifact_dir):
            saved = json.loads(store.path.read_text())
            assert sum(row["status"] == "PENDING" for c in saved["cases"].values()
                       for row in c["slots"].values()) >= 4
            store.count("generation", False)
            for _ in range(100):
                store.count("transformer", False)
                store.count("transformer", True)
            for _ in range(50):
                store.count("scheduler_step", False)
                store.count("scheduler_step", True)
            store.count("generation", True)
            return SimpleNamespace(z0="fake terminal", metadata={"fixed_prefix": True})

        def load_vae(self):
            return {"fake_vae": True}

        def decode(self, terminal):
            store.count("vae_decode", False)
            store.count("vae_decode", True)
            return np.array([0.5], dtype=np.float32)

        def encode(self, rgb):
            store.count("vae_encode", False)
            store.count("vae_encode", True)
            self.encodes += 1
            value = 0.0 if self.zero_lift or self.encodes == 2 else 1.0
            out = np.zeros(carrier.LATENT_SHAPE, dtype=np.float32)
            out[:, :, 1:45] = value
            return out

        def controlled_terminal(self, prefix, direction, target):
            self.native_calls += 1
            store.count("unit_response_probe_step", False)
            store.count("unit_response_probe_step", True)
            return "ZERO_NATIVE_RESPONSE", None, {"unit_D": {"support_rms": 0.0}}

        def resources(self):
            return {"fake_cpu": True}

        def release(self):
            pass

    first = FakeBackend(zero_lift=True)
    second = FakeBackend(zero_lift=False)
    trial.run_case(store, trial.CASE_IDS[0], config, first)
    trial.run_case(store, trial.CASE_IDS[1], config, second)
    trial.finalize_experiment(store)
    assert store.case(trial.CASE_IDS[0])["slots"]["NATIVE_PLUS"]["status"] == "ZERO_LIFT_DIRECTION"
    assert store.case(trial.CASE_IDS[1])["slots"]["NATIVE_PLUS"]["status"] == "ZERO_NATIVE_RESPONSE"
    assert first.native_calls == 0 and second.native_calls == 1
    assert all(store.case(c)["status"] == "FIXED_CONSTRUCTION_NEGATIVE" for c in trial.CASE_IDS)
    assert store.data["status"] == "FIXED_CONSTRUCTION_NEGATIVE"
    assert store.data["calls"]["generation"]["attempted"] == 2
    assert store.data["calls"]["transformer"]["attempted"] == 200
    assert store.data["native_scheduler_calls"]["attempted"] == 101
    assert store.data["scored_media_slots"] == 6 and store.data["invalid_media_slots"] == 2
    assert store.data["attempted_media_slots"] == 6
    assert all(store.case(c)["slots"]["NATIVE_PLUS"]["score"] is None for c in trial.CASE_IDS)
    assert all(store.data["calls"][kind]["attempted"] <= cap for kind, cap in trial.PLAN.items())


def test_channel_gate_and_native_ratio_are_not_detection_thresholds():
    slots = {arm: dict(status="SCORED", score=score) for arm, score in zip(
        trial.ARMS, (0.1, 0.3, -0.1, 0.121)
    )}
    assert trial.channel_gate(slots)["status"] == "POSITIVE"
    result = trial.case_comparison(slots, 0.1)
    assert result["status"] == "DEVELOPMENT_CONTROLLED"
    assert result["native_delta"] == pytest.approx(0.021)
    slots["NATIVE_PLUS"]["score"] = 0.11
    assert trial.case_comparison(slots, 0.1)["status"] == "FIXED_CONSTRUCTION_NEGATIVE"


def test_two_source_positive_path_uses_numpy_rgb_and_fixed_full_budget(tmp_path):
    config, store = _store(tmp_path)
    key = config["key_utf8"].encode()
    scores = {"OFF": 0.0, "PIXEL_PLUS": 0.2, "PIXEL_MINUS": -0.2,
              "NATIVE_PLUS": 0.03}
    media_calls = []

    def fake_encoder(frames, path, fps, crf):
        assert isinstance(frames, np.ndarray) and frames.shape == carrier.RGB_SHAPE
        assert frames.dtype == np.float32 and fps == 8 and crf == 18
        media_calls.append(("save", path.parent.name))
        path.write_bytes(b"fixed fake media")

    def fake_score(path, passed_key):
        assert passed_key == key
        media_calls.append(("score", path.parent.name))
        return dict(status="SCORED", score=scores[path.parent.name], frames_used=181,
                    spec_sha256=receiver.SPEC_SHA256,
                    key_id=config["receiver_key_id"])

    class CompleteFakeBackend:
        def __init__(self):
            self.encodes = 0

        def generate_prefix(self, artifact_dir):
            store.count("generation", False)
            for _ in range(100):
                store.count("transformer", False)
                store.count("transformer", True)
            for _ in range(50):
                store.count("scheduler_step", False)
                store.count("scheduler_step", True)
            store.count("generation", True)
            return SimpleNamespace(z0="off terminal", metadata={"cursor": 49})

        def load_vae(self):
            return {"same_frozen_vae": True}

        def decode(self, terminal):
            store.count("vae_decode", False)
            store.count("vae_decode", True)
            return np.broadcast_to(np.array(0.5, dtype=np.float32).reshape(1, 1, 1, 1),
                                   carrier.RGB_SHAPE)

        def encode(self, rgb):
            assert isinstance(rgb, np.ndarray) and rgb.shape == carrier.RGB_SHAPE
            store.count("vae_encode", False)
            store.count("vae_encode", True)
            self.encodes += 1
            value = 1.0 if self.encodes == 1 else 0.0
            out = np.zeros(carrier.LATENT_SHAPE, dtype=np.float32)
            out[:, :, 1:45] = value
            return out

        def controlled_terminal(self, prefix, masked, target):
            assert masked.shape == carrier.LATENT_SHAPE
            assert np.count_nonzero(masked[:, :, 0]) == 0
            assert np.count_nonzero(masked[:, :, 45]) == 0
            assert target == 0.042943312697648145
            store.count("unit_response_probe_step", False)
            store.count("unit_response_probe_step", True)
            store.count("scheduler_step", False)
            store.count("scheduler_step", True)
            return "READY", "native terminal", {"actual_D": {"support_rms": target}}

        def resources(self):
            return {"fake_cpu": True}

        def release(self):
            pass

    for case_id in trial.CASE_IDS:
        trial.run_case(store, case_id, config, CompleteFakeBackend(),
                       encode_fn=fake_encoder, score_fn=fake_score)
    trial.finalize_experiment(store)
    assert store.data["status"] == "DEVELOPMENT_CONTROLLABILITY_OBSERVED"
    assert store.data["scored_media_slots"] == 8
    assert store.data["attempted_media_slots"] == 8
    assert store.data["scored_frames"] == 1448
    assert store.data["calls"]["generation"] == {"attempted": 2, "completed": 2}
    assert store.data["calls"]["transformer"] == {"attempted": 200, "completed": 200}
    assert store.data["native_scheduler_calls"] == {"attempted": 104, "completed": 104}
    assert store.data["calls"]["vae_decode"] == {"attempted": 4, "completed": 4}
    assert store.data["calls"]["vae_encode"] == {"attempted": 4, "completed": 4}
    assert store.data["calls"]["backward"] == {"attempted": 0, "completed": 0}
    for kind in ("mp4_save", "mp4_read", "score"):
        assert store.data["calls"][kind] == {"attempted": 8, "completed": 8}
    assert len(media_calls) == 16
    for case_id in trial.CASE_IDS:
        assert store.case(case_id)["channel_gate"]["status"] == "POSITIVE"
        assert store.case(case_id)["comparison"]["status"] == "DEVELOPMENT_CONTROLLED"
        assert store.case(case_id)["calls"]["transformer"] == {"attempted": 100, "completed": 100}
        assert store.case(case_id)["calls"]["scheduler_step"] == {"attempted": 51, "completed": 51}


def test_temporary_notebook_builder_binds_one_source_and_run_all(tmp_path):
    with pytest.raises(ValueError):
        builder.build("not-a-sha", tmp_path / "bad.ipynb")
    sha = "b" * 40
    path = builder.build(sha, tmp_path / "temporary-only.ipynb")
    notebook = json.loads(path.read_text())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert codes[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert notebook["metadata"]["source_commit"] == sha
    assert len(codes) == 7 and all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    for code in codes:
        ast.parse(code)
    text = "\n".join(codes)
    assert "SOURCE_SHA = '" + sha + "'" in text
    assert "actual != SOURCE_SHA" in text
    assert "experiments.wan_state_clock.rgb_dct_t49_vae_lift_run" in text
    assert text.count("'--output', str(OUTPUT)") == 1
    assert "--config" not in text and "--arm" not in text and "--key" not in text


@pytest.mark.parametrize(
    ("installed_torch", "cuda_available", "expect_torch_install", "expect_failure"),
    [("2.11.0+cu130", True, False, False),
     ("2.11.0+cu128", True, False, False),
     (None, True, True, False),
     ("2.11.1+cu130", True, True, False),
     ("2.11.0+cu130", False, False, True)],
)
def test_install_cell_accepts_cu130_without_suffix_gate_and_checks_cuda(
    tmp_path, monkeypatch, installed_torch, cuda_available, expect_torch_install, expect_failure,
):
    path = builder.build("b" * 40, tmp_path / "temporary-install-check.ipynb")
    notebook = json.loads(path.read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    install = code[2][code[2].index("import importlib.metadata, subprocess, sys"):]
    ast.parse(install)
    assert "if (version('torch') or '').split('+', 1)[0] != '2.11.0':" in install
    assert "2.11.0+cu128" not in install
    assert "str(torch.__version__).split('+', 1)[0] == '2.11.0'" in install
    assert "assert torch.cuda.is_available()" in install
    assert "assert diffusers.__version__ == '0.40.0'" in install
    assert "from diffusers import WanPipeline, AutoencoderKLWan" in install
    assert "python=sys.version" in code[4]
    assert "torch_cuda_runtime=torch.version.cuda" in code[4]
    assert "device=(torch.cuda.get_device_name(0)" in code[4]

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = installed_torch or "2.11.0+cu128"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.__version__ = "0.40.0"
    fake_diffusers.WanPipeline = type("WanPipeline", (), {})
    fake_diffusers.AutoencoderKLWan = type("AutoencoderKLWan", (), {})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    current = {"torch": installed_torch}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: (
        current["torch"] if name == "torch" else "0.40.0" if name == "diffusers" else "test-version"
    ))
    commands = []

    def fake_logged_run(command, **kwargs):
        commands.append(command)
        if "torch==2.11.0" in command:
            current["torch"] = "2.11.0+cu128"
            fake_torch.__version__ = current["torch"]
        if "-c" in command:
            exec(command[-1], {"__name__": "__main__"})

    if expect_failure:
        with pytest.raises(AssertionError, match="CUDA torch required"):
            exec(install, {"logged_run": fake_logged_run})
    else:
        exec(install, {"logged_run": fake_logged_run})
    torch_installs = [cmd for cmd in commands if any(
        isinstance(arg, str) and arg.startswith("torch==") for arg in cmd
    )]
    assert bool(torch_installs) is expect_torch_install
    assert any("diffusers==0.40.0" in cmd for cmd in commands)


def test_timeout_kills_each_worker_group_and_retains_all_eight_slots(tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    output.mkdir()
    config = trial.load_config()
    killed = []

    class TimedOutWorker:
        next_pid = 10000

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True
            self.pid = TimedOutWorker.next_pid
            TimedOutWorker.next_pid += 1
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fixed worker", timeout)
            return -9

    monkeypatch.setattr(trial.subprocess, "Popen", TimedOutWorker)
    monkeypatch.setattr(trial.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    trial._supervise(output, config, "c" * 40)
    result = json.loads((output / "result.json").read_text())
    assert len(killed) == 2
    assert result["attempted_media_slots"] == 0
    assert result["invalid_media_slots"] == 8
    assert result["pending_media_slots"] == 0
    assert result["status"] == "INCOMPLETE_ENGINEERING_INVALID"
    assert all(slot["status"] == "NOT_RUN_RESOURCE_FAILURE"
               for case in result["cases"].values() for slot in case["slots"].values())
    assert result["calls"]["mp4_save"] == {"attempted": 0, "completed": 0}
