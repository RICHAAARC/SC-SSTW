"""CPU checks for the frozen B2 roster and native terminal direction."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.wan_state_clock import rgb_dct_b2_terminal_receiver_run as trial
from main.tube_state import rgb_dct_group_consistency as receiver
from runtime.wan import trajectory
from runtime.wan.rgb_dct_b2_terminal_receiver_backend import WanB2TerminalReceiverBackend
from runtime.wan.rgb_dct_terminal_gradient_backend import WanTerminalGradientBackend


pytestmark = pytest.mark.unit


class TinyT49Scheduler:
    def __init__(self, sign=-1):
        self.step_index = 49
        self.timesteps = torch.arange(50)
        self.sigmas = torch.ones(51)
        self.sigmas[50] = 0
        self.sign = sign

    def step(self, velocity, timestep, z, return_dict=False):
        assert self.step_index == 49 and int(timestep) == 49
        self.step_index = 50
        return (z + self.sign * self.sigmas[49] * velocity,)


def _tiny_backend(sign=-1):
    counts = []
    backend = WanB2TerminalReceiverBackend({}, lambda kind, completed: counts.append((kind, completed)))
    z = torch.full((1, 1, 46, 1, 1), 2.0)
    v = torch.full_like(z, 0.2)
    snapshot = TinyT49Scheduler(sign)
    off = z + sign * v
    gradient = torch.ones_like(z)
    gradient[:, :, 0] = gradient[:, :, 45] = 0
    backend.phase = "TRANSFORMER"
    backend.nodes[49] = {"z": z, "v": v}
    backend.snapshots[49] = snapshot
    backend.off_terminal = off
    backend.gradient = gradient
    return backend, counts, trajectory.fingerprint(vars(snapshot))


def test_fixed_b2_card_roster_and_call_plan():
    config = trial.load_config()
    assert trial._sha(trial.PROTOCOL_PATH) == config["frozen_task_card_sha256"]
    assert tuple(row["id"] for row in config["cases"]) == trial.CASE_IDS
    assert trial.ARMS == ("OFF", "SINGLE49", "TERMINAL49_RECEIVER")
    assert config["control"]["gradient_index"] == 49
    assert "tail_vjp" not in config["call_plan_max"]
    assert config["fixed_denominator"]["frames"] == 1086


def test_full_terminal_gradient_masks_only_two_time_endpoints(monkeypatch):
    backend = WanB2TerminalReceiverBackend({}, lambda kind, completed: None)

    def fake_full_gradient(self, key):
        self.cotangent = torch.ones((1, 1, 46, 1, 1))
        return None, {"loss": 0.25, "q": [-1.0] * 30}

    monkeypatch.setattr(WanTerminalGradientBackend, "terminal_cotangent", fake_full_gradient)
    _, receipt = backend.terminal_cotangent(b"fixed-key")
    assert receipt["loss"] == 0.25
    assert torch.count_nonzero(backend.gradient) == 44
    assert backend.gradient[:, :, 0].sum() == 0
    assert backend.gradient[:, :, 45].sum() == 0
    assert backend.terminal_gradient_receipt["support_time"] == [1, 45]


def test_native_t49_negative_gradient_probe_and_response():
    backend, counts, original_history = _tiny_backend()
    state, terminal, metrics = backend.control_terminal49_receiver(trial.R_STAR)
    assert state == "READY"
    assert terminal is not None
    assert metrics["g_dot_actual_terminal_delta"] < 0
    assert metrics["actual_D"]["support_rms"] == pytest.approx(trial.R_STAR, rel=2e-5)
    assert torch.equal(terminal[:, :, 0], backend.off_terminal[:, :, 0])
    assert torch.equal(terminal[:, :, 45], backend.off_terminal[:, :, 45])
    assert trajectory.fingerprint(vars(backend.snapshots[49])) == original_history
    assert counts == [
        ("unit_response_probe_step", False), ("unit_response_probe_step", True),
        ("scheduler_step", False), ("scheduler_step", True),
    ]


def test_non_descent_native_mapping_retains_invalid_reason():
    backend, _, original_history = _tiny_backend(sign=1)
    state, terminal, metrics = backend.control_terminal49_receiver(trial.R_STAR)
    assert state == "NON_DESCENT_DIRECTION"
    assert terminal is None
    assert metrics["g_dot_actual_terminal_delta"] >= 0
    assert trajectory.fingerprint(vars(backend.snapshots[49])) == original_history


def test_initial_six_slots_and_fixed_identity(tmp_path):
    config = trial.load_config()
    result = trial.initial_result(config, tmp_path, "a" * 40)
    assert result["frozen_task_card_sha256"] == trial._sha(trial.PROTOCOL_PATH)
    assert result["config_sha256"] == trial._sha(trial.CONFIG_PATH)
    assert result["source_sha"] == "a" * 40
    assert len([slot for case in result["cases"].values() for slot in case["slots"].values()]) == 6
    assert all(slot["status"] == "PENDING" for case in result["cases"].values()
               for slot in case["slots"].values())


class FakeRunBackend:
    def __init__(self, store, *, bad_response=False):
        self.store = store
        self.bad_response = bad_response
        self.terminals = {}
        self.terminal_gradient_receipt = {"masked_gradient": {"support_rms": 1.0}}

    def _calls(self, kind, count):
        for _ in range(count):
            self.store.count(kind, False)
            self.store.count(kind, True)

    def prepare_off(self, artifact_dir):
        self._calls("generation", 1)
        self._calls("transformer_prefix", 100)
        self._calls("scheduler_prefix", 50)
        return {"off_terminal_fingerprint": "fake"}

    def to_vae_phase(self, name):
        return {"name": name}

    def to_transformer_phase(self, name):
        return {"name": name}

    def terminal_cotangent(self, key):
        self._calls("vae_gradient_decode", 1)
        self._calls("vae_vjp", 1)
        return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE), {
            "loss": 1.0, "q": [-1.0] * 30, "proxy_validation": {"matched": True},
        }

    def encode(self, rgb):
        self._calls("vae_encode", 1)
        return np.ones((1, 1, 46, 1, 1), dtype=np.float32)

    def control_terminal49_receiver(self, target):
        self._calls("unit_response_probe_step", 1)
        self._calls("scheduler_step", 1)
        metrics = {"g_dot_actual_terminal_delta": -target,
                   "actual_D": {"support_rms": target}}
        if self.bad_response:
            metrics["g_dot_actual_terminal_delta"] = target
            return "NON_DESCENT_DIRECTION", None, metrics
        self.terminals["TERMINAL49_RECEIVER"] = torch.tensor([49.0])
        return "READY", self.terminals["TERMINAL49_RECEIVER"], metrics

    def control_single49(self, direction, target):
        self._calls("unit_response_probe_step", 1)
        self._calls("scheduler_step", 1)
        self.terminals["SINGLE49"] = torch.tensor([49.0])
        return "READY", self.terminals["SINGLE49"], {"actual_D": {"support_rms": target}}

    def decode(self, terminal):
        self._calls("vae_decode", 1)
        return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)

    def resources(self):
        return {"receipt_valid": True, "cuda": {"peak_allocated": 1, "peak_reserved": 1}}

    def release(self):
        pass


@pytest.mark.parametrize("bad_response", [False, True])
def test_fake_six_slot_run_keeps_negative_direction_failure(tmp_path, monkeypatch, bad_response):
    config = trial.load_config()
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(trial, "apply_carrier", lambda rgb, key, polarity: (rgb, {}))
    monkeypatch.setattr(trial, "lift_direction", lambda raw: (raw, {"status": "READY"}))
    monkeypatch.setattr(trial, "_quantized_rgb", lambda rgb: rgb)
    monkeypatch.setattr(trial, "_paired_quality", lambda off, marked: {
        "rgb_rmse": 0.0, "rgb_psnr_infinite": True, "compared_frames": 181,
    })
    monkeypatch.setattr(trial, "_gradient_effect", lambda off, rgb, key: {
        "actual_float_rgb_loss_delta": -0.5, "gain_groups": [], "loss_groups": [],
    })
    monkeypatch.setattr(trial, "_score_memory_layer", lambda rgb, key: {
        "status": "SCORED", "reason": None, "score": 0.0,
        "group_scores": [-1.0] * 30, "positive_groups": 0,
        "frames_used": 181, "decision": None,
    })

    def encode_fn(rgb, path, fps, crf):
        path.write_bytes(b"fake mp4")

    def score_fn(path, key):
        arm = path.parent.name
        c = 10 if arm == "OFF" else 27
        return ({
            "status": "SCORED", "reason": None, "score": c / 100,
            "group_scores": [1.0] * c + [-1.0] * (30 - c),
            "positive_groups": c, "frames_used": 181,
            "spec_sha256": receiver.SPEC_SHA256,
            "key_id": config["receiver_key_id"],
        }, np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE))

    def worker_fn(output, config, case_id):
        store = trial.Store.open(output / "result.json")
        store.case(case_id)["environment_receipt"] = {"status": "VALID"}
        store.save()
        backend = FakeRunBackend(store, bad_response=bad_response and case_id == trial.CASE_IDS[0])
        trial.run_case(store, case_id, config, backend,
                       encode_fn=encode_fn, score_fn=score_fn)

    trial._supervise(output, config, "a" * 40, worker_fn=worker_fn)
    result = trial.Store.open(output / "result.json").data
    assert len([slot for case in result["cases"].values()
                for slot in case["slots"].values()]) == 6
    if bad_response:
        failed = result["cases"][trial.CASE_IDS[0]]["slots"]["TERMINAL49_RECEIVER"]
        assert failed["status"] == "ENGINEERING_INVALID"
        assert "NON_DESCENT_DIRECTION" in failed["reason"]
        assert result["status"] == "INCOMPLETE"
        assert result["scored_media_slots"] == 5
    else:
        assert result["status"] == "FIXED_B2_TERMINAL_RECEIVER_COMPLETE"
        assert result["scored_media_slots"] == 6
        assert result["scored_frames"] == 1086
