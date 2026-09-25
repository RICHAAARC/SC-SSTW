"""CPU/fake validation for the fixed route-B source; no model, GPU, MP4 or Drive."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.wan_state_clock import rgb_dct_terminal_gradient_run as trial
from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state import rgb_dct_terminal_gradient as proxy
from runtime.wan import rgb_dct_terminal_gradient_backend as backend_module
from runtime.wan.gradient_checkpointing import (
    BoundarySpool, BoundaryStorage, ReplayLedger, checkpoint_call,
)


pytestmark = pytest.mark.unit


class FakeBackend:
    def __init__(self, store, *, fail_vjp=False, fail_resource=False,
                 fail_release=False, zero_single=False, zero_terminal=False,
                 fail_cotangent=False, fail_prepare=False):
        self.store = store
        self.phase = "INITIAL"
        self.terminals = {}
        self.fail_vjp = fail_vjp
        self.fail_resource = fail_resource
        self.fail_release = fail_release
        self.zero_single = zero_single
        self.zero_terminal = zero_terminal
        self.fail_cotangent = fail_cotangent
        self.fail_prepare = fail_prepare
        self.events = []

    def _calls(self, kind, count):
        for _ in range(count):
            self.store.count(kind, False)
            self.store.count(kind, True)

    def prepare_off(self, artifact_dir):
        if self.fail_prepare:
            raise RuntimeError("fake shared OFF failure")
        self._calls("generation", 1)
        self._calls("transformer_prefix", 100)
        self._calls("scheduler_prefix", 50)
        self.phase = "TRANSFORMER"
        self.events.append("OFF")
        return {"fake": True, "history_indices": [46, 49]}

    def forecast_terminal46(self):
        self._calls("transformer_forecast", 6)
        self._calls("scheduler_forecast", 4)
        self.events.append("FORECAST")
        return {"terminal_exact_match": True, "history_exact_match": True}

    def to_vae_phase(self, name):
        assert self.phase == "TRANSFORMER"
        self.phase = "VAE"
        self.events.append(name)
        return {"name": name, "vae_resident": "cuda_fp32"}

    def to_transformer_phase(self, name):
        assert self.phase == "VAE"
        self.phase = "TRANSFORMER"
        self.events.append(name)
        return {"name": name, "identity_verified": True}

    def terminal_cotangent(self, key):
        assert self.phase == "VAE"
        if self.fail_cotangent:
            raise RuntimeError("fake cotangent failure")
        self._calls("vae_gradient_decode", 1)
        self.store.replay({"limits": {"transformer_block": 256, "vae_chunk": 64},
                           "counts": {"vae_chunk": {"forward": {"attempted": 46,
                                                                   "completed": 46}}}})
        self._calls("vae_vjp", 1)
        self.events.append("COTANGENT")
        rgb = np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)
        return rgb, {"loss": 1.0, "q": [-1.0] * 30,
                     "proxy_validation": {"matched": True}}

    def encode(self, rgb):
        assert self.phase == "VAE"
        self._calls("vae_encode", 1)
        return np.ones((1, 16, 46, 40, 64), dtype=np.float32)

    def terminal_vjp(self):
        assert self.phase == "TRANSFORMER"
        self._calls("transformer_replay", 6)
        self._calls("scheduler_replay", 4)
        self.store.replay({"limits": {"transformer_block": 256, "vae_chunk": 64},
                           "counts": {"transformer_block": {
                               "forward": {"attempted": 180, "completed": 180},
                               "recompute": {"attempted": 180, "completed": 180}}}})
        if self.fail_vjp:
            raise RuntimeError("fake attached-history mismatch")
        self._calls("tail_vjp", 1)
        self.events.append("VJP")
        return {"terminal_exact_match": True, "history_exact_match": True,
                "masked_gradient": {"support_rms": 1.0}}

    def control_terminal46(self, target):
        assert target == trial.R_STAR and "VJP" in self.events
        self._calls("scheduler_response_shadow", 1)
        self._calls("unit_response_probe_step", 1)
        self._calls("scheduler_step", 1)
        self._calls("transformer_control_tail", 6)
        self._calls("scheduler_control_tail", 3)
        self.terminals["TERMINAL46"] = torch.tensor([46.0])
        self.events.append("CONTROL46")
        state = "ZERO_NATIVE_RESPONSE" if self.zero_terminal else "READY"
        if state != "READY":
            self.terminals.pop("TERMINAL46")
        return state, self.terminals.get("TERMINAL46"), {
            "actual_D": {"support_rms": target}, "target_D_support_rms": target,
        }

    def control_single49(self, direction, target):
        assert target == trial.R_STAR
        self._calls("unit_response_probe_step", 1)
        self._calls("scheduler_step", 1)
        self.terminals["SINGLE49"] = torch.tensor([49.0])
        self.events.append("CONTROL49")
        state = "ZERO_NATIVE_RESPONSE" if self.zero_single else "READY"
        if state != "READY":
            self.terminals.pop("SINGLE49")
        return state, self.terminals.get("SINGLE49"), {
            "actual_D": {"support_rms": target}, "target_D_support_rms": target,
            "terminal_fingerprint": "fake-terminal",
            "final_history_fingerprint": "fake-history",
            "terminal_from_off": {"support_rms": target},
        }

    def decode(self, terminal):
        assert self.phase == "VAE"
        self._calls("vae_decode", 1)
        return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)

    def resources(self):
        return {"fake_cpu": True, "events": self.events,
                "receipt_valid": not self.fail_resource}

    def release(self):
        if self.fail_release:
            raise RuntimeError("fake release failure")
        self.phase = "RELEASED"


def _fake_run(tmp_path, monkeypatch, *, fail_vjp=False, invalid=None,
              fail_resource=False, fail_release=False, zero_single=False,
              zero_terminal=False, fail_cotangent=False, fail_prepare=False,
              zero_lift=False, negative=False):
    config = trial.load_config()
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(trial, "apply_carrier",
                        lambda rgb, key, polarity: (rgb, {"fake": True}))
    monkeypatch.setattr(
        trial, "lift_direction",
        lambda raw: (raw, {
            "status": "ZERO_LIFT_DIRECTION" if zero_lift else "READY",
            "masked_support_rms": 0.0 if zero_lift else 1.0,
        }),
    )
    monkeypatch.setattr(trial, "_quantized_rgb", lambda rgb: rgb)
    monkeypatch.setattr(trial, "_score_memory_layer", lambda rgb, key: {
        "status": "SCORED", "reason": None, "score": 0.0,
        "group_scores": [0.0] * 30, "positive_groups": 0,
        "frames_used": 181, "decision": None,
    })
    backends, reads = [], []

    def encode_fn(rgb, path, fps, crf):
        assert fps == 8 and crf == 18
        path.write_bytes(b"fake MP4")

    def score_fn(path, key):
        case_id, arm = path.parents[2].name, path.parent.name
        reads.append((case_id, arm))
        if (case_id, arm) == invalid:
            return {"status": "INVALID", "reason": "FAKE_INVALID", "frames_used": 0}, None
        c = 10 if arm == "OFF" else (10 if negative and arm == "TERMINAL46" else 27)
        q = [1.0] * c + [-1.0] * (30 - c)
        return ({
            "status": "SCORED", "score": c / 100, "group_scores": q,
            "positive_groups": c, "frames_used": 181,
            "spec_sha256": receiver.SPEC_SHA256,
            "key_id": config["receiver_key_id"],
        }, np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE))

    def worker_fn(output, config, case_id):
        store = trial.Store.open(output / "result.json")
        store.case(case_id)["environment_receipt"] = {"status": "VALID", "fake": True}
        store.data["runtime_environment"][case_id] = {"status": "VALID", "fake": True}
        store.save()
        backend = FakeBackend(
            store, fail_vjp=fail_vjp and case_id == trial.CASE_IDS[0],
            fail_resource=fail_resource and case_id == trial.CASE_IDS[0],
            fail_release=fail_release and case_id == trial.CASE_IDS[0],
            zero_single=zero_single and case_id == trial.CASE_IDS[0],
            zero_terminal=zero_terminal and case_id == trial.CASE_IDS[0],
            fail_cotangent=fail_cotangent and case_id == trial.CASE_IDS[0],
            fail_prepare=(fail_prepare and case_id == trial.CASE_IDS[0]),
        )
        backends.append(backend)
        trial.run_case(store, case_id, config, backend,
                       encode_fn=encode_fn, score_fn=score_fn)
        return None

    trial._supervise(output, config, "a" * 40, worker_fn=worker_fn)
    return trial.Store.open(output / "result.json"), backends, reads


def test_protocol_roster_denominator_calls_and_no_cuda_suffix_gate(tmp_path):
    config = trial.load_config()
    assert trial._sha(trial.PROTOCOL_PATH) == config["protocol_sha256"]
    assert [row["id"] for row in config["cases"]] == list(trial.CASE_IDS)
    assert config["fixed_denominator"] == {
        "sources": 2, "arms_per_source": 3, "mp4_score_slots": 6, "frames": 1086,
    }
    assert config["call_plan_max"] == trial.PLAN
    assert config["observation_layers"] == [
        "float_rgb", "rgb8_quantized", "mp4_rgb24"
    ]
    assert config["resources"]["termination_grace_seconds"] == 10
    assert config["runtime"]["cuda_build_suffix_policy"] == "record_only"
    assert config["runtime"]["torch_version_policy"] == "record_and_api_check"
    source = inspect.getsource(trial.environment_receipt)
    assert "split(\"+\", 1)[0]" in source
    assert "torch public version mismatch" not in source
    assert "cu128" not in source and "cu130" not in source
    store = trial.Store(tmp_path / "result.json",
                        trial.initial_result(config, tmp_path, "a" * 40))
    store.save()
    assert store.data["pending_media_slots"] == 6
    assert all(list(case["slots"]) == list(trial.ARMS)
               for case in store.data["cases"].values())


def test_full_proxy_matches_numpy_receiver_on_181_frame_input():
    key = b"WanProjection-first-validation-key-v1"
    # Expanded storage avoids allocating a second full input video. The proxy
    # still evaluates all 181 frames, 160 blocks and 30 groups.
    frame = torch.full((1, 320, 512, 3), 0.5, dtype=torch.float32)
    rgb = frame.expand(181, -1, -1, -1).requires_grad_(True)
    result = proxy.torch_proxy(rgb, key)
    receipt = proxy.validate_against_numpy(rgb, key, result, rtol=2e-5, atol=2e-6)
    assert receipt["matched"] and len(receipt["q_torch"]) == 30
    assert result["loss"].requires_grad
    gradient, = torch.autograd.grad(result["loss"], rgb)
    assert gradient.shape == rgb.shape and torch.isfinite(gradient).all()


def test_rgb8_evidence_layer_uses_runtime_rounding_semantics():
    rgb = np.array([[[[0.0, 0.5, 1.0],
                      [1.4 / 255, 1.6 / 255, 254.6 / 255]]]], dtype=np.float32)
    quantized = trial._quantized_rgb(rgb)
    expected = np.rint(rgb * 255).astype(np.uint8).astype(np.float32) / 255
    assert quantized.dtype == np.float32
    assert np.array_equal(quantized, expected)
    assert "quantize_rgb8_no_codec" in inspect.getsource(trial._quantized_rgb)


class TinyHistoryScheduler:
    """UniPC-like history dependence for direct-vs-split VJP validation."""
    def __init__(self):
        self.step_index = 46
        self.timesteps = torch.arange(50, dtype=torch.float64)
        self.model_outputs = [torch.tensor([0.2], dtype=torch.float64),
                              torch.tensor([-0.1], dtype=torch.float64)]
    def step(self, velocity, timestep, sample, return_dict=False):
        converted = sample - 0.17 * velocity
        history = self.model_outputs[-1]
        result = 0.71 * sample + 0.23 * converted + 0.06 * history
        self.model_outputs[0] = self.model_outputs[1]
        self.model_outputs[1] = converted
        self.step_index += 1
        return (result,)


class TinyTransformer(torch.nn.Module):
    def forward(self, hidden_states, timestep, encoder_hidden_states,
                attention_kwargs=None, return_dict=False):
        return (0.31 * hidden_states + 0.07 * encoder_hidden_states,)


def _tiny_rollout(source):
    scheduler = TinyHistoryScheduler()
    pipe = SimpleNamespace(transformer=TinyTransformer())
    prompt = torch.tensor([0.4], dtype=torch.float64)
    negative = torch.tensor([-0.2], dtype=torch.float64)
    counts = []
    def count(kind, completed):
        counts.append((kind, completed))
    z = torch.tensor([0.8], dtype=torch.float64)
    terminal = backend_module._graph_step(scheduler, z, source, 46, count)
    for index in range(47, 50):
        velocity = backend_module._graph_velocity(
            pipe, terminal, scheduler, prompt, negative,
            torch.float64, 5.0, index, count,
        )
        terminal = backend_module._graph_step(scheduler, terminal, velocity, index, count)
    return terminal


def test_tiny_split_vjp_matches_direct_end_to_end_with_attached_history():
    direct_source = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    direct_terminal = _tiny_rollout(direct_source)
    direct_loss = (torch.sin(direct_terminal) + direct_terminal.square()).sum()
    direct_gradient, = torch.autograd.grad(direct_loss, direct_source)

    boundary = _tiny_rollout(torch.tensor([0.3], dtype=torch.float64,
                                          requires_grad=True)).detach().requires_grad_(True)
    boundary_loss = (torch.sin(boundary) + boundary.square()).sum()
    cotangent, = torch.autograd.grad(boundary_loss, boundary)
    replay_source = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    replay_terminal = _tiny_rollout(replay_source)
    split_gradient, = torch.autograd.grad(
        replay_terminal, replay_source, grad_outputs=cotangent,
    )
    assert torch.equal(replay_terminal.detach(), boundary.detach())
    assert torch.allclose(split_gradient, direct_gradient, rtol=1e-12, atol=1e-12)


def test_real_unipc_tiny_cpu_split_vjp_matches_direct_gradient():
    """Exercise the installed native UniPC history mutations, without Wan weights."""
    from diffusers import UniPCMultistepScheduler

    def prefix():
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True,
            flow_shift=3.0, lower_order_final=True, final_sigmas_type="zero",
        )
        scheduler.set_timesteps(5)
        state = torch.tensor([[0.8]], dtype=torch.float64)
        with torch.no_grad():
            for index in range(2):
                state = scheduler.step(
                    0.2 * state, scheduler.timesteps[index], state,
                    return_dict=False,
                )[0]
        return state, scheduler

    def rollout(source):
        state, scheduler = prefix()
        state = scheduler.step(
            source, scheduler.timesteps[2], state, return_dict=False,
        )[0]
        assert scheduler.model_outputs[-1].requires_grad
        for index in (3, 4):
            velocity = 0.3 * state + 0.01 * index
            state = scheduler.step(
                velocity, scheduler.timesteps[index], state, return_dict=False,
            )[0]
        return state

    direct_source = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    direct_terminal = rollout(direct_source)
    direct_gradient, = torch.autograd.grad(direct_terminal.square().sum(), direct_source)
    boundary = rollout(torch.tensor([[0.3]], dtype=torch.float64,
                                    requires_grad=True)).detach().requires_grad_(True)
    cotangent, = torch.autograd.grad(boundary.square().sum(), boundary)
    split_source = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    replay_terminal = rollout(split_source)
    split_gradient, = torch.autograd.grad(
        replay_terminal, split_source, grad_outputs=cotangent,
    )
    assert torch.equal(replay_terminal.detach(), boundary.detach())
    assert torch.allclose(split_gradient, direct_gradient, rtol=1e-12, atol=1e-12)


def test_checkpoint_recompute_has_separate_attempted_completed_ledger():
    updates = []
    ledger = ReplayLedger(
        {"transformer_block": 4, "vae_chunk": 1},
        callback=lambda summary: updates.append(copy.deepcopy(summary)),
    )
    ledger.start_transformer_storage()
    source = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    result = checkpoint_call(
        ledger, "transformer_block", lambda value: torch.sin(value).square(), source,
    )
    result.sum().backward()
    counts = ledger.summary()["counts"]["transformer_block"]
    assert counts["forward"] == {"attempted": 1, "completed": 1}
    assert counts["recompute"] == {"attempted": 1, "completed": 1}
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert updates[-1]["counts"] == ledger.summary()["counts"]
    assert ledger.summary()["transformer_boundary_storage"]["disk_peak_bytes"] > 0
    ledger.release_transformer_storage()
    assert ledger.summary()["transformer_boundary_storage"]["closed"]


def test_transformer_boundary_disk_preserves_alias_history_and_exact_gradient(monkeypatch):
    import runtime.wan.gradient_checkpointing as checkpoints

    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(checkpoints, "TRANSFER_BYTES", 7)
    source = torch.arange(24, dtype=torch.float64).reshape(4, 6)

    def rollout(value, ledger=None):
        history = [value * 0.17]
        for index in range(3):
            def block(x, old, step=index):
                a = x[:, 1:5]
                b = x[1:3, 2:6]
                assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
                return torch.sin(a).sum() * x + old * (step + 1) * 0.03
            if ledger is None:
                value = block(value, history[-1])
            else:
                value = checkpoint_call(ledger, "transformer_block", block,
                                        value, history[-1])
            history.append(value)
        return value.square().mean() + history[-2].sin().mean()

    direct = source.detach().requires_grad_(True)
    direct_value = rollout(direct)
    direct_gradient, = torch.autograd.grad(direct_value, direct)
    attached = source.detach().requires_grad_(True)
    ledger = ReplayLedger({"transformer_block": 3, "vae_chunk": 0})
    ledger.start_transformer_storage()
    value = rollout(attached, ledger)
    gradient, = torch.autograd.grad(value, attached)
    assert torch.equal(value, direct_value)
    assert torch.equal(gradient, direct_gradient)
    counts = ledger.summary()["counts"]["transformer_block"]
    assert counts["forward"] == {"attempted": 3, "completed": 3}
    assert counts["recompute"] == {"attempted": 3, "completed": 3}
    receipt = ledger.summary()["transformer_boundary_storage"]
    assert receipt["disk_peak_bytes"] > 0
    assert receipt["cpu_peak_packed_bytes"] <= 7
    # A saved input may be unpacked more than once during native autograd.
    assert receipt["disk_read_bytes"] >= receipt["disk_written_bytes"]
    assert receipt["d2h_bytes"] == receipt["h2d_bytes"] == 0
    path = ledger.transformer_spool.path
    ledger.release_transformer_storage()
    assert ledger.summary()["transformer_boundary_storage"]["closed"]
    assert not path.exists()


def test_transformer_boundary_budget_failure_closes_spool(monkeypatch):
    import runtime.wan.gradient_checkpointing as checkpoints

    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_LIMIT_BYTES", 1)
    ledger = ReplayLedger({"transformer_block": 1, "vae_chunk": 0})
    ledger.start_transformer_storage()
    path = ledger.transformer_spool.path
    source = torch.arange(8, dtype=torch.float64, requires_grad=True)
    try:
        with pytest.raises(RuntimeError, match="TRANSFORMER_BOUNDARY_DISK_BUDGET"):
            checkpoint_call(ledger, "transformer_block", torch.sin, source)
    finally:
        ledger.release_transformer_storage()
    assert not path.exists()
    assert ledger.summary()["transformer_boundary_storage"]["closed"]


def test_disk_boundary_exact_views_alias_and_gradient(monkeypatch):
    import runtime.wan.gradient_checkpointing as checkpoints

    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(checkpoints, "TRANSFER_BYTES", 7)
    spool = BoundarySpool()
    source = torch.arange(24, dtype=torch.float64).reshape(4, 6)
    view_a = source[:, 1:5]
    view_b = source[1:3, 2:6]
    storage = BoundaryStorage(spool)
    packed_a, packed_b = storage.pack(view_a), storage.pack(view_b)
    assert packed_a[0] == packed_b[0]
    assert spool.summary()["files"] == 1
    storage.copies.clear()
    restored_a, restored_b = storage.unpack(packed_a), storage.unpack(packed_b)
    assert torch.equal(restored_a, view_a)
    assert torch.equal(restored_b, view_b)
    assert restored_a.untyped_storage().data_ptr() == restored_b.untyped_storage().data_ptr()
    restored_a[1, 1] = -123.0
    assert restored_b[0, 0].item() == -123.0
    assert spool.summary()["cpu_live_packed_bytes"] == 0
    assert spool.summary()["cpu_peak_packed_bytes"] <= 7
    assert spool.summary()["disk_written_bytes"] == source.untyped_storage().nbytes()
    assert spool.summary()["d2h_bytes"] == 0
    assert spool.summary()["h2d_bytes"] == 0
    spool.close()
    assert list(spool.path.glob("storage-*")) == []
    assert spool.summary()["disk_live_bytes"] == 0

    def frame(value):
        a = value[:, 1:5]
        b = value[1:3, 2:6]
        return (torch.sin(a).sum() + b.square().sum())

    direct = source.detach().requires_grad_(True)
    direct_value = frame(direct)
    direct_grad, = torch.autograd.grad(direct_value, direct)
    checkpointed = source.detach().requires_grad_(True)
    ledger = ReplayLedger({"transformer_block": 0, "vae_chunk": 1})
    ledger.new_decode()
    value = checkpoint_call(ledger, "vae_chunk", frame, checkpointed)
    grad, = torch.autograd.grad(value, checkpointed)
    assert torch.equal(value, direct_value)
    assert torch.equal(grad, direct_grad)
    receipt = ledger.summary()["boundary_storage"]
    assert receipt["disk_peak_bytes"] > 0
    assert receipt["cpu_peak_packed_bytes"] <= 7
    ledger.release_boundary_storage()
    assert ledger.summary()["boundary_storage"]["closed"]


def test_parent_removes_orphan_spool_after_worker_exit(tmp_path, monkeypatch):
    roots = []

    class FinishedWorker:
        returncode = 0

        def __init__(self, command, **kwargs):
            root = Path(kwargs["env"]["RGB_DCT_BOUNDARY_SPOOL_ROOT"])
            roots.append(root)
            (root / "orphan-storage").write_bytes(b"checkpoint")

        def poll(self):
            return 0

    monkeypatch.setattr(trial.subprocess, "Popen", FinishedWorker)
    receipt = trial._run_worker(tmp_path, trial.load_config(), trial.CASE_IDS[0])
    assert receipt["status"] == "COMPLETED"
    assert receipt["boundary_spool_cleanup"] == "COMPLETED"
    assert len(roots) == 1 and not roots[0].exists()


def test_parent_removes_orphan_spool_after_timeout(tmp_path, monkeypatch):
    roots, signals = [], []

    class TimedOutWorker:
        pid = 123456
        returncode = None

        def __init__(self, command, **kwargs):
            root = Path(kwargs["env"]["RGB_DCT_BOUNDARY_SPOOL_ROOT"])
            roots.append(root)
            (root / "orphan-storage").write_bytes(b"partial checkpoint")

        def poll(self):
            return self.returncode

        def wait(self):
            self.returncode = -9

    monkeypatch.setattr(trial.subprocess, "Popen", TimedOutWorker)
    monkeypatch.setattr(trial.os, "killpg", lambda pid, sig: signals.append(sig))
    config = copy.deepcopy(trial.load_config())
    config["resources"]["case_timeout_seconds"] = 0
    config["resources"]["termination_grace_seconds"] = 0
    receipt = trial._run_worker(tmp_path, config, trial.CASE_IDS[0])
    assert receipt["status"] == "TIMEOUT"
    assert receipt["boundary_spool_cleanup"] == "COMPLETED"
    assert signals == [trial.signal.SIGTERM, trial.signal.SIGKILL]
    assert len(roots) == 1 and not roots[0].exists()


def test_disk_boundary_preflight_and_write_failure_cleanup(monkeypatch):
    import runtime.wan.gradient_checkpointing as checkpoints

    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 10**20)
    with pytest.raises(RuntimeError, match="VAE_BOUNDARY_DISK_PREFLIGHT"):
        BoundarySpool()
    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_LIMIT_BYTES", 1)
    spool = BoundarySpool()
    with pytest.raises(RuntimeError, match="VAE_BOUNDARY_DISK_BUDGET"):
        BoundaryStorage(spool).pack(torch.arange(8, dtype=torch.float64))
    assert spool.summary()["files"] == 0
    assert spool.summary()["disk_live_bytes"] == 0
    spool.close()

    monkeypatch.setattr(checkpoints, "DISK_LIMIT_BYTES", 1024)
    spool = BoundarySpool()
    original = spool._cpu_live

    def fail_transfer(amount):
        if amount:
            raise IOError("injected transfer failure")
        original(amount)

    monkeypatch.setattr(spool, "_cpu_live", fail_transfer)
    with pytest.raises(IOError, match="injected transfer failure"):
        BoundaryStorage(spool).pack(torch.arange(8, dtype=torch.float64))
    assert list(spool.path.glob("storage-*")) == []
    assert spool.summary()["disk_live_bytes"] == 0
    spool.close()


def test_complete_fake_run_has_six_scored_slots_and_exact_call_ledger(tmp_path, monkeypatch):
    config = trial.load_config()
    store, backends, reads = _fake_run(tmp_path, monkeypatch)
    assert len(reads) == 6 and store.data["scored_frames"] == 1086
    assert store.data["status"] == "FIXED_TERMINAL_GRADIENT_COMPLETE"
    for kind, maximum in trial.PLAN.items():
        assert store.data["calls"][kind] == {"attempted": maximum, "completed": maximum}
    for case_id, backend in zip(trial.CASE_IDS, backends, strict=True):
        case = store.case(case_id)
        assert case["slots"]["OFF"]["decision"] == "H0"
        assert case["slots"]["SINGLE49"]["decision"] == "H1"
        assert case["slots"]["TERMINAL46"]["decision"] == "H1"
        assert case["proxy"]["proxy_validation"]["matched"]
        assert case["recompute_ledger"] is not None
        assert case["worker_receipt"]["status"] == "COMPLETED"
        assert case["release_receipt"]["status"] == "COMPLETED"
        assert backend.events.index("COTANGENT") < backend.events.index("VJP")
        assert backend.events.index("VJP") < backend.events.index("CONTROL46")
        assert all(case["slots"][arm]["quality_vs_off"]["compared_frames"] == 181
                   for arm in trial.ARMS[1:])
        for arm in trial.ARMS:
            layers = case["slots"][arm]["layers"]
            assert list(layers) == config["observation_layers"]
            assert all(layer["status"] == "SCORED"
                       and len(layer["group_scores"]) == 30
                       and isinstance(layer["positive_groups"], int)
                       for layer in layers.values())
            assert layers["float_rgb"]["decision"] is None
            assert layers["rgb8_quantized"]["decision"] is None
            assert layers["mp4_rgb24"]["decision"] in ("H0", "H1")
        single = case["controls"]["SINGLE49"]
        assert all(key in single for key in (
            "terminal_fingerprint", "final_history_fingerprint", "terminal_from_off"
        ))


def test_vjp_failure_and_invalid_mp4_preserve_denominator_and_next_source(tmp_path, monkeypatch):
    invalid = (trial.CASE_IDS[0], "OFF")
    store, _, reads = _fake_run(tmp_path, monkeypatch, fail_vjp=True, invalid=invalid)
    first = store.case(trial.CASE_IDS[0])
    assert first["slots"]["OFF"]["status"] == "INVALID"
    assert first["slots"]["TERMINAL46"]["status"] == "ENGINEERING_INVALID"
    assert first["slots"]["TERMINAL46"]["decision"] is None
    assert first["slots"]["SINGLE49"]["status"] == "SCORED"
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert store.data["fixed_denominator"]["mp4_score_slots"] == 6
    assert store.data["status"] == "INCOMPLETE"
    assert len(reads) == 5


def test_gradient_backend_source_does_not_use_detached_trajectory_helpers():
    replay_source = inspect.getsource(backend_module.WanTerminalGradientBackend.terminal_vjp)
    cotangent_source = inspect.getsource(backend_module.WanTerminalGradientBackend.terminal_cotangent)
    assert "trajectory.velocity(" not in replay_source
    assert "trajectory.native_step(" not in replay_source
    assert "checkpoint_decode(" in cotangent_source
    assert "inference_mode" not in cotangent_source
    assert "autograd.grad" in replay_source and "autograd.grad" in cotangent_source


def test_complete_six_slot_misclassification_is_preserved_negative(tmp_path, monkeypatch):
    store, _, _ = _fake_run(tmp_path, monkeypatch, negative=True)
    assert store.data["scored_media_slots"] == 6
    assert all(store.case(case_id)["status"] == "SCORED"
               for case_id in trial.CASE_IDS)
    assert store.data["status"] == "FIXED_TERMINAL_GRADIENT_NEGATIVE"
    errors = store.data["reporting"]["observed_misclassifications"]
    assert [(row["case_id"], row["arm"]) for row in errors] == [
        (trial.CASE_IDS[0], "TERMINAL46"),
        (trial.CASE_IDS[1], "TERMINAL46"),
    ]


def test_cotangent_failure_marks_single49_dependency_not_run(tmp_path, monkeypatch):
    store, _, reads = _fake_run(tmp_path, monkeypatch, fail_cotangent=True)
    first = store.case(trial.CASE_IDS[0])
    assert first["slots"]["OFF"]["status"] == "ENGINEERING_INVALID"
    assert first["slots"]["TERMINAL46"]["status"] == "ENGINEERING_INVALID"
    single = first["slots"]["SINGLE49"]
    assert single["status"] == "NOT_RUN_ENGINEERING_INVALID"
    assert single["reason"].startswith("DEPENDENCY_SHARED_OFF_COTANGENT_FAILED:")
    assert single["attempted"] is False and single["decision"] is None
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert len(reads) == 3 and store.data["status"] == "INCOMPLETE"


def test_shared_off_failure_marks_single49_dependency_not_run(tmp_path, monkeypatch):
    store, _, reads = _fake_run(tmp_path, monkeypatch, fail_prepare=True)
    first = store.case(trial.CASE_IDS[0])
    single = first["slots"]["SINGLE49"]
    assert single["status"] == "NOT_RUN_ENGINEERING_INVALID"
    assert single["reason"].startswith("DEPENDENCY_SHARED_OFF_FAILED:")
    assert single["attempted"] is False and single["decision"] is None
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert len(reads) == 3 and store.data["status"] == "INCOMPLETE"


@pytest.mark.parametrize("which", ["single", "terminal"])
def test_zero_response_without_mp4_is_engineering_invalid(tmp_path, monkeypatch, which):
    store, _, reads = _fake_run(
        tmp_path, monkeypatch,
        zero_single=(which == "single"), zero_terminal=(which == "terminal"),
    )
    arm = "SINGLE49" if which == "single" else "TERMINAL46"
    slot = store.case(trial.CASE_IDS[0])["slots"][arm]
    assert slot["status"] == "ENGINEERING_INVALID"
    assert slot["reason"] == "FINITE_ZERO_NATIVE_RESPONSE_NO_MP4"
    assert slot["decision"] is None
    assert (trial.CASE_IDS[0], arm) not in reads
    assert store.data["status"] == "INCOMPLETE"


def test_zero_lift_without_mp4_is_engineering_invalid(tmp_path, monkeypatch):
    store, _, reads = _fake_run(tmp_path, monkeypatch, zero_lift=True)
    slot = store.case(trial.CASE_IDS[0])["slots"]["SINGLE49"]
    assert slot["status"] == "ENGINEERING_INVALID"
    assert slot["reason"] == "FINITE_ZERO_LIFT_DIRECTION_NO_MP4"
    assert slot["decision"] is None
    assert (trial.CASE_IDS[0], "SINGLE49") not in reads
    assert store.data["status"] == "INCOMPLETE"


@pytest.mark.parametrize("failure", ["resource", "release"])
def test_resource_or_release_failure_forces_incomplete(tmp_path, monkeypatch, failure):
    store, _, _ = _fake_run(
        tmp_path, monkeypatch,
        fail_resource=(failure == "resource"), fail_release=(failure == "release"),
    )
    first = store.case(trial.CASE_IDS[0])
    assert all(slot["status"] == "SCORED" for slot in first["slots"].values())
    assert first["status"] != "SCORED"
    assert store.data["status"] == "INCOMPLETE"
    if failure == "resource":
        assert first["resources"]["receipt_valid"] is False
    else:
        assert first["release_receipt"]["status"] == "FAILED"
        assert any(row["stage"] == "RELEASE" for row in first["failures"])


def test_parent_persists_denominator_before_worker_and_timeout_receipt(tmp_path):
    config = trial.load_config()
    output = tmp_path / "parent"
    output.mkdir()
    observed = []
    def timeout_worker(output, config, case_id):
        store = trial.Store.open(output / "result.json")
        observed.append((case_id, store.data["pending_media_slots"],
                         sum(len(case["slots"]) for case in store.data["cases"].values())))
        return {
            "status": "TIMEOUT", "error": "WORKER_TIMEOUT", "exit_code": None,
            "elapsed_seconds": 14410.0, "parent_observed_peak_rss_kib": 123456,
            "termination": "SIGTERM_THEN_SIGKILL",
        }
    trial._supervise(output, config, "a" * 40, worker_fn=timeout_worker)
    store = trial.Store.open(output / "result.json")
    assert observed == [(trial.CASE_IDS[0], 6, 6), (trial.CASE_IDS[1], 3, 6)]
    for case_id in trial.CASE_IDS:
        case = store.case(case_id)
        assert case["worker_receipt"]["status"] == "TIMEOUT"
        assert case["worker_receipt"]["cuda_peak"]["status"] == "UNAVAILABLE"
        assert case["resources"]["cuda"]["status"] == "UNAVAILABLE"
        assert "WORKER_TIMEOUT" in case["resources"]["cuda"]["reason"]
    source = inspect.getsource(trial._run_worker)
    assert source.index("signal.SIGTERM") < source.index("signal.SIGKILL")
    assert store.data["status"] == "INCOMPLETE"


def test_child_environment_failure_preserves_original_reason(tmp_path, monkeypatch):
    config = trial.load_config()
    result_path = tmp_path / "result.json"
    store = trial.Store(result_path, trial.initial_result(config, tmp_path, "a" * 40))
    store.save()
    monkeypatch.setattr(
        trial, "environment_receipt",
        lambda config: (_ for _ in ()).throw(RuntimeError("fake CUDA API unavailable")),
    )
    assert trial._record_child_environment(store, trial.CASE_IDS[0], config) is False
    case = store.case(trial.CASE_IDS[0])
    assert case["environment_receipt"]["status"] == "FAILED"
    assert case["environment_receipt"]["stage"] == "ENVIRONMENT_CHECK"
    assert "fake CUDA API unavailable" in case["environment_receipt"]["reason"]
    assert all(slot["status"] == "NOT_RUN_ENGINEERING_INVALID"
               and slot["reason"].startswith("ENVIRONMENT_CHECK:")
               for slot in case["slots"].values())
    assert case["resources"]["cuda"]["status"] == "UNAVAILABLE"


def test_invalid_in_memory_layer_is_persisted_before_media(tmp_path, monkeypatch):
    config = trial.load_config()
    output = tmp_path / "layers"
    output.mkdir()
    store = trial.Store(output / "result.json",
                        trial.initial_result(config, output, "a" * 40))
    store.active_case = trial.CASE_IDS[0]
    store.case(trial.CASE_IDS[0])["environment_receipt"] = {"status": "VALID"}
    store.save()
    invalid = {
        "status": "INVALID", "reason": "FAKE_FLOAT_LAYER_INVALID", "score": None,
        "group_scores": None, "positive_groups": None, "frames_used": 0,
        "decision": None,
    }
    monkeypatch.setattr(trial, "_score_memory_layer", lambda rgb, key: invalid.copy())
    encoded = []
    trial._save_and_score(
        store, trial.CASE_IDS[0], "OFF",
        np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE),
        config["key_utf8"].encode(), config,
        encode_fn=lambda *args: encoded.append(args),
        score_fn=lambda *args: pytest.fail("MP4 receiver must not run"),
    )
    slot = store.case(trial.CASE_IDS[0])["slots"]["OFF"]
    assert slot["status"] == "ENGINEERING_INVALID"
    assert slot["layers"]["float_rgb"]["status"] == "INVALID"
    assert slot["layers"]["float_rgb"]["reason"] == "FAKE_FLOAT_LAYER_INVALID"
    assert slot["layers"]["float_rgb"]["decision"] is None
    assert slot["layers"]["rgb8_quantized"]["status"] == "PENDING"
    assert encoded == []
