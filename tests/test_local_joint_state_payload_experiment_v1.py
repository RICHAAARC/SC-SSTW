"""CPU-only runner/lifecycle checks.  No model, VAE, codec, or media execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from runtime.wan import local_joint_state_payload_experiment_v1 as runtime
from experiments.wan_state_clock import local_joint_state_payload_v1_run as runner


pytestmark = pytest.mark.unit


def config() -> dict:
    return {
        "schema": runtime.SCHEMA,
        "model": {"id": "explicit/model", "revision": "immutable-revision"},
        "generation": {
            "height": 320, "width": 512, "frames": 181, "steps": 50,
            "guidance_scale": 5.0, "max_sequence_length": 512,
            "prompt": "explicit prompt", "negative_prompt": "explicit negative", "seed": 7,
        },
        "carrier": {"key": "raw-鍵", "message_hex": "8001a55a", "rho": 0.37, "cap": 0.83},
        "media": {"fps": 8, "codec": "configured-codec", "crf": 21, "pixel_format": "configured-pixfmt"},
        "source": {"source_id": "development-source", "development_only": True},
        "arms": ["OFF", "JOINT"],
        "runtime": {"device": "cuda", "transformer_dtype": "bfloat16"},
    }


def test_config_requires_every_scientific_and_media_value_without_defaults():
    value = config(); runtime.validate_config(value)
    for section, key in (("carrier", "rho"), ("carrier", "cap"), ("generation", "seed"),
                         ("media", "codec"), ("source", "source_id")):
        broken = copy.deepcopy(value); del broken[section][key]
        with pytest.raises(ValueError):
            runtime.validate_config(broken)
    broken = copy.deepcopy(value); broken["arms"] = ["JOINT", "PAYLOAD_ONLY"]
    with pytest.raises(ValueError, match="implemented arms"):
        runtime.validate_config(broken)


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        return (hidden_states * 0.01 + encoder_hidden_states.mean(),)


class AttrDict(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError as exc: raise AttributeError(name) from exc


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])


class TinyScheduler:
    def __init__(self):
        self.step_index = None
        self.timesteps = torch.arange(50, 0, -1, dtype=torch.float32)
        self.sigmas = torch.cat((torch.linspace(1.0, .02, 50), torch.zeros(1)))
        self.config = AttrDict(prediction_type="flow_prediction", thresholding=False,
                               predict_x0=True, lower_order_final=True)
        self.predict_x0 = True
    def step(self, velocity, timestep, state, return_dict=False):
        self.step_index = 1 if self.step_index is None else self.step_index + 1
        return (state - 0.01 * velocity,)


class FakeLoaders:
    def __init__(self):
        self.generation_args = []; self.vae_devices = []

    def prepare_generation(self, cfg, **kwargs):
        self.generation_args.append(kwargs)
        pipe = SimpleNamespace(transformer=TinyTransformer(), scheduler=TinyScheduler(), text_encoder=None, vae=None)
        return pipe, torch.zeros(1, 1, 2, 2, 2), torch.ones(1), -torch.ones(1), torch.float32

    def load_frozen_vae(self, cfg, *, device):
        self.vae_devices.append(str(device)); return TinyVAE().to(device)


def test_serial_residency_uses_load_vae_false_and_preserves_active_scheduler(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = config(); cfg["runtime"] = {"device": "cpu", "transformer_dtype": "float32"}
    # Validation intentionally forbids CPU in production; change only after validating the public contract.
    runtime.validate_config(config())
    cfg["runtime"]["device"] = "cuda"
    events, loaders = [], FakeLoaders()
    residency = runtime.WanSerialResidency.__new__(runtime.WanSerialResidency)
    residency.config, residency.event, residency.device, residency.dtype = cfg, lambda n, r: events.append((n, r)), torch.device("cpu"), torch.float32
    residency.loaders, residency.pipe, residency.initial, residency.prompt, residency.negative = loaders, None, None, None, None
    residency.vae = residency.backend = residency.active_scheduler = None; residency.phase = "EMPTY"; residency.phase_identity = None
    residency.load_generation()
    active = residency.pristine_scheduler(); residency.active_scheduler = active
    before = runtime.trajectory.fingerprint(vars(active))
    residency.enter_vae("enter"); residency.leave_vae("leave")
    assert loaders.generation_args == [{"load_vae": False, "device": torch.device("cpu"), "model_dtype": torch.float32}]
    assert loaders.vae_devices == ["cpu"] and runtime.trajectory.fingerprint(vars(active)) == before
    assert [row[1]["status"] for row in events] == ["ATTEMPTED", "COMPLETED", "ATTEMPTED", "COMPLETED", "ATTEMPTED", "COMPLETED"]
    assert events[-1][1]["identity_verified"] is True


def test_off_arm_uses_real_fifty_step_injection_path_without_vae():
    class Residency:
        pipe = SimpleNamespace(transformer=TinyTransformer())
        initial = torch.zeros(1, 1, 2, 2, 2)
        prompt = torch.tensor([.2]); negative = torch.tensor([-.1]); dtype = torch.float32
        active_scheduler = None
        def pristine_scheduler(self): return TinyScheduler()
    rows, counts = [], {}
    def count(kind, completed):
        pair = counts.setdefault(kind, [0, 0]); pair[int(completed)] += 1
    terminal, receipt, provider = runtime.run_arm(Residency(), config(), "OFF", count=count, record_step=rows.append)
    assert provider is None and terminal.shape == (1, 1, 2, 2, 2)
    assert len(rows) == 50 and [row["cursor_after"] for row in rows] == list(range(1, 51))
    assert all(row["composition"] == "explicit_zero_joint_control" for row in rows)
    assert receipt["final_cursor"] == 50 and counts["injected_joint_control"] == [50, 50]


def test_resident_control_switches_only_enabled_steps_and_always_restores(monkeypatch):
    calls = []
    class Residency:
        backend = object()
        device = torch.device("cpu")
        def enter_vae(self, label): calls.append(("enter", label))
        def leave_vae(self, label): calls.append(("leave", label))
        def event(self, label, row): calls.append(("event", label, row["status"]))
    class Provider:
        def __init__(self, **kwargs): self.kwargs = kwargs; self.calls = {}; self.failures = []
    monkeypatch.setattr(runtime.joint, "LocalJointPosteriorProvider", Provider)
    from runtime.wan import local_joint_state_payload_v1 as adapter
    monkeypatch.setattr(adapter, "make_control_step", lambda **kwargs: lambda **request: (request["z"], {"index": request["index"]}))
    control, _ = runtime.make_resident_control(Residency(), config())
    value = torch.zeros(1)
    control(z=value, conditional=value, unconditional=value, sigma=1.0, index=24, total_steps=50)
    control(z=value, conditional=value, unconditional=value, sigma=1.0, index=25, total_steps=50)
    assert [row for row in calls if row[0] != "event"] == [
        ("enter", "joint_step_25_enter_vae"), ("leave", "joint_step_25_leave_vae")]


def test_provider_failure_keeps_primary_exception_and_does_not_restore(monkeypatch):
    calls = []
    class Residency:
        backend = object()
        device = torch.device("cpu")
        def enter_vae(self, label): calls.append(("enter", label))
        def leave_vae(self, label): calls.append(("leave", label))
        def event(self, label, row): calls.append(("event", label, row["reason"]))
        def stop_in_vae_phase(self, label, *, primary_reason): calls.append(("stop", label, primary_reason))
    class Provider:
        def __init__(self, **kwargs): self.calls = {"decode_attempted": 1, "decode_completed": 0}; self.failures = [{"stage": "decode_clean"}]
    monkeypatch.setattr(runtime.joint, "LocalJointPosteriorProvider", Provider)
    from runtime.wan import local_joint_state_payload_v1 as adapter
    def factory(**kwargs):
        def fail(**request): raise MemoryError("fixture-provider-oom")
        return fail
    monkeypatch.setattr(adapter, "make_control_step", factory)
    control, _ = runtime.make_resident_control(Residency(), config())
    value = torch.zeros(1)
    with pytest.raises(MemoryError, match="fixture-provider-oom"):
        control(z=value, conditional=value, unconditional=value, sigma=1.0, index=25, total_steps=50)
    assert not any(row[0] == "leave" for row in calls)
    assert calls[-1][0] == "stop" and "MemoryError: fixture-provider-oom" in calls[-1][2]


def test_codec_consumes_exact_rgb8_bytes_and_returns_fixed_readback(monkeypatch, tmp_path):
    small = carrier.CarrierProtocol(video_shape=(8, 8, 32, 3), segment_start=0, segment_frames=8,
                                    segment_count=1, rois=((0, 8, 0, 8), (0, 8, 8, 16), (0, 8, 16, 24), (0, 8, 24, 32)))
    monkeypatch.setattr(carrier, "PUBLIC", small)
    pixels = torch.arange(8 * 8 * 32 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(small.video_shape)
    expected = pixels.numpy().tobytes(); invocations = []
    class Result:
        returncode = 0; stderr = b""
        def __init__(self, stdout=b""): self.stdout = stdout
    def invoke(command, **kwargs):
        invocations.append((command, kwargs.get("input")))
        if "pipe:0" in command:
            assert kwargs["input"] == expected
            Path(command[-1]).write_bytes(b"mp4-fixture")
            return Result()
        return Result(expected)
    monkeypatch.setattr(runtime.subprocess, "run", invoke)
    rows = []
    received, receipt = runtime.ExplicitFFmpeg(config()["media"]).roundtrip(pixels, tmp_path / "x.mp4", event=rows.append)
    assert torch.equal(received, pixels) and receipt["input_is_exact_persisted_rgb8"] is True
    assert len(invocations) == 2 and rows[0]["status"] == "ATTEMPTED" and rows[-1]["status"] == "SAVED"


def test_no_git_preflight_subprocess_preserves_fixed_rows(tmp_path):
    root = tmp_path / "release"
    for name in runner.SOURCE_CLOSURE:
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(runner.ROOT / name, target)
    cfg = config(); cfg_path = root / "explicit.json"; cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    output = root / "out"
    child = subprocess.run([
        "/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python", "-B", "-m",
        "experiments.wan_state_clock.local_joint_state_payload_v1_run", "--config", str(cfg_path),
        "--output", str(output), "--preflight-only",
    ], cwd=root, env={**__import__("os").environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "PREFLIGHT_COMPLETE" and result["source_identity"]["git_commit"] is None
    assert all(len(row["steps"]) == 50 for row in result["arms"].values())
    assert all(set(row["layers"]) == set(runtime.LAYERS) for row in result["arms"].values())
