"""CPU-only runner/lifecycle checks.  No model, VAE, codec, or media execution."""
from __future__ import annotations

import copy
import hashlib
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
        "carrier": {"key": "raw-鍵", "wrong_key": "raw-鍵-wrong",
                    "message_hex": "8001a55a", "rho": 0.37, "cap": 0.83},
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


def test_provider_primary_survives_stop_and_event_cleanup_failures(monkeypatch):
    class Residency:
        backend = object(); device = torch.device("cpu")
        def enter_vae(self, label): pass
        def leave_vae(self, label): raise AssertionError("must not restore")
        def event(self, label, row): raise OSError("fixture-event-write")
        def stop_in_vae_phase(self, label, *, primary_reason): raise RuntimeError("fixture-stop")
    class Provider:
        def __init__(self, **kwargs): self.calls = {}; self.failures = []
    monkeypatch.setattr(runtime.joint, "LocalJointPosteriorProvider", Provider)
    from runtime.wan import local_joint_state_payload_v1 as adapter
    monkeypatch.setattr(adapter, "make_control_step", lambda **kwargs: lambda **request: (_ for _ in ()).throw(MemoryError("primary-oom")))
    control, _ = runtime.make_resident_control(Residency(), config())
    with pytest.raises(MemoryError, match="primary-oom"):
        control(z=torch.zeros(1), conditional=torch.zeros(1), unconditional=torch.zeros(1), sigma=1.0, index=25, total_steps=50)


def test_codec_consumes_exact_rgb8_bytes_and_returns_fixed_readback(monkeypatch, tmp_path):
    small = carrier.CarrierProtocol(video_shape=(8, 8, 32, 3), segment_start=0, segment_frames=8,
                                    segment_count=1, rois=((0, 8, 0, 8), (0, 8, 8, 16), (0, 8, 16, 24), (0, 8, 24, 32)))
    monkeypatch.setattr(carrier, "PUBLIC", small)
    pixels = torch.arange(8 * 8 * 32 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(small.video_shape)
    expected = pixels.numpy().tobytes(); invocations = []
    raster = tmp_path / "pixels.rgb"; raster.write_bytes(expected); expected_sha = hashlib.sha256(expected).hexdigest()
    def reopen(path, sha):
        raw = Path(path).read_bytes()
        assert sha == expected_sha and hashlib.sha256(raw).hexdigest() == sha
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(small.video_shape)
    class Result:
        returncode = 0; stderr = b""
        def __init__(self, stdout=b""): self.stdout = stdout
    def invoke(command, **kwargs):
        invocations.append((command, kwargs.get("input")))
        if "pipe:0" in command:
            assert kwargs["input"] == expected
            Path(command[-1]).write_bytes(b"mp4-fixture")
            return Result()
        if command[0] == "ffprobe":
            return Result(json.dumps({"streams": [{"width": 32, "height": 8, "nb_frames": "8"}]}).encode())
        return Result(expected)
    monkeypatch.setattr(runtime.subprocess, "run", invoke)
    rows = []
    received, receipt = runtime.ExplicitFFmpeg(config()["media"], raster_loader=reopen).roundtrip(
        raster, expected_sha, tmp_path / "x.mp4", event=rows.append)
    assert torch.equal(received, pixels) and receipt["input_actual_sha256"] == expected_sha
    assert len(invocations) == 3
    assert [(row["stage"], row["status"]) for row in rows] == [
        ("encode", "ATTEMPTED"), ("encode", "COMPLETED"),
        ("probe", "ATTEMPTED"), ("probe", "COMPLETED"),
        ("read", "ATTEMPTED"), ("read", "COMPLETED")]


def test_codec_wrong_width_height_preserves_encoded_artifact(monkeypatch, tmp_path):
    small = carrier.CarrierProtocol(video_shape=(8, 8, 32, 3), segment_start=0, segment_frames=8,
                                    segment_count=1, rois=((0, 8, 0, 8), (0, 8, 8, 16), (0, 8, 16, 24), (0, 8, 24, 32)))
    monkeypatch.setattr(carrier, "PUBLIC", small)
    pixels = torch.zeros(small.video_shape, dtype=torch.uint8); raw = pixels.numpy().tobytes()
    raster = tmp_path / "pixels.rgb"; raster.write_bytes(raw); sha = hashlib.sha256(raw).hexdigest()
    def invoke(command, **kwargs):
        if "pipe:0" in command:
            Path(command[-1]).write_bytes(b"encoded"); return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{"width": 16, "height": 16, "nb_frames": "8"}]}).encode(), stderr=b"")
    monkeypatch.setattr(runtime.subprocess, "run", invoke)
    rows = []
    with pytest.raises(RuntimeError, match="wrong geometry"):
        runtime.ExplicitFFmpeg(config()["media"], raster_loader=lambda p, s: pixels).roundtrip(
            raster, sha, tmp_path / "wrong.mp4", event=rows.append)
    encoded = [row for row in rows if row["stage"] == "encode" and row["status"] == "COMPLETED"][0]
    assert encoded["artifact"]["sha256"] == hashlib.sha256(b"encoded").hexdigest()
    assert rows[-1]["stage"] == "probe" and rows[-1]["status"] == "FAILED"


CATALOG_PROTOCOL = carrier.CarrierProtocol(
    video_shape=(181, 8, 32, 3), segment_start=1, segment_frames=8, segment_count=22,
    rois=((0, 8, 0, 8), (0, 8, 8, 16), (0, 8, 16, 24), (0, 8, 24, 32)),
)


def test_raw_observations_serializes_all_768_public_catalog_rows():
    rgb = torch.zeros(1, 8, 32, 3, dtype=torch.float32).expand(181, -1, -1, -1)
    rows = runtime.raw_observations(rgb, "raw-key", public_protocol=CATALOG_PROTOCOL)
    assert len(rows) == 8 * 24 * 4
    assert all(set(row) >= {"spec", "state_chips", "payload_chips"} for row in rows)


def test_dual_key_raw_seal_keeps_correct_when_wrong_key_extraction_fails(monkeypatch, tmp_path):
    cfg = _single_off_config()
    store = runner.Store(tmp_path / "dual-key", cfg, Path("<fixture>"), "dependency_injected_cpu_fixture")
    original = runtime.raw_observations
    rgb = torch.zeros(1, 8, 32, 3, dtype=torch.float32).expand(181, -1, -1, -1)
    def extract(value, key, **kwargs):
        if key == cfg["carrier"]["wrong_key"]:
            raise RuntimeError("fixture-wrong-key-only")
        return original(value, key, **kwargs)
    monkeypatch.setattr(runtime, "raw_observations", extract)
    runner._observe_layer(store, "OFF", "float_rgb", rgb, cfg, store.output, CATALOG_PROTOCOL)
    correct = store.data["arms"]["OFF"]["observations"]["float_rgb/CORRECT"]
    wrong = store.data["arms"]["OFF"]["observations"]["float_rgb/WRONG"]
    assert correct["status"] == "SAVED" and correct["rows"] == 768 and Path(correct["path"]).is_file()
    assert wrong["status"] == "FAILED" and "fixture-wrong-key-only" in wrong["reason"]


class FakeExperimentResidency:
    decoded = torch.full(CATALOG_PROTOCOL.video_shape, 0.25, dtype=torch.float32)
    fail_stop = False
    fail_release = False
    def __init__(self, cfg, event):
        self.event = event; self.device = torch.device("cpu"); self.dtype = torch.float32
        self.pipe = SimpleNamespace(transformer=TinyTransformer(), scheduler=TinyScheduler(), text_encoder=None, vae=None)
        self.initial = torch.zeros(1, 1, 2, 2, 2); self.prompt = torch.tensor([.2]); self.negative = torch.tensor([-.1])
        self.active_scheduler = None; self.backend = None; self.phase = "TRANSFORMER"
    def load_generation(self):
        self.event("generation_load", {"status": "COMPLETED", "fixture_dependency_injection": True})
        return {"status": "COMPLETED"}
    def _identity(self): return {"initial": runtime.trajectory.fingerprint(self.initial)}
    def pristine_scheduler(self): return TinyScheduler()
    def enter_vae(self, label):
        self.backend = SimpleNamespace(decode_normalized=lambda value: self.decoded.clone()); self.phase = "VAE"
        self.event(label, {"status": "COMPLETED", "fixture_dependency_injection": True})
    def leave_vae(self, label):
        self.backend = None; self.phase = "TRANSFORMER"; self.event(label, {"status": "COMPLETED"})
    def stop_in_vae_phase(self, label, *, primary_reason=None):
        if self.fail_stop: raise RuntimeError("fixture-stop-cleanup")
        self.backend = None; self.phase = "STOPPED"; self.event(label, {"status": "COMPLETED"})
    def release(self):
        if self.fail_release: raise RuntimeError("fixture-release-cleanup")
        self.phase = "RELEASED"


def small_raster_saver(rgb8, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); raw = rgb8.contiguous().numpy().tobytes(); path.write_bytes(raw)
    return {"status": "SAVED", "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "shape": list(rgb8.shape), "dtype": "uint8"}


class FakeExperimentCodec:
    def __init__(self, media): self.media = media
    def roundtrip(self, raster_path, expected_sha256, mp4_path, *, event):
        raw = Path(raster_path).read_bytes(); assert hashlib.sha256(raw).hexdigest() == expected_sha256
        artifact = {"path": str(mp4_path), "sha256": hashlib.sha256(b"fixture-mp4").hexdigest(), "bytes": 11}
        Path(mp4_path).write_bytes(b"fixture-mp4")
        for stage in ("encode", "probe", "read"):
            event({"stage": stage, "status": "ATTEMPTED", "fixture_dependency_injection": True})
            row = {"stage": stage, "status": "COMPLETED", "fixture_dependency_injection": True}
            if stage == "encode": row["artifact"] = artifact
            event(row)
        received = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(CATALOG_PROTOCOL.video_shape)
        return received, {"status": "SAVED", "artifact": artifact, "fixture_dependency_injection": True}


def _single_off_config():
    value = config(); value["arms"] = ["OFF"]; return value


def test_runner_nonpreflight_fake_success_uses_real_50_steps_and_three_768_catalogs(tmp_path):
    store = runner.run(_single_off_config(), tmp_path / "success",
                       residency_type=FakeExperimentResidency, codec_type=FakeExperimentCodec,
                       public_protocol=CATALOG_PROTOCOL, raster_saver=small_raster_saver,
                       execution_kind="dependency_injected_cpu_fixture")
    arm = store.data["arms"]["OFF"]
    assert store.data["status"] == "COMPLETE" and store.data["actual_model_calls"] is False
    assert store.data["execution"] == dict(kind="dependency_injected_cpu_fixture", attempted=True,
                                           completed=True, scientific_interpretation=False,
                                           blind_recovery=False, fpr_evidence=False)
    assert len(arm["steps"]) == 50 and all(row["status"] == "COMPLETED" for row in arm["steps"])
    assert all(arm["layers"][name]["status"] == "SAVED" for name in runtime.LAYERS)
    assert all(arm["observations"][f"{layer}/{label}"]["rows"] == 768
               for layer in ("float_rgb", "rgb8", "mp4") for label in runner.RAW_KEY_LABELS)
    manifest = json.loads((tmp_path / "success" / "raw_observation_manifest.json").read_text())
    assert manifest["truth_loaded"] is False and set(manifest["arms"]["OFF"]) == set(arm["observations"])
    assert all("key" not in receipt for receipt in manifest["arms"]["OFF"].values())


def test_runner_terminal_nonfinite_keeps_primary_and_records_cleanup_failures(tmp_path):
    class BadResidency(FakeExperimentResidency):
        decoded = torch.full(CATALOG_PROTOCOL.video_shape, float("nan"), dtype=torch.float32)
        fail_stop = True
        fail_release = True
    output = tmp_path / "failure"
    with pytest.raises(FloatingPointError, match="nonfinite"):
        runner.run(_single_off_config(), output,
                   residency_type=BadResidency, codec_type=FakeExperimentCodec,
                   public_protocol=CATALOG_PROTOCOL, raster_saver=small_raster_saver,
                   execution_kind="dependency_injected_cpu_fixture")
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "FAILED" and result["arms"]["OFF"]["layers"]["float_rgb"]["status"] == "FAILED"
    assert result["arms"]["OFF"]["layers"]["rgb8"]["status"] == "MISSING_DEPENDENCY"
    reasons = [row["reason"] for row in result["failures"]]
    assert any("nonfinite" in reason for reason in reasons) and any("fixture-release-cleanup" in reason for reason in reasons)
    assert any(row["event"] == "terminal_failure_cleanup" and row["status"] == "FAILED" for row in result["lifecycle"])


def test_runner_success_then_release_failure_becomes_nonzero_failed_result(monkeypatch, tmp_path):
    class ReleaseFailResidency(FakeExperimentResidency):
        fail_release = True
    monkeypatch.setattr(runtime, "raw_observations", lambda *args, **kwargs: [])
    output = tmp_path / "release-failure"
    with pytest.raises(RuntimeError, match="fixture-release-cleanup"):
        runner.run(_single_off_config(), output,
                   residency_type=ReleaseFailResidency, codec_type=FakeExperimentCodec,
                   public_protocol=CATALOG_PROTOCOL, raster_saver=small_raster_saver,
                   execution_kind="dependency_injected_cpu_fixture")
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "FAILED" and result["stage"] == "RELEASE_FAILED"
    assert result["execution"]["completed"] is False
    assert any(row["stage"] == "residency_release" for row in result["failures"])


def test_runner_residency_constructor_failure_is_sealed_without_model_attempt_or_release(tmp_path):
    original = RuntimeError("fixture-residency-constructor")
    class ConstructorFailure:
        def __init__(self, cfg, event):
            raise original
    output = tmp_path / "constructor-failure"
    with pytest.raises(RuntimeError, match="fixture-residency-constructor") as caught:
        runner.run(config(), output, residency_type=ConstructorFailure,
                   execution_kind="dependency_injected_cpu_fixture")
    assert caught.value is original
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "FAILED" and result["failed_stage"] == "RESIDENCY_CONSTRUCTION"
    assert result["actual_model_calls"] is False and result["calls"] == {}
    assert result["execution"]["attempted"] is False and result["execution"]["completed"] is False
    constructor = [row for row in result["failures"] if row["stage"] == "residency_construction"]
    assert len(constructor) == 1 and constructor[0]["reason"] == "RuntimeError: fixture-residency-constructor"
    assert not any(row["stage"] == "residency_release" for row in result["failures"])
    assert set(result["arms"]) == {"OFF", "JOINT"}
    for arm in result["arms"].values():
        assert arm["status"] == "NOT_RUN"
        assert [(row["index"], row["status"]) for row in arm["steps"]] == [
            (index, "NOT_COMPLETED") for index in range(50)]
        assert set(arm["layers"]) == set(runtime.LAYERS)
        assert all(row["status"] == "MISSING_DEPENDENCY" for row in arm["layers"].values())
        assert set(arm["observations"]) == {
            f"{layer}/{label}" for layer in ("float_rgb", "rgb8", "mp4") for label in runner.RAW_KEY_LABELS}
        assert all(row["status"] == "MISSING_DEPENDENCY" for row in arm["observations"].values())


def test_terminal_wrong_geometry_is_failed_before_completed_or_rgb_artifacts(tmp_path):
    events = []
    residency = FakeExperimentResidency(_single_off_config(), lambda name, row: events.append((name, row)))
    residency.decoded = torch.zeros(180, 8, 32, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="public RGB geometry"):
        runtime.save_terminal_and_rgb_layers(
            residency, torch.zeros(1, 1, 2, 2, 2), tmp_path,
            event=lambda name, row: events.append((name, row)), restore_transformer=False,
            public_protocol=CATALOG_PROTOCOL, raster_saver=small_raster_saver,
        )
    terminal_events = [row for name, row in events if name == "terminal_decode"]
    assert [row["status"] for row in terminal_events] == ["ATTEMPTED", "FAILED"]
    assert not (tmp_path / "float_rgb.pt").exists() and not (tmp_path / "rgb8.rgb").exists()


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
