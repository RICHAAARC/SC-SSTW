"""CPU/static checks for the fixed local-gradient route; no GPU or real MP4."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.wan_state_clock import rgb_dct_local_gradient_run as trial
from main.tube_state import rgb_dct_gradient_proxy as proxy
from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state import rgb_dct_presence as baseline
from runtime.wan.vae import decode_normalized_latent_with_grad
from runtime.wan.rgb_dct_gradient_checkpoint import (
    BoundarySpool, BoundaryStorage, CheckpointLedger, checkpoint_call,
    checkpoint_decode,
)
from scripts import build_rgb_dct_local_gradient_notebook as builder


pytestmark = pytest.mark.unit


def _measure(rms: float) -> dict:
    return dict(support_rms=rms, global_rms=rms, peak_abs=rms)


class FakeBackend:
    def __init__(self, store, *, fail_gradient=False, zero_gradient=False,
                 fail_release=False, fail_resources=False):
        self.store = store
        self.fail_gradient = fail_gradient
        self.zero_gradient = zero_gradient
        self.fail_release = fail_release
        self.fail_resources = fail_resources
        self.off_terminal = torch.tensor([0.0])
        self.terminals = {}
        self.phase = "INITIAL"
        self.local_index = None
        self.events = []

    def prepare_off(self, artifact_dir):
        self.store.count("generation", False)
        for _ in range(100):
            self.store.count("transformer", False); self.store.count("transformer", True)
        for _ in range(50):
            self.store.count("scheduler_step", False); self.store.count("scheduler_step", True)
        self.store.count("generation", True)
        self.phase = "TRANSFORMER"
        self.events.append("OFF")
        return dict(initial_noise_fingerprint="fake", history_indices=[44, 49])

    def to_vae_phase(self, name):
        assert self.phase == "TRANSFORMER"
        self.phase = "VAE"; self.events.append(name)
        return dict(name=name, transformer_resident="cpu", vae_resident="cuda")

    def to_transformer_phase(self, name):
        assert self.phase == "VAE"
        self.phase = "TRANSFORMER"; self.events.append(name)
        return dict(name=name, identity_verified=True)

    def decode(self, latent):
        assert self.phase == "VAE"
        self.store.count("vae_decode", False); self.store.count("vae_decode", True)
        return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)

    def encode(self, rgb):
        assert self.phase == "VAE"
        self.store.count("vae_encode", False); self.store.count("vae_encode", True)
        return np.array([1.0], dtype=np.float32)

    def control_single49(self, direction, target):
        row = self._control(49, target)
        return "READY", torch.tensor([49.0]), row

    def start_local44(self):
        assert self.phase == "TRANSFORMER"
        for _ in range(2):
            self.store.count("transformer", False); self.store.count("transformer", True)
        self.local_index = 44; self.events.append("LIVE_44")
        return dict(index=44, state_fingerprint="z44", velocity_fingerprint="v44",
                    history_fingerprint="h44")

    def local_gradient(self, key):
        assert self.phase == "VAE" and key == b"WanProjection-first-validation-key-v1"
        self.store.count("vae_decode", False); self.store.count("vae_decode", True)
        for _ in range(46):
            self.store.count("vae_chunk_forward", False)
            self.store.count("vae_chunk_forward", True)
        self.store.count("backward", False)
        if self.fail_gradient and self.local_index == 44:
            raise RuntimeError("fake gradient chain broken")
        for _ in range(46):
            self.store.count("vae_chunk_recompute", False)
            self.store.count("vae_chunk_recompute", True)
        self.store.count("backward", True)
        row = dict(index=self.local_index, loss=1.0,
                   q=[-0.1] * 30, positive_groups=0,
                   numpy_q=[-0.1] * 30, numpy_positive_groups=0,
                   proxy_q_max_abs_error=0.0, proxy_q_parity=True,
                   negative_masked_gradient=_measure(1.0))
        self.events.append(f"GRAD_{self.local_index}")
        if self.zero_gradient and self.local_index == 44:
            row["negative_masked_gradient"] = _measure(0.0)
            return "ZERO_LOCAL_GRADIENT", None, row
        return "READY", torch.tensor([1.0]), row

    def verify_local_cfg(self):
        for _ in range(2):
            self.store.count("transformer_validation", False)
            self.store.count("transformer_validation", True)
        self.events.append(f"VERIFY_{self.local_index}")
        return dict(index=self.local_index, numerical_match=True)

    def _control(self, index, target):
        self.store.count("shadow_step", False); self.store.count("shadow_step", True)
        self.store.count("unit_response_probe_step", False)
        self.store.count("unit_response_probe_step", True)
        self.store.count("scheduler_step", False); self.store.count("scheduler_step", True)
        self.events.append(f"CONTROL_{index}")
        return dict(index=index, target_D_support_rms=target,
                    actual_D=_measure(target), unit_D=_measure(1.0))

    def control_local(self, direction, target):
        row = self._control(self.local_index, target)
        self.local_index += 1
        return "READY", row

    def advance_local_to(self, target):
        assert self.local_index == target - 1
        for _ in range(4):
            self.store.count("transformer", False); self.store.count("transformer", True)
        self.store.count("scheduler_step", False); self.store.count("scheduler_step", True)
        self.local_index = target; self.events.append(f"LIVE_{target}")
        return dict(index=target, state_fingerprint=f"z{target}",
                    velocity_fingerprint=f"v{target}", history_fingerprint=f"h{target}")

    def finish_local(self):
        assert self.local_index == 49
        for _ in range(2):
            self.store.count("transformer", False); self.store.count("transformer", True)
        self.store.count("scheduler_step", False); self.store.count("scheduler_step", True)
        self.terminals["LOCAL44_46_48"] = torch.tensor([50.0])
        return self.terminals["LOCAL44_46_48"]

    def net_from_off(self, terminal):
        return _measure(float(terminal[0]))

    def resources(self):
        if self.fail_resources:
            raise RuntimeError("fake resource collection failure after scored media")
        return dict(fake_cpu=True, events=self.events)

    def release(self):
        self.phase = "RELEASED"
        if self.fail_release:
            raise RuntimeError("fake release failure after scored media")


def _run_fake(tmp_path, monkeypatch, *, fail_gradient=False,
              zero_gradient=False, fail_release=False,
              fail_resources=False, fail_quality=False):
    config = trial.load_config()
    output = tmp_path / "fixed"; output.mkdir()
    monkeypatch.setattr(trial, "apply_carrier", lambda rgb, key, polarity: (rgb, {"fake": True}))
    monkeypatch.setattr(trial, "lift_direction",
                        lambda raw: (np.array([1.0], dtype=np.float32), {"status": "READY"}))
    quality_calls = []
    def quality_fn(off, marked):
        quality_calls.append(len(quality_calls))
        if fail_quality and len(quality_calls) == 1:
            raise RuntimeError("fake paired RGB RMSE failure after blind score")
        return dict(compared_frames=181, rgb_rmse=0.0)
    monkeypatch.setattr(trial, "_paired_mp4_quality", quality_fn)
    backends = []

    def encode_fn(rgb, path, fps, crf):
        path.write_bytes(b"fake MP4")

    last_arms = []
    def read_fn(path):
        last_arms.append(path.parent.name)
        return np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)

    def receiver_fn(rgb, key):
        arm = last_arms.pop(0)
        c = 10 if arm == "OFF" else 27
        q = [1.0] * c + [-1.0] * (30 - c)
        return dict(status="SCORED", score=float(c) / 100, group_scores=q,
                    positive_groups=c, frames_used=181,
                    spec_sha256=receiver.SPEC_SHA256,
                    key_id=config["receiver_key_id"])

    def worker_fn(output, config, case_id):
        store = trial.Store.open(output / "result.json")
        first = case_id == trial.CASE_IDS[0]
        backend = FakeBackend(store,
                              fail_gradient=fail_gradient and first,
                              zero_gradient=zero_gradient and first,
                              fail_release=fail_release and first,
                              fail_resources=fail_resources and first)
        backends.append(backend)
        trial.run_case(store, case_id, config, backend, encode_fn=encode_fn,
                       read_fn=read_fn, receiver_fn=receiver_fn,
                       environment_fn=lambda _: {"status": "FAKE_CPU_STATIC"})
        return None

    trial._supervise(output, config, "a" * 40, worker_fn=worker_fn)
    return trial.Store.open(output / "result.json"), backends


def test_protocol_roster_budget_and_six_preserved_slots(tmp_path):
    config = trial.load_config()
    assert trial._sha(trial.PROTOCOL_PATH) == config["protocol_sha256"]
    assert config["control"]["local_indices"] == [44, 46, 48]
    assert config["per_source_call_plan_max"]["backward"] == 3
    assert config["per_source_call_plan_max"]["vae_chunk_forward"] == 138
    assert config["call_plan_max"]["native_scheduler_total"] == 130
    store = trial.Store(tmp_path / "result.json",
                        trial.initial_result(config, tmp_path, "a" * 40))
    store.save()
    assert store.data["pending_media_slots"] == 6
    assert all(list(case["slots"]) == list(trial.ARMS)
               for case in store.data["cases"].values())


def test_full_181_frame_160_block_q_parity():
    key = b"WanProjection-first-validation-key-v1"
    spatial, _, temporal = baseline.key_codes(key)
    block = baseline._KERNEL / np.max(np.abs(baseline._KERNEL))
    signed_blocks = spatial.reshape(10, 16, 1, 1) * block[None, None]
    image = signed_blocks.transpose(0, 2, 1, 3).reshape(320, 512)
    rgb = np.empty((181, 320, 512, 3), dtype=np.float32)
    for index in range(181):
        frame = np.float32(0.5 + 0.02 * temporal[index] * image)
        rgb[index] = frame[:, :, None]
    expected = receiver.score_rgb(rgb, key)
    assert expected["status"] == "SCORED" and expected["frames_used"] == 181
    q, metadata = proxy.group_scores(torch.from_numpy(rgb), key)
    actual = q.detach().numpy()
    assert metadata["features"].shape == (181, 160)
    assert np.allclose(actual, expected["group_scores"], atol=1e-9, rtol=1e-8)
    assert int(np.count_nonzero(actual > 0)) == expected["positive_groups"]


def test_proxy_objective_has_nonzero_gradient_and_vae_keeps_input_graph():
    key = b"WanProjection-first-validation-key-v1"
    spatial, _, temporal = baseline.key_codes(key)
    amplitude = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    features = amplitude * torch.as_tensor(temporal[:, None] * spatial[None, :],
                                           dtype=torch.float64)
    q, _ = proxy.group_scores_from_features(features, key)
    loss = torch.relu(-q).square().mean()
    gradient, = torch.autograd.grad(loss, amplitude)
    assert torch.isfinite(gradient) and gradient.abs().item() > 0

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.config = type("Config", (), {"latents_mean": [0.0],
                                               "latents_std": [1.0]})()
        def decode(self, value, return_dict=False):
            return (value.repeat(1, 3, 1, 1, 1),)
    latent = torch.full((1, 1, 2, 2, 2), 0.2, requires_grad=True)
    decoded = decode_normalized_latent_with_grad(TinyVAE(), latent)
    decoded.sum().backward()
    assert latent.grad is not None and torch.count_nonzero(latent.grad) == latent.numel()


def test_nonreentrant_checkpoint_recomputes_and_counts_separately():
    calls = []
    ledger = CheckpointLedger(lambda kind, complete: calls.append((kind, complete)), 1, 1)
    value = torch.tensor([2.0], requires_grad=True)
    output = checkpoint_call(ledger, lambda item: item.square(), value)
    output.sum().backward()
    assert value.grad.item() == pytest.approx(4.0)
    assert calls == [("vae_chunk_forward", False), ("vae_chunk_forward", True),
                     ("vae_chunk_recompute", False),
                     ("vae_chunk_recompute", True)]


def test_native_shaped_46_chunk_causal_checkpoint_matches_direct_gradient():
    class FakeCausalDecoder(torch.nn.Module):
        def forward(self, value, feat_cache=None, feat_idx=None, first_chunk=False):
            assert len(feat_cache) == 3
            assert torch.is_tensor(feat_cache[0])
            assert feat_cache[1] is None and feat_cache[2] == "Rep"
            previous = feat_cache[0]
            output = torch.sin(value * 1.75 + previous * 0.25)
            feat_cache[0] = torch.tanh(value * 0.5)
            feat_idx[0] = 3
            return output

    class FakeCausalVAE:
        use_tiling = False
        def __init__(self):
            self.decoder = FakeCausalDecoder()
        def decode(self, latent, return_dict=False):
            cache = [torch.zeros_like(latent[:, :, :1]), None, "Rep"]
            chunks = []
            for index in range(latent.shape[2]):
                cursor = [0]
                chunks.append(self.decoder.forward(
                    latent[:, :, index:index + 1], feat_cache=cache,
                    feat_idx=cursor, first_chunk=(index == 0)))
                assert cursor[0] == 3
            return (torch.cat(chunks, dim=2),)

    direct_vae = FakeCausalVAE()
    checkpoint_vae = FakeCausalVAE()
    direct_input = torch.linspace(-1, 1, 46, dtype=torch.float64).reshape(1, 1, 46, 1, 1)
    direct_input.requires_grad_(True)
    checkpoint_input = direct_input.detach().clone().requires_grad_(True)
    direct_output = direct_vae.decode(direct_input, return_dict=False)[0]
    direct_loss = direct_output.square().sum()
    direct_loss.backward()

    calls = []
    ledger = CheckpointLedger(lambda kind, complete: calls.append((kind, complete)), 46, 46)
    original_function = checkpoint_vae.decoder.forward.__func__
    checkpoint_output = checkpoint_decode(checkpoint_vae, checkpoint_input, ledger)
    checkpoint_loss = checkpoint_output.square().sum()
    checkpoint_loss.backward()

    assert torch.allclose(checkpoint_output, direct_output, rtol=0, atol=0)
    assert torch.allclose(checkpoint_input.grad, direct_input.grad, rtol=1e-12, atol=1e-12)
    assert checkpoint_vae.decoder.forward.__func__ is original_function
    assert calls.count(("vae_chunk_forward", False)) == 46
    assert calls.count(("vae_chunk_forward", True)) == 46
    assert calls.count(("vae_chunk_recompute", False)) == 46
    assert calls.count(("vae_chunk_recompute", True)) == 46
    summary = ledger.summary()
    assert len(summary["cache_boundaries"]) == 1
    assert len(summary["cache_boundaries"][0]["forward"]["chunks"]) == 46
    assert all(row["tensor_slots"] == 1
               for row in summary["cache_boundaries"][0]["forward"]["chunks"])
    ledger.release_boundary_storage()


def test_bounded_boundary_spool_preserves_views_and_three_sequential_vjps(
        tmp_path, monkeypatch):
    import runtime.wan.rgb_dct_gradient_checkpoint as checkpoints

    monkeypatch.setenv("RGB_DCT_BOUNDARY_SPOOL_ROOT", str(tmp_path))
    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(checkpoints, "TRANSFER_BYTES", 7)
    spool = BoundarySpool()
    source = torch.arange(24, dtype=torch.float64).reshape(4, 6)
    view_a, view_b = source[:, 1:5], source[1:3, 2:6]
    storage = BoundaryStorage(spool)
    packed_a, packed_b = storage.pack(view_a), storage.pack(view_b)
    assert packed_a[0] == packed_b[0] and spool.summary()["files"] == 1
    storage.copies.clear()
    restored_a, restored_b = storage.unpack(packed_a), storage.unpack(packed_b)
    assert torch.equal(restored_a, view_a) and torch.equal(restored_b, view_b)
    assert restored_a.untyped_storage().data_ptr() == restored_b.untyped_storage().data_ptr()
    restored_a[1, 1] = -123
    assert restored_b[0, 0].item() == -123
    assert spool.summary()["cpu_peak_packed_bytes"] <= 7
    assert spool.summary()["d2h_bytes"] == spool.summary()["h2d_bytes"] == 0
    spool.close()
    assert not spool.path.exists()

    calls = []
    ledger = CheckpointLedger(lambda kind, done: calls.append((kind, done)), 3, 3)
    for _ in (44, 46, 48):
        latent = torch.arange(8, dtype=torch.float64).requires_grad_(True)
        ledger.new_decode()
        actual = checkpoint_call(ledger, lambda value: value.sin().square().sum(), latent)
        actual.backward()
        assert torch.allclose(latent.grad, (2 * latent.detach().sin()
                                             * latent.detach().cos()))
        active = ledger.summary()["boundary_storage_aggregate"]
        assert active["decodes"] == len(ledger.boundary_storage_history) + 1
        assert active["active_disk_live_bytes"] > 0
        ledger.release_boundary_storage()
    summary = ledger.summary()
    history = summary["boundary_storage_history"]
    assert len(history) == 3 and all(row["closed"] for row in history)
    assert all(row["disk_peak_bytes"] > 0 and row["disk_live_bytes"] == 0
               for row in history)
    assert len(list(tmp_path.iterdir())) == 0
    aggregate = summary["boundary_storage_aggregate"]
    assert aggregate["decodes"] == aggregate["closed_decodes"] == 3
    for name in ("d2h_bytes", "h2d_bytes", "disk_written_bytes",
                 "disk_read_bytes", "files"):
        assert aggregate[name] == sum(row[name] for row in history)
    for name in ("disk_peak_bytes", "cpu_peak_packed_bytes"):
        assert aggregate[name] == max(row[name] for row in history)
    assert aggregate["d2h_bytes"] == aggregate["h2d_bytes"] == 0
    assert aggregate["active_disk_live_bytes"] == 0
    assert aggregate["active_cpu_live_packed_bytes"] == 0
    assert calls.count(("vae_chunk_forward", True)) == 3
    assert calls.count(("vae_chunk_recompute", True)) == 3


def test_full_decode_disk_preflight_and_transfer_failure_cleanup(tmp_path, monkeypatch):
    import runtime.wan.rgb_dct_gradient_checkpoint as checkpoints

    monkeypatch.setenv("RGB_DCT_BOUNDARY_SPOOL_ROOT", str(tmp_path))
    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 10**20)
    with pytest.raises(RuntimeError, match="VAE_BOUNDARY_DISK_PREFLIGHT"):
        BoundarySpool()
    monkeypatch.setattr(checkpoints, "EXPECTED_BOUNDARY_BYTES", 0)
    monkeypatch.setattr(checkpoints, "DISK_FREE_MARGIN_BYTES", 0)
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
    assert spool.summary()["disk_live_bytes"] == 0
    assert not list(spool.path.glob("storage-*"))
    spool.close()


def test_net_from_off_moves_measurement_only_to_cpu():
    from runtime.wan.rgb_dct_local_gradient_backend import WanLocalGradientBackend

    backend = object.__new__(WanLocalGradientBackend)
    backend.off_terminal = torch.full((1, 1, 46, 1, 1), 1.0)

    class Terminal:
        def detach(self):
            return self

        def to(self, *, device, dtype):
            assert device == "cpu" and dtype == torch.float32
            return torch.full((1, 1, 46, 1, 1), 2.0, dtype=dtype)

    assert backend.net_from_off(Terminal())["global_rms"] == pytest.approx(1.0)


def test_complete_fake_run_uses_live_histories_and_exact_call_caps(tmp_path, monkeypatch):
    store, backends = _run_fake(tmp_path, monkeypatch)
    assert store.data["status"] == "FIXED_LOCAL_GRADIENT_COMPLETE"
    assert store.data["attempted_media_slots"] == 6
    assert store.data["scored_media_slots"] == 6 and store.data["scored_frames"] == 1086
    for kind, cap in trial.PLAN.items():
        assert store.data["calls"][kind] == {"attempted": cap, "completed": cap}
    for case_id, backend in zip(trial.CASE_IDS, backends, strict=True):
        assert [row["index"] for row in store.case(case_id)["proxy_gradients"]] == [44, 46, 48]
        assert backend.events.index("LIVE_46") < backend.events.index("GRAD_46")
        assert backend.events.index("LIVE_48") < backend.events.index("GRAD_48")
        assert backend.events[-2:] == ["CONTROL_48", "FINAL_MEDIA_VAE"]
        metrics = store.case(case_id)["arm_metrics"]["LOCAL44_46_48"]
        assert metrics["sum_support_rms"] == pytest.approx(trial.R_STAR)
        assert metrics["sum_support_rms_squared"] == pytest.approx(trial.R_STAR ** 2 / 3)


def test_gradient_failure_is_persisted_and_next_source_runs(tmp_path, monkeypatch):
    store, _ = _run_fake(tmp_path, monkeypatch, fail_gradient=True)
    first = store.case(trial.CASE_IDS[0])
    assert first["slots"]["OFF"]["status"] == "SCORED"
    assert first["slots"]["SINGLE49"]["status"] == "SCORED"
    assert first["slots"]["LOCAL44_46_48"]["status"] == "ENGINEERING_INVALID"
    assert first["slots"]["LOCAL44_46_48"]["attempted"] is True
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert store.data["attempted_media_slots"] == 6 and store.data["scored_media_slots"] == 5


def test_exact_zero_local_gradient_is_engineering_invalid_with_receipt(tmp_path, monkeypatch):
    store, _ = _run_fake(tmp_path, monkeypatch, zero_gradient=True)
    first = store.case(trial.CASE_IDS[0])
    slot = first["slots"]["LOCAL44_46_48"]
    assert slot["status"] == "ENGINEERING_INVALID"
    assert slot["reason"] == "ZERO_LOCAL_GRADIENT"
    assert first["proxy_gradients"][0]["loss"] == 1.0
    assert first["proxy_gradients"][0]["negative_masked_gradient"]["support_rms"] == 0
    assert slot["raw_failures"][0]["receipt"] == first["proxy_gradients"][0]
    assert first["status"] == "ENGINEERING_INVALID"
    assert store.case(trial.CASE_IDS[1])["status"] == "SCORED"
    assert store.data["status"] == "INCOMPLETE"


def test_release_failure_prevents_complete_even_when_all_slots_scored(tmp_path, monkeypatch):
    store, _ = _run_fake(tmp_path, monkeypatch, fail_release=True)
    first = store.case(trial.CASE_IDS[0])
    assert all(slot["status"] == "SCORED" for slot in first["slots"].values())
    assert first["status"] == "ENGINEERING_INVALID"
    assert first["failures"][-1]["stage"] == "RELEASE"
    assert store.data["scored_media_slots"] == 6
    assert store.data["status"] == "INCOMPLETE"


def test_resource_receipt_failure_prevents_complete_with_six_scored_slots(
        tmp_path, monkeypatch):
    store, _ = _run_fake(tmp_path, monkeypatch, fail_resources=True)
    first = store.case(trial.CASE_IDS[0])
    assert all(slot["status"] == "SCORED" for slot in first["slots"].values())
    assert first["resources"]["status"] == "UNAVAILABLE"
    failure = next(row for row in first["failures"] if row["stage"] == "RESOURCES")
    assert failure["exception_class"] == "RuntimeError"
    assert "resource collection failure" in failure["message"]
    assert "Traceback" in failure["traceback"]
    assert first["status"] == "ENGINEERING_INVALID"
    assert store.data["scored_media_slots"] == 6
    assert store.data["status"] == "INCOMPLETE"


def test_paired_rgb_quality_failure_keeps_blind_score_but_prevents_complete(
        tmp_path, monkeypatch):
    store, _ = _run_fake(tmp_path, monkeypatch, fail_quality=True)
    first = store.case(trial.CASE_IDS[0])
    slot = first["slots"]["SINGLE49"]
    assert slot["status"] == "SCORED" and slot["decision"] == "H1"
    assert slot["positive_groups"] == 27
    assert slot["quality_vs_off"] is None
    assert "paired RGB RMSE failure" in slot["quality_error"]
    failure = slot["quality_failure"]
    assert failure["stage"] == "SINGLE49_PAIRED_MP4_QUALITY"
    assert failure["exception_class"] == "RuntimeError"
    assert "Traceback" in failure["traceback"]
    assert failure in first["failures"]
    assert all(row["status"] == "SCORED" for row in first["slots"].values())
    assert first["status"] == "ENGINEERING_INVALID"
    assert store.data["scored_media_slots"] == 6
    assert store.data["status"] == "INCOMPLETE"


def _one_slot_store(tmp_path, suffix):
    config = trial.load_config()
    output = tmp_path / suffix
    output.mkdir()
    store = trial.Store(output / "result.json",
                        trial.initial_result(config, output, "a" * 40))
    store.active_case = trial.CASE_IDS[0]
    store.save()
    return config, store


def test_media_failures_retain_raw_subprocess_and_receiver_records(tmp_path):
    rgb = np.broadcast_to(np.float32(0.5), trial.RGB_SHAPE)
    key = b"WanProjection-first-validation-key-v1"

    config, store = _one_slot_store(tmp_path, "encode")
    def encode_failure(*_):
        raise subprocess.CalledProcessError(
            7, ["ffmpeg", "-i", "pipe:0"], output=b"encoder stdout", stderr=b"encoder stderr")
    trial._save_and_score(
        store, trial.CASE_IDS[0], "OFF", rgb, key, config,
        encode_fn=encode_failure, read_fn=lambda _: rgb,
        receiver_fn=lambda *_: {})
    slot = store.case(trial.CASE_IDS[0])["slots"]["OFF"]
    failure = slot["raw_failures"][0]
    assert failure["stage"] == "MP4_ENCODE"
    assert failure["exception_class"] == "CalledProcessError"
    assert failure["command"] == ["ffmpeg", "-i", "pipe:0"]
    assert failure["return_code"] == 7
    assert failure["output"] == "encoder stdout" and failure["stderr"] == "encoder stderr"
    assert store.data["calls"]["mp4_save"] == {"attempted": 1, "completed": 0}
    assert store.data["calls"]["mp4_read"] == {"attempted": 0, "completed": 0}

    config, store = _one_slot_store(tmp_path, "read")
    def encode_ok(_, path, *__):
        path.write_bytes(b"mp4")
    def read_failure(_):
        raise subprocess.CalledProcessError(
            9, ["ffprobe", "bad.mp4"], output="probe out", stderr="probe err")
    trial._save_and_score(
        store, trial.CASE_IDS[0], "OFF", rgb, key, config,
        encode_fn=encode_ok, read_fn=read_failure, receiver_fn=lambda *_: {})
    slot = store.case(trial.CASE_IDS[0])["slots"]["OFF"]
    failure = slot["raw_failures"][0]
    assert failure["stage"] == "MP4_RGB24_READ" and failure["return_code"] == 9
    assert failure["output"] == "probe out" and failure["stderr"] == "probe err"
    assert store.data["calls"]["mp4_read"] == {"attempted": 1, "completed": 0}
    assert store.data["calls"]["score"] == {"attempted": 0, "completed": 0}

    config, store = _one_slot_store(tmp_path, "score_exception")
    def score_failure(*_):
        raise RuntimeError("receiver reduction exploded")
    trial._save_and_score(
        store, trial.CASE_IDS[0], "OFF", rgb, key, config,
        encode_fn=encode_ok, read_fn=lambda _: rgb, receiver_fn=score_failure)
    slot = store.case(trial.CASE_IDS[0])["slots"]["OFF"]
    failure = slot["raw_failures"][0]
    assert failure["stage"] == "BLIND_RECEIVER_SCORE"
    assert failure["exception_class"] == "RuntimeError"
    assert "receiver reduction exploded" in failure["message"]
    assert store.data["calls"]["mp4_read"] == {"attempted": 1, "completed": 1}
    assert store.data["calls"]["score"] == {"attempted": 1, "completed": 0}

    config, store = _one_slot_store(tmp_path, "invalid_receiver")
    invalid = dict(status="INVALID", reason="FULL_RGB_SHAPE_REQUIRED", frames_used=0)
    trial._save_and_score(
        store, trial.CASE_IDS[0], "OFF", rgb, key, config,
        encode_fn=encode_ok, read_fn=lambda _: rgb,
        receiver_fn=lambda *_: invalid)
    slot = store.case(trial.CASE_IDS[0])["slots"]["OFF"]
    assert slot["status"] == "INVALID" and slot["receiver_output"] == invalid
    assert store.data["calls"]["score"] == {"attempted": 1, "completed": 0}


def test_real_cli_worker_failure_persists_all_six_attempted_slots(tmp_path):
    config = trial.load_config()
    output = tmp_path / "real_cli_failure"; output.mkdir()
    # The wrong immutable SHA makes the actual child CLI fail before model load.
    trial._supervise(output, config, "0" * 40, worker_fn=trial._run_worker)
    result = json.loads((output / "result.json").read_text())
    assert result["attempted_media_slots"] == 6
    assert result["scored_media_slots"] == 0 and result["pending_media_slots"] == 0
    assert result["invalid_media_slots"] == 6
    for case_id in trial.CASE_IDS:
        assert Path(result["cases"][case_id]["worker_log_path"]).is_file()
        monitor = json.loads((output / f"{case_id}_worker_monitor.json").read_text())
        assert monitor["boundary_spool_cleanup"] == "COMPLETED"
        assert monitor["parent_observed_peak_rss_kib"] is None or isinstance(
            monitor["parent_observed_peak_rss_kib"], int)
        assert all(slot["status"] == "NOT_RUN_WORKER_FAILURE" and slot["attempted"]
                   for slot in result["cases"][case_id]["slots"].values())


def test_parent_cleans_orphan_spool_on_sigkill_and_monitor_error(tmp_path, monkeypatch):
    roots, signals = [], []

    class KilledWorker:
        pid = 123456
        returncode = -9

        def __init__(self, command, **kwargs):
            root = Path(kwargs["env"]["RGB_DCT_BOUNDARY_SPOOL_ROOT"])
            roots.append(root)
            (root / "orphan-storage").write_bytes(b"partial VAE boundary")

        def poll(self):
            return self.returncode

    monkeypatch.setattr(trial.subprocess, "Popen", KilledWorker)
    monkeypatch.setattr(trial.os, "killpg", lambda pid, sig: signals.append(sig))
    receipt = trial._run_worker(tmp_path, trial.load_config(), trial.CASE_IDS[0])
    assert receipt["status"] == "FAILED" and receipt["error"] == "WORKER_EXIT_-9"
    assert receipt["boundary_spool_cleanup"] == "COMPLETED"
    assert signals == [trial.signal.SIGKILL]
    assert roots and not roots[0].exists()
    assert json.loads((tmp_path / f"{trial.CASE_IDS[0]}_worker_monitor.json").read_text()) == receipt

    class RunningWorker(KilledWorker):
        returncode = None

        def wait(self):
            self.returncode = -9

    monkeypatch.setattr(trial.subprocess, "Popen", RunningWorker)
    monkeypatch.setattr(trial, "_proc_rss_kib", lambda pid: (_ for _ in ()).throw(
        RuntimeError("monitor failed")))
    receipt = trial._run_worker(tmp_path, trial.load_config(), trial.CASE_IDS[1])
    assert receipt["status"] == "FAILED" and "WORKER_MONITOR_FAILED" in receipt["error"]
    assert receipt["termination"] == "MONITOR_ABORT_SIGKILL"
    assert receipt["boundary_spool_cleanup"] == "COMPLETED"
    assert not roots[-1].exists()


def test_worker_spool_creation_and_initial_monitor_write_failures_are_receipted(
        tmp_path, monkeypatch):
    original_tempdir = trial.tempfile.TemporaryDirectory
    monkeypatch.setattr(trial.tempfile, "TemporaryDirectory", lambda **kwargs: (
        _ for _ in ()).throw(OSError("spool creation failed")))
    receipt = trial._run_worker(tmp_path, trial.load_config(), trial.CASE_IDS[0])
    assert receipt["status"] == "FAILED"
    assert "spool creation failed" in receipt["error"]
    assert receipt["boundary_spool_cleanup"] == "NOT_CREATED"
    assert json.loads((tmp_path / f"{trial.CASE_IDS[0]}_worker_monitor.json").read_text()) == receipt

    roots = []

    def tracked_tempdir(**kwargs):
        owner = original_tempdir(**kwargs)
        roots.append(Path(owner.name))
        return owner

    monkeypatch.setattr(trial.tempfile, "TemporaryDirectory", tracked_tempdir)
    original_atomic = trial._atomic_json
    calls = 0

    def fail_first_monitor(path, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("initial monitor write failed")
        return original_atomic(path, data)

    monkeypatch.setattr(trial, "_atomic_json", fail_first_monitor)
    receipt = trial._run_worker(tmp_path, trial.load_config(), trial.CASE_IDS[1])
    assert receipt["status"] == "FAILED"
    assert "initial monitor write failed" in receipt["error"]
    assert receipt["boundary_spool_cleanup"] == "COMPLETED"
    assert roots and not roots[0].exists()
    assert json.loads((tmp_path / f"{trial.CASE_IDS[1]}_worker_monitor.json").read_text()) == receipt


def test_supervisor_exception_retains_fixed_six_invalid_slots(tmp_path):
    output = tmp_path / "supervisor_failure"
    output.mkdir()

    def bad_worker(output, config, case_id):
        raise OSError(f"worker preparation failed for {case_id}")

    trial._supervise(output, trial.load_config(), "a" * 40, worker_fn=bad_worker)
    result = json.loads((output / "result.json").read_text())
    assert result["attempted_media_slots"] == result["invalid_media_slots"] == 6
    assert result["pending_media_slots"] == result["scored_media_slots"] == 0
    for case_id in trial.CASE_IDS:
        case = result["cases"][case_id]
        assert case["worker_receipt"]["status"] == "FAILED"
        assert "worker preparation failed" in case["worker_receipt"]["error"]
        assert all(slot["status"] == "NOT_RUN_WORKER_FAILURE" and slot["attempted"]
                   for slot in case["slots"].values())


def test_final_monitor_write_failure_returns_failed_receipt_and_cleans_spool(
        tmp_path, monkeypatch):
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
    original_atomic = trial._atomic_json
    calls = 0

    def fail_final(path, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("final monitor write failed")
        return original_atomic(path, data)

    monkeypatch.setattr(trial, "_atomic_json", fail_final)
    receipt = trial._run_worker(tmp_path, trial.load_config(), trial.CASE_IDS[0])
    assert receipt["status"] == "FAILED"
    assert "final monitor write failed" in receipt["final_monitor_write_error"]
    assert receipt["boundary_spool_cleanup"] == "COMPLETED"
    assert roots and not roots[0].exists()


def test_notebook_builder_static_binding_and_setup_failure_slots(tmp_path):
    path = builder.build("a" * 40, tmp_path / "fixed.ipynb")
    notebook = json.loads(path.read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]
    assert code[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert notebook["metadata"]["source_commit"] == "a" * 40
    joined = "\n".join(code)
    assert "dev/rgb-dct-local-gradient-v1" in joined
    assert "rgb_dct_local_gradient_run" in joined
    assert "LOCAL44_46_48" in joined and "mp4_score_slots=6" in joined
    assert "split('+', 1)[0]" in joined and "diffusers.__version__ != '0.40.0'" in joined
    assert "ENVIRONMENT_VALIDATION_FAILED" in joined
    assert "guarded_setup('INSTALL'" in joined
    assert "guarded_setup('SOURCE_CHECKOUT'" in joined
    assert "guarded_setup('ENVIRONMENT'" in joined
    assert "guarded_setup('RUN'" in joined
    assert "guarded_setup('RESULT_PRESENTATION'" in joined
    for source in code:
        compile(source, "notebook-cell", "exec")


@pytest.mark.parametrize(("stage", "source", "expected_class"), [
    ("INSTALL", "raise ImportError('missing diffusers')", "ImportError"),
    ("SOURCE_CHECKOUT", "raise RuntimeError('immutable source SHA readback mismatch')",
     "RuntimeError"),
    ("ENVIRONMENT", "raise AssertionError('cuda missing')", "AssertionError"),
    ("RUN", "raise FileNotFoundError('runner produced no retained result.json')",
     "FileNotFoundError"),
    ("INSTALL", "raise subprocess.CalledProcessError(13, ['pip', 'install'], " +
                "output=b'pip out', stderr=b'pip err')", "CalledProcessError"),
])
def test_notebook_failure_helper_retains_six_attempted_placeholders(
        tmp_path, stage, source, expected_class):
    output = tmp_path / (stage.lower() + expected_class.lower())
    output.mkdir()
    slots_path = output / "setup_slots.json"
    slots = dict(status="GLOBAL_SETUP_NOT_COMPLETED",
                 fixed_denominator=dict(sources=2, mp4_score_slots=6, frames=1086),
                 cases={case_id: {arm: dict(status="PENDING", reason=None, attempted=False)
                                  for arm in trial.ARMS}
                        for case_id in trial.CASE_IDS})
    slots_path.write_text(json.dumps(slots) + "\n")
    namespace = dict(OUTPUT=output, SETUP_SLOTS_PATH=slots_path,
                     SOURCE_SHA="a" * 40)
    exec(builder._failure_support_source(), namespace)
    with pytest.raises(Exception):
        namespace["guarded_setup"](stage, source)
    persisted = json.loads(slots_path.read_text())
    flat = [slot for case in persisted["cases"].values() for slot in case.values()]
    assert len(flat) == 6
    assert all(slot["status"] == "NOT_RUN_GLOBAL_SETUP_FAILURE"
               and slot["attempted"] for slot in flat)
    failure = json.loads((output / "setup_failure.json").read_text())
    assert failure["stage"] == stage and failure["exception_class"] == expected_class
    assert "FAILURE " in (output / "setup.log").read_text()
    if expected_class == "CalledProcessError":
        assert failure["command"] == ["pip", "install"]
        assert failure["return_code"] == 13
        assert failure["output"] == "pip out" and failure["stderr"] == "pip err"
