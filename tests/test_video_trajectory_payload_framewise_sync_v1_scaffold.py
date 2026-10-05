"""Candidate-only CPU/fake validation; no real model, codec, GPU, or Drive."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.wan_state_clock import video_trajectory_payload_framewise_sync_v1_run as runner
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as method
from runtime.wan import framewise_autoencoder_kl, rgb8_source


pytestmark = pytest.mark.unit


def small_public(*, formal_time: bool = False) -> method.PublicProtocol:
    return method.PublicProtocol(
        source_frames=181 if formal_time else 9,
        received_frame_counts=(181, 177) if formal_time else (9, 5),
        source_height=32,
        source_width=32,
        framewise_downsample=8,
        latent_channels=4,
        tubelet_length=4,
        patch_height=4,
        patch_width=4,
    )


def test_public_offset_support_and_crop_truth_is_not_public():
    public = method.PUBLIC
    receipt = method.public_receipt(public)
    assert "crop_start" not in receipt
    assert "crop_stop" not in receipt
    assert method.candidate_offsets(181, public) == (0,)
    assert method.candidate_offsets(177, public) == (0, 1, 2, 3, 4)
    correct = method.tubelet_support_rows(177, 1, public)
    assert len(correct) == 45
    assert correct[0]["source_ages"] == [1, 2, 3]
    assert correct[-1]["source_tubelet"] == 44
    assert correct[-1]["source_indices"] == [176, 177]
    assert sum(
        len(method.tubelet_support_rows(177, offset, public)) for offset in range(5)
    ) == 225
    assert receipt["full_singleton_interpretation"].startswith("public geometry")
    assert "target_margin" not in receipt


def test_cached_public_codebook_has_no_sample_message_or_truth_input():
    public = small_public()
    first = method.patch_direction("key", 0, 0, 0, public)
    again = method.patch_direction("key", 0, 0, 0, public)
    wrong = method.patch_direction("wrong", 0, 0, 0, public)
    assert first is again
    assert first.flags.writeable is False
    assert not np.array_equal(first, wrong)
    assert np.isclose(np.sum(first.astype(np.float64) ** 2), 1.0)
    receipt = method.public_receipt(public)
    assert "sample_id" in receipt["codebook_inputs"]
    assert "message" in receipt["codebook_inputs"]
    assert "true crop start" in receipt["codebook_inputs"]
    assert method.patch_direction("key", 2, 0, 0, public).shape[0] == 1


def test_writer_float32_applied_delta_and_adopted_partial_score():
    public = small_public(formal_time=True)
    source = np.zeros((181, 4, 4, 4), dtype=np.float32)
    written, receipt = method.apply_projection_margin(source, "key", public)
    assert written.dtype == np.float32
    assert receipt["projection_rows"] == 46
    assert receipt["raw_target_delta_l2"] == pytest.approx(46.0**0.5)
    assert receipt["applied_float32_delta_l2"] == pytest.approx(46.0**0.5, rel=2e-6)
    assert all(
        row["signed_projection_after"] == pytest.approx(1.0, abs=2e-6)
        for row in receipt["rows"]
    )
    full = method.score_received_latent(written, "key", public)
    crop = method.score_received_latent(written[1:178].copy(), "key", public)
    assert full["counts"] == {
        "candidate_scores": 1,
        "candidate_tubelet_rows": 46,
        "failed_candidates": 0,
    }
    assert full["summary"]["canonical_offset"] == 0
    assert full["summary"]["full_singleton_geometry_only"] is True
    assert crop["counts"] == {
        "candidate_scores": 5,
        "candidate_tubelet_rows": 225,
        "failed_candidates": 0,
    }
    assert crop["summary"]["canonical_offset"] == 1
    assert crop["summary"]["sync_accepted"] is False
    true_rows = [row for row in crop["local_rows"] if row["source_offset"] == 1]
    assert all(row["rho"] > 0.0 and row["q"] is not None for row in true_rows)
    first = true_rows[0]
    assert first["observed_frames"] == 3
    assert first["rho"] == pytest.approx(0.75, abs=2e-6)
    assert first["q"] == pytest.approx(1.0, abs=2e-6)




def test_m05_target_formula_is_not_half_old_delta_and_m1_default_is_stable():
    public = small_public(formal_time=True)
    source = np.zeros((181, 4, 4, 4), dtype=np.float32)
    for tubelet, q_before in enumerate((0.25, 0.75, 1.25)):
        start = tubelet * public.tubelet_length
        stop = start + public.tubelet_length
        direction = method.patch_direction("key", tubelet, 0, 0, public)
        sign = method.sync_sign("key", tubelet, public)
        source[start:stop] += np.float32(q_before * sign) * direction
    original = source.copy()
    default_m1, default_receipt = method.apply_projection_margin(
        source.copy(), "key", public
    )
    explicit_m1, m1_receipt = method.apply_projection_margin(
        source.copy(), "key", public, target_margin=1.0
    )
    m05, m05_receipt = method.apply_projection_margin(
        source.copy(), "key", public, target_margin=0.5
    )
    assert np.array_equal(source, original)
    assert np.array_equal(default_m1, explicit_m1)
    assert default_receipt == m1_receipt
    assert public.method_version == "trajectory-payload-framewise-sync-v1"
    assert [row["signed_projection_before"] for row in m1_receipt["rows"]] == [
        row["signed_projection_before"] for row in m05_receipt["rows"]
    ]
    assert [row["signed_projection_before"] for row in m1_receipt["rows"][:3]] == pytest.approx(
        [0.25, 0.75, 1.25], abs=2e-6
    )
    assert [row["raw_target_delta_l2"] for row in m1_receipt["rows"][:3]] == pytest.approx(
        [0.75, 0.25, 0.0], abs=2e-6
    )
    assert [row["raw_target_delta_l2"] for row in m05_receipt["rows"][:3]] == pytest.approx(
        [0.25, 0.0, 0.0], abs=2e-6
    )
    assert not np.array_equal(m05, original + (explicit_m1 - original) * np.float32(0.5))
    assert m1_receipt["target_margin"] == 1.0
    assert m05_receipt["target_margin"] == 0.5
    with pytest.raises(ValueError, match="two adopted fixed targets"):
        method.apply_projection_margin(source.copy(), "key", public, target_margin=0.75)


def test_nonfinite_retains_all_candidates_and_local_rows():
    public = small_public(formal_time=True)
    received = np.zeros((177, 4, 4, 4), dtype=np.float32)
    received[0, 0, 0, 0] = np.nan
    readout = method.score_received_latent(received, "key", public)
    assert len(readout["candidate_rows"]) == 5
    assert len(readout["local_rows"]) == 225
    assert readout["counts"]["failed_candidates"] >= 1
    assert any(row["status"] == "NONFINITE" for row in readout["local_rows"])
    assert readout["summary"]["sync_accepted"] is False
    json.dumps(readout, allow_nan=False)


class FakeFramewiseVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.18215, force_upcast=True)
        self.encode_batches = []
        self.decode_batches = []

    def encode(self, value):
        self.encode_batches.append(tuple(value.shape))
        pooled = F.avg_pool2d(value.float(), 8)
        latent = torch.cat((pooled, pooled[:, :1]), dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, value):
        self.decode_batches.append(tuple(value.shape))
        sample = F.interpolate(value[:, :3].float(), scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=sample)


def test_framewise_adapter_is_float32_pinned_and_separate_from_wan():
    vae = FakeFramewiseVAE()
    rgb = torch.linspace(0.0, 1.0, 5 * 32 * 32 * 3).reshape(5, 32, 32, 3)
    latent = framewise_autoencoder_kl.encode_rgb_frames(vae, rgb, batch_frames=2)
    assert latent.shape == (5, 4, 4, 4)
    assert latent.dtype == torch.float32
    reconstructed = framewise_autoencoder_kl.decode_rgb_frames(vae, latent, batch_frames=3)
    assert reconstructed.shape == rgb.shape
    receipt = framewise_autoencoder_kl.reconstruction_receipt(
        encoded_frames=5,
        decoded_frames=5,
        batch_frames=2,
        scaling=framewise_autoencoder_kl.scaling_receipt(vae),
    )
    assert receipt["model_id"] == framewise_autoencoder_kl.MODEL_ID
    assert receipt["revision"] == framewise_autoencoder_kl.MODEL_REVISION
    assert receipt["scaling"]["method_coordinate_dtype"] == "torch.float32"
    assert receipt["replaces_wan_native_decoder"] is False


def test_rgb8_source_identity_and_post_mp4_crop(tmp_path):
    shape = (5, 4, 4, 3)
    value = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    path = tmp_path / "source.rgb8"
    path.write_bytes(value.tobytes())
    loaded = rgb8_source.read_rgb8_source(
        path, expected_sha256=hashlib.sha256(value.tobytes()).hexdigest(), shape=shape
    )
    crop, receipt = rgb8_source.crop_received_rgb(
        loaded, source_start=1, source_stop=5
    )
    assert loaded.dtype == torch.uint8
    assert tuple(crop.shape) == (4, 4, 4, 3)
    assert receipt["derived_after_full_mp4_readback"] is True
    assert receipt["second_codec_save"] is False


def _fake_config(tmp_path, source_path, source_sha):
    cfg = copy.deepcopy(runner.load_config())
    cfg["source"].update(
        path=str(source_path),
        sha256=source_sha,
        bytes=181 * 32 * 32 * 3,
        shape=[181, 32, 32, 3],
    )
    cfg["framewise_vae"]["batch_frames"] = 32
    return cfg


def _patch_fake_runtime(monkeypatch, cfg, source_array):
    public = small_public(formal_time=True)
    monkeypatch.setattr(runner.sync_method, "PUBLIC", public)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    frame_vae = FakeFramewiseVAE()
    monkeypatch.setattr(
        runner.framewise, "load_frozen_framewise_vae", lambda *, device: frame_vae
    )

    class FakeWan(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    from runtime.wan import generation

    monkeypatch.setattr(generation, "load_frozen_vae", lambda config, device=None: FakeWan())
    wan_calls = []
    monkeypatch.setattr(
        runner.wan_adapter,
        "reencode_rgb24_readback",
        lambda vae, rgb: wan_calls.append(tuple(rgb.shape)) or torch.zeros((1, 16, 45, 1, 1)),
    )
    payload_calls = []
    truth_bits = runner.payload_method.message_bits(cfg["message"])

    def fake_payload(normalized, key, R):
        payload_calls.append((key, R, tuple(normalized.shape)))
        bits = truth_bits if key == cfg["key"] else [1 - value for value in truth_bits]
        return {
            "status": "READ",
            "decoded_bits": bits,
            "votes": [],
            "truth_used": False,
            "R": R,
            "role": "fake original payload receiver",
        }

    monkeypatch.setattr(runner.payload_method, "payload_read", fake_payload)

    def fake_save_raster(q8, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = q8.numpy().tobytes()
        path.write_bytes(raw)
        return {
            "status": "SAVED",
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "shape": list(q8.shape),
            "dtype": "uint8",
        }

    def fake_roundtrip(raster_path, raster_sha, mp4_path, rgb_path, *, count, event):
        raw = Path(raster_path).read_bytes()
        value = torch.from_numpy(
            np.frombuffer(raw, dtype=np.uint8).reshape(181, 32, 32, 3).copy()
        )
        Path(mp4_path).write_bytes(b"fake-mp4")
        count("mp4_save", False)
        event("mp4", {"status": "SAVED", "path": str(mp4_path), "sha256": hashlib.sha256(b"fake-mp4").hexdigest()})
        count("mp4_save", True)
        count("mp4_probe", False)
        event("probe", {"status": "COMPLETE", "metadata": {"fake": True}})
        count("mp4_probe", True)
        Path(rgb_path).write_bytes(raw)
        count("mp4_readback", False)
        event("rgb24", {"status": "SAVED", "path": str(rgb_path), "sha256": hashlib.sha256(raw).hexdigest(), "shape": list(value.shape)})
        count("mp4_readback", True)
        return value

    monkeypatch.setattr(runner.media, "save_raster", fake_save_raster)
    monkeypatch.setattr(runner.media, "mp4_roundtrip", fake_roundtrip)
    return public, frame_vae, wan_calls, payload_calls


def test_fake_runner_closes_formal_time_roster_and_blind_truth_join(tmp_path, monkeypatch):
    source = (
        np.arange(181 * 32 * 32 * 3, dtype=np.uint32) % 256
    ).astype(np.uint8).reshape(181, 32, 32, 3)
    source_path = tmp_path / "source.rgb8"
    source_path.write_bytes(source.tobytes())
    cfg = _fake_config(tmp_path, source_path, hashlib.sha256(source.tobytes()).hexdigest())
    public, frame_vae, wan_calls, payload_calls = _patch_fake_runtime(
        monkeypatch, cfg, source
    )
    output = tmp_path / "run"
    store = runner.Store(output, create=True)
    runner.media_worker(store, cfg)
    store.data["workers"]["media"] = {"status": "COMPLETE"}
    store.save()
    runner.evaluate(store, cfg)
    assert runner.finish(store, cfg) is True
    final = json.loads((output / "result.json").read_text())
    assert final["status"] == "EXECUTION_COMPLETE"
    assert final["counts"] == {
        "rasters": 4,
        "mp4_save": 4,
        "mp4_read": 4,
        "observations": 8,
        "sync_readouts": 16,
        "sync_candidate_scores": 48,
        "sync_candidate_tubelet_rows": 2168,
        "payload_reads": 16,
        "sync_posthoc": 16,
        "payload_posthoc": 16,
        "quality": 18,
    }
    assert final["calls"]["framewise_writer_encode"] == {"attempted": 1, "completed": 1}
    assert final["calls"]["writer_sync"] == {"attempted": 2, "completed": 2}
    assert final["calls"]["framewise_writer_decode"] == {"attempted": 3, "completed": 3}
    assert final["calls"]["framewise_receiver_encode"] == {"attempted": 8, "completed": 8}
    assert final["calls"]["wan_receiver_encode"] == {"attempted": 8, "completed": 8}
    assert "generation" not in final["calls"]
    assert final["actual_generation_calls"] is False
    assert len(wan_calls) == 8 and len(payload_calls) == 16
    assert all(call[1] == 44 for call in payload_calls)
    assert len(frame_vae.encode_batches) > 1
    assert tuple(final["writers"]) == runner.SYNC_CONDITIONS
    m1_writer = final["writers"]["M1_FRAMEWISE_SYNC"]
    m05_writer = final["writers"]["M05_FRAMEWISE_SYNC"]
    assert m1_writer["target_margin"] == 1.0
    assert m05_writer["target_margin"] == 0.5
    assert m1_writer["input_scaled_latent_sha256"] == m05_writer["input_scaled_latent_sha256"]
    assert m1_writer["independent_copy"] is True
    assert m05_writer["independent_copy"] is True
    assert m1_writer["path"] != m05_writer["path"]
    m1_receipt = runner.read_gzip_json(m1_writer["path"])
    m05_receipt = runner.read_gzip_json(m05_writer["path"])
    assert [row["signed_projection_before"] for row in m1_receipt["rows"]] == [
        row["signed_projection_before"] for row in m05_receipt["rows"]
    ]
    expected_quality_ids = {
        space + "/" + candidate + "_vs_" + reference
        for space in runner.QUALITY_SPACES
        for candidate, reference in runner.QUALITY_PAIRS
    }
    assert set(final["quality"]) == expected_quality_ids
    assert all(row["status"] == "MEASURED" for row in final["quality"].values())
    blind = json.loads((output / "blind_receiver_readouts.json").read_text())
    assert blind["truth_inputs"] is False
    assert "observations" not in blind
    assert final["blind_receiver_sha256"] == hashlib.sha256(
        (output / "blind_receiver_readouts.json").read_bytes()
    ).hexdigest()
    assert all(
        row["full_singleton_geometry_only"]
        for sid, row in final["sync_posthoc"].items()
        if final["observations"][sid.split("/")[0]]["view_truth_join_only"] == "FULL"
    )
    assert all("error" not in row for row in final["sync_posthoc"].values())
    assert all("error" not in row for row in final["payload_posthoc"].values())
    template = final["evidence_provenance"]["candidate_pre_run_template"]
    assert "real-run result is claimed" in template
    assert "real-run result is claimed" not in final["evidence_ceiling"]
    assert "No calibrated threshold or FPR measurement" in final["evidence_ceiling"]
    assert "model or source generalization" in final["evidence_ceiling"]
    assert public.received_frame_counts == (181, 177)


def test_source_failure_retains_fixed_blind_denominator(tmp_path, monkeypatch):
    source_path = tmp_path / "missing.rgb8"
    cfg = copy.deepcopy(runner.load_config())
    cfg["source"]["path"] = str(source_path)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    output = tmp_path / "failed"
    store = runner.Store(output, create=True)
    with pytest.raises(FileNotFoundError) as raised:
        runner.media_worker(store, cfg)
    reason = f"FileNotFoundError: {raised.value}"
    store.failure("SOURCE_READ", raised.value)
    store.data["workers"]["media"] = {"status": "FAILED", "error": reason}
    runner.settle(store, reason)
    runner.evaluate(store, cfg)
    assert runner.finish(store, cfg) is False
    final = json.loads((output / "result.json").read_text())
    snapshot = json.loads((output / "blind_receiver_readouts.json").read_text())
    assert len(snapshot["sync_reads"]) == 16
    assert len(snapshot["payload_reads"]) == 16
    assert all(row["status"] == "NOT_COMPLETED" for row in snapshot["sync_reads"].values())
    assert all(row["status"] == "MISSING_READ" for row in final["sync_posthoc"].values())
    assert all(row["status"] == "MISSING_READ" for row in final["payload_posthoc"].values())
    assert all(row["error"] == reason for row in final["sync_posthoc"].values())
    assert all(row["error"] == reason for row in final["payload_posthoc"].values())
    assert final["failures"] == [{"stage": "SOURCE_READ", "error": reason}]
    assert final["calls"]["source_read"] == {"attempted": 1, "completed": 0}
    assert final["status"] == "INCOMPLETE"
    assert all(
        row["status"] == "NOT_COMPLETED"
        for row in final["writers"].values()
    )
    assert "did not complete" in final["evidence_ceiling"]
    assert "real-run result is claimed" not in final["evidence_ceiling"]


def test_evaluate_clears_only_saved_read_settle_error(tmp_path, monkeypatch):
    cfg = copy.deepcopy(runner.load_config())
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    output = tmp_path / "partial"
    store = runner.Store(output, create=True)
    oid = runner.observation_id("P0_ORIGINAL_RGB", "FULL")
    sid = oid + "/K0"
    sync_row = store.data["sync_reads"][sid]
    runner.dump_gzip_json(
        sync_row["path"],
        {
            "readout": {
                "candidate_rows": [{"source_offset": 0, "score": 0.25}],
                "summary": {"top_offsets": [0]},
            }
        },
    )
    sync_row.update(
        status="SAVED",
        candidate_scores=1,
        candidate_tubelet_rows=46,
    )
    store.data["payload_reads"][sid].update(
        status="READ",
        decoded_bits=runner.payload_method.message_bits(cfg["message"]),
        truth_used=False,
    )
    failure = RuntimeError("real partial media failure")
    reason = "RuntimeError: real partial media failure"
    store.failure("WORKER_MEDIA", failure)
    runner.settle(store, reason)
    runner.evaluate(store, cfg)
    final = json.loads((output / "result.json").read_text())
    assert len(final["sync_posthoc"]) == 16
    assert len(final["payload_posthoc"]) == 16
    assert final["sync_posthoc"][sid]["status"] == "EVALUATED_TRUTH"
    assert final["payload_posthoc"][sid]["status"] == "EVALUATED_TRUTH"
    assert "error" not in final["sync_posthoc"][sid]
    assert "error" not in final["payload_posthoc"][sid]
    missing_sync = [
        row for key, row in final["sync_posthoc"].items() if key != sid
    ]
    missing_payload = [
        row for key, row in final["payload_posthoc"].items() if key != sid
    ]
    assert all(row["status"] == "MISSING_READ" for row in missing_sync)
    assert all(row["error"] == reason for row in missing_sync)
    assert all(row["status"] == "MISSING_READ" for row in missing_payload)
    assert all(row["error"] == reason for row in missing_payload)
    assert final["failures"] == [
        {"stage": "WORKER_MEDIA", "error": reason}
    ]


class _FakeWorkerStdout:
    def __init__(self, *, interrupt):
        self.interrupt = interrupt
        self.closed = False

    def __iter__(self):
        if self.interrupt:
            raise KeyboardInterrupt("operator interrupt")
        return iter(())

    def close(self):
        self.closed = True


class _FakeWorkerProcess:
    def __init__(self, *, interrupt):
        self.stdout = _FakeWorkerStdout(interrupt=interrupt)
        self.interrupt = interrupt
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.interrupt and not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -15 if self.interrupt else 0


def test_run_worker_phase_success_does_not_settle_pending_posthoc(
    tmp_path, monkeypatch
):
    output = tmp_path / "worker-success"
    store = runner.Store(output, create=True)
    child = _FakeWorkerProcess(interrupt=False)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    store, interrupted = runner.run_worker_phase(store)
    assert interrupted is False
    assert store.data["workers"]["media"]["status"] == "COMPLETE"
    assert store.data["failures"] == []
    assert all(row["status"] == "PENDING" for row in store.data["sync_posthoc"].values())
    assert all("error" not in row for row in store.data["sync_posthoc"].values())
    reloaded = runner.Store(output)
    assert reloaded.data["workers"]["media"]["status"] == "COMPLETE"
    assert reloaded.data["workers"]["media"]["returncode"] == 0
    assert all(
        row["status"] == "PENDING"
        for row in reloaded.data["sync_posthoc"].values()
    )
    assert child.stdout.closed is True


def test_run_worker_phase_interrupt_settles_full_roster_with_real_reason(
    tmp_path, monkeypatch
):
    output = tmp_path / "worker-interrupt"
    store = runner.Store(output, create=True)
    child = _FakeWorkerProcess(interrupt=True)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    store, interrupted = runner.run_worker_phase(store)
    assert interrupted is True
    assert store.data["workers"]["media"]["status"] == "FAILED"
    assert store.data["failures"] == [
        {
            "stage": "WORKER_MEDIA",
            "error": "RuntimeError: KeyboardInterrupt: operator interrupt",
        }
    ]
    assert all(row["status"] == "NOT_COMPLETED" for row in store.data["sync_reads"].values())
    assert all(
        row["error"] == "KeyboardInterrupt: operator interrupt"
        for row in store.data["sync_posthoc"].values()
    )
    assert all(
        row["status"] == "NOT_COMPLETED"
        and row["error"] == "KeyboardInterrupt: operator interrupt"
        for row in store.data["writers"].values()
    )
    assert child.terminated is True
    assert child.stdout.closed is True


def test_unpublished_m05_notebook_is_static_and_stops_before_output_creation(tmp_path):
    from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as notebook_builder
    from scripts import build_video_trajectory_payload_framewise_sync_v1_notebook as published_builder

    new_logged = notebook_builder.SETUP[
        notebook_builder.SETUP.index("def logged"):
        notebook_builder.SETUP.index("write_json(OUTPUT/'setup_receipt.json'")
    ]
    published_logged = published_builder.SETUP[
        published_builder.SETUP.index("def logged"):
        published_builder.SETUP.index("write_json(OUTPUT/'setup_receipt.json'")
    ]
    assert new_logged == published_logged
    path = notebook_builder.build(None, tmp_path / "draft.ipynb")
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 5
    first = "".join(code[0]["source"])
    assert first == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert "force_remount" not in first
    for cell in code:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
    setup = "".join(code[1]["source"])
    assert "SOURCE_SHA = None" in setup
    assert setup.index("if SOURCE_SHA is None") < setup.index("OUTPUT.mkdir")
    binding = notebook["metadata"]["candidate_binding"]
    assert binding["candidate"] == "trajectory-payload-framewise-sync-m05"
    assert binding["source_sha"] is None
    assert binding["status"] == "UNPUBLISHED_DRAFT"
    display = "".join(code[-1]["source"])
    assert display.index("P1-P0 compatibility first") < display.index(
        "Same-run incremental quality"
    )
    assert "M1-P1=" in display and "M05-P1=" in display and "M05-M1=" in display
    assert "sync_posthoc" in display
    assert "FULL geometry only" in display
    assert "All payload rows, including K1" in display


def test_m05_display_keeps_all_missing_sync_slots_and_vote_margins(
    tmp_path, capsys
):
    from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as notebook_builder

    source_sha = "a" * 40
    observations = {}
    sync_posthoc = {}
    payload_posthoc = {}
    payload_reads = {}
    first_sid = None
    for condition in runner.CONDITIONS:
        for view in runner.VIEWS:
            oid = runner.observation_id(condition, view)
            observations[oid] = {
                "condition_truth_join_only": condition,
                "view_truth_join_only": view,
            }
            for key_id in runner.KEY_IDS:
                sid = oid + "/" + key_id
                first_sid = first_sid or sid
                sync_posthoc[sid] = {
                    "status": "MISSING_READ",
                    "error": "forced sync missing",
                }
                payload_posthoc[sid] = {
                    "status": "MISSING_READ",
                    "error": "forced payload missing",
                }
                payload_reads[sid] = {
                    "status": "NOT_COMPLETED",
                    "votes": None,
                }
    payload_reads[first_sid] = {
        "status": "READ",
        "votes": [
            {"ones": 10, "zeros": 2, "count": 12},
            {"ones": 7, "zeros": 5, "count": 12},
        ],
    }
    quality_rows = {
        space + "/" + candidate + "_vs_" + reference: {"status": "NOT_COMPLETED"}
        for space in runner.QUALITY_SPACES
        for candidate, reference in runner.QUALITY_PAIRS
    }
    result = {
        "source_sha": source_sha,
        "fixed_denominator": runner.FIXED,
        "status": "INCOMPLETE",
        "counts": {},
        "calls": {},
        "writers": {
            condition: {"status": "NOT_COMPLETED"}
            for condition in runner.SYNC_CONDITIONS
        },
        "observations": observations,
        "sync_posthoc": sync_posthoc,
        "payload_posthoc": payload_posthoc,
        "payload_reads": payload_reads,
        "quality": quality_rows,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    display_prefix = notebook_builder.DISPLAY.split(
        "from IPython.display import Video,display", 1
    )[0]
    exec(
        display_prefix,
        {
            "json": json,
            "Path": Path,
            "RESULT_PATH": result_path,
            "SOURCE_SHA": source_sha,
            "FIXED": runner.FIXED,
        },
    )
    shown = capsys.readouterr().out
    assert shown.count("CROP sync:") == 8
    assert shown.count("FULL geometry only:") == 8
    assert shown.count("forced sync missing") == 16
    assert shown.count("forced payload missing") == 24
    assert "vote_min= 2 vote_median= 5.0" in shown
    assert "vote_min= None vote_median= None" in shown


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_framewise_decode_rejects_nonfinite_sample_before_clamp(bad_value):
    class NonfiniteDecodeVAE(FakeFramewiseVAE):
        def decode(self, value):
            sample = torch.zeros((int(value.shape[0]), 3, 32, 32), dtype=torch.float32)
            sample.reshape(-1)[0] = bad_value
            return SimpleNamespace(sample=sample)

    latent = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="nonfinite sample before RGB clamp"):
        framewise_autoencoder_kl.decode_rgb_frames(
            NonfiniteDecodeVAE(), latent, batch_frames=1
        )


@pytest.mark.parametrize("error_type", [BrokenPipeError, KeyboardInterrupt])
def test_notebook_logged_reaps_owned_process_group_on_stream_failure(
    tmp_path, error_type
):
    import os
    import signal
    import subprocess
    import sys
    import time
    from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as notebook_builder

    setup = notebook_builder.SETUP
    fragment_start = setup.index("def logged")
    fragment = setup[
        fragment_start:setup.index("write_json(OUTPUT/'setup_receipt.json'", fragment_start)
    ]
    failures = []
    namespace = {
        "OUTPUT": tmp_path,
        "os": os,
        "signal": signal,
        "subprocess": subprocess,
        "time": time,
        "failed": lambda stage, exc: failures.append((stage, type(exc).__name__)),
    }

    def broken_print(*args, **kwargs):
        raise error_type("forced notebook stream failure")

    namespace["print"] = broken_print
    exec(fragment, namespace)
    pid_path = tmp_path / ("pids-" + error_type.__name__ + ".txt")
    grand_ready = tmp_path / ("grand-ready-" + error_type.__name__ + ".txt")
    grand_code = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(grand_ready)!r}).write_text('ready');"
        "time.sleep(60)"
    )
    child_code = f"""
import os
import pathlib
import subprocess
import sys
import time
grand = subprocess.Popen([sys.executable, "-c", {grand_code!r}])
ready = pathlib.Path({str(grand_ready)!r})
deadline = time.monotonic() + 5.0
while not ready.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("grandchild readiness timeout")
    time.sleep(0.01)
pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()) + " " + str(grand.pid))
print("ready", flush=True)
time.sleep(60)
"""
    runner_pid = None
    try:
        with pytest.raises(error_type, match="forced notebook stream failure"):
            namespace["logged"](
                [sys.executable, "-c", child_code],
                "CPU_PROCESS_GROUP_TEST",
            )
        runner_pid, worker_pid = [
            int(value) for value in pid_path.read_text().split()
        ]

        def is_running(pid):
            stat = Path("/proc") / str(pid) / "stat"
            if not stat.exists():
                return False
            fields = stat.read_text().split()
            return len(fields) > 2 and fields[2] != "Z"

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (
            is_running(runner_pid) or is_running(worker_pid)
        ):
            time.sleep(0.05)
        assert not is_running(runner_pid)
        assert not is_running(worker_pid)
        assert failures == [("CPU_PROCESS_GROUP_TEST", error_type.__name__)]
    finally:
        if runner_pid is None and pid_path.exists():
            try:
                runner_pid = int(pid_path.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                runner_pid = None
        if runner_pid is not None:
            try:
                os.killpg(runner_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_notebook_failed_serializes_captured_exception_after_except(tmp_path):
    import os
    import traceback
    from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as notebook_builder

    setup = notebook_builder.SETUP
    fragment_start = setup.index("def write_json")
    fragment = setup[fragment_start:setup.index("def logged", fragment_start)]
    namespace = {
        "OUTPUT": tmp_path,
        "FIXED": {"probe": 1},
        "json": json,
        "os": os,
        "traceback": traceback,
    }
    exec(fragment, namespace)

    def throw_from_helper():
        raise ValueError("captured-after-except")

    try:
        throw_from_helper()
    except ValueError as exc:
        captured = exc
        captured.add_note("secondary cleanup note")

    namespace["failed"]("AFTER_EXCEPT", captured)
    saved = json.loads((tmp_path / "setup_failure.json").read_text())
    assert saved["stage"] == "AFTER_EXCEPT"
    assert saved["error"] == "ValueError: captured-after-except"
    assert saved["fixed_denominator"] == {"probe": 1}
    assert "ValueError: captured-after-except" in saved["traceback"]
    assert "throw_from_helper" in saved["traceback"]
    assert "secondary cleanup note" in saved["traceback"]
    assert "NoneType: None" not in saved["traceback"]

def test_notebook_outer_failure_record_preserves_primary_exception():
    from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as notebook_builder

    for source, expected_stage in (
        (notebook_builder.ENVIRONMENT, "SOURCE_OR_ENVIRONMENT"),
        (notebook_builder.RUN, "RUNNER"),
    ):
        parsed = ast.parse(source)
        outer_handlers = [
            node
            for node in parsed.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "failed"
                for handler in node.handlers
                for call in ast.walk(handler)
            )
        ]
        assert len(outer_handlers) == 1
        outer = outer_handlers[0]
        outer.body = ast.parse("logged(['probe'], 'PROBE')").body
        extracted = ast.fix_missing_locations(
            ast.Module(body=[outer], type_ignores=[])
        )
        primary = BrokenPipeError("original stream failure")
        observed = []

        def logged(*args, **kwargs):
            raise primary

        def failed(stage, exc):
            observed.append((stage, exc))
            raise OSError("failure record unavailable")

        with pytest.raises(BrokenPipeError) as raised:
            exec(compile(extracted, f"<{expected_stage}>", "exec"), {
                "logged": logged,
                "failed": failed,
            })
        assert raised.value is primary
        assert observed == [(expected_stage, primary)]
        notes = getattr(primary, "__notes__", [])
        assert any(
            "failure record error" in note
            and "OSError" in note
            and "failure record unavailable" in note
            for note in notes
        )