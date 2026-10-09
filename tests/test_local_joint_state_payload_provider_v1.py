"""CPU-only fake-posterior checks for the real 50-step provider seam."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_v1 as interface
from runtime.wan import local_joint_state_payload_provider_v1 as provider_module


pytestmark = pytest.mark.unit


SMALL_CARRIER = carrier.CarrierProtocol(
    video_shape=(17, 24, 24, 3), segment_start=1, segment_frames=8, segment_count=2,
    rois=((0, 8, 0, 8), (0, 8, 16, 24), (16, 24, 0, 8), (16, 24, 16, 24)),
)
SMALL_SUPPORT = provider_module.LatentSupport(
    time_range=(1, 3),
    rois=((0, 1, 0, 1), (0, 1, 3, 4), (3, 4, 0, 1), (3, 4, 3, 4)),
)


class FakePosteriorBackend:
    """Deterministic CPU stub; it does not claim to exercise a real VAE."""

    def __init__(self):
        generator = torch.Generator().manual_seed(17)
        self.rgb = 0.45 + 0.1 * torch.rand(SMALL_CARRIER.video_shape, generator=generator)
        self.events = []
        self.encode_count = 0

    def decode_normalized(self, normalized):
        self.events.append(("decode", tuple(normalized.shape), normalized.dtype, normalized.device.type))
        return self.rgb.clone()

    def encode_normalized(self, rgb):
        self.encode_count += 1
        self.events.append(("encode", tuple(rgb.shape), rgb.dtype, rgb.device.type))
        if self.encode_count % 2:
            return torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32)
        return torch.ones((1, 2, 3, 4, 4), dtype=torch.float32)


class FailingModifiedEncodeBackend(FakePosteriorBackend):
    def encode_normalized(self, rgb):
        if self.encode_count == 1:
            self.encode_count += 1
            self.events.append(("encode_failed", tuple(rgb.shape), rgb.dtype, rgb.device.type))
            raise RuntimeError("fixture modified encode failure")
        return super().encode_normalized(rgb)


class FakeWanVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=[1.0, -2.0], latents_std=[2.0, 4.0])
        self.clear_calls = 0
        self.decode_raw = None
        self.encode_video = None

    def clear_cache(self):
        self.clear_calls += 1

    def decode(self, raw, return_dict=False):
        self.decode_raw = raw.detach().clone()
        return (torch.zeros((1, 3, 3, 4, 4), dtype=torch.float32),)

    def encode(self, video):
        self.encode_video = video.detach().clone()
        mean = video.new_tensor(self.config.latents_mean).reshape(1, 2, 1, 1, 1)
        std = video.new_tensor(self.config.latents_std).reshape(1, 2, 1, 1, 1)
        normalized = torch.full((1, 2, 3, 4, 4), 0.25, dtype=torch.float32)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mean + std * normalized))


def test_wan_backend_reuses_fp32_normalized_decode_and_posterior_mode_encode():
    vae = FakeWanVAE()
    backend = provider_module.WanPosteriorBackend(vae)
    normalized = torch.full((1, 2, 3, 4, 4), 0.5, dtype=torch.float32)
    rgb = backend.decode_normalized(normalized)
    expected_raw = torch.empty_like(normalized)
    expected_raw[:, 0] = 2.0
    expected_raw[:, 1] = 0.0
    torch.testing.assert_close(vae.decode_raw, expected_raw)
    assert tuple(rgb.shape) == (3, 4, 4, 3) and rgb.dtype == torch.float32

    encoded = backend.encode_normalized(torch.full((3, 4, 4, 3), 0.75))
    torch.testing.assert_close(encoded, torch.full_like(encoded, 0.25))
    assert vae.encode_video.dtype == torch.float32
    assert float(vae.encode_video.min()) == pytest.approx(0.5)
    assert vae.clear_calls == 4

    bad = FakeWanVAE().double()
    with pytest.raises(ValueError, match="FP32"):
        provider_module.WanPosteriorBackend(bad)


def test_mask_then_cap_uses_fp64_norm_and_keeps_only_declared_support():
    raw = torch.ones((1, 2, 3, 4, 4), dtype=torch.float32)
    capped, receipt = provider_module.mask_and_cap(raw, cap=0.5, support=SMALL_SUPPORT)
    assert receipt["full_posterior_difference_l2"] == pytest.approx((96.0) ** 0.5)
    assert receipt["masked_raw_l2"] == pytest.approx(4.0)
    assert receipt["shrink_scale"] == pytest.approx(0.125)
    assert receipt["capped_l2"] == pytest.approx(0.5)
    assert capped.dtype == torch.float32
    mask = torch.zeros_like(capped, dtype=torch.bool)
    for y0, y1, x0, x1 in SMALL_SUPPORT.rois:
        mask[:, :, 1:3, y0:y1, x0:x1] = True
    assert int(torch.count_nonzero(capped[~mask])) == 0
    assert torch.all(capped[mask] == 0.125)
    with pytest.raises(ValueError, match="explicit"):
        provider_module.mask_and_cap(raw, cap=-1, support=SMALL_SUPPORT)


def test_explicit_provider_runs_only_steps_25_to_49_with_25_decode_and_50_encode_calls():
    backend = FakePosteriorBackend()
    control_step, provider = provider_module.make_local_joint_control_step(
        backend=backend, key="provider-key", message=b"ABCD",
        rho=0.25, cap=0.5, carrier_protocol=SMALL_CARRIER,
        latent_support=SMALL_SUPPORT,
    )
    z = torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32)
    conditional = torch.full_like(z, 0.2)
    unconditional = torch.full_like(z, -0.1)
    rows = []
    for index in range(50):
        velocity, row = control_step(
            z=z, conditional=conditional, unconditional=unconditional,
            sigma=1.0, index=index, total_steps=50,
        )
        assert velocity.shape == z.shape
        rows.append(row)
    assert provider.calls == {
        "disabled": 25, "enabled": 25,
        "decode_attempted": 25, "decode_completed": 25,
        "encode_attempted": 50, "encode_completed": 50,
    }
    assert [event[0] for event in backend.events[:6]] == [
        "decode", "encode", "encode", "decode", "encode", "encode"
    ]
    assert all(event[-1] == "cpu" for event in backend.events)
    assert [row["enabled"] for row in rows] == [False] * 25 + [True] * 25
    assert all(row["nominal_provider_operand"]["joint"]["l2"] == 0 for row in rows[:25])
    assert rows[25]["nominal_provider_operand"]["joint"]["l2"] == pytest.approx(0.5)
    assert rows[25]["provider_detail"]["posterior_lift"]["masked_raw_l2"] == pytest.approx(4.0)
    assert rows[25]["requested_budget"] == {
        "rho": 0.25,
        "conditional_clean_l2_cap": 0.5,
        "parameter_status": "caller_supplied_not_experiment_frozen",
    }
    assert rows[24]["provider_detail"]["carrier_math_executed"] is False
    assert rows[25]["provider_detail"]["carrier_math_executed"] is True


def test_rho_and_cap_are_required_without_scientific_defaults():
    signature = inspect.signature(provider_module.make_local_joint_control_step)
    assert signature.parameters["rho"].default is inspect.Parameter.empty
    assert signature.parameters["cap"].default is inspect.Parameter.empty
    backend = FakePosteriorBackend()
    with pytest.raises(ValueError, match="rho"):
        provider_module.LocalJointPosteriorProvider(backend, "k", b"ABCD", 1.01, 1.0)
    with pytest.raises(ValueError, match="cap"):
        provider_module.LocalJointPosteriorProvider(backend, "k", b"ABCD", 0.5, float("inf"))

    receiver_signature = list(inspect.signature(provider_module.read_received_catalog).parameters)
    assert receiver_signature == ["received_rgb", "key", "public_protocol"]
    rows = provider_module.read_received_catalog(
        backend.rgb[:5], "received-only-key", public_protocol=SMALL_CARRIER
    )
    assert len(rows) == 8 * 4 * 4 and all(row.truth_used is False for row in rows)

    zero_cap_backend = FakePosteriorBackend()
    zero_cap = provider_module.LocalJointPosteriorProvider(
        zero_cap_backend, "k", b"ABCD", 0.0, 0.0,
        carrier_protocol=SMALL_CARRIER, latent_support=SMALL_SUPPORT,
    )
    request = interface.JointControlRequest(
        interface.SamplingStepCoord(25, 50, 1.0),
        provider_module.nominal_video_windows(SMALL_CARRIER), {}, {},
        {"rho": 0.0, "cap": 0.0},
    )
    shape = torch.zeros((1, 2, 3, 4, 4))
    result = zero_cap(request, shape, shape, shape)
    assert result.enabled is True and int(torch.count_nonzero(result.joint_delta)) == 0
    assert zero_cap.calls == {
        "disabled": 0, "enabled": 1,
        "decode_attempted": 1, "decode_completed": 1,
        "encode_attempted": 2, "encode_completed": 2,
    }


def test_backend_failure_records_attempted_vs_completed_stage_without_retry():
    backend = FailingModifiedEncodeBackend()
    provider = provider_module.LocalJointPosteriorProvider(
        backend, "failure-key", b"ABCD", 0.2, 0.5,
        carrier_protocol=SMALL_CARRIER, latent_support=SMALL_SUPPORT,
    )
    request = interface.JointControlRequest(
        interface.SamplingStepCoord(25, 50, 1.0),
        provider_module.nominal_video_windows(SMALL_CARRIER), {}, {},
        {"rho": 0.2, "cap": 0.5},
    )
    shape = torch.zeros((1, 2, 3, 4, 4))
    with pytest.raises(RuntimeError, match="fixture modified encode failure"):
        provider(request, shape, shape, shape)
    assert provider.calls == {
        "disabled": 0, "enabled": 1,
        "decode_attempted": 1, "decode_completed": 1,
        "encode_attempted": 2, "encode_completed": 1,
    }
    assert provider.failures == [{
        "sampling_step": 25,
        "stage": "encode_modified",
        "reason": "RuntimeError: fixture modified encode failure",
        "retry": False,
    }]
    assert [event[0] for event in backend.events] == ["decode", "encode", "encode_failed"]
