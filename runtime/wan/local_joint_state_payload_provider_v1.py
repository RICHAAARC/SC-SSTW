"""Posterior-difference provider for the adopted local joint carrier.

The provider never loads a model or chooses a device.  A caller supplies the
posterior backend and the still-unfrozen ``rho`` and cap values explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_v1 as interface
from runtime.wan import local_joint_state_payload_v1 as control_adapter
from runtime.wan import vae as vae_adapter


@dataclass(frozen=True)
class LatentSupport:
    time_range: tuple[int, int]
    rois: tuple[tuple[int, int, int, int], ...]

    def validate_for(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 5 or shape[0] != 1:
            raise ValueError("posterior difference must be [1,C,T,H,W]")
        _, _, time, height, width = shape
        t0, t1 = self.time_range
        if not 0 <= t0 < t1 <= time:
            raise ValueError("latent time support is outside the posterior")
        if len(self.rois) != 4:
            raise ValueError("V1 requires four latent support ROIs")
        occupied: set[tuple[int, int]] = set()
        for y0, y1, x0, x1 in self.rois:
            if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                raise ValueError("latent ROI is outside the posterior")
            cells = {(y, x) for y in range(y0, y1) for x in range(x0, x1)}
            if occupied.intersection(cells):
                raise ValueError("latent support ROIs must be disjoint")
            occupied.update(cells)


PUBLIC_LATENT_SUPPORT = LatentSupport(
    time_range=(1, 45),
    rois=(
        (8, 16, 12, 20),
        (8, 16, 44, 52),
        (28, 36, 12, 20),
        (28, 36, 44, 52),
    ),
)


class WanPosteriorBackend:
    """Thin wrapper around the existing frozen FP32 Wan VAE adapters."""

    def __init__(self, frozen_vae: Any):
        self.vae = frozen_vae
        self._validate_frozen_fp32()

    def _validate_frozen_fp32(self) -> None:
        import torch

        parameters = tuple(self.vae.parameters())
        if not parameters:
            raise ValueError("Wan posterior backend requires VAE parameters")
        if any(value.dtype != torch.float32 for value in parameters):
            raise ValueError("joint posterior backend requires an FP32 VAE")
        if any(value.requires_grad for value in parameters):
            raise ValueError("joint posterior backend requires a frozen VAE")

    def decode_normalized(self, normalized: Any) -> Any:
        return vae_adapter.decode_normalized_latent(self.vae, normalized.float())

    def encode_normalized(self, rgb: Any) -> Any:
        return vae_adapter.reencode_rgb24_readback(self.vae, rgb.float())


def mask_and_cap(
    raw_difference: Any, *, cap: float, support: LatentSupport,
) -> tuple[Any, dict[str, Any]]:
    """Mask first, then apply one shrink-only FP64 global L2 cap."""
    import torch

    if not torch.is_tensor(raw_difference) or raw_difference.ndim != 5:
        raise ValueError("raw posterior difference must be a five-dimensional tensor")
    if not math.isfinite(float(cap)) or float(cap) < 0:
        raise ValueError("cap must be explicit, finite, and nonnegative")
    raw = raw_difference.detach().float()
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("nonfinite raw posterior difference")
    support.validate_for(tuple(raw.shape))
    masked = torch.zeros_like(raw)
    t0, t1 = support.time_range
    for y0, y1, x0, x1 in support.rois:
        masked[:, :, t0:t1, y0:y1, x0:x1] = raw[:, :, t0:t1, y0:y1, x0:x1]
    full_l2 = float(raw.double().square().sum().sqrt())
    masked_l2 = float(masked.double().square().sum().sqrt())
    scale = min(1.0, float(cap) / masked_l2) if masked_l2 > 0 else 0.0
    capped = (masked * scale).float()
    capped_l2 = float(capped.double().square().sum().sqrt())
    return capped, dict(
        coordinate="normalized_conditional_clean_latent",
        full_posterior_difference_l2=full_l2,
        masked_raw_l2=masked_l2,
        requested_cap_l2=float(cap),
        shrink_scale=scale,
        capped_l2=capped_l2,
        reduction_dtype="torch.float64",
        output_dtype="torch.float32",
        time_range=list(support.time_range),
        spatial_rois=[list(value) for value in support.rois],
        all_channels=True,
        mask_before_cap=True,
        adaptive_amplification=False,
    )


@dataclass
class LocalJointPosteriorProvider:
    backend: Any
    key: str
    message: bytes
    rho: float
    cap: float
    carrier_protocol: carrier.CarrierProtocol = carrier.PUBLIC
    latent_support: LatentSupport = PUBLIC_LATENT_SUPPORT
    calls: dict[str, int] = field(default_factory=lambda: {
        "disabled": 0, "enabled": 0,
        "decode_attempted": 0, "decode_completed": 0,
        "encode_attempted": 0, "encode_completed": 0,
    }, init=False)
    failures: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError("public key must be the original Unicode string")
        carrier.message_fragments(self.message)
        if not math.isfinite(float(self.rho)) or not 0 <= float(self.rho) <= 1:
            raise ValueError("rho must be explicit, finite, and in [0,1]")
        if not math.isfinite(float(self.cap)) or float(self.cap) < 0:
            raise ValueError("cap must be explicit, finite, and nonnegative")
        if not callable(getattr(self.backend, "decode_normalized", None)):
            raise TypeError("posterior backend must provide decode_normalized")
        if not callable(getattr(self.backend, "encode_normalized", None)):
            raise TypeError("posterior backend must provide encode_normalized")

    def _backend_call(self, kind: str, stage: str, index: int, value: Any) -> Any:
        attempted = f"{kind}_attempted"
        completed = f"{kind}_completed"
        self.calls[attempted] += 1
        function = getattr(self.backend, f"{kind}_normalized")
        try:
            result = function(value)
        except Exception as exc:
            self.failures.append(dict(
                sampling_step=index, stage=stage,
                reason=f"{type(exc).__name__}: {exc}", retry=False,
            ))
            raise
        self.calls[completed] += 1
        return result

    def __call__(self, request: interface.JointControlRequest, z: Any,
                 conditional: Any, unconditional: Any) -> interface.JointControlResult:
        import torch

        if request.sampling.total_steps != 50:
            raise ValueError("joint provider requires the native 50-step history")
        index = request.sampling.index
        if index < 25:
            self.calls["disabled"] += 1
            return interface.JointControlResult(
                joint_delta=torch.zeros_like(z, dtype=torch.float32),
                composition="local_paired_energy_posterior_difference",
                enabled=False,
                detail=dict(
                    schedule="native_steps_25_through_49",
                    carrier_math_executed=False,
                    rho=float(self.rho), cap=float(self.cap),
                ),
            )
        self.calls["enabled"] += 1
        sigma = float(request.sampling.sigma)
        clean = z.detach().float() - sigma * conditional.detach().float()
        with torch.inference_mode():
            decoded = self._backend_call("decode", "decode_clean", index, clean)
            modified, carrier_receipt = carrier.apply_carrier_rgb(
                decoded, key=self.key, message=self.message, rho=self.rho,
                protocol=self.carrier_protocol,
            )
            baseline = self._backend_call("encode", "encode_baseline", index, decoded)
            changed = self._backend_call("encode", "encode_modified", index, modified)
        if not torch.is_tensor(baseline) or not torch.is_tensor(changed):
            raise TypeError("posterior backend must return normalized latent tensors")
        if baseline.shape != clean.shape or changed.shape != clean.shape:
            raise ValueError("posterior backend output must match the clean latent shape")
        raw = changed.float() - baseline.float()
        joint, lift_receipt = mask_and_cap(raw, cap=self.cap, support=self.latent_support)
        return interface.JointControlResult(
            joint_delta=joint,
            composition="local_paired_energy_posterior_difference",
            enabled=True,
            detail=dict(
                schedule="native_steps_25_through_49",
                carrier_math_executed=True,
                rho=float(self.rho), cap=float(self.cap),
                carrier=carrier_receipt,
                posterior_lift=lift_receipt,
                backend=type(self.backend).__name__,
                backend_calls=dict(self.calls),
                backend_failures=list(self.failures),
                no_denoiser_backward=True,
                no_tail_backward=True,
            ),
        )


def nominal_video_windows(
    protocol: carrier.CarrierProtocol = carrier.PUBLIC,
) -> tuple[interface.VideoWindowCoord, ...]:
    rows = []
    for segment in range(protocol.segment_count):
        frames = tuple(
            protocol.segment_start + protocol.segment_frames * segment + value
            for value in range(protocol.segment_frames)
        )
        for roi_index, region in enumerate(protocol.rois):
            rows.append(interface.VideoWindowCoord(
                window_id=f"segment{segment}:roi{roi_index}",
                time_indices=frames,
                coordinate_space="output_frame",
                layer="rgb_float_posterior_candidate",
                spatial_region=region,
            ))
    return tuple(rows)


def read_received_catalog(
    received_rgb: Any, key: str, *, public_protocol: carrier.CarrierProtocol = carrier.PUBLIC,
) -> tuple[carrier.RawWindowObservation, ...]:
    """Public receiver entry: received RGB + key + public protocol only."""
    return carrier.observe_catalog(received_rgb, key, protocol=public_protocol)


def make_local_joint_control_step(
    *, backend: Any, key: str, message: bytes, rho: float, cap: float,
    guidance: float = 5.0,
    carrier_protocol: carrier.CarrierProtocol = carrier.PUBLIC,
    latent_support: LatentSupport = PUBLIC_LATENT_SUPPORT,
) -> tuple[Any, LocalJointPosteriorProvider]:
    """Bind the provider to the existing real 50-step injection seam.

    ``rho`` and ``cap`` are required and recorded as caller supplied values;
    this factory does not make them an experiment configuration.
    """
    provider = LocalJointPosteriorProvider(
        backend=backend, key=key, message=message, rho=rho, cap=cap,
        carrier_protocol=carrier_protocol, latent_support=latent_support,
    )
    control_step = control_adapter.make_control_step(
        windows=nominal_video_windows(carrier_protocol),
        state_spec={"kind": "balanced_rm_1_5_public_key_order"},
        payload_spec={"kind": "four_exact_bytes_msb_fragments"},
        requested_budget={
            "rho": float(rho),
            "conditional_clean_l2_cap": float(cap),
            "parameter_status": "caller_supplied_not_experiment_frozen",
        },
        provider=provider,
        guidance=guidance,
    )
    return control_step, provider
