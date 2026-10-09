"""Runtime adapter for externally supplied local joint state/payload control."""
from __future__ import annotations

import math

from main.tube_state import local_joint_state_payload_v1 as interface


def _delta_measurement(value):
    import torch
    flat = value.detach().double().reshape(-1)
    return dict(
        shape=list(value.shape),
        rms=float(flat.square().mean().sqrt()),
        l2=float(flat.square().sum().sqrt()),
        abs_max=float(flat.abs().max()),
        nonzero=int(torch.count_nonzero(flat)),
    )


def _validated_delta(value, reference, name):
    import torch
    if not torch.is_tensor(value) or value.shape != reference.shape:
        raise ValueError(f"{name} must match the current latent shape")
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")
    return value


def apply_joint_control(z, conditional, unconditional, request, result, *, guidance=5.0):
    """Apply the provider's explicit joint clean-estimate delta before FP32 CFG."""
    import torch
    if not isinstance(request, interface.JointControlRequest):
        raise TypeError("public joint-control request required")
    if not isinstance(result, interface.JointControlResult):
        raise TypeError("public joint-control result required")
    if request.sampling.total_steps != 50:
        raise ValueError("the current Wan adapter requires the native 50-step history")
    if not math.isfinite(float(guidance)):
        raise ValueError("finite CFG guidance required")
    z = z.detach().float()
    conditional = conditional.detach().float()
    unconditional = unconditional.detach().float()
    if conditional.shape != z.shape or unconditional.shape != z.shape:
        raise ValueError("conditional branches must match the current latent")
    joint = _validated_delta(result.joint_delta, z, "joint_delta")
    if not result.enabled and int(torch.count_nonzero(joint)):
        raise ValueError("disabled external control must return a zero joint_delta")
    state = (_validated_delta(result.state_delta,z,"state_delta")
             if result.state_delta is not None else None)
    payload = (_validated_delta(result.payload_delta,z,"payload_delta")
               if result.payload_delta is not None else None)
    sigma = float(request.sampling.sigma)
    clean_before = z - sigma * conditional
    baseline_velocity = unconditional + float(guidance) * (conditional - unconditional)
    controlled = conditional - joint / sigma
    clean_after = z - sigma * controlled
    velocity = unconditional + float(guidance) * (controlled - unconditional)
    if not all(bool(torch.isfinite(value).all()) for value in (
            clean_before,clean_after,baseline_velocity,velocity)):
        raise FloatingPointError("nonfinite injected joint-control arithmetic")
    receipt = dict(
        enabled=result.enabled,
        control_kind="external_local_joint_state_payload",
        sampling_step=request.sampling.index,
        total_sampling_steps=request.sampling.total_steps,
        video_windows=[dict(
            window_id=x.window_id,
            time_indices=list(x.time_indices),
            coordinate_space=x.coordinate_space,
            layer=x.layer,
            spatial_region=list(x.spatial_region) if x.spatial_region is not None else None,
        ) for x in request.windows],
        requested_budget=dict(request.requested_budget),
        nominal_provider_operand=dict(
            coordinate="conditional_clean_estimate",
            joint=_delta_measurement(joint),
            state_diagnostic=_delta_measurement(state) if state is not None else None,
            payload_diagnostic=_delta_measurement(payload) if payload is not None else None,
        ),
        realized_control_delta=dict(
            conditional_clean_estimate=_delta_measurement(clean_after-clean_before),
            guided_velocity=_delta_measurement(velocity-baseline_velocity),
            guidance=float(guidance),
            arithmetic_dtype="torch.float32",
        ),
        provider_composition=result.composition,
        provider_composition_unverified=True,
        provider_detail=dict(result.detail),
        branch_dtype="float32",
        cfg_dtype="float32",
    )
    return velocity.detach(),receipt


def make_control_step(*, windows, state_spec, payload_spec, requested_budget,
                      provider, guidance=5.0):
    """Bind one provider for all native steps; no carrier or budget is defaulted."""
    windows = tuple(windows)
    if not callable(provider):
        raise TypeError("joint-control provider must be callable")
    # Validate the static window denominator before entering a model loop.
    interface.JointControlRequest(
        sampling=interface.SamplingStepCoord(0,50,1.0),
        windows=windows,state_spec=state_spec,payload_spec=payload_spec,
        requested_budget=requested_budget,
    )

    def control_step(*, z, conditional, unconditional, sigma, index, total_steps):
        request = interface.JointControlRequest(
            sampling=interface.SamplingStepCoord(index,total_steps,sigma),
            windows=windows,state_spec=state_spec,payload_spec=payload_spec,
            requested_budget=requested_budget,
        )
        result = provider(request,z,conditional,unconditional)
        return apply_joint_control(
            z,conditional,unconditional,request,result,guidance=guidance)

    return control_step


def collect_window_observations(received, public_protocol, key, windows, observe_window):
    """Collect one received-only record per declared window in input order.

    The observer receives only the received object, public protocol, current
    key, and one public window coordinate.  Returning ``None`` records a
    missing observation; exceptions are retained as failed observations.
    """
    if not callable(observe_window):
        raise TypeError("window observer must be callable")
    rows=[]
    for window in tuple(windows):
        if not isinstance(window,interface.VideoWindowCoord):
            raise TypeError("public video-window coordinate required")
        try:
            row=observe_window(received,public_protocol,key,window)
            if row is None:
                row=interface.WindowObservation(
                    window=window,state_status="MISSING",payload_status="MISSING",
                    state_error="observer returned no observation",
                    payload_error="observer returned no observation")
            elif not isinstance(row,interface.WindowObservation) or row.window != window:
                raise ValueError("observer returned a mismatched window record")
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            row=interface.WindowObservation(
                window=window,state_status="FAILED",payload_status="FAILED",
                state_error=error,payload_error=error)
        rows.append(row)
    return rows
