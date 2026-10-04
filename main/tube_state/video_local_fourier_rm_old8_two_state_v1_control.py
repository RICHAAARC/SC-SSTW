"""Six-arm OLD8 two-state writer; original payload and shrink-only update."""
from __future__ import annotations
import math
from main.tube_state import video_local_fourier_rm_control as old
from main.tube_state import video_local_fourier_rm_old8_two_state_v1 as state

ARMS = ("OFF", "PAYLOAD_MULTI", "STATE_A_MULTI", "STATE_B_MULTI", "STATE_AB_MULTI", "STATE_BA_MULTI")
STATE_ARMS = dict(STATE_A_MULTI="AA", STATE_B_MULTI="BB", STATE_AB_MULTI="AB", STATE_BA_MULTI="BA")
message_bits = old.message_bits; payload_read = old.payload_read; project_tensor = old.project_tensor; delta_closure = old.delta_closure


def control_enabled(arm, index):
    if arm not in ARMS or not 0 <= index < 50: raise ValueError("fixed arm/index required")
    return arm != "OFF" and index >= 25


def build_targets(latent, key, bits, arm):
    import torch
    if arm not in ARMS: raise ValueError("fixed arm required")
    targets = old.build_targets(latent, key, bits); sequence = STATE_ARMS.get(arm)
    if sequence is not None:
        signs = torch.tensor(state.composite_signs(key, sequence), device=latent.device, dtype=torch.float32)
        targets["pilot_target"] = signs * state.PUBLIC.alpha; targets["pilot_active"] = signs != 0
    targets.update(carrier=state.PUBLIC.method_version, sequence=sequence,
                   selection=state.selection_receipt(key), blocks=state.PUBLIC.blocks)
    return targets


def guided_velocity(z, conditional, unconditional, sigma, targets, arm, index, count=None, *, diagnostics=None):
    import torch
    if not math.isfinite(float(sigma)) or sigma <= 0: raise ValueError("positive native sigma required")
    z, c, u = z.float(), conditional.float(), unconditional.float(); enabled = control_enabled(arm, index)
    detail = dict(enabled=enabled, state_control_cfg="float32", arm=arm,
                  carrier=targets["carrier"], sequence=targets["sequence"])
    if enabled:
        clean = z - float(sigma) * c
        if count: count("payload_gradient", False)
        dp, pinfo = old.payload.local_delta(clean, targets["payload_target"], targets["payload_mask"])
        if count: count("payload_gradient", True)
        ds = torch.zeros_like(dp); sinfo = None
        if arm in STATE_ARMS:
            if count: count("pilot_gradient", False)
            ds, sinfo = old.pilot_delta(clean, targets)
            if count: count("pilot_gradient", True)
        delta = dp + ds; p2 = float(dp.double().square().sum()); s2 = float(ds.double().square().sum())
        cross = float((dp.double() * ds.double()).sum()); d2 = float(delta.double().square().sum())
        detail.update(payload=pinfo, pilot=sinfo, payload_delta_l2=math.sqrt(p2), pilot_delta_l2=math.sqrt(s2),
                      merged_delta_l2=math.sqrt(d2), merged_delta_rms=math.sqrt(d2 / delta.numel()),
                      cross_inner_product=cross, norm_decomposition_error=d2 - (p2 + s2 + 2 * cross),
                      cfg_clean_delta_l2=5 * math.sqrt(d2), cross_arm_total_budget_matched=False,
                      budget_semantics="independent OLD8 eta696 cap1 shrink-only; same cap is not equal actual budget")
        c = c - delta / float(sigma)
    if diagnostics is not None:
        try:
            ds_applied = ds if enabled and arm in STATE_ARMS else torch.zeros_like(z)
            projected = old.project_tensor(ds_applied, targets["basis"])
            diagnostics.update(pilot_delta=projected.detach(), closure=old.delta_closure(ds_applied, projected, targets))
        except Exception as exc: diagnostics.update(error=f"{type(exc).__name__}: {exc}")
    velocity = u + 5.0 * (c - u)
    if not bool(torch.isfinite(velocity).all()): raise FloatingPointError("nonfinite CFG velocity")
    return velocity.detach(), detail

