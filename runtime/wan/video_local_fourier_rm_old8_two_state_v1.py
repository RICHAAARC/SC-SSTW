"""Native six-arm OLD8 two-state candidate; real fifty-step generation only."""
from __future__ import annotations
import hashlib
import numpy as np
from main.tube_state import video_local_fourier_rm_old8_two_state_v1_control as method
from main.tube_state import video_local_fourier_rm_old8_two_state_v1 as state
from runtime.wan import trajectory
from runtime.wan.video_temporal_sync_bridge import execution_device_dtype, predict_branches

DIAGNOSTIC_ARRAYS = ("conditional_clean", "pilot_delta", "z_pre", "z_post", "cfg_clean")
def file_array_sha256(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_trajectory(pipe, initial, scheduler, prompt, negative, dtype, arm, key, bits, count, record, *, diagnostic=None):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None: raise ValueError("fresh full native history required")
    z = initial.detach().float().clone(); targets = method.build_targets(z, key, bits, arm)
    before = None; last_projection = None; statuses = []
    for index in range(50):
        c, u = predict_branches(pipe, z, scheduler, prompt, negative, dtype, index, count)
        if index == 25:
            before = dict(z=trajectory.fingerprint(z), history=trajectory.fingerprint(vars(scheduler)),
                          conditional=trajectory.fingerprint(c), unconditional=trajectory.fingerprint(u))
        sigma = float(scheduler.sigmas[index]); sigma_post = float(scheduler.sigmas[index + 1])
        enabled = method.control_enabled(arm, index); capture = {} if diagnostic is not None and index >= 25 else None
        if enabled: count("local_control", False)
        velocity, row = method.guided_velocity(z, c, u, sigma, targets, arm, index, count, diagnostics=capture)
        if enabled: count("local_control", True)
        if capture is not None and "error" not in capture:
            try:
                capture.update(conditional_clean=method.project_tensor(z - sigma * c.float(), targets["basis"]),
                               z_pre=method.project_tensor(z, targets["basis"]),
                               cfg_clean=method.project_tensor(z - sigma * velocity, targets["basis"]))
            except Exception as exc: capture["error"] = f"{type(exc).__name__}: {exc}"
        z = trajectory.native_step(scheduler, z, velocity, index, count, kind="native_step")
        if capture is not None:
            metadata = dict(index=index, sigma_pre=sigma, sigma_post=sigma_post, arm=arm,
                            pilot_enabled=arm in method.STATE_ARMS, sequence=targets["sequence"],
                            method_version=state.PUBLIC.method_version, dtype="float32", shape=[45, 4, 4, 8],
                            role="writer diagnostic only; never blind input")
            arrays = None
            try:
                if "error" in capture: raise RuntimeError(capture["error"])
                capture["z_post"] = method.project_tensor(z, targets["basis"])
                arrays = {name: capture[name].detach().float().cpu().numpy() for name in DIAGNOSTIC_ARRAYS}
                if not all(np.isfinite(value).all() for value in arrays.values()): raise ValueError("nonfinite writer projection")
                metadata.update(status="COMPLETE", closure=capture["closure"])
                if index == 49: last_projection = arrays["z_post"].copy()
            except Exception as exc: metadata.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            try: diagnostic(index, arrays, metadata)
            except Exception as exc: metadata.update(status="FAILED", error=f"diagnostic persistence: {type(exc).__name__}: {exc}")
            statuses.append(metadata["status"]); row["writer_diagnostic"] = dict(status=metadata["status"], index=index, error=metadata.get("error"))
        record(dict(index=index, sigma=sigma, cursor_after=scheduler.step_index, **row))
    if scheduler.step_index != 50: raise RuntimeError("native trajectory incomplete")
    terminal_info = dict(status="NOT_REQUESTED", last_z_post_matches_terminal=None, terminal_projection_sha256=None)
    if diagnostic is not None:
        try:
            projection = method.project_tensor(z, targets["basis"]).detach().float().cpu().numpy()
            if not np.isfinite(projection).all(): raise ValueError("nonfinite terminal diagnostic projection")
            terminal_info.update(status="COMPLETE", last_z_post_matches_terminal=bool(np.array_equal(last_projection, projection)),
                                 terminal_projection_sha256=file_array_sha256(projection))
        except Exception as exc: terminal_info.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    return z.detach().cpu(), dict(before_step25=before, final_history_sha256=trajectory.fingerprint(vars(scheduler)),
                                  terminal_sha256=trajectory.fingerprint(z), transformer_dtype=str(dtype),
                                  state_control_cfg="float32", method_version=state.PUBLIC.method_version,
                                  sequence=targets["sequence"], selection=targets["selection"],
                                  writer_diagnostics=dict(expected=25, completed=statuses.count("COMPLETE"), **terminal_info))
