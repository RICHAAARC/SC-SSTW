"""B2 terminal receiver-gradient control at native same-history T49."""
from __future__ import annotations

import copy

from main.tube_state.rgb_dct_t49_carrier import classify_unit_response
from runtime.wan.rgb_dct_terminal_gradient_backend import (
    WanTerminalGradientBackend, _move_scheduler,
)


class WanB2TerminalReceiverBackend(WanTerminalGradientBackend):
    """Reuse the B1 full-RGB frozen-VAE gradient; act on terminal z50 at T49."""

    def prepare_off(self, artifact_dir):
        from runtime.wan import trajectory

        receipt = super().prepare_off(artifact_dir)
        # B2 differentiates the actual OFF z50. There is no T46 forecast or VJP.
        self.forecast = dict(
            terminal=self.off_terminal,
            terminal_fingerprint=trajectory.fingerprint(self.off_terminal),
        )
        return receipt

    def terminal_cotangent(self, key):
        import torch
        from runtime.wan import trajectory

        off_rgb, proxy = super().terminal_cotangent(key)
        full = trajectory.measures(self.cotangent)
        gradient = self.cotangent.clone()
        gradient[:, :, 0] = 0
        gradient[:, :, 45] = 0
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite masked terminal z50 gradient")
        masked = trajectory.measures(gradient)
        if masked["support_rms"] == 0:
            raise RuntimeError("ZERO_TERMINAL_Z50_GRADIENT")
        self.gradient = gradient
        self.terminal_gradient_receipt = dict(
            source="full_181_frame_frozen_fp32_vae_loss_to_off_z50",
            full_gradient=full, masked_gradient=masked,
            masked_gradient_fingerprint=trajectory.fingerprint(gradient),
            support_time=[1, 45],
            loss=proxy["loss"], q=proxy["q"],
        )
        return off_rgb, proxy

    def control_terminal49_receiver(self, target: float):
        """Map desired -dL/dz50 through the original SINGLE49 native T49 step."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER" or self.gradient is None:
            raise RuntimeError("terminal receiver gradient unavailable")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        snapshot_cpu = self.snapshots[49]
        before = trajectory.fingerprint(vars(snapshot_cpu))
        snapshot = _move_scheduler(copy.deepcopy(snapshot_cpu), device)
        z = self.nodes[49]["z"].to(device=device, dtype=torch.float32)
        v = self.nodes[49]["v"].to(device=device, dtype=torch.float32)
        raw = -self.gradient.to(device=device, dtype=torch.float32)
        sigma = float(snapshot.sigmas[49])
        support = trajectory.measures(raw)["support_rms"]
        if support <= 0 or sigma <= 0 or float(snapshot.sigmas[50]) != 0:
            raise ValueError("valid terminal gradient/sigma required")
        unit = raw / support
        off = self.off_terminal.to(device=device, dtype=torch.float32)
        probe, _ = trajectory.zero_step(
            snapshot, z, v - unit / sigma, 49, self.count,
            "unit_response_probe_step",
        )
        unit_response = trajectory.measures(probe - off)
        state = classify_unit_response(unit_response["support_rms"])
        metrics = dict(
            index=49, history_fingerprint=before,
            gradient_fingerprint=trajectory.fingerprint(self.gradient),
            desired_direction=trajectory.measures(raw),
            unit_D=unit_response, target_D_support_rms=target,
            sigma49=sigma,
        )
        if state != "READY":
            return state, None, metrics
        epsilon = target / unit_response["support_rms"]
        terminal, controlled_history = trajectory.zero_step(
            snapshot, z, v - epsilon * unit / sigma, 49, self.count,
            "scheduler_step",
        )
        delta = terminal - off
        actual = trajectory.measures(delta)
        relative = abs(actual["support_rms"] - target) / target
        if relative > 2e-5 or trajectory.fingerprint(vars(snapshot_cpu)) != before:
            raise RuntimeError("TERMINAL49 native response mismatch/history pollution")
        # The linear prediction and actual finite loss change are diagnostics only.
        dot = float(torch.sum(
            self.gradient.to(device=device, dtype=torch.float64)
            * delta.to(dtype=torch.float64)
        ).item())
        metrics.update(
            epsilon=epsilon, actual_D=actual,
            response_relative_error=relative,
            g_dot_actual_terminal_delta=dot,
            off_terminal_fingerprint=trajectory.fingerprint(off),
            terminal_fingerprint=trajectory.fingerprint(terminal),
            final_history_fingerprint=trajectory.fingerprint(vars(controlled_history)),
            terminal_from_off=actual,
        )
        if not bool(torch.isfinite(torch.as_tensor(dot))) or dot >= 0:
            return "NON_DESCENT_DIRECTION", None, metrics
        self.terminals["TERMINAL49_RECEIVER"] = terminal.detach().cpu()
        return "READY", self.terminals["TERMINAL49_RECEIVER"], metrics
