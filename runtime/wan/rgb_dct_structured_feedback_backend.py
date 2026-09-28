"""No-gradient native UniPC tail probes for structured RGB-DCT feedback."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

from runtime.wan.rgb_dct_terminal_gradient_backend import (
    WanTerminalGradientBackend, _move_scheduler,
)


def _assert_finite_history(scheduler) -> None:
    """Check dynamic native history for every computed, including unscored, tail."""
    import numpy as np
    import torch
    for name in ("sigmas", "timesteps", "model_outputs", "timestep_list"):
        value = getattr(scheduler, name, None)
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            if item is None:
                continue
            if torch.is_tensor(item):
                valid = bool(torch.isfinite(item).all())
            elif isinstance(item, np.ndarray):
                valid = bool(np.isfinite(item).all())
            elif isinstance(item, (float, int)):
                valid = math.isfinite(item)
            else:
                valid = True
            if not valid:
                raise FloatingPointError(f"nonfinite scheduler {name}")


class WanStructuredFeedbackBackend(WanTerminalGradientBackend):
    def prepare_off(self, artifact_dir: Path) -> dict:
        import torch
        from runtime.wan import trajectory
        from runtime.wan.generation import prepare_generation

        if self.phase != "INITIAL":
            raise RuntimeError("OFF generation may run once")
        started = time.perf_counter()
        self.count("generation", False)
        self.pipe, initial, self.prompt, self.negative, self.dtype = prepare_generation(
            self.config, load_vae=False)
        scheduler = self.pipe.scheduler
        trajectory.validate_scheduler(scheduler)
        z = initial.detach()
        initial_fp = trajectory.fingerprint(z)
        guidance = self.config["generation"]["guidance_scale"]
        with torch.no_grad():
            for index in range(50):
                def count_transformer(kind, completed):
                    if kind != "transformer":
                        raise RuntimeError("unexpected OFF transformer call")
                    self.count("transformer_prefix", completed)
                v = trajectory.velocity(self.pipe, z, scheduler, self.prompt,
                                        self.negative, self.dtype, guidance,
                                        index, count_transformer)
                if index in (46, 47, 48, 49):
                    self.nodes[index] = dict(z=z.detach().cpu().clone(),
                                             v=v.detach().cpu().clone())
                    self.snapshots[index] = _move_scheduler(
                        copy.deepcopy(scheduler), torch.device("cpu"))
                def count_step(kind, completed):
                    if kind != "scheduler_step":
                        raise RuntimeError("unexpected OFF scheduler call")
                    self.count("scheduler_prefix", completed)
                z = trajectory.native_step(scheduler, z, v, index, count_step)
        if scheduler.step_index != 50:
            raise RuntimeError("OFF trajectory incomplete")
        self.off_terminal = z.detach().cpu().clone()
        self.count("generation", True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for index in (46, 47, 48, 49):
            for kind, value in self.nodes[index].items():
                path = artifact_dir / f"{kind}{index}.pt"
                torch.save(value, path)
                artifacts[f"{kind}{index}"] = dict(path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    fingerprint=trajectory.fingerprint(value))
            path = artifact_dir / f"scheduler{index}.pt"
            torch.save(self.snapshots[index], path)
            artifacts[f"scheduler{index}"] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fingerprint=trajectory.fingerprint(vars(self.snapshots[index])))
        path = artifact_dir / "off_terminal.pt"
        torch.save(self.off_terminal, path)
        artifacts["off_terminal"] = dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fingerprint=trajectory.fingerprint(self.off_terminal))
        self.phase = "TRANSFORMER"
        self._phase_started = time.perf_counter()
        return dict(initial_noise_fingerprint=initial_fp,
            scheduler_class=type(scheduler).__name__,
            scheduler_config=dict(scheduler.config),
            off_terminal_fingerprint=trajectory.fingerprint(self.off_terminal),
            transformer_dtype=str(self.dtype), artifacts=artifacts,
            phase=self._phase_receipt("OFF_GENERATION", started))

    def build_basis(self, key: bytes):
        from main.tube_state.rgb_dct_t49_carrier import apply_carrier
        from main.tube_state.rgb_dct_t49_carrier import lift_direction
        from main.tube_state.rgb_dct_structured_feedback import split_basis
        from runtime.wan import trajectory

        off_rgb = self.decode(self.off_terminal)
        positive_rgb, clipping = apply_carrier(off_rgb, key, +1)
        plus = self.encode(positive_rgb)
        base = self.encode(off_rgb)
        lift = plus - base
        directions, bins = split_basis(lift)
        single_direction, single_geometry = lift_direction(lift)
        return off_rgb, directions, single_direction, dict(clipping=clipping,
            bins=bins, single_geometry=single_geometry,
            plus_fingerprint=trajectory.fingerprint(plus),
            base_fingerprint=trajectory.fingerprint(base),
            direction_fingerprints=[trajectory.fingerprint(x) for x in directions])

    def state(self, index: int):
        if index not in self.nodes:
            raise ValueError("frozen control index required")
        return (self.nodes[index]["z"].clone(),
                copy.deepcopy(self.snapshots[index]),
                self.nodes[index]["v"].clone())

    def velocity(self, index: int, z, scheduler):
        import torch
        from runtime.wan import trajectory
        def count(kind, completed):
            if kind != "transformer":
                raise RuntimeError("unexpected controlled transformer call")
            self.count("transformer_feedback", completed)
        with torch.no_grad():
            return trajectory.velocity(
                self.pipe, z, scheduler, self.prompt, self.negative,
                self.dtype, self.config["generation"]["guidance_scale"],
                index, count)

    def rollout(self, index: int, z_cpu, scheduler_cpu, v_cpu,
                velocity_delta=None, baseline_next=None):
        """Independent native current step and genuine no-grad CFG tail."""
        import torch
        from runtime.wan import trajectory
        if self.phase != "TRANSFORMER":
            raise RuntimeError("tail requires Transformer phase")
        before = trajectory.fingerprint(vars(scheduler_cpu))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scheduler = _move_scheduler(copy.deepcopy(scheduler_cpu), device)
        z = z_cpu.to(device=device, dtype=torch.float32)
        v = v_cpu.to(device=device, dtype=torch.float32)
        if velocity_delta is not None:
            v = v + velocity_delta.to(device=device, dtype=torch.float32)
        def count_step(kind, completed):
            if kind != "scheduler_step":
                raise RuntimeError("unexpected controlled scheduler call")
            self.count("scheduler_feedback", completed)
        with torch.no_grad():
            first_z = trajectory.native_step(scheduler, z, v, index,
                                             count_step)
            _assert_finite_history(scheduler)
            first_state = _move_scheduler(copy.deepcopy(scheduler), torch.device("cpu"))
            immediate = (trajectory.measures(first_z - baseline_next.to(device))
                         if baseline_next is not None else None)
            terminal = first_z
            for step in range(index + 1, 50):
                velocity = self.velocity(step, terminal, scheduler)
                terminal = trajectory.native_step(scheduler, terminal,
                                                  velocity, step, count_step)
            _assert_finite_history(scheduler)
        if trajectory.fingerprint(vars(scheduler_cpu)) != before:
            raise RuntimeError("tail polluted frozen UniPC history")
        return dict(terminal=terminal.detach().cpu(),
                    next_z=first_z.detach().cpu(), next_scheduler=first_state,
                    immediate=immediate, source_history_fingerprint=before,
                    immediate_tensor=((first_z - baseline_next.to(device)).detach().cpu()
                                      if baseline_next is not None else None),
                    next_history_fingerprint=trajectory.fingerprint(vars(first_state)),
                    final_history_fingerprint=trajectory.fingerprint(vars(scheduler)),
                    terminal_fingerprint=trajectory.fingerprint(terminal))

    def score_terminal(self, terminal, key: bytes):
        from main.tube_state import rgb_dct_group_consistency as receiver
        if self.phase != "VAE":
            raise RuntimeError("score requires VAE phase")
        rgb = self.decode(terminal)
        self.count("memory_receiver_score", False)
        scored = receiver.score_rgb(rgb, key)
        if scored["status"] != "SCORED" or scored["frames_used"] != 181:
            raise RuntimeError(f"invalid terminal RGB score: {scored['reason']}")
        self.count("memory_receiver_score", True)
        return dict(q=[float(x) for x in scored["group_scores"]],
                    C=scored["positive_groups"], score=scored["score"])
