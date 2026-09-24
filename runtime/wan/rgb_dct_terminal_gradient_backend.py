"""Same-history T46 terminal VJP backend for the fixed RGB-DCT experiment."""
from __future__ import annotations

import copy
import gc
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable

from main.tube_state.rgb_dct_t49_carrier import classify_unit_response


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cuda_memory() -> dict:
    import torch
    if not torch.cuda.is_available():
        return dict(allocated=None, reserved=None, peak_allocated=None, peak_reserved=None)
    return dict(
        allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )


def _move(value: Any, device: Any) -> Any:
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict) and type(value) is dict:
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _move_scheduler(scheduler: Any, device: Any) -> Any:
    from runtime.wan import trajectory
    before = trajectory.fingerprint(vars(scheduler))
    for name, value in list(vars(scheduler).items()):
        setattr(scheduler, name, _move(value, device))
    if trajectory.fingerprint(vars(scheduler)) != before:
        raise RuntimeError("scheduler history changed during device transfer")
    return scheduler


def _graph_velocity(pipe, z, scheduler, prompt, negative, dtype, guidance,
                    index: int, count: Callable[[str, bool], None]):
    """CFG call with input gradients enabled and frozen Transformer parameters."""
    import torch
    timestep = scheduler.timesteps[index].expand(z.shape[0])
    values = []
    for embedding in (prompt, negative):
        count("transformer_replay", False)
        values.append(pipe.transformer(
            hidden_states=z.to(dtype), timestep=timestep,
            encoder_hidden_states=embedding, attention_kwargs=None,
            return_dict=False,
        )[0])
        count("transformer_replay", True)
    result = (values[1] + guidance * (values[0] - values[1])).float()
    if not result.requires_grad or not bool(torch.isfinite(result).all()):
        raise RuntimeError("differentiable CFG chain is detached or nonfinite")
    return result


def _graph_step(scheduler, z, v, index: int,
                count: Callable[[str, bool], None]):
    """Native UniPC step without no_grad or detach, including live history writes."""
    import torch
    cursor = scheduler.step_index
    if cursor != index:
        raise ValueError("differentiable scheduler cursor mismatch")
    count("scheduler_replay", False)
    result = scheduler.step(
        v, scheduler.timesteps[index], z.clone(), return_dict=False
    )[0]
    count("scheduler_replay", True)
    if scheduler.step_index != index + 1 or not result.requires_grad:
        raise RuntimeError("native UniPC replay detached the terminal graph")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite native replay state")
    attached = [item for item in getattr(scheduler, "model_outputs", [])
                if torch.is_tensor(item) and item.requires_grad]
    if not attached:
        raise RuntimeError("UniPC model-output history lost the live graph")
    return result


class WanTerminalGradientBackend:
    """One source worker with separate Transformer and gradient-VAE phases."""

    def __init__(self, config: dict, count: Callable[[str, bool], None],
                 replay_callback=None):
        self.config = config
        self.count = count
        self.replay_callback = replay_callback
        self.pipe = self.prompt = self.negative = self.dtype = None
        self.vae = None
        self.phase = "INITIAL"
        self.nodes: dict[int, dict] = {}
        self.snapshots: dict[int, Any] = {}
        self.off_terminal = None
        self.terminals: dict[str, Any] = {}
        self.forecast = None
        self.cotangent = None
        self.gradient = None
        self.phase_log: list[dict] = []
        self.replay_ledger = None
        self._phase_identity = None
        self._phase_started = time.perf_counter()

    def _phase_receipt(self, name: str, started: float, **extra) -> dict:
        row = dict(name=name, elapsed_seconds=time.perf_counter() - started,
                   cuda=_cuda_memory(), **extra)
        self.phase_log.append(row)
        return row

    def _identity(self) -> dict:
        from runtime.wan import trajectory
        return dict(
            prompt=trajectory.fingerprint(self.prompt),
            negative=trajectory.fingerprint(self.negative),
            snapshots={index: trajectory.fingerprint(vars(snapshot))
                       for index, snapshot in self.snapshots.items()},
            nodes={index: {key: trajectory.fingerprint(value)
                           for key, value in node.items()}
                   for index, node in self.nodes.items()},
            transformer_id=id(self.pipe.transformer),
        )

    def prepare_off(self, artifact_dir: Path) -> dict:
        import torch
        from runtime.wan import trajectory
        from runtime.wan.generation import prepare_generation

        if self.phase != "INITIAL":
            raise RuntimeError("OFF generation may run once")
        started = time.perf_counter()
        self.count("generation", False)
        self.pipe, initial, self.prompt, self.negative, self.dtype = prepare_generation(
            self.config, load_vae=False
        )
        scheduler = self.pipe.scheduler
        trajectory.validate_scheduler(scheduler)
        z = initial.detach()
        initial_fp = trajectory.fingerprint(z)
        guidance = self.config["generation"]["guidance_scale"]
        with torch.no_grad():
            for index in range(50):
                def count_transformer(kind, completed):
                    if kind != "transformer":
                        raise RuntimeError("unexpected OFF call kind")
                    self.count("transformer_prefix", completed)
                v = trajectory.velocity(
                    self.pipe, z, scheduler, self.prompt, self.negative,
                    self.dtype, guidance, index, count_transformer,
                )
                if index in (46, 49):
                    self.nodes[index] = dict(
                        z=z.detach().cpu().clone(), v=v.detach().cpu().clone()
                    )
                    self.snapshots[index] = _move_scheduler(
                        copy.deepcopy(scheduler), torch.device("cpu")
                    )
                def count_step(kind, completed):
                    if kind != "scheduler_step":
                        raise RuntimeError("unexpected OFF scheduler kind")
                    self.count("scheduler_prefix", completed)
                z = trajectory.native_step(scheduler, z, v, index, count_step)
        if scheduler.step_index != 50:
            raise RuntimeError("OFF trajectory did not reach step 50")
        self.off_terminal = z.detach().cpu().clone()
        self.count("generation", True)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for index in (46, 49):
            for kind, value in self.nodes[index].items():
                path = artifact_dir / f"{kind}{index}.pt"
                torch.save(value, path)
                artifacts[f"{kind}{index}"] = dict(
                    path=str(path), sha256=_sha(path),
                    fingerprint=trajectory.fingerprint(value),
                )
            path = artifact_dir / f"scheduler{index}.pt"
            torch.save(self.snapshots[index], path)
            artifacts[f"scheduler{index}"] = dict(
                path=str(path), sha256=_sha(path),
                fingerprint=trajectory.fingerprint(vars(self.snapshots[index])),
            )
        path = artifact_dir / "off_terminal.pt"
        torch.save(self.off_terminal, path)
        artifacts["off_terminal"] = dict(
            path=str(path), sha256=_sha(path),
            fingerprint=trajectory.fingerprint(self.off_terminal),
        )
        self.phase = "TRANSFORMER"
        self._phase_started = time.perf_counter()
        return dict(
            initial_noise_fingerprint=initial_fp,
            scheduler_class=type(scheduler).__name__,
            scheduler_config=dict(scheduler.config),
            off_terminal_fingerprint=trajectory.fingerprint(self.off_terminal),
            transformer_dtype=str(self.dtype), artifacts=artifacts,
            phase=self._phase_receipt("OFF_GENERATION", started),
        )

    def forecast_terminal46(self) -> dict:
        """No-gradient same-history terminal used by the split VJP boundary."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER":
            raise RuntimeError("terminal forecast requires Transformer phase")
        snapshot_cpu = self.snapshots[46]
        source_history = trajectory.fingerprint(vars(snapshot_cpu))
        scheduler = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
        z = self.nodes[46]["z"].to(device="cuda", dtype=torch.float32)
        v46 = self.nodes[46]["v"].to(device="cuda", dtype=torch.float32)
        guidance = self.config["generation"]["guidance_scale"]

        def count_step(kind, completed):
            if kind != "scheduler_step":
                raise RuntimeError("unexpected forecast scheduler kind")
            self.count("scheduler_forecast", completed)

        def count_transformer(kind, completed):
            if kind != "transformer":
                raise RuntimeError("unexpected forecast Transformer kind")
            self.count("transformer_forecast", completed)

        with torch.no_grad():
            z47 = trajectory.native_step(scheduler, z, v46, 46, count_step)
            for index in range(47, 50):
                velocity = trajectory.velocity(
                    self.pipe, z47, scheduler, self.prompt, self.negative,
                    self.dtype, guidance, index, count_transformer,
                )
                z47 = trajectory.native_step(scheduler, z47, velocity, index, count_step)
        terminal = z47.detach().cpu().clone()
        terminal_fp = trajectory.fingerprint(terminal)
        off_fp = trajectory.fingerprint(self.off_terminal)
        history_fp = trajectory.fingerprint(vars(scheduler))
        if terminal_fp != off_fp:
            raise RuntimeError("same-history T46 forecast differs from OFF terminal")
        if trajectory.fingerprint(vars(snapshot_cpu)) != source_history:
            raise RuntimeError("terminal forecast polluted source T46 history")
        self.forecast = dict(
            terminal=terminal, terminal_fingerprint=terminal_fp,
            history_fingerprint=history_fp,
            source_history_fingerprint=source_history,
        )
        return dict(
            terminal_fingerprint=terminal_fp, off_terminal_fingerprint=off_fp,
            final_history_fingerprint=history_fp,
            source_history_fingerprint=source_history,
            exact_terminal_match=True,
        )

    def to_vae_phase(self, name: str) -> dict:
        import torch
        from runtime.wan.generation import load_frozen_vae

        if self.phase != "TRANSFORMER" or self.vae is not None:
            raise RuntimeError("VAE phase requires a live Transformer phase")
        started = time.perf_counter()
        prior = started - self._phase_started
        self._phase_identity = self._identity()
        self.pipe.transformer.to(torch.device("cpu"))
        self.prompt = self.prompt.cpu()
        self.negative = self.negative.cpu()
        if any(parameter.is_cuda for parameter in self.pipe.transformer.parameters()):
            raise RuntimeError("Transformer remained on CUDA")
        gc.collect()
        torch.cuda.empty_cache()
        self.vae = load_frozen_vae(self.config)
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise RuntimeError("VAE parameters are not frozen")
        if any(parameter.dtype != torch.float32 for parameter in self.vae.parameters()):
            raise RuntimeError("gradient VAE must be FP32")
        self.phase = "VAE"
        self._phase_started = time.perf_counter()
        return self._phase_receipt(
            name, started, transformer_resident="cpu", vae_resident="cuda_fp32",
            prior_transformer_phase_elapsed_seconds=prior,
        )

    def to_transformer_phase(self, name: str) -> dict:
        import torch
        from runtime.wan.vae import _clear_cache

        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("Transformer restore requires VAE phase")
        started = time.perf_counter()
        prior = started - self._phase_started
        _clear_cache(self.vae)
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipe.transformer.to(torch.device("cuda"))
        self.prompt = self.prompt.to(torch.device("cuda"))
        self.negative = self.negative.to(torch.device("cuda"))
        if self._identity() != self._phase_identity:
            raise RuntimeError("source states/history changed across VAE phase")
        self.phase = "TRANSFORMER"
        self._phase_started = time.perf_counter()
        return self._phase_receipt(
            name, started, transformer_resident="cuda", vae_resident="absent",
            identity_verified=True, prior_vae_phase_elapsed_seconds=prior,
        )

    def terminal_cotangent(self, key: bytes) -> tuple[Any, dict]:
        """Exact gradient-VAE phase: full RGB proxy then dL/dz_terminal."""
        import torch
        from main.tube_state.rgb_dct_terminal_gradient import (
            torch_proxy, validate_against_numpy,
        )
        from runtime.wan.gradient_checkpointing import ReplayLedger, checkpoint_decode
        from runtime.wan.vae import _clear_cache, _scale_tensors

        if self.phase != "VAE" or self.forecast is None:
            raise RuntimeError("terminal cotangent requires forecast and VAE phase")
        limits = self.config["resources"]["replay_limits_per_source"]
        self.replay_ledger = ReplayLedger(limits, callback=self.replay_callback)
        device = next(self.vae.parameters()).device
        terminal = self.forecast["terminal"].to(
            device=device, dtype=torch.float32
        ).detach().requires_grad_(True)
        mean, std = _scale_tensors(self.vae, terminal)
        _clear_cache(self.vae)
        self.count("vae_gradient_decode", False)
        try:
            decoded = checkpoint_decode(
                self.vae, terminal * std + mean, self.replay_ledger
            )
            rgb = (decoded[0].permute(1, 2, 3, 0) / 2 + 0.5).clamp(0, 1)
            if tuple(rgb.shape) != (181, 320, 512, 3) or rgb.dtype != torch.float32:
                raise RuntimeError("gradient VAE output is not full FP32 RGB")
            proxy = torch_proxy(rgb, key)
            validation = validate_against_numpy(
                rgb, key, proxy,
                rtol=self.config["proxy"]["numpy_rtol"],
                atol=self.config["proxy"]["numpy_atol"],
            )
            self.count("vae_gradient_decode", True)
            self.count("vae_vjp", False)
            cotangent, = torch.autograd.grad(proxy["loss"], terminal)
            self.count("vae_vjp", True)
        finally:
            _clear_cache(self.vae)
        if not bool(torch.isfinite(cotangent).all()):
            raise FloatingPointError("nonfinite VAE terminal cotangent")
        cotangent_measure = float(cotangent.detach().double().square().mean().sqrt())
        if cotangent_measure == 0:
            raise RuntimeError("ZERO_TERMINAL_COTANGENT")
        # BF16 is never exposed to NumPy; media/proxy handoff is explicit FP32.
        off_rgb = rgb.detach().to(dtype=torch.float32, device="cpu").numpy()
        self.cotangent = cotangent.detach().to(device="cpu", dtype=torch.float32)
        receipt = dict(
            loss=float(proxy["loss"].detach().double()),
            q=[float(value) for value in proxy["q"].detach().double().cpu()],
            score=float(proxy["score"].detach().double()),
            terminal_cotangent_rms=cotangent_measure,
            terminal_cotangent_fingerprint=(
                __import__("runtime.wan.trajectory", fromlist=["fingerprint"])
                .fingerprint(self.cotangent)
            ),
            proxy_validation=validation,
            replay=self.replay_ledger.summary(),
        )
        return off_rgb, receipt

    def encode(self, rgb: Any) -> Any:
        import numpy as np
        import torch
        from runtime.wan.vae import reencode_rgb24_readback
        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("encode requires VAE phase")
        self.count("vae_encode", False)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb))
        encoded = reencode_rgb24_readback(self.vae, tensor).detach().to(
            dtype=torch.float32, device="cpu"
        )
        self.count("vae_encode", True)
        return encoded.numpy()

    def decode(self, terminal: Any) -> Any:
        import torch
        from runtime.wan.vae import decode_normalized_latent
        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("decode requires VAE phase")
        self.count("vae_decode", False)
        rgb = decode_normalized_latent(
            self.vae, terminal.to(next(self.vae.parameters()).device)
        ).detach().to(dtype=torch.float32, device="cpu")
        self.count("vae_decode", True)
        return rgb.numpy()

    def terminal_vjp(self) -> dict:
        """Replay T46–49 with attached CFG/UniPC history and apply cotangent."""
        import torch
        from runtime.wan import trajectory
        from runtime.wan.gradient_checkpointing import (
            disable_transformer_checkpointing, enable_transformer_checkpointing,
        )

        if self.phase != "TRANSFORMER" or self.cotangent is None or self.forecast is None:
            raise RuntimeError("terminal VJP requires restored Transformer and cotangent")
        if self.replay_ledger is None:
            raise RuntimeError("replay ledger missing")
        snapshot_cpu = self.snapshots[46]
        source_fp = trajectory.fingerprint(vars(snapshot_cpu))
        scheduler = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
        z = self.nodes[46]["z"].to(device="cuda", dtype=torch.float32)
        v46 = self.nodes[46]["v"].to(
            device="cuda", dtype=torch.float32
        ).detach().requires_grad_(True)
        guidance = self.config["generation"]["guidance_scale"]
        enable_transformer_checkpointing(self.pipe.transformer, self.replay_ledger)
        try:
            terminal = _graph_step(scheduler, z, v46, 46, self.count)
            for index in range(47, 50):
                velocity = _graph_velocity(
                    self.pipe, terminal, scheduler, self.prompt, self.negative,
                    self.dtype, guidance, index, self.count,
                )
                terminal = _graph_step(scheduler, terminal, velocity, index, self.count)
            replay_terminal_fp = trajectory.fingerprint(terminal)
            replay_history_fp = trajectory.fingerprint(vars(scheduler))
            if replay_terminal_fp != self.forecast["terminal_fingerprint"]:
                raise RuntimeError("gradient replay terminal differs from no-gradient forecast")
            if replay_history_fp != self.forecast["history_fingerprint"]:
                raise RuntimeError("gradient replay UniPC history differs from forecast")
            if trajectory.fingerprint(vars(snapshot_cpu)) != source_fp:
                raise RuntimeError("gradient replay polluted source T46 history")
            self.count("tail_vjp", False)
            gradient, = torch.autograd.grad(
                terminal, v46,
                grad_outputs=self.cotangent.to(device=terminal.device, dtype=terminal.dtype),
            )
            self.count("tail_vjp", True)
        finally:
            disable_transformer_checkpointing(self.pipe.transformer)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite full T46 velocity gradient")
        full = trajectory.measures(gradient)
        masked = gradient.detach().clone()
        masked[:, :, 0] = 0
        masked[:, :, 45] = 0
        masked_measure = trajectory.measures(masked)
        if masked_measure["support_rms"] == 0:
            raise RuntimeError("ZERO_T46_GRADIENT")
        self.gradient = masked.to(device="cpu", dtype=torch.float32)
        return dict(
            replay_terminal_fingerprint=replay_terminal_fp,
            forecast_terminal_fingerprint=self.forecast["terminal_fingerprint"],
            replay_history_fingerprint=replay_history_fp,
            forecast_history_fingerprint=self.forecast["history_fingerprint"],
            terminal_exact_match=True, history_exact_match=True,
            full_gradient=full, masked_gradient=masked_measure,
            masked_gradient_fingerprint=trajectory.fingerprint(self.gradient),
            support_time=[1, 45], replay=self.replay_ledger.summary(),
        )

    def control_single49(self, masked_direction: Any, target: float):
        """Original SINGLE49 VAE-lift control, unchanged from the N2 family."""
        import torch
        from runtime.wan import trajectory

        snapshot_cpu = self.snapshots[49]
        before = trajectory.fingerprint(vars(snapshot_cpu))
        snapshot = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
        z = self.nodes[49]["z"].to(device="cuda", dtype=torch.float32)
        v = self.nodes[49]["v"].to(device="cuda", dtype=torch.float32)
        raw = torch.as_tensor(masked_direction, device="cuda", dtype=torch.float32)
        sigma = float(snapshot.sigmas[49])
        support = trajectory.measures(raw)["support_rms"]
        if support <= 0 or sigma <= 0 or float(snapshot.sigmas[50]) != 0:
            raise ValueError("valid SINGLE49 lift/sigma required")
        unit = raw / support
        probe, _ = trajectory.zero_step(
            snapshot, z, v - unit / sigma, 49, self.count,
            "unit_response_probe_step",
        )
        unit_response = trajectory.measures(probe - self.off_terminal.to("cuda"))
        state = classify_unit_response(unit_response["support_rms"])
        metrics = dict(
            index=49, history_fingerprint=before,
            raw_direction=trajectory.measures(raw), unit_D=unit_response,
            target_D_support_rms=target,
        )
        if state != "READY":
            return state, None, metrics
        epsilon = target / unit_response["support_rms"]
        terminal, controlled_history = trajectory.zero_step(
            snapshot, z, v - epsilon * unit / sigma, 49, self.count,
            "scheduler_step",
        )
        actual = trajectory.measures(terminal - self.off_terminal.to("cuda"))
        relative = abs(actual["support_rms"] - target) / target
        if relative > 2e-5 or trajectory.fingerprint(vars(snapshot_cpu)) != before:
            raise RuntimeError("SINGLE49 response mismatch/history pollution")
        metrics.update(epsilon=epsilon, actual_D=actual,
                       response_relative_error=relative,
                       terminal_fingerprint=trajectory.fingerprint(terminal),
                       final_history_fingerprint=trajectory.fingerprint(
                           vars(controlled_history)
                       ),
                       terminal_from_off=trajectory.measures(
                           terminal - self.off_terminal.to("cuda")
                       ))
        self.terminals["SINGLE49"] = terminal.detach().cpu()
        return "READY", self.terminals["SINGLE49"], metrics

    def control_terminal46(self, target: float):
        """Scale the negative full T46 gradient by native same-history response."""
        import torch
        from runtime.wan import trajectory

        if self.gradient is None:
            raise RuntimeError("full T46 gradient unavailable")
        snapshot_cpu = self.snapshots[46]
        before = trajectory.fingerprint(vars(snapshot_cpu))
        z46 = self.nodes[46]["z"].to(device="cuda", dtype=torch.float32)
        v46 = self.nodes[46]["v"].to(device="cuda", dtype=torch.float32)
        raw = -self.gradient.to(device="cuda", dtype=torch.float32)
        support = trajectory.measures(raw)["support_rms"]
        if support <= 0:
            raise RuntimeError("zero negative-gradient direction")
        unit_velocity = raw / support

        def one_step(velocity, kind):
            snapshot = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
            def counter(_kind, completed):
                self.count(kind, completed)
            return trajectory.native_step(snapshot, z46, velocity, 46, counter), snapshot

        normal_z47, _ = one_step(v46, "scheduler_response_shadow")
        probe_z47, _ = one_step(v46 + unit_velocity, "unit_response_probe_step")
        unit_response = trajectory.measures(probe_z47 - normal_z47)
        state = classify_unit_response(unit_response["support_rms"])
        metrics = dict(
            index=46, history_fingerprint=before,
            gradient=trajectory.measures(self.gradient),
            negative_unit_velocity=trajectory.measures(unit_velocity),
            unit_D=unit_response, target_D_support_rms=target,
        )
        if state != "READY":
            return state, None, metrics
        epsilon = target / unit_response["support_rms"]
        controlled_z47, scheduler = one_step(
            v46 + epsilon * unit_velocity, "scheduler_step"
        )
        actual = trajectory.measures(controlled_z47 - normal_z47)
        relative = abs(actual["support_rms"] - target) / target
        if relative > 2e-5:
            raise RuntimeError("TERMINAL46 native response differs from target")
        guidance = self.config["generation"]["guidance_scale"]

        def count_transformer(kind, completed):
            if kind != "transformer":
                raise RuntimeError("unexpected controlled-tail Transformer kind")
            self.count("transformer_control_tail", completed)

        def count_step(kind, completed):
            if kind != "scheduler_step":
                raise RuntimeError("unexpected controlled-tail scheduler kind")
            self.count("scheduler_control_tail", completed)

        terminal = controlled_z47
        with torch.no_grad():
            for index in range(47, 50):
                velocity = trajectory.velocity(
                    self.pipe, terminal, scheduler, self.prompt, self.negative,
                    self.dtype, guidance, index, count_transformer,
                )
                terminal = trajectory.native_step(
                    scheduler, terminal, velocity, index, count_step
                )
        if trajectory.fingerprint(vars(snapshot_cpu)) != before:
            raise RuntimeError("TERMINAL46 control polluted source history")
        terminal = terminal.detach().cpu()
        metrics.update(
            epsilon=epsilon, actual_D=actual,
            response_relative_error=relative,
            controlled_velocity_fingerprint=trajectory.fingerprint(
                (v46 + epsilon * unit_velocity).detach()
            ),
            terminal_fingerprint=trajectory.fingerprint(terminal),
            final_history_fingerprint=trajectory.fingerprint(vars(scheduler)),
            terminal_from_off=trajectory.measures(terminal - self.off_terminal),
        )
        self.terminals["TERMINAL46"] = terminal
        return "READY", terminal, metrics

    def resources(self) -> dict:
        cuda = _cuda_memory()
        return dict(
            receipt_valid=(cuda["peak_allocated"] is not None
                           and cuda["peak_reserved"] is not None),
            cuda=cuda, phase=self.phase,
            current_phase_elapsed_seconds=time.perf_counter() - self._phase_started,
            phases=list(self.phase_log),
            replay=(self.replay_ledger.summary() if self.replay_ledger else None),
        )

    def release(self) -> None:
        import torch
        from runtime.wan.gradient_checkpointing import disable_transformer_checkpointing
        from runtime.wan.vae import _clear_cache
        if self.vae is not None:
            _clear_cache(self.vae)
        self.vae = None
        if self.pipe is not None and self.pipe.transformer is not None:
            disable_transformer_checkpointing(self.pipe.transformer)
            self.pipe.transformer = None
            self.pipe.text_encoder = None
            self.pipe.vae = None
        self.pipe = self.prompt = self.negative = None
        self.cotangent = self.gradient = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.phase = "RELEASED"
