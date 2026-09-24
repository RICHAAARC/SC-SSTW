"""Fresh Wan OFF trajectory with isolated T49, T46, and T44/T46 forks."""
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
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())


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


class WanMultiBackend:
    """One source per process; Transformer and VAE occupy GPU in separate phases."""

    def __init__(self, config: dict, count: Callable[[str, bool], None]):
        self.config = config
        self.count = count
        self.pipe = self.prompt = self.negative = self.dtype = None
        self.vae = None
        self.phase = "INITIAL"
        self.phase_log: list[dict] = []
        self.nodes: dict[int, dict] = {}
        self.snapshots: dict[int, Any] = {}
        self.off_terminal = None
        self.terminals: dict[str, Any] = {}
        self.multi46: dict | None = None
        self.initial_noise_fingerprint = None
        self.initial_noise = None
        self._phase_identity = None
        self._phase_started = time.perf_counter()

    def _phase_receipt(self, name: str, start: float, **extra) -> dict:
        row = dict(name=name, elapsed_seconds=time.perf_counter() - start,
                   cuda=_cuda_memory(), **extra)
        self.phase_log.append(row)
        return row

    def _phase_identity_record(self) -> dict:
        """Device-independent identity of live conditioning and full histories."""
        from runtime.wan import trajectory

        return dict(
            initial_noise=trajectory.fingerprint(self.initial_noise),
            prompt=trajectory.fingerprint(self.prompt),
            negative=trajectory.fingerprint(self.negative),
            off_histories={index: trajectory.fingerprint(vars(snapshot))
                           for index, snapshot in self.snapshots.items()},
            multi46_history=(trajectory.fingerprint(vars(self.multi46["snapshot"]))
                             if self.multi46 else None),
            transformer_id=id(self.pipe.transformer),
        )

    def prepare_off(self, artifact_dir: Path) -> dict:
        import torch
        from runtime.wan import trajectory
        from runtime.wan.generation import prepare_generation

        if self.phase != "INITIAL":
            raise RuntimeError("OFF generation may run only once")
        started = time.perf_counter()
        self.count("generation", False)
        self.pipe, initial, self.prompt, self.negative, self.dtype = prepare_generation(
            self.config, load_vae=False)
        scheduler = self.pipe.scheduler
        trajectory.validate_scheduler(scheduler)
        z = initial.detach()
        self.initial_noise_fingerprint = trajectory.fingerprint(z)
        self.initial_noise = z.detach().cpu().clone()
        guidance = self.config["generation"]["guidance_scale"]
        with torch.no_grad():
            for index in range(50):
                v = trajectory.velocity(self.pipe, z, scheduler, self.prompt,
                                        self.negative, self.dtype, guidance, index, self.count)
                if index in (44, 46, 49):
                    self.nodes[index] = dict(z=z.detach().cpu().clone(),
                                             v=v.detach().cpu().clone())
                    self.snapshots[index] = _move_scheduler(copy.deepcopy(scheduler),
                                                            torch.device("cpu"))
                    if self.snapshots[index].step_index != index:
                        raise RuntimeError("frozen OFF scheduler cursor mismatch")
                z = trajectory.native_step(scheduler, z, v, index, self.count)
        if scheduler.step_index != 50:
            raise RuntimeError("normal OFF did not finish all 50 steps")
        self.off_terminal = z.detach().cpu().clone()
        self.count("generation", True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for index in (44, 46, 49):
            for kind, value in self.nodes[index].items():
                path = artifact_dir / f"{kind}{index}.pt"
                torch.save(value, path)
                artifacts[f"{kind}{index}"] = dict(path=str(path), sha256=_sha(path),
                                                    fingerprint=trajectory.fingerprint(value))
            path = artifact_dir / f"scheduler{index}.pt"
            torch.save(self.snapshots[index], path)
            artifacts[f"scheduler{index}"] = dict(
                path=str(path), sha256=_sha(path),
                fingerprint=trajectory.fingerprint(vars(self.snapshots[index])))
        path = artifact_dir / "off_terminal.pt"
        torch.save(self.off_terminal, path)
        artifacts["off_terminal"] = dict(path=str(path), sha256=_sha(path),
                                         fingerprint=trajectory.fingerprint(self.off_terminal))
        path = artifact_dir / "initial_noise.pt"
        torch.save(self.initial_noise, path)
        artifacts["initial_noise"] = dict(path=str(path), sha256=_sha(path),
                                          fingerprint=self.initial_noise_fingerprint)
        self.phase = "TRANSFORMER"
        self._phase_started = time.perf_counter()
        return dict(initial_noise_fingerprint=self.initial_noise_fingerprint,
                    scheduler_class=type(scheduler).__name__,
                    scheduler_config=dict(scheduler.config),
                    off_terminal_fingerprint=trajectory.fingerprint(self.off_terminal),
                    transformer_dtype=str(self.dtype), artifacts=artifacts,
                    phase=self._phase_receipt("OFF_GENERATION", started))

    def to_vae_phase(self, name: str) -> dict:
        import torch
        from runtime.wan.generation import load_frozen_vae

        if self.phase != "TRANSFORMER" or self.vae is not None:
            raise RuntimeError("VAE phase requires a live Transformer phase")
        started = time.perf_counter()
        previous_elapsed = started - self._phase_started
        self._phase_identity = self._phase_identity_record()
        model = self.pipe.transformer
        model.to(torch.device("cpu"))
        self.prompt = self.prompt.cpu()
        self.negative = self.negative.cpu()
        _move_scheduler(self.pipe.scheduler, torch.device("cpu"))
        if any(parameter.is_cuda for parameter in model.parameters()):
            raise RuntimeError("Transformer remained on CUDA during VAE phase")
        gc.collect()
        torch.cuda.empty_cache()
        self.vae = load_frozen_vae(self.config)
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise RuntimeError("frozen VAE has gradient parameters")
        if next(self.vae.parameters()).device.type != "cuda":
            raise RuntimeError("VAE is not resident on CUDA")
        self.phase = "VAE"
        self._phase_started = time.perf_counter()
        return self._phase_receipt(name, started, transformer_resident="cpu",
                                   vae_resident="cuda",
                                   prior_transformer_phase_elapsed_seconds=previous_elapsed)

    def to_transformer_phase(self, name: str) -> dict:
        import torch
        from runtime.wan.vae import _clear_cache

        if self.phase != "VAE":
            raise RuntimeError("Transformer restore requires VAE phase")
        started = time.perf_counter()
        previous_elapsed = started - self._phase_started
        _clear_cache(self.vae)
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipe.transformer.to(torch.device("cuda"))
        self.prompt = self.prompt.to(torch.device("cuda"))
        self.negative = self.negative.to(torch.device("cuda"))
        if next(self.pipe.transformer.parameters()).device.type != "cuda":
            raise RuntimeError("Transformer did not resume on CUDA")
        restored_identity = self._phase_identity_record()
        if restored_identity != self._phase_identity:
            raise RuntimeError("noise/conditioning/scheduler identity changed across VAE phase")
        self.phase = "TRANSFORMER"
        self._phase_started = time.perf_counter()
        return self._phase_receipt(name, started, transformer_resident="cuda",
                                   vae_resident="absent", identity_verified=True,
                                   prior_vae_phase_elapsed_seconds=previous_elapsed)

    def verify_cfg(self, index: int, *, multi: bool = False) -> dict:
        """One same-input numerical CFG check; it never advances scheduler."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER" or index not in (44, 46):
            raise ValueError("restored CFG verification requires T44/T46 Transformer phase")
        node = self.multi46 if multi else self.nodes[index]
        snapshot_cpu = node["snapshot"] if multi else self.snapshots[index]
        if multi and index != 46:
            raise ValueError("MULTI recovery check must use its own T46")
        before = trajectory.fingerprint(vars(snapshot_cpu))
        snapshot = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
        z = node["z"].to(device="cuda", dtype=torch.float32)
        expected = node["v"].to(device="cuda", dtype=torch.float32)
        if snapshot.step_index != index:
            raise RuntimeError("restored CFG cursor mismatch")
        def count_validation(kind: str, completed: bool) -> None:
            if kind != "transformer":
                raise RuntimeError("unexpected model call during CFG validation")
            self.count("transformer_validation", completed)
        actual = trajectory.velocity(self.pipe, z, snapshot, self.prompt,
                                     self.negative, self.dtype,
                                     self.config["generation"]["guidance_scale"],
                                     index, count_validation)
        if trajectory.fingerprint(vars(snapshot)) != before or trajectory.fingerprint(vars(snapshot_cpu)) != before:
            raise RuntimeError("CFG validation advanced or changed scheduler history")
        difference = (actual - expected).abs()
        max_abs = float(difference.max())
        same = bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5))
        receipt = dict(index=index, multi=multi, history_fingerprint=before,
                       state_fingerprint=trajectory.fingerprint(z),
                       expected_velocity_fingerprint=trajectory.fingerprint(expected),
                       actual_velocity_fingerprint=trajectory.fingerprint(actual),
                       max_abs_error=max_abs, rtol=1e-4, atol=1e-5,
                       numerical_match=same)
        if not same:
            raise RuntimeError(f"restored same-input CFG mismatch at T{index}: max_abs={max_abs}")
        return receipt

    def decode(self, normalized_latent: Any) -> Any:
        from runtime.wan.vae import decode_normalized_latent

        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("decode requires isolated VAE phase")
        self.count("vae_decode", False)
        device = next(self.vae.parameters()).device
        rgb = decode_normalized_latent(self.vae, normalized_latent.to(device)).detach().cpu()
        self.count("vae_decode", True)
        return rgb.numpy()

    def encode(self, rgb: Any) -> Any:
        import numpy as np
        import torch
        from runtime.wan.vae import reencode_rgb24_readback

        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("encode requires isolated VAE phase")
        self.count("vae_encode", False)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb))
        encoded = reencode_rgb24_readback(self.vae, tensor).detach().cpu()
        self.count("vae_encode", True)
        return encoded.numpy()

    def clean_proxy(self, index: int) -> Any:
        from runtime.wan import trajectory

        if index not in (44, 46):
            raise ValueError("early clean proxy requires T44 or T46")
        node = self.nodes[index]
        sigma = float(self.snapshots[index].sigmas[index])
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive early scheduler sigma required")
        proxy = node["z"] - sigma * node["v"]
        if not bool(proxy.isfinite().all()):
            raise FloatingPointError("nonfinite early clean proxy")
        return proxy

    def multi_clean_proxy(self) -> Any:
        if self.multi46 is None:
            raise RuntimeError("MULTI live T46 missing")
        sigma = float(self.multi46["snapshot"].sigmas[46])
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive MULTI T46 sigma required")
        proxy = self.multi46["z"] - sigma * self.multi46["v"]
        if not bool(proxy.isfinite().all()):
            raise FloatingPointError("nonfinite MULTI clean proxy")
        return proxy

    def _control(self, index: int, z_cpu: Any, v_cpu: Any, snapshot_cpu: Any,
                 masked_direction: Any, target: float) -> tuple[str, Any, Any, dict]:
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER":
            raise RuntimeError("native control requires Transformer phase")
        if not math.isfinite(target) or target <= 0:
            raise ValueError("positive finite response target required")
        snapshot = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
        trajectory.validate_scheduler(snapshot)
        if snapshot.step_index != index:
            raise ValueError("native control history cursor mismatch")
        sigma = float(snapshot.sigmas[index])
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive controlled-step sigma required")
        z = z_cpu.to(device="cuda", dtype=torch.float32)
        v = v_cpu.to(device="cuda", dtype=torch.float32)
        raw = torch.as_tensor(masked_direction, device="cuda", dtype=torch.float32)
        if raw.shape != z.shape or not bool(torch.isfinite(raw).all()):
            raise ValueError("finite matched masked lift direction required")
        if bool(torch.count_nonzero(raw[:, :, 0])) or bool(torch.count_nonzero(raw[:, :, 45])):
            raise ValueError("lift endpoints must be zero")
        before = trajectory.fingerprint(vars(snapshot))
        shadow, _ = trajectory.zero_step(snapshot, z, v, index, self.count, "shadow_step")
        if index == 49 and trajectory.fingerprint(shadow) != trajectory.fingerprint(self.off_terminal):
            raise RuntimeError("T49 normal shadow differs from OFF terminal")
        support = trajectory.measures(raw)["support_rms"]
        if not math.isfinite(support) or support <= 0:
            return "ZERO_LIFT_DIRECTION", None, None, dict(
                index=index, history_fingerprint=before, shadow_fingerprint=trajectory.fingerprint(shadow),
                raw_direction=trajectory.measures(raw), target_D_support_rms=target)
        unit = raw / support
        probe, _ = trajectory.zero_step(snapshot, z, v - unit / sigma, index,
                                        self.count, "unit_response_probe_step")
        unit_D = trajectory.measures(probe - shadow)
        state = classify_unit_response(unit_D["support_rms"])
        metrics = dict(index=index, sigma=sigma, history_fingerprint=before,
                       state_fingerprint=trajectory.fingerprint(z),
                       velocity_fingerprint=trajectory.fingerprint(v),
                       shadow_fingerprint=trajectory.fingerprint(shadow),
                       raw_direction_fingerprint=trajectory.fingerprint(raw),
                       raw_direction=trajectory.measures(raw),
                       unit_direction=trajectory.measures(unit), unit_D=unit_D,
                       target_D_support_rms=target)
        if trajectory.fingerprint(vars(snapshot)) != before:
            raise RuntimeError("shadow/probe polluted full source history")
        if state == "ZERO_NATIVE_RESPONSE":
            return state, None, None, metrics
        epsilon = target / unit_D["support_rms"]
        controlled_v = v - epsilon * unit / sigma
        if not bool(torch.isfinite(controlled_v).all()):
            raise FloatingPointError("nonfinite controlled CFG velocity")
        next_z = trajectory.native_step(snapshot, z, controlled_v, index,
                                        self.count, "scheduler_step")
        actual = trajectory.measures(next_z - shadow)
        relative_error = abs(actual["support_rms"] - target) / target
        metrics.update(epsilon=epsilon, actual_D=actual,
                       response_relative_error=relative_error,
                       controlled_velocity_fingerprint=trajectory.fingerprint(controlled_v),
                       next_state_fingerprint=trajectory.fingerprint(next_z),
                       next_history_fingerprint=trajectory.fingerprint(vars(snapshot)))
        if relative_error > 2e-5:
            raise RuntimeError("controlled native response differs from fixed target")
        return "READY", next_z, snapshot, metrics

    def control_from_off(self, index: int, masked_direction: Any, target: float) -> tuple[str, Any, Any, dict]:
        if index not in (44, 46, 49):
            raise ValueError("OFF fork requires T44/T46/T49")
        node = self.nodes[index]
        return self._control(index, node["z"], node["v"], self.snapshots[index],
                             masked_direction, target)

    def control_multi46(self, masked_direction: Any, target: float) -> tuple[str, Any, Any, dict]:
        if self.multi46 is None:
            raise RuntimeError("MULTI T46 live state missing")
        node = self.multi46
        return self._control(46, node["z"], node["v"], node["snapshot"],
                             masked_direction, target)

    def continue_normal(self, z: Any, scheduler: Any, start: int, stop: int) -> Any:
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER":
            raise RuntimeError("continuation requires Transformer phase")
        guidance = self.config["generation"]["guidance_scale"]
        with torch.no_grad():
            for index in range(start, stop):
                v = trajectory.velocity(self.pipe, z, scheduler, self.prompt,
                                        self.negative, self.dtype, guidance, index, self.count)
                z = trajectory.native_step(scheduler, z, v, index, self.count)
        return z.detach().cpu()

    def multi44_to_46(self, masked_direction: Any, target: float) -> tuple[str, dict]:
        import torch
        from runtime.wan import trajectory

        state, z45, scheduler, metrics = self.control_from_off(44, masked_direction, target)
        if state != "READY":
            return state, metrics
        guidance = self.config["generation"]["guidance_scale"]
        v45 = trajectory.velocity(self.pipe, z45, scheduler, self.prompt,
                                  self.negative, self.dtype, guidance, 45, self.count)
        z46 = trajectory.native_step(scheduler, z45, v45, 45, self.count)
        v46 = trajectory.velocity(self.pipe, z46, scheduler, self.prompt,
                                  self.negative, self.dtype, guidance, 46, self.count)
        self.multi46 = dict(z=z46.detach().cpu().clone(), v=v46.detach().cpu().clone(),
                            snapshot=_move_scheduler(copy.deepcopy(scheduler), torch.device("cpu")))
        if self.multi46["snapshot"].step_index != 46:
            raise RuntimeError("MULTI evolved history did not reach T46")
        metrics["multi46_state_fingerprint"] = trajectory.fingerprint(self.multi46["z"])
        metrics["multi46_velocity_fingerprint"] = trajectory.fingerprint(self.multi46["v"])
        metrics["multi46_history_fingerprint"] = trajectory.fingerprint(vars(self.multi46["snapshot"]))
        return "READY", metrics

    def net_from_off(self, terminal: Any) -> dict:
        from runtime.wan import trajectory

        return trajectory.measures(terminal - self.off_terminal)

    def resources(self) -> dict:
        return dict(cuda=_cuda_memory(), phase=self.phase,
                    current_phase_elapsed_seconds=time.perf_counter() - self._phase_started,
                    phases=list(self.phase_log))

    def release(self) -> None:
        import torch
        from runtime.wan.vae import _clear_cache

        if self.vae is not None:
            _clear_cache(self.vae)
        self.vae = None
        if self.pipe is not None:
            self.pipe.transformer = None
            self.pipe.text_encoder = None
            self.pipe.vae = None
        self.pipe = self.prompt = self.negative = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.phase = "RELEASED"
