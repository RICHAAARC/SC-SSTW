"""Live-history T44/T46/T48 local RGB-DCT gradient backend."""
from __future__ import annotations

import copy
import gc
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from main.tube_state import rgb_dct_gradient_proxy as proxy
from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state.rgb_dct_t49_carrier import classify_unit_response
from runtime.wan.rgb_dct_gradient_checkpoint import CheckpointLedger, checkpoint_decode


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


class WanLocalGradientBackend:
    """One source process with mutually exclusive Transformer and VAE phases."""

    def __init__(self, config: dict, count: Callable[[str, bool], None]):
        self.config = config
        self.count = count
        self.pipe = self.prompt = self.negative = self.dtype = None
        self.vae = None
        self.phase = "INITIAL"
        self.phase_log: list[dict] = []
        self.nodes: dict[int, dict] = {}
        self.snapshots: dict[int, Any] = {}
        self.local: dict | None = None
        self.off_terminal = None
        self.terminals: dict[str, Any] = {}
        self.initial_noise = None
        self.initial_noise_fingerprint = None
        self._phase_identity = None
        self._phase_started = time.perf_counter()
        self.gradient_attempts: list[dict] = []
        checkpoint = config["checkpoint"]
        self.checkpoint_ledger = CheckpointLedger(
            count, checkpoint["vae_chunk_forward_max_per_source"],
            checkpoint["vae_chunk_recompute_max_per_source"])

    def _phase_receipt(self, name: str, started: float, **extra: Any) -> dict:
        receipt = dict(name=name, elapsed_seconds=time.perf_counter() - started,
                       cuda=_cuda_memory(), **extra)
        self.phase_log.append(receipt)
        return receipt

    def _phase_identity_record(self) -> dict:
        from runtime.wan import trajectory

        return dict(
            initial_noise=trajectory.fingerprint(self.initial_noise),
            prompt=trajectory.fingerprint(self.prompt),
            negative=trajectory.fingerprint(self.negative),
            off_histories={index: trajectory.fingerprint(vars(snapshot))
                           for index, snapshot in self.snapshots.items()},
            local_history=(trajectory.fingerprint(vars(self.local["snapshot"]))
                           if self.local else None),
            local_state=(trajectory.fingerprint(self.local["z"]) if self.local else None),
            local_velocity=(trajectory.fingerprint(self.local["v"])
                            if self.local and self.local.get("v") is not None else None),
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
        self.initial_noise = z.detach().cpu().clone()
        self.initial_noise_fingerprint = trajectory.fingerprint(z)
        guidance = self.config["generation"]["guidance_scale"]
        with torch.no_grad():
            for index in range(50):
                v = trajectory.velocity(self.pipe, z, scheduler, self.prompt,
                                        self.negative, self.dtype, guidance, index, self.count)
                if index in (44, 49):
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
        for index in (44, 49):
            for kind, value in self.nodes[index].items():
                path = artifact_dir / f"{kind}{index}.pt"
                torch.save(value, path)
                artifacts[f"{kind}{index}"] = dict(
                    path=str(path), sha256=_sha(path), fingerprint=trajectory.fingerprint(value))
            path = artifact_dir / f"scheduler{index}.pt"
            torch.save(self.snapshots[index], path)
            artifacts[f"scheduler{index}"] = dict(
                path=str(path), sha256=_sha(path),
                fingerprint=trajectory.fingerprint(vars(self.snapshots[index])))
        for name, value in (("off_terminal", self.off_terminal),
                            ("initial_noise", self.initial_noise)):
            path = artifact_dir / f"{name}.pt"
            torch.save(value, path)
            artifacts[name] = dict(path=str(path), sha256=_sha(path),
                                   fingerprint=trajectory.fingerprint(value))
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
        if self.local is not None:
            self.local["snapshot"] = _move_scheduler(
                self.local["snapshot"], torch.device("cpu"))
            self.local["z"] = self.local["z"].detach().cpu()
            if self.local.get("v") is not None:
                self.local["v"] = self.local["v"].detach().cpu()
        self._phase_identity = self._phase_identity_record()
        self.pipe.transformer.to(torch.device("cpu"))
        self.prompt = self.prompt.cpu()
        self.negative = self.negative.cpu()
        _move_scheduler(self.pipe.scheduler, torch.device("cpu"))
        if any(parameter.is_cuda for parameter in self.pipe.transformer.parameters()):
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
        restored = self._phase_identity_record()
        if restored != self._phase_identity:
            raise RuntimeError("state/conditioning/history identity changed across VAE phase")
        self.phase = "TRANSFORMER"
        self._phase_started = time.perf_counter()
        return self._phase_receipt(name, started, transformer_resident="cuda",
                                   vae_resident="absent", identity_verified=True,
                                   prior_vae_phase_elapsed_seconds=previous_elapsed)

    def decode(self, normalized_latent: Any) -> np.ndarray:
        from runtime.wan.vae import decode_normalized_latent

        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("decode requires isolated VAE phase")
        self.count("vae_decode", False)
        device = next(self.vae.parameters()).device
        rgb = decode_normalized_latent(self.vae, normalized_latent.to(device)).detach()
        self.count("vae_decode", True)
        # Explicit float() handles a BF16-producing implementation before NumPy.
        return rgb.float().cpu().numpy()

    def encode(self, rgb: Any) -> np.ndarray:
        import torch
        from runtime.wan.vae import reencode_rgb24_readback

        if self.phase != "VAE" or self.vae is None:
            raise RuntimeError("encode requires isolated VAE phase")
        self.count("vae_encode", False)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb))
        encoded = reencode_rgb24_readback(self.vae, tensor).detach()
        self.count("vae_encode", True)
        return encoded.float().cpu().numpy()

    def start_local44(self) -> dict:
        """Recompute the local arm's own live T44 CFG velocity."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER" or self.local is not None:
            raise RuntimeError("local arm must start once in Transformer phase")
        snapshot = _move_scheduler(copy.deepcopy(self.snapshots[44]), torch.device("cuda"))
        z = self.nodes[44]["z"].to(device="cuda", dtype=torch.float32)
        v = trajectory.velocity(self.pipe, z, snapshot, self.prompt, self.negative,
                                self.dtype, self.config["generation"]["guidance_scale"],
                                44, self.count)
        self.local = dict(index=44, z=z.detach().cpu(), v=v.detach().cpu(),
                          snapshot=_move_scheduler(snapshot, torch.device("cpu")))
        return self._local_identity()

    def _local_identity(self) -> dict:
        from runtime.wan import trajectory

        if self.local is None:
            raise RuntimeError("local arm absent")
        return dict(index=self.local["index"],
                    state_fingerprint=trajectory.fingerprint(self.local["z"]),
                    velocity_fingerprint=(trajectory.fingerprint(self.local["v"])
                                          if self.local.get("v") is not None else None),
                    history_fingerprint=trajectory.fingerprint(vars(self.local["snapshot"])))

    def local_gradient(self, key: bytes) -> tuple[str, Any | None, dict]:
        """Compute and parity-check the fixed objective gradient at the live step."""
        import torch
        from runtime.wan import trajectory
        from runtime.wan.vae import _clear_cache, decode_normalized_latent_with_grad

        if self.phase != "VAE" or self.vae is None or self.local is None:
            raise RuntimeError("live local gradient requires VAE phase and local state")
        index = self.local["index"]
        if index not in (44, 46, 48):
            raise ValueError("local gradient step must be T44, T46, or T48")
        sigma = float(self.local["snapshot"].sigmas[index])
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive local proxy sigma required")
        device = next(self.vae.parameters()).device
        z = self.local["z"].to(device=device, dtype=torch.float32)
        velocity = self.local["v"].to(device=device, dtype=torch.float32).detach()
        velocity.requires_grad_(True)
        clean = z - sigma * velocity
        identity = self._local_identity()
        rgb = loss = q = metadata = gradient = None
        try:
            self.count("vae_decode", False)
            rgb = decode_normalized_latent_with_grad(
                self.vae, clean,
                decode_fn=lambda vae, raw: checkpoint_decode(
                    vae, raw, self.checkpoint_ledger))
            self.count("vae_decode", True)
            loss, q, metadata = proxy.loss_from_rgb(rgb, key)
            # NumPy receives the exact decoded FP32 values used by the proxy.
            numpy_rgb = rgb.detach().float().cpu().numpy()
            baseline = receiver.score_rgb(numpy_rgb, key)
            if baseline["status"] != "SCORED" or len(baseline["group_scores"]) != 30:
                raise RuntimeError("same-RGB NumPy proxy reference was invalid")
            q_np = np.asarray(baseline["group_scores"], dtype=np.float64)
            q_torch = q.detach().cpu().numpy().astype(np.float64, copy=False)
            absolute = np.abs(q_torch - q_np)
            atol = self.config["control"]["proxy_q_atol"]
            rtol = self.config["control"]["proxy_q_rtol"]
            parity = bool(np.allclose(q_torch, q_np, atol=atol, rtol=rtol))
            receipt = dict(**identity, sigma=sigma, loss=float(loss.detach()),
                           q=[float(value) for value in q_torch],
                           positive_groups=int(np.count_nonzero(q_torch > 0)),
                           numpy_q=[float(value) for value in q_np],
                           numpy_positive_groups=baseline["positive_groups"],
                           proxy_q_max_abs_error=float(absolute.max()),
                           proxy_q_atol=atol, proxy_q_rtol=rtol,
                           proxy_q_parity=parity,
                           decoded_rgb_dtype=str(rgb.dtype),
                           decoded_rgb_shape=list(rgb.shape),
                           clean_proxy_fingerprint=trajectory.fingerprint(clean))
            self.gradient_attempts.append(receipt)
            if not parity:
                raise RuntimeError("differentiable q does not match same-RGB NumPy receiver")
            self.count("backward", False)
            gradient = torch.autograd.grad(loss, velocity, retain_graph=False,
                                           create_graph=False, allow_unused=False)[0]
            self.count("backward", True)
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("nonfinite or missing local velocity gradient")
            masked = -gradient.detach().float()
            masked[:, :, 0] = 0
            masked[:, :, 45] = 0
            measures = trajectory.measures(masked)
            receipt.update(gradient=trajectory.measures(gradient),
                           negative_masked_gradient=measures,
                           negative_masked_gradient_fingerprint=trajectory.fingerprint(masked))
            if not math.isfinite(measures["support_rms"]) or measures["support_rms"] <= 0:
                return "ZERO_LOCAL_GRADIENT", None, receipt
            return "READY", masked.cpu(), receipt
        finally:
            rgb = loss = q = metadata = gradient = velocity = clean = None
            gc.collect()
            _clear_cache(self.vae)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def verify_local_cfg(self) -> dict:
        """Recompute CFG after phase restoration without advancing live history."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER" or self.local is None:
            raise RuntimeError("local CFG check requires Transformer phase")
        index = self.local["index"]
        before = trajectory.fingerprint(vars(self.local["snapshot"]))
        snapshot = _move_scheduler(copy.deepcopy(self.local["snapshot"]),
                                   torch.device("cuda"))
        z = self.local["z"].to(device="cuda", dtype=torch.float32)
        expected = self.local["v"].to(device="cuda", dtype=torch.float32)
        def count_validation(kind: str, completed: bool) -> None:
            if kind != "transformer":
                raise RuntimeError("unexpected call in local CFG validation")
            self.count("transformer_validation", completed)
        actual = trajectory.velocity(
            self.pipe, z, snapshot, self.prompt, self.negative, self.dtype,
            self.config["generation"]["guidance_scale"], index, count_validation)
        if (trajectory.fingerprint(vars(snapshot)) != before
                or trajectory.fingerprint(vars(self.local["snapshot"])) != before):
            raise RuntimeError("CFG validation changed live history")
        difference = (actual - expected).abs()
        same = bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5))
        receipt = dict(**self._local_identity(), max_abs_error=float(difference.max()),
                       rtol=1e-4, atol=1e-5, numerical_match=same)
        if not same:
            raise RuntimeError(f"restored same-input CFG mismatch at T{index}")
        return receipt

    def _control(self, index: int, z_cpu: Any, v_cpu: Any, snapshot_cpu: Any,
                 velocity_direction: Any, target: float) -> tuple[str, Any, Any, dict]:
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER":
            raise RuntimeError("native control requires Transformer phase")
        if not math.isfinite(target) or target <= 0:
            raise ValueError("positive finite native response target required")
        snapshot = _move_scheduler(copy.deepcopy(snapshot_cpu), torch.device("cuda"))
        trajectory.validate_scheduler(snapshot)
        if snapshot.step_index != index:
            raise ValueError("native control history cursor mismatch")
        z = z_cpu.to(device="cuda", dtype=torch.float32)
        v = v_cpu.to(device="cuda", dtype=torch.float32)
        raw = torch.as_tensor(velocity_direction, device="cuda", dtype=torch.float32)
        if raw.shape != z.shape or not bool(torch.isfinite(raw).all()):
            raise ValueError("finite matched velocity direction required")
        if bool(torch.count_nonzero(raw[:, :, 0])) or bool(torch.count_nonzero(raw[:, :, 45])):
            raise ValueError("velocity direction endpoints must be zero")
        before = trajectory.fingerprint(vars(snapshot))
        shadow, _ = trajectory.zero_step(snapshot, z, v, index, self.count, "shadow_step")
        support = trajectory.measures(raw)["support_rms"]
        metrics = dict(index=index, history_fingerprint=before,
                       state_fingerprint=trajectory.fingerprint(z),
                       velocity_fingerprint=trajectory.fingerprint(v),
                       shadow_fingerprint=trajectory.fingerprint(shadow),
                       raw_direction_fingerprint=trajectory.fingerprint(raw),
                       raw_direction=trajectory.measures(raw),
                       target_D_support_rms=target)
        if not math.isfinite(support) or support <= 0:
            return "ZERO_LOCAL_GRADIENT", None, None, metrics
        unit = raw / support
        probe, _ = trajectory.zero_step(snapshot, z, v + unit, index, self.count,
                                        "unit_response_probe_step")
        unit_response = trajectory.measures(probe - shadow)
        metrics.update(unit_direction=trajectory.measures(unit), unit_D=unit_response)
        if trajectory.fingerprint(vars(snapshot)) != before:
            raise RuntimeError("shadow/probe polluted live source history")
        state = classify_unit_response(unit_response["support_rms"])
        if state == "ZERO_NATIVE_RESPONSE":
            return state, None, None, metrics
        epsilon = target / unit_response["support_rms"]
        controlled_velocity = v + epsilon * unit
        if not bool(torch.isfinite(controlled_velocity).all()):
            raise FloatingPointError("nonfinite controlled CFG velocity")
        next_z = trajectory.native_step(snapshot, z, controlled_velocity, index,
                                        self.count, "scheduler_step")
        actual = trajectory.measures(next_z - shadow)
        relative_error = abs(actual["support_rms"] - target) / target
        metrics.update(epsilon=epsilon, actual_D=actual,
                       response_relative_error=relative_error,
                       controlled_velocity_fingerprint=trajectory.fingerprint(controlled_velocity),
                       next_state_fingerprint=trajectory.fingerprint(next_z),
                       next_history_fingerprint=trajectory.fingerprint(vars(snapshot)))
        if relative_error > 2e-5:
            raise RuntimeError("controlled native response differs from fixed target")
        return "READY", next_z, snapshot, metrics

    def control_single49(self, latent_direction: Any, target: float) -> tuple[str, Any, dict]:
        """Existing positive reference maps a clean-latent direction into velocity."""
        snapshot = self.snapshots[49]
        sigma = float(snapshot.sigmas[49])
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive T49 sigma required")
        raw = -np.asarray(latent_direction, dtype=np.float32) / sigma
        state, terminal, _, metrics = self._control(
            49, self.nodes[49]["z"], self.nodes[49]["v"], snapshot, raw, target)
        return state, terminal, metrics

    def control_local(self, velocity_direction: Any, target: float) -> tuple[str, dict]:
        if self.local is None:
            raise RuntimeError("local arm absent")
        index = self.local["index"]
        if index not in (44, 46, 48):
            raise ValueError("local control step must be T44, T46, or T48")
        state, next_z, snapshot, metrics = self._control(
            index, self.local["z"], self.local["v"], self.local["snapshot"],
            velocity_direction, target)
        if state == "READY":
            self.local = dict(index=index + 1, z=next_z.detach().cpu(), v=None,
                              snapshot=_move_scheduler(snapshot, __import__("torch").device("cpu")))
        return state, metrics

    def advance_local_to(self, target_index: int) -> dict:
        """Free-step the live arm, then recompute its target-step CFG velocity."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER" or self.local is None:
            raise RuntimeError("local advance requires Transformer phase")
        if target_index not in (46, 48) or self.local["index"] != target_index - 1:
            raise ValueError("local arm may advance only 45->46 or 47->48")
        scheduler = _move_scheduler(copy.deepcopy(self.local["snapshot"]),
                                    torch.device("cuda"))
        z = self.local["z"].to(device="cuda", dtype=torch.float32)
        guidance = self.config["generation"]["guidance_scale"]
        index = target_index - 1
        v = trajectory.velocity(self.pipe, z, scheduler, self.prompt, self.negative,
                                self.dtype, guidance, index, self.count)
        z = trajectory.native_step(scheduler, z, v, index, self.count)
        live_v = trajectory.velocity(self.pipe, z, scheduler, self.prompt, self.negative,
                                     self.dtype, guidance, target_index, self.count)
        self.local = dict(index=target_index, z=z.detach().cpu(),
                          v=live_v.detach().cpu(),
                          snapshot=_move_scheduler(scheduler, torch.device("cpu")))
        return self._local_identity()

    def finish_local(self) -> Any:
        """After controlled T48, run the normal free T49 step."""
        import torch
        from runtime.wan import trajectory

        if self.phase != "TRANSFORMER" or self.local is None or self.local["index"] != 49:
            raise RuntimeError("local terminal requires post-control T49 state")
        scheduler = _move_scheduler(copy.deepcopy(self.local["snapshot"]),
                                    torch.device("cuda"))
        z = self.local["z"].to(device="cuda", dtype=torch.float32)
        v = trajectory.velocity(self.pipe, z, scheduler, self.prompt, self.negative,
                                self.dtype, self.config["generation"]["guidance_scale"],
                                49, self.count)
        terminal = trajectory.native_step(scheduler, z, v, 49, self.count)
        self.terminals["LOCAL44_46_48"] = terminal.detach().cpu()
        return self.terminals["LOCAL44_46_48"]

    def net_from_off(self, terminal: Any) -> dict:
        from runtime.wan import trajectory

        return trajectory.measures(terminal.float() - self.off_terminal.float())

    def resources(self) -> dict:
        return dict(cuda=_cuda_memory(), phase=self.phase, phase_log=self.phase_log,
                    checkpoint=self.checkpoint_ledger.summary(),
                    gradient_attempts=self.gradient_attempts)

    def release(self) -> None:
        import torch
        from runtime.wan.vae import _clear_cache

        if self.vae is not None:
            _clear_cache(self.vae)
        self.vae = None
        if self.pipe is not None:
            self.pipe.transformer = None
            self.pipe.text_encoder = None
        self.pipe = self.prompt = self.negative = self.local = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
