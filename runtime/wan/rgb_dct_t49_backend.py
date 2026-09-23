"""One no-gradient, same-history Wan T49 backend for the fixed RGB/DCT trial."""
from __future__ import annotations

import copy
import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from main.tube_state.rgb_dct_t49_carrier import classify_unit_response


@dataclass
class Prefix49:
    z49: Any
    v49: Any
    z0: Any
    snapshot: Any
    metadata: dict


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _release_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class WanT49Backend:
    def __init__(self, config: dict, count: Callable[[str, bool], None]):
        self.config = config
        self.count = count
        self.vae = None

    def generate_prefix(self, artifact_dir: Path) -> Prefix49:
        """One shared normal 0..48 prefix, v49, then normal OFF step 49."""
        import torch
        from runtime.wan import trajectory
        from runtime.wan.generation import prepare_generation

        pipe = initial = prompt = negative = None
        self.count("generation", False)
        try:
            pipe, initial, prompt, negative, dtype = prepare_generation(self.config, load_vae=False)
            scheduler = pipe.scheduler
            trajectory.validate_scheduler(scheduler)
            z = initial.detach()
            initial_fp = trajectory.fingerprint(z)
            guidance = self.config["generation"]["guidance_scale"]
            with torch.no_grad():
                for index in range(49):
                    v = trajectory.velocity(pipe, z, scheduler, prompt, negative, dtype,
                                            guidance, index, self.count)
                    z = trajectory.native_step(scheduler, z, v, index, self.count)
                if scheduler.step_index != 49:
                    raise ValueError("T49 native scheduler cursor mismatch")
                z49 = z.detach().float()
                v49 = trajectory.velocity(pipe, z49, scheduler, prompt, negative,
                                          dtype, guidance, 49, self.count)
                snapshot = copy.deepcopy(scheduler)
                history_fp = trajectory.fingerprint(vars(snapshot))
                sigma49, sigma50 = float(snapshot.sigmas[49]), float(snapshot.sigmas[50])
                if not math.isfinite(sigma49) or sigma49 <= 0 or sigma50 != 0:
                    raise ValueError("fixed T49 positive sigma and zero sigma50 required")
                z0, _ = trajectory.zero_step(snapshot, z49, v49, 49, self.count,
                                             "scheduler_step")
                if trajectory.fingerprint(vars(snapshot)) != history_fp:
                    raise RuntimeError("OFF step polluted shared T49 history")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for name, value in (("z49", z49), ("v49", v49), ("z0", z0)):
                path = artifact_dir / f"{name}.pt"
                torch.save(value.detach().cpu(), path)
                artifacts[name] = dict(path=str(path), sha256=_sha(path),
                                       fingerprint=trajectory.fingerprint(value))
            history_path = artifact_dir / "scheduler49_history.pt"
            torch.save(snapshot, history_path)
            artifacts["scheduler49_history"] = dict(
                path=str(history_path), sha256=_sha(history_path), fingerprint=history_fp,
            )
            metadata = dict(
                initial_noise_fingerprint=initial_fp,
                scheduler_class=type(snapshot).__name__,
                scheduler_config=dict(snapshot.config),
                scheduler_cursor=49, sigma49=sigma49, sigma50=sigma50,
                transformer_dtype=str(dtype),
                normal_terminal_fingerprint=trajectory.fingerprint(z0),
                artifacts=artifacts,
            )
            self.count("generation", True)
            return Prefix49(z49.detach().cpu(), v49.detach().cpu(), z0.detach().cpu(),
                            snapshot, metadata)
        finally:
            if pipe is not None:
                pipe.transformer = None
                pipe.text_encoder = None
            pipe = initial = prompt = negative = z = v = None
            _release_memory()

    def load_vae(self) -> dict:
        import torch
        from runtime.wan.generation import load_frozen_vae

        # generate_prefix has returned: its transformer locals are now gone.
        _release_memory()
        self.vae = load_frozen_vae(self.config)
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise ValueError("frozen VAE unexpectedly has gradient parameters")
        clear = getattr(self.vae, "clear_cache", None) or getattr(self.vae, "_clear_cache", None)
        return dict(dtype=str(next(self.vae.parameters()).dtype),
                    cache_clear_available=callable(clear),
                    cache_policy="shared VAE adapter clears before and after decode/encode",
                    cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                               if torch.cuda.is_available() else None))

    def decode(self, normalized_latent) -> Any:
        from runtime.wan.vae import decode_normalized_latent

        self.count("vae_decode", False)
        device = next(self.vae.parameters()).device
        rgb = decode_normalized_latent(self.vae, normalized_latent.to(device)).detach().cpu()
        self.count("vae_decode", True)
        return rgb.numpy()

    def encode(self, rgb) -> Any:
        import numpy as np
        import torch
        from runtime.wan.vae import reencode_rgb24_readback

        self.count("vae_encode", False)
        frame_tensor = torch.from_numpy(np.ascontiguousarray(rgb))
        encoded = reencode_rgb24_readback(self.vae, frame_tensor).detach().cpu()
        self.count("vae_encode", True)
        return encoded.numpy()

    def controlled_terminal(self, prefix: Prefix49, masked_direction, target: float) -> tuple[str, Any, dict]:
        import torch
        from runtime.wan import trajectory

        snapshot = prefix.snapshot
        trajectory.validate_scheduler(snapshot)
        if snapshot.step_index != 49 or len(snapshot.sigmas) != 51:
            raise ValueError("fixed T49 scheduler cursor/history required")
        sigma49, sigma50 = float(snapshot.sigmas[49]), float(snapshot.sigmas[50])
        if not math.isfinite(sigma49) or sigma49 <= 0 or sigma50 != 0:
            raise ValueError("fixed T49 positive sigma and zero sigma50 required")
        if not math.isfinite(target) or target <= 0:
            raise ValueError("positive finite native response target required")
        before = trajectory.fingerprint(vars(snapshot))
        device = snapshot.timesteps.device
        z49 = prefix.z49.to(device=device, dtype=torch.float32)
        v49 = prefix.v49.to(device=device, dtype=torch.float32)
        z0 = prefix.z0.to(device=device, dtype=torch.float32)
        raw = torch.as_tensor(masked_direction, device=device, dtype=torch.float32)
        if tuple(raw.shape) != tuple(z49.shape) or not bool(torch.isfinite(raw).all()):
            raise ValueError("finite matched masked lift direction required")
        if bool(torch.count_nonzero(raw[:, :, 0])) or bool(torch.count_nonzero(raw[:, :, 45])):
            raise ValueError("lift endpoints must be zero")
        norm = trajectory.measures(raw)["support_rms"]
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("zero/nonfinite lift direction before native control")
        unit = raw / norm
        probe, _ = trajectory.zero_step(snapshot, z49, v49 - unit / sigma49, 49,
                                        self.count, "unit_response_probe_step")
        unit_response = trajectory.measures(probe - z0)
        response = unit_response["support_rms"]
        state = classify_unit_response(response)
        if trajectory.fingerprint(vars(snapshot)) != before:
            raise RuntimeError("unit probe polluted source scheduler history")
        metrics = dict(
            index=49, sigma49=sigma49, sigma50=sigma50,
            history_fingerprint=before, raw_direction=trajectory.measures(raw),
            unit_direction=trajectory.measures(unit), unit_D=unit_response,
            target_D_support_rms=target,
        )
        if state == "ZERO_NATIVE_RESPONSE":
            return state, None, metrics
        epsilon = target / response
        controlled_velocity = v49 - epsilon * unit / sigma49
        if not bool(torch.isfinite(controlled_velocity).all()):
            raise FloatingPointError("nonfinite controlled T49 velocity")
        z_native, _ = trajectory.zero_step(snapshot, z49, controlled_velocity, 49,
                                           self.count, "scheduler_step")
        actual = trajectory.measures(z_native - z0)
        relative_error = abs(actual["support_rms"] - target) / target
        if relative_error > 2e-5 or trajectory.fingerprint(vars(snapshot)) != before:
            raise RuntimeError("T49 response mismatch or source history pollution")
        metrics.update(epsilon=epsilon, actual_D=actual,
                       matching_relative_error=relative_error,
                       controlled_terminal_fingerprint=trajectory.fingerprint(z_native),
                       zero_terminal_fingerprint=trajectory.fingerprint(z0))
        return "READY", z_native.detach().cpu(), metrics

    def resources(self) -> dict:
        import torch

        return dict(
            cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                       if torch.cuda.is_available() else None),
            cuda_peak_reserved_bytes=(torch.cuda.max_memory_reserved()
                                      if torch.cuda.is_available() else None),
        )

    def release(self) -> None:
        if self.vae is not None:
            from runtime.wan.vae import _clear_cache

            _clear_cache(self.vae)
            self.vae = None
        _release_memory()
