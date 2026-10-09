"""Real Wan runner pieces for the optional local joint carrier experiment.

This module owns model residency and artifact mechanics.  It does not select
scientific parameter values, an arm roster, a receiver threshold, or a blind
path.  The experiment entry point must supply every such value explicitly.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from runtime.wan import grow_video_reference as grow
from runtime.wan import local_joint_state_payload_provider_v1 as joint
from runtime.wan import trajectory
from runtime.wan import vae as vae_adapter


SCHEMA = "local-joint-state-payload-real-v1"
ARMS = ("OFF", "JOINT")
LAYERS = ("terminal_latent", "float_rgb", "rgb8", "mp4")


def _resources(device: Any) -> dict[str, Any]:
    """Process CUDA counters; peak is cumulative since process/device reset."""
    import torch

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return dict(device=str(target), allocated_bytes=None, reserved_bytes=None,
                    peak_allocated_bytes=None, peak_scope="process_cumulative_since_cuda_reset")
    return dict(device=str(target), allocated_bytes=int(torch.cuda.memory_allocated(target)),
                reserved_bytes=int(torch.cuda.memory_reserved(target)),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(target)),
                peak_scope="process_cumulative_since_cuda_reset")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the explicit real-run contract without choosing defaults."""
    required = {"schema", "model", "generation", "carrier", "media", "source", "arms", "runtime"}
    if set(config) != required or config["schema"] != SCHEMA:
        raise ValueError(f"config must contain exactly {sorted(required)} and schema {SCHEMA}")
    model = config["model"]
    if set(model) != {"id", "revision"} or not all(isinstance(model[x], str) and model[x] for x in model):
        raise ValueError("model id and immutable revision must be explicit nonempty strings")
    generation = config["generation"]
    generation_fields = {"height", "width", "frames", "steps", "guidance_scale", "max_sequence_length", "prompt", "negative_prompt", "seed"}
    if set(generation) != generation_fields:
        raise ValueError(f"generation must contain exactly {sorted(generation_fields)}")
    if (generation["frames"], generation["height"], generation["width"], generation["steps"]) != (181, 320, 512, 50):
        raise ValueError("V1 real path requires 181x320x512 and the complete native 50-step schedule")
    if type(generation["seed"]) is not int or generation["seed"] < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    if not isinstance(generation["prompt"], str) or not isinstance(generation["negative_prompt"], str):
        raise TypeError("prompt strings must be explicit")
    if not math.isfinite(float(generation["guidance_scale"])):
        raise ValueError("guidance_scale must be finite")
    spec = config["carrier"]
    if set(spec) != {"key", "wrong_key", "message_hex", "rho", "cap"}:
        raise ValueError("carrier requires explicit key, wrong_key, message_hex, rho, and cap")
    if not isinstance(spec["key"], str) or not isinstance(spec["wrong_key"], str):
        raise TypeError("keys must be original Unicode strings")
    if spec["wrong_key"] == spec["key"]:
        raise ValueError("wrong_key must differ from the writer key")
    try:
        message = bytes.fromhex(spec["message_hex"])
    except (TypeError, ValueError) as exc:
        raise ValueError("message_hex must encode exactly four bytes") from exc
    carrier.message_fragments(message)
    if not math.isfinite(float(spec["rho"])) or not 0 <= float(spec["rho"]) <= 1:
        raise ValueError("rho must be explicit, finite, and in [0,1]")
    if not math.isfinite(float(spec["cap"])) or float(spec["cap"]) < 0:
        raise ValueError("cap must be explicit, finite, and nonnegative")
    if not isinstance(config["arms"], list) or not config["arms"] or len(set(config["arms"])) != len(config["arms"]):
        raise ValueError("arms must be an explicit nonempty unique list")
    if any(value not in ARMS for value in config["arms"]):
        raise ValueError(f"implemented arms are {ARMS}")
    runtime = config["runtime"]
    if set(runtime) != {"device", "transformer_dtype"} or runtime["device"] != "cuda" or runtime["transformer_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("runtime requires explicit cuda device and supported transformer_dtype")
    media = config["media"]
    if set(media) != {"fps", "codec", "crf", "pixel_format"}:
        raise ValueError("media requires explicit fps, codec, crf, and pixel_format")
    if type(media["fps"]) is not int or media["fps"] <= 0 or type(media["crf"]) is not int:
        raise ValueError("media fps and crf must be explicit integers")
    if not all(isinstance(media[x], str) and media[x] for x in ("codec", "pixel_format")):
        raise ValueError("codec and pixel_format must be explicit nonempty strings")
    source = config["source"]
    if set(source) != {"source_id", "development_only"} or not isinstance(source["source_id"], str) or not source["source_id"]:
        raise ValueError("source identity and development_only declaration are required")
    if type(source["development_only"]) is not bool:
        raise TypeError("development_only must be boolean")


def _tensor_file(path: Path, value: Any) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value.detach().cpu(), temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(status="SAVED", path=str(path), sha256=_sha(path), shape=list(value.shape), dtype=str(value.dtype))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _move_scheduler(scheduler: Any, device: Any) -> Any:
    import torch

    for name, value in vars(scheduler).items():
        if torch.is_tensor(value):
            setattr(scheduler, name, value.to(device))
        elif isinstance(value, list):
            setattr(scheduler, name, [item.to(device) if torch.is_tensor(item) else item for item in value])
    return scheduler


class WanSerialResidency:
    """Own one transformer and load one frozen VAE only during each VAE phase."""

    def __init__(self, config: dict[str, Any], event: Callable[[str, dict[str, Any]], None], *, loaders: Any = None):
        import torch
        from runtime.wan import generation

        validate_config(config)
        self.config, self.event = config, event
        self.device = torch.device(config["runtime"]["device"])
        self.dtype = getattr(torch, config["runtime"]["transformer_dtype"])
        self.loaders = loaders or generation
        self.pipe = self.initial = self.prompt = self.negative = None
        self.vae = None
        self.backend = None
        self.active_scheduler = None
        self.phase = "EMPTY"
        self.phase_identity = None

    def load_generation(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.event("generation_load", {"status": "ATTEMPTED", "load_vae": False, "resources": _resources(self.device)})
        try:
            values = self.loaders.prepare_generation(self.config, load_vae=False, device=self.device, model_dtype=self.dtype)
            self.pipe, self.initial, self.prompt, self.negative, self.dtype = values
            self.phase = "TRANSFORMER"
            row = dict(status="COMPLETED", load_vae=False, transformer_resident=str(self.device), vae_resident="absent",
                       initial_fingerprint=trajectory.fingerprint(self.initial), prompt_fingerprint=trajectory.fingerprint(self.prompt),
                       negative_fingerprint=trajectory.fingerprint(self.negative),
                       elapsed_seconds=time.perf_counter() - started, resources=_resources(self.device))
            self.event("generation_load", row)
            return row
        except Exception as exc:
            self.event("generation_load", {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "retry": False,
                                           "elapsed_seconds": time.perf_counter() - started, "resources": _resources(self.device)})
            raise

    def pristine_scheduler(self) -> Any:
        if self.phase != "TRANSFORMER":
            raise RuntimeError("pristine scheduler requires transformer phase")
        value = copy.deepcopy(self.pipe.scheduler)
        if value.step_index is not None:
            raise RuntimeError("loaded scheduler is not pristine")
        return value

    def _identity(self) -> dict[str, str]:
        return dict(initial=trajectory.fingerprint(self.initial), prompt=trajectory.fingerprint(self.prompt),
                    negative=trajectory.fingerprint(self.negative),
                    scheduler=trajectory.fingerprint(vars(self.active_scheduler)) if self.active_scheduler is not None else None)

    def enter_vae(self, label: str) -> dict[str, Any]:
        import torch

        if self.phase != "TRANSFORMER" or self.vae is not None:
            raise RuntimeError("VAE phase requires one live transformer phase")
        started = time.perf_counter()
        self.event(label, {"status": "ATTEMPTED", "transition": "transformer_to_vae", "resources": _resources(self.device)})
        try:
            self.phase_identity = self._identity()
            self.pipe.transformer.to(torch.device("cpu"))
            if self.active_scheduler is None:
                raise RuntimeError("VAE phase requires the active arm scheduler")
            _move_scheduler(self.active_scheduler, torch.device("cpu"))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.vae = self.loaders.load_frozen_vae(self.config, device=self.device)
            self.backend = joint.WanPosteriorBackend(self.vae)
            self.phase = "VAE"
            row = dict(status="COMPLETED", transition="transformer_to_vae", transformer_resident="cpu",
                       vae_resident=str(self.device), scheduler_resident="cpu",
                       conditioning_resident=str(self.prompt.device),
                       conditioning_deliberately_retained=True, identity_before=self.phase_identity,
                       elapsed_seconds=time.perf_counter() - started, resources=_resources(self.device))
            self.event(label, row)
            return row
        except Exception as exc:
            self.event(label, {"status": "FAILED", "transition": "transformer_to_vae", "reason": f"{type(exc).__name__}: {exc}", "retry": False,
                               "elapsed_seconds": time.perf_counter() - started, "resources": _resources(self.device)})
            raise

    def leave_vae(self, label: str) -> dict[str, Any]:
        import torch

        if self.phase != "VAE":
            raise RuntimeError("transformer restore requires VAE phase")
        started = time.perf_counter()
        self.event(label, {"status": "ATTEMPTED", "transition": "vae_to_transformer", "resources": _resources(self.device)})
        try:
            vae_adapter._clear_cache(self.vae)
            self.backend = self.vae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.pipe.transformer.to(self.device)
            _move_scheduler(self.active_scheduler, self.device)
            actual = self._identity()
            if actual != self.phase_identity:
                raise RuntimeError("initial/conditioning/base scheduler identity changed across VAE phase")
            self.phase = "TRANSFORMER"
            row = dict(status="COMPLETED", transition="vae_to_transformer", transformer_resident=str(self.device),
                       vae_resident="absent", scheduler_resident=str(self.device), identity_verified=True)
            row.update(elapsed_seconds=time.perf_counter() - started, resources=_resources(self.device))
            self.event(label, row)
            return row
        except Exception as exc:
            self.event(label, {"status": "FAILED", "transition": "vae_to_transformer", "reason": f"{type(exc).__name__}: {exc}", "retry": False,
                               "elapsed_seconds": time.perf_counter() - started, "resources": _resources(self.device)})
            raise

    def stop_in_vae_phase(self, label: str, *, primary_reason: str | None = None) -> dict[str, Any]:
        """Best-effort VAE cleanup without restoring the large transformer.

        Used after provider/final-decode failure and after the final successful
        decode.  Cleanup errors are records and never replace a primary error.
        """
        import torch

        started = time.perf_counter()
        row = dict(status="ATTEMPTED", transition="vae_release_without_transformer_restore",
                   primary_reason=primary_reason, retry=False, resources=_resources(self.device), cleanup_record_errors=[])
        try:
            self.event(label, dict(row))
        except Exception as exc:
            row["cleanup_record_errors"].append(f"attempt_event: {type(exc).__name__}: {exc}")
        try:
            if self.vae is not None:
                vae_adapter._clear_cache(self.vae)
            self.backend = self.vae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.phase = "STOPPED_AFTER_VAE"
            row.update(status="COMPLETED", transformer_resident="cpu", vae_resident="absent",
                       elapsed_seconds=time.perf_counter() - started, resources=_resources(self.device))
        except Exception as exc:
            self.phase = "FAILED_CLEANUP"
            row.update(status="FAILED", cleanup_reason=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.perf_counter() - started, resources=_resources(self.device))
        try:
            self.event(label, dict(row))
        except Exception as exc:
            row["cleanup_record_errors"].append(f"completion_event: {type(exc).__name__}: {exc}")
        return row

    def release(self) -> None:
        import torch

        if self.vae is not None:
            vae_adapter._clear_cache(self.vae)
        self.vae = self.backend = None
        if self.pipe is not None:
            self.pipe.transformer = None
            self.pipe.text_encoder = None
            self.pipe.vae = None
        self.pipe = self.initial = self.prompt = self.negative = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.phase = "RELEASED"


class _DynamicBackend:
    def __init__(self, residency: WanSerialResidency):
        self.residency = residency

    def decode_normalized(self, value: Any) -> Any:
        if self.residency.backend is None:
            raise RuntimeError("decode outside VAE residency phase")
        return self.residency.backend.decode_normalized(value)

    def encode_normalized(self, value: Any) -> Any:
        if self.residency.backend is None:
            raise RuntimeError("encode outside VAE residency phase")
        return self.residency.backend.encode_normalized(value)


def _best_effort_stop(residency: Any, label: str, reason: str) -> None:
    """Never replace the provider/decode exception with cleanup bookkeeping."""
    try:
        residency.stop_in_vae_phase(label, primary_reason=reason)
    except Exception as cleanup_exc:
        try:
            residency.event(label, dict(status="FAILED", transition="failure_cleanup",
                            primary_reason=reason,
                            cleanup_reason=f"{type(cleanup_exc).__name__}: {cleanup_exc}", retry=False))
        except Exception:
            pass


def make_resident_control(residency: WanSerialResidency, config: dict[str, Any]) -> tuple[Any, joint.LocalJointPosteriorProvider]:
    """Bind the reviewed provider to VAE residency around enabled steps only."""
    from runtime.wan import local_joint_state_payload_v1 as adapter

    spec, generation = config["carrier"], config["generation"]
    provider = joint.LocalJointPosteriorProvider(
        backend=_DynamicBackend(residency), key=spec["key"], message=bytes.fromhex(spec["message_hex"]),
        rho=float(spec["rho"]), cap=float(spec["cap"]),
    )
    base = adapter.make_control_step(
        windows=joint.nominal_video_windows(),
        state_spec={"kind": "balanced_rm_1_5_public_key_order"},
        payload_spec={"kind": "four_exact_bytes_msb_fragments"},
        requested_budget={"rho": float(spec["rho"]), "conditional_clean_l2_cap": float(spec["cap"]), "parameter_status": "explicit_run_config"},
        provider=provider, guidance=float(generation["guidance_scale"]),
    )

    def control(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        index = int(kwargs["index"])
        if index < 25:
            return base(**kwargs)
        residency.enter_vae(f"joint_step_{index:02d}_enter_vae")
        try:
            result = base(**kwargs)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            try:
                residency.event(f"joint_step_{index:02d}_provider", dict(status="FAILED", reason=reason, retry=False,
                                 backend_calls=dict(provider.calls), backend_failures=list(provider.failures), resources=_resources(residency.device)))
            except Exception:
                pass
            _best_effort_stop(residency, f"joint_step_{index:02d}_failure_cleanup", reason)
            raise
        else:
            residency.event(f"joint_step_{index:02d}_provider", dict(status="COMPLETED",
                             backend_calls=dict(provider.calls), backend_failures=list(provider.failures), resources=_resources(residency.device)))
            residency.leave_vae(f"joint_step_{index:02d}_leave_vae")
        return result

    return control, provider


def run_arm(residency: WanSerialResidency, config: dict[str, Any], arm: str, *, count: Callable, record_step: Callable) -> tuple[Any, dict[str, Any], Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    scheduler = residency.pristine_scheduler()
    residency.active_scheduler = scheduler
    if arm == "OFF":
        injected, provider = None, None
    else:
        injected, provider = make_resident_control(residency, config)
    # OFF uses an explicit zero injected control so it never falls into legacy GROW target math.
    if arm == "OFF":
        import torch

        def injected(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
            c, u = kwargs["conditional"], kwargs["unconditional"]
            guidance = float(config["generation"]["guidance_scale"])
            return u + guidance * (c - u), dict(enabled=False, composition="explicit_zero_joint_control", joint_delta_l2=0.0)
    try:
        terminal, receipt = grow.run_trajectory(
            residency.pipe, residency.initial, scheduler, residency.prompt, residency.negative,
            residency.dtype, arm, "unused", (), count, record_step,
            injected_control=injected,
        )
    finally:
        residency.active_scheduler = None
    return terminal, receipt, provider


def save_terminal_and_rgb_layers(residency: WanSerialResidency, terminal: Any, output: Path,
                                 *, event: Callable[[str, dict[str, Any]], None],
                                 restore_transformer: bool,
                                 public_protocol: carrier.CarrierProtocol = carrier.PUBLIC,
                                 raster_saver: Callable[[Any, Path], dict[str, Any]] | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Save normalized terminal, adapter-clamped float RGB, and its exact RGB8 raster."""
    import torch
    from runtime.wan import fixed_rgb_media

    layers: dict[str, Any] = {}
    layers["terminal_latent"] = _tensor_file(output / "terminal_latent.pt", terminal)
    event("terminal_latent", layers["terminal_latent"])
    residency.active_scheduler = residency.pipe.scheduler
    residency.enter_vae("terminal_enter_vae")
    started = time.perf_counter()
    residency.event("terminal_decode", dict(status="ATTEMPTED", resources=_resources(residency.device)))
    try:
        decoded = residency.backend.decode_normalized(terminal.to(residency.device))
        if (not torch.is_tensor(decoded) or decoded.dtype != torch.float32 or
                tuple(decoded.shape) != public_protocol.video_shape):
            raise ValueError("terminal decoder must return FP32 public RGB geometry")
        if not bool(torch.isfinite(decoded).all()):
            raise FloatingPointError("terminal decoder returned nonfinite FP32 RGB")
        if bool((decoded < 0).any()) or bool((decoded > 1).any()):
            raise ValueError("terminal decoder output must be adapter-clamped to [0,1]")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        try:
            residency.event("terminal_decode", dict(status="FAILED", reason=reason, retry=False))
        except Exception:
            pass
        _best_effort_stop(residency, "terminal_failure_cleanup", reason)
        residency.active_scheduler = None
        raise
    residency.event("terminal_decode", dict(status="COMPLETED", elapsed_seconds=time.perf_counter() - started,
                    resources=_resources(residency.device)))
    if restore_transformer:
        residency.leave_vae("terminal_leave_vae")
    else:
        residency.stop_in_vae_phase("terminal_vae_release")
    residency.active_scheduler = None
    layers["float_rgb"] = _tensor_file(output / "float_rgb.pt", decoded)
    layers["float_rgb"]["coordinate"] = "vae_adapter_clamped_rgb_0_1"
    event("float_rgb", layers["float_rgb"])
    rgb8 = vae_adapter.quantize_rgb8_no_codec(decoded)
    save = fixed_rgb_media.save_raster if raster_saver is None else raster_saver
    layers["rgb8"] = save(rgb8, output / "rgb8.rgb")
    event("rgb8", layers["rgb8"])
    return decoded.detach().cpu(), rgb8, layers


class ExplicitFFmpeg:
    """One no-retry RGB24 round trip using only explicit media config."""
    def __init__(self, media: dict[str, Any], *, raster_loader: Callable | None = None):
        self.media = dict(media)
        self.raster_loader = raster_loader

    def roundtrip(self, raster_path: Path, expected_sha256: str, path: Path,
                  *, event: Callable[[dict[str, Any]], None]) -> tuple[Any, dict[str, Any]]:
        import numpy as np
        import torch
        from runtime.wan import fixed_rgb_media

        raster_path = Path(raster_path)
        reopen = fixed_rgb_media.reopen_raster if self.raster_loader is None else self.raster_loader
        rgb8 = reopen(raster_path, expected_sha256)
        t, h, w, _ = carrier.PUBLIC.video_shape
        partial = path.with_name(path.stem + ".partial.mp4")
        if path.exists() or partial.exists():
            raise FileExistsError("MP4 output already exists")
        raw = rgb8.detach().cpu().contiguous().numpy().tobytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        identity_matches = actual_sha256 == expected_sha256 if expected_sha256 is not None else None
        save = ["ffmpeg", "-v", "error", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.media["fps"]), "-i", "pipe:0", "-an", "-c:v", self.media["codec"], "-crf", str(self.media["crf"]), "-pix_fmt", self.media["pixel_format"], "-n", str(partial)]
        row = {"stage": "encode", "status": "ATTEMPTED", "command": save,
               "input_raster_path": str(raster_path), "input_expected_sha256": expected_sha256,
               "input_actual_sha256": actual_sha256, "input_bytes_sha256": actual_sha256,
               "input_sha256_matches": identity_matches, "retry": False}
        event(dict(row))
        child = None
        try:
            child = subprocess.run(save, input=raw, capture_output=True, check=False)
            if child.returncode:
                raise RuntimeError(f"FFmpeg encode exited {child.returncode}")
            os.replace(partial, path)
            artifact = dict(path=str(path), sha256=_sha(path), bytes=path.stat().st_size)
            row.update(status="COMPLETED", returncode=0, artifact=artifact,
                       stdout=child.stdout.decode(errors="replace"), stderr=child.stderr.decode(errors="replace"))
            event(dict(row))
        except Exception as exc:
            row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            if child is not None:
                row.update(returncode=child.returncode, stdout=child.stdout.decode(errors="replace"),
                           stderr=child.stderr.decode(errors="replace"))
            event(dict(row)); raise
        probe_command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(path)]
        probe = {"stage": "probe", "status": "ATTEMPTED", "command": probe_command, "mp4_sha256": artifact["sha256"]}
        event(dict(probe))
        child = None
        try:
            child = subprocess.run(probe_command, capture_output=True, check=False)
            metadata = json.loads(child.stdout)
            stream = metadata["streams"][0]
            if child.returncode or (int(stream["width"]), int(stream["height"])) != (w, h):
                raise ValueError("ffprobe width/height mismatch")
        except Exception as exc:
            probe.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            if child is not None:
                probe.update(returncode=child.returncode, stdout=child.stdout.decode(errors="replace"),
                             stderr=child.stderr.decode(errors="replace"))
            event(dict(probe)); raise RuntimeError("FFprobe failed or reported wrong geometry") from exc
        probe.update(status="COMPLETED", returncode=0, metadata=metadata,
                     stdout=child.stdout.decode(errors="replace"), stderr=child.stderr.decode(errors="replace"))
        event(dict(probe))
        read = ["ffmpeg", "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        read_row = {"stage": "read", "status": "ATTEMPTED", "command": read, "mp4_sha256": artifact["sha256"]}
        event(dict(read_row))
        child = None
        try:
            child = subprocess.run(read, capture_output=True, check=False)
            if child.returncode or len(child.stdout) != t * h * w * 3:
                raise RuntimeError("FFmpeg readback failed or changed fixed geometry")
        except Exception as exc:
            read_row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            if child is not None:
                read_row.update(returncode=child.returncode, read_bytes=len(child.stdout),
                                stderr=child.stderr.decode(errors="replace"))
            event(dict(read_row)); raise
        received = torch.from_numpy(np.frombuffer(child.stdout, dtype=np.uint8).reshape(carrier.PUBLIC.video_shape).copy())
        read_row.update(status="COMPLETED", returncode=0, read_bytes=len(child.stdout),
                        stderr=child.stderr.decode(errors="replace"), full_frame_count_by_raw_bytes=t)
        event(dict(read_row))
        return received, dict(status="SAVED", artifact=artifact, input_raster_path=str(raster_path),
                              input_expected_sha256=expected_sha256, input_actual_sha256=actual_sha256,
                              input_sha256_matches=identity_matches,
                              encode=row, probe=probe, read=read_row)


def raw_observations(rgb: Any, key: str, *, public_protocol: carrier.CarrierProtocol = carrier.PUBLIC) -> list[dict[str, Any]]:
    """Serialize received-only raw rows; no reducer, truth, threshold, or path selection."""
    values = rgb.float() / 255.0 if str(rgb.dtype).endswith("uint8") else rgb.float()
    return [row.to_dict() for row in joint.read_received_catalog(values, key, public_protocol=public_protocol)]
