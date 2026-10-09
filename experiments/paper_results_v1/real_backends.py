"""Concrete lazy loaders for the staged Paper Results V1 evaluation.

Nothing in this module loads a model at import time.  Every loader requires an
explicit local source tree and checkpoint/config paths; there is no download or
model-name fallback.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from experiments.paper_results_v1.native_adapters import (
    RivaGANLoadedTensorBackend,
    RivaGANNativeAdapter,
    VideoSealNativeAdapter,
)


class RealBackendError(RuntimeError):
    """An explicit local backend does not satisfy its declared interface."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_local_file(path, *, expected_sha256=None, label="file"):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    try:
        actual = file_sha256(resolved)
    except OSError:
        # Digest collection is provenance only. The actual parser/model loader
        # below remains responsible for deciding whether the file is usable.
        actual = None
    return resolved, actual


def _digest_status(expected, actual):
    if actual is None:
        return "OBSERVATION_UNAVAILABLE"
    if not expected:
        return "UNDECLARED"
    return "MATCH" if expected == actual else "RECORDED_DIFFERENCE"


def observe_source_identity(root, expected_commit=None):
    """Record an available source tree's Git identity without gating its use."""

    observation = {
        "expected_commit": expected_commit,
        "actual_commit": None,
        "dirty": None,
        "status": "GIT_IDENTITY_UNAVAILABLE",
    }
    if not (Path(root) / ".git").exists():
        return observation
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True, capture_output=True, check=False,
        )
    except OSError as exc:
        observation.update(status="GIT_IDENTITY_QUERY_FAILED", reason=f"{type(exc).__name__}: {exc}")
        return observation
    if head.returncode or dirty.returncode:
        observation.update(
            status="GIT_IDENTITY_QUERY_FAILED",
            reason=(head.stderr or dirty.stderr).strip() or "git identity query failed",
        )
        return observation
    actual = head.stdout.strip()
    is_dirty = bool(dirty.stdout)
    observation.update(
        actual_commit=actual,
        dirty=is_dirty,
        status=(
            "MATCH_CLEAN"
            if expected_commit and actual == expected_commit and not is_dirty
            else "RECORDED_DIFFERENCE"
        ),
    )
    return observation


def require_source_root(path, *, package_path, label):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir() or not (resolved / package_path).exists():
        raise FileNotFoundError(f"{label} source root missing {package_path}: {resolved}")
    return resolved


def require_local_snapshot(path, *, required, label):
    resolved = Path(path).expanduser().resolve()
    missing = [name for name in required if not (resolved / name).exists()]
    if not resolved.is_dir() or missing:
        raise FileNotFoundError(f"{label} local snapshot missing {missing or ['directory']}: {resolved}")
    return resolved


@contextlib.contextmanager
def _source_import(root, *, working_directory=None):
    root = str(Path(root).resolve())
    old_path = list(sys.path)
    old_cwd = Path.cwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    if working_directory is not None:
        os.chdir(working_directory)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


class VideoSealRGB8Backend:
    """VideoSeal model call-through for full RGB uint8 frame tensors.

    The wrapper follows the current official ``embed`` and ``detect`` calls.
    It does not call ``extract_message`` or choose a capacity reducer.
    """

    def __init__(self, model, *, torch_module, device, lowres_attenuation):
        self.model = model
        self.torch = torch_module
        self.device = device
        self.lowres_attenuation = bool(lowres_attenuation)

    def build_message(self, bits):
        return self.torch.tensor([list(bits)], dtype=self.torch.float32, device=self.device)

    def _input(self, media):
        value = media
        if not hasattr(value, "shape"):
            value = self.torch.as_tensor(value)
        if tuple(int(part) for part in value.shape)[-1:] != (3,) or len(value.shape) != 4:
            raise RealBackendError("VideoSeal RGB8 media must have shape [T,H,W,3]")
        if value.dtype != self.torch.uint8:
            raise RealBackendError("VideoSeal RGB8 media must use torch.uint8")
        return value.permute(0, 3, 1, 2).to(device=self.device, dtype=self.torch.float32).div(255.0)

    def _rgb8(self, value):
        if value.ndim != 4 or int(value.shape[1]) != 3:
            raise RealBackendError("VideoSeal imgs_w must have shape [T,3,H,W]")
        if not bool(self.torch.isfinite(value).all()):
            raise RealBackendError("VideoSeal imgs_w contains non-finite values")
        # Current official inference uses multiply, byte truncation, then THWC.
        return value.detach().clamp(0.0, 1.0).mul(255.0).to(self.torch.uint8).permute(0, 2, 3, 1).cpu().contiguous()

    def embed(self, media, prepared_message, output_uri=None):
        del output_uri
        with self.torch.inference_mode():
            output = self.model.embed(
                self._input(media), msgs=prepared_message, is_video=True,
                lowres_attenuation=self.lowres_attenuation,
            )
        if not isinstance(output, Mapping) or "imgs_w" not in output:
            raise RealBackendError("VideoSeal embed output lacks imgs_w")
        return {**output, "imgs_w": self._rgb8(output["imgs_w"])}

    @staticmethod
    def embedded_media(output):
        return output["imgs_w"]

    def extract(self, media):
        with self.torch.inference_mode():
            return self.model.detect(self._input(media), is_video=True)


class LocalFramewiseBackend:
    """Frozen framewise VAE loaded from an explicit local snapshot only."""

    def __init__(self, config):
        snapshot = require_local_snapshot(
            config["local_snapshot_path"],
            required=("config.json", "diffusion_pytorch_model.safetensors"),
            label="framewise VAE",
        )
        torch = importlib.import_module("torch")
        AutoencoderKL = importlib.import_module("diffusers").AutoencoderKL
        from runtime.wan import framewise_autoencoder_kl as framewise

        self.torch = torch
        self.framewise = framewise
        self.batch_frames = config["batch_frames"]
        self.device = torch.device(config["device"])
        self.vae = AutoencoderKL.from_pretrained(
            str(snapshot), torch_dtype=torch.float32, use_safetensors=True,
            local_files_only=True,
        )
        self.vae.requires_grad_(False)
        self.vae.eval().to(self.device)
        framewise.verify_scaling_factor(self.vae)

    def encode(self, rgb):
        return self.framewise.encode_rgb_frames(
            self.vae, rgb.float().div(255), batch_frames=self.batch_frames,
        ).numpy()

    @staticmethod
    def score(latent, key, protocol):
        from main.tube_state import video_trajectory_conditional_joint_v1 as method
        from main.tube_state import video_trajectory_receiver_estimated_align_v1 as receiver

        if protocol == "GLOBAL":
            return receiver.sync.score_received_latent(latent, key, receiver.PUBLIC)
        if protocol == "SINGLE_JUMP":
            return method.deletion.score(latent, key)
        raise ValueError("public protocol")

    @staticmethod
    def write(latent, key):
        from main.tube_state import video_trajectory_receiver_estimated_align_v1 as receiver

        return receiver.sync.apply_projection_margin(
            latent, key, receiver.sync.PUBLIC, target_margin=0.5,
        )

    def decode(self, latent):
        from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

        value = self.torch.from_numpy(latent.copy())
        rgb = self.framewise.decode_rgb_frames(
            self.vae, value, batch_frames=self.batch_frames,
        )
        return runtime.vae.quantize_rgb8_no_codec(rgb)

    def close(self):
        self.vae = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def load_local_framewise_backend(config):
    return LocalFramewiseBackend(config)


def _videoseal_model_message_length(model):
    embedder = getattr(model, "embedder", None)
    processor = getattr(embedder, "msg_processor", None)
    value = getattr(processor, "nbits", None)
    return value if type(value) is int and value > 0 else None


def load_videoseal_adapter(config, *, native_output_store):
    """Load VideoSeal from a local card and local checkpoint only."""

    source = require_source_root(
        config["source_root"], package_path="videoseal/__init__.py", label="VideoSeal",
    )
    source_identity = observe_source_identity(source, config.get("source_commit"))
    card, card_sha = require_local_file(
        config["card_path"], expected_sha256=config.get("card_sha256"), label="VideoSeal card",
    )
    checkpoint, checkpoint_sha = require_local_file(
        config["checkpoint_path"], expected_sha256=config.get("checkpoint_sha256"),
        label="VideoSeal checkpoint",
    )
    with _source_import(source, working_directory=source):
        torch = importlib.import_module("torch")
        OmegaConf = importlib.import_module("omegaconf").OmegaConf
        setup_model = importlib.import_module("videoseal.utils.cfg").setup_model
        card_config = OmegaConf.load(card)
        card_length = int(card_config.args.nbits)
        model = setup_model(card_config, str(checkpoint))
    device = torch.device(config["device"])
    model.eval().to(device)
    model_length = _videoseal_model_message_length(model)
    if model_length is not None and model_length != card_length:
        raise RealBackendError(
            f"VideoSeal model msg_processor nbits {model_length} != card args.nbits {card_length}"
        )
    native_length = card_length
    if native_length != config["native_message_length"]:
        raise RealBackendError(
            f"VideoSeal loaded nbits {native_length} != declared {config['native_message_length']}"
        )
    backend = VideoSealRGB8Backend(
        model, torch_module=torch, device=device,
        lowres_attenuation=config["lowres_attenuation"],
    )
    adapter = VideoSealNativeAdapter(
        backend,
        native_message_length=native_length,
        native_output_store=native_output_store,
        # Real evaluation always uses a lossless sidecar.  Even the usual
        # T x 257 output can fit the generic adapter's inline threshold, but
        # edit-by-edit native records must remain bounded and independently
        # addressable in JSON.
        inline_element_limit=1,
        backend_metadata={
            "source_version": source_identity.get("actual_commit") or config.get("source_commit") or "UNDECLARED",
            "source_identity_observation": source_identity,
            "model_version": config["model_card_name"],
            "weight_identity": checkpoint_sha,
            "detect_output_layout": config["detect_output_layout"],
            "card_path": str(card),
            "card_sha256": card_sha,
            "card_declared_sha256": config.get("card_sha256"),
            "card_sha256_status": _digest_status(config.get("card_sha256"), card_sha),
            "card_args_nbits": card_length,
            "model_msg_processor_nbits": model_length,
            "checkpoint_path": str(checkpoint),
            "checkpoint_declared_sha256": config.get("checkpoint_sha256"),
            "checkpoint_sha256_status": _digest_status(config.get("checkpoint_sha256"), checkpoint_sha),
            "loader": "videoseal.utils.cfg.setup_model(OmegaConf.load(card), local_checkpoint)",
        },
    )
    return adapter


def _install_rivagan_pickle_classes():
    import __main__

    modules = {
        "Adversary": ("rivagan.adversary", "Adversary"),
        "Critic": ("rivagan.adversary", "Critic"),
        "AttentiveDecoder": ("rivagan.attention", "AttentiveDecoder"),
        "AttentiveEncoder": ("rivagan.attention", "AttentiveEncoder"),
        "RivaGAN": ("rivagan.rivagan", "RivaGAN"),
    }
    for name, (module, attribute) in modules.items():
        setattr(__main__, name, getattr(importlib.import_module(module), attribute))


def load_rivagan_adapter(config):
    """Load the explicit local community checkpoint and tensor-level backend."""

    source = require_source_root(
        config["source_root"], package_path="rivagan/rivagan.py", label="RivaGAN",
    )
    source_identity = observe_source_identity(source, config.get("source_commit"))
    checkpoint, checkpoint_sha = require_local_file(
        config["checkpoint_path"], expected_sha256=config.get("checkpoint_sha256"),
        label="RivaGAN checkpoint",
    )
    with _source_import(source):
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        _install_rivagan_pickle_classes()
        # This is an explicitly supplied local legacy pickle, never a URL.  The
        # The unsafe format is disclosed in metadata. Its actual digest is
        # recorded, while a declared digest difference is provenance only.
        model = torch.load(checkpoint, map_location=config["device"], weights_only=False)
    for name in ("encoder", "decoder"):
        module = getattr(model, name, None)
        if module is None:
            raise RealBackendError(f"RivaGAN checkpoint lacks {name}")
        module.eval()
        move = getattr(module, "to", None)
        if callable(move):
            move(config["device"])
    backend = RivaGANLoadedTensorBackend(
        model, tensor_module=torch, array_module=numpy, device=config["device"],
        input_color_layout=config["input_color_layout"],
    )
    return RivaGANNativeAdapter(
        backend,
        backend_metadata={
            "source_version": source_identity.get("actual_commit") or config.get("source_commit") or "UNDECLARED",
            "source_identity_observation": source_identity,
            "model_version": config["model_name"],
            "weight_identity": checkpoint_sha,
            "checkpoint_path": str(checkpoint),
            "checkpoint_declared_sha256": config.get("checkpoint_sha256"),
            "checkpoint_sha256_status": _digest_status(config.get("checkpoint_sha256"), checkpoint_sha),
            "checkpoint_provenance": config["checkpoint_provenance"],
            "checkpoint_format": "LOCAL_LEGACY_TORCH_PICKLE_WEIGHTS_ONLY_FALSE",
            "loader": "torch.load(local_checkpoint,map_location=device,weights_only=False)",
        },
    )


def close_adapter(adapter):
    """Drop explicitly loaded model references and release an available cache."""

    backend = getattr(adapter, "backend", None)
    if backend is not None:
        for owner in (backend, getattr(backend, "model", None)):
            if owner is not None and hasattr(owner, "model"):
                owner.model = None
        if hasattr(backend, "model"):
            backend.model = None
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass
