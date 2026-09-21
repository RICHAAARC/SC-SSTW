"""One no-gradient Wan terminal-latent generation reused by C2A.

This is the small, endpoint-only part of the prior public-statistic loader:
it retains the WanPipeline prompt/latent/scheduler path and its independently
loaded FP32 VAE, but drops feedback, autograd, checkpointing, and the old
three-arm controller.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any


@dataclass
class GeneratedTerminal:
    normalized_latent: Any
    vae: Any
    metadata: dict[str, Any]


def _load_args(model: dict[str, Any], *, torch_dtype: Any, subfolder: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"torch_dtype": torch_dtype}
    if subfolder is not None:
        args["subfolder"] = subfolder
    if model.get("revision"):
        args["revision"] = model["revision"]
    return args


def load_frozen_vae(config: dict[str, Any]) -> Any:
    """Load only the fixed FP32 Wan VAE for a C2A readout.

    This intentionally does not instantiate ``WanPipeline`` or a transformer.
    """

    import torch
    from diffusers import AutoencoderKLWan

    model = config["model"]
    vae = AutoencoderKLWan.from_pretrained(model["id"], **_load_args(model, torch_dtype=torch.float32, subfolder="vae")).eval()
    disable = getattr(vae, "disable_gradient_checkpointing", None)
    if callable(disable):
        disable()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae.to(torch.device("cuda"))


def prepare_generation(config: dict[str, Any], *, load_vae: bool = True):
    """Prepare a fresh prompt-conditioned Wan noise state and native scheduler.

    ``load_vae=False`` is the generation-worker path: the transformer process
    owns no VAE, and the later media worker loads the VAE independently.
    """

    import torch
    from diffusers import WanPipeline

    model, generation = config["model"], config["generation"]
    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(
        model["id"],
        **_load_args(model, torch_dtype=torch.bfloat16),
        **({} if load_vae else {"vae": None}),
    )
    if (
        getattr(pipe, "transformer_2", None) is not None
        or getattr(pipe.config, "boundary_ratio", None) is not None
        or getattr(pipe.config, "expand_timesteps", False)
    ):
        raise ValueError("the shared adapter supports the fixed single-transformer Wan path only")
    pipe.text_encoder.to(device)
    with torch.no_grad():
        prompt, negative = pipe.encode_prompt(
            prompt=generation["prompt"],
            negative_prompt=generation["negative_prompt"],
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=generation["max_sequence_length"],
            device=device,
        )
    prompt, negative = prompt.to(torch.bfloat16), negative.to(torch.bfloat16)
    pipe.text_encoder = None
    pipe.vae = None
    gc.collect()
    torch.cuda.empty_cache()
    if load_vae:
        vae = load_frozen_vae(config)
        pipe.vae = vae
        pipe.vae_scale_factor_temporal = getattr(vae.config, "scale_factor_temporal", None) or 2 ** sum(vae.config.temperal_downsample)
        pipe.vae_scale_factor_spatial = getattr(vae.config, "scale_factor_spatial", None) or 2 ** len(vae.config.temperal_downsample)
    if (
        generation["height"] % pipe.vae_scale_factor_spatial
        or generation["width"] % pipe.vae_scale_factor_spatial
        or (generation["frames"] - 1) % pipe.vae_scale_factor_temporal
    ):
        raise ValueError("the fixed Wan VAE cannot represent the configured video geometry")
    pipe.transformer.eval()
    disable = getattr(pipe.transformer, "disable_gradient_checkpointing", None)
    if callable(disable):
        disable()
    for parameter in pipe.transformer.parameters():
        parameter.requires_grad_(False)
    pipe.transformer.to(device)
    with torch.no_grad():
        latent = pipe.prepare_latents(
            1,
            int(pipe.transformer.config.in_channels),
            generation["height"],
            generation["width"],
            generation["frames"],
            torch.float32,
            device,
            torch.Generator(device=device).manual_seed(generation["seed"]),
            None,
        ).detach()
    pipe.scheduler.set_timesteps(generation["steps"], device=device)
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(0)
    weight = getattr(getattr(pipe.transformer, "patch_embedding", None), "weight", None)
    input_dtype = weight.dtype if weight is not None else next(pipe.transformer.parameters()).dtype
    return pipe, latent, prompt, negative, input_dtype


def generate_terminal_latent(config: dict[str, Any], progress: Any = None) -> GeneratedTerminal:
    """Generate one normalized terminal latent by the existing Wan step path."""

    import torch

    pipe, latent, prompt, negative, input_dtype = prepare_generation(config)
    model, generation, vae = config["model"], config["generation"], pipe.vae
    transformer_calls = 0
    transformer_attempted = 0
    def record_forward(completed: bool) -> None:
        nonlocal transformer_calls, transformer_attempted
        if completed:
            transformer_calls += 1
        else:
            transformer_attempted += 1
        if progress is not None:
            progress({"transformer_attempted": transformer_attempted, "transformer_completed": transformer_calls})
    with torch.no_grad():
        for index, timestep in enumerate(pipe.scheduler.timesteps):
            hidden = latent.to(input_dtype)
            time = timestep.expand(latent.shape[0])
            record_forward(False)
            conditional = pipe.transformer(hidden_states=hidden, timestep=time, encoder_hidden_states=prompt, attention_kwargs=None, return_dict=False)[0]
            record_forward(True)
            record_forward(False)
            unconditional = pipe.transformer(hidden_states=hidden, timestep=time, encoder_hidden_states=negative, attention_kwargs=None, return_dict=False)[0]
            record_forward(True)
            velocity = unconditional + generation["guidance_scale"] * (conditional - unconditional)
            latent = pipe.scheduler.step(velocity, timestep, latent, return_dict=False)[0]
            if hasattr(pipe.scheduler, "step_index") and pipe.scheduler.step_index not in (None, index + 1):
                raise RuntimeError("Wan scheduler cursor diverged during C2A terminal generation")
    metadata = {
        "generation_entry": "reused_public_statistic_single_transformer_wan_endpoint",
        "model": model,
        "generation": generation,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "terminal_latent_shape": list(latent.shape),
        "terminal_latent_dtype": str(latent.dtype),
        "transformer_dtype": str(input_dtype),
        "vae_dtype": str(next(vae.parameters()).dtype),
        "transformer_calls": transformer_calls,
        "generation_invocations": 1,
    }
    return GeneratedTerminal(latent.detach(), vae, metadata)
