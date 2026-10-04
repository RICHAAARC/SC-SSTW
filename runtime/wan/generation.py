"""Frozen Wan model loaders for the successful native GROW reference chain."""
from __future__ import annotations
import gc
from typing import Any

def _load_args(model: dict[str, Any], *, torch_dtype: Any, subfolder: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"torch_dtype": torch_dtype}
    if subfolder is not None:
        args["subfolder"] = subfolder
    if model.get("revision"):
        args["revision"] = model["revision"]
    return args


def load_frozen_vae(config: dict[str, Any], *, device: Any = None) -> Any:
    """Load only the fixed FP32 Wan VAE for a readout.

    Optional device preserves the legacy CUDA default for existing callers.

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
    return vae.to(torch.device("cuda") if device is None else torch.device(device))


def prepare_generation(config: dict[str, Any], *, load_vae: bool = True,
                       device: Any = None, model_dtype: Any = None):
    """Prepare a fresh prompt-conditioned Wan noise state and native scheduler.

    ``load_vae=False`` is the generation-worker path: the transformer process
    owns no VAE, and the later media worker loads the VAE independently.
    Optional device/model_dtype preserve old CUDA/BF16 defaults; callers can select
    CPU/FP32 without a hardware-name gate.
    """

    import torch
    from diffusers import WanPipeline

    model, generation = config["model"], config["generation"]
    device = torch.device("cuda") if device is None else torch.device(device)
    model_dtype = torch.bfloat16 if model_dtype is None else model_dtype
    pipe = WanPipeline.from_pretrained(
        model["id"],
        **_load_args(model, torch_dtype=model_dtype),
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
    prompt, negative = prompt.to(model_dtype), negative.to(model_dtype)
    pipe.text_encoder = None
    pipe.vae = None
    gc.collect()
    torch.cuda.empty_cache()
    if load_vae:
        vae = load_frozen_vae(config, device=device)
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
