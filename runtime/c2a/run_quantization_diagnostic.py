"""Three-arm RGB8-only C2A diagnostic over one persisted second-axis package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.c2a.chain import quantize_rgb8_no_codec, reencode_rgb24_readback
from runtime.c2a.generation import load_frozen_vae
from runtime.c2a.protocol import C2AConfig, read_q


ARMS = ("ZERO", "PLUS_E2", "MINUS_E2")
FIXED_INPUT = "/content/drive/MyDrive/Video-WM/C2A_SecondAxis_Diagnostic/c2a_second_axis_20260914T145106Z"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _torch() -> Any:
    import torch

    return torch


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model, source, c2a = config.get("model", {}), config.get("source", {}), config.get("c2a", {})
    if not isinstance(model.get("id"), str) or not model["id"] or model["id"].startswith("REQUIRED_"):
        raise ValueError("quantization diagnostic requires a concrete model identifier")
    if model.get("revision") is not None and not isinstance(model.get("revision"), str):
        raise ValueError("model revision is a record string or null")
    if source.get("second_axis_run") != FIXED_INPUT:
        raise ValueError("quantization diagnostic fixes the already-persisted second-axis input path")
    if list(source.get("arms", ())) != list(ARMS):
        raise ValueError("quantization diagnostic arm roster is fixed")
    if c2a.get("channels_zero_based") != [0, 1] or c2a.get("ordinary_groups") != [1, 2, 3] or c2a.get("read_group") != 2:
        raise ValueError("C2A read support is fixed")
    if c2a.get("central_block") != "floor((H-8)/2):+8, floor((W-8)/2):+8":
        raise ValueError("C2A central support is fixed")
    if c2a.get("beta") != 0.25 or c2a.get("rho") != 0.5 or c2a.get("posterior") != "mode" or c2a.get("cache_reset") != "before and after each encode":
        raise ValueError("C2A carrier, cache, and posterior semantics are fixed")
    return config


def _summary(q_by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    output: dict[str, Any] = {"ratio_policy": "O2/M2 ratios are not reported; O2 and M2 vectors/norms remain explicit."}
    for layer in ("source_pre_codec", "rgb8_quantized", "source_post_mp4"):
        zero, plus, minus = (q_by_arm[arm][layer].to(dtype=torch.float64) for arm in ARMS)
        odd = (plus - minus) / 2.0
        midpoint = (plus + minus) / 2.0 - zero
        output[layer] = {"O2": odd.detach().cpu().tolist(), "M2": midpoint.detach().cpu().tolist(), "O2_l2": float(odd.norm().item()), "M2_l2": float(midpoint.norm().item())}
    return output


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing quantization diagnostic output: {output}")
    source_root = Path(config["source"]["second_axis_run"])
    source_config_path, source_result_path = source_root / "config.json", source_root / "result.json"
    required = [source_config_path, source_result_path, *(source_root / "pre_codec_float_rgb" / f"{arm}.pt" for arm in ARMS)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fixed quantization inputs missing: {missing}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_config.get("model") != config["model"]:
        raise ValueError("fixed input model record differs from quantization configuration")
    if list(source_config.get("c2a", {}).get("arms", ())) != list(ARMS):
        raise ValueError("fixed input is not the required three-arm second-axis package")

    output.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "SETUP_RUNNING",
        "science_denominator": 0,
        "diagnostic_scope": "RGB8 np.rint/clamp/255 only; no FFmpeg, color conversion, video compression, generation, transformer, or VAE decode",
        "source": {"second_axis_run": str(source_root), "config": "config.json", "result": "result.json", "pre_codec_float_rgb": "pre_codec_float_rgb/<arm>.pt"},
        "arms": {arm: {"status": "NOT_STARTED"} for arm in ARMS},
        "calls": {"generation_invocations": 0, "transformer_forwards": 0, "vae_loads": 0, "vae_decode_attempts": 0, "vae_encode_attempts": 0},
        "acceptance": config["acceptance"],
    }
    _write_json(output / "config.json", config)
    try:
        vae = load_frozen_vae(config)
        result["calls"]["vae_loads"] = 1
    except BaseException as exc:
        result["status"] = "FATAL_SETUP"
        result["setup_error"] = repr(exc)
        _write_json(output / "result.json", result)
        raise

    torch = _torch()
    c2a = C2AConfig()
    q_by_arm: dict[str, dict[str, Any]] = {}
    result["status"] = "EXECUTED_REQUIRES_REVIEW"
    for arm in ARMS:
        row = result["arms"][arm]
        row["status"] = "PENDING"
        try:
            source_rgb = torch.load(source_root / "pre_codec_float_rgb" / f"{arm}.pt", map_location="cpu")
            if source_rgb.ndim != 4 or source_rgb.shape[-1] != 3:
                raise ValueError("persisted source float RGB has incompatible shape")
            source_q = source_result["arms"][arm]["q"]
            pre_q = torch.tensor(source_q["pre_codec"], dtype=torch.float64)
            post_q = torch.tensor(source_q["post_mp4"], dtype=torch.float64)
            pixels = quantize_rgb8_no_codec(source_rgb)
            pixels_path = output / "quantized_rgb8_uint8" / f"{arm}.pt"
            pixels_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pixels, pixels_path)
            quantized_rgb = pixels.to(dtype=torch.float32) / 255.0
            difference = quantized_rgb.to(dtype=torch.float64) - source_rgb.detach().cpu().to(dtype=torch.float64)
            result["calls"]["vae_encode_attempts"] += 1
            q = read_q(reencode_rgb24_readback(vae, quantized_rgb), config=c2a)
            quantized_q = q.detach().cpu().to(dtype=torch.float64)
            row["input_float_rgb"] = {"path": str((source_root / "pre_codec_float_rgb" / f"{arm}.pt")), "shape": list(source_rgb.shape), "dtype": str(source_rgb.dtype)}
            row["quantized_rgb8_uint8"] = {"path": str(pixels_path.relative_to(output)), "shape": list(pixels.shape), "dtype": str(pixels.dtype), "bytes": pixels.nelement() * pixels.element_size()}
            row["rgb8_pixel_error_not_q"] = {"mse": float(difference.square().mean().item()), "max_abs": float(difference.abs().max().item())}
            row["q"] = {"source_pre_codec": pre_q.tolist(), "rgb8_quantized": quantized_q.tolist(), "source_post_mp4": post_q.tolist(), "rgb8_minus_source_pre_codec": (quantized_q - pre_q).tolist(), "rgb8_minus_source_post_mp4": (quantized_q - post_q).tolist()}
            q_by_arm[arm] = {"source_pre_codec": pre_q, "rgb8_quantized": quantized_q, "source_post_mp4": post_q}
            row["status"] = "COMPLETE"
        except BaseException as exc:
            row["status"] = "FAILED"
            row["error"] = repr(exc)
        _write_json(output / "result.json", result)

    if set(q_by_arm) == set(ARMS):
        result["response_comparison"] = _summary(q_by_arm)
    else:
        result["response_comparison"] = {"status": "NOT_COMPUTED_INCOMPLETE_FIXED_DENOMINATOR", "available_arms": sorted(q_by_arm)}
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("prepared only: --execute is the sole real-work entrypoint")
    run(_load_config(args.config), args.output)


if __name__ == "__main__":
    main()
