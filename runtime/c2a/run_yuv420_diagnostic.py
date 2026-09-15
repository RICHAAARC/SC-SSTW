"""Raw YUV420p-only diagnostic over the persisted RGB8 quantization package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.c2a.chain import raw_yuv420p_roundtrip_rgb8, reencode_rgb24_readback
from runtime.c2a.generation import load_frozen_vae
from runtime.c2a.protocol import C2AConfig, read_q


ARMS = ("ZERO", "PLUS_E2", "MINUS_E2")
FIXED_INPUT = "/content/drive/MyDrive/Video-WM/C2A_Quantization_Diagnostic/c2a_quantization_20260915T005839Z"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _torch() -> Any:
    import torch

    return torch


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model, source, c2a = config.get("model", {}), config.get("source", {}), config.get("c2a", {})
    if not isinstance(model.get("id"), str) or not model["id"] or model["id"].startswith("REQUIRED_"):
        raise ValueError("raw YUV420p diagnostic requires a concrete model identifier")
    if model.get("revision") is not None and not isinstance(model.get("revision"), str):
        raise ValueError("model revision is a record string or null")
    if source.get("quantization_run") != FIXED_INPUT or list(source.get("arms", ())) != list(ARMS):
        raise ValueError("raw YUV420p diagnostic fixes one completed RGB8 three-arm input package")
    if (source.get("frames"), source.get("height"), source.get("width"), source.get("fps")) != (49, 320, 512, 8):
        raise ValueError("raw YUV420p diagnostic fixes the source frame geometry and rate")
    if c2a.get("channels_zero_based") != [0, 1] or c2a.get("ordinary_groups") != [1, 2, 3] or c2a.get("read_group") != 2 or c2a.get("central_block") != "floor((H-8)/2):+8, floor((W-8)/2):+8":
        raise ValueError("C2A read support is fixed")
    if c2a.get("beta") != 0.25 or c2a.get("rho") != 0.5 or c2a.get("posterior") != "mode" or c2a.get("cache_reset") != "before and after each encode":
        raise ValueError("C2A carrier, cache, and posterior semantics are fixed")
    return config


def _summary(q_by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    output: dict[str, Any] = {"ratio_policy": "O2/M2 ratios are not reported; O2 and M2 vectors/norms remain explicit."}
    for layer in ("source_pre_codec", "source_rgb8_quantized", "raw_yuv420p_rgb8", "source_post_mp4"):
        zero, plus, minus = (q_by_arm[arm][layer].to(dtype=torch.float64) for arm in ARMS)
        odd = (plus - minus) / 2.0
        midpoint = (plus + minus) / 2.0 - zero
        output[layer] = {"O2": odd.detach().cpu().tolist(), "M2": midpoint.detach().cpu().tolist(), "O2_l2": float(odd.norm().item()), "M2_l2": float(midpoint.norm().item())}
    return output


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing raw YUV420p diagnostic output: {output}")
    source_root = Path(config["source"]["quantization_run"])
    source_config_path, source_result_path = source_root / "config.json", source_root / "result.json"
    required = [source_config_path, source_result_path, *(source_root / "quantized_rgb8_uint8" / f"{arm}.pt" for arm in ARMS)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fixed raw YUV420p inputs missing: {missing}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_config.get("model") != config["model"] or list(source_config.get("source", {}).get("arms", ())) != list(ARMS):
        raise ValueError("fixed input is not the required RGB8 three-arm package")

    output.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "SETUP_RUNNING",
        "science_denominator": 0,
        "diagnostic_scope": "RGB8 -> rawvideo YUV420p -> RGB8 only; no H.264/container, generation, transformer, or VAE decode",
        "source": {"quantization_run": str(source_root), "config": "config.json", "result": "result.json", "quantized_rgb8_uint8": "quantized_rgb8_uint8/<arm>.pt"},
        "arms": {arm: {"status": "NOT_STARTED"} for arm in ARMS},
        "calls": {"generation_invocations": 0, "transformer_forwards": 0, "vae_loads": 0, "vae_decode_attempts": 0, "vae_encode_attempts": 0, "raw_yuv_encode_attempts": 0, "raw_yuv_decode_attempts": 0},
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
            source_rgb8 = torch.load(source_root / "quantized_rgb8_uint8" / f"{arm}.pt", map_location="cpu")
            if tuple(source_rgb8.shape) != (49, 320, 512, 3) or source_rgb8.dtype != torch.uint8:
                raise ValueError("persisted RGB8 source has incompatible shape or dtype")
            source_q = source_result["arms"][arm]["q"]
            pre_q = torch.tensor(source_q["source_pre_codec"], dtype=torch.float64)
            rgb8_q = torch.tensor(source_q["rgb8_quantized"], dtype=torch.float64)
            post_q = torch.tensor(source_q["source_post_mp4"], dtype=torch.float64)
            raw_path = output / "raw_yuv420p" / f"{arm}.yuv"
            result["calls"]["raw_yuv_encode_attempts"] += 1
            result["calls"]["raw_yuv_decode_attempts"] += 1
            roundtrip_rgb8, conversion = raw_yuv420p_roundtrip_rgb8(source_rgb8, raw_path, fps=int(config["source"]["fps"]))
            rgb8_path = output / "roundtrip_rgb8_uint8" / f"{arm}.pt"
            rgb8_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(roundtrip_rgb8, rgb8_path)
            difference = roundtrip_rgb8.to(dtype=torch.float64) - source_rgb8.to(dtype=torch.float64)
            result["calls"]["vae_encode_attempts"] += 1
            q = read_q(reencode_rgb24_readback(vae, roundtrip_rgb8.to(dtype=torch.float32) / 255.0), config=c2a)
            yuv_q = q.detach().cpu().to(dtype=torch.float64)
            row["input_rgb8"] = {"path": str(source_root / "quantized_rgb8_uint8" / f"{arm}.pt"), "shape": list(source_rgb8.shape), "dtype": str(source_rgb8.dtype)}
            row["raw_yuv420p"] = {"path": str(raw_path.relative_to(output)), **conversion}
            row["roundtrip_rgb8_uint8"] = {"path": str(rgb8_path.relative_to(output)), "shape": list(roundtrip_rgb8.shape), "dtype": str(roundtrip_rgb8.dtype), "bytes": roundtrip_rgb8.nelement() * roundtrip_rgb8.element_size()}
            row["rgb8_pixel_error_not_q"] = {"mse": float(difference.square().mean().item()), "max_abs": float(difference.abs().max().item())}
            row["q"] = {"source_pre_codec": pre_q.tolist(), "source_rgb8_quantized": rgb8_q.tolist(), "raw_yuv420p_rgb8": yuv_q.tolist(), "source_post_mp4": post_q.tolist(), "raw_yuv420p_minus_source_rgb8": (yuv_q - rgb8_q).tolist(), "raw_yuv420p_minus_source_post_mp4": (yuv_q - post_q).tolist()}
            q_by_arm[arm] = {"source_pre_codec": pre_q, "source_rgb8_quantized": rgb8_q, "raw_yuv420p_rgb8": yuv_q, "source_post_mp4": post_q}
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
