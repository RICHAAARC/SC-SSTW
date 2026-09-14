"""Three-arm, second-axis C2A diagnostic with persisted causal intermediates.

This is a separate 2A diagnostic entrypoint.  It does not replace the six-arm
interface executor or reinterpret its already-recorded results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.c2a.chain import decode_normalized_latent, ffmpeg_roundtrip, reencode_rgb24_readback
from runtime.c2a.generation import generate_terminal_latent
from runtime.c2a.protocol import (
    C2AConfig,
    actual_write_measurement,
    actual_written_block_snapshots,
    read_q,
    write_covariance_state,
)


SECOND_AXIS_STATES: dict[str, tuple[float, float]] = {
    "ZERO": (0.0, 0.0),
    "PLUS_E2": (0.0, 0.5),
    "MINUS_E2": (0.0, -0.5),
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _torch() -> Any:
    import torch

    return torch


def _save_tensor(path: Path, tensor: Any) -> dict[str, Any]:
    """Persist an exact CPU tensor, preserving its dtype rather than compressing it."""

    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = tensor.detach().to(device="cpu").contiguous().clone()
    torch.save(saved, path)
    return {"shape": list(saved.shape), "dtype": str(saved.dtype), "bytes": saved.element_size() * saved.nelement()}


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model, generation, c2a, codec = config.get("model", {}), config.get("generation", {}), config.get("c2a", {}), config.get("codec", {})
    if not isinstance(model.get("id"), str) or not model["id"] or model["id"].startswith("REQUIRED_"):
        raise ValueError("C2A requires a concrete model identifier")
    if model.get("revision") is not None and not isinstance(model.get("revision"), str):
        raise ValueError("model revision is a record string or null")
    fixed_generation = {
        "prompt": "locked camera, a single matte white cube moving slowly across a dark plain table, stable lighting, no people, no cuts",
        "negative_prompt": "text, watermark, logo, camera motion, cuts, multiple objects, flicker",
        "seed": 1275,
        "height": 320,
        "width": 512,
        "frames": 49,
        "fps": 8,
        "steps": 50,
        "guidance_scale": 5.0,
        "max_sequence_length": 512,
    }
    if any(generation.get(key) != value for key, value in fixed_generation.items()):
        raise ValueError("second-axis diagnostic freezes the prior terminal-generation inputs")
    if list(c2a.get("arms", ())) != list(SECOND_AXIS_STATES):
        raise ValueError("second-axis diagnostic arm roster is fixed and ordered")
    if c2a.get("channels_zero_based") != [0, 1] or c2a.get("ordinary_groups") != [1, 2, 3] or c2a.get("read_group") != 2:
        raise ValueError("C2A support is fixed")
    if c2a.get("beta") != 0.25 or c2a.get("rho") != 0.5 or c2a.get("posterior") != "mode":
        raise ValueError("C2A carrier and posterior are fixed")
    if c2a.get("central_block") != "floor((H-8)/2):+8, floor((W-8)/2):+8" or c2a.get("near_singular_rule") != "reject if lambda_min <= 1e-8 * max(lambda_max, 1); no regularization or pseudoinverse" or c2a.get("cache_reset") != "before and after each decode and encode":
        raise ValueError("C2A spatial, singularity, and cache semantics are fixed")
    if codec.get("name") != "h264_yuv420p" or codec.get("crf") != 18 or codec.get("readback") != "ffmpeg_rgb24" or codec.get("thread_count") != 1:
        raise ValueError("C2A codec/readback contract is fixed")
    return config


def _second_axis_summary(q_by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    summary: dict[str, Any] = {"ratio_policy": "O2/M2 ratios are intentionally not reported; vector values and norms remain visible when O2 is close to zero."}
    for layer in ("pre_codec", "post_mp4"):
        zero, plus, minus = (q_by_arm[arm][layer].to(dtype=torch.float64) for arm in ("ZERO", "PLUS_E2", "MINUS_E2"))
        odd = (plus - minus) / 2.0
        midpoint = (plus + minus) / 2.0 - zero
        summary[layer] = {
            "O2": odd.detach().cpu().tolist(),
            "M2": midpoint.detach().cpu().tolist(),
            "O2_l2": float(odd.norm().item()),
            "M2_l2": float(midpoint.norm().item()),
        }
    return summary


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing second-axis diagnostic output: {output}")
    output.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "SETUP_RUNNING",
        "science_denominator": 0,
        "diagnostic_scope": "new fixed-terminal second-axis localization; not recovery of the earlier terminal tensor",
        "arms": {arm: {"state": state, "status": "NOT_STARTED"} for arm, state in SECOND_AXIS_STATES.items()},
        "calls": {"generation_invocations": 0, "transformer_forwards": 0, "vae_decode_attempts": 0, "vae_encode_attempts": 0},
        "acceptance": config["acceptance"],
    }
    _write_json(output / "config.json", config)
    try:
        generated = generate_terminal_latent(config)
        terminal, vae = generated.normalized_latent, generated.vae
        if terminal.dtype != _torch().float32:
            raise ValueError(f"second-axis diagnostic requires an actual FP32 terminal, got {terminal.dtype}")
        result["generation"] = generated.metadata
        result["calls"]["generation_invocations"] = int(generated.metadata["generation_invocations"])
        result["calls"]["transformer_forwards"] = int(generated.metadata["transformer_calls"])
        result["shared_terminal"] = _save_tensor(output / "shared_terminal_normalized.pt", terminal)
        result["shared_terminal"]["path"] = "shared_terminal_normalized.pt"
    except BaseException as exc:
        result["status"] = "FATAL_SETUP"
        result["setup_error"] = repr(exc)
        _write_json(output / "result.json", result)
        raise

    config_c2a = C2AConfig()
    q_by_arm: dict[str, dict[str, Any]] = {}
    result["status"] = "EXECUTED_REQUIRES_REVIEW"
    for arm, state in SECOND_AXIS_STATES.items():
        row = result["arms"][arm]
        row["status"] = "PENDING"
        try:
            marked, analytic_records = write_covariance_state(terminal, state, config=config_c2a)
            row["analytic_transport_records_float64"] = analytic_records
            row["actual_write_measurement"] = actual_write_measurement(marked, state, config=config_c2a)
            blocks_path = output / "actual_written_blocks" / f"{arm}.pt"
            blocks_path.parent.mkdir(parents=True, exist_ok=True)
            _torch().save(actual_written_block_snapshots(marked, config=config_c2a), blocks_path)
            row["actual_written_blocks"] = {"path": str(blocks_path.relative_to(output)), "groups": [1, 2, 3], "dtype": str(marked.dtype), "block_shape_per_group": [2, 8, 8]}

            result["calls"]["vae_decode_attempts"] += 1
            rgb = decode_normalized_latent(vae, marked)
            float_path = output / "pre_codec_float_rgb" / f"{arm}.pt"
            row["pre_codec_float_rgb"] = _save_tensor(float_path, rgb)
            row["pre_codec_float_rgb"]["path"] = str(float_path.relative_to(output))

            result["calls"]["vae_encode_attempts"] += 1
            pre_reencoded = reencode_rgb24_readback(vae, rgb)
            pre_q = read_q(pre_reencoded, config=config_c2a)
            mp4_path = output / arm / "saved.mp4"
            readback = ffmpeg_roundtrip(rgb, mp4_path, fps=int(config["generation"]["fps"]), crf=int(config["codec"]["crf"]))
            result["calls"]["vae_encode_attempts"] += 1
            post_reencoded = reencode_rgb24_readback(vae, readback)
            post_q = read_q(post_reencoded, config=config_c2a)
            row["q"] = {"pre_codec": pre_q.detach().cpu().tolist(), "post_mp4": post_q.detach().cpu().tolist()}
            row["saved_mp4"] = str(mp4_path.relative_to(output))
            q_by_arm[arm] = {"pre_codec": pre_q, "post_mp4": post_q}
            row["status"] = "COMPLETE"
        except BaseException as exc:
            row["status"] = "FAILED"
            row["error"] = repr(exc)

    if set(q_by_arm) == set(SECOND_AXIS_STATES):
        result["second_axis_response"] = _second_axis_summary(q_by_arm)
    else:
        result["second_axis_response"] = {"status": "NOT_COMPUTED_INCOMPLETE_FIXED_DENOMINATOR", "available_arms": sorted(q_by_arm)}
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
