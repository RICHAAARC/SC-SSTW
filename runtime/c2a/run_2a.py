"""Explicitly gated six-arm C2A interface executor.

This file does no model work on import. It consumes one supplied terminal
latent; `--execute` is its sole explicit real-work entrypoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.c2a.chain import decode_normalized_latent, ffmpeg_roundtrip, reencode_rgb24_readback
from runtime.c2a.generation import generate_terminal_latent
from runtime.c2a.protocol import ARM_STATES, C2AConfig, axis_response_summary, latent_change_metrics, read_q, rgb_quality_metrics, write_covariance_state


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model = config.get("model", {})
    if not isinstance(model.get("id"), str) or not model["id"] or model["id"].startswith("REQUIRED_"):
        raise ValueError("C2A requires the identifier of an already-present local model")
    if model.get("revision") is not None and not isinstance(model["revision"], str):
        raise ValueError("model revision is a record string or null")
    c2a = config.get("c2a", {})
    if list(c2a.get("arms", ())) != list(ARM_STATES):
        raise ValueError("C2A arm roster is fixed and ordered")
    if c2a.get("channels_zero_based") != [0, 1] or c2a.get("ordinary_groups") != [1, 2, 3] or c2a.get("read_group") != 2:
        raise ValueError("C2A channel and temporal support are fixed by the prepared method")
    if c2a.get("beta") != 0.25 or c2a.get("rho") != 0.5 or c2a.get("posterior") != "mode":
        raise ValueError("C2A state domain and posterior rule are fixed by the prepared method")
    if c2a.get("central_block") != "floor((H-8)/2):+8, floor((W-8)/2):+8" or c2a.get("near_singular_rule") != "reject if lambda_min <= 1e-8 * max(lambda_max, 1); no regularization or pseudoinverse" or c2a.get("cache_reset") != "before and after each decode and encode":
        raise ValueError("C2A spatial, singularity, and VAE-cache semantics are fixed by the prepared method")
    codec = config.get("codec", {})
    if codec.get("name") != "h264_yuv420p" or codec.get("readback") != "ffmpeg_rgb24" or codec.get("thread_count") != 1 or codec.get("crf") != 18:
        raise ValueError("C2A codec/readback contract is fixed")
    return config


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing C2A output: {output}")
    output.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "SETUP_RUNNING",
        "science_denominator": 0,
        "arms": {arm: {"state": state, "status": "NOT_STARTED"} for arm, state in ARM_STATES.items()},
        "acceptance": config["acceptance"],
    }
    _write_json(output / "config.json", config)
    try:
        generated = generate_terminal_latent(config)
        terminal, vae = generated.normalized_latent, generated.vae
        config_c2a = C2AConfig()
        result["generation"] = generated.metadata
    except BaseException as exc:
        result["status"] = "FATAL_SETUP"
        result["setup_error"] = repr(exc)
        _write_json(output / "result.json", result)
        raise
    result["status"] = "EXECUTED_REQUIRES_REVIEW"
    q_by_arm: dict[str, Any] = {}
    off_rgb = None
    off_readback = None
    zero_rgb = None
    zero_readback = None
    for arm, state in ARM_STATES.items():
        row = result["arms"][arm]
        row["status"] = "PENDING"
        try:
            marked, write_records = (terminal.detach().clone(), []) if state is None else write_covariance_state(terminal, state, config=config_c2a)
            row["write_records"] = write_records
            row["latent_metrics_vs_off"] = latent_change_metrics(terminal, marked, config=config_c2a)
            rgb = decode_normalized_latent(vae, marked)
            readback = ffmpeg_roundtrip(rgb, output / arm / "saved.mp4", fps=int(config["generation"]["fps"]), crf=int(config["codec"]["crf"]))
            reencoded = reencode_rgb24_readback(vae, readback)
            row["decoded_frame_shape"] = list(rgb.shape)
            row["reencoded_latent_shape"] = list(reencoded.shape)
            q_value = read_q(reencoded, config=config_c2a)
            row["q"] = q_value.detach().cpu().tolist()
            q_by_arm[arm] = q_value
            if arm == "OFF":
                off_rgb = rgb
                off_readback = readback
            else:
                if off_rgb is not None:
                    row["quality_off_to_arm_pre_codec"] = rgb_quality_metrics(off_rgb, rgb)
                if off_readback is not None:
                    row["quality_off_to_arm_mp4_rgb24"] = rgb_quality_metrics(off_readback, readback)
                if arm == "ZERO":
                    zero_rgb = rgb
                    zero_readback = readback
                elif zero_rgb is not None:
                    row["quality_zero_to_arm_pre_codec"] = rgb_quality_metrics(zero_rgb, rgb)
                    if zero_readback is not None:
                        row["quality_zero_to_arm_mp4_rgb24"] = rgb_quality_metrics(zero_readback, readback)
            row["status"] = "COMPLETE"
        except BaseException as exc:
            row["status"] = "FAILED"
            row["error"] = repr(exc)
    if set(q_by_arm) == set(ARM_STATES):
        result["axis_response"] = axis_response_summary(q_by_arm)
    else:
        result["axis_response"] = {"status": "NOT_COMPUTED_INCOMPLETE_FIXED_DENOMINATOR", "available_arms": sorted(q_by_arm)}
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
