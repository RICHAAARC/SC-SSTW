"""Fixed single-versus-four-block C2A diagnostic over a saved terminal latent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.c2a.chain import decode_normalized_latent, ffmpeg_roundtrip, reencode_rgb24_readback
from runtime.c2a.generation import load_frozen_vae
from runtime.c2a.protocol import (
    C2AConfig,
    latent_change_metrics_at_blocks,
    read_q_at_block,
    write_covariance_state_at_blocks,
    written_block_snapshots_at_blocks,
    rgb_quality_metrics,
)


TERMINAL_PATH = "/content/drive/MyDrive/Video-WM/C2A_SecondAxis_Diagnostic/c2a_second_axis_20260914T145106Z/shared_terminal_normalized.pt"
LAYOUTS: dict[str, tuple[tuple[int, int], ...]] = {
    "single": ((16, 28),),
    "four": ((16, 28), (4, 16), (4, 40), (28, 28)),
}
STATES: dict[str, tuple[float, float]] = {
    "ZERO": (0.0, 0.0),
    "PLUS_E1": (0.5, 0.0),
    "MINUS_E1": (-0.5, 0.0),
    "PLUS_E2": (0.0, 0.5),
    "MINUS_E2": (0.0, -0.5),
}
ARM_ORDER = ("OFF", "SINGLE_ZERO", "SINGLE_PLUS_E1", "SINGLE_MINUS_E1", "SINGLE_PLUS_E2", "SINGLE_MINUS_E2", "FOUR_ZERO", "FOUR_PLUS_E1", "FOUR_MINUS_E1", "FOUR_PLUS_E2", "FOUR_MINUS_E2")
SINGULAR_RELATIVE = 1e-8


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _torch() -> Any:
    import torch

    return torch


def _save_tensor(path: Path, tensor: Any, *, output: Path) -> dict[str, Any]:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = tensor.detach().to(device="cpu").contiguous().clone()
    torch.save(saved, path)
    return {"path": str(path.relative_to(output)), "shape": list(saved.shape), "dtype": str(saved.dtype), "bytes": saved.nelement() * saved.element_size()}


def _layout_payload() -> dict[str, list[list[int]]]:
    return {name: [[top, left] for top, left in blocks] for name, blocks in LAYOUTS.items()}


def _validate_layouts() -> None:
    for name, blocks in LAYOUTS.items():
        if not blocks:
            raise ValueError(f"{name} layout is empty")
        for top, left in blocks:
            if top < 0 or left < 0 or top + 8 > 40 or left + 8 > 64:
                raise ValueError(f"{name} block lies outside fixed 40x64 geometry")
        for index, (top, left) in enumerate(blocks):
            for other_top, other_left in blocks[index + 1:]:
                overlap = top < other_top + 8 and other_top < top + 8 and left < other_left + 8 and other_left < left + 8
                if overlap:
                    raise ValueError(f"{name} layout blocks overlap")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model, source, c2a, codec = config.get("model", {}), config.get("source", {}), config.get("c2a", {}), config.get("codec", {})
    if not isinstance(model.get("id"), str) or not model["id"] or model["id"].startswith("REQUIRED_"):
        raise ValueError("multiblock diagnostic requires a concrete model identifier")
    if model.get("revision") is not None and not isinstance(model.get("revision"), str):
        raise ValueError("model revision is a record string or null")
    if source.get("shared_terminal") != TERMINAL_PATH or source.get("terminal_shape") != [1, 16, 13, 40, 64] or source.get("terminal_dtype") != "torch.float32":
        raise ValueError("multiblock diagnostic fixes the persisted terminal provenance")
    if c2a.get("layouts") != _layout_payload() or c2a.get("beta") != 0.25 or c2a.get("rho") != 0.5 or c2a.get("channels_zero_based") != [0, 1] or c2a.get("ordinary_groups") != [1, 2, 3] or c2a.get("read_group") != 2 or c2a.get("posterior") != "mode":
        raise ValueError("multiblock carrier/layout semantics are fixed")
    if c2a.get("block_size") != 8 or c2a.get("aggregation") != "equal_weight_mean_of_per_block_inverse_calibrated_state" or c2a.get("inverse_rule") != "reject if sigma_min <= 1e-8 * max(sigma_max, 1); no pseudoinverse or reweighting":
        raise ValueError("multiblock calibration semantics are fixed")
    if codec.get("name") != "h264_yuv420p" or codec.get("crf") != 18 or codec.get("readback") != "ffmpeg_rgb24" or codec.get("thread_count") != 1 or config.get("generation", {}).get("fps") != 8:
        raise ValueError("MP4/readback contract is fixed")
    _validate_layouts()
    return config


def _arm_spec(arm: str) -> tuple[str | None, str | None]:
    if arm == "OFF":
        return None, None
    layout, state = arm.split("_", 1)
    return layout.lower(), state


def _q_for_blocks(reencoded: Any, blocks: tuple[tuple[int, int]], config: C2AConfig) -> dict[str, Any]:
    return {f"{top}_{left}": read_q_at_block(reencoded, top_left=(top, left), config=config) for top, left in blocks}


def _vectors(plus: Any, minus: Any, zero: Any) -> dict[str, Any]:
    odd = (plus - minus) / 2.0
    midpoint = (plus + minus) / 2.0 - zero
    return {"O": odd.detach().cpu().tolist(), "M": midpoint.detach().cpu().tolist(), "O_l2": float(odd.norm().item()), "M_l2": float(midpoint.norm().item())}


def _calibrate_layout(layout: str, q_rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Fit b/A only on ZERO,+E1,+E2; retain both negatives as holdouts."""

    torch = _torch()
    blocks = LAYOUTS[layout]
    expected = {state: q_rows[f"{layout.upper()}_{state}"] for state in STATES}
    per_block: dict[str, Any] = {}
    estimates: dict[str, list[Any]] = {state: [] for state in STATES}
    unsupported: list[dict[str, Any]] = []
    for top, left in blocks:
        key = f"{top}_{left}"
        zero = expected["ZERO"][key].to(dtype=torch.float64)
        plus1 = expected["PLUS_E1"][key].to(dtype=torch.float64)
        plus2 = expected["PLUS_E2"][key].to(dtype=torch.float64)
        matrix = torch.stack(((plus1 - zero) / 0.5, (plus2 - zero) / 0.5), dim=1)
        singular_values = torch.linalg.svdvals(matrix)
        record: dict[str, Any] = {"top_left": [top, left], "b": zero.detach().cpu().tolist(), "A": matrix.detach().cpu().tolist(), "singular_values_absolute": singular_values.detach().cpu().tolist()}
        if float(singular_values[-1].item()) <= SINGULAR_RELATIVE * max(float(singular_values[0].item()), 1.0):
            record["status"] = "UNSUPPORTED_INVERSE_NEAR_SINGULAR"
            unsupported.append(record)
            per_block[key] = record
            continue
        inverse = torch.linalg.inv(matrix)
        record["status"] = "CALIBRATED_FROM_ZERO_PLUS_E1_PLUS_E2_ONLY"
        per_block[key] = record
        for state, values in expected.items():
            estimates[state].append(inverse @ (values[key].to(dtype=torch.float64) - zero))
    if unsupported:
        return {"status": "UNSUPPORTED_ANY_BLOCK_INVERSE", "per_block": per_block, "unsupported_blocks": unsupported, "aggregation": "not computed; no pseudoinverse, omission, or reweighting"}

    aggregate = {state: torch.stack(values).mean(dim=0) for state, values in estimates.items()}
    holdouts: dict[str, Any] = {}
    for state, target_values in (("MINUS_E1", (-0.5, 0.0)), ("MINUS_E2", (0.0, -0.5))):
        target = torch.tensor(target_values, dtype=torch.float64, device=estimates[state][0].device)
        errors = torch.stack([estimate - target for estimate in estimates[state]])
        mean_error = errors.mean(dim=0)
        holdouts[state] = {
            "target_state": target.detach().cpu().tolist(),
            "per_block_state_estimates": [value.detach().cpu().tolist() for value in estimates[state]],
            "per_block_error_vectors": [value.detach().cpu().tolist() for value in errors],
            "mean_error_vector": mean_error.detach().cpu().tolist(),
            "mean_error_vector_l2": float(mean_error.norm().item()),
            "mean_of_per_block_error_l2": float(errors.norm(dim=1).mean().item()),
            "aggregate_state": aggregate[state].detach().cpu().tolist(),
            "aggregate_error_vector": (aggregate[state] - target).detach().cpu().tolist(),
            "aggregate_error_l2": float((aggregate[state] - target).norm().item()),
        }
    q_vectors = {
        f"{top}_{left}": {
            axis: _vectors(expected[plus][f"{top}_{left}"], expected[minus][f"{top}_{left}"], expected["ZERO"][f"{top}_{left}"])
            for axis, plus, minus in (("E1", "PLUS_E1", "MINUS_E1"), ("E2", "PLUS_E2", "MINUS_E2"))
        }
        for top, left in blocks
    }
    aggregate_raw_q = {
        state: torch.stack([values[f"{top}_{left}"].to(dtype=torch.float64) for top, left in blocks]).mean(dim=0)
        for state, values in expected.items()
    }
    aggregate_raw_q_vectors = {axis: _vectors(aggregate_raw_q[plus], aggregate_raw_q[minus], aggregate_raw_q["ZERO"]) for axis, plus, minus in (("E1", "PLUS_E1", "MINUS_E1"), ("E2", "PLUS_E2", "MINUS_E2"))}
    state_vectors = {axis: _vectors(aggregate[plus], aggregate[minus], aggregate["ZERO"]) for axis, plus, minus in (("E1", "PLUS_E1", "MINUS_E1"), ("E2", "PLUS_E2", "MINUS_E2"))}
    heldout_distance = float((aggregate["MINUS_E1"] - aggregate["MINUS_E2"]).norm().item())
    return {"status": "CALIBRATED_DESCRIPTIVE_NO_ACCEPTANCE", "fit_points": ["ZERO", "PLUS_E1", "PLUS_E2"], "holdouts": ["MINUS_E1", "MINUS_E2"], "per_block": per_block, "aggregate_state_equal_weight": {state: value.detach().cpu().tolist() for state, value in aggregate.items()}, "state_O_M": state_vectors, "raw_q_O_M_per_block": q_vectors, "equal_weight_raw_q_descriptive_only": {state: value.detach().cpu().tolist() for state, value in aggregate_raw_q.items()}, "equal_weight_raw_q_O_M_descriptive_only": aggregate_raw_q_vectors, "negative_holdouts": holdouts, "heldout_state_pair_distance": heldout_distance, "no_success_threshold": True}


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing multiblock diagnostic output: {output}")
    terminal_path = Path(config["source"]["shared_terminal"])
    if not terminal_path.is_file():
        raise FileNotFoundError(f"fixed shared terminal missing: {terminal_path}")
    output.mkdir(parents=True)
    result: dict[str, Any] = {"status": "SETUP_RUNNING", "science_denominator": 0, "diagnostic_scope": "fixed single-versus-four-block spatial repetition; independent arms, not blind public calibration", "source": {"shared_terminal": str(terminal_path)}, "layouts": {name: {"blocks_top_left": [[top, left] for top, left in blocks], "spatial_write_support_multiplier_vs_single": len(blocks)} for name, blocks in LAYOUTS.items()}, "arms": {arm: {"status": "NOT_STARTED"} for arm in ARM_ORDER}, "calls": {"generation_invocations": 0, "transformer_forwards": 0, "vae_loads": 0, "vae_decode_attempts": 0, "vae_encode_attempts": 0}, "acceptance": config["acceptance"]}
    _write_json(output / "config.json", config)
    try:
        torch = _torch()
        terminal_cpu = torch.load(terminal_path, map_location="cpu")
        if tuple(terminal_cpu.shape) != (1, 16, 13, 40, 64) or terminal_cpu.dtype != torch.float32:
            raise ValueError("persisted shared terminal does not match fixed FP32 geometry")
        vae = load_frozen_vae(config)
        terminal = terminal_cpu.to(device=next(vae.parameters()).device)
        result["calls"]["vae_loads"] = 1
        result["terminal"] = {"shape": list(terminal_cpu.shape), "dtype": str(terminal_cpu.dtype), "source_reused_without_generation": True}
    except BaseException as exc:
        result["status"] = "FATAL_SETUP"
        result["setup_error"] = repr(exc)
        _write_json(output / "result.json", result)
        raise

    c2a = C2AConfig()
    q_rows: dict[str, dict[str, dict[str, Any]]] = {"single": {}, "four": {}}
    off_rgb = off_readback = None
    zero_rgb: dict[str, Any] = {}
    zero_readback: dict[str, Any] = {}
    result["status"] = "EXECUTED_REQUIRES_REVIEW"
    for arm in ARM_ORDER:
        row = result["arms"][arm]
        row["status"] = "PENDING"
        try:
            layout, state_name = _arm_spec(arm)
            if arm == "OFF":
                marked, blocks_for_row = terminal.detach().clone(), LAYOUTS["four"]
            else:
                blocks_for_row = LAYOUTS[layout]
                marked, records = write_covariance_state_at_blocks(terminal, STATES[state_name], blocks=blocks_for_row, config=c2a)
                row["actual_write_records"] = records
                row["latent_cost"] = latent_change_metrics_at_blocks(terminal, marked, blocks=blocks_for_row, config=c2a)
                snapshots_path = output / "actual_written_blocks" / f"{arm}.pt"
                snapshots_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(written_block_snapshots_at_blocks(marked, blocks=blocks_for_row, config=c2a), snapshots_path)
                row["actual_written_blocks"] = str(snapshots_path.relative_to(output))
            result["calls"]["vae_decode_attempts"] += 1
            rgb = decode_normalized_latent(vae, marked)
            row["pre_codec_float_rgb"] = _save_tensor(output / "pre_codec_float_rgb" / f"{arm}.pt", rgb, output=output)
            readback = ffmpeg_roundtrip(rgb, output / arm / "saved.mp4", fps=8, crf=18)
            result["calls"]["vae_encode_attempts"] += 1
            reencoded = reencode_rgb24_readback(vae, readback)
            row["post_mp4_reencoded_normalized"] = _save_tensor(output / "post_mp4_reencoded_normalized" / f"{arm}.pt", reencoded, output=output)
            row["saved_mp4"] = str((output / arm / "saved.mp4").relative_to(output))
            row["post_mp4_q_by_layout"] = {}
            for name, blocks in LAYOUTS.items():
                q_values = _q_for_blocks(reencoded, blocks, c2a)
                row["post_mp4_q_by_layout"][name] = {key: value.detach().cpu().tolist() for key, value in q_values.items()}
                q_rows[name][arm] = q_values
            if arm == "OFF":
                off_rgb, off_readback = rgb, readback
            else:
                if off_rgb is not None:
                    row["quality_off_to_arm_pre_codec"] = rgb_quality_metrics(off_rgb, rgb)
                if off_readback is not None:
                    row["quality_off_to_arm_mp4_rgb24"] = rgb_quality_metrics(off_readback, readback)
                if state_name == "ZERO":
                    zero_rgb[layout], zero_readback[layout] = rgb, readback
                else:
                    if layout in zero_rgb:
                        row["quality_zero_to_arm_pre_codec"] = rgb_quality_metrics(zero_rgb[layout], rgb)
                    if layout in zero_readback:
                        row["quality_zero_to_arm_mp4_rgb24"] = rgb_quality_metrics(zero_readback[layout], readback)
            row["status"] = "COMPLETE"
        except BaseException as exc:
            row["status"] = "FAILED"
            row["error"] = repr(exc)
        _write_json(output / "result.json", result)

    for name in LAYOUTS:
        required = {f"{name.upper()}_{state}" for state in STATES}
        if required.issubset(q_rows[name]):
            result["layouts"][name]["calibration"] = _calibrate_layout(name, q_rows[name])
        else:
            status = "UNSUPPORTED_ANY_BLOCK_OR_ARM_FAILURE" if name == "four" else "NOT_COMPUTED_INCOMPLETE_FIXED_DENOMINATOR"
            result["layouts"][name]["calibration"] = {"status": status, "available_arms": sorted(q_rows[name]), "aggregation": "not computed; no failed-block omission, reweighting, or pseudoinverse"}
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
