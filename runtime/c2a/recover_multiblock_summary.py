"""Pure-CPU recovery of multiblock calibration from persisted JSON q records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOURCE_RESULT = "/content/drive/MyDrive/Video-WM/C2A_Multiblock_Diagnostic/c2a_multiblock_20260915T022411Z/result.json"
LAYOUTS = {"single": ((16, 28),), "four": ((16, 28), (4, 16), (4, 40), (28, 28))}
STATES = ("ZERO", "PLUS_E1", "MINUS_E1", "PLUS_E2", "MINUS_E2")
SINGULAR_RELATIVE = 1e-8


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _add(left: list[float], right: list[float]) -> list[float]:
    return [left[0] + right[0], left[1] + right[1]]


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [left[0] - right[0], left[1] - right[1]]


def _scale(value: list[float], scalar: float) -> list[float]:
    return [value[0] * scalar, value[1] * scalar]


def _norm(value: list[float]) -> float:
    return math.hypot(value[0], value[1])


def _mean(values: list[list[float]]) -> list[float]:
    if not values:
        raise ValueError("cannot average an empty fixed block set")
    return _scale([sum(value[0] for value in values), sum(value[1] for value in values)], 1.0 / len(values))


def _vectors(plus: list[float], minus: list[float], zero: list[float]) -> dict[str, Any]:
    odd = _scale(_sub(plus, minus), 0.5)
    midpoint = _sub(_scale(_add(plus, minus), 0.5), zero)
    return {"O": odd, "M": midpoint, "O_l2": _norm(odd), "M_l2": _norm(midpoint)}


def _matrix_singular_values(matrix: list[list[float]]) -> list[float]:
    a00, a01 = matrix[0]
    a10, a11 = matrix[1]
    trace = a00 * a00 + a01 * a01 + a10 * a10 + a11 * a11
    determinant_ata = (a00 * a11 - a01 * a10) ** 2
    discriminant = max(trace * trace - 4.0 * determinant_ata, 0.0)
    largest_sq = max((trace + math.sqrt(discriminant)) / 2.0, 0.0)
    smallest_sq = max((trace - math.sqrt(discriminant)) / 2.0, 0.0)
    return [math.sqrt(largest_sq), math.sqrt(smallest_sq)]


def _inverse_apply(matrix: list[list[float]], value: list[float]) -> list[float]:
    a00, a01 = matrix[0]
    a10, a11 = matrix[1]
    determinant = a00 * a11 - a01 * a10
    return [(a11 * value[0] - a01 * value[1]) / determinant, (-a10 * value[0] + a00 * value[1]) / determinant]


def _q_rows(source: dict[str, Any], layout: str) -> dict[str, dict[str, list[float]]]:
    rows: dict[str, dict[str, list[float]]] = {}
    for state in STATES:
        arm = f"{layout.upper()}_{state}"
        row = source["arms"].get(arm)
        if not isinstance(row, dict) or row.get("status") != "COMPLETE":
            raise ValueError(f"source arm unavailable: {arm}")
        block_values = row.get("post_mp4_q_by_layout", {}).get(layout, {})
        rows[state] = {key: [float(value[0]), float(value[1])] for key, value in block_values.items()}
    return rows


def _calibrate_layout(source: dict[str, Any], layout: str) -> dict[str, Any]:
    rows = _q_rows(source, layout)
    per_block: dict[str, Any] = {}
    estimates: dict[str, list[list[float]]] = {state: [] for state in STATES}
    unsupported: list[dict[str, Any]] = []
    for top, left in LAYOUTS[layout]:
        key = f"{top}_{left}"
        try:
            zero, plus1, plus2 = rows["ZERO"][key], rows["PLUS_E1"][key], rows["PLUS_E2"][key]
        except KeyError as exc:
            raise ValueError(f"source q missing fixed block {key}") from exc
        matrix = [[(plus1[0] - zero[0]) / 0.5, (plus2[0] - zero[0]) / 0.5], [(plus1[1] - zero[1]) / 0.5, (plus2[1] - zero[1]) / 0.5]]
        singular_values = _matrix_singular_values(matrix)
        record: dict[str, Any] = {"top_left": [top, left], "b": zero, "A": matrix, "singular_values_absolute": singular_values}
        if singular_values[1] <= SINGULAR_RELATIVE * max(singular_values[0], 1.0):
            record["status"] = "UNSUPPORTED_INVERSE_NEAR_SINGULAR"
            per_block[key] = record
            unsupported.append(record)
            continue
        record["status"] = "CALIBRATED_FROM_ZERO_PLUS_E1_PLUS_E2_ONLY"
        per_block[key] = record
        for state in STATES:
            estimates[state].append(_inverse_apply(matrix, _sub(rows[state][key], zero)))
    if unsupported:
        return {"status": "UNSUPPORTED_ANY_BLOCK_INVERSE", "per_block": per_block, "unsupported_blocks": unsupported, "aggregation": "not computed; no pseudoinverse, omission, or reweighting"}

    aggregate = {state: _mean(values) for state, values in estimates.items()}
    holdouts: dict[str, Any] = {}
    for state, target in (("MINUS_E1", [-0.5, 0.0]), ("MINUS_E2", [0.0, -0.5])):
        errors = [_sub(value, target) for value in estimates[state]]
        mean_error = _mean(errors)
        holdouts[state] = {"target_state": target, "per_block_state_estimates": estimates[state], "per_block_error_vectors": errors, "mean_error_vector": mean_error, "mean_error_vector_l2": _norm(mean_error), "mean_of_per_block_error_l2": sum(_norm(value) for value in errors) / len(errors), "aggregate_state": aggregate[state], "aggregate_error_vector": _sub(aggregate[state], target), "aggregate_error_l2": _norm(_sub(aggregate[state], target))}
    raw_per_block = {f"{top}_{left}": {axis: _vectors(rows[plus][f"{top}_{left}"], rows[minus][f"{top}_{left}"], rows["ZERO"][f"{top}_{left}"]) for axis, plus, minus in (("E1", "PLUS_E1", "MINUS_E1"), ("E2", "PLUS_E2", "MINUS_E2"))} for top, left in LAYOUTS[layout]}
    aggregate_raw_q = {state: _mean([rows[state][f"{top}_{left}"] for top, left in LAYOUTS[layout]]) for state in STATES}
    return {"status": "CALIBRATED_DESCRIPTIVE_NO_ACCEPTANCE", "fit_points": ["ZERO", "PLUS_E1", "PLUS_E2"], "holdouts": ["MINUS_E1", "MINUS_E2"], "per_block": per_block, "aggregate_state_equal_weight": aggregate, "state_O_M": {axis: _vectors(aggregate[plus], aggregate[minus], aggregate["ZERO"]) for axis, plus, minus in (("E1", "PLUS_E1", "MINUS_E1"), ("E2", "PLUS_E2", "MINUS_E2"))}, "raw_q_O_M_per_block": raw_per_block, "equal_weight_raw_q_descriptive_only": aggregate_raw_q, "equal_weight_raw_q_O_M_descriptive_only": {axis: _vectors(aggregate_raw_q[plus], aggregate_raw_q[minus], aggregate_raw_q["ZERO"]) for axis, plus, minus in (("E1", "PLUS_E1", "MINUS_E1"), ("E2", "PLUS_E2", "MINUS_E2"))}, "negative_holdouts": holdouts, "heldout_state_pair_distance": _norm(_sub(aggregate["MINUS_E1"], aggregate["MINUS_E2"])), "no_success_threshold": True}


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("source_result") != SOURCE_RESULT:
        raise ValueError("recovery fixes the failed multiblock result path")
    if config.get("layouts") != {name: [[top, left] for top, left in blocks] for name, blocks in LAYOUTS.items()}:
        raise ValueError("recovery fixes the known multiblock geometry")
    return config


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing recovered summary: {output}")
    source_path = Path(config["source_result"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    recovered: dict[str, Any] = {"status": "RECOVERED_FROM_PERSISTED_Q_REQUIRES_REVIEW", "science_denominator": 0, "source_result": str(source_path), "recovery_scope": "pure CPU JSON q aggregation only; no model, tensor, video, generation, decode, encode, or FFmpeg", "source_calls": source.get("calls"), "layouts": {}}
    for layout in LAYOUTS:
        try:
            recovered["layouts"][layout] = _calibrate_layout(source, layout)
        except Exception as exc:
            recovered["layouts"][layout] = {"status": "UNSUPPORTED_RECOVERY_INPUT", "error": repr(exc)}
    _write_json(output / "recovered_summary.json", recovered)
    _write_json(output / "config.json", config)
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(_load_config(args.config), args.output)


if __name__ == "__main__":
    main()
