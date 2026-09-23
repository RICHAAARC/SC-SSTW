"""Fixed input auditing and identity-path measurements for media retention."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split_receiver

PARTITIONS = ("TOTAL", "A", "B")
SCALARS = ("score", "matched_score", "state_innovation_mean")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def partition_contract() -> dict[str, dict[str, int]]:
    result = {}
    for name, indices in (("A", split_receiver.A), ("B", split_receiver.B)):
        indices = np.asarray(indices)
        row = {
            "support_count": int(len(indices)),
            "q_axis_0_count": int(np.sum(indices % 2 == 0)),
            "q_axis_1_count": int(np.sum(indices % 2 == 1)),
        }
        if row != {"support_count": 80, "q_axis_0_count": 40, "q_axis_1_count": 40}:
            raise ValueError(f"frozen partition {name} is not 80 supports with 40/40 q axes")
        result[name] = row
    if set(map(int, split_receiver.A)) & set(map(int, split_receiver.B)):
        raise ValueError("frozen A/B partitions overlap")
    if set(map(int, split_receiver.A)) | set(map(int, split_receiver.B)) != set(range(160)):
        raise ValueError("frozen A/B partitions do not cover the 160 supports")
    return result


def empty_measurement() -> dict:
    return {
        "status": "NOT_RUN",
        "partition": None,
        "score": None,
        "matched_score": None,
        "state_innovation_mean": None,
        "q_by_window": [None] * 11,
        "windows": [],
        "failure": None,
    }


def _compact_score(row: dict, partition: str) -> dict:
    if row.get("status") != "SCORED":
        raise ValueError(f"{partition} identity path was not scored")
    windows = row.get("windows", [])
    if len(windows) != 11 or any(not window.get("valid") for window in windows):
        raise ValueError(f"{partition} identity path does not retain all 11 valid windows")
    expected_components = 7040 if partition == "TOTAL" else 3520
    expected_blocks = 1760 if partition == "TOTAL" else 880
    if row.get("matched_components") != expected_components:
        raise ValueError(
            f"{partition} matched component denominator changed: "
            f"{row.get('matched_components')} != {expected_components}"
        )
    if row.get("full_blocks") != expected_blocks or row.get("partial_blocks") != 0:
        raise ValueError(f"{partition} does not have fixed complete support")
    result = {
        "status": "SCORED",
        "partition": partition,
        "path": row.get("path"),
        "score": row.get("score"),
        "matched_score": row.get("matched_score"),
        "state_innovation_mean": row.get("state_innovation_mean"),
        "q_by_window": [window.get("q") for window in windows],
        "innovation_by_window": row.get("innovation_by_window"),
        "matched_windows": row.get("matched_windows"),
        "matched_components": row.get("matched_components"),
        "full_blocks": row.get("full_blocks"),
        "partial_blocks": row.get("partial_blocks"),
        "windows": windows,
        "failure": None,
    }
    return _plain(result)


def score_identity_layer(tensor: Any, book: dict, identity_path: dict, expected_shape: list[int]) -> dict:
    """Score TOTAL, A, and B independently on the fixed complete identity path."""

    shape = tuple(int(value) for value in tensor.shape)
    if shape != tuple(expected_shape):
        raise ValueError(f"latent shape mismatch: {shape} != {tuple(expected_shape)}")
    array = tensor.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError("latent contains nonfinite values")
    if identity_path != fixed_key.REFERENCE_PATHS["IDENTITY"]:
        raise ValueError("only the frozen IDENTITY path is allowed")
    contract = partition_contract()
    observations = {0: array}
    rows = {
        "TOTAL": fixed_key.score_path(observations, book, identity_path),
        "A": split_receiver.PartitionEvidence(
            observations, book, split_receiver.A, "A",
        ).score_path(identity_path),
        "B": split_receiver.PartitionEvidence(
            observations, book, split_receiver.B, "B",
        ).score_path(identity_path),
    }
    return {
        "status": "SCORED",
        "partition_contract": contract,
        "partitions": {
            name: _compact_score(rows[name], name) for name in PARTITIONS
        },
    }


def numeric_marker(value: Any, epsilon: float) -> str:
    if value is None:
        return "UNDEFINED"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "NONFINITE"
    if abs(float(value)) <= epsilon:
        return "ZERO_OR_NEAR_ZERO"
    return "POSITIVE" if value > 0 else "NEGATIVE"


def _difference(after: Any, before: Any, epsilon: float) -> dict:
    valid = all(isinstance(value, (int, float)) and math.isfinite(value) for value in (after, before))
    delta = float(after - before) if valid else None
    before_marker = numeric_marker(before, epsilon)
    after_marker = numeric_marker(after, epsilon)
    signs = {before_marker, after_marker}
    return {
        "before": before,
        "after": after,
        "delta": delta,
        "delta_marker": numeric_marker(delta, epsilon),
        "endpoint_sign_change": signs == {"NEGATIVE", "POSITIVE"},
    }


def compare_measurements(after: dict, before: dict, epsilon: float) -> dict:
    if after.get("status") != "SCORED" or before.get("status") != "SCORED":
        return {
            "status": "UNDEFINED",
            "failure": "both measurement endpoints must be SCORED",
            "scalars": {name: _difference(None, None, epsilon) for name in SCALARS},
            "q_by_window": [
                [_difference(None, None, epsilon), _difference(None, None, epsilon)] for _ in range(11)
            ],
        }
    return {
        "status": "COMPARED",
        "failure": None,
        "scalars": {
            name: _difference(after.get(name), before.get(name), epsilon) for name in SCALARS
        },
        "q_by_window": [
            [
                _difference(after["q_by_window"][window][axis], before["q_by_window"][window][axis], epsilon)
                for axis in (0, 1)
            ]
            for window in range(11)
        ],
    }


def compare_increments(after: dict, before: dict, epsilon: float) -> dict:
    if after.get("status") != "COMPARED" or before.get("status") != "COMPARED":
        return {
            "status": "UNDEFINED",
            "failure": "both marked-minus-OFF increments must be available",
            "scalars": {name: _difference(None, None, epsilon) for name in SCALARS},
            "q_by_window": [
                [_difference(None, None, epsilon), _difference(None, None, epsilon)] for _ in range(11)
            ],
        }
    return {
        "status": "COMPARED",
        "failure": None,
        "scalars": {
            name: _difference(
                after["scalars"][name]["delta"], before["scalars"][name]["delta"], epsilon,
            )
            for name in SCALARS
        },
        "q_by_window": [
            [
                _difference(
                    after["q_by_window"][window][axis]["delta"],
                    before["q_by_window"][window][axis]["delta"],
                    epsilon,
                )
                for axis in (0, 1)
            ]
            for window in range(11)
        ],
    }
