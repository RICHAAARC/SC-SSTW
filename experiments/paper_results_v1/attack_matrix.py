"""Fixed temporal-edit recipes and post-hoc weighted truth for two-pilot evaluation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from typing import Any, Iterable


SOURCE_FRAMES = 181
GROUP_SIZE = math.floor(0.1 * SOURCE_FRAMES + 0.5)
GROUP_SEED_BASE = 20261010


def _sample_group(family_index: int, group: int, population: int) -> list[int]:
    return sorted(random.Random(GROUP_SEED_BASE + family_index * 100 + group).sample(
        range(population), GROUP_SIZE,
    ))


def fixed_attack_specs() -> list[dict[str, Any]]:
    rows = [
        {"attack_id": "full", "family": "FULL"},
        {"attack_id": "crop37_126", "family": "CROP", "start": 37, "stop": 126},
        {"attack_id": "speed075", "family": "SPEED_NN", "rate": 0.75},
        {"attack_id": "speed125", "family": "SPEED_NN", "rate": 1.25},
        {"attack_id": "mean3", "family": "MEAN", "window": 3},
        {"attack_id": "mean5", "family": "MEAN", "window": 5},
    ]
    for family_index, (name, family, population) in enumerate((
        ("delete", "DELETE", 181),
        ("repeat", "REPEAT_INSERT", 181),
        ("interp", "ADJACENT_EQUAL_INSERT", 180),
    )):
        for group in range(3):
            rows.append({
                "attack_id": f"{name}_g{group}", "family": family,
                "group": group, "seed": GROUP_SEED_BASE + family_index * 100 + group,
                "positions": _sample_group(family_index, group, population),
            })
    if len(rows) != 15 or len({row["attack_id"] for row in rows}) != 15:
        raise RuntimeError("fixed attack roster must contain fifteen unique instances")
    return rows


def _combined(items: Iterable[tuple[int, float]]) -> list[dict[str, float | int]]:
    combined: dict[int, float] = defaultdict(float)
    for source_index, weight in items:
        if type(source_index) is not int or source_index not in range(SOURCE_FRAMES):
            raise ValueError("weighted truth source index outside 0..180")
        if not math.isfinite(float(weight)) or float(weight) <= 0.0:
            raise ValueError("weighted truth requires finite positive weights")
        combined[source_index] += float(weight)
    total = sum(combined.values())
    return [
        {"source_index": source_index, "weight": weight / total}
        for source_index, weight in sorted(combined.items())
    ]


def expand_weighted_truth(spec: dict[str, Any]) -> list[list[dict[str, float | int]]]:
    family = spec.get("family")
    if family == "FULL":
        rows = [_combined(((index, 1.0),)) for index in range(SOURCE_FRAMES)]
    elif family == "CROP":
        start, stop = spec.get("start"), spec.get("stop")
        if (start, stop) != (37, 126):
            raise ValueError("fixed crop must be source[37:126]")
        rows = [_combined(((index, 1.0),)) for index in range(start, stop)]
    elif family == "SPEED_NN":
        rate = float(spec.get("rate"))
        if rate not in (0.75, 1.25):
            raise ValueError("fixed speed rate must be 0.75 or 1.25")
        length = math.floor(SOURCE_FRAMES / rate + 0.5)
        rows = [
            _combined(((min(180, math.floor(output_index * rate + 0.5)), 1.0),))
            for output_index in range(length)
        ]
    elif family == "MEAN":
        window = spec.get("window")
        if window not in (3, 5):
            raise ValueError("fixed mean window must be 3 or 5")
        radius = window // 2
        rows = [
            _combined((
                (min(180, max(0, source_index + delta)), 1.0 / window)
                for delta in range(-radius, radius + 1)
            ))
            for source_index in range(SOURCE_FRAMES)
        ]
    elif family in ("DELETE", "REPEAT_INSERT", "ADJACENT_EQUAL_INSERT"):
        positions = spec.get("positions")
        limit = 180 if family == "ADJACENT_EQUAL_INSERT" else 181
        if (
            not isinstance(positions, list) or len(positions) != GROUP_SIZE
            or len(set(positions)) != GROUP_SIZE
            or any(type(index) is not int or index not in range(limit) for index in positions)
        ):
            raise ValueError("fixed edit positions must be eighteen unique valid integers")
        selected = set(positions)
        rows = []
        for source_index in range(SOURCE_FRAMES):
            if family != "DELETE" or source_index not in selected:
                rows.append(_combined(((source_index, 1.0),)))
            if family == "REPEAT_INSERT" and source_index in selected:
                rows.append(_combined(((source_index, 1.0),)))
            if family == "ADJACENT_EQUAL_INSERT" and source_index in selected:
                rows.append(_combined(((source_index, 0.5), (source_index + 1, 0.5))))
    else:
        raise ValueError(f"unknown fixed attack family {family!r}")
    expected = {
        "FULL": 181, "CROP": 89, "SPEED_NN": 241 if spec.get("rate") == 0.75 else 145,
        "MEAN": 181, "DELETE": 163, "REPEAT_INSERT": 199,
        "ADJACENT_EQUAL_INSERT": 199,
    }[family]
    if len(rows) != expected:
        raise RuntimeError(f"fixed {spec['attack_id']} length {len(rows)} != {expected}")
    return rows


def recipe_receipt(spec: dict[str, Any]) -> dict[str, Any]:
    truth = expand_weighted_truth(spec)
    raw = json.dumps(truth, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "attack_id": spec["attack_id"], "family": spec["family"],
        "source_frames": SOURCE_FRAMES, "output_frames": len(truth),
        "weighted_truth_sha256": hashlib.sha256(raw).hexdigest(),
        "weighted_truth": truth,
        "construction_rule": (
            "float64 weighted sum, np.rint, clip[0,255], uint8"
            if any(len(row) > 1 or row[0]["weight"] != 1.0 for row in truth)
            else "exact source frame selection in RGB8"
        ),
        "truth_is_posthoc_only": True,
    }


def apply_edit_rgb8(source_rgb8, spec: dict[str, Any]):
    """Apply the adopted edit exactly; imports NumPy only in the execution phase."""

    import numpy as np

    candidate = source_rgb8
    for method in ("detach", "cpu"):
        function = getattr(candidate, method, None)
        if callable(function):
            candidate = function()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    source = np.asarray(candidate)
    if source.dtype != np.uint8 or source.shape != (181, 320, 512, 3):
        raise ValueError("attack source must be uint8 [181,320,512,3]")
    truth = expand_weighted_truth(spec)
    output = np.empty((len(truth), 320, 512, 3), dtype=np.uint8)
    for output_index, weights in enumerate(truth):
        if len(weights) == 1 and weights[0]["weight"] == 1.0:
            output[output_index] = source[weights[0]["source_index"]]
            continue
        value = np.zeros((320, 512, 3), dtype=np.float64)
        for item in weights:
            value += source[item["source_index"]].astype(np.float64) * item["weight"]
        output[output_index] = np.clip(np.rint(value), 0.0, 255.0).astype(np.uint8)
    return output, recipe_receipt(spec)


def path_error_rows(path: list[int], weighted_truth: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Post-hoc error retains both support distance and expected absolute error."""

    if len(path) != len(weighted_truth):
        raise ValueError("estimated path and weighted truth lengths differ")
    rows = []
    for received_index, (estimated, support) in enumerate(zip(path, weighted_truth)):
        distances = [abs(estimated - item["source_index"]) for item in support]
        rows.append({
            "received_index": received_index, "estimated_source_index": estimated,
            "minimum_support_absolute_error": min(distances),
            "weighted_absolute_error": sum(
                item["weight"] * abs(estimated - item["source_index"]) for item in support
            ),
        })
    return {
        "status": "EVALUATED_POSTHOC", "frames": len(rows), "rows": rows,
        "minimum_support_absolute_error_sum": sum(row["minimum_support_absolute_error"] for row in rows),
        "minimum_support_absolute_error_mean": sum(row["minimum_support_absolute_error"] for row in rows) / len(rows),
        "minimum_support_absolute_error_max": max(row["minimum_support_absolute_error"] for row in rows),
        "first_frame_offset_error": rows[0]["minimum_support_absolute_error"],
        "weighted_absolute_error_sum": sum(row["weighted_absolute_error"] for row in rows),
        "weighted_absolute_error_mean": sum(row["weighted_absolute_error"] for row in rows) / len(rows),
        "weighted_absolute_error_max": max(row["weighted_absolute_error"] for row in rows),
        "weighted_support_note": "mixed-frame source spread can make weighted absolute error positive even when the estimate belongs to the truth support",
        "weighted_truth_not_selector_input": True,
    }
