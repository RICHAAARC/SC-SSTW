"""Pure frame/support geometry for a caller-supplied received-to-source map.

This module estimates no time path. A supplied map may be an estimate, a clean
identity map, or an oracle used for conditional evaluation.
"""
from __future__ import annotations
from typing import Any


def plan_given_time_mapping(path) -> dict[str, Any]:
    """Map the caller-supplied source span back to received-frame indices."""

    source_path = list(path)
    if not source_path or any(type(item) is not int for item in source_path):
        raise ValueError("estimated path must contain integer source positions")
    if any(item < 0 or item >= 181 for item in source_path):
        raise ValueError("estimated path leaves the public source clock")
    if any(right < left for left, right in zip(source_path, source_path[1:])):
        raise ValueError("estimated path must be nondecreasing")
    source_start, source_end = source_path[0], source_path[-1]
    anchor = (source_start // 4) * 4
    available = source_end - anchor + 1
    output_frames = ((available - 1) // 4) * 4 + 1
    if output_frames < 1:
        raise ValueError("estimated visible span is empty")
    output_sources = list(range(anchor, anchor + output_frames))
    received_map = []
    synthetic = []
    exact_source_hits = set(source_path)
    for output_index, source_index in enumerate(output_sources):
        nearest = min(
            range(len(source_path)),
            key=lambda received_index: (abs(source_path[received_index] - source_index), received_index),
        )
        received_map.append(nearest)
        if source_index not in exact_source_hits:
            synthetic.append(output_index)
    repeated = [
        index for index in range(1, len(received_map))
        if received_map[index] == received_map[index - 1]
    ]
    selected = set(received_map)
    dropped = [index for index in range(len(source_path)) if index not in selected]
    support = min(44, (output_frames - 1) // 4)
    status = "SUPPORTED" if support >= 1 else "UNSUPPORTED"
    return {
        "status": status,
        "reason": None if status == "SUPPORTED" else "INSUFFICIENT_VISIBLE_WAN_SUPPORT",
        "source_start": source_start, "source_end": source_end,
        "visible_span_frames": available,
        "anchor": anchor, "front_boundary_copies": source_start - anchor,
        "tail_discarded_source_positions": available - output_frames,
        "output_frames": output_frames, "source_coordinate_map": output_sources,
        "received_index_map": received_map,
        "synthetic_output_indices": synthetic,
        "repeated_output_indices": repeated,
        "dropped_received_indices": dropped,
        "wan_support": support,
        "wan_input_rule": "output_frames == 1 + 4*k; R=min(44,k); first Wan latent excluded",
        "mapping_rule": "nearest received source estimate; equal distance chooses earlier received index",
        "mapping_origin": "CALLER_SUPPLIED",
        "conditional_on_given_mapping": True,
        "blind_sync_performed": False,
    }
