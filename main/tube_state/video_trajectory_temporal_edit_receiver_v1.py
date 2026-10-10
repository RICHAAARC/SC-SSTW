"""Blind temporal-edit receiver over the public framewise clock.

The selector accepts only a received framewise latent, a key, and this public
protocol.  Attack recipes, edit labels, message bits, and post-hoc source truth
are deliberately absent.  The dynamic program uses every received frame and a
fixed 181-position public source clock with nondecreasing stay/advance/skip
transitions and free endpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class PublicProtocol:
    method_version: str = "trajectory-temporal-edit-receiver-v1"
    source_frames: int = 181
    latent_channels: int = 4
    latent_height: int = 40
    latent_width: int = 64
    minimum_wan_support: int = 1
    maximum_wan_support: int = 44


PUBLIC = PublicProtocol()


def public_receipt(public: PublicProtocol = PUBLIC) -> dict[str, Any]:
    _require_public(public)
    return {
        **asdict(public),
        "selector_inputs": "received framewise latent + key + public protocol only",
        "local_score": "signed_projection/rho",
        "path_objective": "sum local_score over one source index per received frame",
        "transition_grammar": "nondecreasing source indices: stay, advance-one, arbitrary-skip",
        "endpoints": "free source start and free source end",
        "tie_rule": "exact equal best complete paths are UNRESOLVED",
        "payload_selection": False,
        "truth_inputs": False,
    }


def _require_public(public: PublicProtocol) -> None:
    if not isinstance(public, PublicProtocol):
        raise TypeError("public must be PublicProtocol")
    if public.source_frames != 181:
        raise ValueError("public source clock must contain 181 positions")
    if (public.latent_channels, public.latent_height, public.latent_width) != (4, 40, 64):
        raise ValueError("public framewise latent geometry changed")
    if not 1 <= public.minimum_wan_support <= public.maximum_wan_support == 44:
        raise ValueError("invalid public Wan support")


def score_framewise(received_latent, key: str, public: PublicProtocol = PUBLIC) -> dict[str, Any]:
    """Build the full received-frame x public-source clock evidence matrix."""

    import numpy as np
    from main.tube_state import video_trajectory_payload_framewise_sync_v1 as clock

    _require_public(public)
    value = np.asarray(received_latent)
    if value.dtype != np.float32 or value.ndim != 4:
        raise TypeError("received framewise latent must be float32 [N,4,40,64]")
    if tuple(value.shape[1:]) != (4, 40, 64) or int(value.shape[0]) < 1:
        raise ValueError("received framewise latent geometry mismatch")
    if not bool(np.isfinite(value).all()):
        raise ValueError("received framewise latent contains nonfinite values")

    source_weights = np.empty((public.source_frames, 4, 40, 64), dtype=np.float32)
    for source_index in range(public.source_frames):
        tubelet = source_index // clock.PUBLIC.tubelet_length
        age = source_index % clock.PUBLIC.tubelet_length
        sign = clock.sync_sign(key, tubelet, clock.PUBLIC)
        for patch_y, patch_x in clock.spatial_patch_coordinates(40, 64, clock.PUBLIC):
            source_weights[source_index, :, patch_y:patch_y + 4, patch_x:patch_x + 4] = (
                np.float32(sign) * clock.patch_direction(
                    key, tubelet, patch_y, patch_x, clock.PUBLIC,
                )[age]
            )
    flat_received = value.astype(np.float64, copy=False).reshape(len(value), -1)
    flat_weights = source_weights.astype(np.float64, copy=False).reshape(public.source_frames, -1)
    signed_projection = flat_received @ flat_weights.T
    rho_source = np.sum(flat_weights * flat_weights, axis=1)
    rho = np.broadcast_to(rho_source, signed_projection.shape).copy()
    if not bool(np.isfinite(signed_projection).all()) or not bool(np.isfinite(rho).all()):
        raise ValueError("framewise clock evidence is nonfinite")
    if bool((rho <= 0.0).any()):
        raise ValueError("framewise clock direction has nonpositive rho")
    return {
        "status": "SCORED",
        "method_version": public.method_version,
        "received_frames": int(value.shape[0]),
        "source_frames": public.source_frames,
        "signed_projection": signed_projection,
        "rho": rho,
        "truth_inputs": False,
        "message_inputs": False,
        "attack_inputs": False,
    }


def _finite_matrix(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    rows = [list(row) for row in value]
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be a nonempty matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    result = []
    for row in rows:
        converted = [float(item) for item in row]
        if any(not math.isfinite(item) for item in converted):
            raise ValueError(f"{name} contains nonfinite values")
        result.append(converted)
    return result


def solve_monotone(
    signed_projection: Sequence[Sequence[float]],
    rho: Sequence[Sequence[float]],
    public: PublicProtocol = PUBLIC,
) -> dict[str, Any]:
    """Return a unique best complete monotone path or retain an exact tie.

    Two ranks per state are retained with explicit backpointers, so the reported
    runner-up is a different complete path rather than a duplicate terminal
    state or a local alternative.
    """

    _require_public(public)
    numerator = _finite_matrix(signed_projection, "signed_projection")
    denominator = _finite_matrix(rho, "rho")
    if len(numerator) != len(denominator) or len(numerator[0]) != len(denominator[0]):
        raise ValueError("signed_projection and rho shapes differ")
    if len(numerator[0]) != public.source_frames:
        raise ValueError("clock evidence must have 181 public source columns")
    if any(item <= 0.0 for row in denominator for item in row):
        raise ValueError("rho must be positive")
    local = [
        [num / den for num, den in zip(num_row, den_row)]
        for num_row, den_row in zip(numerator, denominator)
    ]
    received_frames = len(local)
    source_frames = public.source_frames
    neg_inf = float("-inf")
    scores = [[[neg_inf, neg_inf] for _ in range(source_frames)] for _ in range(received_frames)]
    back = [[[None, None] for _ in range(source_frames)] for _ in range(received_frames)]
    for source_index in range(source_frames):
        scores[0][source_index][0] = local[0][source_index]
    for received_index in range(1, received_frames):
        prefix_top: list[tuple[float, int, int]] = []
        for source_index in range(source_frames):
            for rank in range(2):
                prior = scores[received_index - 1][source_index][rank]
                if prior != neg_inf:
                    prefix_top.append((prior, source_index, rank))
                    prefix_top.sort(key=lambda item: item[0], reverse=True)
                    del prefix_top[2:]
            for rank, (prior, previous_source, previous_rank) in enumerate(prefix_top):
                scores[received_index][source_index][rank] = prior + local[received_index][source_index]
                back[received_index][source_index][rank] = (previous_source, previous_rank)
    finals = sorted(
        (
            (scores[-1][source_index][rank], source_index, rank)
            for source_index in range(source_frames)
            for rank in range(2)
            if scores[-1][source_index][rank] != neg_inf
        ),
        key=lambda item: item[0], reverse=True,
    )
    if not finals:
        return {
            "status": "UNRESOLVED", "reason": "NO_COMPLETE_PATH",
            "path": None, "best_score": None, "runner_up_score": None,
            "score_gap": None, "exact_tie": False, "truth_inputs": False,
        }
    best_score, best_source, best_rank = finals[0]
    runner_up_score = finals[1][0] if len(finals) > 1 else None

    def reconstruct(final_source: int, final_rank: int) -> list[int]:
        result = [final_source]
        source_index, rank = final_source, final_rank
        for received_index in range(received_frames - 1, 0, -1):
            source_index, rank = back[received_index][source_index][rank]
            result.append(source_index)
        result.reverse()
        return result

    best_path = reconstruct(best_source, best_rank)
    runner_up_path = reconstruct(finals[1][1], finals[1][2]) if len(finals) > 1 else None
    if runner_up_path == best_path:
        raise RuntimeError("dynamic-program runner-up duplicated the best complete path")
    exact_tie = runner_up_score is not None and runner_up_score == best_score
    if exact_tie:
        return {
            "status": "UNRESOLVED", "reason": "EXACT_BEST_PATH_TIE",
            "path": None, "best_path": best_path, "runner_up_path": runner_up_path,
            "best_score": best_score,
            "runner_up_score": runner_up_score, "score_gap": 0.0,
            "exact_tie": True, "received_frames": received_frames,
            "source_frames": source_frames, "truth_inputs": False,
        }
    path = best_path
    transitions = [right - left for left, right in zip(path, path[1:])]
    return {
        "status": "ESTIMATED", "reason": None, "path": path,
        "best_path": best_path, "runner_up_path": runner_up_path,
        "best_score": best_score, "runner_up_score": runner_up_score,
        "score_gap": best_score - runner_up_score if runner_up_score is not None else None,
        "exact_tie": False, "received_frames": received_frames,
        "source_frames": source_frames, "source_start": path[0],
        "source_end": path[-1],
        "transition_counts": {
            "stay": sum(delta == 0 for delta in transitions),
            "advance_one": sum(delta == 1 for delta in transitions),
            "skip": sum(delta > 1 for delta in transitions),
        },
        "truth_inputs": False,
    }


def decode_visible_span(path: Sequence[int], public: PublicProtocol = PUBLIC) -> dict[str, Any]:
    """Map the estimated visible source span back to received-frame indices."""

    _require_public(public)
    source_path = list(path)
    if not source_path or any(type(item) is not int for item in source_path):
        raise ValueError("estimated path must contain integer source positions")
    if any(item < 0 or item >= public.source_frames for item in source_path):
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
    support = min(public.maximum_wan_support, (output_frames - 1) // 4)
    status = "SUPPORTED" if support >= public.minimum_wan_support else "UNSUPPORTED"
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
        "truth_inputs": False,
    }


def blind_receive(received_latent, key: str, public: PublicProtocol = PUBLIC) -> dict[str, Any]:
    evidence = score_framewise(received_latent, key, public)
    estimate = solve_monotone(evidence["signed_projection"], evidence["rho"], public)
    operation = decode_visible_span(estimate["path"], public) if estimate["status"] == "ESTIMATED" else None
    return {
        "status": operation["status"] if operation is not None else "UNRESOLVED",
        "public": public_receipt(public),
        "evidence_shape": [evidence["received_frames"], evidence["source_frames"]],
        "estimate": estimate,
        "operation": operation,
        "truth_inputs": False,
    }
