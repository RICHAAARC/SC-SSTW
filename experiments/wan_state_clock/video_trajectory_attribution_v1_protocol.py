"""Frozen experiment roster and posthoc truth for trajectory attribution V1.

This module is intentionally outside the receiver core. Semantic source,
condition, view, and expected-action labels are experiment preparation/posthoc
data and never appear in a blind evidence seal.
"""
from __future__ import annotations

import hashlib
import json

from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align

SOURCE_IDS = ("DEV", "C1", "C2")
VIDEO_IDS = ("OFF", "A_P1", "A_M05", "B_M05")
KEY_LABELS = ("K0", "K1")
VIEW_MAPS = {
    "FULL181": tuple(range(181)),
    "CROP177_P1": tuple(range(1, 178)),
    "CROP177_P2": tuple(range(2, 179)),
    "CROP177_P3": tuple(range(3, 180)),
    "SHORT89_S37": tuple(range(37, 126)),
    "SHORT89_S38": tuple(range(38, 127)),
    "SHORT89_S39": tuple(range(39, 128)),
    "DELETE177_B2K88": tuple(range(2, 90)) + tuple(range(91, 180)),
}


def fixed_denominator(sources: int = 3) -> dict[str, int]:
    if type(sources) is not int or sources < 1:
        raise ValueError("positive source count")
    per_video_r = 44 + 3 * 44 + 3 * 22 + 44
    return {
        "sources": sources,
        "native_trajectories": 3 * sources,
        "received_conditions": 4 * sources,
        "physical_observations": 32 * sources,
        "logical_queries": 64 * sources,
        "sync_candidates": 24928 * sources,
        "framewise_frames": 4624 * sources,
        "framewise_batches": 604 * sources,
        "max_payload_reads": 64 * sources,
        "logical_payload_votes": 32 * 30 * per_video_r * 8 * sources,
        "logical_time_bit_rows": 32 * per_video_r * 8 * sources,
        "logical_final_bits": 32 * 64 * sources,
    }


def query_roster(source_ids: tuple[str, ...] = SOURCE_IDS) -> list[dict]:
    rows = []
    source_ordinals = {source_id: index for index, source_id in enumerate(SOURCE_IDS)}
    for source_id in source_ids:
        source_ordinal = source_ordinals[source_id]
        for video_ordinal, video_id in enumerate(VIDEO_IDS):
            for view_ordinal, (view_id, mapping) in enumerate(VIEW_MAPS.items()):
                observation_ordinal = source_ordinal * 32 + video_ordinal * 8 + view_ordinal
                observation_id = f"O{observation_ordinal:03d}"
                for key_ordinal, key_label in enumerate(KEY_LABELS):
                    query_id = f"Q{observation_ordinal * 2 + key_ordinal:03d}"
                    rows.append(
                        {
                            "query_id": query_id,
                            "observation_id": observation_id,
                            "source_id": source_id,
                            "video_id": video_id,
                            "view_id": view_id,
                            "key_label": key_label,
                            "frames": len(mapping),
                        }
                    )
    return rows


def true_action_for_view(view_id: str) -> list[int]:
    if view_id == "FULL181":
        return align.phase_map(181, 0)
    if view_id.startswith("CROP177_P"):
        b = int(view_id[-1])
        return deletion.correction({"family": "H0", "b": b, "k": None})["received_index_map"]
    if view_id.startswith("SHORT89_S"):
        start = int(view_id.rsplit("S", 1)[1])
        return align.phase_map(89, start % 4)
    if view_id == "DELETE177_B2K88":
        return deletion.correction({"family": "H1", "b": 2, "k": 88})["received_index_map"]
    raise ValueError("fixed view id")


def calibration_label(query: dict) -> dict:
    video_id = query["video_id"]
    key_label = query["key_label"]
    positive = video_id == "A_M05" and key_label == "K0"
    identity_class = None
    if (video_id, key_label) in (("OFF", "K0"), ("A_M05", "K1"), ("B_M05", "K0")):
        identity_class = video_id + "/" + key_label
    return {
        "sync_null": video_id in ("OFF", "A_P1") or key_label == "K1",
        "identity_null_class": identity_class,
        "positive": positive,
    }


def role(query: dict) -> dict:
    expected = query["video_id"] == "A_M05" and query["key_label"] == "K0"
    return {
        "expected_accept": expected,
        "required_for_false_claim": True,
        "control": (
            "SYNC_NULL_PAYLOAD_BEARING"
            if query["video_id"] == "A_P1"
            else "PRIMARY_POSITIVE"
            if expected
            else "IDENTITY_OR_SYNC_NULL"
        ),
    }


def roster_sha256(config_sources: dict) -> str:
    raw = json.dumps(config_sources, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
