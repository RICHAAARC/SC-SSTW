"""Frozen saved-RGB roster and posthoc truth for uncertainty follow-up V1.

The receiver seals contain only opaque observation/query identifiers, public N,
RGB identities and query-key identities. Source, condition, construction and
coverage roles are joined only after the complete decision seal.
"""
from __future__ import annotations

import hashlib
import json

from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align

MARKER = "uncertainty-v1-followup-20261008"
ROOT_PROTOCOL_FREEZE_SHA256 = "6fa6cabd599104302ff5a019d749ea543816da3f47774905f3f3ac9d15fd5aeb"
FROZEN_RULE_SHA256 = "4c0a14c1493ff642405d13111866b53d9373477668e572e964bb414239b796fa"
FROZEN_RULE_FILE_SHA256 = "9b4449569daabfee6d572fb849afa61d80175433b0585fbf3eaac4d27e84ed7c"
PRIOR_RUN_ID = "20261008T085244020205Z"
SOURCE_IDS = ("C1", "C2")
KEY_LABELS = ("K0", "K1")

FIXED_DENOMINATOR = {
    "sources": 2,
    "queries": 22,
    "observations": 20,
    "primary_queries": 10,
    "alias_controls": 4,
    "regression_controls": 8,
    "positive_queries": 16,
    "negative_queries": 6,
    "sync_candidates": 11902,
    "framewise_frames": 3012,
    "framewise_batch8": 394,
    "max_payload_reads": 22,
    "max_votes": 802560,
    "max_time_bit_rows": 26752,
    "max_bit_rows": 704,
    "sync_coverage_denominator": 10,
    "weak_identity_coverage_denominator": 10,
}


def _view(view: str, **parameters) -> dict:
    return {"view": view, "parameters": parameters}


_PER_SOURCE = (
    dict(condition="A_M05", key="K0", **_view("SHORT89", start=0), role="PRIMARY_PROBE", focus="IDENTITY_FOCUS"),
    dict(condition="A_M05", key="K0", **_view("SHORT89", start=46), role="PRIMARY_PROBE", focus="IDENTITY_FOCUS"),
    dict(condition="A_M05", key="K0", **_view("SHORT89", start=92), role="PRIMARY_PROBE", focus="IDENTITY_FOCUS"),
    dict(condition="A_M05", key="K0", **_view("DELETE177", b=2, k=1), role="ALIAS_CONTROL", focus="ALIAS"),
    dict(condition="A_M05", key="K0", **_view("DELETE177", b=2, k=44), role="PRIMARY_PROBE", focus="SYNC_FOCUS"),
    dict(condition="A_M05", key="K0", **_view("DELETE177", b=2, k=132), role="PRIMARY_PROBE", focus="SYNC_FOCUS"),
    dict(condition="A_M05", key="K0", **_view("DELETE177", b=2, k=176), role="ALIAS_CONTROL", focus="ALIAS"),
    dict(condition="A_M05", key="K0", **_view("CROP177", b=2), role="REGRESSION_CONTROL"),
    dict(condition="A_M05", key="K1", **_view("CROP177", b=2), role="REGRESSION_CONTROL"),
    dict(condition="OFF", key="K0", **_view("CROP177", b=2), role="REGRESSION_CONTROL"),
    dict(condition="B_M05", key="K0", **_view("CROP177", b=2), role="REGRESSION_CONTROL"),
)


def received_index_map(view: str, parameters: dict) -> tuple[int, ...]:
    if view == "SHORT89":
        start = parameters.get("start")
        if type(start) is not int or start not in range(93):
            raise ValueError("fixed SHORT89 start")
        return tuple(range(start, start + 89))
    if view == "CROP177":
        b = parameters.get("b")
        if b != 2:
            raise ValueError("fixed CROP177 b2")
        return tuple(range(b, b + 177))
    if view == "DELETE177":
        b, k = parameters.get("b"), parameters.get("k")
        if b != 2 or type(k) is not int or k not in range(1, 177):
            raise ValueError("fixed DELETE177 b2/k")
        return tuple(range(b, b + k)) + tuple(range(b + k + 1, b + 178))
    raise ValueError("fixed view")


def true_action(view: str, parameters: dict) -> list[int]:
    if view == "SHORT89":
        return align.phase_map(89, parameters["start"] % 4)
    if view == "CROP177":
        return deletion.correction({"family": "H0", "b": parameters["b"], "k": None})[
            "received_index_map"
        ]
    if view == "DELETE177":
        return deletion.correction(
            {"family": "H1", "b": parameters["b"], "k": parameters["k"]}
        )["received_index_map"]
    raise ValueError("fixed view")


def query_roster() -> list[dict]:
    rows = []
    observations = {}
    for source in SOURCE_IDS:
        for template in _PER_SOURCE:
            observation_key = (
                source,
                template["condition"],
                template["view"],
                tuple(sorted(template["parameters"].items())),
            )
            if observation_key not in observations:
                observations[observation_key] = f"UO{len(observations):03d}"
            mapping = received_index_map(template["view"], template["parameters"])
            rows.append(
                {
                    "query_id": f"UQ{len(rows):03d}",
                    "observation_id": observations[observation_key],
                    "source": source,
                    **template,
                    "frames": len(mapping),
                }
            )
    return rows


def blind_roster() -> list[dict]:
    return [
        {name: row[name] for name in ("query_id", "observation_id", "frames")}
        for row in query_roster()
    ]


def role(row: dict) -> dict:
    expected_accept = row["condition"] == "A_M05" and row["key"] == "K0"
    return {
        "expected_accept": expected_accept,
        "required_for_false_claim": True,
        "positive_or_negative": "POSITIVE" if expected_accept else "NEGATIVE",
    }


def regression_expected(row: dict) -> dict | None:
    if row["role"] != "REGRESSION_CONTROL":
        return None
    key = (row["condition"], row["key"])
    expected = {
        ("A_M05", "K0"): ("ACCEPT", "ACCEPT"),
        ("A_M05", "K1"): ("REJECT", "REJECT_LOW_SYNC"),
        ("OFF", "K0"): ("REJECT", "REJECT_LOW_SYNC"),
        ("B_M05", "K0"): ("REJECT", "REJECT_IDENTITY"),
    }[key]
    return {"decision": expected[0], "reason": expected[1]}


def roster_sha256() -> str:
    raw = json.dumps(query_roster(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def input_contract_sha256(inputs: dict) -> str:
    raw = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_roster() -> None:
    rows = query_roster()
    if len(rows) != 22 or len({row["query_id"] for row in rows}) != 22:
        raise ValueError("fixed 22-query roster")
    if len({row["observation_id"] for row in rows}) != 20:
        raise ValueError("fixed 20-observation roster")
    if sum(row["role"] == "PRIMARY_PROBE" for row in rows) != 10:
        raise ValueError("fixed primary coverage denominator")
    if sum(row["role"] == "ALIAS_CONTROL" for row in rows) != 4:
        raise ValueError("fixed alias controls")
    if sum(row["role"] == "REGRESSION_CONTROL" for row in rows) != 8:
        raise ValueError("fixed regression controls")
    if sum(role(row)["expected_accept"] for row in rows) != 16:
        raise ValueError("fixed positive denominator")


validate_roster()
