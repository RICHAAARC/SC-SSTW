"""Fixed wrong-key background normalization for the ORIGINAL blind receiver."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import fixed_key

PROTOCOL_ID = "SC-SSTW-Content-Background-Existence-V1"
RECEIVER_ID = "CONTENT_BACKGROUND_Z"
EPSILON = 1e-6
PATH_COUNT = 4284
WRONG_KEY_HEX = (
    "9dca8e4c40959bbf126f524fcba265cc336b4303226e4f87cbd2dcffe2b375c5",
    "8d0c3ccd4ccee1ce2e877d28dcad2f3fc67ee0105f800cfe8c0b31e4f984d6ec",
    "daf439c90c825b7921273ed89af57ccb75915cde7433705a043f4cf7d58db16f",
    "363db89fe336f8da9042f9bc866e1932d17e3e15a930f2b79ff5a7a11ec913d0",
    "e56325da82f9b6a197647a90d211dfd4ef61031390e664c98fca4dd615207e2e",
    "e4377efe95d3a36234e0f84e7cd8aaef4619171793f088658a361e7fc697324d",
    "cef5abcdd4eb81fb123d96feb3e987750d2eb9c38b61b950da410bcf644410e1",
    "5809292bb9e16fba722c2fa8045e65462379681fd7f22df021bdfc5da9d66986",
    "25e6de36409a9c8681c670d30a6161d5ca81a2be9dbab09cc9bbe959e96da5a3",
    "9f6b39425650a2fe5c934470c97d8849358e161abe0bd68459e91f7d7181402c",
    "ec8fb5e649de70e7d5f75e179b1dfca7faec90f02769efa476ced7391595bc70",
    "37006db392771e3b67f0a53259bafca760a46c8d868cab2a926a683bb9203a34",
    "62c07d122be704ad72ec0129ed6ae13d0c97c70def0c05153210dd3929e1d6ba",
    "81641c8441992c011292d5bc9e7bf0218aa01e04e3ef38cd3ba624d516fcab5f",
    "65f6f780f2b7a286fe6b5f1a9cf0d63373152c9a991be4bba87f9431bf31b36f",
    "a1fff9f728667bfa961469607248be87171acc76659cd854d9ae6338568bf427",
)
WRONG_KEYS = tuple(bytes.fromhex(value) for value in WRONG_KEY_HEX)


def _derived_wrong_keys() -> tuple[bytes, ...]:
    prefix = "SC-SSTW-Content-Background-Existence-V1/wrong/"
    return tuple(hashlib.sha256((prefix + f"{index:02d}").encode("utf-8")).digest() for index in range(16))


assert WRONG_KEYS == _derived_wrong_keys()
assert len(set(WRONG_KEYS)) == 16


def _empty_search(label: str, key: bytes, status: str = "NOT_RUN") -> dict:
    return {
        "label": label,
        "key_hex": key.hex(),
        "key_id": fixed_key.key_identifier(key),
        "status": status,
        "attempted_path_count": 0,
        "scored_path_count": 0,
        "score": None,
        "detection": None,
    }


def _validate_observations(observations) -> str | None:
    if set(observations) != set(range(4)):
        return "exact phases 0,1,2,3 are required"
    for phase in range(4):
        value = observations[phase]
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 5
            or value.shape[:2] != (1, 16)
            or value.shape[3:] != (40, 64)
            or not np.isfinite(value).all()
        ):
            return f"phase {phase} must be one finite (1,16,T,40,64) ndarray"
    return None


def _search(observations, label: str, key: bytes) -> dict:
    row = _empty_search(label, key)
    try:
        detection = fixed_key.read(observations, fixed_key.codebook(key))
        row.update(
            status=detection.get("status", "INVALID"),
            attempted_path_count=detection.get("attempted_path_count", 0),
            scored_path_count=detection.get("scored_path_count", 0),
            score=detection.get("existence_statistic"),
            detection=detection,
        )
        if (
            row["status"] != "SCORED"
            or row["attempted_path_count"] != PATH_COUNT
            or row["score"] is None
            or not math.isfinite(row["score"])
        ):
            row["status"] = "INVALID"
    except Exception as exc:  # Retain this fixed key's failure and continue the denominator.
        row.update(status="INVALID", error=repr(exc))
    return row


def read(observations, correct_key: bytes, spec_sha256: str) -> dict:
    """Score the correct key once and all sixteen frozen wrong keys."""
    if not isinstance(correct_key, bytes):
        raise TypeError("correct_key must be bytes")
    searches = [_empty_search("correct", correct_key)] + [
        _empty_search(f"wrong/{index:02d}", key) for index, key in enumerate(WRONG_KEYS)
    ]
    base = {
        "status": "INVALID",
        "receiver_protocol_id": PROTOCOL_ID,
        "candidate_id": RECEIVER_ID,
        "spec_sha256": spec_sha256,
        "correct_key_id": fixed_key.key_identifier(correct_key),
        "wrong_key_count": 16,
        "expected_search_count": 17,
        "expected_path_attempts": 17 * PATH_COUNT,
        "epsilon": EPSILON,
        "std_ddof": 0,
        "searches": searches,
        "wrong_scores": [None] * 16,
        "wrong_mean": None,
        "wrong_population_std": None,
        "statistic": None,
        "correct_score": None,
        "correct_detection": None,
        "actual_search_count": 0,
        "actual_path_attempts": 0,
    }
    if len(set(WRONG_KEYS)) != 16 or correct_key in WRONG_KEYS:
        return dict(base, reason="wrong keys must be unique and distinct from the correct key")
    invalid_reason = _validate_observations(observations)
    if invalid_reason is not None:
        return dict(base, reason=invalid_reason)

    searches = [_search(observations, "correct", correct_key)]
    searches.extend(_search(observations, f"wrong/{index:02d}", key) for index, key in enumerate(WRONG_KEYS))
    result = dict(
        base,
        searches=searches,
        correct_score=searches[0]["score"],
        correct_detection=searches[0]["detection"],
        actual_search_count=len(searches),
        actual_path_attempts=sum(row["attempted_path_count"] for row in searches),
    )
    if any(row["status"] != "SCORED" for row in searches):
        return dict(result, reason="all seventeen complete finite searches are required")
    wrong_scores = [row["score"] for row in searches[1:]]
    mean = float(np.mean(np.asarray(wrong_scores, dtype=np.float64)))
    std = float(np.std(np.asarray(wrong_scores, dtype=np.float64), ddof=0))
    result.update(wrong_scores=wrong_scores, wrong_mean=mean, wrong_population_std=std)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= EPSILON:
        return dict(result, reason="finite wrong-key population std must exceed 1e-6")
    statistic = (searches[0]["score"] - mean) / (std + EPSILON)
    if not math.isfinite(statistic):
        return dict(result, reason="normalized statistic is non-finite")
    result.update(
        status="SCORED",
        statistic=float(statistic),
    )
    return result


def source_statistic(views: dict, spec_sha256: str, correct_key_id: str) -> dict:
    binding = {
        "receiver_protocol_id": PROTOCOL_ID,
        "candidate_id": RECEIVER_ID,
        "spec_sha256": spec_sha256,
        "correct_key_id": correct_key_id,
    }
    if set(views) != set(fixed_key.FIXED_VIEWS) or any(
        row.get("status") != "SCORED"
        or any(row.get(key) != value for key, value in binding.items())
        or not math.isfinite(row.get("statistic", math.nan))
        for row in views.values()
    ):
        return dict(status="INVALID", statistic=None, winning_view=None, **binding)
    winner = max(fixed_key.FIXED_VIEWS, key=lambda view: (views[view]["statistic"], view))
    return dict(status="SCORED", statistic=views[winner]["statistic"], winning_view=winner, **binding)


def calibrate(calibration_sources: dict, expected_source_ids, spec_sha256: str, correct_key_id: str) -> dict:
    expected_source_ids = tuple(expected_source_ids)
    binding = {
        "receiver_protocol_id": PROTOCOL_ID,
        "candidate_id": RECEIVER_ID,
        "spec_sha256": spec_sha256,
        "correct_key_id": correct_key_id,
    }
    ok = (
        len(expected_source_ids) == 4
        and set(calibration_sources) == set(expected_source_ids)
        and all(
            row.get("status") == "SCORED"
            and math.isfinite(row.get("statistic", math.nan))
            and all(row.get(key) == value for key, value in binding.items())
            for row in calibration_sources.values()
        )
    )
    return {
        "status": "FROZEN" if ok else "UNCALIBRATED",
        "threshold": max(row["statistic"] for row in calibration_sources.values()) + EPSILON if ok else None,
        "guard": EPSILON,
        "source_count": 4,
        "rank_resolution": "1/5",
        "sources": calibration_sources,
        **binding,
    }


def decide(row: dict, calibration: dict) -> dict:
    statistic = row.get("statistic")
    if row.get("status") != "SCORED" or statistic is None or not math.isfinite(statistic):
        return {"status": "INVALID", "detected": None, "statistic": statistic}
    binding = {
        "receiver_protocol_id": PROTOCOL_ID,
        "candidate_id": RECEIVER_ID,
        "spec_sha256": row.get("spec_sha256"),
        "correct_key_id": row.get("correct_key_id"),
    }
    if calibration.get("status") != "FROZEN" or any(calibration.get(key) != value for key, value in binding.items()):
        return {"status": "UNCALIBRATED", "detected": None, "statistic": statistic}
    detected = statistic > calibration["threshold"]
    return {
        "status": "DETECTED" if detected else "REJECTED",
        "detected": detected,
        "statistic": statistic,
        "threshold": calibration["threshold"],
    }
