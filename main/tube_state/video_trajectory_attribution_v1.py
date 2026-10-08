"""Pure receiver math, calibration, and decisions for attribution V1.

The core accepts sealed receiver evidence plus explicit opaque development labels.
Experiment source/condition/view rosters and posthoc truth live outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align

CLAIM_MESSAGE = "OKOK"
CLAIM_BITS = tuple(payload.message_bits(CLAIM_MESSAGE))
RULE_FORMULA_VERSION = "trajectory-attribution-v1-action-folded-development-maxima"
SCIENTIFIC_REASONS = frozenset(
    {
        "ACCEPT",
        "REJECT_LOW_SYNC",
        "UNCERTAIN_SYNC",
        "REJECT_IDENTITY",
        "UNCERTAIN_IDENTITY_WEAK",
    }
)


def _action(mapping: list[int]) -> str:
    raw = json.dumps(list(mapping), separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _candidate_rows(readout: dict, frames: int, key: str) -> tuple[list[dict], str]:
    if frames == 177:
        if readout.get("truth_inputs") is not False or readout.get("key_id") != align.sync.key_identifier(key):
            raise ValueError("blind/key mismatch")
        rebuilt = deletion.reduce_cache(readout["frame_cache"], key)
        if (
            readout.get("status") != rebuilt["status"]
            or readout.get("candidate_rows") != rebuilt["candidate_rows"]
            or readout.get("local_grid") != rebuilt["local_grid"]
            or readout.get("counts") != rebuilt["counts"]
        ):
            raise ValueError("incomplete/inconsistent J709 support")
        if rebuilt["status"] != "COMPLETE":
            raise ValueError("J709 not complete")
        rows = []
        for index, row in enumerate(rebuilt["candidate_rows"]):
            if row["status"] != "SCORED" or not math.isfinite(row["score"]):
                raise ValueError("nonfinite J709 row")
            path = {name: row[name] for name in ("family", "b", "k")}
            operation = deletion.correction(path)
            rows.append(
                {
                    "candidate_index": index,
                    "path": path,
                    "source_map": deletion.path_map(path),
                    "score": float(row["score"]),
                    "operation": operation,
                    "action_id": _action(operation["received_index_map"]),
                }
            )
        if len(rows) != 709:
            raise ValueError("J709 count")
        return rows, "J709"

    if frames not in (89, 181):
        raise ValueError("public N must be 181/177/89")
    estimate = align.estimate(readout, frames, key)
    if estimate["status"] not in ("ESTIMATED", "UNRESOLVED") or (
        estimate["status"] == "UNRESOLVED" and estimate.get("reason") != "TIED_TOP"
    ):
        raise ValueError("G search not complete")
    rows = []
    for index, row in enumerate(readout["candidate_rows"]):
        if row["status"] != "SCORED" or not math.isfinite(row["score"]):
            raise ValueError("nonfinite G row")
        offset = row["source_offset"]
        phase = offset % 4
        mapping = align.phase_map(frames, phase)
        operation = {
            "received_index_map": mapping,
            "phase": phase,
            "synthetic_output_indices": list(range(phase)),
            "dropped_received_indices": sorted(set(range(frames)) - set(mapping)),
            "inserted_gap_output_index": None,
            "operation": "prepend+truncate",
        }
        rows.append(
            {
                "candidate_index": index,
                "path": {"family": "G", "offset": offset, "phase": phase},
                "source_map": list(range(offset, offset + frames)),
                "score": float(row["score"]),
                "operation": operation,
                "action_id": _action(mapping),
            }
        )
    expected = 1 if frames == 181 else 93
    if len(rows) != expected:
        raise ValueError("G candidate count")
    return rows, "SINGLETON" if frames == 181 else "G93"


def _q177(readout: dict) -> dict[tuple[int, int], float]:
    cache = readout["frame_cache"]
    if len(cache) != 885:
        raise ValueError("fixed frame cache count")
    values = {}
    for expected, row in zip(((r, d) for r in range(177) for d in range(5)), cache):
        r, d = expected
        if (row.get("r"), row.get("d")) != expected or row.get("source_index") != r + d:
            raise ValueError("frame cache geometry")
        numerator = row.get("numerator")
        rho = row.get("rho")
        if (
            row.get("status") != "SCORED"
            or not isinstance(numerator, (int, float))
            or not isinstance(rho, (int, float))
            or not math.isfinite(numerator)
            or not math.isfinite(rho)
            or rho <= 0
        ):
            raise ValueError("frame cache q unavailable")
        values[expected] = numerator / rho
    return values


def _q89(readout: dict) -> dict[int, float]:
    grouped: dict[int, list[dict]] = {offset: [] for offset in range(93)}
    for row in readout["local_rows"]:
        offset = row.get("source_offset")
        if offset not in grouped:
            raise ValueError("local offset geometry")
        grouped[offset].append(row)
    q_values = {}
    for offset, rows in grouped.items():
        expected = align.sync.tubelet_support_rows(89, offset, align.PUBLIC)
        if len(rows) != len(expected):
            raise ValueError("local support count")
        weighted = 0.0
        numerator_total = 0.0
        rho_total = 0.0
        observed = 0
        for row, support in zip(rows, expected):
            for name in ("source_tubelet", "received_indices", "source_indices", "source_ages", "observed_frames"):
                if row.get(name) != support[name]:
                    raise ValueError("local support geometry")
            numerator = row.get("signed_projection")
            rho = row.get("rho")
            q = row.get("q")
            if (
                row.get("status") != "SCORED"
                or not isinstance(numerator, (int, float))
                or not isinstance(rho, (int, float))
                or not isinstance(q, (int, float))
                or not math.isfinite(numerator)
                or not math.isfinite(rho)
                or not math.isfinite(q)
                or rho <= 0
                or not math.isclose(q, numerator / rho, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("local q/numerator/rho mismatch")
            count = row["observed_frames"]
            weighted += count * q
            numerator_total += numerator
            rho_total += rho
            observed += count
        if observed != 89:
            raise ValueError("local observed-frame denominator")
        candidate = readout["candidate_rows"][offset]
        recomputed = numerator_total / rho_total
        if (
            candidate.get("source_offset") != offset
            or candidate.get("numerator") != numerator_total
            or candidate.get("denominator") != rho_total
            or candidate.get("score") != recomputed
        ):
            raise ValueError("original candidate S mismatch")
        q_values[offset] = weighted / 89.0
    return q_values


def _local_action_contrast(readout: dict, frames: int, rows: list[dict], top: list[dict]) -> float | None:
    if frames == 181:
        return None
    contrasts = []
    if frames == 177:
        q = _q177(readout)
        for high in top:
            for other in rows:
                if other["action_id"] == high["action_id"]:
                    continue
                differing = [
                    r
                    for r, (source_high, source_other) in enumerate(zip(high["source_map"], other["source_map"]))
                    if source_high != source_other
                ]
                if not differing:
                    raise ValueError("different actions without different source map")
                contrasts.append(
                    sum(
                        q[(r, high["source_map"][r] - r)] - q[(r, other["source_map"][r] - r)]
                        for r in differing
                    )
                    / len(differing)
                )
    elif frames == 89:
        q = _q89(readout)
        for high in top:
            high_q = q[high["path"]["offset"]]
            for other in rows:
                if other["action_id"] != high["action_id"]:
                    contrasts.append(high_q - q[other["path"]["offset"]])
    if not contrasts or any(not math.isfinite(value) for value in contrasts):
        raise ValueError("local different-action contrast unavailable")
    return min(contrasts)


def summarize_sync(readout: dict, frames: int, key: str) -> dict:
    try:
        rows, search = _candidate_rows(readout, frames, key)
        M = max(row["score"] for row in rows)
        atol = align.PUBLIC.tie_atol
        top = [row for row in rows if abs(row["score"] - M) <= atol]
        action_order = []
        representatives = {}
        for row in top:
            if row["action_id"] not in representatives:
                action_order.append(row["action_id"])
                representatives[row["action_id"]] = row
        m = _local_action_contrast(readout, frames, rows, top)
        chosen = representatives[action_order[0]] if len(action_order) == 1 else None
        return {
            "status": "COMPLETE",
            "search": search,
            "frames": frames,
            "M": M,
            "m": m,
            "m_definition": None if frames == 181 else "minimum local q contrast over every top path and every different-action path",
            "top_paths": [row["path"] for row in top],
            "top_action_ids": action_order,
            "top_action_count": len(action_order),
            "unique_action": len(action_order) == 1,
            "representative_path": chosen["path"] if chosen else top[0]["path"],
            "chosen_action": chosen["operation"] if chosen else None,
            "candidate_count": len(rows),
            "truth_inputs": False,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "TECHNICAL_INCOMPLETE",
            "frames": frames,
            "error": str(exc),
            "M": None,
            "m": None,
            "top_paths": [],
            "top_action_ids": [],
            "top_action_count": 0,
            "unique_action": False,
            "representative_path": None,
            "chosen_action": None,
            "truth_inputs": False,
        }


def identity_evidence(detail: dict, claim_bits=CLAIM_BITS) -> dict:
    try:
        if detail["status"] != "READ" or detail.get("truth_inputs") is not False:
            raise ValueError("blind payload unavailable")
        rows = detail["bit_rows"]
        if len(rows) != 32 or tuple(claim_bits) != CLAIM_BITS:
            raise ValueError("fixed public OKOK claim required")
        margins = []
        decoded = []
        for index, row in enumerate(rows):
            if row["bit_index"] != index:
                raise ValueError("bit roster")
            values = (row["ones"], row["zeros"], row["count"], row["decoded"])
            if any(type(value) is not int for value in values):
                raise ValueError("integer vote evidence required")
            ones, zeros, count, value = values
            if count <= 0 or ones < 0 or zeros < 0 or ones + zeros != count:
                raise ValueError("vote counts")
            if value not in (0, 1):
                raise ValueError("decoded bit")
            reconstructed = Counter(([1] * ones) + ([0] * zeros)).most_common(1)[0][0]
            if reconstructed != value and ones != zeros:
                raise ValueError("decoded/count mismatch")
            decoded.append(value)
            margins.append((2 * claim_bits[index] - 1) * (ones - zeros) / count)
        exact = decoded == list(claim_bits)
        return {
            "status": "COMPLETE",
            "attempted": True,
            "I": min(margins),
            "exact32": exact,
            "decoded_bits": decoded,
            "signed_claim_margins": margins,
            "claim_message": CLAIM_MESSAGE,
            "truth_inputs": False,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "TECHNICAL_INCOMPLETE",
            "attempted": True,
            "error": str(exc),
            "I": None,
            "exact32": None,
            "claim_message": CLAIM_MESSAGE,
            "truth_inputs": False,
        }


def _required(rows: list[dict], predicate, label: str) -> list[dict]:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        raise ValueError("missing " + label)
    if any(row["sync"]["status"] != "COMPLETE" for row in selected):
        raise ValueError("technical sync failure in " + label)
    return selected


def freeze_rules(
    development_rows: list[dict],
    development_labels: dict[str, dict],
    source_roster_sha256: str,
    config_sha256: str,
) -> dict:
    """Freeze numeric maxima once from DEV64 plus explicit post-seal labels."""
    try:
        query_ids = [row["query_id"] for row in development_rows]
        if len(query_ids) != 64 or len(set(query_ids)) != 64 or set(query_ids) != set(development_labels):
            raise ValueError("complete DEV64 opaque roster and labels required")
        labeled = [dict(row, calibration=development_labels[row["query_id"]]) for row in development_rows]
        if any(
            row["identity"].get("attempted") is True and row["identity"].get("status") != "COMPLETE"
            for row in labeled
        ):
            raise ValueError("attempted DEV payload failure")
        thresholds = {}
        for N in (181, 177, 89):
            same = [row for row in labeled if row["frames"] == N]
            sync_null = _required(same, lambda row: row["calibration"].get("sync_null") is True, "sync-null")
            M_values = [row["sync"]["M"] for row in sync_null]
            if any(not math.isfinite(value) for value in M_values):
                raise ValueError("nonfinite M")
            m_values = [row["sync"]["m"] for row in sync_null if row["sync"].get("m") is not None]
            if N != 181 and (not m_values or any(not math.isfinite(value) for value in m_values)):
                raise ValueError("missing/nonfinite m")
            identity = []
            for identity_class in ("OFF/K0", "A_M05/K1", "B_M05/K0"):
                group = _required(
                    same,
                    lambda row, expected=identity_class: row["calibration"].get("identity_null_class") == expected,
                    identity_class,
                )
                if any(
                    row["identity"].get("attempted") is True and row["identity"].get("status") != "COMPLETE"
                    for row in group
                ):
                    raise ValueError("technical identity failure")
                eligible = [
                    row
                    for row in group
                    if row["sync"].get("unique_action") and row["identity"].get("status") == "COMPLETE"
                ]
                if not eligible:
                    raise ValueError("no eligible identity-null row for " + identity_class)
                identity.extend(eligible)
            I_values = [row["identity"]["I"] for row in identity]
            if any(not math.isfinite(value) for value in I_values):
                raise ValueError("nonfinite I")
            thresholds[str(N)] = {
                "tau_M": max(M_values),
                "tau_m": None if N == 181 else max(0.0, max(m_values)),
                "tau_I": max(0.0, max(I_values)),
                "M_rows": len(M_values),
                "m_rows": len(m_values),
                "I_rows": len(I_values),
            }
            positives = _required(same, lambda row: row["calibration"].get("positive") is True, "positive")
            threshold = thresholds[str(N)]
            if any(
                not row["sync"].get("unique_action")
                or row["identity"].get("status") != "COMPLETE"
                or row["identity"].get("exact32") is not True
                or not (row["sync"]["M"] > threshold["tau_M"])
                or (N != 181 and not (row["sync"]["m"] > threshold["tau_m"]))
                or not (row["identity"]["I"] > threshold["tau_I"])
                for row in positives
            ):
                raise ValueError("development positive not strictly separable")
        frozen = {
            "status": "FROZEN",
            "formula_version": RULE_FORMULA_VERSION,
            "thresholds": thresholds,
            "source_roster_sha256": source_roster_sha256,
            "config_sha256": config_sha256,
            "claim_message": CLAIM_MESSAGE,
            "claim_bits": list(CLAIM_BITS),
        }
        frozen["rule_sha256"] = hashlib.sha256(
            json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return frozen
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "NOT_FREEZABLE",
            "formula_version": RULE_FORMULA_VERSION,
            "error": str(exc),
            "thresholds": {},
            "source_roster_sha256": source_roster_sha256,
            "config_sha256": config_sha256,
            "claim_message": CLAIM_MESSAGE,
            "claim_bits": list(CLAIM_BITS),
        }


def decide(sync: dict, identity: dict, frames: int, rules: dict) -> dict:
    base = {
        "frames": frames,
        "rule_sha256": rules.get("rule_sha256") if isinstance(rules, dict) else None,
        "claim_message": CLAIM_MESSAGE,
    }
    try:
        if not isinstance(rules, dict) or rules.get("status") != "FROZEN":
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_RULE_NOT_FREEZABLE")
        if frames not in (181, 177, 89) or str(frames) not in rules.get("thresholds", {}):
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_UNSUPPORTED_PROTOCOL")
        if sync.get("status") != "COMPLETE" or not math.isfinite(sync.get("M")):
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_TECHNICAL_SYNC")
        threshold = rules["thresholds"][str(frames)]
        if not math.isfinite(threshold["tau_M"]) or not math.isfinite(threshold["tau_I"]):
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_RULE_NOT_FREEZABLE")
        if sync["M"] <= threshold["tau_M"]:
            return dict(base, decision="REJECT", reason="REJECT_LOW_SYNC")
        if frames != 181 and (
            not math.isfinite(sync.get("m")) or not math.isfinite(threshold.get("tau_m"))
        ):
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_TECHNICAL_SYNC")
        if not sync.get("unique_action") or (frames != 181 and sync["m"] <= threshold["tau_m"]):
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_SYNC")
        if identity.get("status") != "COMPLETE" or not math.isfinite(identity.get("I")):
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_TECHNICAL_PAYLOAD")
        I = identity["I"]
        if I < 0:
            return dict(base, decision="REJECT", reason="REJECT_IDENTITY")
        if I <= threshold["tau_I"]:
            return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_IDENTITY_WEAK")
        if identity.get("exact32") is True:
            return dict(base, decision="ACCEPT", reason="ACCEPT")
        return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_INTERNAL_INCONSISTENCY")
    except (KeyError, TypeError, ValueError):
        return dict(base, decision="UNCERTAIN", reason="UNCERTAIN_TECHNICAL_EVIDENCE")


def posthoc_false_claim(
    decision: dict,
    role: dict,
    action_map_correct: bool | None,
    *,
    true_action_available: bool = True,
) -> dict:
    required = role.get("required_for_false_claim", True)
    accepted = decision.get("decision") == "ACCEPT"
    expected = role.get("expected_accept") is True
    if accepted and (not expected or action_map_correct is False):
        return {
            "false_claim": True,
            "false_claim_reason": "ACCEPT_WRONG_ACTION" if action_map_correct is False else "ACCEPT_NEGATIVE_CONTROL",
            "required": required,
            "technical_complete": True,
        }
    technical_complete = decision.get("reason") in SCIENTIFIC_REASONS
    if not true_action_available or (accepted and expected and action_map_correct is None):
        technical_complete = False
    return {
        "false_claim": False,
        "false_claim_reason": None,
        "required": required,
        "technical_complete": technical_complete,
    }


def aggregate_false_claim(rows: list[dict]):
    required = [row for row in rows if row["posthoc"]["required"]]
    if any(row["posthoc"]["false_claim"] for row in required):
        return True
    if not required or any(not row["posthoc"]["technical_complete"] for row in required):
        return "UNRESOLVED"
    return False
