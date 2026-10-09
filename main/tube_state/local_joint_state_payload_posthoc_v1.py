"""Adopted known-grid descriptive reducer for saved local-joint observations.

This module never selects a phase/path, decodes a message, reads media, or
defines a scientific PASS.  Truth is supplied only after the complete raw
observation directory has been saved and validated.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier


LAYERS = ("float_rgb", "rgb8", "mp4")
ARMS = ("OFF", "JOINT")
KEY_LABELS = ("CORRECT", "WRONG")


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _chip_reason(row: Mapping[str, Any], chip: Mapping[str, Any], component: str) -> str | None:
    if chip.get("status") != "SCORED":
        return f"chip_status={chip.get('status')}"
    if not _finite_number(chip.get("q")):
        return "q_missing_or_nonfinite"
    if type(chip.get("supported_samples")) is not int or chip["supported_samples"] <= 0:
        return "zero_supported_samples"
    if chip.get("failed_samples") != 0:
        return f"failed_samples={chip.get('failed_samples')}"
    if chip.get("available_frames") != chip.get("expected_frames"):
        return "incomplete_frame_support"
    if chip.get("available_samples") != chip.get("expected_samples"):
        return "incomplete_sample_support"
    return None


def _catalog(rows: list[dict[str, Any]], protocol: carrier.CarrierProtocol) -> dict[str, dict[str, Any]]:
    expected = carrier.phase_window_catalog(protocol)
    if len(rows) != len(expected):
        raise ValueError(f"raw directory requires exactly {len(expected)} rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("spec"), dict):
            raise TypeError("serialized RawWindowObservation row required")
        spec = row["spec"]
        observation_id = spec.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in by_id:
            raise ValueError("raw observation ids must be unique strings")
        if row.get("truth_used") is not False:
            raise ValueError("raw observation directory must remain truth-free")
        by_id[observation_id] = row
    if set(by_id) != {item.observation_id for item in expected}:
        raise ValueError("raw observation directory does not match the fixed catalog")
    for item in expected:
        spec = by_id[item.observation_id]["spec"]
        for field, value in (
            ("phase", item.phase), ("slot", item.slot), ("roi_index", item.roi_index),
            ("requested_frames", list(item.requested_frames)),
            ("received_frames", list(item.received_frames)), ("availability", item.availability),
        ):
            if spec.get(field) != value:
                raise ValueError(f"raw observation spec mismatch for {item.observation_id}:{field}")
    return by_id


def _evidence(row: Mapping[str, Any], chip: Mapping[str, Any], reason: str | None) -> dict[str, Any]:
    return dict(
        observation_id=row["spec"]["observation_id"],
        phase=row["spec"]["phase"], slot=row["spec"]["slot"], roi_index=row["spec"]["roi_index"],
        status=chip.get("status"), q=chip.get("q"), energy_plus=chip.get("energy_plus"),
        energy_minus=chip.get("energy_minus"), expected_samples=chip.get("expected_samples"),
        available_samples=chip.get("available_samples"), supported_samples=chip.get("supported_samples"),
        expected_frames=chip.get("expected_frames"), available_frames=chip.get("available_frames"),
        zero_energy_samples=chip.get("zero_energy_samples"), failed_samples=chip.get("failed_samples"),
        valid_for_metric=(reason is None), missing_reason=reason,
    )


def evaluate_observations(
    rows: list[dict[str, Any]], *, key: str, message: bytes,
    protocol: carrier.CarrierProtocol = carrier.PUBLIC,
) -> dict[str, Any]:
    """Evaluate only the adopted phase-1/slot-0..21 known-grid diagnostic."""
    by_id = _catalog(rows, protocol)
    if protocol.segment_count != 22 or len(protocol.rois) != 4:
        raise ValueError("adopted posthoc requires 22 segments and four ROIs")
    state_truth = carrier.state_matrix(key, 22)
    fragments = carrier.message_fragments(message)

    state_items = []
    for slot in range(22):
        for roi in range(4):
            row = by_id[f"phase1:slot{slot}:roi{roi}"]
            chips = row.get("state_chips")
            pairs = carrier.coefficient_pairs(key, roi)
            if not isinstance(chips, list) or len(chips) != 8:
                raise ValueError("each known-grid row requires eight state chips")
            for local_bit, chip in enumerate(chips):
                global_bit = 8 * roi + local_bit
                if (chip.get("component") != "state" or chip.get("component_index") != global_bit
                        or chip.get("pair_index") != local_bit
                        or chip.get("plus_coord") != list(pairs[local_bit][0])
                        or chip.get("minus_coord") != list(pairs[local_bit][1])):
                    raise ValueError("state chip index mapping mismatch")
                reason = _chip_reason(row, chip, "state")
                state_items.append(dict(slot=slot, chip_index=global_bit,
                                        truth=state_truth[slot][global_bit],
                                        evidence=_evidence(row, chip, reason)))
    invalid_state = [item for item in state_items if not item["evidence"]["valid_for_metric"]]
    correlations = []
    if invalid_state:
        for offset in range(22):
            correlations.append(dict(offset=offset, status="MISSING", value=None,
                                     missing_items=len(invalid_state), reason="required_state_evidence_missing"))
        state_summary = dict(status="MISSING", correlations=correlations, c0=None,
                             max_other=None, c0_minus_max_other=None, global_max_ties=[],
                             invalid_items=invalid_state, evidence=state_items)
    else:
        for offset in range(22):
            value = sum(
                state_truth[(item["slot"] + offset) % 22][item["chip_index"]]
                * float(item["evidence"]["q"])
                for item in state_items
            ) / 704.0
            correlations.append(dict(offset=offset, status="SCORED", value=value,
                                     missing_items=0, reason=None))
        values = [item["value"] for item in correlations]
        maximum = max(values)
        maximum_other = max(values[1:])
        state_summary = dict(status="SCORED", correlations=correlations, c0=values[0],
                             max_other=maximum_other, c0_minus_max_other=values[0] - maximum_other,
                             global_max_ties=[index for index, value in enumerate(values) if value == maximum],
                             invalid_items=[], evidence=state_items)

    payload_metrics = []
    expected_counts = (24, 24, 20, 20)
    for fragment in range(4):
        for bit in range(8):
            sign = 2 * fragments[fragment][bit] - 1
            items = []
            for slot in range(fragment, 22, 4):
                for roi in range(4):
                    row = by_id[f"phase1:slot{slot}:roi{roi}"]
                    chips = row.get("payload_chips")
                    if not isinstance(chips, list) or len(chips) != 8:
                        raise ValueError("each known-grid row requires eight payload chips")
                    chip = chips[bit]
                    if (chip.get("component") != "payload" or chip.get("component_index") != bit
                            or chip.get("pair_index") != 8 + bit
                            or chip.get("plus_coord") != list(carrier.coefficient_pairs(key, roi)[8 + bit][0])
                            or chip.get("minus_coord") != list(carrier.coefficient_pairs(key, roi)[8 + bit][1])):
                        raise ValueError("payload chip index mapping mismatch")
                    reason = _chip_reason(row, chip, "payload")
                    items.append(_evidence(row, chip, reason))
            if len(items) != expected_counts[fragment]:
                raise RuntimeError("adopted payload evidence count mismatch")
            invalid = [item for item in items if not item["valid_for_metric"]]
            payload_metrics.append(dict(
                fragment=fragment, bit=bit, truth_bit=fragments[fragment][bit], sign=sign,
                expected_evidence_count=expected_counts[fragment], evidence=items,
                status="MISSING" if invalid else "SCORED",
                signed_mean=None if invalid else sum(sign * float(item["q"]) for item in items) / len(items),
                missing_items=len(invalid), reason="required_payload_evidence_missing" if invalid else None,
            ))
    payload_status = "SCORED" if all(item["status"] == "SCORED" for item in payload_metrics) else "MISSING"
    return dict(
        schema="local-joint-known-grid-descriptive-v1",
        scope=dict(phase=1, slots=list(range(22)), state_denominator=704,
                   payload_evidence_counts=list(expected_counts), blind_path=False,
                   message_decoding=False, truth_loaded_after_raw_directory=True),
        state=state_summary,
        payload=dict(status=payload_status, metrics=payload_metrics),
    )


def descriptive_condition(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    state, payload = evaluation["state"], evaluation["payload"]
    if state["status"] != "SCORED" or payload["status"] != "SCORED":
        invalid = [item["evidence"] for item in state.get("invalid_items", [])]
        invalid.extend(evidence for metric in payload["metrics"]
                       for evidence in metric["evidence"] if not evidence["valid_for_metric"])
        def is_support_gap(item: Mapping[str, Any]) -> bool:
            # A clean, finite zero-energy item is a carrier-support gap.  A
            # failed/non-finite observation remains an engineering failure
            # even when it consequently has no supported samples.
            return (item.get("supported_samples") == 0
                    and item.get("failed_samples") == 0
                    and item.get("available_frames") == item.get("expected_frames")
                    and item.get("available_samples") == item.get("expected_samples")
                    and _finite_number(item.get("energy_plus")) and item.get("energy_plus") == 0
                    and _finite_number(item.get("energy_minus")) and item.get("energy_minus") == 0)

        support_gaps = [item for item in invalid if is_support_gap(item)]
        engineering = [item for item in invalid if not is_support_gap(item)]
        classification = "ENGINEERING_FAILURE" if engineering else "CONSTRUCTION_SUPPORT_GAP"
        return dict(status="MISSING", met=None, reason="required_metric_missing",
                    classification=classification, construction_support_gap_items=len(support_gaps),
                    engineering_failure_items=len(engineering), scientific_pass=False)
    state_met = state["c0_minus_max_other"] > 0
    payload_met = all(item["signed_mean"] > 0 for item in payload["metrics"])
    met = bool(state_met and payload_met)
    return dict(status="SCORED", met=met,
                state_gap_positive=state_met, all_32_payload_signed_means_positive=payload_met,
                classification="DESCRIPTIVE_CONDITION_MET" if met else "VALID_FINITE_NEGATIVE",
                scientific_pass=False)


def compare_arms(off: Mapping[str, Any], joint: Mapping[str, Any]) -> dict[str, Any]:
    def difference(left: Any, right: Any) -> float | None:
        return float(left) - float(right) if _finite_number(left) and _finite_number(right) else None
    state_deltas = [
        dict(offset=index, value=difference(joint["state"]["correlations"][index]["value"],
                                            off["state"]["correlations"][index]["value"]))
        for index in range(22)
    ]
    payload_deltas = [
        dict(fragment=item["fragment"], bit=item["bit"],
             value=difference(item["signed_mean"], off["payload"]["metrics"][index]["signed_mean"]))
        for index, item in enumerate(joint["payload"]["metrics"])
    ]
    return dict(
        state_correlations=state_deltas,
        state_gap=difference(joint["state"]["c0_minus_max_other"], off["state"]["c0_minus_max_other"]),
        payload_signed_means=payload_deltas,
        missing_values=sum(item["value"] is None for item in (*state_deltas, *payload_deltas)),
    )
