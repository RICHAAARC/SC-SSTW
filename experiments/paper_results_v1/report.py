"""Fixed-denominator reporting over already-saved experiment results.

The manifest is the denominator authority. Saved results only fill planned rows;
they never create successful denominator rows or select comparisons.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from experiments.paper_results_v1 import SCHEMA_VERSION


TERMINAL_STATES = ("OBSERVED", "FAILED", "MISSING", "EXCLUDED", "UNSUPPORTED", "CONFLICT")
MODE_MAP = {
    "BASELINE": "RAW",
    "RAW": "RAW",
    "EST_ALIGN": "GLOBAL",
    "GLOBAL_ALIGN": "GLOBAL",
    "PATH_ALIGN": "PATH",
    "TRUTH_PATH": "ORACLE",
}


class ManifestError(ValueError):
    """The planned denominator is malformed and cannot be interpreted safely."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _require(row, fields, where):
    missing = [field for field in fields if field not in row]
    if missing:
        raise ManifestError(f"{where} missing fields: {', '.join(missing)}")


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    _require(manifest, ("study_id", "method", "comparability", "result_bindings", "slots", "pairs", "measurements"), "manifest")
    if not isinstance(manifest["result_bindings"], dict):
        raise ManifestError("result_bindings must be an object")
    for result_id, binding in manifest["result_bindings"].items():
        if not isinstance(binding, dict) or not isinstance(binding.get("expected_top_level", {}), dict):
            raise ManifestError(f"result_bindings[{result_id!r}] must contain an object expected_top_level")
    if any(not isinstance(manifest[field], list) for field in ("slots", "pairs", "measurements")):
        raise ManifestError("slots, pairs, and measurements must be lists")
    _require(manifest["method"], ("name", "claim_scope", "evidence_ceiling"), "method")
    _require(
        manifest["comparability"],
        ("message_length_bits", "redundancy", "codec", "auxiliary_inputs", "matching_rule_status"),
        "comparability",
    )
    if type(manifest["comparability"]["message_length_bits"]) is not int or manifest["comparability"]["message_length_bits"] != 32:
        raise ManifestError("comparability.message_length_bits must be integer 32")
    if manifest["comparability"]["matching_rule_status"] != "DESCRIPTIVE_ONLY_NOT_ADOPTED":
        raise ManifestError("matching_rule_status must remain DESCRIPTIVE_ONLY_NOT_ADOPTED")
    for index, slot in enumerate(manifest["slots"]):
        if not isinstance(slot, dict):
            raise ManifestError(f"slots[{index}] must be an object")
        _require(
            slot,
            (
                "slot_id", "source_id", "arm", "receiver_mode", "result_id", "adapter",
                "locator", "planned_bits", "key_label", "included", "supported",
            ),
            f"slots[{index}]",
        )
        if slot["adapter"] != "conditional_joint_v1":
            raise ManifestError(f"slots[{index}] unsupported adapter {slot['adapter']!r}")
        if slot["receiver_mode"] not in ("RAW", "GLOBAL", "PATH", "ORACLE"):
            raise ManifestError(f"slots[{index}] has invalid receiver_mode")
        if type(slot["planned_bits"]) is not int or slot["planned_bits"] != 32:
            raise ManifestError(f"slots[{index}].planned_bits must be integer 32")
        if slot["key_label"] not in ("K0", "K1"):
            raise ManifestError(f"slots[{index}].key_label must be K0 or K1")
        if type(slot["included"]) is not bool or type(slot["supported"]) is not bool:
            raise ManifestError(f"slots[{index}] included/supported must be booleans")
        if slot["result_id"] not in manifest["result_bindings"]:
            raise ManifestError(f"slots[{index}].result_id has no result_bindings entry")
    for index, pair in enumerate(manifest["pairs"]):
        if not isinstance(pair, dict):
            raise ManifestError(f"pairs[{index}] must be an object")
        _require(pair, ("pair_id", "raw_slot_id", "sync_slot_id", "included"), f"pairs[{index}]")
        if type(pair["included"]) is not bool:
            raise ManifestError(f"pairs[{index}].included must be a boolean")
    for index, measurement in enumerate(manifest["measurements"]):
        if not isinstance(measurement, dict):
            raise ManifestError(f"measurements[{index}] must be an object")
        _require(
            measurement,
            ("measurement_id", "source_id", "arm", "kind", "result_id", "locator", "included", "supported"),
            f"measurements[{index}]",
        )
        if measurement["kind"] not in ("QUALITY", "COST"):
            raise ManifestError(f"measurements[{index}].kind must be QUALITY or COST")
        if not isinstance(measurement["locator"], list) or not measurement["locator"]:
            raise ManifestError(f"measurements[{index}].locator must be a nonempty list")
        if measurement["result_id"] not in manifest["result_bindings"]:
            raise ManifestError(f"measurements[{index}].result_id has no result_bindings entry")
    return manifest


def load_inputs(entries):
    """Load RESULT_ID=PATH entries while retaining duplicate and read failures."""
    loaded = defaultdict(list)
    for result_id, path in entries:
        record = {"result_id": result_id, "path": str(Path(path).resolve())}
        try:
            raw = Path(path).read_bytes()
            data = json.loads(raw, parse_constant=_reject_nonfinite)
            if not isinstance(data, dict):
                raise ValueError("result top level must be an object")
            record.update(
                status="LOADED", sha256=hashlib.sha256(raw).hexdigest(), data=data,
                saved_status=data.get("status") if isinstance(data, dict) else None,
                source_sha=data.get("source_sha") if isinstance(data, dict) else None,
                content_sha256=data.get("content_sha256") if isinstance(data, dict) else None,
                config_sha256=data.get("config_sha256") if isinstance(data, dict) else None,
            )
        except Exception as exc:  # result failures belong in the retained report
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        loaded[result_id].append(record)
    return dict(loaded)


def _base_slot(slot, index):
    return {
        "manifest_index": index,
        "slot_id": slot["slot_id"],
        "source_id": slot["source_id"],
        "arm": slot["arm"],
        "receiver_mode": slot["receiver_mode"],
        "view": slot.get("view"),
        "key_role": slot.get("key_role"),
        "key_label": slot["key_label"],
        "result_id": slot["result_id"],
        "adapter": slot["adapter"],
        "locator": slot["locator"],
        "planned_bits": slot["planned_bits"],
        "state": None,
        "reason": None,
        "blind": {},
        "truth": {},
    }


def _safe_blind(payload):
    """Copy receiver-visible bookkeeping without posthoc truth fields."""
    allowed = (
        "status", "view_id", "input_id", "protocol", "frames", "R", "key_label", "mode",
        "oracle", "physical_read", "alias_of", "operation_cost", "planned_final_bits", "error",
    )
    return {key: payload[key] for key in allowed if key in payload}


def _binding_error(manifest, result_id, result):
    if not isinstance(result, dict):
        return "result top level is not an object"
    binding = manifest["result_bindings"][result_id]
    expected = binding.get("expected_top_level", {})
    if not isinstance(expected, dict):
        return "result binding expected_top_level must be an object"
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        return "result identity mismatch: " + ", ".join(sorted(mismatches))
    return None


def _conditional_slot(slot, result):
    posthoc = result.get("posthoc")
    payloads = result.get("payload_reads")
    if not isinstance(posthoc, dict) or not isinstance(payloads, dict):
        return "CONFLICT", "conditional_joint_v1 result lacks posthoc/payload_reads objects", {}, {}
    locator = slot["locator"]
    if locator not in posthoc and locator not in payloads:
        return "MISSING", "planned locator absent from result", {}, {}
    if locator not in posthoc or locator not in payloads:
        return "CONFLICT", "posthoc and payload_reads disagree on locator", {}, {}
    payload_row = payloads[locator]
    if not isinstance(payload_row, dict):
        return "CONFLICT", "payload_reads row is not an object", {}, {}
    blind = _safe_blind(payload_row)
    truth_row = posthoc[locator]
    if not isinstance(truth_row, dict):
        return "CONFLICT", "posthoc row is not an object", blind, {}
    actual_mode = MODE_MAP.get(payload_row.get("mode"))
    if actual_mode != slot["receiver_mode"]:
        return "CONFLICT", f"manifest receiver_mode {slot['receiver_mode']} != saved mode {actual_mode}", blind, {}
    if slot.get("view") is not None and payload_row.get("view_id") != slot["view"]:
        return "CONFLICT", "manifest view does not match saved view_id", blind, {}
    if payload_row.get("key_label") != slot["key_label"]:
        return "CONFLICT", "manifest key_label does not match saved key_label", blind, {}
    if type(payload_row.get("planned_final_bits")) is not int or payload_row["planned_final_bits"] != 32:
        return "CONFLICT", "saved planned_final_bits must be integer 32", blind, {}
    truth = {
        key: truth_row[key]
        for key in (
            "status", "key_role", "bit_errors", "error_bits", "absolute_path_correct", "phase_correct",
            "family_correct", "b_correct", "k_signed_error", "k_absolute_error", "source_map_correct",
            "source_map_denominator", "geometry_only", "oracle", "conditional_interpretation", "error",
        )
        if key in truth_row
    }
    status = truth_row.get("status")
    if status != "EVALUATED_TRUTH":
        return "FAILED", truth_row.get("error", f"posthoc status {status!r}"), blind, truth
    if slot.get("key_role") is not None and truth_row.get("key_role") != slot["key_role"]:
        return "CONFLICT", "manifest key_role does not match posthoc key_role", blind, truth
    expected_oracle = slot["receiver_mode"] == "ORACLE"
    if payload_row.get("oracle") is not expected_oracle or truth_row.get("oracle") is not expected_oracle:
        return "CONFLICT", "oracle status does not match receiver_mode", blind, truth
    errors = truth_row.get("bit_errors")
    if type(errors) is not int or not 0 <= errors <= slot["planned_bits"]:
        return "CONFLICT", "evaluated row has invalid bit_errors", blind, truth
    if blind.get("status") != "READ":
        return "CONFLICT", "truth evaluated although blind payload row is not READ", blind, truth
    truth["bit_denominator"] = slot["planned_bits"]
    truth["exact_recovery"] = errors == 0
    return "OBSERVED", None, blind, truth


def normalize_slots(manifest, inputs):
    slots = manifest["slots"]
    duplicate_ids = {key for key, count in Counter(row["slot_id"] for row in slots).items() if count > 1}
    locators = Counter((row["result_id"], row["locator"]) for row in slots)
    out = []
    for index, slot in enumerate(slots):
        row = _base_slot(slot, index)
        if not slot["included"]:
            row.update(state="EXCLUDED", reason=slot.get("reason") or "manifest exclusion")
        elif not slot["supported"]:
            row.update(state="UNSUPPORTED", reason=slot.get("reason") or "unsupported planned slot")
        elif slot["slot_id"] in duplicate_ids:
            row.update(state="CONFLICT", reason="duplicate slot_id in manifest")
        elif locators[(slot["result_id"], slot["locator"])] > 1:
            row.update(state="CONFLICT", reason="duplicate result locator in manifest")
        else:
            candidates = inputs.get(slot["result_id"], [])
            if not candidates:
                row.update(state="MISSING", reason="result_id not supplied")
            elif len(candidates) > 1:
                row.update(state="CONFLICT", reason="multiple result files supplied for result_id")
            elif candidates[0]["status"] != "LOADED":
                row.update(state="FAILED", reason=candidates[0]["error"])
            else:
                identity_error = _binding_error(manifest, slot["result_id"], candidates[0]["data"])
                if identity_error:
                    row.update(state="CONFLICT", reason=identity_error)
                else:
                    state, reason, blind, truth = _conditional_slot(slot, candidates[0]["data"])
                    row.update(state=state, reason=reason, blind=blind, truth=truth)
        out.append(row)
    return out


def _lookup(data, locator):
    value = data
    for part in locator:
        if not isinstance(value, dict) or part not in value:
            raise KeyError(part)
        value = value[part]
    return value


def normalize_measurements(manifest, inputs):
    planned = manifest["measurements"]
    duplicate_ids = {key for key, count in Counter(x["measurement_id"] for x in planned).items() if count > 1}
    rows = []
    for index, item in enumerate(planned):
        row = {
            "manifest_index": index,
            "measurement_id": item["measurement_id"],
            "source_id": item["source_id"],
            "arm": item["arm"],
            "kind": item["kind"],
            "metric": item.get("metric"),
            "unit": item.get("unit"),
            "result_id": item["result_id"],
            "locator": item["locator"],
            "state": None,
            "reason": None,
            "value": None,
        }
        candidates = inputs.get(item["result_id"], [])
        if not item["included"]:
            row.update(state="EXCLUDED", reason=item.get("reason") or "manifest exclusion")
        elif not item["supported"]:
            row.update(state="UNSUPPORTED", reason=item.get("reason") or "unsupported measurement")
        elif item["measurement_id"] in duplicate_ids:
            row.update(state="CONFLICT", reason="duplicate measurement_id in manifest")
        elif not candidates:
            row.update(state="MISSING", reason="result_id not supplied")
        elif len(candidates) > 1:
            row.update(state="CONFLICT", reason="multiple result files supplied for result_id")
        elif candidates[0]["status"] != "LOADED":
            row.update(state="FAILED", reason=candidates[0]["error"])
        else:
            identity_error = _binding_error(manifest, item["result_id"], candidates[0]["data"])
            if identity_error:
                row.update(state="CONFLICT", reason=identity_error)
            else:
                try:
                    value = _lookup(candidates[0]["data"], item["locator"])
                    if value is None:
                        row.update(state="MISSING", reason="saved measurement value is null")
                    elif isinstance(value, float) and not math.isfinite(value):
                        row.update(state="CONFLICT", reason="saved measurement value is non-finite")
                    else:
                        row.update(state="OBSERVED", value=value)
                except KeyError:
                    row.update(state="MISSING", reason="planned measurement locator absent from result")
        rows.append(row)
    return rows


def aggregate_slots(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_id"], row["arm"], row["receiver_mode"], row.get("key_role"))].append(row)
    output = []
    for (source_id, arm, mode, key_role), members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        eligible = [row for row in members if row["state"] not in ("EXCLUDED", "UNSUPPORTED")]
        evaluable = [row for row in eligible if row["state"] == "OBSERVED"]
        exact = [row for row in evaluable if row["truth"].get("exact_recovery")]
        physical_reads = {row["blind"].get("physical_read") for row in members if row["blind"].get("physical_read")}
        output.append({
            "source_id": source_id,
            "arm": arm,
            "receiver_mode": mode,
            "key_role": key_role,
            "manifest_slots": len(members),
            "fixed_denominator_slots": len(members),
            "eligible_recovery_slots": len(eligible),
            "unique_physical_key_read_identities": len(physical_reads),
            "state_counts": dict(sorted(Counter(row["state"] for row in members).items())),
            "fixed_bit_denominator": sum(row["planned_bits"] for row in members),
            "eligible_bit_denominator": sum(row["planned_bits"] for row in eligible),
            "evaluable_slots": len(evaluable),
            "evaluable_bit_denominator": sum(row["truth"]["bit_denominator"] for row in evaluable),
            "bit_errors": sum(row["truth"]["bit_errors"] for row in evaluable),
            "exact_32bit_numerator": len(exact),
            "exact_32bit_fixed_denominator": len(members),
            "exact_32bit_eligible_denominator": len(eligible),
            "exact_32bit_evaluable_denominator": len(evaluable),
        })
    return output


def paired_comparisons(manifest, rows):
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["slot_id"]].append(row)
    duplicate_pair_ids = {key for key, count in Counter(pair["pair_id"] for pair in manifest["pairs"]).items() if count > 1}
    output = []
    for pair in manifest["pairs"]:
        pair_id = pair["pair_id"]
        base = {
            "pair_id": pair_id,
            "raw_slot_id": pair["raw_slot_id"],
            "sync_slot_id": pair["sync_slot_id"],
            "state": "NOT_EVALUABLE",
            "reason": None,
        }
        if not pair["included"]:
            base.update(state="EXCLUDED", reason=pair.get("reason") or "manifest pair exclusion")
        elif pair_id in duplicate_pair_ids:
            base["reason"] = "duplicate pair_id in manifest"
        else:
            raw_rows, sync_rows = by_id[pair["raw_slot_id"]], by_id[pair["sync_slot_id"]]
            if len(raw_rows) != 1 or len(sync_rows) != 1:
                base["reason"] = "pair references missing or duplicate slot_id"
            else:
                raw, sync = raw_rows[0], sync_rows[0]
                base.update(raw_state=raw["state"], sync_state=sync["state"])
                if raw["receiver_mode"] != "RAW" or sync["receiver_mode"] not in ("GLOBAL", "PATH"):
                    base["reason"] = "pair must reference RAW then GLOBAL/PATH slots"
                elif any((
                    raw["source_id"] != sync["source_id"], raw["arm"] != sync["arm"],
                    raw.get("key_role") != sync.get("key_role"), raw["key_label"] != sync["key_label"],
                    raw["result_id"] != sync["result_id"], raw.get("view") != sync.get("view"),
                )):
                    base["reason"] = "pair source, arm, key, result, or view mismatch"
                elif raw["state"] == sync["state"] == "OBSERVED":
                    if raw["blind"].get("input_id") != sync["blind"].get("input_id") or raw["blind"].get("key_label") != sync["blind"].get("key_label"):
                        base["reason"] = "pair saved input_id or key_label mismatch"
                    else:
                        raw_errors = raw["truth"]["bit_errors"]
                        sync_errors = sync["truth"]["bit_errors"]
                        delta = sync_errors - raw_errors
                        interpretation = "BER_IMPROVEMENT_OBSERVED" if delta < 0 else "BER_WORSENING_OBSERVED" if delta > 0 else "NO_BER_GAIN_OBSERVED"
                        base.update(
                            state="EVALUABLE", raw_bit_errors=raw_errors, sync_bit_errors=sync_errors,
                            sync_minus_raw_bit_errors=delta, interpretation=interpretation,
                        )
                else:
                    base["reason"] = "RAW or SYNC member is not evaluable"
        output.append(base)
    return output


def _unplanned_observations(manifest, inputs):
    planned = defaultdict(set)
    for slot in manifest["slots"]:
        planned[slot["result_id"]].add(slot["locator"])
    out = []
    for result_id, records in sorted(inputs.items()):
        for record_index, record in enumerate(records):
            if record["status"] != "LOADED":
                continue
            data = record["data"]
            payloads = data.get("payload_reads", {}) if isinstance(data, dict) else {}
            posthoc = data.get("posthoc", {}) if isinstance(data, dict) else {}
            payloads = payloads if isinstance(payloads, dict) else {}
            posthoc = posthoc if isinstance(posthoc, dict) else {}
            for locator in sorted(set(payloads) | set(posthoc)):
                if locator not in planned[result_id]:
                    payload_value, posthoc_value = payloads.get(locator), posthoc.get(locator)
                    out.append({
                        "result_id": result_id,
                        "record_index": record_index,
                        "locator": locator,
                        "state": "UNPLANNED",
                        "payload_exists": locator in payloads,
                        "payload_status": payload_value.get("status") if isinstance(payload_value, dict) else None,
                        "posthoc_exists": locator in posthoc,
                        "posthoc_status": posthoc_value.get("status") if isinstance(posthoc_value, dict) else None,
                    })
    return out


def build_report(manifest, inputs):
    validate_manifest(manifest)
    rows = normalize_slots(manifest, inputs)
    measurements = normalize_measurements(manifest, inputs)
    pairs = paired_comparisons(manifest, rows)
    state_counts = dict(sorted(Counter(row["state"] for row in rows).items()))
    pair_counts = dict(sorted(Counter(row["state"] for row in pairs).items()))
    issues = sum(state_counts.get(state, 0) for state in ("FAILED", "MISSING", "CONFLICT"))
    issues += sum(row["state"] in ("FAILED", "MISSING", "CONFLICT") for row in measurements)
    issues += sum(row["state"] == "NOT_EVALUABLE" for row in pairs)
    independent_sources = {row["source_id"] for row in rows if row["state"] not in ("EXCLUDED", "UNSUPPORTED")}
    physical_reads = {row["blind"].get("physical_read") for row in rows if row["blind"].get("physical_read")}
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "report_status": "COMPLETE_WITH_RETAINED_ISSUES" if issues else "COMPLETE",
        "method": manifest["method"],
        "comparability": manifest["comparability"],
        "arm_catalog": manifest.get("arm_catalog", []),
        "result_bindings": manifest["result_bindings"],
        "manifest_denominator": {
            "slots": len(rows),
            "measurements": len(measurements),
            "pairs": len(pairs),
            "independent_source_labels": len(independent_sources),
            "observed_unique_physical_key_read_identities": len(physical_reads),
            "slot_state_counts": state_counts,
            "pair_state_counts": pair_counts,
        },
        "input_records": [
            {key: value for key, value in record.items() if key != "data"}
            for result_id in sorted(inputs) for record in inputs[result_id]
        ],
        "blind_rows": [
            {key: value for key, value in row.items() if key != "truth"}
            for row in rows
        ],
        "truth_rows": [
            {key: value for key, value in row.items() if key != "blind"}
            for row in rows
        ],
        "conditional_recovery": aggregate_slots(rows),
        "paired_sync_effect": pairs,
        "measurements": measurements,
        "unplanned_observations": _unplanned_observations(manifest, inputs),
        "claim_guard": (
            "Descriptive fixed-denominator report only. Correct-key, wrong-key, and oracle rows remain separate. "
            "Missing or failed slots are not successes. A zero-error RAW/SYNC pair is NO_BER_GAIN_OBSERVED, "
            "not evidence of synchronization benefit."
        ),
    }


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def render_markdown(report):
    denominator = report["manifest_denominator"]
    lines = [
        f"# {report['study_id']}", "",
        f"Report status: `{report['report_status']}`", "",
        f"Method: **{report['method']['name']}**", "",
        report["method"]["claim_scope"], "",
        f"Evidence ceiling: {report['method']['evidence_ceiling']}", "",
        "## Comparability description", "",
        "These fields describe the planned comparison. They do not adopt a matching rule.", "",
        "| Field | Value |", "| --- | --- |",
    ]
    for field in ("message_length_bits", "redundancy", "codec", "auxiliary_inputs", "matching_rule_status"):
        lines.append(f"| {field} | `{_csv_value(report['comparability'].get(field))}` |")
    lines += ["", "## Fixed denominator", "", "| Item | Count |", "| --- | ---: |"]
    for field in ("slots", "measurements", "pairs", "independent_source_labels", "observed_unique_physical_key_read_identities"):
        lines.append(f"| {field} | {denominator[field]} |")
    for state in TERMINAL_STATES:
        lines.append(f"| slots `{state}` | {denominator['slot_state_counts'].get(state, 0)} |")
    lines += ["", "## Result bindings", "", "| Result | Load | Saved status | Source SHA | Content identity | Config identity | Path |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["input_records"]:
        lines.append(
            f"| {row['result_id']} | {row['status']} | {row.get('saved_status') or ''} | "
            f"`{row.get('source_sha') or ''}` | `{row.get('content_sha256') or ''}` | "
            f"`{row.get('config_sha256') or ''}` | `{row['path']}` |"
        )
    if not report["input_records"]:
        lines.append("| *(none supplied)* |  |  |  |  |  |  |")
    lines += [
        "", "Logical slots, unique physical key-read identities, input media, actual calls, and independent source labels are separate counts.",
        "", "## Conditional 32-bit recovery", "",
        "| Source | Arm | Receiver | Key role | Physical key-read IDs | Exact / full fixed | Exact / eligible | Exact / evaluable | Bit errors / evaluable bits | States |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["conditional_recovery"]:
        lines.append(
            f"| {row['source_id']} | {row['arm']} | {row['receiver_mode']} | {row.get('key_role') or ''} | "
            f"{row['unique_physical_key_read_identities']} | {row['exact_32bit_numerator']} / {row['exact_32bit_fixed_denominator']} | "
            f"{row['exact_32bit_numerator']} / {row['exact_32bit_eligible_denominator']} | "
            f"{row['exact_32bit_numerator']} / {row['exact_32bit_evaluable_denominator']} | "
            f"{row['bit_errors']} / {row['evaluable_bit_denominator']} | `{_csv_value(row['state_counts'])}` |"
        )
    lines += [
        "", "## Planned RAW/SYNC pairs", "",
        "| Pair | State | RAW errors | SYNC errors | SYNC - RAW | Interpretation / reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["paired_sync_effect"]:
        lines.append(
            f"| {row['pair_id']} | {row['state']} | {row.get('raw_bit_errors', '')} | "
            f"{row.get('sync_bit_errors', '')} | {row.get('sync_minus_raw_bit_errors', '')} | "
            f"{row.get('interpretation') or row.get('reason')} |"
        )
    if not report["paired_sync_effect"]:
        lines.append("| *(none planned)* |  |  |  |  |  |")
    lines += [
        "", "A RAW=0 and SYNC=0 pair is reported as `NO_BER_GAIN_OBSERVED`. It is not counted as a synchronization benefit.",
        "", "## Quality and cost measurements", "",
        "| Measurement | Source | Arm | Kind | State | Value | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["measurements"]:
        lines.append(
            f"| {row['measurement_id']} | {row['source_id']} | {row['arm']} | {row['kind']} | "
            f"{row['state']} | `{_csv_value(row['value'])}` | {row.get('reason') or ''} |"
        )
    if not report["measurements"]:
        lines.append("| *(none planned)* |  |  |  |  |  |  |")
    lines += ["", "## Retained unplanned observations", ""]
    if report["unplanned_observations"]:
        lines += ["| Result | Locator | Payload exists/status | Posthoc exists/status |", "| --- | --- | --- | --- |"]
        for row in report["unplanned_observations"]:
            lines.append(
                f"| {row['result_id']} | `{row['locator']}` | {row['payload_exists']} / {row.get('payload_status') or ''} | "
                f"{row['posthoc_exists']} / {row.get('posthoc_status') or ''} |"
            )
    else:
        lines.append("None.")
    lines += ["", report["claim_guard"], ""]
    return "\n".join(lines)


def write_report(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    blind_fields = (
        "manifest_index", "slot_id", "source_id", "arm", "receiver_mode", "view", "key_role", "key_label",
        "result_id", "locator", "planned_bits", "state", "reason", "blind",
    )
    truth_fields = (
        "manifest_index", "slot_id", "source_id", "arm", "receiver_mode", "view", "key_role", "key_label",
        "result_id", "locator", "planned_bits", "state", "reason", "truth",
    )
    _write_csv(output / "blind_rows.csv", report["blind_rows"], blind_fields)
    _write_csv(output / "truth_rows.csv", report["truth_rows"], truth_fields)
    pair_fields = (
        "pair_id", "state", "raw_slot_id", "sync_slot_id", "raw_state", "sync_state",
        "raw_bit_errors", "sync_bit_errors", "sync_minus_raw_bit_errors", "interpretation", "reason",
    )
    _write_csv(output / "paired_sync_effect.csv", report["paired_sync_effect"], pair_fields)
    measurement_fields = (
        "manifest_index", "measurement_id", "source_id", "arm", "kind", "metric", "unit", "result_id",
        "locator", "state", "value", "reason",
    )
    _write_csv(output / "measurements.csv", report["measurements"], measurement_fields)
