"""Read saved tables against an explicit plan. No receiver or model is imported.

The adapter accepts the already-scored segmented row format. It does not score
messages, select paths, reconstruct votes, or prove execution-time isolation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import SCHEMA

LAYOUT = {"segments": 4, "bits_per_segment": 8, "net_bits": 32}
STATES = {"READ", "FAILED", "MISSING", "UNRESOLVED", "UNSUPPORTED", "CONFLICT", "UNREAD"}
INFORMATION = {"BLIND", "ORACLE", "CALLER_SUPPLIED"}
METRIC_FIELDS = {
    "structurally_present_bits": "eligible_planned_bits",
    "structural_missing_bits": "structural_missing_bits",
    "read_present_bits": "scored_present_bits",
    "present_unread_bits": "missing_present_bits",
    "error_bits": "wrong_present_bits",
    "missing_segment_output_bits": "false_attribution_bits",
}
OUTPUT_FIELDS = ("decoded_bits", "decoded_segment_hex", "blind_claimed_segments",
                 "selector_status", "score_gap")
POSTHOC_FIELDS = ("true_missing_segments", "nominal_true_cells", "path_exact",
                  "interval_exact", "absolute_path_error", "wrong_path_positions",
                  "falsely_claimed_missing_segments", "matches_other_expected_byte",
                  "exact_present_segments", "bit_error_rate",
                  "complete_present_denominator_bit_error_rate", "bit_support",
                  "expected_message", "expected_bits", "expected_segment_hex")


class ReportFormatError(ValueError):
    """Invalid saved data or plan, not a scientific outcome."""


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportFormatError("non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReportFormatError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReportFormatError(f"invalid JSON: {exc}") from exc
    _finite(value)
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ReportFormatError(f"{name} must be a nonempty string")
    return value


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ReportFormatError(f"protocol schema must be {SCHEMA}")
    _finite(protocol)
    if protocol.get("payload_layout") != LAYOUT:
        raise ReportFormatError("payload_layout must declare 4 x 8 = 32")
    slots = protocol.get("slots")
    bindings = protocol.get("result_bindings")
    if not isinstance(slots, list) or not isinstance(bindings, list):
        raise ReportFormatError("slots and result_bindings must be lists")
    if slots and protocol.get("plan_status") != "ADOPTED":
        raise ReportFormatError("a nonempty denominator requires an explicitly ADOPTED plan")
    binding_ids = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ReportFormatError("each result binding must be an object")
        identity = _text(binding.get("binding_id"), "binding_id")
        if identity in binding_ids:
            raise ReportFormatError(f"duplicate binding_id: {identity}")
        binding_ids.add(identity)
        _text(binding.get("rows_path"), "rows_path")
    slot_ids, targets = set(), set()
    for slot in slots:
        if not isinstance(slot, dict):
            raise ReportFormatError("each slot must be an object")
        for field in ("slot_id", "saved_slot_id", "binding_id", "source_id", "arm",
                      "view", "information_condition", "evidence_role", "message_id"):
            _text(slot.get(field), field)
        if slot["slot_id"] in slot_ids:
            raise ReportFormatError(f"duplicate planned slot: {slot['slot_id']}")
        slot_ids.add(slot["slot_id"])
        target = (slot["binding_id"], slot["saved_slot_id"])
        if target in targets:
            raise ReportFormatError("one saved row cannot fill two planned slots")
        targets.add(target)
        if slot["binding_id"] not in binding_ids:
            raise ReportFormatError("slot references unknown binding")
        if slot["information_condition"] not in INFORMATION:
            raise ReportFormatError("information_condition must be BLIND, ORACLE or CALLER_SUPPLIED")
    return protocol


def _count(value, name, maximum=32):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= maximum:
        raise ReportFormatError(f"{name} must be an integer in 0..{maximum} or null")
    return value


def _validate_known_counts(metrics):
    """Check whether known counts can satisfy the fixed equations; fill nothing."""
    present, missing, read, unread, wrong, false = (metrics[key] for key in (
        "structurally_present_bits", "structural_missing_bits", "read_present_bits",
        "present_unread_bits", "error_bits", "missing_segment_output_bits"))
    lower, upper = 0, 32
    if present is not None:
        lower, upper = present, present
    if missing is not None:
        lower, upper = max(lower, 32 - missing), min(upper, 32 - missing)
    if read is not None:
        lower = max(lower, read)
    if unread is not None:
        lower = max(lower, unread)
    if read is not None and unread is not None:
        lower, upper = max(lower, read + unread), min(upper, read + unread)
    if wrong is not None:
        if read is not None and wrong > read:
            raise ReportFormatError("error bits exceed known read present bits")
        lower = max(lower, wrong + (unread if unread is not None else 0))
    if false is not None:
        upper = min(upper, 32 - false)
    if lower > upper:
        raise ReportFormatError(
            "known counts cannot satisfy planned=present+missing, present=read+unread "
            "and error/missing-output support bounds; null fields do not remove known contradictions"
        )


def _validated_support(records, metrics, *, decoded=None, missing_segments=None):
    """Validate saved support against other saved facts, without message scoring."""
    if not isinstance(records, list) or len(records) != 32:
        raise ReportFormatError("bit_support requires all 32 planned positions")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReportFormatError("bit_support record must be an object")
        bit = _count(record.get("bit"), "bit", 31)
        if bit is None or bit in result or type(record.get("present")) is not bool or type(record.get("read")) is not bool:
            raise ReportFormatError("invalid/duplicate support position or mask")
        correct = record.get("correct")
        if record["read"] and record["present"]:
            if type(correct) is not bool:
                raise ReportFormatError("read present bit requires saved bool correctness")
        elif correct is not None:
            raise ReportFormatError("unread/missing bit cannot have correctness")
        result[bit] = record
    counts = {
        "structurally_present_bits": sum(r['present'] for r in result.values()),
        "structural_missing_bits": sum(not r['present'] for r in result.values()),
        "read_present_bits": sum(r['present'] and r['read'] for r in result.values()),
        "present_unread_bits": sum(r['present'] and not r['read'] for r in result.values()),
        "error_bits": sum(r['present'] and r['read'] and not r['correct'] for r in result.values()),
        "missing_segment_output_bits": sum(not r['present'] and r['read'] for r in result.values()),
    }
    for key, derived in counts.items():
        known = _count(metrics.get(key), key)
        if known is not None and known != derived:
            raise ReportFormatError(f"bit_support contradicts saved {key}")
    if decoded is not None:
        if not isinstance(decoded, list) or len(decoded) != 32:
            raise ReportFormatError("decoded support must contain 32 positions")
        if any(result[i]['read'] != (decoded[i] is not None) for i in result):
            raise ReportFormatError("bit_support read mask contradicts saved decoded support")
    if missing_segments is not None:
        if not isinstance(missing_segments, list) or any(type(g) is not int or not 0 <= g < 4 for g in missing_segments):
            raise ReportFormatError("invalid saved true_missing_segments mask")
        if any(result[i]['present'] != (i // 8 not in missing_segments) for i in result):
            raise ReportFormatError("bit_support present mask contradicts saved true_missing_segments")
    return result


def _empty_row(slot, status, reason, *, report_state="INPUT_UNAVAILABLE", originals=None):
    return {
        **slot, "status": status, "saved_status": None, "saved_reason": None,
        "report_state": report_state, "report_reason": reason,
        "planned_bits": 32, "planned_segments": 4,
        "metrics": {**{key: None for key in METRIC_FIELDS}, "observed_ber": None, "complete_present_ber": None},
        "blind_output": None, "oracle_output": None, "caller_map_output": None,
        "posthoc": {}, "raw_saved_rows": originals or [],
        "physical_read_unit": None, "read_alias_of": None, "alias_record_present": False,
        "general_source_cell_false_attribution": "UNKNOWN",
    }


def adapt_segmented_row(slot, saved):
    """Copy saved decisions and saved posthoc counts into separate columns."""
    row = _empty_row(slot, "UNSUPPORTED", None, report_state="INVALID_SAVED_FORMAT", originals=[saved])
    if not isinstance(saved, dict):
        row["report_reason"] = "saved row must be an object; not a scientific failure"
        return row
    row["saved_status"], row["saved_reason"] = saved.get("status"), saved.get("reason")
    try:
        _finite(saved)
        if not isinstance(saved.get("status"), str) or saved["status"] not in STATES:
            raise ReportFormatError("unsupported saved status")
        row["status"] = saved["status"]
        if saved.get("planned_bits") != 32 or saved.get("planned_segments") != 4:
            raise ReportFormatError("saved denominator must be 4 segments / 32 bits")
        for field, expected in (("case_id", slot["source_id"]), ("receiver", slot["arm"]),
                                ("view", slot["view"]), ("slot_id", slot["saved_slot_id"])):
            if saved.get(field) != expected:
                raise ReportFormatError(f"saved {field} disagrees with explicit planned identity")
        condition = slot["information_condition"]
        if type(saved.get("oracle")) is not bool or saved["oracle"] != (condition == "ORACLE"):
            raise ReportFormatError("saved oracle flag disagrees with planned information condition")
        if condition == "BLIND" and (saved.get("mapping_origin") in ("ORACLE", "CALLER_SUPPLIED")
                                      or saved.get("conditional_on_given_mapping") is True):
            raise ReportFormatError("conditional/oracle saved mapping cannot enter the blind output column")
        metrics = {key: _count(saved.get(source), source) for key, source in METRIC_FIELDS.items()}
        _validate_known_counts(metrics)
        present, missing, read, unread, wrong = (metrics[k] for k in (
            "structurally_present_bits", "structural_missing_bits", "read_present_bits",
            "present_unread_bits", "error_bits"))
        if present is not None and missing is not None and present + missing != 32:
            raise ReportFormatError("planned != structurally present + structural missing")
        if all(x is not None for x in (present, read, unread)) and present != read + unread:
            raise ReportFormatError("structurally present != read + present unread")
        if wrong is not None and read is not None and wrong > read:
            raise ReportFormatError("error bits exceed read present bits")
        false = metrics["missing_segment_output_bits"]
        if false is not None and missing is not None and false > missing:
            raise ReportFormatError("missing-segment output exceeds structural missing bits")
        decoded = saved.get("decoded_bits")
        if decoded is not None and (not isinstance(decoded, list) or len(decoded) != 32 or any(
                bit is not None and (type(bit) is not int or bit not in (0, 1)) for bit in decoded)):
            raise ReportFormatError("decoded_bits must contain exactly 32 integer bits/nulls")
        missing_segments = saved.get("true_missing_segments")
        if missing_segments is not None:
            if not isinstance(missing_segments, list) or any(type(g) is not int or not 0 <= g < 4 for g in missing_segments) or len(set(missing_segments)) != len(missing_segments):
                raise ReportFormatError("true_missing_segments must contain distinct indices 0..3")
            if missing is not None and 8 * len(missing_segments) != missing:
                raise ReportFormatError("structural mask disagrees with saved missing count")
            if present is not None and 32 - 8 * len(missing_segments) != present:
                raise ReportFormatError("structural mask disagrees with saved present count")
            if decoded is not None:
                observed_present = sum(bit is not None and i // 8 not in missing_segments for i, bit in enumerate(decoded))
                observed_missing = sum(bit is not None and i // 8 in missing_segments for i, bit in enumerate(decoded))
                if read is not None and read != observed_present:
                    raise ReportFormatError("saved read count disagrees with explicit decoded support")
                if false is not None and false != observed_missing:
                    raise ReportFormatError("saved missing-segment output count disagrees with decoded support")
        for field in ("path_exact", "interval_exact"):
            if saved.get(field) is not None and type(saved[field]) is not bool:
                raise ReportFormatError(f"{field} must be bool or null")
        for field in ("absolute_path_error", "wrong_path_positions"):
            value = saved.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ReportFormatError(f"{field} must be nonnegative integer or null")
        for field in ("physical_read_unit", "read_alias_of"):
            if saved.get(field) is not None and not isinstance(saved[field], str):
                raise ReportFormatError(f"{field} must be a string or null")
        if saved.get("bit_support") is not None:
            _validated_support(saved["bit_support"], metrics, decoded=decoded,
                               missing_segments=missing_segments)
        output = {key: saved[key] for key in OUTPUT_FIELDS if key in saved}
        output["mapping_origin"] = saved.get("mapping_origin", condition)
        if condition != "BLIND":
            output.update(conditional_on_given_mapping=True, blind_sync_performed=False)
        row[{"BLIND": "blind_output", "ORACLE": "oracle_output", "CALLER_SUPPLIED": "caller_map_output"}[condition]] = output
        row.update(metrics=metrics, report_state="ADAPTED", report_reason=None,
                   posthoc={key: saved[key] for key in POSTHOC_FIELDS if key in saved},
                   physical_read_unit=saved.get("physical_read_unit"),
                   read_alias_of=saved.get("read_alias_of"), alias_record_present="read_alias_of" in saved,
                   physical_id=saved.get("physical_id"))
        metrics["observed_ber"] = wrong / read if wrong is not None and read else None
        metrics["complete_present_ber"] = wrong / present if wrong is not None and present and unread == 0 and read == present else None
    except ReportFormatError as exc:
        row["report_reason"] = str(exc) + "; format/identity issue, not a scientific failure"
    return row


def summarize(rows):
    statuses, report_states = {}, {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        report_states[row["report_state"]] = report_states.get(row["report_state"], 0) + 1
    totals, known, unknown = {}, {}, {}
    for field in METRIC_FIELDS:
        values = [row["metrics"].get(field) for row in rows]
        known[field] = sum(value for value in values if value is not None)
        unknown[field] = sum(value is None for value in values)
        totals[field] = known[field] if rows and unknown[field] == 0 else None
    read, error = totals["read_present_bits"], totals["error_bits"]
    present, unread = totals["structurally_present_bits"], totals["present_unread_bits"]
    path_values = [row["posthoc"].get("absolute_path_error") for row in rows]
    return {
        "planned_rows": len(rows), "planned_segments": 4 * len(rows), "planned_bits": 32 * len(rows),
        "status_counts": statuses, "report_state_counts": report_states, **totals,
        "known_subtotals": known, "unknown_metric_rows": unknown,
        "observed_ber": error / read if error is not None and read else None,
        "complete_present_ber": error / present if error is not None and present and unread == 0 and read == present else None,
        "path": {"planned_rows": len(rows), "resolved_rows": sum(x is not None for x in path_values),
                 "exact_rows": sum(row["posthoc"].get("path_exact") is True for row in rows),
                 "resolved_absolute_error": sum(x for x in path_values if x is not None),
                 "fixed_denominator_absolute_error": sum(path_values) if rows and all(x is not None for x in path_values) else None},
        "general_source_cell_false_attribution": "UNKNOWN",
    }


def compare_saved_support(left, right):
    """Pure table comparison of explicit per-bit correctness/support records.

    No decoded-message scoring is done. Historical rows without these records
    remain UNKNOWN, even when two aggregate counts happen to be equal.
    """
    for side, row in (("left", left), ("right", right)):
        if row.get("report_state") != "ADAPTED":
            return {"status": "INVALID_SAVED_FORMAT" if row.get("report_state") == "INVALID_SAVED_FORMAT" else "UNAVAILABLE",
                    "reason": f"{side} row is not adapted: {row.get('report_reason')}"}
    for key in ("source_id", "view", "message_id", "information_condition", "evidence_role"):
        if left.get(key) != right.get(key):
            return {"status": "INCOMPARABLE", "reason": f"different {key}"}
    a, b = left.get("posthoc", {}).get("bit_support"), right.get("posthoc", {}).get("bit_support")
    if a is None or b is None:
        return {"status": "UNKNOWN", "reason": "explicit per-bit support/correctness records not saved"}
    def validated(row, records):
        result = _validated_support(records, row["metrics"],
                                    missing_segments=row["posthoc"].get("true_missing_segments"))
        for field in ("blind_output", "oracle_output", "caller_map_output"):
            output = row.get(field)
            if output is not None:
                _validated_support(records, row["metrics"], decoded=output.get("decoded_bits"))
        return result
    try:
        a, b = validated(left, a), validated(right, b)
    except ReportFormatError as exc:
        return {"status": "INVALID_SAVED_FORMAT", "reason": str(exc)}
    if any(a[i]["present"] != b[i]["present"] for i in a):
        return {"status": "INCOMPARABLE", "reason": "different structural support masks"}
    ar = {i for i in a if a[i]["present"] and a[i]["read"]}
    br = {i for i in b if b[i]["present"] and b[i]["read"]}
    ac = {i for i in ar if a[i]["correct"]}
    bc = {i for i in br if b[i]["correct"]}
    common = ar & br
    return {"status": "SAVED_SUPPORT_REDUCTION", "common_read_bits": len(common),
            "left_common_ber": sum(not a[i]["correct"] for i in common) / len(common) if common else None,
            "right_common_ber": sum(not b[i]["correct"] for i in common) / len(common) if common else None,
            "correct_coverage_retained": len(ac & bc), "correct_coverage_lost": len(ac - bc),
            "correct_coverage_added": len(bc - ac), "newly_read_bits": len(br - ar)}


def build_report(protocol, *, base_dir="."):
    validate_protocol(protocol)
    bindings, inputs, unplanned = {}, [], []
    targets = {(s["binding_id"], s["saved_slot_id"]) for s in protocol["slots"]}
    for binding in protocol["result_bindings"]:
        name = binding["binding_id"]
        record = {"binding_id": name, "rows_path": binding["rows_path"],
                  "expected_top_level_record_only": binding.get("expected_top_level", {}),
                  "saved_summary": None, "summary_read_issue": None}
        indexed = {}
        try:
            content = load_json(Path(base_dir) / binding["rows_path"])
            if not isinstance(content, list):
                raise ReportFormatError("saved rows file must be a JSON list")
            for ordinal, row in enumerate(content):
                key = row.get("slot_id") if isinstance(row, dict) else None
                if not isinstance(key, str):
                    unplanned.append({"binding_id": name, "ordinal": ordinal, "reason": "no usable saved slot identity", "raw_saved_row": row})
                    continue
                indexed.setdefault(key, []).append(row)
                if (name, key) not in targets:
                    unplanned.append({"binding_id": name, "ordinal": ordinal, "reason": "not in adopted plan; no denominator expansion", "raw_saved_row": row})
            record.update(read_status="READ", row_count=len(content), issue=None)
        except (OSError, ReportFormatError) as exc:
            record.update(read_status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", issue=str(exc))
        if binding.get("summary_path"):
            try:
                record["saved_summary"] = load_json(Path(base_dir) / binding["summary_path"])
            except (OSError, ReportFormatError) as exc:
                record["summary_read_issue"] = str(exc)
        inputs.append(record)
        bindings[name] = (indexed, record)
    rows = []
    for slot in protocol["slots"]:
        indexed, record = bindings[slot["binding_id"]]
        candidates = indexed.get(slot["saved_slot_id"], [])
        if record["read_status"] != "READ":
            row = _empty_row(slot, record["read_status"], record["issue"])
        elif not candidates:
            row = _empty_row(slot, "MISSING", "planned slot absent in saved rows")
        elif len(candidates) != 1:
            row = _empty_row(slot, "CONFLICT", "duplicate saved slot; no answer-based selection", originals=candidates)
        else:
            row = adapt_segmented_row(slot, candidates[0])
        rows.append(row)
    groups = {}
    for row in rows:
        key = (row["binding_id"], row["arm"], row["information_condition"], row["evidence_role"],
               "CLEAN" if row["view"] == "CLEAN" else "ATTACK")
        groups.setdefault(key, []).append(row)
    return {
        "schema": SCHEMA, "protocol": protocol, "inputs": inputs, "rows": rows,
        "summary": summarize(rows),
        "groups": [{"binding_id": key[0], "arm": key[1], "information_condition": key[2],
                    "evidence_role": key[3], "view_scope": key[4], **summarize(group)} for key, group in groups.items()],
        "unplanned_rows": unplanned,
        "counts": {"planned_source_labels": len({r['source_id'] for r in rows}),
                   "planned_message_labels": len({r['message_id'] for r in rows}),
                   "logical_read_rows": sum(r['status'] == 'READ' and r['report_state'] == 'ADAPTED' for r in rows),
                   "saved_alias_rows": sum(bool(r['read_alias_of']) for r in rows) if any(r['alias_record_present'] for r in rows) else None,
                   "alias_record_unknown_rows": sum(not r['alias_record_present'] for r in rows),
                   "physical_read_unit_labels": len({(r['binding_id'], r['physical_read_unit']) for r in rows if r['physical_read_unit'] is not None}),
                   "physical_read_unit_unknown_rows": sum(r['physical_read_unit'] is None for r in rows),
                   "independence": "NOT_INFERRED_FROM_LABEL_COUNTS; see saved audit records"},
        "claim_ceiling": "Saved evidence collation only. No new physical run, independent confirmation, blind-isolation proof, fair baseline or scientific PASS.",
        "general_source_cell_false_attribution": "UNKNOWN",
    }


def render_markdown(report):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ') if value is not None else "UNKNOWN"
    summary = report["summary"]
    lines = ["# Saved segmented results", "", report["claim_ceiling"], "",
             f"Fixed plan: {summary['planned_rows']} rows / {summary['planned_segments']} segments / {summary['planned_bits']} bits.",
             "", "Structural absence, unread present bits and errors have separate denominators. General source/cell attribution remains UNKNOWN.", "",
             "| Binding | Arm | Information | Evidence role | View scope | Rows | Present | Missing | Read | Unread | Errors | Observed BER | Complete present BER |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields = ("binding_id", "arm", "information_condition", "evidence_role", "view_scope", "planned_rows",
              "structurally_present_bits", "structural_missing_bits", "read_present_bits", "present_unread_bits", "error_bits", "observed_ber", "complete_present_ber")
    lines.extend("| " + " | ".join(cell(g.get(k)) for k in fields) + " |" for g in report["groups"])
    lines += ["", "| Slot | Status | Saved reason | Report state / issue |", "|---|---|---|---|"]
    lines.extend("| " + " | ".join(cell(v) for v in (r['slot_id'], r['status'], r['saved_reason'], r['report_state'] + ': ' + (r['report_reason'] or ''))) + " |" for r in report['rows'])
    lines += ["", f"Unplanned saved rows retained separately: {len(report['unplanned_rows'])}.",
              "Full saved rows, blind/oracle/caller-map columns, posthoc fields, saved summaries and provenance are retained in report.json."]
    return "\n".join(lines) + "\n"


def write_report(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ("report.json", "report.csv", "report.md")]
    if any(path.exists() for path in targets):
        raise FileExistsError("report outputs already exist; choose a fresh output directory")
    targets[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ("slot_id", "source_id", "arm", "view", "information_condition", "evidence_role", "status",
              "saved_status", "saved_reason", "report_state", "report_reason", "planned_bits", *METRIC_FIELDS,
              "observed_ber", "complete_present_ber", "blind_output", "oracle_output", "caller_map_output", "posthoc")
    with targets[1].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            flat = {**row, **row["metrics"]}
            writer.writerow({key: json.dumps(flat[key], ensure_ascii=False, allow_nan=False) if isinstance(flat.get(key), (dict, list)) else flat.get(key) for key in fields})
    targets[2].write_text(render_markdown(report), encoding="utf-8")
    return targets
