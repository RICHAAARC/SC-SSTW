"""CDC32: payload-free planning and conditional, non-voting aggregation.

Pure Python. No model, codec, FFT, Counter, IO, truth or path selection.
The caller persists/re-reads plans before calling read_local/read_comparisons.
"""
from copy import deepcopy
import json
import math

H = tuple((p, s) for p in range(4) for s in range(1 if p == 0 else 2, 36))
REGIONS = ("R0", "R1")
REGION_SLICES = {"R0": [8, 24, 8, 24], "R1": [8, 24, 40, 56]}
TECHNICAL = {"TECHNICAL_UNRESOLVED", "UNRESOLVED_MAPPING", "UNRESOLVED_MAPPING_CONFLICT",
             "UNRESOLVED_EVIDENCE_OWNERSHIP", "TECHNICAL_CACHE", "NOT_ENTERED", "UNSUPPORTED",
             "MISSING_VIEW", "CALIBRATION_UNRESOLVED"}
FAMILY_AVAILABLE = {None, "READY", "DETECTED_PROVISIONAL", "LOCAL_GATE_REJECT", "REJECT_SYNC", "SCORES_COMPLETE"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _unique(rows):
    """Canonical data ordering, not a digest or identity check."""
    return [json.loads(k) for k in sorted({json.dumps(r, sort_keys=True) for r in rows})]


def evidence_id(view_id, local_row, region):
    """Actual view identity is supplied by runtime; hypotheses/key are absent."""
    return f"{view_id}/row{local_row}/{region}"


def _candidate_id(row):
    return f"p{row['phase']}:s{row['source_start']}"


def _code(value):
    """Only structured local status/reason codes, never arbitrary error text."""
    if value is None:
        return None
    if isinstance(value, str) and value and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:" for c in value):
        return value
    return "UNSTRUCTURED_LOCAL_STATUS"


def _causes(value):
    if not isinstance(value, (list, tuple)):
        return [] if value is None else ["INVALID_CAUSE_SET"]
    return sorted({_code(x) for x in value if x is not None})


def local_failure_metadata(prepared):
    """Narrow B2 failure allowlist, usable before/after opaque bundle rebasing."""
    result = {k: _code(prepared[k]) for k in ("upstream_status", "upstream_score_status", "score_status", "threshold_status") if k in prepared}
    result["family_status"] = _code(prepared.get("status"))
    result["cause_set"] = _causes(prepared.get("cause_set"))
    for name in ("missing_members", "unexpected_members"):
        records = prepared.get(name, [])
        output = []
        for item in records if isinstance(records, (list, tuple)) else []:
            if not isinstance(item, dict):
                output.append(dict(reason="INVALID_LOCAL_FAILURE_RECORD"))
                continue
            row = {k: item[k] for k in ("phase", "source_start", "index") if type(item.get(k)) is int}
            if "cause_set" in item:
                row["cause_set"] = _causes(item["cause_set"])
            for k in ("reason", "score_status", "upstream_score_status"):
                if k in item:
                    row[k] = _code(item[k])
            output.append(row)
        result[name] = output
    if type(prepared.get("submitted_member_count")) is int:
        result["submitted_member_count"] = prepared["submitted_member_count"]
    return result


def _family(prepared):
    members = prepared.get("members", prepared.get("candidates", []))
    valid = (isinstance(members, list) and
             all(isinstance(r, dict) for r in members) and
             [(r.get("phase"), r.get("source_start")) for r in members] == list(H) and
             all(r.get("score_status") == "RETURNED" and _finite(r.get("S")) and
                 0 <= r["S"] <= 1 and not _causes(r.get("cause_set")) and
                 r.get("upstream_score_status", "RETURNED") == "RETURNED" for r in members))
    tau = prepared.get("tau")
    threshold = prepared.get("threshold", {})
    if tau is None and isinstance(threshold, dict):
        tau = threshold.get("tau")
    valid = (valid and _finite(tau) and 0 <= tau <= 1 and prepared.get("score_complete") is not False and
             prepared.get("completeness", "COMPLETE") == "COMPLETE" and
             prepared.get("threshold_status", "FROZEN") == "FROZEN" and
             prepared.get("status") in FAMILY_AVAILABLE and not _causes(prepared.get("cause_set")) and
             prepared.get("score_status") in (None, "SCORES_COMPLETE") and
             prepared.get("upstream_status") in (None, "SCORES_COMPLETE"))
    c = [r for r in members if r["S"] >= tau] if valid else []
    maximum = max(r["S"] for r in members) if valid else None
    detected = bool(valid and maximum > tau)
    # Full C is determined exclusively by the complete saved local family.
    # A supplied pruned C, D or membership is a semantic error, never a rescue.
    if valid and "C" in prepared:
        supplied_rows = prepared["C"]
        valid = isinstance(supplied_rows, list) and all(isinstance(r, dict) for r in supplied_rows)
        supplied = [(r.get("phase"), r.get("source_start")) for r in supplied_rows] if valid else None
        valid = valid and supplied == [(r["phase"], r["source_start"]) for r in c]
    if valid and type(prepared.get("D")) is bool:
        valid = prepared["D"] == detected
    if valid:
        valid = all("in_C" not in r or r["in_C"] == (r["S"] >= tau) for r in members)
    status = "READY" if valid and detected else "REJECT_SYNC" if valid else "TECHNICAL_UNRESOLVED"
    if prepared.get("status") == "NOT_ENTERED" or prepared.get("upstream_status") == "NOT_ENTERED":
        status = "NOT_ENTERED"
    elif not valid and prepared.get("status") in ("CALIBRATION_UNRESOLVED", "UNSUPPORTED"):
        status = prepared["status"]
    return members, c if valid else [], maximum, tau, detected if valid and status == "READY" else False, status


def _source_frames(h, w, received):
    """The H137 local-contiguous hypothesis, never a whole-input translation."""
    if "local_received_to_source_map" in h:
        mapping = h["local_received_to_source_map"]
        if isinstance(mapping, dict):
            values = [mapping.get(str(x), mapping.get(x)) for x in received]
        elif isinstance(mapping, list) and len(mapping) == 45:
            values = [mapping[x - w] for x in received]
        else:
            return None
        return values if all(type(x) is int for x in values) else None
    origin = 4 * (h["source_start"] - 1) - h["phase"]
    supplied = h.get("query_source_origin_hypothesis", origin)
    if supplied != origin:
        return None
    return [origin + x - w for x in received]


def _geometry(h, w):
    cells = {}
    support = []
    unknown = False
    for row in range(1, 11):
        start = w + h["phase"] + 4 * row - 3
        received = list(range(start, start + 4))
        source = _source_frames(h, w, received)
        t = (source[0] + 3) // 4 if source is not None else None
        full = (t in range(1, 45) and source == list(range(4*t-3, 4*t+1)))
        kind = "FULL_UNDER_H" if full else "UNRESOLVED_MAPPING" if source is None else "MIXED_PARTIAL_OR_GUARD_UNDER_H"
        unknown |= source is None
        support.append(dict(local_row=row, nominal_received_cell=received,
                            source_frames_under_h=source, canonical_cell=t if full else None, status=kind))
        if full:
            cells[t] = row
    return cells, support, unknown


def _presence(h, w, observed):
    source = _source_frames(h, w, list(range(w, w+45)))
    source_set = set(source) if source is not None else None
    result = []
    for t in range(1, 45):
        n = len(source_set & set(range(4*t-3, 4*t+1))) if source_set is not None else None
        state = ("UNRESOLVED_MAPPING" if n is None else "STRUCTURALLY_ABSENT_UNDER_H" if n == 0 else
                 "PARTIAL_UNDER_H" if n < 4 else "FULL_OBSERVED_UNDER_H" if t in observed else
                 "PRESENT_UNOBSERVED_UNDER_H")
        result.append(dict(canonical_cell=t, present_nominal_frames=n, status=state))
    return result


def _view_table(views):
    values = list(views.values()) if isinstance(views, dict) else list(views)
    table = {}
    for view in values:
        p = view.get("phase")
        if p not in range(4) or p in table:
            raise ValueError("one planned view per phase 0..3 required; aliases reuse its view_id")
        table[p] = view
    return table


def _reference(view, p, row, region, h, t, w):
    start = w + p + 4*row - 3
    received = list(range(start, start + 4))
    if view is None:
        return dict(candidate_id=_candidate_id(h), phase=p, local_row=row,
                    observed_region_id=region, carrier_region_id=region, canonical_cell=t,
                    evidence_id=None, view_id=None, status="MISSING_VIEW",
                    nominal_received_cell=received, encoder_input_support=[], encoder_context=None)
    view_id = view.get("view_id")
    unit = view.get("units", {}).get(str(row), {}).get(region, {})
    eid = unit.get("evidence_id", evidence_id(view_id, row, region))
    view_status = view.get("status")
    status = unit.get("status", view_status) if view_status == "RETURNED" else view_status
    status = status if isinstance(status, str) else "TECHNICAL_UNRESOLVED"
    observed = unit.get("observed_region_id", region)
    carrier = unit.get("carrier_region_id", region)
    support = view.get("encoder_input_support", list(range(w+p, w+p+41)))
    if isinstance(support, dict):
        support = support.get("received_frames", [])
    if (not isinstance(view_id, str) or not isinstance(eid, str) or
            observed != region or carrier != region or
            support != list(range(w+p, w+p+41)) or unit.get("diagnostic_only", False)):
        status = "TECHNICAL_UNRESOLVED"
    return dict(candidate_id=_candidate_id(h), phase=p, local_row=row,
                observed_region_id=observed, carrier_region_id=carrier, canonical_cell=t,
                evidence_id=eid, view_id=view_id, status=status,
                view_status=_code(view_status), unit_status=_code(unit.get("status")),
                cause_set=_causes(view.get("cause_set")) + _causes(unit.get("cause_set")),
                nominal_received_cell=received, encoder_input_support=deepcopy(support),
                encoder_context=deepcopy(view.get("encoder_context")),
                region_slice=deepcopy(REGION_SLICES[region]))


def plan_local(prepared_local, views, required_regions=REGIONS):
    """Plan four groups from complete local scores. Never accepts a bit cache.

    prepared_local: query_id, input_id, key_id, received_query_start, members
    (the ordered 137 phase/source_start/S/score_status rows), tau, optional C/D.
    views: four metadata dicts, one per phase; optional units[row][region].
    Explicit maps use received coordinates or a 45-element query-local list.
    """
    regions = tuple(required_regions)
    if regions not in (REGIONS, ("R0",), ("R1",)):
        raise ValueError("production R0/R1 or one independent diagnostic region required")
    w = prepared_local.get("received_query_start", 0)
    if type(w) is not int or w < 0:
        raise ValueError("nonnegative received query start required")
    members, c, maximum, tau, detected, status = _family(prepared_local)
    failures = local_failure_metadata(prepared_local)
    failures["member_failures"] = []
    for member in members if isinstance(members, list) else []:
        if isinstance(member, dict) and (member.get("score_status") != "RETURNED" or
                                        _causes(member.get("cause_set")) or
                                        member.get("upstream_score_status", "RETURNED") != "RETURNED"):
            failures["member_failures"].append(dict(
                **{k: member[k] for k in ("phase", "source_start") if type(member.get(k)) is int},
                score_status=_code(member.get("score_status")),
                upstream_score_status=_code(member.get("upstream_score_status")),
                cause_set=_causes(member.get("cause_set"))))
    table = _view_table(views)
    geometries = [_geometry(h, w) for h in c]
    presence = [_presence(h, w, geometry[0]) for h, geometry in zip(c, geometries)]
    mapping_unknown = any(item[2] for item in geometries)
    groups = []
    for g in range(4):
        common = set(range(11*g+1, 11*g+12))
        for cells, _, _ in geometries:
            common.intersection_update(cells)
        common = sorted(common) if c else []
        selected = min(common) if common and status == "READY" and not mapping_unknown else None
        reason = status if status != "READY" else "UNRESOLVED_MAPPING" if mapping_unknown else "READY" if selected is not None else "NO_COMMON_FULL_CELL"
        refs = []
        if selected is not None:
            for h, (cells, _, _) in zip(c, geometries):
                row = cells[selected]
                for region in regions:
                    ref = _reference(table.get(h["phase"]), h["phase"], row, region, h, selected, w)
                    ref["input_id"] = prepared_local.get("input_id")
                    refs.append(ref)
        if any(ref["status"] == "UNRESOLVED_EVIDENCE_OWNERSHIP" for ref in refs):
            reason = "UNRESOLVED_EVIDENCE_OWNERSHIP"
        causes = [] if reason == "READY" else [reason]
        if status != "READY":
            causes.extend(failures["cause_set"])
            for name in ("missing_members", "unexpected_members"):
                for record in failures[name]:
                    causes.extend(record.get("cause_set", []))
                    if record.get("reason") is not None:
                        causes.append(record["reason"])
            for member in failures["member_failures"]:
                causes.extend(member["cause_set"])
                if member["upstream_score_status"] is not None:
                    causes.append("UPSTREAM:" + member["upstream_score_status"])
            if failures.get("family_status") not in FAMILY_AVAILABLE:
                causes.append("FAMILY:" + failures["family_status"])
            for field, available in (("score_status", "SCORES_COMPLETE"), ("upstream_status", "SCORES_COMPLETE"),
                                     ("threshold_status", "FROZEN")):
                if failures.get(field) not in (None, available):
                    causes.append(field.upper() + ":" + failures[field])
        if reason == "NO_COMMON_FULL_CELL":
            for entries in presence:
                states = {entry["status"] for entry in entries[11*g:11*g+11]}
                if "FULL_OBSERVED_UNDER_H" in states:
                    causes.append("NO_COMMON_CELL_IDENTITY")
                elif "PRESENT_UNOBSERVED_UNDER_H" in states:
                    causes.append("PRESENT_UNOBSERVED_UNDER_H")
                elif "PARTIAL_UNDER_H" in states:
                    causes.append("PARTIAL_ONLY_NO_FULL_CELL_UNDER_H")
                else:
                    causes.append("STRUCTURALLY_ABSENT_UNDER_H")
        groups.append(dict(query_id=prepared_local.get("query_id"), group=g,
                           selected_canonical_cell=selected, common_cells=common,
                           required_replica_ids=list(regions), references=_unique(refs), status=reason,
                           cause_set=sorted(set(causes)), failure_details=deepcopy(failures)))
    # Copy only semantic fields. Arbitrary metadata/truth/payload never enters a plan.
    safe_rows = members if isinstance(members, list) and len(members) == 137 and all(isinstance(r, dict) for r in members) else [{}]*137
    safe_members = [dict(phase=p, source_start=s,
                         S=safe_rows[i].get("S") if _finite(safe_rows[i].get("S")) else None,
                         score_status=safe_rows[i].get("score_status", "MISSING"),
                         upstream_score_status=_code(safe_rows[i].get("upstream_score_status")),
                         cause_set=_causes(safe_rows[i].get("cause_set")),
                         in_C=any(h["phase"] == p and h["source_start"] == s for h in c) if status != "TECHNICAL_UNRESOLVED" else None)
                    for i, (p, s) in enumerate(H)]
    result = dict(query_id=prepared_local.get("query_id"), input_id=prepared_local.get("input_id"),
                key_id=prepared_local.get("key_id"), received_query_start=w, status=status, D=detected,
                maximum=maximum, tau=tau, domain_count=137, members=safe_members,
                failure_details=failures, missing_members=deepcopy(failures["missing_members"]),
                unexpected_members=deepcopy(failures["unexpected_members"]),
                full_local_C=[dict(candidate_id=_candidate_id(h), phase=h["phase"], source_start=h["source_start"],
                                   query_source_origin_hypothesis=4*(h["source_start"]-1)-h["phase"])
                              for h in c], groups=groups,
                support_under_h=[dict(candidate_id=_candidate_id(h), rows=geometry[1], source_cell_presence=present)
                                 for h, geometry, present in zip(c, geometries, presence)],
                mapping_scope="LOCAL_FULL_C", references_frozen_before_bits=True, message_used=False)
    return reconcile_plans([result])[0]


def reconcile_plans(plans, *, ownership_only=False):
    """Mark conflicting assignments before reading; do not prune any candidate.

    Within one query only the same h is simultaneous. Across queries local C
    is factorized, so assignments coexist; no external path is used to prune it.
    """
    result = deepcopy(list(plans))
    by_evidence = {}
    for pi, plan in enumerate(result):
        for gi, group in enumerate(plan["groups"]):
            for ref in group["references"]:
                if ref["evidence_id"] is not None:
                    by_evidence.setdefault(ref["evidence_id"], []).append((pi, gi, ref))
    bad = {}
    for entries in by_evidence.values():
        for i, (pi, gi, a) in enumerate(entries):
            for pj, gj, b in entries[i+1:]:
                same_query = result[pi]["query_id"] == result[pj]["query_id"]
                simultaneous = not same_query or a["candidate_id"] == b["candidate_id"]
                owner_conflict = any(a.get(k) != b.get(k) for k in ("view_id", "local_row", "observed_region_id"))
                conflict = (a["canonical_cell"] != b["canonical_cell"] or
                            a["observed_region_id"] != b["observed_region_id"] or
                            a["carrier_region_id"] != b["carrier_region_id"])
                reasons = (["UNRESOLVED_EVIDENCE_OWNERSHIP"] if owner_conflict else [])
                if not ownership_only and simultaneous and conflict:
                    reasons.append("UNRESOLVED_MAPPING_CONFLICT")
                if owner_conflict:
                    for ref in (a, b):
                        ref["status"] = "UNRESOLVED_EVIDENCE_OWNERSHIP"
                        ref["cause_set"] = sorted(set(ref.get("cause_set", [])) | {"UNRESOLVED_EVIDENCE_OWNERSHIP"})
                for key in ((pi, gi), (pj, gj)):
                    if reasons:
                        bad.setdefault(key, set()).update(reasons)
    for (pi, gi), reasons in bad.items():
        group = result[pi]["groups"][gi]
        group["status"] = "UNRESOLVED_EVIDENCE_OWNERSHIP" if "UNRESOLVED_EVIDENCE_OWNERSHIP" in reasons else "UNRESOLVED_MAPPING_CONFLICT"
        group["cause_set"] = sorted(set(group["cause_set"]) | reasons)
    return result


def _cache_value(cache, address):
    e, key, bit = address
    try:
        value = cache(e, key, bit) if callable(cache) else cache[e][key][str(bit)]
    except Exception as error:
        return None, f"TECHNICAL_CACHE:{type(error).__name__}"
    if not isinstance(value, dict):
        return None, "TECHNICAL_CACHE:FORMAT"
    if value.get("diagnostic_only", False):
        return None, "TECHNICAL_CACHE:DIAGNOSTIC_ONLY"
    v = value.get("value")
    if (type(v) is not int or v not in (0, 1) or type(value.get("count")) is not int or value["count"] != 8 or
            value.get("status") not in ("RETURNED", "READ", "AVAILABLE", "CONDITIONAL_READ")):
        return None, "TECHNICAL_CACHE"
    return v, None


def read_local(plan, cache, *, memo=None):
    """Consume a previously persisted plan; memo shares actual cache addresses."""
    memo = {} if memo is None else memo
    slots = []
    used = set()
    attempts = 0
    for group in plan["groups"]:
        for bit in range(8):
            causes = set(group["cause_set"])
            refs = _unique(group["references"])
            missing = []
            values = []
            if group["status"] == "READY":
                for ref in refs:
                    if ref["status"] != "RETURNED" or ref["evidence_id"] is None:
                        missing.append(ref)
                        causes.add("TECHNICAL_CACHE")
                        causes.update(ref.get("cause_set", []))
                        causes.add("OBSERVATION:" + str(_code(ref["status"])))
                        continue
                    address = (ref["evidence_id"], plan["key_id"], bit)
                    used.add(ref["evidence_id"])
                    if address not in memo:
                        memo[address] = _cache_value(cache, address)
                        attempts += 1
                    value, error = memo[address]
                    if error:
                        missing.append(ref)
                        causes.update(("TECHNICAL_CACHE", error))
                    else:
                        values.append((ref, value))
            unique_values = {v for _, v in values}
            conflict = unique_values == {0, 1}
            technical = bool(missing) or bool(causes & TECHNICAL)
            if conflict:
                value, status = None, "LOCAL_CONFLICT"
                causes.add("BIT_CONFLICT")
            elif technical:
                value, status = None, "LOCAL_TECHNICAL_UNREAD"
            elif unique_values:
                value, status = next(iter(unique_values)), "LOCAL_CONDITIONAL_READ"
            else:
                value, status = None, group["status"]
            slots.append(dict(slot=8*group["group"]+bit, group=group["group"], bit=bit, value=value, status=status,
                              cause_set=sorted(causes), selected_refs=refs,
                              failure_details=deepcopy(group.get("failure_details", {})),
                              conflict_refs=_unique([r for r, _ in values]) if conflict else [], missing_refs=_unique(missing),
                              technical_failure=technical,
                              conditional_candidate_ids=sorted({r["candidate_id"] for r in refs}),
                              selected_canonical_cell=group["selected_canonical_cell"]))
    not_entered = 32 if plan["status"] == "NOT_ENTERED" else 0
    failed = 0 if not_entered else sum(s["status"] == "LOCAL_TECHNICAL_UNREAD" for s in slots)
    return dict(query_id=plan["query_id"], input_id=plan["input_id"], key_id=plan["key_id"],
                D=plan["D"], status=plan["status"], slots=slots, used_evidence_ids=sorted(used),
                cache_accesses=attempts, planned_slots=32,
                counts=dict(planned=32, returned=32-failed-not_entered, failed=failed, not_entered=not_entered),
                support_under_h=deepcopy(plan["support_under_h"]),
                mapping_scope="LOCAL_FULL_C")


def _dependencies(refs):
    units = {}
    for ref in refs:
        if ref["evidence_id"] is not None:
            units[ref["evidence_id"]] = ref
    result = []
    ids = sorted(units)
    for i, left in enumerate(ids):
        for right in ids[i+1:]:
            a, b = units[left], units[right]
            if a["input_id"] == b["input_id"] and set(a["encoder_input_support"]) & set(b["encoder_input_support"]):
                result.append(dict(evidence_ids=[left, right], kind="ENCODER_INPUT_OVERLAP",
                                   independent=False))
    return result


def aggregate(local_decisions, source_relation="SINGLE_INPUT_CONDITIONAL", *, path_context=None,
              propagate_conflicts=True):
    """Conditional union, never source authentication. One value per unique slot."""
    decisions = deepcopy(list(local_decisions))
    # Normal orchestration reconciles the plans before persisting/reading them.
    # Also retain this semantic failure if a caller aggregates separate reads:
    # no already-read value may make a contradictory e->t assignment acceptable.
    geometry_plans = [dict(query_id=d["query_id"], groups=[dict(
        references=d["slots"][8*g]["selected_refs"], status="READY", cause_set=[])
        for g in range(4)]) for d in decisions]
    for decision, checked in zip(decisions, reconcile_plans(geometry_plans)):
        for g, group in enumerate(checked["groups"]):
            if group["status"] in ("UNRESOLVED_MAPPING_CONFLICT", "UNRESOLVED_EVIDENCE_OWNERSHIP"):
                for slot in decision["slots"][8*g:8*g+8]:
                    slot.update(value=None, status="LOCAL_TECHNICAL_UNREAD", technical_failure=True,
                                cause_set=sorted(set(slot["cause_set"] + group["cause_set"])))
    inputs = {d["input_id"] for d in decisions}
    keys = {d["key_id"] for d in decisions}
    unsupported = (len(keys) > 1 or source_relation not in ("SINGLE_INPUT_CONDITIONAL", "SEPARATE", "JOIN_CONDITIONAL") or
                   (source_relation == "JOIN_CONDITIONAL" and len(inputs) > 2) or
                   (source_relation != "JOIN_CONDITIONAL" and len(inputs) > 1))
    rows = []
    for slot in range(32):
        parts = [d["slots"][slot] for d in decisions]
        if any(p["slot"] != slot for p in parts):
            raise ValueError("ordered fixed32 slot records required")
        values = {p["value"] for p in parts if p["value"] is not None}
        known_conflict = any(p["status"] == "LOCAL_CONFLICT" for p in parts)
        technical = any(p["technical_failure"] for p in parts)
        causes = sorted({c for p in parts for c in p["cause_set"]})
        if unsupported:
            value, status = None, "UNSUPPORTED_ARITY_OR_SEMANTICS"
            causes = sorted(set(causes + [status]))
        elif values == {0, 1} or (propagate_conflicts and known_conflict):
            value, status = None, "CONFLICT_ERASURE"
        elif values:
            value = next(iter(values))
            status = "PARTIAL_EVIDENCE_READ" if technical else "CONDITIONAL_READ"
        else:
            value, status = None, "TECHNICAL_UNREAD" if technical else "ERASURE_OR_REJECT"
        rows.append(dict(slot=slot, value=value, status=status, cause_set=causes,
                         selected_refs=_unique([r for p in parts for r in p["selected_refs"]]),
                         supporting_refs=_unique([r for p in parts if value is not None and p["value"] == value and
                                                  p["status"] == "LOCAL_CONDITIONAL_READ" and not p["technical_failure"]
                                                  for r in p["selected_refs"]]),
                         failure_details=_unique([dict(query_id=d["query_id"], input_id=d["input_id"],
                                                       details=d["slots"][slot].get("failure_details", {})) for d in decisions]),
                         conflict_refs=_unique([r for p in parts for r in p["conflict_refs"]]),
                         missing_refs=_unique([r for p in parts for r in p["missing_refs"]]),
                         conditional_candidate_ids=sorted({r for p in parts for r in p["conditional_candidate_ids"]})))
    refs = _unique([r for row in rows for r in row["selected_refs"]])
    unique_decisions = {(d["input_id"], d["query_id"]): d for d in decisions}
    canonical = sorted({r["canonical_cell"] for row in rows for r in row["supporting_refs"] if r["canonical_cell"] is not None})
    planned_canonical = sorted({r["canonical_cell"] for r in refs if r["canonical_cell"] is not None})
    not_entered = 32 if decisions and all(d["status"] == "NOT_ENTERED" for d in decisions) else 0
    failed = 0 if not_entered else sum(r["status"] == "TECHNICAL_UNREAD" for r in rows)
    buckets = []
    for decision in decisions:
        for part in decision["slots"]:
            row = rows[part["slot"]]
            supports_value = (row["value"] is not None and part["value"] == row["value"] and
                              part["status"] == "LOCAL_CONDITIONAL_READ" and not part["technical_failure"])
            for t in sorted({r["canonical_cell"] for r in part["selected_refs"] if r["canonical_cell"] is not None}):
                local_refs = [r for r in part["selected_refs"] if r["canonical_cell"] == t]
                buckets.append(dict(canonical_cell=t, slot=part["slot"], query_id=decision["query_id"],
                    input_id=decision["input_id"], conditional=True, vote_weight=None,
                    status=part["status"], value=part["value"], aggregate_status=row["status"],
                    supports_aggregate_value=supports_value, cause_set=part["cause_set"],
                    evidence_ids=sorted({r["evidence_id"] for r in local_refs if r["evidence_id"] is not None}),
                    conditional_candidate_ids=sorted({r["candidate_id"] for r in local_refs}),
                    missing_refs=_unique([r for r in part["missing_refs"] if r["canonical_cell"] == t])))
    buckets = _unique(buckets)
    return dict(slots=rows, status="UNSUPPORTED" if unsupported else "RETURNED", source_relation=source_relation,
                required_assumptions=["single source under declared input model", "common key/carrier/clock/group semantics"],
                source_authenticated=False, path_context=deepcopy(path_context), mapping_scope="LOCAL_FULL_C",
                used_evidence_ids=sorted({e for d in decisions for e in d["used_evidence_ids"]}),
                dependency_refs=_dependencies(refs), per_query_slots=[deepcopy(unique_decisions[k]) for k in sorted(unique_decisions)],
                canonical_cells_under_h=canonical, canonical_coverage_certified=False,
                planned_canonical_cells_under_h=planned_canonical,
                source_position_buckets=buckets,
                recovered_slot_count=sum(r["value"] is not None for r in rows), planned_slots=32,
                unique_query_count=len(unique_decisions), logical_query_references=len(decisions),
                detection_status="ANY_LOCAL_DETECTED" if any(d["D"] for d in decisions) else
                "TECHNICAL_UNRESOLVED" if any(d["status"] in TECHNICAL for d in decisions) else "REJECT_ALL",
                counts=dict(planned=32, returned=32-failed-not_entered, failed=failed,
                            technical_unread=sum(r["status"] == "TECHNICAL_UNREAD" for r in rows),
                            conflict=sum(r["status"] == "CONFLICT_ERASURE" for r in rows),
                            not_entered=not_entered))


def union_reference(local_decisions, source_relation="SINGLE_INPUT_CONDITIONAL", *, path_context=None):
    return aggregate(local_decisions, source_relation, path_context=path_context, propagate_conflicts=False)


def plan_comparisons(plans_by_window):
    """No payload access: freeze E/M variants, aliases and reversed refs."""
    if set(plans_by_window) != {"L", "M", "R"}:
        raise ValueError("fixed L/M/R plans including failures required")
    m = reconcile_plans([plans_by_window[k] for k in ("L", "M", "R")])
    e = reconcile_plans([plans_by_window[k] for k in ("L", "R")])
    alias = deepcopy(m)
    for group in alias[1]["groups"]:
        group["references"] *= 10
    reverse = deepcopy(list(reversed(m)))
    for plan in reverse:
        for group in plan["groups"]:
            group["references"].reverse()
    return dict(CDC_E=e, CDC_M=m, UNION_M=deepcopy(m), ALIAS_M_X10=alias, REVERSE_ORDER=reverse)


def comparison_signature(result):
    """Exclude raw alias/access counts; include semantic values/reasons/support."""
    return {k: result[k] for k in ("slots", "used_evidence_ids", "canonical_cells_under_h", "planned_canonical_cells_under_h",
                                   "recovered_slot_count", "unique_query_count", "source_position_buckets",
                                   "dependency_refs", "counts")}


def read_comparisons(plan_pack, cache, *, path_context=None):
    memo = {}
    outputs = {}
    for name, plans in plan_pack.items():
        reads = [read_local(plan, cache, memo=memo) for plan in plans]
        fn = union_reference if name == "UNION_M" else aggregate
        outputs[name] = fn(reads, path_context=(path_context or {}).get("E2" if name == "CDC_E" else "M3"))
    outputs["invariance"] = {name: comparison_signature(outputs[name]) == comparison_signature(outputs["CDC_M"])
                             for name in ("ALIAS_M_X10", "REVERSE_ORDER")}
    outputs["unique_cache_accesses"] = len(memo)
    return outputs
