"""Two-fragment CDC32 isolation and durable plan-before-cache consumption.

This adapter only reads already computed per-unit bits. It never receives or
reuses query-level payload decisions, parent paths, Q scoring or physical calls.
RawFragment is a private runtime envelope; only its allowlisted, rebased plan
is passed to the pure method and written to the blind output directory.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

from main.tube_state import b_cdc32_aggregation_v1 as method


@dataclass
class RawFragment:
    prepared_local: dict
    views: list
    cache: object


def _save_read(path, value):
    """Ordinary durable JSON; no digest, identity comparison or version gate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _lookup(cache, eid, key, bit):
    return cache(eid, key, bit) if callable(cache) else cache[eid][key][str(bit)]


def _failure(reason, status="NOT_ENTERED"):
    outputs = {}
    for name, relation in (("SEP_A", "SEPARATE"), ("SEP_B", "SEPARATE"), ("JOIN_CONDITIONAL", "JOIN_CONDITIONAL")):
        outputs[name] = dict(slots=[dict(slot=i, value=None, status=status, cause_set=[reason],
                                         conflict_refs=[], missing_refs=[], selected_refs=[]) for i in range(32)],
                             source_relation=relation, source_authenticated=False,
                             required_assumptions=["same source only under JOIN_CONDITIONAL", "common key/carrier/clock/group semantics"],
                             path_context="NOT_APPLICABLE_FRAGMENT_PAIR", used_evidence_ids=[], dependency_refs=[],
                             planned_slots=32, recovered_slot_count=0,
                             counts=dict(planned=32, returned=0, failed=32 if status != "NOT_ENTERED" else 0,
                                         not_entered=32 if status == "NOT_ENTERED" else 0))
    return dict(status=status, cause_set=[reason], outputs=outputs, planned_local_slots=64,
                planned_aggregate_slots=96, unique_cache_accesses=0, physical_calls=0)


def prepare_bundle(fragments):
    """Return a blind plan and an unexposed lazy cache accessor. No bit reads.

    Input fragments contain only one 45-frame query each. Arbitrary parent
    metadata is not copied, and mappings are rebuilt from that local family.
    No full-parent path or previous payload decision is an accepted input.
    """
    if len(fragments) != 2 or any(not isinstance(f, RawFragment) for f in fragments):
        raise ValueError("exactly two RawFragment envelopes required")
    namespace = uuid.uuid4().hex
    view_names, evidence_names, access = {}, {}, {}
    locals_ = []
    for index, fragment in enumerate(fragments):
        prepared = fragment.prepared_local
        old_w = prepared.get("received_query_start", 0)
        if type(old_w) is not int or old_w < 0:
            raise ValueError("nonnegative original received query start required")
        local = dict(input_id=f"{namespace}/input{index}", query_id=f"{namespace}/query{index}",
                     key_id=prepared.get("key_id"), received_query_start=0,
                     tau=prepared.get("tau", prepared.get("threshold", {}).get("tau")),
                     status=prepared.get("status"), members=[])
        for field in ("score_complete", "completeness", "threshold_status", "upstream_status"):
            if field in prepared:
                local[field] = deepcopy(prepared[field])
        # Only local state rows cross the boundary; C/D are validated by planner.
        for raw in prepared.get("members", prepared.get("candidates", [])):
            row = {k: deepcopy(raw[k]) for k in ("phase", "source_start", "S", "score_status", "in_C",
                                                 "query_source_origin_hypothesis") if k in raw}
            if "local_received_to_source_map" in raw:
                original = raw["local_received_to_source_map"]
                if isinstance(original, dict):
                    row["local_received_to_source_map"] = {str(x): original.get(str(old_w+x), original.get(old_w+x)) for x in range(45)}
                else:
                    row["local_received_to_source_map"] = deepcopy(original)
            local["members"].append(row)
        if "C" in prepared:
            local["C"] = (None if prepared["C"] is None else
                          [{k: h.get(k) for k in ("phase", "source_start")} for h in prepared["C"]])
        if "D" in prepared:
            local["D"] = prepared["D"]
        views = []
        for original in fragment.views:
            old_view = original.get("view_id")
            if old_view not in view_names:
                view_names[old_view] = f"{namespace}/view{len(view_names)}"
            p = original.get("phase")
            support = original.get("encoder_input_support", list(range(old_w+p, old_w+p+41)))
            if isinstance(support, dict):
                support = support.get("received_frames", [])
            # Encoder context is a public operational description, not file/parent identity.
            context = original.get("encoder_context")
            if isinstance(context, dict):
                context = {k: deepcopy(context[k]) for k in ("reset", "posterior_mode", "preprocessing", "spatial_input") if k in context}
            elif context is not None:
                context = None
            view = dict(view_id=view_names[old_view], input_id=local["input_id"], query_id=local["query_id"],
                        phase=p, received_query_start=0, encoder_input_support=[x-old_w for x in support],
                        encoder_context=context, status=original.get("status", "RETURNED"), units={})
            for r in range(1, 11):
                view["units"][str(r)] = {}
                for region in method.REGIONS:
                    unit = original.get("units", {}).get(str(r), {}).get(region, {})
                    old_eid = unit.get("evidence_id", method.evidence_id(old_view, r, region))
                    if old_eid not in evidence_names:
                        new_eid = f"{namespace}/e{len(evidence_names)}"
                        evidence_names[old_eid] = new_eid
                        access[new_eid] = (fragment.cache, old_eid)
                    view["units"][str(r)][region] = dict(
                        evidence_id=evidence_names[old_eid], observed_region_id=unit.get("observed_region_id", region),
                        carrier_region_id=unit.get("carrier_region_id", region),
                        diagnostic_only=bool(unit.get("diagnostic_only", False)),
                        status=unit.get("status", original.get("status", "RETURNED")))
            views.append(view)
        locals_.append(method.plan_local(local, views))
    plan = dict(source_relations=["SEPARATE", "JOIN_CONDITIONAL"],
                required_assumptions=["same source for conditional JOIN only", "common key/carrier/clock/group semantics"],
                path_context="NOT_APPLICABLE_FRAGMENT_PAIR", source_authenticated=False,
                SEP_A=method.reconcile_plans([locals_[0]]), SEP_B=method.reconcile_plans([locals_[1]]),
                JOIN_CONDITIONAL=method.reconcile_plans(locals_), planned_local_slots=64,
                planned_aggregate_slots=96, message_used=False)

    def cache_reader(eid, key, bit):
        original_cache, original_eid = access[eid]
        return _lookup(original_cache, original_eid, key, bit)

    return plan, cache_reader


def read_bundle_plan(plan, cache_reader):
    """Use the returned/read-back plan, with no access to original envelopes."""
    memo, outputs = {}, {}
    for name in ("SEP_A", "SEP_B", "JOIN_CONDITIONAL"):
        reads = [method.read_local(p, cache_reader, memo=memo) for p in plan[name]]
        outputs[name] = method.aggregate(reads, "JOIN_CONDITIONAL" if name == "JOIN_CONDITIONAL" else "SEPARATE",
                                         path_context="NOT_APPLICABLE_FRAGMENT_PAIR")
    return dict(status="RETURNED", outputs=outputs, planned_local_slots=64, planned_aggregate_slots=96,
                unique_cache_accesses=len(memo), physical_calls=0, source_authenticated=False)


def run_bundle(fragments, output_directory):
    """Persist/re-read this bundle's plan before the first cache callback."""
    directory = Path(output_directory)
    try:
        plan, reader = prepare_bundle(fragments)
        plan = _save_read(directory / "plan.json", plan)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return _failure(f"PLAN_UNAVAILABLE:{type(error).__name__}", "UNRESOLVED")
    try:
        result = read_bundle_plan(plan, reader)
        return _save_read(directory / "decision.json", result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return _failure(f"DECISION_UNAVAILABLE:{type(error).__name__}", "UNRESOLVED")


def run_fixed_bundles(requests, output_directory):
    """Private scheduler supplies 72 envelopes/None; no source roster goes blind.

    Each call has a fresh namespace. Missing requests retain all three 32-slot
    outputs. Shared physical caches do not reduce any logical denominator.
    """
    if len(requests) != 72:
        raise ValueError("c0 requires exactly 72 planned bundle entries, including None failures")
    directory = Path(output_directory)
    rows = []
    for index, request in enumerate(requests):
        result = _failure("UPSTREAM_NOT_ENTERED") if request is None else run_bundle(request, directory / f"bundle-{index:03}")
        rows.append(dict(bundle_slot=index, **result))
    result = dict(bundles=rows, planned_bundles=72, planned_local_slots=4608,
                  planned_sep_slots=4608, planned_join_slots=2304, physical_calls=0,
                  returned_bundles=sum(r["status"] == "RETURNED" for r in rows),
                  not_entered_bundles=sum(r["status"] == "NOT_ENTERED" for r in rows),
                  unresolved_bundles=sum(r["status"] == "UNRESOLVED" for r in rows))
    return _save_read(directory / "bundles.json", result)
