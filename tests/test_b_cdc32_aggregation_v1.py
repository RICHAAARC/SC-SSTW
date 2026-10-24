"""Small synthetic engineering cases; no saved scientific observations."""
import copy
import json
import pytest
from main.tube_state import b_cdc32_aggregation_v1 as m

pytestmark = pytest.mark.unit


def fixture(query="L", start=1, w=0, input_id="opaque", candidates=None):
    active = [(0, start)] if candidates is None else candidates
    members = [dict(phase=p, source_start=s, S=.8 if (p, s) in active else .1,
                    score_status="RETURNED", in_C=(p, s) in active,
                    local_received_to_source_map=[4*(s-1)-p+j for j in range(45)]) for p, s in m.H]
    prepared = dict(query_id=query, input_id=input_id, key_id="key00", received_query_start=w,
                    members=members, tau=.5, C=[h for h in members if h["in_C"]], D=bool(active))
    views = [dict(view_id=f"{input_id}-{query}-p{p}", phase=p, status="RETURNED",
                  encoder_input_support=list(range(w+p, w+p+41)), encoder_context=dict(reset=True)) for p in range(4)]
    cache = {m.evidence_id(view["view_id"], row, region): {"key00": {str(b): dict(value=1, count=8, status="RETURNED")
                     for b in range(8)}} for view in views for row in range(1, 11) for region in m.REGIONS}
    return prepared, views, cache


def read_fixture(**kwargs):
    p, v, c = fixture(**kwargs)
    return m.plan_local(p, v), c


def test_full_c_intersection_earliest_and_two_replicas():
    plan, cache = read_fixture(candidates=[(0, 1), (0, 2)])
    assert plan["groups"][0]["selected_canonical_cell"] == 2
    assert len(plan["groups"][0]["references"]) == 4
    assert plan["groups"][1]["selected_canonical_cell"] is None
    result = m.read_local(plan, cache)
    assert [s["value"] for s in result["slots"]] == [1]*8+[None]*24
    assert len(result["slots"]) == 32


def test_equal_tau_rejects_without_cache_access_and_incomplete_family_retains137():
    p, views, _ = fixture()
    p["members"][0]["S"] = .5
    p["D"] = False
    plan = m.plan_local(p, views)
    assert len(plan["full_local_C"]) == 1 and not plan["D"]
    m.read_local(plan, lambda *args: pytest.fail("equal threshold must not read"))
    p["members"][-1]["S"] = None
    p["members"][-1]["score_status"] = "FAILED"
    plan = m.plan_local(p, views)
    assert len(plan["members"]) == 137 and plan["status"] == "TECHNICAL_UNRESOLVED"
    result = m.read_local(plan, lambda *args: pytest.fail("incomplete family must not read"))
    assert all(s["status"] == "LOCAL_TECHNICAL_UNREAD" for s in result["slots"])


def test_pruned_c_cannot_silently_widen_support():
    p, views, _ = fixture(candidates=[(0, 1), (0, 12)])
    p["C"] = p["C"][:1]
    plan = m.plan_local(p, views)
    assert plan["status"] == "TECHNICAL_UNRESOLVED"


def test_region_conflict_and_missing_preserve_both_causes_no_next_cell():
    plan, cache = read_fixture(candidates=[(0, 1), (0, 2)])
    refs = plan["groups"][0]["references"]
    cache[refs[0]["evidence_id"]]["key00"]["0"]["value"] = 0
    del cache[refs[-1]["evidence_id"]]["key00"]["0"]
    result = m.read_local(plan, cache)
    slot = result["slots"][0]
    assert slot["status"] == "LOCAL_CONFLICT" and slot["technical_failure"]
    assert "BIT_CONFLICT" in slot["cause_set"] and "TECHNICAL_CACHE" in slot["cause_set"]
    assert slot["selected_canonical_cell"] == 2 and slot["missing_refs"]


def test_known_local_conflict_vetoes_union_but_failure_allows_partial_evidence():
    p1, c1 = read_fixture(query="L")
    p2, c2 = read_fixture(query="R", start=2)
    ref = p1["groups"][0]["references"][0]
    c1[ref["evidence_id"]]["key00"]["0"]["value"] = 0
    reads = [m.read_local(p1, c1), m.read_local(p2, c2)]
    assert m.aggregate(reads)["slots"][0]["status"] == "CONFLICT_ERASURE"
    assert m.union_reference(reads)["slots"][0]["value"] == 1
    del c1[ref["evidence_id"]]["key00"]["0"]
    reads = [m.read_local(p1, c1), m.read_local(p2, c2)]
    output = m.aggregate(reads)
    assert output["slots"][0]["value"] == 1
    assert output["slots"][0]["status"] == "PARTIAL_EVIDENCE_READ"
    assert reads[0]["slots"][0]["value"] is None


def test_alias_x10_and_reverse_are_invariant_and_cache_is_unique():
    plans, cache = {}, {}
    for q, start, w in (("L", 1, 0), ("M", 12, 53), ("R", 34, 106)):
        p, c = read_fixture(query=q, start=start, w=w)
        plans[q] = p
        cache.update(c)
    addresses = []
    def read(e, k, b):
        addresses.append((e, k, b))
        return cache[e][k][str(b)]
    pack = m.plan_comparisons(plans)
    output = m.read_comparisons(pack, read, path_context={"E2": "AVAILABLE", "M3": "EMPTY"})
    assert all(output["invariance"].values())
    assert len(addresses) == len(set(addresses)) == output["unique_cache_accesses"]
    assert output["CDC_M"]["recovered_slot_count"] > output["CDC_E"]["recovered_slot_count"]
    assert output["CDC_M"]["path_context"] == "EMPTY"


def test_same_evidence_mapping_conflict_marks_all_affected_plans():
    p, views, _ = fixture()
    a = m.plan_local(p, views)
    p2 = copy.deepcopy(p)
    p2["query_id"] = "R"
    p2["members"][0]["S"] = .1
    p2["members"][0]["in_C"] = False
    p2["members"][1]["S"] = .8
    p2["members"][1]["in_C"] = True
    p2["C"] = [p2["members"][1]]
    b = m.plan_local(p2, views)
    reconciled = m.reconcile_plans([a, b])
    assert all(p["groups"][0]["status"] == "UNRESOLVED_MAPPING_CONFLICT" for p in reconciled)
    reads = [m.read_local(p, lambda *args: pytest.fail("conflicting map must not read")) for p in reconciled]
    assert m.aggregate(reads)["canonical_cells_under_h"] == []


def test_direct_aggregate_cannot_hide_mapping_conflict_after_separate_reads():
    p, views, cache = fixture()
    a = m.plan_local(p, views)
    p2, _, _ = fixture(query="R", start=2)
    b = m.plan_local(p2, views)
    result = m.aggregate([m.read_local(a, cache), m.read_local(b, cache)])
    assert result["slots"][0]["value"] is None
    assert "UNRESOLVED_MAPPING_CONFLICT" in result["slots"][0]["cause_set"]
    assert result["canonical_cells_under_h"] == []


def test_explicit_mixed_and_unknown_mapping_never_use_equal_bits():
    p, views, cache = fixture()
    p["members"][0]["local_received_to_source_map"] = [None]*45
    plan = m.plan_local(p, views)
    assert all(g["status"] == "UNRESOLVED_MAPPING" for g in plan["groups"])
    p["members"][0]["local_received_to_source_map"] = [j+1 for j in range(45)]
    plan = m.plan_local(p, views)
    assert all(g["selected_canonical_cell"] is None for g in plan["groups"])
    result = m.read_local(plan, lambda *args: pytest.fail("mixed cell cannot be rescued by bits"))
    assert len(result["slots"]) == 32


def test_region_diagnostic_cannot_fill_cdc_missing_replica():
    p, views, cache = fixture()
    plan = m.plan_local(p, views)
    r1 = next(r for r in plan["groups"][0]["references"] if r["observed_region_id"] == "R1")
    del cache[r1["evidence_id"]]
    assert m.read_local(m.plan_local(p, views, ("R0",)), cache)["slots"][0]["value"] == 1
    assert m.read_local(plan, cache)["slots"][0]["status"] == "LOCAL_TECHNICAL_UNREAD"


def test_plan_is_json_and_contains_no_payload_or_parent_metadata():
    p, views, _ = fixture(w=53)
    p.update(writer_bits="FORBIDDEN", source_id="FORBIDDEN", path_context="FORBIDDEN")
    plan = m.plan_local(p, views)
    assert "FORBIDDEN" not in json.dumps(plan)
    assert plan["full_local_C"][0]["query_source_origin_hypothesis"] == 0
    assert plan["groups"][0]["references"][0]["nominal_received_cell"] == [54,55,56,57]


def test_presence_categories_and_not_entered_denominator_remain_distinct():
    # Origin3 retains part of cell1/12 and no cell13. Origin0 has a whole
    # cell11 outside its fixed phase0 E41. Keep these distinct under-h states.
    p, views, _ = fixture(candidates=[(1, 2)])
    plan = m.plan_local(p, views)
    presence = plan["support_under_h"][0]["source_cell_presence"]
    assert presence[0]["status"] == "PARTIAL_UNDER_H"
    assert presence[12]["status"] == "STRUCTURALLY_ABSENT_UNDER_H"
    other, other_views, _ = fixture()
    assert m.plan_local(other, other_views)["support_under_h"][0]["source_cell_presence"][10]["status"] == "PRESENT_UNOBSERVED_UNDER_H"
    p["status"] = "NOT_ENTERED"
    plan = m.plan_local(p, views)
    result = m.aggregate([m.read_local(plan, lambda *args: pytest.fail("not entered"))])
    counts = result["counts"]
    assert counts["planned"] == counts["returned"] + counts["failed"] + counts["not_entered"] == 32
    assert counts["not_entered"] == 32


def test_three_independent_inputs_never_become_a_join_or_lose_individual_slots():
    reads = []
    for i in range(3):
        plan, cache = read_fixture(input_id=f"input{i}")
        reads.append(m.read_local(plan, cache))
    result = m.aggregate(reads, "JOIN_CONDITIONAL")
    assert result["recovered_slot_count"] == 0
    assert len(result["per_query_slots"]) == 3
    assert all(s["status"] == "UNSUPPORTED_ARITY_OR_SEMANTICS" for s in result["slots"])
    assert result["source_authenticated"] is False


def test_unexpected_member_flag_cannot_be_erased_by_complete_looking_rows():
    p, views, _ = fixture()
    p.update(score_complete=False, completeness="UNRESOLVED")
    plan = m.plan_local(p, views)
    assert plan["status"] == "TECHNICAL_UNRESOLVED" and plan["full_local_C"] == []
    p, views, _ = fixture(candidates=[])
    p["C"] = None  # Unresolved is not an empty, completely decided C.
    assert m.plan_local(p, views)["status"] == "TECHNICAL_UNRESOLVED"


@pytest.mark.parametrize("record", [
    {"value": 1}, {"value": 1, "status": "AVAILABLE"}, {"value": 1, "count": 8},
    {"value": True, "count": 8, "status": "RETURNED"},
    {"value": 1, "count": True, "status": "RETURNED"},
    {"value": 1, "count": 7, "status": "RETURNED"},
    {"value": 1, "count": 8, "status": "FAILED"},
    {"value": 1, "count": 8, "status": "RETURNED", "diagnostic_only": True},
])
def test_review_r1_invalid_or_diagnostic_cache_retains_failed_slot(record):
    plan, cache = read_fixture()
    for ref in plan["groups"][0]["references"]:
        cache[ref["evidence_id"]]["key00"]["0"] = record
    result = m.read_local(plan, cache)
    slot = result["slots"][0]
    assert slot["value"] is None and slot["status"] == "LOCAL_TECHNICAL_UNREAD"
    assert "TECHNICAL_CACHE" in slot["cause_set"] and len(slot["missing_refs"]) == 2
    assert len(result["slots"]) == 32 and result["counts"]["failed"] == 1
    assert result["slots"][1]["value"] == 1
    if record.get("diagnostic_only"):
        assert "TECHNICAL_CACHE:DIAGNOSTIC_ONLY" in slot["cause_set"]


def test_review_r1_available_cache_and_implicit_units_remain_supported():
    plan, cache = read_fixture()
    for ref in plan["groups"][0]["references"]:
        for record in cache[ref["evidence_id"]]["key00"].values():
            record["status"] = "AVAILABLE"
    assert all(s["value"] == 1 for s in m.read_local(plan, cache)["slots"][:8])


def test_review_r1_failed_view_cannot_be_overridden_by_returned_units():
    p, views, _ = fixture()
    views[0].update(status="FAILED", cause_set=["ENCODER_FAILED"],
                    units={"1": {r: dict(status="RETURNED") for r in m.REGIONS}})
    plan = m.plan_local(p, views)
    result = m.read_local(plan, lambda *args: pytest.fail("failed view must not consume bits"))
    assert all(s["value"] is None for s in result["slots"])
    assert result["slots"][0]["status"] == "LOCAL_TECHNICAL_UNREAD"
    assert {"OBSERVATION:FAILED", "ENCODER_FAILED"} <= set(result["slots"][0]["cause_set"])
    assert all(r["view_status"] == "FAILED" and r["unit_status"] == "RETURNED"
               for r in result["slots"][0]["missing_refs"])


@pytest.mark.parametrize("failure", [
    {"status": "TECHNICAL_UNRESOLVED"}, {"score_status": "SCORE_UNRESOLVED"},
])
def test_review_r1_explicit_family_failure_cannot_be_recomputed_as_ready(failure):
    p, views, _ = fixture()
    p.update(failure)
    plan = m.plan_local(p, views)
    assert plan["status"] == "TECHNICAL_UNRESOLVED" and plan["full_local_C"] == []
    result = m.read_local(plan, lambda *args: pytest.fail("failed family must not consume bits"))
    assert all(s["status"] == "LOCAL_TECHNICAL_UNREAD" for s in result["slots"])
    assert result["counts"] == dict(planned=32, returned=0, failed=32, not_entered=0)


def test_review_r2_partial_read_keeps_failed_cell_out_of_recovered_coverage():
    left, left_cache = read_fixture(query="L", start=1)
    right, right_cache = read_fixture(query="R", start=2)
    for ref in left["groups"][0]["references"]:
        del left_cache[ref["evidence_id"]]
    result = m.aggregate([m.read_local(left, left_cache), m.read_local(right, right_cache)])
    assert result["canonical_cells_under_h"] == [2]
    assert result["planned_canonical_cells_under_h"] == [1, 2]
    assert result["slots"][0]["value"] == 1 and result["slots"][0]["status"] == "PARTIAL_EVIDENCE_READ"
    assert {r["canonical_cell"] for r in result["slots"][0]["supporting_refs"]} == {2}
    buckets = {row["canonical_cell"]: row for row in result["source_position_buckets"] if row["slot"] == 0}
    assert buckets[1]["query_id"] == "L" and buckets[1]["status"] == "LOCAL_TECHNICAL_UNREAD"
    assert buckets[1]["value"] is None and not buckets[1]["supports_aggregate_value"]
    assert len(buckets[1]["missing_refs"]) == 2 and "TECHNICAL_CACHE" in buckets[1]["cause_set"]
    assert buckets[1]["conditional_candidate_ids"] == ["p0:s1"]
    assert buckets[2]["query_id"] == "R" and buckets[2]["status"] == "LOCAL_CONDITIONAL_READ"
    assert buckets[2]["supports_aggregate_value"] and buckets[2]["conditional_candidate_ids"] == ["p0:s2"]


def test_review_r3_actual_owner_conflict_marks_all_core_refs_before_read():
    plans = []
    for query in ("A", "B"):
        p, views, _ = fixture(query=query)
        views[0]["units"] = {"1": {r: dict(evidence_id=f"shared/{r}") for r in m.REGIONS}}
        plans.append(m.plan_local(p, views))
    checked = m.reconcile_plans(plans)
    for plan in checked:
        group = plan["groups"][0]
        assert group["status"] == "UNRESOLVED_EVIDENCE_OWNERSHIP"
        assert all(r["status"] == "UNRESOLVED_EVIDENCE_OWNERSHIP" for r in group["references"])
        result = m.read_local(plan, lambda *args: pytest.fail("ambiguous owner must not read"))
        assert result["slots"][0]["value"] is None
        assert "UNRESOLVED_EVIDENCE_OWNERSHIP" in result["slots"][0]["cause_set"]
