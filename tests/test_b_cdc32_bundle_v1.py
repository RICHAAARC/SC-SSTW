"""Bundle isolation/order/fixed-denominator engineering tests only."""
import json
import copy
import pytest
from runtime.wan import b_cdc32_bundle_v1 as b
from test_b_cdc32_aggregation_v1 import fixture

pytestmark = pytest.mark.unit


def fragments(tmp_path, *, failing=False):
    items = []
    for index, (start, w) in enumerate(((5, 0), (28, 106))):
        p, views, cache = fixture(query=f"secret{index}", start=start, w=w, input_id=f"parent-source-{index}")
        p.update(parent_paths="DO_NOT_COPY", writer_bits="DO_NOT_COPY", aggregate="DO_NOT_COPY")
        for v in views:
            v["encoder_context"]["private_parent"] = "DO_NOT_COPY"
        def reader(e, k, bit, cache=cache):
            assert (tmp_path/"plan.json").is_file()
            plan = json.loads((tmp_path/"plan.json").read_text())
            assert plan["message_used"] is False
            if failing:
                raise KeyError("missing cached unit")
            return cache[e][k][str(bit)]
        items.append(b.RawFragment(p, views, reader))
    return items


def test_bundle_plan_before_first_cache_and_no_parent_information(tmp_path):
    result = b.run_bundle(fragments(tmp_path), tmp_path)
    assert result["status"] == "RETURNED"
    plan = json.loads((tmp_path/"plan.json").read_text())
    text = json.dumps(plan)
    assert "DO_NOT_COPY" not in text and "parent-source" not in text and "secret" not in text
    assert all(p["received_query_start"] == 0 for p in plan["JOIN_CONDITIONAL"])
    right = plan["SEP_B"][0]
    assert right["full_local_C"][0]["query_source_origin_hypothesis"] == 108
    assert right["groups"][2]["references"][0]["encoder_input_support"] == list(range(41))
    assert set(result["outputs"]) == {"SEP_A", "SEP_B", "JOIN_CONDITIONAL"}
    assert all(len(x["slots"]) == 32 and x["source_authenticated"] is False for x in result["outputs"].values())
    assert result["outputs"]["JOIN_CONDITIONAL"]["recovered_slot_count"] == 32
    assert all(x["path_context"] == "NOT_APPLICABLE_FRAGMENT_PAIR" for x in result["outputs"].values())


def test_preparing_bundle_never_reads_cache_and_namespaces_do_not_link():
    items = []
    for q in ("A", "B"):
        p, v, _ = fixture(query=q)
        items.append(b.RawFragment(p, v, lambda *args: pytest.fail("planning must not consume payload")))
    first, _ = b.prepare_bundle(items)
    second, _ = b.prepare_bundle(items)
    assert first["SEP_A"][0]["input_id"] != second["SEP_A"][0]["input_id"]


def test_failed_plan_persistence_never_calls_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "_save_read", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    result = b.run_bundle(fragments(tmp_path), tmp_path)
    assert result["status"] == "UNRESOLVED" and result["unique_cache_accesses"] == 0
    assert sum(len(x["slots"]) for x in result["outputs"].values()) == 96


def test_cache_failures_are_not_replaced_by_other_region_or_source(tmp_path):
    result = b.run_bundle(fragments(tmp_path, failing=True), tmp_path)
    assert result["status"] == "RETURNED"
    assert result["outputs"]["JOIN_CONDITIONAL"]["recovered_slot_count"] == 0
    assert result["outputs"]["JOIN_CONDITIONAL"]["counts"]["technical_unread"] == 32


def test_all72_failed_upstream_bundles_retain_all_slots(tmp_path):
    result = b.run_fixed_bundles([None]*72, tmp_path)
    assert result["not_entered_bundles"] == 72
    assert result["planned_sep_slots"] == 4608 and result["planned_join_slots"] == 2304
    assert sum(len(o["slots"]) for row in result["bundles"] for o in row["outputs"].values()) == 6912
    assert result["physical_calls"] == 0


def test_one_incomplete_fragment_keeps_other_fragment_and_conditional_partial(tmp_path):
    items = fragments(tmp_path)
    items[1].prepared_local["members"][-1]["S"] = None
    items[1].prepared_local["members"][-1]["score_status"] = "FAILED"
    items[1].prepared_local["C"] = None
    items[1].prepared_local["score_complete"] = False
    result = b.run_bundle(items, tmp_path)
    assert result["status"] == "RETURNED"
    assert result["outputs"]["SEP_A"]["recovered_slot_count"] == 16
    assert result["outputs"]["SEP_B"]["recovered_slot_count"] == 0
    join = result["outputs"]["JOIN_CONDITIONAL"]
    assert join["recovered_slot_count"] == 16
    assert all(s["status"] == "PARTIAL_EVIDENCE_READ" for s in join["slots"][:16])
    assert join["source_authenticated"] is False


def test_review_r1_bundle_preserves_failed_view_over_returned_units(tmp_path):
    items = []
    for query in ("A", "B"):
        p, views, _ = fixture(query=query)
        views[0].update(status="FAILED", cause_set=["ENCODER_FAILED"],
                        units={"1": {r: dict(status="RETURNED") for r in ("R0", "R1")}})
        items.append(b.RawFragment(p, views, lambda *args: pytest.fail("failed view must not consume bits")))
    result = b.run_bundle(items, tmp_path)
    assert result["status"] == "RETURNED" and result["unique_cache_accesses"] == 0
    for output in result["outputs"].values():
        assert output["slots"][0]["value"] is None
        assert {"OBSERVATION:FAILED", "ENCODER_FAILED"} <= set(output["slots"][0]["cause_set"])


def test_review_r3_bundle_rejects_duplicate_id_with_different_actual_owners(tmp_path):
    items = []
    for index, query in enumerate(("A", "B")):
        p, views, _ = fixture(query=query)
        views[0]["units"] = {"1": {r: dict(evidence_id=f"shared/{r}") for r in ("R0", "R1")}}
        def reader(*args, value=index):
            pytest.fail(f"ambiguous owner must not consume either cache ({value})")
        items.append(b.RawFragment(p, views, reader))
    result = b.run_bundle(items, tmp_path)
    assert result["status"] == "RETURNED" and result["unique_cache_accesses"] == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    for name, output in result["outputs"].items():
        assert len(output["slots"]) == 32 and output["recovered_slot_count"] == 0
        assert "UNRESOLVED_EVIDENCE_OWNERSHIP" in output["slots"][0]["cause_set"]
        for local in plan[name]:
            group = local["groups"][0]
            assert group["status"] == "UNRESOLVED_EVIDENCE_OWNERSHIP"
            assert all(r["status"] == "UNRESOLVED_EVIDENCE_OWNERSHIP" for r in group["references"])


def test_review_r3_bundle_true_alias_shares_evidence_across_distinct_cache_objects(tmp_path):
    p, views, cache = fixture()
    calls = []
    items = []
    for index in range(2):
        def reader(eid, key, bit, local_cache=copy.deepcopy(cache), index=index):
            assert (tmp_path / "plan.json").is_file()
            calls.append((index, eid, key, bit))
            return local_cache[eid][key][str(bit)]
        items.append(b.RawFragment(copy.deepcopy(p), copy.deepcopy(views), reader))
    result = b.run_bundle(items, tmp_path)
    assert result["status"] == "RETURNED" and result["unique_cache_accesses"] == len(calls) == 16
    assert len({(eid, key, bit) for _, eid, key, bit in calls}) == 16
    plan = json.loads((tmp_path / "plan.json").read_text())
    a = plan["SEP_A"][0]["groups"][0]["references"]
    c = plan["SEP_B"][0]["groups"][0]["references"]
    assert {r["evidence_id"] for r in a} == {r["evidence_id"] for r in c}
    for output in result["outputs"].values():
        assert output["recovered_slot_count"] == 8
        assert all(row["value"] == 1 for row in output["slots"][:8])


def test_review_r4_bundle_saved_plan_and_slots_preserve_local_failure_details(tmp_path):
    p, views, _ = fixture()
    missing = [dict(phase=0, source_start=1, cause_set=["MISSING_MEMBER", "INVALID_SCORE"], private="DO_NOT_COPY")]
    extra = [dict(index=137, reason="OUT_OF_DOMAIN_MEMBER", parent_path="DO_NOT_COPY")]
    p.update(status="TECHNICAL_UNRESOLVED", score_status="SCORE_UNRESOLVED", score_complete=False,
             upstream_status="SCORE_FAILED", cause_set=["INCOMPLETE_H137", "UNEXPECTED_MEMBERS"],
             missing_members=missing, unexpected_members=extra, upstream_error="DO_NOT_COPY", C=None)
    p["members"][0].update(S=None, score_status="TECHNICAL_UNRESOLVED", upstream_score_status="ENCODER_FAILED",
                           cause_set=["MISSING_MEMBER", "INVALID_SCORE"], parent_path="DO_NOT_COPY")
    items = [b.RawFragment(copy.deepcopy(p), copy.deepcopy(views), lambda *args: pytest.fail("failed family must not read"))
             for _ in range(2)]
    result = b.run_bundle(items, tmp_path)
    assert result["status"] == "RETURNED" and result["unique_cache_accesses"] == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert "DO_NOT_COPY" not in json.dumps(plan) and "DO_NOT_COPY" not in json.dumps(result)
    for name, output in result["outputs"].items():
        for local in plan[name]:
            assert len(local["members"]) == 137
            member = local["members"][0]
            assert member["phase"] == 0 and member["source_start"] == 1
            assert member["cause_set"] == ["INVALID_SCORE", "MISSING_MEMBER"]
            assert member["upstream_score_status"] == "ENCODER_FAILED"
            assert local["missing_members"] == [dict(phase=0, source_start=1, cause_set=["INVALID_SCORE", "MISSING_MEMBER"])]
            assert local["unexpected_members"] == [dict(index=137, reason="OUT_OF_DOMAIN_MEMBER")]
        assert len(output["slots"]) == 32 and output["counts"]["failed"] == 32
        for slot in output["slots"]:
            assert {"INVALID_SCORE", "MISSING_MEMBER", "UPSTREAM:ENCODER_FAILED", "OUT_OF_DOMAIN_MEMBER"} <= set(slot["cause_set"])
            for failure in slot["failure_details"]:
                assert failure["details"]["missing_members"][0]["source_start"] == 1
                assert failure["details"]["unexpected_members"][0]["index"] == 137
                assert failure["details"]["member_failures"][0]["upstream_score_status"] == "ENCODER_FAILED"
