"""Bundle isolation/order/fixed-denominator engineering tests only."""
import json
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
