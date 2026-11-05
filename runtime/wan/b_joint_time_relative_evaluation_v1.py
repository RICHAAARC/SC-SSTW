"""Closed saved-output time evaluation, separate from received-only comparison."""
from collections import Counter
from pathlib import Path

from main.tube_state.b_joint_time_relative_validation_v1 import units, denominators
from runtime.wan.native_baseline_storage_v1 import read_json, write_json, now, recover

NODE_METRICS = ("original_correct", "conditional_correct", "original_wrong", "conditional_wrong",
                "newly_correct", "new_wrong_singleton", "previous_correct_lost", "ambiguity_retained",
                "comparable", "base_output_without_written_time")
EDGE_METRICS = ("original_correct", "conditional_correct", "original_wrong", "conditional_wrong",
                "newly_correct", "new_wrong_exact", "previous_correct_lost", "base_output_without_written_time")


def execution_rows(output):
    """Read saved operation evidence; never start or reconstruct a scientific read."""
    path = output / "ledger.json"
    rows = {r["id"]: r for r in read_json(path)} if path.is_file() else {}
    stage = recover(output / "stages/receive")
    for row in stage["operations"]:
        # A bare stage plan must not erase a later persisted parent ledger event.
        if row["id"] not in rows or row["status"] != "NOT_STARTED":
            rows[row["id"]] = row
    return rows, stage["read_errors"]


def unit_execution(unit, recorded_status, available, operations, read_errors):
    rows = [r for key, r in operations.items() if key.startswith(unit["id"] + "/")]
    states = {r["kind"]: r["status"] for r in rows}
    failed = [r for r in rows if r["status"] == "FAILED"]
    interrupted = [r for r in rows if r["status"] in ("STARTED", "INTERRUPTED_CONSUMED")]
    if available:
        status, reason = "BLIND_SAVED", "Actual saved relative output is available"
    elif failed:
        status, reason = "FAILED", "A unit operation failed before blind output was saved"
    elif interrupted:
        status, reason = "INTERRUPTED", "A consumed unit operation has no returned terminal event"
    elif states.get("relative_compare") == "RETURNED" or recorded_status == "BLIND_SAVED":
        status, reason = "OUTPUT_MISSING", "Returned or previously saved comparison has no readable output file"
    elif "RETURNED" in states.values():
        status, reason = "INCOMPLETE", "Some operations returned but no conditional output was saved"
    elif len(rows) == 6 and set(states.values()) == {"NOT_STARTED"} and not read_errors:
        status, reason = "NOT_STARTED", "No unit operation started"
    else:
        status, reason = "UNKNOWN", "Saved accounting is insufficient to classify execution"
    return dict(unit=unit["id"], status=status, reason=reason, recorded_unit_status=recorded_status,
                blind_output_available=available, operation_states=states,
                failure_operations=failed, interrupted_operations=interrupted,
                accounting_sources=["ledger.json", "stages/receive/events.jsonl"],
                relative_file="blind/" + unit["id"] + "/relative.json")


def singleton(values):
    return values[0] if isinstance(values, list) and len(values) == 1 else None


def edge_pair(row):
    if row and row.get("exact_singleton_pair") and len(row.get("pairs", [])) == 1:
        return (row["pairs"][0]["left"], row["pairs"][0]["right"])
    return None


def evaluate_saved(output):
    output = Path(output)
    if read_json(output / "blind_terminated.json")["status"] != "BLIND_TERMINATED":
        raise ValueError("Blind output closure required before evaluation truth access")
    write_json(output / "post_truth_access.json", dict(time=now(), after_blind_terminated=True))
    config = read_json(output / "protocol.json")
    maps = read_json(output / "private/attack_truth.json")
    statuses = read_json(output / "unit_statuses.json")
    operations, accounting_errors = execution_rows(output)
    node_rows, edge_rows, summaries = [], [], []
    for unit in units(config):
        g = maps[unit["view"]]
        if len(g) != unit["frames"]:
            raise ValueError("Fixed evaluation mapping length mismatch")
        path = output / "blind" / unit["id"] / "relative.json"
        value = read_json(path) if path.is_file() else None
        available = value is not None
        execution = unit_execution(unit, statuses[unit["id"]]["status"], available, operations, accounting_errors)
        old_nodes = value["original_f42"]["nodes"] if value else [None]*unit["frames"]
        new_nodes = value["conditional_nodes"] if value else [None]*unit["frames"]
        old_edges = value["original_f42"]["edges"] if value else [None]*unit["edges"]
        new_edges = value["conditional_edges"] if value else [None]*unit["edges"]
        if [len(old_nodes), len(new_nodes), len(old_edges), len(new_edges)] != [unit["frames"], unit["frames"], unit["edges"], unit["edges"]]:
            raise ValueError("Saved result truncated; fixed rows may not be dropped")
        counts = Counter()
        for r, (old, new) in enumerate(zip(old_nodes, new_nodes)):
            old_w = old.get("W") if old else None
            new_w = new.get("conditional_candidates") if new else None
            a, b = singleton(old_w), singleton(new_w)
            row = dict(unit=unit["id"], arm=unit["arm"], received_frame=r, truth_n=g[r],
                       startup_source_position=g[r] < 4, original_W=old_w, conditional_candidates=new_w,
                       original_correct=a == g[r], conditional_correct=b == g[r],
                       original_wrong=a is not None and a != g[r], conditional_wrong=b is not None and b != g[r],
                       newly_correct=a != g[r] and b == g[r],
                       new_wrong_singleton=a is None and b is not None and b != g[r],
                       previous_correct_lost=a == g[r] and b != g[r],
                       ambiguity_retained=isinstance(old_w, list) and len(old_w)>1 and new_w == old_w,
                       comparable=bool(new and new.get("comparison_performed")),
                       base_output_without_written_time=unit["arm"] == "BASE" and b is not None,
                       comparison_status=new["status"] if new else statuses[unit["id"]]["status"],
                       condition_result_is_presence=False)
            row.update(unit_execution_status=execution["status"], evaluable=available,
                       conditional_read_status=("READ_SINGLETON" if b is not None else "READ_UNRESOLVED")
                       if available else "NOT_EVALUABLE")
            if not available:
                row.update({key: None for key in NODE_METRICS})
                row["comparison_status"] = None
            counts.update(k for k, v in row.items() if v is True)
            counts["nodes"] += 1
            counts["nodes_with_blind_output"] += available
            counts["nodes_without_blind_output"] += not available
            if available:
                counts["conditional_unresolved"] += b is None
            node_rows.append(row)
        for r, (old, new) in enumerate(zip(old_edges, new_edges)):
            truth = (g[r], g[r+1])
            a, b = edge_pair(old), edge_pair(new)
            row = dict(unit=unit["id"], arm=unit["arm"], received_edge=r, truth_pair=list(truth),
                       original_exact_pair=a, conditional_exact_pair=b, original_correct=a == truth,
                       conditional_correct=b == truth, original_wrong=a is not None and a != truth,
                       conditional_wrong=b is not None and b != truth, newly_correct=a != truth and b == truth,
                       new_wrong_exact=a is None and b is not None and b != truth,
                       previous_correct_lost=a == truth and b != truth,
                       base_output_without_written_time=unit["arm"] == "BASE" and b is not None,
                       condition_result_is_presence=False)
            row.update(unit_execution_status=execution["status"], evaluable=available,
                       conditional_read_status=("READ_EXACT_PAIR" if b is not None else "READ_UNRESOLVED")
                       if available else "NOT_EVALUABLE")
            if not available:
                row.update({key: None for key in EDGE_METRICS})
            counts.update("edge_" + k for k, v in row.items() if v is True)
            counts["edges"] += 1
            counts["edges_with_blind_output"] += available
            counts["edges_without_blind_output"] += not available
            if available:
                counts["edge_conditional_unresolved"] += b is None
            edge_rows.append(row)
        for key in (*NODE_METRICS, "conditional_unresolved",
                    *("edge_" + key for key in EDGE_METRICS), "edge_conditional_unresolved"):
            counts[key] = counts[key] if available else None
        summaries.append(dict(unit=unit["id"], arm=unit["arm"], planned_nodes=unit["frames"],
                              planned_edges=unit["edges"], counts=dict(counts), status=execution["status"],
                              evaluable=available, execution=execution))
    write_json(output / "node_results.json", node_rows)
    write_json(output / "edge_results.json", edge_rows)
    comparable_new = sum(s["counts"].get("comparable", 0) or 0 for s in summaries if s["arm"] == "NEW")
    complete_new = sum(s["status"] == "BLIND_SAVED" for s in summaries if s["arm"] == "NEW")
    ceiling = "retention only: no comparable NEW ambiguity" if comparable_new == 0 else "single new content conditional-time diagnostic"
    if not complete_new:
        ceiling = "no completed NEW read; execution incomplete"
    elif complete_new < 5:
        ceiling = "partial NEW execution/saved output; " + ceiling
    elif any(s["status"] != "BLIND_SAVED" for s in summaries):
        ceiling = "paired BASE execution incomplete; " + ceiling
    execution_totals = {}
    for s in summaries:
        total = execution_totals.setdefault(s["status"], dict(units=0, nodes=0, edges=0))
        total["units"] += 1
        total["nodes"] += s["planned_nodes"]
        total["edges"] += s["planned_edges"]
    result = dict(status="SAVED_OUTPUT_EVALUATION", denominators=denominators(config), summaries=summaries,
                  comparable_new_nodes=comparable_new,
                  completed_new_units=complete_new, execution_status_counts=execution_totals,
                  execution_accounting_errors=accounting_errors,
                  evaluated_nodes=sum(s["counts"]["nodes_with_blind_output"] for s in summaries),
                  evaluated_edges=sum(s["counts"]["edges_with_blind_output"] for s in summaries),
                  evidence_ceiling=ceiling,
                  presence_claim=False, scientific_pass=None, identity_and_CDC32_new_scores="NOT_IN_THIS_TIME_CONFIRMATION",
                  counterexamples_retained=True, no_tuning_or_retry=True,
                  reports=["node_results.json", "edge_results.json"],
                  note="Repeated views/positions are not independent sources; BASE contains DCT2 identity but no Joint-A time packet")
    write_json(output / "evaluation.json", result)
    return result
