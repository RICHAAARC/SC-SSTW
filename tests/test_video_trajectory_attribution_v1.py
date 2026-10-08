import ast
import copy
import json
from pathlib import Path

import pytest

from main.tube_state import video_trajectory_attribution_v1 as method
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align
from experiments.wan_state_clock import video_trajectory_attribution_v1_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.quick
KEY = "watermark"


def cache177(level=1.0, tie=False, key=KEY):
    rows = []
    for r in range(177):
        for d in range(5):
            rho = 160 if r + d == 180 else 40
            q = 0.0 if tie else level if d == 2 else 0.0
            rows.append(
                {
                    "r": r,
                    "d": d,
                    "source_index": r + d,
                    "source_tubelet": (r + d) // 4,
                    "source_age": (r + d) % 4,
                    "status": "SCORED",
                    "numerator": q * rho,
                    "rho": rho,
                }
            )
    return method.deletion.reduce_cache(rows, key)


def rows89(level=1.0, tie_offsets=(), key=KEY):
    local = []
    candidates = []
    active = set(tie_offsets or (38,))
    for offset in range(93):
        numerator = 0.0
        denominator = 0.0
        for support in align.sync.tubelet_support_rows(89, offset, align.PUBLIC):
            rho = support["observed_frames"] * (
                160 if support["source_tubelet"] == 45 else 40
            )
            q = level if offset in active else 0.0
            row = {
                "source_offset": offset,
                **support,
                "status": "SCORED",
                "q": q,
                "signed_projection": q * rho,
                "rho": rho,
            }
            row["q"] = row["signed_projection"] / row["rho"]
            local.append(row)
            numerator += row["signed_projection"]
            denominator += rho
        candidates.append(
            {
                "source_offset": offset,
                "status": "SCORED",
                "numerator": numerator,
                "denominator": denominator,
                "score": numerator / denominator,
                "tubelet_rows": len(align.sync.tubelet_support_rows(89, offset, align.PUBLIC)),
            }
        )
    scores = [row["score"] for row in candidates]
    best = max(scores)
    top = [index for index, score in enumerate(scores) if abs(score - best) <= align.PUBLIC.tie_atol]
    return {
        "status": "COMPLETE",
        "truth_inputs": False,
        "key_id": align.sync.key_identifier(key),
        "candidate_offsets": list(range(93)),
        "candidate_rows": candidates,
        "local_rows": local,
        "counts": {
            "candidate_scores": 93,
            "candidate_tubelet_rows": len(local),
            "failed_candidates": 0,
        },
        "summary": {
            "top_offsets": top,
            "canonical_offset": top[0] if len(top) == 1 else None,
            "unique": len(top) == 1,
            "best_score": best,
            "sync_accepted": False,
        },
    }


def frozen_rules():
    return {
        "status": "FROZEN",
        "rule_sha256": "a" * 64,
        "thresholds": {
            "181": {"tau_M": 1.0, "tau_m": None, "tau_I": 0.2},
            "177": {"tau_M": 1.0, "tau_m": 0.1, "tau_I": 0.2},
            "89": {"tau_M": 1.0, "tau_m": 0.1, "tau_I": 0.2},
        },
    }


def sync(frames=177, M=2.0, m=0.5, actions=1, status="COMPLETE"):
    return {
        "status": status,
        "M": M,
        "m": None if frames == 181 else m,
        "unique_action": actions == 1,
        "top_action_count": actions,
        "top_paths": [{"fixture": True}],
        "chosen_action": {
            "received_index_map": list(range(frames))
        }
        if actions == 1
        else None,
    }


def identity(I=0.8, exact=True, status="COMPLETE", attempted=True):
    return {"status": status, "I": I, "exact32": exact, "attempted": attempted}


def development_rows():
    rows = []
    labels = {}
    for row in protocol.query_roster(("DEV",)):
        N = row["frames"]
        label = protocol.calibration_label(row)
        null = label["sync_null"]
        positive = label["positive"]
        rows.append(
            {
                **row,
                "sync": sync(N, M=1.0 if null else 3.0, m=0.1 if null else 0.5),
                "identity": identity(I=0.8 if positive else 0.1, exact=positive),
            }
        )
        labels[row["query_id"]] = label
    return rows, labels


def test_fixed_roster_denominator_and_opaque_ids():
    rows = protocol.query_roster()
    assert len(rows) == 192
    assert len({row["query_id"] for row in rows}) == 192
    assert len({row["observation_id"] for row in rows}) == 96
    assert all(row["query_id"].startswith("Q") for row in rows)
    assert all(row["observation_id"].startswith("O") for row in rows)
    assert protocol.fixed_denominator() == {
        "sources": 3,
        "native_trajectories": 9,
        "received_conditions": 12,
        "physical_observations": 96,
        "logical_queries": 192,
        "sync_candidates": 74784,
        "framewise_frames": 13872,
        "framewise_batches": 1812,
        "max_payload_reads": 192,
        "logical_payload_votes": 6589440,
        "logical_time_bit_rows": 219648,
        "logical_final_bits": 6144,
    }


def test_177_local_contrast_restores_adopted_counterexample():
    raw = cache177()
    original = copy.deepcopy(raw)
    result = method.summarize_sync(raw, 177, KEY)
    rows, _ = method._candidate_rows(raw, 177, KEY)
    assert result["status"] == "COMPLETE"
    assert result["representative_path"] == {"family": "H0", "b": 2, "k": None}
    assert result["M"] == 1.0
    top = next(row for row in rows if row["path"] == result["representative_path"])
    global_action_gap = top["score"] - max(
        row["score"] for row in rows if row["action_id"] != top["action_id"]
    )
    assert global_action_gap == pytest.approx(0.011299435028248594)
    assert result["m"] == pytest.approx(1.0)
    assert raw == original


def test_89_local_q_recomputation_and_every_top_path():
    raw = rows89()
    result = method.summarize_sync(raw, 89, KEY)
    assert result["status"] == "COMPLETE"
    assert result["M"] == 1.0 and result["m"] == pytest.approx(1.0)
    same_action = method.summarize_sync(rows89(tie_offsets=(0, 4)), 89, KEY)
    assert same_action["unique_action"] is True
    assert len(same_action["top_paths"]) == 2
    assert same_action["m"] == pytest.approx(1.0)
    different_action = method.summarize_sync(rows89(tie_offsets=(0, 1)), 89, KEY)
    assert different_action["unique_action"] is False
    assert different_action["top_action_count"] == 2
    assert different_action["m"] == pytest.approx(0.0)


def test_local_formula_rejects_tampered_q_and_original_score():
    raw = rows89()
    raw["local_rows"][0]["q"] += 0.01
    assert method.summarize_sync(raw, 89, KEY)["status"] == "TECHNICAL_INCOMPLETE"
    raw = rows89()
    raw["candidate_rows"][0]["score"] += 0.01
    assert method.summarize_sync(raw, 89, KEY)["status"] == "TECHNICAL_INCOMPLETE"


def test_low_sync_precedes_action_tie_and_payload():
    decision = method.decide(
        sync(M=1.0, actions=2),
        identity(status="TECHNICAL_INCOMPLETE"),
        177,
        frozen_rules(),
    )
    assert decision["reason"] == "REJECT_LOW_SYNC"
    assert method.decide(sync(M=2.0, actions=2), identity(), 177, frozen_rules())[
        "reason"
    ] == "UNCERTAIN_SYNC"


def test_identity_gray_negative_accept_and_internal_uncertain():
    assert method.decide(sync(), identity(I=-0.01, exact=False), 177, frozen_rules())[
        "reason"
    ] == "REJECT_IDENTITY"
    assert method.decide(sync(), identity(I=0.1, exact=False), 177, frozen_rules())[
        "reason"
    ] == "UNCERTAIN_IDENTITY_WEAK"
    assert method.decide(sync(), identity(), 177, frozen_rules())["decision"] == "ACCEPT"
    assert method.decide(sync(), identity(I=0.8, exact=False), 177, frozen_rules())[
        "reason"
    ] == "UNCERTAIN_INTERNAL_INCONSISTENCY"


def test_freeze_once_from_dev64_explicit_labels_and_fail_closed():
    rows, labels = development_rows()
    frozen = method.freeze_rules(rows, labels, "b" * 64, "c" * 64)
    assert frozen["status"] == "FROZEN"
    assert frozen["thresholds"]["177"]["tau_M"] == 1.0
    assert frozen["thresholds"]["177"]["tau_m"] == 0.1
    assert frozen["thresholds"]["177"]["tau_I"] == 0.1
    assert method.freeze_rules(rows[:-1], labels, "b" * 64, "c" * 64)[
        "status"
    ] == "NOT_FREEZABLE"
    broken = copy.deepcopy(rows)
    broken[0]["sync"]["status"] = "TECHNICAL_INCOMPLETE"
    assert method.freeze_rules(broken, labels, "b" * 64, "c" * 64)[
        "status"
    ] == "NOT_FREEZABLE"


def test_noneligible_identity_null_retained_but_attempted_failure_blocks():
    rows, labels = development_rows()
    tied = next(
        row
        for row in rows
        if row["frames"] == 177
        and labels[row["query_id"]]["identity_null_class"] == "OFF/K0"
    )
    tied["sync"].update(unique_action=False, top_action_count=2, chosen_action=None)
    tied["identity"] = identity(
        status="NOT_ELIGIBLE", I=None, exact=None, attempted=False
    )
    assert method.freeze_rules(rows, labels, "b" * 64, "c" * 64)["status"] == "FROZEN"
    rows, labels = development_rows()
    failed = next(
        row
        for row in rows
        if row["frames"] == 177
        and labels[row["query_id"]]["identity_null_class"] == "OFF/K0"
    )
    failed["identity"] = identity(
        status="TECHNICAL_INCOMPLETE", I=None, exact=None, attempted=True
    )
    assert method.freeze_rules(rows, labels, "b" * 64, "c" * 64)[
        "status"
    ] == "NOT_FREEZABLE"


@pytest.mark.parametrize(
    "reason",
    [
        "UNCERTAIN_TECHNICAL_SOURCE",
        "UNCERTAIN_TECHNICAL_EVIDENCE",
        "UNCERTAIN_INTERNAL_INCONSISTENCY",
        "UNCERTAIN_UNSUPPORTED_PROTOCOL",
        "UNCERTAIN_RULE_NOT_FREEZABLE",
        "UNCERTAIN_TECHNICAL_INTERRUPTION",
    ],
)
def test_unknown_and_technical_reasons_are_unresolved(reason):
    row = {
        "decision": "UNCERTAIN",
        "reason": reason,
    }
    posthoc = method.posthoc_false_claim(
        row, {"expected_accept": False, "required_for_false_claim": True}, None
    )
    assert posthoc["technical_complete"] is False
    assert (
        method.aggregate_false_claim([{"posthoc": posthoc}]) == "UNRESOLVED"
    )


def test_accept_wrong_action_and_negative_accept_are_false_claims():
    positive_role = {"expected_accept": True, "required_for_false_claim": True}
    wrong = method.posthoc_false_claim(
        {"decision": "ACCEPT", "reason": "ACCEPT"}, positive_role, False
    )
    assert wrong["false_claim"] is True
    assert wrong["false_claim_reason"] == "ACCEPT_WRONG_ACTION"
    missing_map = method.posthoc_false_claim(
        {"decision": "ACCEPT", "reason": "ACCEPT"}, positive_role, None
    )
    assert missing_map["technical_complete"] is False
    unavailable = method.posthoc_false_claim(
        {"decision": "REJECT", "reason": "REJECT_LOW_SYNC"},
        positive_role,
        None,
        true_action_available=False,
    )
    assert unavailable["technical_complete"] is False
    negative = method.posthoc_false_claim(
        {"decision": "ACCEPT", "reason": "ACCEPT"},
        {"expected_accept": False, "required_for_false_claim": True},
        True,
    )
    assert negative["false_claim"] is True
    assert method.aggregate_false_claim(
        [{"posthoc": wrong}, {"posthoc": missing_map}]
    ) is True


def test_unsupported_missing_nonfinite_and_fractional_votes_fail_stably():
    assert method.decide(sync(), identity(), 90, frozen_rules())["reason"] == "UNCERTAIN_UNSUPPORTED_PROTOCOL"
    broken = frozen_rules()
    del broken["thresholds"]["177"]
    assert method.decide(sync(), identity(), 177, broken)["reason"] == "UNCERTAIN_UNSUPPORTED_PROTOCOL"
    assert method.decide(sync(M=float("nan")), identity(), 177, frozen_rules())[
        "reason"
    ] == "UNCERTAIN_TECHNICAL_SYNC"
    rows = []
    for index, bit in enumerate(method.CLAIM_BITS):
        rows.append(
            {
                "bit_index": index,
                "ones": 9.0 if bit else 1.0,
                "zeros": 1.0 if bit else 9.0,
                "count": 10.0,
                "decoded": bit,
            }
        )
    assert method.identity_evidence(
        {"status": "READ", "truth_inputs": False, "bit_rows": rows}
    )["status"] == "TECHNICAL_INCOMPLETE"


def test_core_has_no_experiment_truth_roster_and_protocol_has_it():
    source = (ROOT / "main/tube_state/video_trajectory_attribution_v1.py").read_text()
    for forbidden in (
        "SOURCE_IDS",
        "VIDEO_IDS",
        "VIEW_MAPS",
        "query_roster",
        "true_action_for_view",
        "experiments",
        "runtime",
    ):
        assert forbidden not in source
    assert len(protocol.true_action_for_view("DELETE177_B2K88")) == 177
    ast.parse(source)


def test_config_and_notebook_draft_schema():
    from experiments.wan_state_clock import video_trajectory_attribution_v1_run as runner

    cfg = runner.load_config()
    assert cfg["fixed_denominator"] == protocol.fixed_denominator(3)
    assert list(cfg["sources"]) == ["DEV", "C1", "C2"]
    assert [cfg["sources"][source]["seed"] for source in cfg["sources"]] == [
        2026100701,
        2026100802,
        2026100803,
    ]
    assert {
        name: tuple(row["received_index_map"]) for name, row in cfg["views"].items()
    } == protocol.VIEW_MAPS
    nb = json.loads(
        (ROOT / "notebooks/video_trajectory_attribution_v1_colab.ipynb").read_text()
    )
    assert nb["metadata"]["candidate_binding"] == {
        "candidate": "trajectory-attribution-v1",
        "source_sha": None,
        "status": "UNPUBLISHED_DRAFT",
    }
    code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == (
        "from google.colab import drive\ndrive.mount('/content/drive')\n"
    )
    for cell in code:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        ast.parse("".join(cell["source"]))


def _fill_fake_source(store, source_id, rules_value=None, fail_attempt=False):
    for query in store.data["queries"].values():
        if query["source_id"] != source_id:
            continue
        label = protocol.calibration_label(query)
        null = label["sync_null"]
        positive = label["positive"]
        query["sync"] = sync(
            query["frames"],
            M=1.0 if null else 3.0,
            m=0.1 if null else 0.5,
        )
        query["sync"]["chosen_action"] = {
            "received_index_map": protocol.true_action_for_view(query["view_id"])
        }
        query["identity"] = identity(
            I=0.8 if positive else 0.1, exact=positive, attempted=True
        )
        query["status"] = "PAYLOAD_READ"
    if fail_attempt and source_id == "DEV":
        row = next(
            query
            for query in store.data["queries"].values()
            if query["source_id"] == "DEV"
            and query["frames"] == 177
            and protocol.calibration_label(query)["identity_null_class"] == "OFF/K0"
        )
        row["identity"] = identity(
            status="TECHNICAL_INCOMPLETE", I=None, exact=None, attempted=True
        )
    seal_id = runner_module()._source_seal_id(source_id)
    store.data["seals"][source_id + "/sync"] = {
        "path": str(store.output / "seals" / (seal_id + ".sync.json")),
        "sha256": "a" * 64,
    }
    store.data["seals"][source_id + "/payload"] = {
        "path": str(store.output / "seals" / (seal_id + ".payload.json")),
        "sha256": "b" * 64,
    }
    store.save()


def runner_module():
    from experiments.wan_state_clock import video_trajectory_attribution_v1_run as runner

    return runner


def test_fake_full_dev_confirmation_has_all_decision_seals(tmp_path, monkeypatch):
    runner = runner_module()
    calls = []

    def prepared(store, source_id):
        calls.append(source_id)
        store.data["sources"][source_id]["status"] = "COMPLETE"
        store.save()
        return {"received": {name: {} for name in protocol.VIDEO_IDS}, "status": "COMPLETE"}

    monkeypatch.setattr(runner, "prepare_source", prepared)
    monkeypatch.setattr(
        runner,
        "collect_source",
        lambda store, source_id, record, rules=None: _fill_fake_source(
            store, source_id, rules
        ),
    )
    result = runner.run(tmp_path / "complete", runner.load_config())
    assert result["status"] == "COMPLETE"
    assert result["rules"]["status"] == "FROZEN"
    assert calls == ["DEV", "C1", "C2"]
    assert all(source + "/decisions" in result["seals"] for source in protocol.SOURCE_IDS)
    assert all(result["sources"][source]["any_false_claim"] is False for source in protocol.SOURCE_IDS)


def test_rule_failure_retains_128_not_run_unresolved(tmp_path, monkeypatch):
    runner = runner_module()
    calls = []

    def prepared(store, source_id):
        calls.append(source_id)
        store.data["sources"][source_id]["status"] = "COMPLETE"
        store.save()
        return {"received": {name: {} for name in protocol.VIDEO_IDS}, "status": "COMPLETE"}

    monkeypatch.setattr(runner, "prepare_source", prepared)
    monkeypatch.setattr(
        runner,
        "collect_source",
        lambda store, source_id, record, rules=None: _fill_fake_source(
            store, source_id, rules, fail_attempt=True
        ),
    )
    result = runner.run(tmp_path / "blocked", runner.load_config())
    confirmation = [
        query
        for query in result["queries"].values()
        if query["source_id"] in ("C1", "C2")
    ]
    assert calls == ["DEV"]
    assert len(confirmation) == 128
    assert all(query["status"] == "NOT_RUN" for query in confirmation)
    assert all(
        result["sources"][source]["any_false_claim"] == "UNRESOLVED"
        for source in ("C1", "C2")
    )
    assert all(source + "/decisions" in result["seals"] for source in protocol.SOURCE_IDS)


def test_partial_confirmation_failure_is_unresolved_and_c2_continues(tmp_path, monkeypatch):
    runner = runner_module()
    calls = []

    def prepared(store, source_id):
        calls.append(source_id)
        if source_id == "C1":
            raise RuntimeError("fixture source failure")
        store.data["sources"][source_id]["status"] = "COMPLETE"
        store.save()
        return {"received": {name: {} for name in protocol.VIDEO_IDS}, "status": "COMPLETE"}

    monkeypatch.setattr(runner, "prepare_source", prepared)
    monkeypatch.setattr(
        runner,
        "collect_source",
        lambda store, source_id, record, rules=None: _fill_fake_source(
            store, source_id, rules
        ),
    )
    result = runner.run(tmp_path / "partial", runner.load_config())
    assert calls == ["DEV", "C1", "C2"]
    assert result["status"] == "RETAINED_INCOMPLETE"
    assert result["sources"]["C1"]["any_false_claim"] == "UNRESOLVED"
    assert result["sources"]["C2"]["status"] == "COMPLETE"
    assert all(source + "/decisions" in result["seals"] for source in protocol.SOURCE_IDS)


def test_c1_keyboard_interrupt_preserves_dev_partial_c1_and_notruns_c2(tmp_path, monkeypatch):
    runner = runner_module()

    def prepared(store, source_id):
        if source_id == "C1":
            raise KeyboardInterrupt("fixture cancel")
        store.data["sources"][source_id]["status"] = "COMPLETE"
        store.save()
        return {"received": {name: {} for name in protocol.VIDEO_IDS}, "status": "COMPLETE"}

    monkeypatch.setattr(runner, "prepare_source", prepared)
    monkeypatch.setattr(
        runner,
        "collect_source",
        lambda store, source_id, record, rules=None: _fill_fake_source(
            store, source_id, rules
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run(tmp_path / "interrupt", runner.load_config())
    result = json.loads((tmp_path / "interrupt" / "result.json").read_text())
    dev = [row for row in result["queries"].values() if row["source_id"] == "DEV"]
    c1 = [row for row in result["queries"].values() if row["source_id"] == "C1"]
    c2 = [row for row in result["queries"].values() if row["source_id"] == "C2"]
    assert all(row["decision"] is not None for row in dev)
    assert all(row["decision"]["reason"] == "UNCERTAIN_TECHNICAL_INTERRUPTION" for row in c1)
    assert all(row["status"] == "NOT_RUN" for row in c2)
    assert all(row["decision"]["reason"] == "UNCERTAIN_TECHNICAL_INTERRUPTION" for row in c2)
    assert not any(row["status"] == "PENDING" for row in result["queries"].values())
    assert all(source + "/decisions" in result["seals"] for source in protocol.SOURCE_IDS)


def test_worker_wait_interrupt_always_reaps_group(tmp_path, monkeypatch):
    runner = runner_module()
    store = type(
        "WorkerStore",
        (),
        {
            "output": tmp_path,
            "cfg": {"_config_path": str(runner.CONFIG)},
        },
    )()
    calls = []

    class Child:
        pid = 123456

        def wait(self):
            raise KeyboardInterrupt("fixture")

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(runner, "_stop_process_group", lambda child: calls.append(child.pid))
    with pytest.raises(KeyboardInterrupt):
        runner.run_worker(store, "DEV", "generation")
    assert calls == [123456]


def test_observation_failures_retain_both_keys_and_other_slots(tmp_path, monkeypatch):
    runner = runner_module()
    store = runner.Store(tmp_path / "slots", runner.load_config())
    record = {
        "received": {
            name: {"condition": name} for name in protocol.VIDEO_IDS
        }
    }

    class Inputs:
        @staticmethod
        def read(spec):
            return {"condition": spec["condition"]}

        @staticmethod
        def construct(full, mapping):
            if full["condition"] == "OFF" and tuple(mapping) == protocol.VIEW_MAPS["CROP177_P1"]:
                raise RuntimeError("fixture sync/read failure")
            return {"condition": full["condition"], "mapping": tuple(mapping)}

        @staticmethod
        def receipt(observed):
            frames = len(observed["mapping"])
            return {
                "sha256": "d" * 64,
                "shape": [frames, 320, 512, 3],
                "bytes": frames * 320 * 512 * 3,
                "dtype": "uint8",
            }

    class Framewise:
        def __init__(self, cfg):
            pass

        def encode(self, observed):
            if (
                observed["condition"] == "B_M05"
                and observed["mapping"] == protocol.VIEW_MAPS["CROP177_P2"]
            ):
                raise RuntimeError("fixture framewise encode failure")
            return observed

        def close(self):
            pass

    class Wan:
        def __init__(self, cfg):
            pass

        def encode(self, observed):
            if (
                observed["condition"] == "A_P1"
                and observed["mapping"] == protocol.VIEW_MAPS["SHORT89_S37"]
            ):
                raise RuntimeError("fixture payload encode failure")
            return observed

        def read(self, latent, key, frames):
            return {"fixture": True}

        def close(self):
            pass

    def fake_summary(raw, frames, key):
        return {
            **sync(frames),
            "top_paths": [{"fixture": True}],
            "truth_inputs": False,
        }

    monkeypatch.setattr(runner.runtime, "score_received", lambda backend, latent, key, frames: {"fixture": True})
    monkeypatch.setattr(runner.method, "summarize_sync", fake_summary)
    monkeypatch.setattr(runner.runtime, "apply_action", lambda observed, action: observed)
    monkeypatch.setattr(
        runner.method,
        "identity_evidence",
        lambda detail: identity(I=0.5, exact=True, attempted=True),
    )
    runner.collect_source(
        store,
        "DEV",
        record,
        rules=None,
        input_type=Inputs,
        framewise_type=Framewise,
        wan_type=Wan,
    )
    failed_sync = [
        row
        for row in store.data["queries"].values()
        if row["source_id"] == "DEV"
        and row["video_id"] == "OFF"
        and row["view_id"] == "CROP177_P1"
    ]
    failed_payload = [
        row
        for row in store.data["queries"].values()
        if row["source_id"] == "DEV"
        and row["video_id"] == "A_P1"
        and row["view_id"] == "SHORT89_S37"
    ]
    failed_framewise = [
        row
        for row in store.data["queries"].values()
        if row["source_id"] == "DEV"
        and row["video_id"] == "B_M05"
        and row["view_id"] == "CROP177_P2"
    ]
    completed = next(
        row
        for row in store.data["queries"].values()
        if row["source_id"] == "DEV"
        and row["video_id"] == "A_M05"
        and row["view_id"] == "FULL181"
        and row["key_label"] == "K0"
    )
    assert len(failed_sync) == 2
    assert all(row["sync"]["status"] == "TECHNICAL_INCOMPLETE" for row in failed_sync)
    assert len(failed_framewise) == 2
    assert all(row["sync"]["status"] == "TECHNICAL_INCOMPLETE" for row in failed_framewise)
    assert len(failed_payload) == 2
    assert all(row["identity"]["status"] == "TECHNICAL_INCOMPLETE" for row in failed_payload)
    assert len(store.data["failures"]) >= 3
    assert completed["identity"]["status"] == "COMPLETE"
    seal = json.loads(
        Path(store.data["seals"]["DEV/sync"]["path"]).read_text()
    )
    encoded = json.dumps(seal)
    assert "DEV" not in encoded
    assert "A_P1" not in encoded
    assert "CROP177_P1" not in encoded
    assert all("received_index_map" not in row for row in seal["observations"].values())
    assert all(key.startswith("O") for key in seal["observations"])
    assert all(key.startswith("Q") for key in seal["queries"])
