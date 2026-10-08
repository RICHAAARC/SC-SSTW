"""Fixed saved-RGB uncertainty follow-up for trajectory attribution V1."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal

from main.tube_state import video_trajectory_attribution_v1 as method
from runtime.wan import video_trajectory_attribution_v1 as runtime
from runtime.wan.provenance import source_identity
from experiments.wan_state_clock import video_trajectory_attribution_uncertainty_v1_protocol as protocol

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/wan_state_clock/configs/video_trajectory_attribution_uncertainty_v1.json"
ENTRY = "experiments.wan_state_clock.video_trajectory_attribution_uncertainty_v1_run"


class TerminationRequested(BaseException):
    pass


def _handle_sigterm(signum, frame):
    raise TerminationRequested(f"signal {signum}")


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path) -> str:
    return digest_bytes(Path(path).read_bytes())


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    os.replace(temp, path)
    return {"path": str(path), "bytes": len(raw), "sha256": digest_bytes(raw)}


def _semantic_slots():
    return [
        {
            name: row[name]
            for name in ("source", "condition", "key", "view", "parameters", "role", "focus")
            if name in row
        }
        for row in protocol.query_roster()
    ]


def load_config(path=CONFIG):
    path = Path(path).resolve()
    cfg = read(path)
    fixed = read(CONFIG)
    if cfg != fixed:
        raise ValueError("fixed uncertainty config required")
    if (
        cfg["name"] != "video_trajectory_attribution_uncertainty_v1"
        or cfg["marker"] != protocol.MARKER
        or cfg["formula_version"] != method.RULE_FORMULA_VERSION
        or cfg["root_protocol_freeze_sha256"] != protocol.ROOT_PROTOCOL_FREEZE_SHA256
        or cfg["frozen_rules"]["rule_sha256"] != protocol.FROZEN_RULE_SHA256
        or cfg["frozen_rules"]["file_sha256"] != protocol.FROZEN_RULE_FILE_SHA256
        or cfg["fixed_denominator"] != protocol.FIXED_DENOMINATOR
        or cfg["slots"] != _semantic_slots()
        or cfg["coverage"]["sync_denominator"] != 10
        or cfg["coverage"]["weak_identity_denominator"] != 10
    ):
        raise ValueError("fixed uncertainty protocol mismatch")
    if set(cfg["inputs"]) != set(protocol.SOURCE_IDS):
        raise ValueError("fixed source inputs")
    for source in protocol.SOURCE_IDS:
        if set(cfg["inputs"][source]) != {"A_M05", "OFF", "B_M05"}:
            raise ValueError("fixed condition inputs")
        for spec in cfg["inputs"][source].values():
            if spec["shape"] != [181, 320, 512, 3] or spec["bytes"] != 88965120:
                raise ValueError("fixed input geometry")
    cfg["_config_path"] = str(path)
    return cfg


def load_frozen_rules(cfg):
    path = Path(cfg["frozen_rules"]["path"])
    raw = path.read_bytes()
    if digest_bytes(raw) != cfg["frozen_rules"]["file_sha256"]:
        raise ValueError("frozen rule byte identity mismatch")
    rules = json.loads(raw)
    if (
        rules.get("status") != "FROZEN"
        or rules.get("formula_version") != method.RULE_FORMULA_VERSION
        or rules.get("rule_sha256") != protocol.FROZEN_RULE_SHA256
        or any(rules["thresholds"][str(n)]["tau_I"] != 0.0 for n in (181, 177, 89))
    ):
        raise ValueError("frozen attribution rule mismatch")
    return rules, raw


def _technical_sync(error):
    return {
        "status": "TECHNICAL_INCOMPLETE",
        "error": str(error),
        "M": None,
        "m": None,
        "top_paths": [],
        "top_action_ids": [],
        "top_action_count": 0,
        "unique_action": False,
        "chosen_action": None,
        "truth_inputs": False,
    }


def _technical_identity(error, attempted=False):
    return {
        "status": "TECHNICAL_INCOMPLETE",
        "attempted": attempted,
        "error": str(error),
        "I": None,
        "exact32": None,
        "truth_inputs": False,
    }


def _not_eligible_identity():
    return {
        "status": "NOT_ELIGIBLE",
        "attempted": False,
        "error": "PAYLOAD_NOT_RUN_WITHOUT_QUALIFIED_UNIQUE_ACTION",
        "I": None,
        "exact32": None,
        "truth_inputs": False,
    }


class Store:
    def __init__(self, output, cfg):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.cfg = cfg
        self.sync_records = {}
        self.observation_records = {}
        self.payload_records = {}
        provenance = source_identity(ROOT)
        self.data = {
            "status": "RUNNING",
            "stage": "INITIALIZED",
            "marker": protocol.MARKER,
            "source_sha": provenance["git_commit"],
            "source_provenance": provenance,
            "config_sha256": digest(cfg["_config_path"]),
            "roster_sha256": protocol.roster_sha256(),
            "input_contract_sha256": protocol.input_contract_sha256(cfg["inputs"]),
            "prior_run": cfg["prior_run"],
            "fixed_denominator": cfg["fixed_denominator"],
            "rules": {
                "status": "PENDING",
                "expected_rule_sha256": cfg["frozen_rules"]["rule_sha256"],
                "expected_file_sha256": cfg["frozen_rules"]["file_sha256"],
            },
            "queries": {
                row["query_id"]: dict(
                    row,
                    claim_message=method.CLAIM_MESSAGE,
                    status="PENDING",
                    sync=None,
                    identity=None,
                    decision=None,
                )
                for row in protocol.blind_roster()
            },
            "seals": {},
            "coverage": {"status": "PENDING"},
            "controls": {"status": "PENDING"},
            "false_attribution": {"status": "PENDING"},
            "failures": [],
            "evidence_ceiling": cfg["evidence_ceiling"],
        }
        self.save()

    def save(self):
        dump(self.output / "result.json", self.data)

    def failure(self, stage, exc, *, observation_id=None, query_id=None):
        row = {"stage": stage, "error": f"{type(exc).__name__}: {exc}"}
        if observation_id is not None:
            row["observation_id"] = observation_id
        if query_id is not None:
            row["query_id"] = query_id
        self.data["failures"].append(row)
        self.save()


def _rows_by_observation():
    groups = {}
    for row in protocol.query_roster():
        groups.setdefault(row["observation_id"], []).append(row)
    return groups


def _query(store, query_id):
    return store.data["queries"][query_id]


def _observed(cfg, row, input_type):
    spec = cfg["inputs"][row["source"]][row["condition"]]
    full = input_type.read(spec)
    mapping = protocol.received_index_map(row["view"], row["parameters"])
    observed = input_type.construct(full, mapping)
    receipt = input_type.receipt(observed)
    return observed, receipt


def _sync_eligible(sync, frames, rules):
    if sync.get("status") != "COMPLETE" or not sync.get("unique_action"):
        return False
    threshold = rules["thresholds"][str(frames)]
    return sync["M"] > threshold["tau_M"] and (
        frames == 181 or sync["m"] > threshold["tau_m"]
    )


def seal_sync(store):
    expected_observations = set(_rows_by_observation())
    expected_queries = {row["query_id"] for row in protocol.query_roster()}
    if set(store.observation_records) != expected_observations:
        raise ValueError("complete 20-row opaque observation roster required")
    if set(store.sync_records) != expected_queries:
        raise ValueError("complete 22-row opaque sync roster required")
    for query_id in expected_queries:
        summary = store.sync_records[query_id].get("summary")
        if summary is None or _query(store, query_id).get("sync") != summary:
            raise ValueError("sync record/query mismatch")
    if "sync" in store.data["seals"]:
        return
    receipt = dump(
        store.output / "seals/blind_sync.json",
        {
            "observations": store.observation_records,
            "queries": store.sync_records,
            "truth_inputs": False,
        },
    )
    store.data["seals"]["sync"] = receipt
    store.save()


def collect_sync(store, rules, input_type=runtime.Inputs, framewise_type=runtime.FramewiseBackend):
    groups = _rows_by_observation()
    backend = None
    try:
        try:
            backend = framewise_type(store.cfg)
        except Exception as exc:
            store.failure("SYNC_BACKEND", exc)
            for observation_id, rows in groups.items():
                store.observation_records[observation_id] = {
                    "observation_id": observation_id,
                    "frames": rows[0]["frames"],
                    "status": "TECHNICAL_INCOMPLETE",
                    "error": str(exc),
                }
            for row in protocol.query_roster():
                summary = _technical_sync(exc)
                _query(store, row["query_id"]).update(status="SYNC_FAILED", sync=summary)
                store.sync_records[row["query_id"]] = {
                    "query_id": row["query_id"],
                    "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                    "summary": summary,
                }
            store.save()
            return
        for observation_id, rows in groups.items():
            anchor = rows[0]
            try:
                observed, receipt = _observed(store.cfg, anchor, input_type)
                latent = backend.encode(observed)
                store.observation_records[observation_id] = {
                    "observation_id": observation_id,
                    "frames": len(protocol.received_index_map(anchor["view"], anchor["parameters"])),
                    "rgb_sha256": receipt["sha256"],
                    "rgb_shape": receipt["shape"],
                    "rgb_bytes": receipt["bytes"],
                    "rgb_dtype": receipt["dtype"],
                }
            except Exception as exc:
                store.failure("SYNC_OBSERVATION", exc, observation_id=observation_id)
                store.observation_records[observation_id] = {
                    "observation_id": observation_id,
                    "frames": anchor["frames"],
                    "status": "TECHNICAL_INCOMPLETE",
                    "error": str(exc),
                }
                for row in rows:
                    summary = _technical_sync(exc)
                    _query(store, row["query_id"]).update(status="SYNC_FAILED", sync=summary)
                    store.sync_records[row["query_id"]] = {
                        "query_id": row["query_id"],
                        "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                        "summary": summary,
                    }
                store.save()
                continue
            for row in rows:
                key = store.cfg["keys"][row["key"]]
                try:
                    raw = runtime.score_received(backend, latent, key, row["frames"])
                    raw_receipt = dump(store.output / "blind_sync" / (row["query_id"] + ".json.gz"), raw)
                    summary = method.summarize_sync(raw, row["frames"], key)
                    store.sync_records[row["query_id"]] = {
                        "query_id": row["query_id"],
                        "key_id": hashlib.sha256(key.encode()).hexdigest(),
                        "raw": raw_receipt,
                        "summary": summary,
                    }
                    _query(store, row["query_id"]).update(status="SYNC_READ", sync=summary)
                except Exception as exc:
                    store.failure("SYNC_QUERY", exc, observation_id=observation_id, query_id=row["query_id"])
                    summary = _technical_sync(exc)
                    store.sync_records[row["query_id"]] = {
                        "query_id": row["query_id"],
                        "key_id": hashlib.sha256(key.encode()).hexdigest(),
                        "summary": summary,
                    }
                    _query(store, row["query_id"]).update(status="SYNC_FAILED", sync=summary)
                store.save()
            observed = latent = None
    finally:
        if backend is not None:
            backend.close()
        if len(store.observation_records) == 20 and len(store.sync_records) == 22 and all(
            row.get("sync") is not None for row in store.data["queries"].values()
        ):
            seal_sync(store)


def _add_exact_vote_differences(identity, detail):
    if identity.get("status") != "COMPLETE":
        return identity
    differences = []
    for bit, row in zip(method.CLAIM_BITS, detail["bit_rows"]):
        differences.append((2 * bit - 1) * (row["ones"] - row["zeros"]))
    identity["signed_vote_differences"] = differences
    identity["zero_signed_vote_bits"] = sum(value == 0 for value in differences)
    return identity


def seal_payload(store):
    expected_queries = {row["query_id"] for row in protocol.query_roster()}
    if set(store.payload_records) != expected_queries:
        raise ValueError("complete 22-row opaque payload roster required")
    for query_id in expected_queries:
        identity = store.payload_records[query_id].get("identity")
        if identity is None or _query(store, query_id).get("identity") != identity:
            raise ValueError("payload record/query mismatch")
    if "payload" in store.data["seals"]:
        return
    receipt = dump(
        store.output / "seals/blind_payload.json",
        {"queries": store.payload_records, "truth_inputs": False},
    )
    store.data["seals"]["payload"] = receipt
    store.save()


def collect_payload(store, rules, input_type=runtime.Inputs, wan_type=runtime.WanBackend):
    groups = _rows_by_observation()
    eligible_ids = {
        row["query_id"]
        for row in protocol.query_roster()
        if _sync_eligible(_query(store, row["query_id"]).get("sync") or {}, row["frames"], rules)
    }
    for row in protocol.query_roster():
        if row["query_id"] not in eligible_ids:
            identity = _not_eligible_identity()
            _query(store, row["query_id"]).update(status="PAYLOAD_NOT_ELIGIBLE", identity=identity)
            store.payload_records[row["query_id"]] = {
                "query_id": row["query_id"],
                "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                "identity": identity,
            }
    store.save()
    if not eligible_ids:
        seal_payload(store)
        return

    backend = None
    try:
        try:
            backend = wan_type(store.cfg)
        except Exception as exc:
            store.failure("PAYLOAD_BACKEND", exc)
            for row in protocol.query_roster():
                if row["query_id"] not in eligible_ids:
                    continue
                identity = _technical_identity(exc, attempted=True)
                _query(store, row["query_id"]).update(status="PAYLOAD_FAILED", identity=identity)
                store.payload_records[row["query_id"]] = {
                    "query_id": row["query_id"],
                    "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                    "identity": identity,
                }
            store.save()
            return
        for observation_id, rows in groups.items():
            selected = [row for row in rows if row["query_id"] in eligible_ids]
            if not selected:
                continue
            anchor = rows[0]
            try:
                observed, _ = _observed(store.cfg, anchor, input_type)
            except Exception as exc:
                store.failure("PAYLOAD_OBSERVATION", exc, observation_id=observation_id)
                for row in selected:
                    identity = _technical_identity(exc, attempted=True)
                    _query(store, row["query_id"]).update(status="PAYLOAD_FAILED", identity=identity)
                    store.payload_records[row["query_id"]] = {
                        "query_id": row["query_id"],
                        "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                        "identity": identity,
                    }
                store.save()
                continue
            for row in selected:
                query = _query(store, row["query_id"])
                key = store.cfg["keys"][row["key"]]
                try:
                    corrected = runtime.apply_action(observed, query["sync"]["chosen_action"])
                    latent = backend.encode(corrected)
                    detail = backend.read(latent, key, row["frames"])
                    detail_receipt = dump(
                        store.output / "blind_payload" / (row["query_id"] + ".json.gz"), detail
                    )
                    identity = _add_exact_vote_differences(method.identity_evidence(detail), detail)
                    store.payload_records[row["query_id"]] = {
                        "query_id": row["query_id"],
                        "key_id": hashlib.sha256(key.encode()).hexdigest(),
                        "detail": detail_receipt,
                        "identity": identity,
                    }
                    query.update(
                        status="PAYLOAD_READ" if identity["status"] == "COMPLETE" else "PAYLOAD_FAILED",
                        identity=identity,
                    )
                    corrected = latent = None
                except Exception as exc:
                    store.failure("PAYLOAD_QUERY", exc, observation_id=observation_id, query_id=row["query_id"])
                    identity = _technical_identity(exc, attempted=True)
                    query.update(status="PAYLOAD_FAILED", identity=identity)
                    store.payload_records[row["query_id"]] = {
                        "query_id": row["query_id"],
                        "key_id": hashlib.sha256(key.encode()).hexdigest(),
                        "identity": identity,
                    }
                store.save()
            observed = None
    finally:
        if backend is not None:
            backend.close()
        if len(store.payload_records) == 22 and all(
            row.get("identity") is not None for row in store.data["queries"].values()
        ):
            seal_payload(store)


def settle_evidence(store, reason):
    for observation_id, rows in _rows_by_observation().items():
        if observation_id not in store.observation_records:
            store.observation_records[observation_id] = {
                "observation_id": observation_id,
                "frames": rows[0]["frames"],
                "status": "TECHNICAL_INCOMPLETE",
                "error": str(reason),
            }
    for row in protocol.query_roster():
        query = _query(store, row["query_id"])
        sync_record = store.sync_records.get(row["query_id"])
        if sync_record is not None and sync_record.get("summary") is not None:
            query["sync"] = sync_record["summary"]
            query["status"] = (
                "SYNC_READ" if query["sync"].get("status") == "COMPLETE" else "SYNC_FAILED"
            )
        elif query.get("sync") is None:
            query["sync"] = _technical_sync(reason)
            query["status"] = "SYNC_FAILED"
        if row["query_id"] not in store.sync_records:
            store.sync_records[row["query_id"]] = {
                "query_id": row["query_id"],
                "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                "summary": query["sync"],
            }
        payload_record = store.payload_records.get(row["query_id"])
        if payload_record is not None and payload_record.get("identity") is not None:
            query["identity"] = payload_record["identity"]
            if query["identity"].get("status") == "COMPLETE":
                query["status"] = "PAYLOAD_READ"
            elif query["identity"].get("status") == "NOT_ELIGIBLE":
                query["status"] = "PAYLOAD_NOT_ELIGIBLE"
            else:
                query["status"] = "PAYLOAD_FAILED"
        elif query.get("identity") is None:
            query["identity"] = _technical_identity(reason, attempted=False)
            if query["status"] == "PENDING":
                query["status"] = "PAYLOAD_FAILED"
        if row["query_id"] not in store.payload_records:
            store.payload_records[row["query_id"]] = {
                "query_id": row["query_id"],
                "key_id": hashlib.sha256(store.cfg["keys"][row["key"]].encode()).hexdigest(),
                "identity": query["identity"],
            }
    store.save()
    seal_sync(store)
    seal_payload(store)


def decide_all(store, rules, interruption=False):
    for row in protocol.query_roster():
        query = _query(store, row["query_id"])
        incomplete = (
            query["sync"].get("status") != "COMPLETE"
            or query["identity"].get("status") == "TECHNICAL_INCOMPLETE"
        )
        decision = method.decide(query["sync"], query["identity"], row["frames"], rules)
        if interruption and incomplete and decision.get("reason") not in method.SCIENTIFIC_REASONS:
            decision = {
                "decision": "UNCERTAIN",
                "reason": "UNCERTAIN_TECHNICAL_INTERRUPTION",
                "rule_sha256": rules.get("rule_sha256"),
                "frames": row["frames"],
                "claim_message": method.CLAIM_MESSAGE,
            }
        query["decision"] = decision
    store.save()


def seal_decisions(store, rules):
    if "decisions" in store.data["seals"]:
        return
    if any(row.get("decision") is None for row in store.data["queries"].values()):
        raise ValueError("complete 22-row decision roster required")
    receipt = dump(
        store.output / "seals/blind_decisions.json",
        {
            "rule_sha256": rules.get("rule_sha256"),
            "decisions": {
                query_id: row["decision"] for query_id, row in store.data["queries"].items()
            },
            "truth_inputs": False,
        },
    )
    store.data["seals"]["decisions"] = receipt
    store.save()


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _sync_trigger(query, frames, rules):
    sync = query["sync"]
    if query["decision"].get("reason") != "UNCERTAIN_SYNC":
        return False
    threshold = rules.get("thresholds", {}).get(str(frames))
    if not isinstance(threshold, dict):
        return False
    if (
        sync.get("status") != "COMPLETE"
        or not _finite(sync.get("M"))
        or not (sync["M"] > threshold["tau_M"])
    ):
        return False
    distinct_action_tie = type(sync.get("top_action_count")) is int and sync["top_action_count"] > 1
    low_local_margin = (
        frames != 181
        and sync.get("unique_action") is True
        and _finite(sync.get("m"))
        and sync["m"] <= threshold["tau_m"]
    )
    return distinct_action_tie or low_local_margin


def _weak_identity_trigger(query, action_map_correct, frames, rules):
    sync, identity = query["sync"], query["identity"]
    if query["decision"].get("reason") != "UNCERTAIN_IDENTITY_WEAK":
        return False
    threshold = rules.get("thresholds", {}).get(str(frames))
    if not isinstance(threshold, dict):
        return False
    differences = identity.get("signed_vote_differences")
    sync_qualified = (
        sync.get("status") == "COMPLETE"
        and sync.get("unique_action") is True
        and _finite(sync.get("M"))
        and sync["M"] > threshold["tau_M"]
        and (frames == 181 or (_finite(sync.get("m")) and sync["m"] > threshold["tau_m"]))
    )
    return (
        sync_qualified
        and action_map_correct is True
        and identity.get("status") == "COMPLETE"
        and identity.get("I") == 0.0
        and isinstance(differences, list)
        and len(differences) == 32
        and all(type(value) is int and value >= 0 for value in differences)
        and any(value == 0 for value in differences)
    )


def _coverage_status(valid, technical):
    if valid:
        return "OBSERVED"
    if technical:
        return "UNRESOLVED_TECHNICAL"
    return "NOT_OBSERVED_FIXED_ROSTER"


def join_truth_and_report(store, rules):
    if "decisions" not in store.data["seals"]:
        raise ValueError("decision seal required before truth join")
    primary = []
    controls = []
    aliases = []
    all_posthoc = []
    for truth in protocol.query_roster():
        query = _query(store, truth["query_id"])
        true_action = protocol.true_action(truth["view"], truth["parameters"])
        chosen = query["sync"].get("chosen_action")
        action_map_correct = (
            chosen.get("received_index_map") == true_action if isinstance(chosen, dict) else None
        )
        role = protocol.role(truth)
        false_claim = method.posthoc_false_claim(
            query["decision"], role, action_map_correct, true_action_available=True
        )
        expected = protocol.regression_expected(truth)
        posthoc = {
            "source": truth["source"],
            "condition": truth["condition"],
            "key": truth["key"],
            "view": truth["view"],
            "parameters": truth["parameters"],
            "role": truth["role"],
            "focus": truth.get("focus"),
            "expected_accept": role["expected_accept"],
            "action_map_correct": action_map_correct,
            "absolute_source_path_ambiguous": len(query["sync"].get("top_paths", [])) > 1,
            "same_action_path_alias_only": (
                len(query["sync"].get("top_paths", [])) > 1
                and query["sync"].get("top_action_count") == 1
            ),
            "regression_expected": expected,
            "regression_match": (
                expected is not None
                and query["decision"].get("decision") == expected["decision"]
                and query["decision"].get("reason") == expected["reason"]
                and (
                    not role["expected_accept"]
                    or (
                        action_map_correct is True
                        and query["identity"].get("status") == "COMPLETE"
                        and query["identity"].get("exact32") is True
                    )
                )
            ),
            **false_claim,
        }
        posthoc["valid_sync_uncertainty_coverage"] = (
            truth["role"] == "PRIMARY_PROBE" and _sync_trigger(query, truth["frames"], rules)
        )
        posthoc["valid_weak_identity_coverage"] = (
            truth["role"] == "PRIMARY_PROBE"
            and _weak_identity_trigger(query, action_map_correct, truth["frames"], rules)
        )
        query["posthoc"] = posthoc
        all_posthoc.append(query)
        if truth["role"] == "PRIMARY_PROBE":
            primary.append(query)
        elif truth["role"] == "REGRESSION_CONTROL":
            controls.append(query)
        else:
            aliases.append(query)

    scientific_reasons = method.SCIENTIFIC_REASONS
    technical_primary = sum(
        row["decision"].get("reason") not in scientific_reasons for row in primary
    )
    sync_actual = sum(row["decision"].get("reason") == "UNCERTAIN_SYNC" for row in primary)
    sync_valid = sum(row["posthoc"]["valid_sync_uncertainty_coverage"] for row in primary)
    weak_actual = sum(
        row["decision"].get("reason") == "UNCERTAIN_IDENTITY_WEAK" for row in primary
    )
    weak_valid = sum(row["posthoc"]["valid_weak_identity_coverage"] for row in primary)
    store.data["coverage"] = {
        "same_ten_primary_queries": True,
        "primary_denominator": 10,
        "probe_focus": {
            "IDENTITY_FOCUS": sum(row["posthoc"]["focus"] == "IDENTITY_FOCUS" for row in primary),
            "SYNC_FOCUS": sum(row["posthoc"]["focus"] == "SYNC_FOCUS" for row in primary),
        },
        "sync_unreliable_action": {
            "denominator": 10,
            "actual_trigger": sync_actual,
            "valid_coverage": sync_valid,
            "technical": technical_primary,
            "status": _coverage_status(sync_valid, technical_primary),
        },
        "weak_identity": {
            "denominator": 10,
            "actual_trigger": weak_actual,
            "valid_coverage": weak_valid,
            "technical": technical_primary,
            "tau_I": 0.0,
            "status": _coverage_status(weak_valid, technical_primary),
        },
        "alias_controls_excluded": len(aliases),
        "regression_controls_excluded": len(controls),
    }
    store.data["controls"] = {
        "status": "COMPLETE" if all(row["posthoc"]["technical_complete"] for row in controls) else "UNRESOLVED_TECHNICAL",
        "denominator": 8,
        "positive": {
            "denominator": 2,
            "matched": sum(row["posthoc"]["expected_accept"] and row["posthoc"]["regression_match"] for row in controls),
        },
        "negative": {
            "denominator": 6,
            "matched": sum((not row["posthoc"]["expected_accept"]) and row["posthoc"]["regression_match"] for row in controls),
        },
        "rows_matching_prior_decision_reason": sum(row["posthoc"]["regression_match"] for row in controls),
    }
    aggregate = method.aggregate_false_claim(all_posthoc)
    store.data["false_attribution"] = {
        "status": "COMPLETE" if aggregate != "UNRESOLVED" else "UNRESOLVED",
        "all_queries_denominator": 22,
        "positive_denominator": 16,
        "negative_denominator": 6,
        "false_accept_queries": sum(row["posthoc"]["false_claim"] for row in all_posthoc),
        "unknown_queries": sum(not row["posthoc"]["technical_complete"] for row in all_posthoc),
        "any_false_claim": aggregate,
    }
    store.data["posthoc_seal"] = dump(
        store.output / "seals/posthoc_truth.json",
        {
            "queries": {
                row["query_id"]: row["posthoc"] for row in store.data["queries"].values()
            },
            "joined_after_decision_seal_sha256": store.data["seals"]["decisions"]["sha256"],
        },
    )
    store.save()


def run(
    output,
    cfg=None,
    *,
    input_type=runtime.Inputs,
    framewise_type=runtime.FramewiseBackend,
    wan_type=runtime.WanBackend,
):
    cfg = load_config() if cfg is None else cfg
    store = Store(output, cfg)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    rules = {"status": "NOT_FREEZABLE", "thresholds": {}}
    try:
        store.data["stage"] = "RULE_AND_INPUT_CONTRACT"
        store.save()
        rules, raw_rules = load_frozen_rules(cfg)
        store.data["rules"] = {
            "status": "FROZEN",
            "rule_sha256": rules["rule_sha256"],
            "formula_version": rules["formula_version"],
            "thresholds": rules["thresholds"],
        }
        if raw_rules is None:
            store.data["seals"]["rules"] = dump(store.output / "seals/frozen_rules.json", rules)
        else:
            rule_path = store.output / "seals/frozen_rules.json"
            rule_path.parent.mkdir(parents=True, exist_ok=True)
            rule_path.write_bytes(raw_rules)
            store.data["seals"]["rules"] = {
                "path": str(rule_path),
                "bytes": len(raw_rules),
                "sha256": digest_bytes(raw_rules),
            }
        store.save()

        store.data["stage"] = "BLIND_SYNC"
        store.save()
        collect_sync(store, rules, input_type, framewise_type)
        store.data["stage"] = "BLIND_PAYLOAD"
        store.save()
        collect_payload(store, rules, input_type, wan_type)
        store.data["stage"] = "BLIND_DECISIONS"
        store.save()
        settle_evidence(store, "missing blind evidence")
        decide_all(store, rules)
        seal_decisions(store, rules)
        store.data["stage"] = "POSTHOC_TRUTH"
        store.save()
        join_truth_and_report(store, rules)
        store.data.update(
            status="COMPLETE" if not store.data["failures"] else "RETAINED_INCOMPLETE",
            stage="FINISHED",
        )
        store.save()
        return store.data
    except BaseException as exc:
        store.failure(store.data["stage"], exc)
        interruption = not isinstance(exc, Exception)
        try:
            settle_evidence(store, f"{type(exc).__name__}: {exc}")
            decide_all(store, rules, interruption=interruption)
            seal_decisions(store, rules)
            join_truth_and_report(store, rules)
        except BaseException as settle_error:
            if hasattr(exc, "add_note"):
                exc.add_note("settlement error: " + repr(settle_error))
        store.data.update(status="RETAINED_INCOMPLETE", stage="FINISHED")
        store.save()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.output, load_config(args.config))


if __name__ == "__main__":
    main()
