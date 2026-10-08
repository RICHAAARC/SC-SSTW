"""Fixed development-freeze then unseen-confirmation trajectory attribution run."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from main.tube_state import video_trajectory_attribution_v1 as method
from runtime.wan import video_trajectory_attribution_v1 as runtime
from runtime.wan.provenance import source_identity
from experiments.wan_state_clock import video_trajectory_attribution_v1_prepare as prepare
from experiments.wan_state_clock import video_trajectory_attribution_v1_protocol as protocol

ROOT = Path(__file__).resolve().parents[2]
ENTRY = "experiments.wan_state_clock.video_trajectory_attribution_v1_run"
CONFIG = ROOT / "experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json"


class TerminationRequested(BaseException):
    pass


def _handle_sigterm(signum, frame):
    raise TerminationRequested(f"signal {signum}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load_config(path=CONFIG):
    path = Path(path).resolve()
    cfg = read(path)
    fixed = read(CONFIG)
    if (
        cfg != fixed
        or cfg["formula_version"] != method.RULE_FORMULA_VERSION
        or cfg["fixed_denominator"] != protocol.fixed_denominator(3)
    ):
        raise ValueError("fixed attribution config required")
    cfg["_config_path"] = str(path)
    return cfg


class Store:
    def __init__(self, output, cfg):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.cfg = cfg
        provenance = source_identity(ROOT)
        roster = protocol.query_roster()
        self.data = {
            "status": "RUNNING",
            "stage": "INITIALIZED",
            "source_sha": provenance["git_commit"],
            "source_provenance": provenance,
            "config_sha256": digest(cfg["_config_path"]),
            "source_roster_sha256": protocol.roster_sha256(cfg["sources"]),
            "fixed_denominator": cfg["fixed_denominator"],
            "rules": {"status": "PENDING"},
            "sources": {
                source_id: {"status": "PENDING", "role": cfg["sources"][source_id]["role"]}
                for source_id in protocol.SOURCE_IDS
            },
            "queries": {
                row["query_id"]: dict(
                    row,
                    claim_message=method.CLAIM_MESSAGE,
                    claim_bits=list(method.CLAIM_BITS),
                    status="PENDING",
                    sync=None,
                    identity=None,
                    decision=None,
                )
                for row in roster
            },
            "aggregates": {},
            "seals": {},
            "failures": [],
            "evidence_ceiling": cfg["evidence_ceiling"],
        }
        self.save()

    def save(self):
        prepare.dump(self.output / "result.json", self.data)

    def failure(self, stage, exc, *, observation_id=None, query_id=None):
        row = {"stage": stage, "error": f"{type(exc).__name__}: {exc}"}
        if observation_id is not None:
            row["observation_id"] = observation_id
        if query_id is not None:
            row["query_id"] = query_id
        self.data["failures"].append(row)
        self.save()


def _source_queries(store, source_id):
    return [row for row in store.data["queries"].values() if row["source_id"] == source_id]


def _source_seal_id(source_id):
    return f"S{protocol.SOURCE_IDS.index(source_id):02d}"


def _stop_process_group(child):
    if child is None:
        return
    if os.name == "posix":
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10)


def run_worker(store, source_id, phase):
    out = store.output / "prepared" / source_id
    log = store.output / "workers" / (source_id + "." + phase + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-B",
        "-m",
        ENTRY,
        "--config",
        store.cfg["_config_path"],
        "--output",
        str(out),
        "--worker",
        phase,
        "--source-id",
        source_id,
    ]
    child = None
    primary = None
    primary_tb = None
    try:
        with log.open("wb") as stream:
            child = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            returncode = child.wait()
            if returncode:
                raise RuntimeError(f"{source_id}/{phase} worker exited {returncode}")
    except BaseException as exc:
        primary = exc
        primary_tb = exc.__traceback__
    finally:
        try:
            _stop_process_group(child)
        except BaseException as cleanup_error:
            if primary is None:
                primary = cleanup_error
                primary_tb = cleanup_error.__traceback__
            elif hasattr(primary, "add_note"):
                primary.add_note("worker cleanup error: " + repr(cleanup_error))
    if primary is not None:
        raise primary.with_traceback(primary_tb)
    return read(out / "source_preparation.json")


def prepare_source(store, source_id):
    store.data["sources"][source_id]["status"] = "RUNNING"
    store.save()
    for phase in ("generation", "native_decode", "framewise_media"):
        store.data["stage"] = source_id + "/" + phase
        store.save()
        run_worker(store, source_id, phase)
    record = read(store.output / "prepared" / source_id / "source_preparation.json")
    if record["status"] != "COMPLETE" or set(record["received"]) != set(protocol.VIDEO_IDS):
        raise RuntimeError("source preparation incomplete")
    store.data["sources"][source_id].update(
        status="COMPLETE",
        preparation_receipt=dump(store.output / "receipts" / (source_id + ".json"), record),
    )
    store.save()
    return record


def _observation(runtime_record, video_id, view_id, input_type):
    full = input_type.read(runtime_record["received"][video_id])
    mapping = protocol.VIEW_MAPS[view_id]
    observed = input_type.construct(full, mapping)
    receipt = input_type.receipt(observed)
    return observed, {"frames": len(mapping), **receipt}


def _technical_sync(error):
    return {
        "status": "TECHNICAL_INCOMPLETE",
        "error": str(error),
        "M": None,
        "m": None,
        "top_paths": [],
        "top_action_count": 0,
        "unique_action": False,
        "chosen_action": None,
        "truth_inputs": False,
    }


def _technical_identity(error, attempted):
    return {
        "status": "TECHNICAL_INCOMPLETE",
        "attempted": attempted,
        "error": str(error),
        "I": None,
        "exact32": None,
        "truth_inputs": False,
    }


def collect_source(
    store,
    source_id,
    record,
    rules=None,
    input_type=runtime.Inputs,
    framewise_type=runtime.FramewiseBackend,
    wan_type=runtime.WanBackend,
):
    keys = store.cfg["keys"]
    source_rows = _source_queries(store, source_id)
    by_observation = {}
    for query in source_rows:
        by_observation.setdefault(query["observation_id"], []).append(query)

    sync_rows = {}
    observation_receipts = {}
    framewise = None
    try:
        try:
            framewise = framewise_type(store.cfg)
        except Exception as exc:
            store.failure(source_id + "/SYNC_BACKEND", exc)
            for query in source_rows:
                summary = _technical_sync(exc)
                query.update(status="SYNC_FAILED", sync=summary)
                sync_rows[query["query_id"]] = {
                    "query_id": query["query_id"],
                    "key_id": hashlib.sha256(keys[query["key_label"]].encode()).hexdigest(),
                    "error": str(exc),
                    "summary": summary,
                }
                store.save()
        if framewise is not None:
            for observation_id, queries in by_observation.items():
                anchor = queries[0]
                try:
                    observed, receipt = _observation(
                        record, anchor["video_id"], anchor["view_id"], input_type
                    )
                    latent = framewise.encode(observed)
                    observation_receipts[observation_id] = {
                        "observation_id": observation_id,
                        "frames": receipt["frames"],
                        "rgb_sha256": receipt["sha256"],
                        "rgb_shape": receipt["shape"],
                        "rgb_bytes": receipt["bytes"],
                        "rgb_dtype": receipt["dtype"],
                    }
                except Exception as exc:
                    store.failure(
                        source_id + "/SYNC_OBSERVATION",
                        exc,
                        observation_id=observation_id,
                    )
                    observation_receipts[observation_id] = {
                        "observation_id": observation_id,
                        "frames": anchor["frames"],
                        "status": "TECHNICAL_INCOMPLETE",
                        "error": str(exc),
                    }
                    for query in queries:
                        summary = _technical_sync(exc)
                        query.update(status="SYNC_FAILED", sync=summary)
                        sync_rows[query["query_id"]] = {
                            "query_id": query["query_id"],
                            "key_id": hashlib.sha256(keys[query["key_label"]].encode()).hexdigest(),
                            "error": str(exc),
                            "summary": summary,
                        }
                    store.save()
                    continue
                for query in queries:
                    key = keys[query["key_label"]]
                    try:
                        raw = runtime.score_received(framewise, latent, key, receipt["frames"])
                        raw_receipt = dump(
                            store.output / "blind_sync" / (query["query_id"] + ".json.gz"), raw
                        )
                        summary = method.summarize_sync(raw, receipt["frames"], key)
                        sync_rows[query["query_id"]] = {
                            "query_id": query["query_id"],
                            "key_id": hashlib.sha256(key.encode()).hexdigest(),
                            "raw": raw_receipt,
                            "summary": summary,
                        }
                        query.update(status="SYNC_READ", sync=summary)
                    except Exception as exc:
                        store.failure(
                            source_id + "/SYNC_QUERY",
                            exc,
                            observation_id=observation_id,
                            query_id=query["query_id"],
                        )
                        summary = _technical_sync(exc)
                        sync_rows[query["query_id"]] = {
                            "query_id": query["query_id"],
                            "key_id": hashlib.sha256(key.encode()).hexdigest(),
                            "error": str(exc),
                            "summary": summary,
                        }
                        query.update(status="SYNC_FAILED", sync=summary)
                    store.save()
                observed = latent = None
    finally:
        if framewise is not None:
            framewise.close()

    seal_id = _source_seal_id(source_id)
    sync_seal = dump(
        store.output / "seals" / (seal_id + ".sync.json"),
        {
            "seal_id": seal_id,
            "observations": observation_receipts,
            "queries": sync_rows,
            "truth_inputs": False,
        },
    )
    store.data["seals"][source_id + "/sync"] = sync_seal
    store.save()

    payload_rows = {}
    wan = None
    try:
        try:
            wan = wan_type(store.cfg)
        except Exception as exc:
            store.failure(source_id + "/PAYLOAD_BACKEND", exc)
            for query in source_rows:
                identity = _technical_identity(exc, attempted=True)
                query.update(status="PAYLOAD_UNAVAILABLE", identity=identity)
                payload_rows[query["query_id"]] = {
                    "query_id": query["query_id"],
                    "key_id": hashlib.sha256(keys[query["key_label"]].encode()).hexdigest(),
                    "identity": identity,
                }
                store.save()
        if wan is not None:
            for observation_id, queries in by_observation.items():
                anchor = queries[0]
                try:
                    observed, receipt = _observation(
                        record, anchor["video_id"], anchor["view_id"], input_type
                    )
                except Exception as exc:
                    store.failure(
                        source_id + "/PAYLOAD_OBSERVATION",
                        exc,
                        observation_id=observation_id,
                    )
                    for query in queries:
                        identity = _technical_identity(exc, attempted=True)
                        query.update(status="PAYLOAD_UNAVAILABLE", identity=identity)
                        payload_rows[query["query_id"]] = {
                            "query_id": query["query_id"],
                            "key_id": hashlib.sha256(keys[query["key_label"]].encode()).hexdigest(),
                            "identity": identity,
                        }
                    store.save()
                    continue
                for query in queries:
                    sync = query["sync"] or _technical_sync("missing sync")
                    eligible = sync.get("status") == "COMPLETE" and sync.get("unique_action")
                    if rules is not None and eligible:
                        threshold = rules["thresholds"][str(receipt["frames"])]
                        eligible = sync["M"] > threshold["tau_M"] and (
                            receipt["frames"] == 181 or sync["m"] > threshold["tau_m"]
                        )
                    if not eligible:
                        identity = {
                            "status": "NOT_ELIGIBLE",
                            "attempted": False,
                            "error": "PAYLOAD_NOT_RUN_WITHOUT_QUALIFIED_UNIQUE_ACTION",
                            "I": None,
                            "exact32": None,
                            "truth_inputs": False,
                        }
                    else:
                        key = keys[query["key_label"]]
                        try:
                            corrected = runtime.apply_action(observed, sync["chosen_action"])
                            latent = wan.encode(corrected)
                            detail = wan.read(latent, key, receipt["frames"])
                            detail_receipt = dump(
                                store.output / "blind_payload" / (query["query_id"] + ".json.gz"),
                                detail,
                            )
                            identity = method.identity_evidence(detail)
                            payload_rows[query["query_id"]] = {
                                "query_id": query["query_id"],
                                "key_id": hashlib.sha256(key.encode()).hexdigest(),
                                "detail": detail_receipt,
                                "identity": identity,
                            }
                            corrected = latent = None
                        except Exception as exc:
                            store.failure(
                                source_id + "/PAYLOAD_QUERY",
                                exc,
                                observation_id=observation_id,
                                query_id=query["query_id"],
                            )
                            identity = _technical_identity(exc, attempted=True)
                    query.update(
                        status="PAYLOAD_READ" if identity["status"] == "COMPLETE" else "PAYLOAD_UNAVAILABLE",
                        identity=identity,
                    )
                    payload_rows.setdefault(
                        query["query_id"],
                        {
                            "query_id": query["query_id"],
                            "key_id": hashlib.sha256(
                                keys[query["key_label"]].encode()
                            ).hexdigest(),
                            "identity": identity,
                        },
                    )
                    store.save()
                observed = None
    finally:
        if wan is not None:
            wan.close()

    payload_seal = dump(
        store.output / "seals" / (seal_id + ".payload.json"),
        {
            "seal_id": seal_id,
            "queries": payload_rows,
            "truth_inputs": False,
        },
    )
    store.data["seals"][source_id + "/payload"] = payload_seal
    store.save()


def decision_seal(store, source_id):
    rows = _source_queries(store, source_id)
    if len(rows) != 64 or any(row.get("decision") is None for row in rows):
        raise ValueError("complete 64-row decision roster required before seal")
    seal_id = _source_seal_id(source_id)
    receipt = dump(
        store.output / "seals" / (seal_id + ".decisions.json"),
        {
            "seal_id": seal_id,
            "rule_sha256": store.data["rules"].get("rule_sha256"),
            "decisions": {row["query_id"]: row["decision"] for row in rows},
            "truth_inputs": False,
        },
    )
    store.data["seals"][source_id + "/decisions"] = receipt
    store.save()


def posthoc_roles(store, source_id):
    rows = []
    for query in _source_queries(store, source_id):
        role = protocol.role(query)
        action_map_correct = None
        true_action_available = True
        try:
            true_action = protocol.true_action_for_view(query["view_id"])
        except (KeyError, TypeError, ValueError):
            true_action = None
            true_action_available = False
        sync = query.get("sync") or {}
        if sync.get("chosen_action") is not None and true_action_available:
            action_map_correct = sync["chosen_action"]["received_index_map"] == true_action
        query["posthoc"] = {
            "role": role,
            "action_map_correct": action_map_correct,
            "true_action_available": true_action_available,
            "absolute_source_path_ambiguous": len(sync.get("top_paths", [])) > 1,
        }
        query["posthoc"].update(
            method.posthoc_false_claim(
                query["decision"],
                role,
                action_map_correct,
                true_action_available=true_action_available,
            )
        )
        rows.append(query)

    observation = {}
    for video_id in protocol.VIDEO_IDS:
        for view_id in protocol.VIEW_MAPS:
            group = [
                row
                for row in rows
                if row["video_id"] == video_id and row["view_id"] == view_id
            ]
            observation[video_id + "/" + view_id] = method.aggregate_false_claim(group)
    any_edit = {}
    for video_id in protocol.VIDEO_IDS:
        for key_label in protocol.KEY_LABELS:
            group = [
                row
                for row in rows
                if row["video_id"] == video_id and row["key_label"] == key_label
            ]
            any_edit[video_id + "/" + key_label] = method.aggregate_false_claim(group)
    any_key = {}
    for video_id in protocol.VIDEO_IDS:
        group = [row for row in rows if row["video_id"] == video_id]
        any_key[video_id] = method.aggregate_false_claim(group)
    source_value = method.aggregate_false_claim(rows)
    store.data["aggregates"][source_id] = {
        "observation_any_key": observation,
        "condition_key_any_edit": any_edit,
        "condition_any_key_any_edit": any_key,
        "source_any_false_claim": source_value,
    }
    source_row = store.data["sources"][source_id]
    source_row["any_false_claim"] = source_value
    source_row["decision_counts"] = {
        state: sum(row["decision"]["decision"] == state for row in rows)
        for state in ("ACCEPT", "REJECT", "UNCERTAIN")
    }
    source_row["technical_incomplete_queries"] = sum(
        not row["posthoc"]["technical_complete"] for row in rows
    )
    if source_row.get("status") == "COMPLETE" and source_row["technical_incomplete_queries"]:
        source_row["status"] = "RETAINED_INCOMPLETE"
    store.save()


def settle_source(store, source_id, reason, *, interruption=False):
    decision_reason = (
        "UNCERTAIN_TECHNICAL_INTERRUPTION" if interruption else "UNCERTAIN_TECHNICAL_SOURCE"
    )
    for query in _source_queries(store, source_id):
        if query.get("sync") is None:
            query["sync"] = _technical_sync(reason)
        if query.get("identity") is None:
            query["identity"] = _technical_identity(reason, attempted=False)
        if query.get("decision") is None:
            query["decision"] = {
                "decision": "UNCERTAIN",
                "reason": decision_reason,
                "rule_sha256": store.data["rules"].get("rule_sha256"),
                "frames": query["frames"],
                "claim_message": method.CLAIM_MESSAGE,
            }
        if query["status"] == "PENDING":
            query["status"] = "TECHNICAL_INCOMPLETE"
    store.data["sources"][source_id].update(status="RETAINED_INCOMPLETE", error=reason)
    store.save()


def not_run_sources(store, source_ids, reason):
    for source_id in source_ids:
        store.data["sources"][source_id].update(status="NOT_RUN", reason=reason)
        for query in _source_queries(store, source_id):
            if query.get("sync") is None:
                query["sync"] = _technical_sync(reason)
            if query.get("identity") is None:
                query["identity"] = _technical_identity(reason, attempted=False)
            if query.get("decision") is None:
                query["decision"] = {
                    "decision": "UNCERTAIN",
                    "reason": reason,
                    "rule_sha256": store.data["rules"].get("rule_sha256"),
                    "frames": query["frames"],
                    "claim_message": method.CLAIM_MESSAGE,
                }
            query["status"] = "NOT_RUN"
        decision_seal(store, source_id)
        posthoc_roles(store, source_id)
    store.save()


def _decide_source(store, source_id, rules):
    for query in _source_queries(store, source_id):
        query["decision"] = method.decide(query["sync"], query["identity"], query["frames"], rules)
    store.save()


def _settle_and_seal(store, source_id, reason, *, interruption=False):
    settle_source(store, source_id, reason, interruption=interruption)
    decision_seal(store, source_id)
    posthoc_roles(store, source_id)


def run(output, cfg=None):
    cfg = load_config() if cfg is None else cfg
    store = Store(output, cfg)
    current_source = "DEV"
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        store.data["stage"] = "DEV_PREPARATION"
        store.save()
        dev = prepare_source(store, "DEV")
        store.data["stage"] = "DEV_BLIND"
        store.save()
        collect_source(store, "DEV", dev, rules=None)

        development = _source_queries(store, "DEV")
        labels = {
            query["query_id"]: protocol.calibration_label(query) for query in development
        }
        rules = method.freeze_rules(
            development,
            labels,
            store.data["source_roster_sha256"],
            store.data["config_sha256"],
        )
        store.data["rules"] = rules
        store.data["seals"]["rules"] = dump(
            store.output / "seals" / "frozen_rules.json", rules
        )
        store.save()
        _decide_source(store, "DEV", rules)
        decision_seal(store, "DEV")
        posthoc_roles(store, "DEV")

        if rules["status"] != "FROZEN":
            not_run_sources(store, ("C1", "C2"), "UNCERTAIN_RULE_NOT_FREEZABLE")
            store.data.update(status="RETAINED_INCOMPLETE", stage="FINISHED")
            store.save()
            return store.data

        for source_id in ("C1", "C2"):
            current_source = source_id
            try:
                store.data["stage"] = source_id + "_PREPARATION"
                store.save()
                record = prepare_source(store, source_id)
                store.data["stage"] = source_id + "_BLIND"
                store.save()
                collect_source(store, source_id, record, rules=rules)
                _decide_source(store, source_id, rules)
                decision_seal(store, source_id)
                posthoc_roles(store, source_id)
            except BaseException as exc:
                store.failure(store.data["stage"], exc)
                _settle_and_seal(
                    store,
                    source_id,
                    f"{type(exc).__name__}: {exc}",
                    interruption=not isinstance(exc, Exception),
                )
                if not isinstance(exc, Exception):
                    remaining = tuple(
                        candidate
                        for candidate in ("C1", "C2")
                        if protocol.SOURCE_IDS.index(candidate)
                        > protocol.SOURCE_IDS.index(source_id)
                    )
                    not_run_sources(
                        store, remaining, "UNCERTAIN_TECHNICAL_INTERRUPTION"
                    )
                    store.data.update(status="RETAINED_INCOMPLETE", stage="FINISHED")
                    store.save()
                    raise
        store.data.update(
            status="COMPLETE" if not store.data["failures"] else "RETAINED_INCOMPLETE",
            stage="FINISHED",
        )
        store.save()
        return store.data
    except BaseException as exc:
        if store.data["stage"] != "FINISHED":
            store.failure(store.data["stage"], exc)
            if source_id := current_source:
                if source_id + "/decisions" not in store.data["seals"]:
                    _settle_and_seal(
                        store,
                        source_id,
                        f"{type(exc).__name__}: {exc}",
                        interruption=not isinstance(exc, Exception),
                    )
                elif source_id not in store.data["aggregates"]:
                    posthoc_roles(store, source_id)
            remaining = tuple(
                candidate
                for candidate in ("C1", "C2")
                if candidate + "/decisions" not in store.data["seals"]
            )
            if remaining:
                not_run_sources(
                    store,
                    remaining,
                    "UNCERTAIN_TECHNICAL_INTERRUPTION"
                    if not isinstance(exc, Exception)
                    else "UNCERTAIN_TECHNICAL_SOURCE",
                )
            store.data.update(status="RETAINED_INCOMPLETE", stage="FINISHED")
            store.save()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--worker", choices=("generation", "native_decode", "framewise_media"))
    parser.add_argument("--source-id", choices=protocol.SOURCE_IDS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.worker:
        if args.source_id is None:
            raise ValueError("worker source-id required")
        prepare.run_phase(args.output, args.worker, cfg, args.source_id)
    else:
        run(args.output, cfg)


if __name__ == "__main__":
    main()
