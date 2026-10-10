"""Fixed three-receiver payload follow-up on saved M05 observations only.

Selection sees received scores only. Media/model work is explicit in read(), never
in prepare/report. Historical inputs are read-only. Completed physical attempts
are not repeated when read evidence was lost; the corresponding rows stay failed.
"""
from __future__ import annotations

from collections import Counter
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import time

from experiments.paper_results_v1.receiver_controls_cli import ATTACKS

RECEIVERS = ("RAW_U", "CENTERED_U", "CENTERED_D4")
CASES = ("pilot_01", "pilot_02")
ARM = "PAYLOAD_FRAMEWISE_M05"
SCHEMA = "receiver-payload-followup-v1"
COHORT = "PILOT_EXCLUDED_FROM_CONFIRMATION"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def reason(exc):
    return f"{type(exc).__name__}: {exc}"


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / "followup_state.json"
        self.data = read_json(self.path)
        if self.data["schema"] != SCHEMA or len(self.data["rows"]) != 90:
            raise ValueError("follow-up schema or fixed denominator changed")

    def save(self):
        atomic_json(self.path, self.data)

    def resolve(self, value, document=None):
        """Read ordinary locators; support relocating the saved run without Git."""
        path = Path(value)
        if path.is_absolute() and path.is_file():
            return path
        if document is not None:
            candidate = Path(document).parent / (path.name if path.is_absolute() else path)
            if candidate.is_file():
                return candidate
        source = Path(self.data["source_state"]).parent
        if "/run_state/" in str(path):
            candidate = source / str(path).split("/run_state/", 1)[1]
            if candidate.is_file():
                return candidate
        candidate = source / path
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"saved evidence unavailable: {path}")


def initialize(source_run, output, config, temp_root=None):
    output = Path(output).resolve()
    if (output / "followup_state.json").is_file():
        return Store(output)
    source_run = Path(source_run).resolve()
    source_state = source_run / "run_state/attack_run_state.json"
    if not source_state.is_file():
        source_state = source_run / "attack_run_state.json"
    source = read_json(source_state)
    cfg = read_json(config)
    original_config = source_state.parent / "effective_attack_config.json"
    if not original_config.is_file():
        original_config = source_state.parent.parent / "effective_attack_config.json"
    source_cfg = read_json(original_config)
    if cfg["keys"]["K0"] != source_cfg["keys"]["K0"]:
        raise ValueError("follow-up must reuse the saved run's K0 key")
    if not isinstance(cfg["keys"]["K0"], str) or not cfg["keys"]["K0"]:
        raise ValueError("source K0 is required")
    if len(cfg["payload_bits"]) != 32 or any(type(x) is not int or x not in (0, 1) for x in cfg["payload_bits"]):
        raise ValueError("source payload must be 32 bits")
    rows = [dict(slot_id=f"{case}/{attack}/{receiver}", case_id=case, attack_id=attack,
                 arm=ARM, key_label="K0", receiver=receiver, cohort=COHORT,
                 status="NOT_EXECUTED", reason="not_prepared", planned_bits=32, missing_bits=32)
            for case in CASES for attack in ATTACKS for receiver in RECEIVERS]
    state = dict(schema=SCHEMA, status="INITIALIZED", source_state=str(source_state),
                 source_config=str(original_config.resolve()), key=cfg["keys"]["K0"],
                 wan=cfg["models"]["wan"], temp_root=str(Path(temp_root or output / "temp_rgb").resolve()),
                 fixed_denominator=dict(cases=list(CASES), conditions=30, receivers=list(RECEIVERS),
                     rows=90, planned_bits=2880, cohort=COHORT, confirmation_cases=8,
                     confirmation_status="NOT_EXECUTED", RAW_D4="PRIOR_CPU_PATH_CONTROL_ONLY"),
                 rows=rows, physical_inputs=[], media_decodes=[], vae_loads=[], errors=[], phases={},
                 source_history=dict(status=source.get("status"), recovery=source.get("recovery", {}).get("source_run_id")),
                 actual_other_model_calls=0, media_encode_calls=0)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "followup_state.json", state)
    store = Store(output); report(store)
    return store


def source_rows(source, case, attack):
    return [r for r in source["receiver_rows"] if (r["case_id"], r["arm"], r["attack_id"], r["key_label"])
            == (case, ARM, attack, "K0")]


def same_input(row, operation):
    return row.get("operation", {}).get("received_index_map") == operation["received_index_map"]


def reannotate_detail(detail, operation):
    result = copy.deepcopy(detail)
    result.pop("signed_votes", None); result.pop("zero_mask", None)
    for row in result["time_bit_rows"]:
        stride = 4 * row["receiver_latent_index"]
        if row["output_stride_coordinate"] != stride:
            raise ValueError("old per-time row stride differs from reader contract")
        row["estimated_source_coordinate"] = operation["source_coordinate_map"][stride]
    result["operation"] = copy.deepcopy(operation)
    result["annotation_policy"] = "copy_on_write_current_operation_not_source_receptive_field"
    return result


def validate_detail(detail, sidecar, operation):
    import numpy as np
    if len(detail["decoded_bits"]) != 32 or any(x not in (0, 1) for x in detail["decoded_bits"]):
        raise ValueError("saved read has incomplete decoded bits")
    if detail["R"] != operation["wan_support"] or detail["output_frames"] != operation["output_frames"]:
        raise ValueError("saved read shape differs from input map")
    if len(detail["time_bit_rows"]) != 32 * detail["R"]:
        raise ValueError("saved per-time counts are incomplete")
    with np.load(sidecar, allow_pickle=False) as arrays:
        signs, zeros = arrays["signed_votes"], arrays["zero_mask"]
        if signs.ndim != 3 or signs.shape[:2] != (4, detail["R"]) or signs.shape != zeros.shape:
            raise ValueError("saved vote sidecar geometry mismatch")
        if not np.isin(signs, [-1, 0, 1]).all() or not np.array_equal(signs == 0, zeros):
            raise ValueError("saved vote sign/zero inconsistency")


def adopt_detail(store, row, detail_path, *, reused=False, source_slot=None):
    detail_path = Path(detail_path); detail = read_json(detail_path)
    sidecar = store.resolve(detail["vote_sidecar"]["path"], detail_path)
    validate_detail(detail, sidecar, row["operation"])
    if reused:
        target = store.output / "reused_votes" / (source_slot.replace("/", "_") + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            temporary = target.with_suffix(".npz.tmp"); shutil.copyfile(sidecar, temporary); os.replace(temporary, target)
        sidecar = target
    result = reannotate_detail(detail, row["operation"])
    result["vote_sidecar"]["path"] = str(sidecar)
    result["evidence_origin"] = dict(kind="REUSED_OLD_READ" if reused else "NEW_PHYSICAL_READ",
                                     source_detail=str(detail_path), source_slot=source_slot)
    target = store.output / "reads" / row["case_id"] / row["attack_id"] / (row["receiver"] + ".json")
    atomic_json(target, result)
    row.update(status="READ", reason=None, missing_bits=0, decoded_bits=result["decoded_bits"],
               votes=result["votes"], R=result["R"], detail_record=str(target), reused=reused,
               source_slot=source_slot, original_reader_match=result.get("original_reader_match"))
    store.save()


def _load_evidence(store, old):
    import numpy as np
    clock = store.resolve(old["clock_record"]["path"])
    document = read_json(clock)
    path = store.resolve(document["evidence"]["path"], clock)
    with np.load(path, allow_pickle=False) as arrays:
        numerator, rho = arrays["signed_projection"].copy(), arrays["rho"].copy()
    return numerator, rho, str(path)


def prepare(store):
    from main.tube_state.receiver_controls_v1 import receive_scores
    source = read_json(store.data["source_state"])
    store.data["phases"]["prepare"] = "RUNNING"; store.save()
    try:
        for case in CASES:
            for attack in ATTACKS:
                rows = [r for r in store.data["rows"] if r["case_id"] == case and r["attack_id"] == attack]
                pending = [r for r in rows if r["status"] == "NOT_EXECUTED"]
                if not pending:
                    continue
                try:
                    olds = source_rows(source, case, attack)
                    blind = next(r for r in olds if r["mode"] == "BLIND_PATH")
                    num, rho, score_source = _load_evidence(store, blind)
                    artifact_id = f"{case}/{ARM}/{attack}/RECEIVED"
                    artifact = next(a for a in source["artifacts"] if a["artifact_id"] == artifact_id)
                    # All selectors are sealed before payload/message/truth evaluation.
                    for row in pending:
                        selected = receive_scores(num, rho, row["receiver"])
                        target = store.output / "selectors" / case / attack / (row["receiver"] + ".json")
                        atomic_json(target, selected)
                        row.update(selector_record=str(target), score_source=score_source,
                                   operation=selected["operation"], observation=copy.deepcopy(artifact),
                                   received_frames=len(num), status="SELECTED", reason=None)
                        if selected["status"] != "SUPPORTED":
                            row.update(status=selected["status"], reason=selected["reason"])
                        store.save()
                except Exception as exc:
                    for row in pending:
                        if row["status"] == "NOT_EXECUTED":
                            row.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
                    store.save()
                # SELECTED is durable, so re-entry can finish reuse/planning after interruption.
        for row in store.data["rows"]:
            if row["status"] != "SELECTED":
                continue
            try:
                olds = source_rows(source, row["case_id"], row["attack_id"])
                matches = [old for old in olds if old["status"] == "EVALUATED" and same_input(old, row["operation"])]
                if matches:
                    old = next((x for x in matches if x["mode"] == "BLIND_PATH"), matches[0])
                    row["reuse_candidate"] = old["slot_id"]; store.save()
                    adopt_detail(store, row, store.resolve(old["detail_record"]["path"]),
                                 reused=True, source_slot=old["slot_id"])
                else:
                    if row["receiver"] == "RAW_U":
                        raise ValueError("original U has no matching evaluated reference; do not replace old failure")
                    row.update(status="PENDING_READ", reason="new_corrected_input")
            except Exception as exc:
                row.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
            store.save()
        for row in store.data["rows"]:
            if row["status"] != "PENDING_READ" or row.get("physical_id"):
                continue
            operation = row["operation"]; artifact_id = row["observation"]["artifact_id"]
            group = next((g for g in store.data["physical_inputs"] if g["artifact_id"] == artifact_id
                          and g["received_index_map"] == operation["received_index_map"]), None)
            if group is None:
                group = dict(physical_id=f"map_{len(store.data['physical_inputs']):03d}", artifact_id=artifact_id,
                             received_index_map=operation["received_index_map"], output_frames=operation["output_frames"],
                             status="PLANNED", logical_slots=[])
                store.data["physical_inputs"].append(group)
            if row["slot_id"] not in group["logical_slots"]:
                group["logical_slots"].append(row["slot_id"])
            row["physical_id"] = group["physical_id"]; store.save()
        store.data["phases"]["prepare"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["prepare"] = "INTERRUPTED"
        store.data["errors"].append(reason(exc)); raise
    finally:
        store.save(); report(store)
    return store


def _group_rows(store, group):
    return [r for r in store.data["rows"] if r.get("physical_id") == group["physical_id"]]


def _restore_group(store, group):
    detail = store.output / "physical" / group["physical_id"] / "read.json"
    if not detail.is_file():
        return False
    for row in _group_rows(store, group):
        if row["status"] not in ("READ", "EVALUATED"):
            try:
                adopt_detail(store, row, detail)
            except Exception as exc:
                row.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
                store.save()
    group["read_status"] = "DURABLE"; store.save()
    return True


def record_external_failure(store, why):
    recovered_failure = False
    for group in store.data["physical_inputs"]:
        if _restore_group(store, group):
            continue
        if group["status"] == "RUNNING":
            group.update(status="INTERRUPTED", reason=why, finished_at=time.time())
        if "started_at" in group and group["status"] in ("INTERRUPTED", "COMPLETE", "FAILED"):
            recovered_failure = True
            for row in _group_rows(store, group):
                if row["status"] not in ("READ", "EVALUATED"):
                    row.update(status="FAILED", reason="PRIOR_ATTEMPT_NO_DURABLE_READ: " + why)
    for call in store.data["media_decodes"]:
        if call["status"] == "RUNNING":
            recovered_failure = True
            call.update(status="INTERRUPTED", reason=why, finished_at=time.time())
    for call in store.data.setdefault("vae_loads", []):
        if call["status"] == "RUNNING":
            recovered_failure = True
            call.update(status="INTERRUPTED", reason=why, finished_at=time.time())
    if recovered_failure and why not in store.data["errors"]:
        store.data["errors"].append(why)
    store.save(); report(store)


def _received_rgb(store, row):
    import numpy as np
    import torch
    from runtime.wan.variable_rgb_media import decode_published
    artifact = row["observation"]; shape = artifact["shape"]
    cache = Path(store.data["temp_root"]) / row["case_id"] / row["attack_id"] / "received.rgb8"
    source_path = Path(artifact.get("path", ""))
    path = cache if cache.is_file() else source_path
    if path.is_file():
        data = np.fromfile(path, dtype=np.uint8)
        if data.size != int(np.prod(shape)):
            raise ValueError("saved RGB byte count differs from recorded observation")
        return torch.from_numpy(data.reshape(shape).copy())
    if artifact.get("source_mp4"):
        mp4 = Path(artifact["source_mp4"])
    else:
        source = read_json(store.data["source_state"])
        original = Path(source["recovery"]["source_run"])
        mp4 = original / "run_state/media" / row["case_id"] / ARM / row["attack_id"] / "published.mp4"
    call = dict(artifact_id=artifact["artifact_id"], source_mp4=str(mp4), status="RUNNING", started_at=time.time())
    store.data["media_decodes"].append(call); store.save()
    try:
        data, receipt = decode_published(mp4, cache, expected_frames=shape[0])
        call.update(status="COMPLETE", finished_at=time.time(), receipt=receipt); store.save()
        return torch.from_numpy(data)
    except BaseException as exc:
        call.update(status="FAILED", reason=reason(exc), finished_at=time.time()); store.save(); raise


def _save_shared_read(store, group, result):
    import numpy as np
    root = store.output / "physical" / group["physical_id"]; root.mkdir(parents=True, exist_ok=True)
    sidecar = root / "votes.npz"; temporary = sidecar.with_suffix(".npz.tmp")
    result = dict(result)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, signed_votes=result.pop("signed_votes"), zero_mask=result.pop("zero_mask"))
    os.replace(temporary, sidecar)
    result["vote_sidecar"] = dict(path=str(sidecar), arrays=["signed_votes", "zero_mask"],
                                  coordinate_order="[payload_channel,receiver_latent,frequency_coordinate]")
    atomic_json(root / "read.json", result)
    group["read_status"] = "DURABLE"; store.save()


def read(store, *, load_vae=None, encode=None, payload_read=None, load_rgb=None):
    # Injection points serve CPU engineering tests, never config-selectable model replacements.
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan.vae import reencode_rgb24_readback
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map, read_payload_general
    load_vae = load_vae or (lambda: load_frozen_vae({"model": {"id": store.data["wan"]["local_snapshot_path"]}},
                                                 device=store.data["wan"].get("device", "cuda")))
    encode = encode or reencode_rgb24_readback
    payload_read = payload_read or read_payload_general
    load_rgb = load_rgb or (lambda row: _received_rgb(store, row))
    record_external_failure(store, "read re-entry; retain prior unfinished model attempts")
    vae = None; store.data["phases"]["read"] = "RUNNING"; store.save()
    try:
        todo = [g for g in store.data["physical_inputs"] if g["status"] == "PLANNED"]
        if todo:
            load_call = dict(status="RUNNING", started_at=time.time())
            store.data.setdefault("vae_loads", []).append(load_call); store.save()
            try:
                vae = load_vae()
            except BaseException as exc:
                load_call.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                 finished_at=time.time(), reason=reason(exc)); store.save(); raise
            load_call.update(status="COMPLETE", finished_at=time.time()); store.save()
        cached_artifact = None; received = None
        for group in todo:
            row = _group_rows(store, group)[0]
            encoded = normalized = corrected = result = None
            try:
                if cached_artifact != group["artifact_id"]:
                    received = load_rgb(row); cached_artifact = group["artifact_id"]
                corrected = operate_map(received, group["received_index_map"]).float().div(255.0)
                # Preprocessing failures do not count as model invocations.
                group.update(status="RUNNING", started_at=time.time()); store.save()
                try:
                    encoded = encode(vae, corrected)
                except BaseException as exc:
                    group.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                 reason=reason(exc), finished_at=time.time()); store.save(); raise
                # Model return is recorded before any CPU transfer or evidence writing.
                group.update(status="COMPLETE", finished_at=time.time()); store.save()
                normalized = encoded.detach().cpu()
                result = payload_read(normalized, store.data["key"], row["operation"]["output_frames"],
                                      row["operation"]["source_coordinate_map"])
                _save_shared_read(store, group, result)
                _restore_group(store, group)
            except Exception as exc:
                group["read_status"] = "FAILED"; group["read_reason"] = reason(exc)
                if group["status"] == "PLANNED":
                    group.update(status="PREPROCESS_FAILED", reason=reason(exc))
                for dependent in _group_rows(store, group):
                    if dependent["status"] not in ("READ", "EVALUATED"):
                        dependent.update(status="FAILED", reason=reason(exc))
                store.save()
            finally:
                encoded = normalized = corrected = result = None
        store.data["phases"]["read"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["read"] = "INTERRUPTED" if not isinstance(exc, Exception) else "FAILED"
        for group in store.data["physical_inputs"]:
            if group["status"] == "PLANNED":
                for row in _group_rows(store, group):
                    row.update(reason="remaining_model_work_not_attempted: " + reason(exc))
        record_external_failure(store, reason(exc)); raise
    finally:
        vae = None
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        store.save(); report(store)
    return store


def report(store):
    # This is the only message/truth join; selection records are already durable.
    from experiments.paper_results_v1.receiver_controls_cli import truth_for_report, evaluate_path
    try:
        bits = read_json(store.data["source_config"])["payload_bits"]
    except Exception as exc:
        bits = None
        store.data["report_reason"] = reason(exc)
    compact = []
    for row in store.data["rows"]:
        if row["status"] in ("READ", "EVALUATED") and bits is not None:
            row.update(status="EVALUATED", bit_errors=sum(a != b for a, b in zip(bits, row["decoded_bits"])),
                       exact_recovery=row["decoded_bits"] == bits, missing_bits=0)
        metrics = {}
        if row.get("selector_record"):
            try:
                selected = read_json(row["selector_record"]); path = selected["estimate"]["path"]
                if path is not None:
                    metrics = evaluate_path(path, truth_for_report(row["attack_id"]))
                else:
                    metrics = dict(path_evaluation_status=selected["estimate"]["status"])
            except Exception as exc:
                metrics = dict(path_evaluation_status="MISSING", path_evaluation_reason=reason(exc))
            row["path_metrics"] = {k:v for k,v in metrics.items() if k != "frame_rows"}
        c = {k:row.get(k) for k in ("slot_id", "case_id", "attack_id", "receiver", "cohort", "status", "reason",
                                    "reused", "source_slot", "physical_id", "R", "missing_bits", "bit_errors", "exact_recovery")}
        c.update(row.get("path_metrics", {})); c["status"] = row["status"]
        compact.append(c)
    actual = [g for g in store.data["physical_inputs"] if "started_at" in g]
    by_receiver = {}
    for receiver in RECEIVERS:
        rows = [r for r in store.data["rows"] if r["receiver"] == receiver]
        scored = [r for r in rows if r["status"] == "EVALUATED"]
        by_receiver[receiver] = dict(planned_rows=30, status_counts=dict(Counter(r["status"] for r in rows)),
            evaluated_rows=len(scored), decoded_bits=len(scored)*32, bit_errors=sum(r["bit_errors"] for r in scored),
            exact_recovery_rows=sum(r["exact_recovery"] for r in scored), missing_bits=sum(r["missing_bits"] for r in rows))
    summary = dict(schema=SCHEMA, fixed_denominator=store.data["fixed_denominator"], by_receiver=by_receiver,
                   planned_new_physical_inputs=len(store.data["physical_inputs"]),
                   reused_rows=sum(bool(r.get("reused")) for r in store.data["rows"]),
                   actual_wan_calls=dict(attempted=len(actual), statuses=dict(Counter(g["status"] for g in actual))),
                   actual_vae_loads=dict(attempted=len(store.data.get("vae_loads", [])),
                                        statuses=dict(Counter(x["status"] for x in store.data.get("vae_loads", [])))),
                   actual_media_decodes=dict(attempted=len(store.data["media_decodes"]),
                                            statuses=dict(Counter(x["status"] for x in store.data["media_decodes"]))),
                   other_model_calls=0, media_encode_calls=0, evidence_level="SAME_BATCH_DEVELOPMENT_NOT_CONFIRMATION")
    store.save(); atomic_json(store.output / "summary.json", summary)
    atomic_json(store.output / "evaluation_report.json", dict(summary=summary, rows=store.data["rows"]))
    fields = list(dict.fromkeys(k for r in compact for k in r))
    with (store.output / "receiver_payload_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(compact)
    return summary
