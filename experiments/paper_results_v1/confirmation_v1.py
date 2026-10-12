"""Fixed eight-source confirmation receiver over the adopted attack run.

The temporal score evidence is produced by ``attack_eval.receiver-clock``.
This module freezes RAW_U, CENTERED_U and CENTERED_D4 before payload/truth
evaluation, shares only the VAE latent for identical received-media maps, and
keeps key-specific FFT/vote/bit evidence separate.
"""
from __future__ import annotations

import copy
import csv
import gc
import json
import os
from collections import Counter
from pathlib import Path
import time

from experiments.paper_results_v1 import attack_eval
from experiments.paper_results_v1.resume_state import StateJournal, read_state, remember, unit_attempt


SCHEMA = "paper-results-v1-confirmation-receiver-v1"
RECEIVERS = ("RAW_U", "CENTERED_U", "CENTERED_D4")
CASES = attack_eval.CONFIRMATION_CASES
ALL_CASES = attack_eval.ALL_CASES
ARMS = attack_eval.MAIN_ARMS
KEYS = attack_eval.KEY_LABELS


def _reason(exc):
    return f"{type(exc).__name__}: {exc}"


def _atomic_json(path, value):
    return attack_eval.atomic_json(path, value)


def _read(path):
    return attack_eval.read_json(path)


def validate_config(config):
    config = attack_eval.validate_config(config)
    attempted, not_executed = attack_eval._execution_cases(config)
    if attempted != CASES or not_executed != attack_eval.PILOT_CASES:
        raise ValueError("confirmation V1 requires the fixed eight-source execution scope")
    receiver = config["receiver"]
    if tuple(receiver.get("confirmation_receivers", ())) != RECEIVERS:
        raise ValueError("fixed confirmation receiver roster changed")
    if receiver.get("preferred") != "CENTERED_U":
        raise ValueError("CENTERED_U must remain the fixed preferred receiver")
    if receiver.get("raw_control") != "RAW_U" or receiver.get("duration_control") != "CENTERED_D4":
        raise ValueError("fixed receiver controls changed")
    return config


def build_plan(config):
    config = validate_config(config)
    attacks = [row["attack_id"] for row in config["attacks"]]
    rows = []
    for case in config["cases"]:
        case_id = case["case_id"]
        initial = "PLANNED" if case_id in CASES else "NOT_EXECUTED_BY_NOTEBOOK"
        for arm in ARMS:
            for attack_id in attacks:
                for key_label in KEYS:
                    for receiver in RECEIVERS:
                        rows.append({
                            "slot_id": f"{case_id}/{arm}/{attack_id}/{key_label}/{receiver}",
                            "case_id": case_id, "cohort": case["cohort"], "arm": arm,
                            "attack_id": attack_id, "key_label": key_label,
                            "receiver": receiver, "status": initial,
                            "planned_bits": 32, "missing_bits": 32,
                        })
    return {
        "rows": rows,
        "fixed_denominator": {
            "confirmation_sources": 8,
            "attacks_per_source": 15,
            "main_arms": 4,
            "keys": 2,
            "receivers": 3,
            "confirmation_rows": 2880,
            "confirmation_bits": 92160,
            "preferred_m05_k0_rows": 120,
            "preferred_m05_k0_nonfull_rows": 112,
            "off_rows": 720,
            "wrong_key_rows": 1440,
            "off_wrong_key_overlap_rows": 360,
            "physical_wan_encode_unconditional_range": [0, 2880],
        },
    }


class ConfirmationStore:
    def __init__(self, output, config, *, create=False, attack_output=None, temp_root=None):
        self.output = Path(output).resolve()
        self.path = self.output / "confirmation_state.json"
        self.journal = StateJournal(self.path)
        self.config = validate_config(config)
        self.temp_root = Path(temp_root).resolve() if temp_root else self.output / "temporary_rgb"
        from experiments.paper_results_v1.resume_state import state_exists
        if create and state_exists(self.path):
            create = False
        if create:
            if attack_output is None:
                raise ValueError("attack_output is required for initialization")
            attack_path = Path(attack_output).resolve() / "attack_run_state.json"
            if not state_exists(attack_path):
                raise FileNotFoundError(f"attack state is unavailable: {attack_path}")
            self.output.mkdir(parents=True, exist_ok=True)
            plan = build_plan(config)
            self.data = {
                "schema": SCHEMA, "status": "INITIALIZED",
                "attack_state": str(attack_path), "temp_root": str(self.temp_root),
                "rows": plan["rows"], "fixed_denominator": plan["fixed_denominator"],
                "physical_inputs": [], "vae_loads": [], "media_decodes": [],
                "phases": {
                    name: {case: {"status": "PLANNED", "failures": []} for case in CASES}
                    for name in ("prepare", "read")
                },
                "errors": [],
            }
            self.save()
        else:
            self.data = read_state(self.path)
            self.journal.seed_cache(self.data)
            if temp_root is not None:
                self.data["temp_root"] = str(self.temp_root)

    def save(self):
        self.journal.save(self.data)

    def rows(self, case_id=None, **match):
        return [row for row in self.data["rows"] if (case_id is None or row["case_id"] == case_id)
                and all(row.get(key) == value for key, value in match.items())]

    def resolve(self, candidate, record=None):
        path = Path(candidate)
        if path.is_file():
            return path
        if record is not None and not path.is_absolute():
            relative = Path(record).parent / path
            if relative.is_file():
                return relative
        attack_root = Path(self.data["attack_state"]).parent
        text = str(path)
        if "/run_state/" in text:
            relocated = attack_root / text.split("/run_state/", 1)[1]
            if relocated.is_file():
                return relocated
        relocated = attack_root / path
        return relocated if relocated.is_file() else path


def initialize(attack_output, output, config, *, temp_root=None):
    return ConfirmationStore(output, config, create=True, attack_output=attack_output, temp_root=temp_root)


def _attack_state(store):
    return read_state(store.data["attack_state"])


def _attack_row(source, case_id, arm, attack_id, key_label):
    return next(row for row in source["receiver_rows"] if (
        row["case_id"], row["arm"], row["attack_id"], row["key_label"], row["mode"]
    ) == (case_id, arm, attack_id, key_label, "BLIND_PATH"))


def _artifact(source, artifact_id):
    return next(row for row in source["artifacts"] if row["artifact_id"] == artifact_id)


def _load_clock_evidence(store, source_row):
    import numpy as np
    record = store.resolve(source_row["clock_record"]["path"])
    document = _read(record)
    sidecar = store.resolve(document["evidence"]["path"], record)
    with np.load(sidecar, allow_pickle=False) as arrays:
        numerator = arrays["signed_projection"].copy()
        rho = arrays["rho"].copy()
    return numerator, rho, str(sidecar)


def _group_for(store, artifact_id, operation):
    mapping = operation["received_index_map"]
    group = next((row for row in store.data["physical_inputs"]
                  if row["artifact_id"] == artifact_id and row["received_index_map"] == mapping), None)
    if group is None:
        occupied = {row["physical_id"] for row in store.data["physical_inputs"]}
        number = len(occupied)
        while f"map_{number:04d}" in occupied:
            number += 1
        group = {
            "physical_id": f"map_{number:04d}",
            "artifact_id": artifact_id,
            "received_index_map": mapping,
            "output_frames": operation["output_frames"],
            "status": "PLANNED", "logical_slots": [], "key_reads": {},
        }
        store.data["physical_inputs"].append(group)
    return group


def prepare_case(store, case_id):
    with store.journal.periodic():
        return _prepare_case(store, case_id)


def _prepare_case(store, case_id):
    from main.tube_state.receiver_controls_v1 import receive_scores

    phase = store.data["phases"]["prepare"][case_id]
    for row in store.rows(case_id):
        if row["status"] in ("FAILED", "MISSING") and not row.get("operation"):
            remember(row, "retry missing selector evidence")
            row.update(status="PLANNED", reason=None)
            phase["status"] = "PARTIAL"
    if phase["status"] == "COMPLETE":
        return store
    phase["status"] = "RUNNING"; store.save()
    source = _attack_state(store)
    try:
        for arm in ARMS:
            for attack in store.config["attacks"]:
                attack_id = attack["attack_id"]
                artifact_id = f"{case_id}/{arm}/{attack_id}/RECEIVED"
                artifact = _artifact(source, artifact_id)
                for key_label in KEYS:
                    source_row = _attack_row(source, case_id, arm, attack_id, key_label)
                    pending = store.rows(case_id, arm=arm, attack_id=attack_id, key_label=key_label)
                    pending = [row for row in pending if row["status"] == "PLANNED"]
                    if not pending:
                        continue
                    try:
                        numerator, rho, evidence_path = _load_clock_evidence(store, source_row)
                        for row in pending:
                            selected = receive_scores(numerator, rho, row["receiver"])
                            selector = store.output / "selectors" / case_id / arm / attack_id / key_label / (row["receiver"] + ".json")
                            row.update(
                                selector_record=_atomic_json(selector, selected),
                                score_evidence=evidence_path,
                                operation=selected["operation"],
                                observation=copy.deepcopy(artifact),
                                clock_status=selected["estimate"]["status"],
                                reason=selected.get("reason"), truth_inputs=False,
                            )
                            if selected["status"] != "SUPPORTED":
                                row.update(status=selected["status"], missing_bits=32)
                                store.save(); continue
                            group = _group_for(store, artifact_id, selected["operation"])
                            if row["slot_id"] not in group["logical_slots"]:
                                group["logical_slots"].append(row["slot_id"])
                            row.update(status="PENDING_READ", physical_id=group["physical_id"])
                            store.save()
                    except Exception as exc:
                        status = "MISSING" if isinstance(exc, FileNotFoundError) else "FAILED"
                        for row in pending:
                            if row["status"] == "PLANNED":
                                row.update(status=status, reason=_reason(exc), missing_bits=32)
                        store.save()
        phase["status"] = "PARTIAL" if any(row["status"] in ("FAILED", "MISSING", "PLANNED") and not row.get("operation") for row in store.rows(case_id)) else "COMPLETE"
    except BaseException as exc:
        phase["status"] = "INTERRUPTED" if not isinstance(exc, Exception) else "FAILED"
        phase["failures"].append(_reason(exc)); raise
    finally:
        store.save()
        store.journal.checkpoint("prepare-exit")
    return store


def _rows_for_group(store, group, key_label=None):
    slots = set(group["logical_slots"])
    return [row for row in store.data["rows"] if row["slot_id"] in slots
            and (key_label is None or row["key_label"] == key_label)]


def _save_key_read(store, group, key_label, result):
    import numpy as np
    root = store.output / "physical" / group["physical_id"] / key_label
    root.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    sidecar = root / "votes.npz"
    temporary = sidecar.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, signed_votes=result.pop("signed_votes"), zero_mask=result.pop("zero_mask"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, sidecar)
    result["vote_sidecar"] = {
        "path": str(sidecar), "arrays": ["signed_votes", "zero_mask"],
        "coordinate_order": "[payload_channel,receiver_latent,frequency_coordinate]",
    }
    record = root / "read.json"
    _atomic_json(record, result)
    group["key_reads"][key_label] = {"status": "DURABLE", "record": str(record)}
    store.save()
    return result, record


def _adopt_key_read(store, group, key_label, result, record):
    for row in _rows_for_group(store, group, key_label):
        if row["status"] not in ("PENDING_READ", "FAILED", "MISSING"):
            continue
        operation = row["operation"]
        annotation = {
            "schema": "paper-results-v1-logical-time-annotation-v1",
            "slot_id": row["slot_id"], "physical_read_record": str(record),
            "source_coordinate_map": operation["source_coordinate_map"],
            "time_row_mapping": (
                "For every physical time_bit_row, index source_coordinate_map by "
                "that row's output_stride_coordinate. The physical read deliberately "
                "stores no shared source-coordinate claim."
            ),
        }
        annotation_path = store.output / "logical" / Path(*row["slot_id"].split("/"))
        annotation_path = annotation_path.with_suffix(".json")
        _atomic_json(annotation_path, annotation)
        row.update(
            status="READ", reason=None, missing_bits=0,
            decoded_bits=result["decoded_bits"], votes=result["votes"],
            bit_rows=result["bit_rows"], R=result["R"],
            original_reader_match=result.get("original_reader_match"),
            physical_read_record={"path": str(record), "bytes": record.stat().st_size},
            detail_record={"path": str(annotation_path), "bytes": annotation_path.stat().st_size},
        )
    store.save()


def _restore_key_reads(store, group):
    restored = set()
    changed = False
    for key_label in _pending_keys(store, group):
        record = store.output / "physical" / group["physical_id"] / key_label / "read.json"
        if not record.is_file():
            continue
        try:
            result = _read(record)
            if len(result["decoded_bits"]) != 32 or any(bit not in (0, 1) for bit in result["decoded_bits"]):
                raise ValueError("key read must contain 32 binary bits")
            if not isinstance(result["votes"], list) or not isinstance(result["bit_rows"], list) or int(result["R"]) < 1:
                raise ValueError("key read is missing valid vote evidence")
            with Path(result["vote_sidecar"]["path"]).open("rb") as stream:
                if not stream.read(1): raise ValueError("empty vote sidecar")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            group.setdefault("checkpoint_errors", []).append({"path": str(record), "reason": _reason(exc)})
            changed = True
            continue
        _adopt_key_read(store, group, key_label, result, record)
        group["key_reads"][key_label] = {"status": "DURABLE", "record": str(record)}
        restored.add(key_label)
    if restored or changed:
        store.save()
    return restored


def _received_rgb(store, group, artifact):
    from experiments.paper_results_v1.resume_media import load_rgb
    return load_rgb(store, artifact, artifacts=_attack_state(store)["artifacts"])


def _save_latent(path, value):
    import torch
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 16 or not bool(torch.isfinite(value).all()):
        raise ValueError("Wan checkpoint must be a finite [1,16,T,H,W] latent")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_latent(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 16 or not bool(torch.isfinite(value).all()):
        raise ValueError("saved Wan checkpoint has invalid shape/nonfinite values")
    return value


def _pending_keys(store, group):
    return [key for key in KEYS if any(
        row["status"] not in ("READ", "EVALUATED", "UNSUPPORTED", "UNRESOLVED", "NOT_EXECUTED_BY_NOTEBOOK")
        for row in _rows_for_group(store, group, key)
    )]


def _resume_group(store, group):
    # Durable successful or negative scientific output is never reconsidered.
    changed = False
    for row in _rows_for_group(store, group):
        if row["status"] in ("FAILED", "MISSING"):
            remember(row, "resume technical read failure")
            row.update(status="PENDING_READ", reason=None)
            changed = True
    _restore_key_reads(store, group)
    if _pending_keys(store, group) and group.get("status") != "PLANNED":
        remember(group, "continue missing keys; preserve prior encode/key attempts")
        changed = True
    if changed:
        store.save()


def read_case(
    store, case_id, *, load_vae=None, encode=None, payload_read=None,
    load_rgb=None, operate=None, save_latent=None, load_latent=None,
):
    phase = store.data["phases"]["read"][case_id]
    source = _attack_state(store)
    groups = [row for row in store.data["physical_inputs"] if any(
        slot.startswith(case_id + "/") for slot in row["logical_slots"]
    )]
    def case_complete():
        return all(row["status"] in ("READ", "EVALUATED", "UNSUPPORTED", "UNRESOLVED", "NOT_EXECUTED_BY_NOTEBOOK")
                   for row in store.rows(case_id))
    for group in groups:
        if _pending_keys(store, group):
            _resume_group(store, group)
    todo = [group for group in groups if _pending_keys(store, group)]
    if not todo:
        phase["status"] = "COMPLETE" if case_complete() else "PARTIAL"; store.save()
        store.journal.checkpoint("read-already-complete")
        return store
    remember(phase, "resume read at missing physical map/key")
    phase["status"] = "RUNNING"; store.save()
    save_latent = save_latent or _save_latent
    load_latent = load_latent or _load_latent
    vae = None
    try:
        with store.journal.periodic():
            for group in todo:
                normalized = None
                checkpoint = store.output / "physical" / group["physical_id"] / "latent.pt"
                try:
                    if checkpoint.is_file():
                        try:
                            normalized = load_latent(checkpoint)
                        except Exception as exc:
                            group.setdefault("checkpoint_errors", []).append(
                                {"path": str(checkpoint), "reason": _reason(exc), "at": time.time()})
                            store.save()
                    if normalized is None:
                        if load_vae is None:
                            from runtime.wan.generation import load_frozen_vae
                            load_vae = lambda: load_frozen_vae(
                                {"model": {"id": store.config["models"]["wan"]["local_snapshot_path"]}},
                                device=store.config["models"]["wan"].get("device", "cuda"))
                        if vae is None:
                            with unit_attempt(store, case_id + "/vae-load", operation="model_load"):
                                vae = load_vae()
                        if encode is None:
                            from runtime.wan.vae import reencode_rgb24_readback
                            encode = reencode_rgb24_readback
                        if operate is None:
                            from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map
                            operate = operate_map
                        artifact = _artifact(source, group["artifact_id"])
                        received = load_rgb(group, artifact) if load_rgb else _received_rgb(store, group, artifact)
                        corrected = operate(received, group["received_index_map"]).float().div(255.0)
                        attempt = {"attempt": len(group.setdefault("encode_attempts", [])) + 1,
                                   "status": "RUNNING", "started_at": time.time()}
                        for prior in group["encode_attempts"]:
                            if prior["status"] == "RUNNING":
                                prior.update(status="INTERRUPTED", previous_status="RUNNING")
                        group["encode_attempts"].append(attempt)
                        group["status"] = "RUNNING"; store.save()

                        def observe(event, error):
                            attempt["last_event"] = event
                            if event == "RETURNED":
                                attempt.update(status="COMPLETE", finished_at=time.time())
                            elif event == "FAILED":
                                attempt.update(status="FAILED" if isinstance(error, Exception) else "INTERRUPTED",
                                               reason=_reason(error), finished_at=time.time())
                            store.save()
                        try:
                            normalized = encode(vae, corrected, encode_observer=observe).detach().cpu()
                            attempt.update(status="COMPLETE", finished_at=time.time())
                            # This durable boundary precedes both keyed reads.
                            save_latent(checkpoint, normalized)
                            group.update(status="LATENT_SAVED", latent_checkpoint=str(checkpoint))
                            store.save()
                        except BaseException as exc:
                            attempt.setdefault("reason", _reason(exc))
                            if attempt["status"] == "RUNNING":
                                attempt.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED")
                            store.save()
                            raise
                    if payload_read is None:
                        from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import read_payload_general
                        payload_read = read_payload_general
                    for key in _pending_keys(store, group):
                        try:
                            with unit_attempt(store, group["physical_id"] + "/" + key, operation="key_payload_read"):
                                result = payload_read(normalized, store.config["keys"][key],
                                                      group["output_frames"], [None] * group["output_frames"])
                                result, record = _save_key_read(store, group, key, result)
                                _adopt_key_read(store, group, key, result, record)
                        except Exception as exc:
                            group["key_reads"][key] = {"status": "FAILED", "reason": _reason(exc)}
                            for row in _rows_for_group(store, group, key):
                                if row["status"] not in ("READ", "EVALUATED"):
                                    row.update(status="FAILED", reason=_reason(exc), missing_bits=32)
                            store.save()
                    if not _pending_keys(store, group):
                        group["status"] = "COMPLETE"; store.save()
                        # Delete only this owned temporary checkpoint, after all keys are durable.
                        if checkpoint.is_file():
                            checkpoint.unlink()
                        group["latent_checkpoint_released"] = True; store.save()
                except Exception as exc:
                    group.update(status="FAILED", reason=_reason(exc))
                    for row in _rows_for_group(store, group):
                        if row["status"] not in ("READ", "EVALUATED", "UNSUPPORTED", "UNRESOLVED"):
                            row.update(status="FAILED", reason=_reason(exc), missing_bits=32)
                    store.save()
                finally:
                    normalized = None
            phase["status"] = "COMPLETE" if case_complete() else "PARTIAL"
    except BaseException as exc:
        phase["status"] = "INTERRUPTED" if not isinstance(exc, Exception) else "FAILED"
        phase.setdefault("failures", []).append(_reason(exc))
        raise
    finally:
        vae = None; gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        store.save(); store.journal.checkpoint("read-exit")
    return store


def record_external_failure(store, case_id, reason):
    phase = store.data["phases"]["read"][case_id]
    phase.setdefault("failures", []).append(reason)
    phase["status"] = "INTERRUPTED"
    # Keep pending rows and latent/key checkpoints resumable.
    store.save(); store.journal.checkpoint("external-interruption")


def _path_metrics(store, row):
    from experiments.paper_results_v1.attack_matrix import expand_weighted_truth, path_error_rows
    try:
        selected = _read(store.resolve(row["selector_record"]["path"]))
        path = selected["estimate"]["path"]
        if path is None:
            return {"path_evaluation_status": selected["estimate"]["status"]}
        spec = next(item for item in store.config["attacks"] if item["attack_id"] == row["attack_id"])
        value = path_error_rows(path, expand_weighted_truth(spec))
        value.update(
            score_gap=selected["estimate"].get("score_gap"),
            exact_tie=selected["estimate"].get("exact_tie"),
            runner_up_is_distinct=(
                selected["estimate"].get("runner_up_path") is None
                or selected["estimate"].get("runner_up_path") != selected["estimate"].get("best_path")
            ),
        )
        return value
    except Exception as exc:
        return {"path_evaluation_status": "MISSING", "path_evaluation_reason": _reason(exc)}


def _fixed_summary(rows):
    evaluated = [row for row in rows if row["status"] == "EVALUATED"]
    errors = sum(row["bit_errors"] for row in evaluated)
    return {
        "fixed_rows": len(rows), "fixed_bits": len(rows) * 32,
        "evaluated_rows": len(evaluated), "missing_rows": len(rows) - len(evaluated),
        "exact_recoveries": sum(bool(row["exact_recovery"]) for row in evaluated),
        "bit_errors": errors,
        "fixed_denominator_exact_rate": sum(bool(row["exact_recovery"]) for row in evaluated) / len(rows) if rows else None,
        "scorable_ber": errors / (len(evaluated) * 32) if evaluated else None,
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    }


def report(store):
    expected = store.config["payload_bits"]
    for row in store.data["rows"]:
        if row["case_id"] in CASES and row["status"] == "READ":
            errors = sum(left != right for left, right in zip(row["decoded_bits"], expected))
            row.update(status="EVALUATED", bit_errors=errors, exact_recovery=errors == 0, missing_bits=0)
        if row["case_id"] in CASES and row.get("selector_record"):
            row["path_metrics"] = _path_metrics(store, row)
    store.save()

    source_summaries = []
    for case_id in CASES:
        rows = store.rows(case_id)
        preferred = [row for row in rows if row["arm"] == "PAYLOAD_FRAMEWISE_M05"
                     and row["key_label"] == "K0" and row["receiver"] == "CENTERED_U"]
        source_summaries.append({
            "case_id": case_id,
            "all_rows": _fixed_summary(rows),
            "preferred_m05_k0_centered_u": _fixed_summary(preferred),
            "preferred_nonfull": _fixed_summary([row for row in preferred if row["attack_id"] != "full"]),
            "preferred_full": _fixed_summary([row for row in preferred if row["attack_id"] == "full"]),
        })

    grouped = {}
    for arm in ARMS:
        for key_label in KEYS:
            for receiver in RECEIVERS:
                grouped[f"{arm}|{key_label}|{receiver}"] = _fixed_summary([
                    row for row in store.data["rows"] if row["case_id"] in CASES
                    and row["arm"] == arm and row["key_label"] == key_label and row["receiver"] == receiver
                ])

    attack = _attack_state(store)
    baselines = []
    for row in attack["baseline_rows"]:
        if row["case_id"] not in CASES:
            continue
        item = copy.deepcopy(row)
        item["missing_bits"] = 0 if item.get("status") == "EVALUATED" else item.get("missing_bits", 32)
        baselines.append(item)
    qualities = [copy.deepcopy(row) for row in attack["quality_rows"] if row["case_id"] in CASES]

    baseline_index = {(row["case_id"], row["attack_id"], row["method"]): row for row in baselines}
    comparison_rows = []
    for case_id in CASES:
        for attack_spec in store.config["attacks"]:
            attack_id = attack_spec["attack_id"]
            main = next(row for row in store.data["rows"] if (
                row["case_id"], row["arm"], row["attack_id"], row["key_label"], row["receiver"]
            ) == (case_id, "PAYLOAD_FRAMEWISE_M05", attack_id, "K0", "CENTERED_U"))
            for method in attack_eval.BASELINES:
                baseline = baseline_index[(case_id, attack_id, method)]
                paired = main["status"] == baseline["status"] == "EVALUATED"
                comparison_rows.append({
                    "pair_id": f"{case_id}/{attack_id}/{method}", "case_id": case_id,
                    "attack_id": attack_id, "baseline_method": method,
                    "status": "PAIRED" if paired else "UNEVALUABLE_PAIR",
                    "main_status": main["status"], "baseline_status": baseline["status"],
                    "main_bit_errors": main.get("bit_errors"),
                    "baseline_bit_errors": baseline.get("bit_errors"),
                    "baseline_minus_main_bit_errors": (
                        baseline["bit_errors"] - main["bit_errors"] if paired else None
                    ),
                })

    actual = [attempt for group in store.data["physical_inputs"] for attempt in group.get(
        "encode_attempts", [dict(status=group["status"], started_at=group["started_at"])] if "started_at" in group else [])]
    unit_calls = store.data.get("unit_attempts", {})
    vae_calls = [attempt for rows in unit_calls.values() for attempt in rows if attempt.get("operation") == "model_load"]
    decode_calls = [attempt for rows in unit_calls.values() for attempt in rows if attempt.get("operation") == "decode_existing_mp4"]
    pending = {
        "receiver_rows": sum(row["case_id"] in CASES and row["status"] not in ("EVALUATED", "UNSUPPORTED", "UNRESOLVED") for row in store.data["rows"]),
        "baseline_rows": sum(row["status"] not in ("EVALUATED", "UNSUPPORTED", "UNRESOLVED") for row in baselines),
        "quality_rows": sum(row["status"] != "EVALUATED" for row in qualities),
    }
    result = {
        "schema": SCHEMA, "status": "PARTIAL_RESUMABLE" if any(pending.values()) else "COMPLETE_WITH_RETAINED_ITEM_STATUSES",
        "pending_technical_units": pending,
        "fixed_denominator": store.data["fixed_denominator"],
        "receiver_status_counts": dict(sorted(Counter(
            row["status"] for row in store.data["rows"] if row["case_id"] in CASES
        ).items())),
        "receiver_groups": grouped, "source_summaries": source_summaries,
        "receiver_rows": store.data["rows"], "baseline_rows": baselines,
        "quality_rows": qualities, "comparison_rows": comparison_rows,
        "cost": {
            "planned_logical_rows": 2880,
            "planned_physical_wan_encode_range": [0, 2880],
            "planned_conditional_shared_map_lower_bound": 480,
            "physical_inputs_planned": len(store.data["physical_inputs"]),
            "physical_wan_calls": {
                "attempted": len(actual),
                "status_counts": dict(sorted(Counter(group["status"] for group in actual).items())),
                "extra_attempts": sum(max(0, len(group.get("encode_attempts", [])) - 1) for group in store.data["physical_inputs"]),
                "budget_semantics": "0..2880 is unique maps; interrupted missing units may add explicitly counted attempts",
            },
            "vae_loads": {
                "attempted": len(store.data["vae_loads"]) + len(vae_calls),
                "status_counts": dict(sorted(Counter(row["status"] for row in [*store.data["vae_loads"], *vae_calls]).items())),
            },
            "media_decodes": {
                "attempted": len(store.data["media_decodes"]) + len(decode_calls),
                "status_counts": dict(sorted(Counter(row["status"] for row in [*store.data["media_decodes"], *decode_calls]).items())),
            },
            "unit_attempts": unit_calls,
            "attack_unit_attempts": attack.get("unit_attempts", {}),
            "attack_actual_records": attack.get("records", {}),
        },
    }
    _atomic_json(store.output / "evaluation_report.json", result)
    compact = []
    for row in store.data["rows"]:
        value = {key: row.get(key) for key in (
            "slot_id", "case_id", "cohort", "arm", "attack_id", "key_label", "receiver",
            "status", "reason", "physical_id", "missing_bits", "bit_errors", "exact_recovery",
        )}
        value["path_metrics"] = json.dumps(row.get("path_metrics", {}), sort_keys=True, allow_nan=False)
        compact.append(value)
    with (store.output / "receiver_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(compact)
    store.data["status"] = result["status"]; store.save()
    store.journal.checkpoint("report-complete")
    return result
