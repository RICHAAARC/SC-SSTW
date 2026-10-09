"""Explicit outer workflow for future paper artifacts and native baselines.

The workflow executes only caller-supplied operations and adapters. Its
manifest, rather than successful outputs, defines every denominator.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path


WORKFLOW_SCHEMA_VERSION = "paper-workflow-v1"
STAGES = ("OFF_NATIVE", "PAYLOAD_NATIVE", "PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_FRAMEWISE_M05")


class WorkflowManifestError(ValueError):
    """The workflow plan is ambiguous or changes a fixed protocol relation."""


def _require(row, fields, where):
    if not isinstance(row, dict):
        raise WorkflowManifestError(f"{where} must be an object")
    missing = [field for field in fields if field not in row]
    if missing:
        raise WorkflowManifestError(f"{where} missing fields: {', '.join(missing)}")


def _identifier(value, where):
    if not isinstance(value, str) or not value.strip():
        raise WorkflowManifestError(f"{where} must be a nonempty string")


def _message_bits(value, where):
    if not isinstance(value, list) or not value:
        raise WorkflowManifestError(f"{where} must be a nonempty list")
    if any(type(bit) is not int or bit not in (0, 1) for bit in value):
        raise WorkflowManifestError(f"{where} must contain integer 0/1 values")


def validate_workflow_manifest(manifest):
    _require(
        manifest,
        (
            "schema_version", "study_id", "evidence_role", "backend_policy", "cases",
            "native_jobs", "quality_pairs", "main_report",
        ),
        "workflow manifest",
    )
    if manifest["schema_version"] != WORKFLOW_SCHEMA_VERSION:
        raise WorkflowManifestError(f"schema_version must be {WORKFLOW_SCHEMA_VERSION!r}")
    if manifest["evidence_role"] not in ("SYNTHETIC_FIXTURE_ONLY", "PLANNED_FUTURE_EVALUATION"):
        raise WorkflowManifestError("evidence_role must be synthetic fixture or planned future evaluation")
    if manifest["backend_policy"] != "EXPLICIT_INJECTION_ONLY":
        raise WorkflowManifestError("backend_policy must remain EXPLICIT_INJECTION_ONLY")
    if any(not isinstance(manifest[field], list) for field in ("cases", "native_jobs", "quality_pairs")):
        raise WorkflowManifestError("cases, native_jobs, and quality_pairs must be lists")
    if not isinstance(manifest["main_report"], dict):
        raise WorkflowManifestError("main_report must be an object")

    case_ids = set()
    artifact_ids = set()
    artifact_index = {}
    for index, case in enumerate(manifest["cases"]):
        where = f"cases[{index}]"
        _require(case, ("case_id", "source_id", "source", "noise", "codec", "stages"), where)
        _identifier(case["case_id"], f"{where}.case_id")
        if case["case_id"] in case_ids:
            raise WorkflowManifestError(f"duplicate case_id {case['case_id']!r}")
        case_ids.add(case["case_id"])
        _identifier(case["source_id"], f"{where}.source_id")
        source = case["source"]
        _require(source, ("artifact_id", "content_id", "input"), f"{where}.source")
        _identifier(source["artifact_id"], f"{where}.source.artifact_id")
        _identifier(source["content_id"], f"{where}.source.content_id")
        source_input = source["input"]
        _require(source_input, ("kind",), f"{where}.source.input")
        if source_input["kind"] == "INLINE_ARRAY":
            if manifest["evidence_role"] != "SYNTHETIC_FIXTURE_ONLY" or "value" not in source_input:
                raise WorkflowManifestError("INLINE_ARRAY is restricted to an explicit synthetic fixture")
        elif source_input["kind"] == "FILE":
            _identifier(source_input.get("path"), f"{where}.source.input.path")
        else:
            raise WorkflowManifestError(f"{where}.source.input.kind must be INLINE_ARRAY or FILE")
        noise = case["noise"]
        _require(noise, ("noise_id", "seed"), f"{where}.noise")
        _identifier(noise["noise_id"], f"{where}.noise.noise_id")
        if type(noise["seed"]) is not int:
            raise WorkflowManifestError(f"{where}.noise.seed must be an integer")
        codec = case["codec"]
        _require(codec, ("codec_id", "operation", "parameters"), f"{where}.codec")
        _identifier(codec["codec_id"], f"{where}.codec.codec_id")
        _identifier(codec["operation"], f"{where}.codec.operation")
        if not isinstance(codec["parameters"], dict):
            raise WorkflowManifestError(f"{where}.codec.parameters must be an object")
        if not isinstance(case["stages"], dict) or set(case["stages"]) != set(STAGES):
            raise WorkflowManifestError(f"{where}.stages must declare exactly {', '.join(STAGES)}")

        ids = [source["artifact_id"]]
        stage_rows = case["stages"]
        for stage in STAGES:
            row = stage_rows[stage]
            _require(
                row,
                ("input_artifact_id", "pre_artifact_id", "post_artifact_id", "transform_operation"),
                f"{where}.stages.{stage}",
            )
            for field in ("input_artifact_id", "pre_artifact_id", "post_artifact_id", "transform_operation"):
                _identifier(row[field], f"{where}.stages.{stage}.{field}")
            ids.extend((row["pre_artifact_id"], row["post_artifact_id"]))
        expected_inputs = {
            "OFF_NATIVE": source["artifact_id"],
            "PAYLOAD_NATIVE": source["artifact_id"],
            "PAYLOAD_FRAMEWISE_RECON": stage_rows["PAYLOAD_NATIVE"]["pre_artifact_id"],
            "PAYLOAD_FRAMEWISE_M05": stage_rows["PAYLOAD_FRAMEWISE_RECON"]["pre_artifact_id"],
        }
        for stage, expected in expected_inputs.items():
            if stage_rows[stage]["input_artifact_id"] != expected:
                raise WorkflowManifestError(f"{where}.stages.{stage}.input_artifact_id must be {expected!r}")
        for artifact_id in ids:
            if artifact_id in artifact_ids:
                raise WorkflowManifestError(f"duplicate artifact_id {artifact_id!r}")
            artifact_ids.add(artifact_id)
        artifact_index[source["artifact_id"]] = (case["case_id"], "SOURCE", "SOURCE")
        for stage in STAGES:
            artifact_index[stage_rows[stage]["pre_artifact_id"]] = (case["case_id"], stage, "PRE")
            artifact_index[stage_rows[stage]["post_artifact_id"]] = (case["case_id"], stage, "POST")

    native_job_ids = set()
    for index, job in enumerate(manifest["native_jobs"]):
        where = f"native_jobs[{index}]"
        _require(
            job,
            (
                "job_id", "case_id", "method", "input_artifact_id", "pre_artifact_id",
                "post_artifact_id", "native_message_bits",
            ),
            where,
        )
        _identifier(job["job_id"], f"{where}.job_id")
        if job["job_id"] in native_job_ids:
            raise WorkflowManifestError(f"duplicate native job_id {job['job_id']!r}")
        native_job_ids.add(job["job_id"])
        if job["case_id"] not in case_ids:
            raise WorkflowManifestError(f"{where}.case_id is not declared")
        if job["method"] not in ("videoseal", "rivagan"):
            raise WorkflowManifestError(f"{where}.method must be videoseal or rivagan")
        if job["input_artifact_id"] not in artifact_index:
            raise WorkflowManifestError(f"{where}.input_artifact_id is not declared")
        input_case, input_stage, input_phase = artifact_index[job["input_artifact_id"]]
        if (input_case, input_stage, input_phase) != (job["case_id"], "OFF_NATIVE", "PRE"):
            raise WorkflowManifestError(f"{where}.input_artifact_id must be the case's OFF_NATIVE PRE artifact")
        for field, phase in (("pre_artifact_id", "NATIVE_PRE"), ("post_artifact_id", "NATIVE_POST")):
            _identifier(job[field], f"{where}.{field}")
            if job[field] in artifact_ids:
                raise WorkflowManifestError(f"duplicate artifact_id {job[field]!r}")
            artifact_ids.add(job[field])
            artifact_index[job[field]] = (job["case_id"], job["method"], phase)
        _message_bits(job["native_message_bits"], f"{where}.native_message_bits")
        if job["method"] == "rivagan" and len(job["native_message_bits"]) != 32:
            raise WorkflowManifestError(f"{where}.native_message_bits must contain exactly 32 bits")

    quality_ids = set()
    for index, pair in enumerate(manifest["quality_pairs"]):
        where = f"quality_pairs[{index}]"
        _require(
            pair,
            ("quality_id", "case_id", "comparison", "reference_artifact_id", "candidate_artifact_id", "data_range"),
            where,
        )
        _identifier(pair["quality_id"], f"{where}.quality_id")
        if pair["quality_id"] in quality_ids:
            raise WorkflowManifestError(f"duplicate quality_id {pair['quality_id']!r}")
        quality_ids.add(pair["quality_id"])
        if pair["case_id"] not in case_ids:
            raise WorkflowManifestError(f"{where}.case_id is not declared")
        for field in ("reference_artifact_id", "candidate_artifact_id"):
            artifact_id = pair[field]
            if artifact_id not in artifact_index:
                raise WorkflowManifestError(f"{where}.{field} is not declared")
            case_id, _stage, phase = artifact_index[artifact_id]
            if case_id != pair["case_id"] or phase != "POST":
                raise WorkflowManifestError(f"{where}.{field} must be a POST artifact from the same case")
        if not isinstance(pair["data_range"], (int, float)) or isinstance(pair["data_range"], bool):
            raise WorkflowManifestError(f"{where}.data_range must be positive and finite")
        if not math.isfinite(float(pair["data_range"])) or pair["data_range"] <= 0:
            raise WorkflowManifestError(f"{where}.data_range must be positive and finite")
    return manifest


def _case_metadata(case):
    return {
        "case_id": case["case_id"],
        "source_id": case["source_id"],
        "source_content_id": case["source"]["content_id"],
        "noise_id": case["noise"]["noise_id"],
        "noise_seed": case["noise"]["seed"],
        "codec_id": case["codec"]["codec_id"],
        "codec_parameters": case["codec"]["parameters"],
    }


def expand_plan(manifest):
    validate_workflow_manifest(manifest)
    rows = []
    for case in manifest["cases"]:
        for stage in STAGES:
            stage_row = case["stages"][stage]
            rows.append({
                "step_id": f"{case['case_id']}/{stage}/TRANSFORM",
                "case_id": case["case_id"], "stage": stage, "phase": "TRANSFORM",
                "operation": stage_row["transform_operation"],
                "input_artifact_id": stage_row["input_artifact_id"],
                "output_artifact_id": stage_row["pre_artifact_id"],
                "state": "PLANNED", "reason": None,
            })
            rows.append({
                "step_id": f"{case['case_id']}/{stage}/CODEC",
                "case_id": case["case_id"], "stage": stage, "phase": "CODEC",
                "operation": case["codec"]["operation"],
                "input_artifact_id": stage_row["pre_artifact_id"],
                "output_artifact_id": stage_row["post_artifact_id"],
                "state": "PLANNED", "reason": None,
            })
    return rows


def _nested_shape(value):
    if not isinstance(value, (list, tuple)):
        return []
    if not value:
        return [0]
    child = _nested_shape(value[0])
    if any(_nested_shape(item) != child for item in value[1:]):
        raise ValueError("array is ragged")
    return [len(value), *child]


def _flatten_numeric(value):
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("quality input must be a numeric array")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("quality input contains a non-finite value")
    return [number]


def _public_artifact(row):
    return {key: value for key, value in row.items() if key != "_value"}


def _artifact(artifact_id, status, metadata, *, role, stage, phase, reason=None, value=None):
    row = {
        "artifact_id": artifact_id, "status": status, "role": role, "stage": stage,
        "phase": phase, "reason": reason, **metadata,
    }
    if status == "AVAILABLE":
        row["value_shape"] = _nested_shape(value) if isinstance(value, (list, tuple)) else None
        if isinstance(value, (str, Path)):
            row["uri"] = str(value)
        row["_value"] = value
    return row


def _execute_step(row, case, artifacts, operations):
    metadata = _case_metadata(case)
    source = artifacts[row["input_artifact_id"]]
    output_role = "MAIN_METHOD"
    output_phase = "PRE" if row["phase"] == "TRANSFORM" else "POST"
    if source["status"] != "AVAILABLE":
        row.update(state="BLOCKED_DEPENDENCY", reason=f"input artifact state is {source['status']}")
        artifacts[row["output_artifact_id"]] = _artifact(
            row["output_artifact_id"], "BLOCKED", metadata, role=output_role,
            stage=row["stage"], phase=output_phase, reason=row["reason"],
        )
        return None
    operation = operations.get(row["operation"])
    if operation is None:
        row.update(state="SKIPPED_BACKEND_UNAVAILABLE", reason=f"operation {row['operation']!r} was not injected")
        artifacts[row["output_artifact_id"]] = _artifact(
            row["output_artifact_id"], "SKIPPED", metadata, role=output_role,
            stage=row["stage"], phase=output_phase, reason=row["reason"],
        )
        return None
    context = {
        **metadata,
        "stage": row["stage"],
        "phase": row["phase"],
        "codec": case["codec"],
        "input_artifact_id": row["input_artifact_id"],
        "output_artifact_id": row["output_artifact_id"],
    }
    started = time.perf_counter()
    try:
        value = operation(source["_value"], context)
        if value is None:
            raise ValueError("operation returned no artifact value or URI")
        elapsed = time.perf_counter() - started
        row.update(state="SUCCEEDED", reason=None)
        artifacts[row["output_artifact_id"]] = _artifact(
            row["output_artifact_id"], "AVAILABLE", metadata, role=output_role,
            stage=row["stage"], phase=output_phase, value=value,
        )
        return elapsed
    except Exception as exc:  # execution failures are fixed-denominator records
        elapsed = time.perf_counter() - started
        row.update(state="FAILED", reason=f"{type(exc).__name__}: {exc}")
        artifacts[row["output_artifact_id"]] = _artifact(
            row["output_artifact_id"], "FAILED", metadata, role=output_role,
            stage=row["stage"], phase=output_phase, reason=row["reason"],
        )
        return elapsed


def _quality_record(pair, artifacts):
    base = {
        "quality_id": pair["quality_id"], "case_id": pair["case_id"],
        "comparison": pair["comparison"],
        "reference_artifact_id": pair["reference_artifact_id"],
        "candidate_artifact_id": pair["candidate_artifact_id"],
        "metric_semantics": "PAIRWISE_ABSOLUTE_NOT_ADDITIVE_DECOMPOSITION",
        "state": None, "reason": None,
    }
    reference = artifacts[pair["reference_artifact_id"]]
    candidate = artifacts[pair["candidate_artifact_id"]]
    if reference["status"] != "AVAILABLE" or candidate["status"] != "AVAILABLE":
        base.update(state="BLOCKED_DEPENDENCY", reason="reference or candidate artifact is unavailable")
        return base
    identity_fields = (
        "case_id", "source_id", "source_content_id", "noise_id", "noise_seed",
        "codec_id", "codec_parameters",
    )
    mismatches = [field for field in identity_fields if reference.get(field) != candidate.get(field)]
    if mismatches:
        base.update(state="CONFLICT", reason="paired artifact identity mismatch: " + ", ".join(mismatches))
        return base
    try:
        if _nested_shape(reference["_value"]) != _nested_shape(candidate["_value"]):
            raise ValueError("paired arrays have different shapes")
        left = _flatten_numeric(reference["_value"])
        right = _flatten_numeric(candidate["_value"])
        if not left:
            raise ValueError("paired arrays are empty")
        mse = sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)
        rmse = math.sqrt(mse)
        if mse == 0.0:
            psnr = None
            psnr_status = "IDENTICAL_INFINITE"
        else:
            psnr = 20.0 * math.log10(float(pair["data_range"]) / rmse)
            psnr_status = "FINITE"
        base.update(
            state="OBSERVED", sample_count=len(left), shape=_nested_shape(reference["_value"]),
            mse=mse, rmse=rmse, psnr_db=psnr, psnr_status=psnr_status,
            declared_pairing_receipt={field: reference[field] for field in identity_fields},
            pairing_verification_scope="DECLARATION_AND_CALLBACK_RECEIPT_ONLY_NOT_PHYSICAL_MEDIA_VERIFICATION",
        )
    except Exception as exc:
        base.update(state="FAILED", reason=f"{type(exc).__name__}: {exc}")
    return base


def run_workflow(manifest, *, operations=None, native_adapters=None, base_dir=None, main_report_link=None):
    validate_workflow_manifest(manifest)
    operations = dict(operations or {})
    native_adapters = dict(native_adapters or {})
    base_dir = Path(base_dir or ".").resolve()
    plan = expand_plan(manifest)
    plan_by_case = {}
    for row in plan:
        plan_by_case.setdefault(row["case_id"], []).append(row)
    artifacts = {}
    source_rows = []
    cost_rows = []
    for case in manifest["cases"]:
        metadata = _case_metadata(case)
        source = case["source"]
        source_input = source["input"]
        if source_input["kind"] == "INLINE_ARRAY":
            try:
                source_row = _artifact(
                    source["artifact_id"], "AVAILABLE", metadata, role="SOURCE", stage="SOURCE",
                    phase="SOURCE", value=source_input["value"],
                )
            except Exception as exc:
                source_row = _artifact(
                    source["artifact_id"], "FAILED", metadata, role="SOURCE", stage="SOURCE",
                    phase="SOURCE", reason=f"{type(exc).__name__}: {exc}",
                )
        else:
            path = Path(source_input["path"])
            path = path if path.is_absolute() else base_dir / path
            if path.is_file():
                source_row = _artifact(
                    source["artifact_id"], "AVAILABLE", metadata, role="SOURCE", stage="SOURCE",
                    phase="SOURCE", value=str(path.resolve()),
                )
                source_row["uri"] = str(path.resolve())
            else:
                source_row = _artifact(
                    source["artifact_id"], "MISSING", metadata, role="SOURCE", stage="SOURCE",
                    phase="SOURCE", reason=f"source file not found: {path}",
                )
                source_row["uri"] = str(path)
        artifacts[source["artifact_id"]] = source_row
        source_rows.append(source_row)
        for row in plan_by_case[case["case_id"]]:
            elapsed = _execute_step(row, case, artifacts, operations)
            cost_rows.append({
                "cost_id": row["step_id"], "kind": "WALL_SECONDS_OBSERVED_CALL",
                "state": row["state"], "seconds": elapsed, "budget_status": "NO_BUDGET_ADOPTED",
                "reason": row["reason"],
            })

    case_lookup = {case["case_id"]: case for case in manifest["cases"]}
    native_records = []
    for job in manifest["native_jobs"]:
        case = case_lookup[job["case_id"]]
        metadata = _case_metadata(case)
        record = {
            "job_id": job["job_id"], "case_id": job["case_id"], "method": job["method"],
            "input_artifact_id": job["input_artifact_id"],
            "pre_artifact_id": job["pre_artifact_id"], "post_artifact_id": job["post_artifact_id"],
            "status": None, "reason": None,
        }
        source = artifacts[job["input_artifact_id"]]
        adapter = native_adapters.get(job["method"])
        phase_costs = [
            {"cost_id": f"native/{job['job_id']}/EMBED", "kind": "WALL_SECONDS_OBSERVED_CALL", "state": "BLOCKED_DEPENDENCY", "seconds": None, "budget_status": "NO_BUDGET_ADOPTED", "reason": None},
            {"cost_id": f"native/{job['job_id']}/CODEC", "kind": "WALL_SECONDS_OBSERVED_CALL", "state": "BLOCKED_DEPENDENCY", "seconds": None, "budget_status": "NO_BUDGET_ADOPTED", "reason": None},
            {"cost_id": f"native/{job['job_id']}/EXTRACT", "kind": "WALL_SECONDS_OBSERVED_CALL", "state": "BLOCKED_DEPENDENCY", "seconds": None, "budget_status": "NO_BUDGET_ADOPTED", "reason": None},
        ]
        if source["status"] != "AVAILABLE":
            record.update(status="BLOCKED_DEPENDENCY", reason=f"input artifact state is {source['status']}")
            for phase, field in (("NATIVE_PRE", "pre_artifact_id"), ("NATIVE_POST", "post_artifact_id")):
                artifacts[job[field]] = _artifact(
                    job[field], "BLOCKED", metadata, role="NATIVE_BASELINE",
                    stage=job["method"], phase=phase, reason=record["reason"],
                )
            for cost in phase_costs:
                cost["reason"] = record["reason"]
        elif adapter is None:
            record.update(status="SKIPPED_BACKEND_UNAVAILABLE", reason="native adapter was not injected")
            for phase, field in (("NATIVE_PRE", "pre_artifact_id"), ("NATIVE_POST", "post_artifact_id")):
                artifacts[job[field]] = _artifact(
                    job[field], "SKIPPED", metadata, role="NATIVE_BASELINE",
                    stage=job["method"], phase=phase, reason=record["reason"],
                )
            for cost in phase_costs:
                cost.update(state="SKIPPED_BACKEND_UNAVAILABLE", reason=record["reason"])
        else:
            try:
                output_uri = job.get("output_uri")
                if output_uri:
                    path = Path(output_uri)
                    output_uri = str(path if path.is_absolute() else base_dir / path)
                started = time.perf_counter()
                pre_value, embed = adapter.embed(
                    source["_value"], list(job["native_message_bits"]), output_uri=output_uri,
                )
                if pre_value is None:
                    raise ValueError("native embed returned no artifact value or URI")
                phase_costs[0].update(state="SUCCEEDED", seconds=time.perf_counter() - started)
                artifacts[job["pre_artifact_id"]] = _artifact(
                    job["pre_artifact_id"], "AVAILABLE", metadata, role="NATIVE_BASELINE",
                    stage=job["method"], phase="NATIVE_PRE", value=pre_value,
                )
                native = {
                    "method": job["method"], "status": "PARTIAL",
                    "backend_metadata": adapter.backend_metadata,
                    "native_message": embed["native_message"], "embed": embed,
                }
                record["native"] = native
                codec_operation = operations.get(case["codec"]["operation"])
                if codec_operation is None:
                    raise LookupError(f"shared codec operation {case['codec']['operation']!r} was not injected")
                codec_context = {
                    **metadata, "stage": job["method"], "phase": "CODEC",
                    "codec": case["codec"], "input_artifact_id": job["pre_artifact_id"],
                    "output_artifact_id": job["post_artifact_id"],
                }
                started = time.perf_counter()
                post_value = codec_operation(pre_value, codec_context)
                if post_value is None:
                    raise ValueError("shared codec callback returned no artifact value or URI")
                phase_costs[1].update(state="SUCCEEDED", seconds=time.perf_counter() - started)
                artifacts[job["post_artifact_id"]] = _artifact(
                    job["post_artifact_id"], "AVAILABLE", metadata, role="NATIVE_BASELINE",
                    stage=job["method"], phase="NATIVE_POST", value=post_value,
                )
                native["shared_codec_callback_receipt"] = {
                    **case["codec"],
                    "verification_scope": "DECLARATION_AND_CALLBACK_RECEIPT_ONLY_NOT_PHYSICAL_MEDIA_VERIFICATION",
                }
                started = time.perf_counter()
                extract = adapter.extract(post_value)
                phase_costs[2].update(state="SUCCEEDED", seconds=time.perf_counter() - started)
                native.update(status="SUCCEEDED", extract=extract)
                if job["method"] == "videoseal":
                    native["main_32bit_mapping"] = "PENDING_NOT_APPLIED"
                else:
                    native["sequence_32bit_recovery"] = "PENDING_REDUCER_NOT_APPLIED"
                    native["per_frame_accuracy"] = "NOT_COMPUTED_WITHOUT_TRUTH_EVALUATION"
                record.update(status="SUCCEEDED", native=native)
            except Exception as exc:
                failure_status = "SKIPPED_BACKEND_UNAVAILABLE" if isinstance(exc, LookupError) else "FAILED"
                record.update(status=failure_status, reason=f"{type(exc).__name__}: {exc}")
                if "native" in record:
                    record["native"].update(status="PARTIAL_WITH_RETAINED_FAILURE", failure_reason=record["reason"])
                if job["pre_artifact_id"] not in artifacts:
                    artifacts[job["pre_artifact_id"]] = _artifact(
                        job["pre_artifact_id"], "FAILED", metadata, role="NATIVE_BASELINE",
                        stage=job["method"], phase="NATIVE_PRE", reason=record["reason"],
                    )
                if job["post_artifact_id"] not in artifacts:
                    artifacts[job["post_artifact_id"]] = _artifact(
                        job["post_artifact_id"], "SKIPPED" if isinstance(exc, LookupError) else "BLOCKED",
                        metadata, role="NATIVE_BASELINE",
                        stage=job["method"], phase="NATIVE_POST", reason=record["reason"],
                    )
                first_pending = next((cost for cost in phase_costs if cost["seconds"] is None), None)
                if first_pending is not None:
                    if isinstance(exc, LookupError):
                        first_pending["state"] = "SKIPPED_BACKEND_UNAVAILABLE"
                    else:
                        first_pending["state"] = "FAILED"
                    first_pending["reason"] = record["reason"]
                for cost in phase_costs:
                    if cost["reason"] is None and cost["state"] == "BLOCKED_DEPENDENCY":
                        cost["reason"] = "earlier native phase did not succeed"
        cost_rows.extend(phase_costs)
        native_records.append(record)

    quality_records = [_quality_record(pair, artifacts) for pair in manifest["quality_pairs"]]
    plan_counts = dict(sorted(Counter(row["state"] for row in plan).items()))
    source_counts = dict(sorted(Counter(row["status"] for row in source_rows).items()))
    native_counts = dict(sorted(Counter(row["status"] for row in native_records).items()))
    quality_counts = dict(sorted(Counter(row["state"] for row in quality_records).items()))
    issue_states = {"FAILED", "MISSING", "SKIPPED_BACKEND_UNAVAILABLE", "BLOCKED_DEPENDENCY", "CONFLICT"}
    has_issues = any(state in issue_states for counts in (plan_counts, source_counts, native_counts, quality_counts) for state in counts)
    link = main_report_link or {"status": "PENDING_NOT_LINKED"}
    if manifest["cases"] and link.get("status") != "LINKED":
        has_issues = True
    report_status = "PENDING_EMPTY_ROSTER" if not manifest["cases"] else "COMPLETE_WITH_RETAINED_ISSUES" if has_issues else "COMPLETE"
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "evidence_role": manifest["evidence_role"],
        "report_status": report_status,
        "backend_policy": manifest["backend_policy"],
        "manifest_denominator": {
            "cases": len(manifest["cases"]), "source_artifacts": len(source_rows),
            "plan_steps": len(plan), "native_jobs": len(native_records),
            "quality_pairs": len(quality_records), "cost_rows": len(cost_rows),
            "source_state_counts": source_counts, "plan_state_counts": plan_counts,
            "native_state_counts": native_counts, "quality_state_counts": quality_counts,
        },
        "main_report": link,
        "plan": plan,
        "artifacts": [_public_artifact(artifacts[key]) for key in sorted(artifacts)],
        "native_records": native_records,
        "quality_records": quality_records,
        "cost_records": cost_rows,
        "claim_guard": (
            "Workflow execution records engineering calls only. Native-capacity outputs remain separate from the "
            "strict main 32-bit report. Pairwise quality metrics are not additive VAE/M05 decompositions."
        ),
    }


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def render_workflow_markdown(report):
    denominator = report["manifest_denominator"]
    lines = [
        f"# {report['study_id']} workflow", "", f"Status: `{report['report_status']}`", "",
        f"Evidence role: `{report['evidence_role']}`", "", "## Fixed denominators", "",
        "| Item | Count |", "| --- | ---: |",
    ]
    for key in ("cases", "source_artifacts", "plan_steps", "native_jobs", "quality_pairs", "cost_rows"):
        lines.append(f"| {key} | {denominator[key]} |")
    lines += ["", "## Plan execution", "", "| Step | Stage | Phase | State | Reason |", "| --- | --- | --- | --- | --- |"]
    for row in report["plan"]:
        lines.append(f"| {row['step_id']} | {row['stage']} | {row['phase']} | {row['state']} | {row.get('reason') or ''} |")
    lines += ["", "## Native records", "", "| Job | Method | State | 32-bit relation | Reason |", "| --- | --- | --- | --- | --- |"]
    for row in report["native_records"]:
        native = row.get("native", {})
        relation = native.get("main_32bit_mapping") or native.get("sequence_32bit_recovery") or ""
        lines.append(f"| {row['job_id']} | {row['method']} | {row['status']} | {relation} | {row.get('reason') or ''} |")
    lines += ["", "## Quality pairs", "", "| Pair | Comparison | State | MSE | RMSE | PSNR dB | Semantics |", "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for row in report["quality_records"]:
        lines.append(
            f"| {row['quality_id']} | {row['comparison']} | {row['state']} | {row.get('mse', '')} | "
            f"{row.get('rmse', '')} | {row.get('psnr_db', '')} | {row['metric_semantics']} |"
        )
    lines += ["", "## Linked strict main report", "", f"`{_csv_value(report['main_report'])}`", "", report["claim_guard"], ""]
    return "\n".join(lines)


def write_workflow_report(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "workflow_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "workflow_report.md").write_text(render_workflow_markdown(report), encoding="utf-8")
    _write_csv(
        output / "plan.csv", report["plan"],
        ("step_id", "case_id", "stage", "phase", "operation", "input_artifact_id", "output_artifact_id", "state", "reason"),
    )
    _write_csv(
        output / "artifacts.csv", report["artifacts"],
        (
            "artifact_id", "status", "role", "stage", "phase", "case_id", "source_id",
            "source_content_id", "noise_id", "noise_seed", "codec_id", "codec_parameters",
            "value_shape", "uri", "reason",
        ),
    )
    _write_csv(
        output / "quality.csv", report["quality_records"],
        (
            "quality_id", "case_id", "comparison", "reference_artifact_id", "candidate_artifact_id",
            "state", "sample_count", "shape", "mse", "rmse", "psnr_db", "psnr_status",
            "metric_semantics", "declared_pairing_receipt", "pairing_verification_scope", "reason",
        ),
    )
    _write_csv(
        output / "cost.csv", report["cost_records"],
        ("cost_id", "kind", "state", "seconds", "budget_status", "reason"),
    )
    (output / "native_records.json").write_text(
        json.dumps(report["native_records"], indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
