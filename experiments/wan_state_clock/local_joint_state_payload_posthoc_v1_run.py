"""Read-only known-grid posthoc for already saved local-joint raw observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from main.tube_state import local_joint_state_payload_posthoc_v1 as method
from runtime.wan import local_joint_state_payload_experiment_v1 as runtime
from runtime.wan.provenance import content_id
from experiments.wan_state_clock import local_joint_state_payload_v1_run as experiment


ROOT = Path(__file__).resolve().parents[2]
FIXED_CONFIG = ROOT / "experiments/wan_state_clock/configs/local_joint_state_payload_v1.json"
SOURCE_CLOSURE = tuple(dict.fromkeys((*experiment.SOURCE_CLOSURE,
    "main/tube_state/local_joint_state_payload_posthoc_v1.py",
    "experiments/wan_state_clock/local_joint_state_payload_posthoc_v1_run.py",
    "experiments/wan_state_clock/configs/local_joint_state_payload_v1.json")))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode()
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(path=str(path), sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded))


def _json_safe(value: Any) -> Any:
    """Make derived output strict JSON without modifying sealed source files."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _truth_free_manifest_from_result(run_result: dict[str, Any]) -> dict[str, Any]:
    arms = run_result.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("run receipt arms missing")
    return dict(
        schema="local-joint-raw-observation-manifest-v1", truth_loaded=False,
        arms={arm: {
            name: {key: value for key, value in receipt.items() if key != "key"}
            for name, receipt in row.get("observations", {}).items()
        } for arm, row in arms.items()},
    )


def _run_receipt(run_result_path: Path, run_identity: str, run_result: dict[str, Any]) -> dict[str, Any]:
    return dict(
        path=str(run_result_path), sha256=run_identity, status=run_result.get("status"),
        stage=run_result.get("stage"), execution=run_result.get("execution"),
        actual_model_calls=run_result.get("actual_model_calls"),
        source_identity=run_result.get("source_identity"),
        arms={arm: dict(
            status=row.get("status"), initial_fingerprint=row.get("initial_fingerprint"),
            terminal_fingerprint=row.get("terminal_fingerprint"),
        ) for arm, row in run_result.get("arms", {}).items()},
    )


def _run_is_complete(run_result: dict[str, Any]) -> bool:
    execution = run_result.get("execution", {})
    arms = run_result.get("arms", {})
    complete_statuses = {"COMPLETE", "COMPLETE_WITH_OBSERVATION_FAILURE"}
    return (run_result.get("status") in complete_statuses
            and execution.get("attempted") is True and execution.get("completed") is True
            and set(arms) == set(method.ARMS)
            and all(row.get("status") in complete_statuses for row in arms.values()))


def _source_identity() -> dict[str, Any]:
    files = {name: _sha(ROOT / name) for name in SOURCE_CLOSURE}
    # The experiment entry already implements exact-root Git detection.  Its
    # receipt is nested while this posthoc adds the complete posthoc file set.
    return dict(experiment_source=experiment.source_identity(ROOT), files=files,
                content_sha256=content_id(files), parent_git_inheritance=False)


def _load_raw_receipt(
    receipt: dict[str, Any], *, expected_layer: str, expected_label: str,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    if receipt.get("status") != "SAVED":
        return None, dict(status="ENGINEERING_FAILURE", reason=receipt.get("reason", "raw observation not saved"),
                          source_receipt=receipt)
    path = Path(receipt["path"])
    try:
        if receipt.get("layer") != expected_layer or receipt.get("key_label") != expected_label:
            raise ValueError("raw observation receipt layer/key label mismatch")
        actual = _sha(path)
        if actual != receipt["sha256"]:
            raise ValueError("raw observation SHA mismatch")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != 768 or receipt.get("rows") != 768:
            raise ValueError("raw observation fixed directory must contain 768 rows")
        return rows, dict(status="SEALED", path=str(path), sha256=actual, rows=768,
                          key_label=receipt["key_label"], layer=receipt["layer"], truth_used=False)
    except Exception as exc:
        return None, dict(status="ENGINEERING_FAILURE", path=str(path),
                          reason=f"{type(exc).__name__}: {exc}", source_receipt=receipt)


def _condition_or_missing(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "EVALUATED":
        return dict(status="ENGINEERING_FAILURE", met=None, reason=row.get("reason", "evaluation unavailable"),
                    classification="ENGINEERING_FAILURE", scientific_pass=False)
    return method.descriptive_condition(row["evaluation"])


def _attribution(off: dict[str, Any], joint: dict[str, Any]) -> str:
    if joint["status"] != "SCORED":
        return joint["classification"]
    if not joint["met"]:
        return "VALID_FINITE_NEGATIVE"
    if off.get("status") == "SCORED" and off.get("met"):
        return "ATTRIBUTION_UNRESOLVED_OFF_ALSO_MEETS"
    if off.get("status") == "SCORED" and not off.get("met"):
        return "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    return "ATTRIBUTION_UNRESOLVED_OFF_UNAVAILABLE"


def run(run_result_path: Path, config_path: Path, output: Path) -> dict[str, Any]:
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    run_result_path, config_path = Path(run_result_path), Path(config_path)
    run_identity = _sha(run_result_path)
    manifest_path = run_result_path.with_name("raw_observation_manifest.json")
    manifest_identity = _sha(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "local-joint-raw-observation-manifest-v1"
            or manifest.get("truth_loaded") is not False):
        raise ValueError("truth-free raw observation manifest required")
    raw: dict[str, dict[str, list[dict[str, Any]] | None]] = {}
    seal = dict(schema="local-joint-raw-seal-v1", run_result_path=str(run_result_path),
                run_result_sha256=run_identity, manifest_path=str(manifest_path),
                manifest_sha256=manifest_identity, truth_loaded=False, entries={})
    for arm in method.ARMS:
        raw[arm] = {}
        observations = manifest.get("arms", {}).get(arm, {})
        for layer in method.LAYERS:
            raw[arm][layer] = {}
            for label in method.KEY_LABELS:
                entry = f"{layer}/{label}"
                rows, receipt = _load_raw_receipt(
                    observations.get(entry, dict(status="MISSING", reason="fixed raw slot absent")),
                    expected_layer=layer, expected_label=label)
                raw[arm][layer][label] = rows
                seal["entries"][f"{arm}/{entry}"] = receipt
    seal_receipt = _write(output / "raw_observation_seal.json", seal)

    # Truth/config is loaded only after the raw directory and its identities
    # have been independently sealed above.
    run_result = json.loads(run_result_path.read_text(encoding="utf-8"))
    if _truth_free_manifest_from_result(run_result) != manifest:
        raise ValueError("sealed raw manifest does not match the run receipt observations")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed = json.loads(FIXED_CONFIG.read_text(encoding="utf-8"))
    runtime.validate_config(config)
    if config != fixed or run_result.get("config") != config:
        raise ValueError("adopted posthoc requires the exact fixed run config and matching run receipt")
    truth = dict(key=config["carrier"]["key"], wrong_key=config["carrier"]["wrong_key"],
                 message_hex=config["carrier"]["message_hex"], loaded_after_raw_seal=True)
    message = bytes.fromhex(truth["message_hex"])

    evaluations: dict[str, Any] = {}
    for arm in method.ARMS:
        evaluations[arm] = {}
        for layer in method.LAYERS:
            evaluations[arm][layer] = {}
            for label, key in (("CORRECT", truth["key"]), ("WRONG", truth["wrong_key"])):
                rows = raw[arm][layer][label]
                if rows is None:
                    evaluations[arm][layer][label] = dict(
                        status="ENGINEERING_FAILURE", reason="raw observation unavailable",
                        auxiliary_wrong_key=(label == "WRONG"))
                    continue
                try:
                    value = method.evaluate_observations(rows, key=key, message=message)
                    evaluations[arm][layer][label] = dict(
                        status="EVALUATED", evaluation=value, auxiliary_wrong_key=(label == "WRONG"))
                except Exception as exc:
                    evaluations[arm][layer][label] = dict(
                        status="ENGINEERING_FAILURE", reason=f"{type(exc).__name__}: {exc}",
                        auxiliary_wrong_key=(label == "WRONG"))

    comparisons = {}
    for layer in method.LAYERS:
        off, joint = evaluations["OFF"][layer]["CORRECT"], evaluations["JOINT"][layer]["CORRECT"]
        comparisons[layer] = (method.compare_arms(off["evaluation"], joint["evaluation"])
                              if off["status"] == joint["status"] == "EVALUATED"
                              else dict(status="ENGINEERING_FAILURE", reason="correct-key arm evaluation unavailable"))
    correct_conditions = {
        arm: {layer: _condition_or_missing(evaluations[arm][layer]["CORRECT"])
              for layer in method.LAYERS}
        for arm in method.ARMS
    }
    wrong_conditions = {
        arm: {layer: dict(
            **_condition_or_missing(evaluations[arm][layer]["WRONG"]), auxiliary_only=True)
              for layer in method.LAYERS}
        for arm in method.ARMS
    }
    off_mp4 = correct_conditions["OFF"]["mp4"]
    joint_mp4 = correct_conditions["JOINT"]["mp4"]
    attribution = _attribution(off_mp4, joint_mp4)
    engineering_failure = any(
        correct_conditions[arm][layer]["classification"] == "ENGINEERING_FAILURE"
        for arm in method.ARMS for layer in method.LAYERS
    )
    run_complete = _run_is_complete(run_result)
    if not run_complete:
        outcome = "INCOMPLETE"
    elif engineering_failure:
        outcome = "ENGINEERING_FAILURE"
    else:
        outcome = attribution
    result = dict(
        schema="local-joint-known-grid-posthoc-v1",
        status=("INCOMPLETE" if not run_complete else
                "COMPLETE_WITH_ENGINEERING_FAILURE" if engineering_failure else "COMPLETE"),
        source_identity=_source_identity(), run_result=_run_receipt(run_result_path, run_identity, run_result),
        config=dict(path=str(config_path), sha256=_sha(config_path)), raw_seal=seal_receipt, truth=truth,
        evaluations=evaluations, correct_key_arm_differences=comparisons,
        correct_key_conditions=correct_conditions, wrong_key_conditions=wrong_conditions,
        mp4_conditions=dict(OFF=off_mp4, JOINT=joint_mp4), mp4_attribution=attribution,
        outcome_classification=outcome,
        wrong_key_role="auxiliary_only_not_a_primary_decision_or_FPR_estimate",
        evidence_ceiling="known-grid descriptive posthoc; not blind recovery, message reconstruction, FPR, or scientific PASS",
        scientific_pass=False, automatic_retry=False,
    )
    result = _json_safe(result)
    _write(output / "posthoc_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.run_result, args.config, args.output)
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
