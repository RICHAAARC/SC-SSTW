"""Fixed six saved-state real-tail points; continuous surrogate solver candidate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from main.tube_state import rgb_dct_continuous_solver as solver
from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as shared

ROOT = shared.ROOT
CONFIG_PATH = Path(__file__).parent / "configs/video_trajectory_blind_validation_v1.json"
PROTOCOL_PATH = ROOT / "docs/video_trajectory_blind_validation_v1.md"
MODULE = "experiments.wan_state_clock.video_trajectory_blind_validation_run"
POINTS = (46, 47, 48)
PLAN = dict(model_setup=2, transformer_feedback=168, scheduler_feedback=126,
            vae_decode=44, vae_encode=4, memory_receiver_score=42)


def load_config():
    candidate = json.loads(CONFIG_PATH.read_text())
    historical = shared.load_config()
    if (shared._sha(PROTOCOL_PATH) != candidate["protocol_sha256"]
            or shared._sha(shared.CONFIG_PATH) != candidate["historical_config_sha256"]
            or shared._sha(shared.PROTOCOL_PATH) != candidate["historical_protocol_sha256"]
            or candidate["solver"] != solver.SPEC
            or candidate["points"] != list(POINTS)
            or candidate["fixed_denominator"] != dict(sources=2,
                independent_points_per_source=3, real_tail_points=6, mp4_slots=0)
            or candidate["call_plan_max"] != PLAN):
        raise ValueError("candidate protocol/config binding mismatch")
    return candidate, historical


def source_receipt():
    paths = sorted(path for directory in ("main", "runtime", "experiments")
                   for path in (ROOT / directory).rglob("*.py"))
    paths += [CONFIG_PATH, PROTOCOL_PATH, shared.CONFIG_PATH, shared.PROTOCOL_PATH]
    files = {str(p.relative_to(ROOT)): shared._sha(p) for p in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                files=files, source_tree_sha256=digest,
                published_binding=False)


def preflight(saved_run: Path):
    candidate, historical = load_config()
    result_path = saved_run / "result.json"
    if shared._sha(result_path) != candidate["saved_result_sha256"]:
        raise ValueError("saved result byte identity mismatch")
    saved = json.loads(result_path.read_text())
    receipt_path = saved_run / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (saved["source_sha"] != candidate["saved_source_sha"]
            or receipt["status"] != "VALID"
            or receipt["expected_sha"] != candidate["saved_source_sha"]
            or receipt["actual_sha"] != candidate["saved_source_sha"]
            or receipt["files"]["config"]["sha256"] != candidate["historical_config_sha256"]
            or receipt["files"]["protocol"]["sha256"] != candidate["historical_protocol_sha256"]):
        raise ValueError("saved source/config identity mismatch")
    manifest = {}
    for case_id in shared.CASE_IDS:
        case = saved["cases"][case_id]
        required = {f"{kind}{index}" for index in (46, 47, 48, 49)
                    for kind in ("z", "v", "scheduler")} | {"off_terminal"}
        artifacts = case["generation"]["artifacts"]
        if set(artifacts) != required:
            raise ValueError("saved 13-artifact denominator mismatch")
        manifest[case_id] = {}
        for name in sorted(required):
            path = saved_run / case_id / "state" / f"{name}.pt"
            digest = shared._sha(path)
            if digest != artifacts[name]["sha256"]:
                raise ValueError(f"saved artifact identity mismatch: {case_id}/{name}")
            manifest[case_id][name] = dict(path=str(path), sha256=digest,
                                           bytes=path.stat().st_size)
    return saved, dict(status="VERIFIED", result_sha256=shared._sha(result_path),
        source_receipt_sha256=shared._sha(receipt_path), artifact_count=26,
        artifacts=manifest, no_model_loaded=True)


def initial_result(output, saved_run, candidate, historical, receipt, inputs):
    return dict(status="PREPARED", experiment_id=candidate["name"],
        config_sha256=shared._sha(CONFIG_PATH), protocol_sha256=shared._sha(PROTOCOL_PATH),
        source_receipt=receipt, input_receipt=inputs, saved_run=str(saved_run),
        fixed_denominator=candidate["fixed_denominator"], solver=solver.SPEC,
        historical_control=historical["control"], model=historical["model"],
        receiver_spec_sha256=historical["receiver_spec_sha256"], key_id=historical["receiver_key_id"],
        output_dir=str(output), result_path=str(output / "result.json"),
        evidence_ceiling=candidate["evidence_ceiling"],
        call_plan_max=PLAN, calls={k: dict(attempted=0, completed=0) for k in PLAN},
        cases={case_id: dict(status="PENDING", stage="PENDING", seed=seed,
            controls={f"FIXED_T{i}": [dict(index=i, outcome="PENDING")] for i in POINTS},
            budgets={}, phase_receipts=[], failures=[], basis=None,
            calls={k: dict(attempted=0, completed=0) for k in PLAN},
            environment_receipt=None, resources=None, release_receipt=None)
            for case_id, seed in zip(shared.CASE_IDS, shared.SEEDS, strict=True)})


class SixPointStore:
    def __init__(self, path, data):
        self.path, self.data, self.active_case = path, data, None

    @classmethod
    def open(cls, path):
        return cls(path, json.loads(path.read_text()))

    def case(self, case_id):
        return self.data["cases"][case_id]

    def save(self):
        rows = [row for case in self.data["cases"].values()
                for points in case["controls"].values() for row in points]
        if len(rows) != 6:
            raise RuntimeError("six-point denominator changed")
        outcomes = [r["outcome"] for r in rows]
        complete = ("ACCEPT", "ZERO_OPTIMUM", "NO_ACCEPTABLE_CANDIDATE")
        self.data["point_counts"] = dict(expected=6, accepted=outcomes.count("ACCEPT"),
            evaluated=sum(x in complete for x in outcomes),
            pending=outcomes.count("PENDING"),
            failed=sum(x not in (*complete, "PENDING") for x in outcomes))
        shared._atomic_json(self.path, self.data)

    def count(self, kind, completed):
        if kind not in PLAN:
            raise RuntimeError(f"unplanned call: {kind}")
        field = "completed" if completed else "attempted"
        for ledger, cap in ((self.data["calls"], PLAN[kind]),
                            (self.case(self.active_case)["calls"], PLAN[kind] // 2)):
            row = ledger[kind]
            row[field] += 1
            if row["attempted"] > cap or row["completed"] > row["attempted"]:
                raise RuntimeError(f"call cap/order: {kind}")
        self.save()


def feedback_arm(store, case_id, arm, backend, directions, key, **kwargs):
    """Candidate sequential entry: q/J are refreshed inside the shared loop."""
    return shared._feedback_arm(store, case_id, arm, backend, directions, key,
                               coefficient_solver=solver.select_coefficients, **kwargs)


def restore_backend(backend, saved_case, case_id, input_receipt):
    """Load pinned trusted local torch.save artifacts before any model load."""
    import torch
    from runtime.wan import trajectory
    from runtime.wan.rgb_dct_structured_feedback_backend import _assert_finite_history
    for name, entry in input_receipt["artifacts"][case_id].items():
        path = Path(entry["path"])
        if shared._sha(path) != entry["sha256"]:
            raise RuntimeError("IDENTITY_MISMATCH: artifact changed after preflight")
        is_scheduler = name.startswith("scheduler")
        value = torch.load(path, map_location="cpu", weights_only=not is_scheduler)
        fingerprint = trajectory.fingerprint(vars(value) if is_scheduler else value)
        if fingerprint != saved_case["generation"]["artifacts"][name]["fingerprint"]:
            raise RuntimeError(f"IDENTITY_MISMATCH: restored {name}")
        if is_scheduler:
            index = int(name.removeprefix("scheduler"))
            trajectory.validate_scheduler(value)
            _assert_finite_history(value)
            if value.step_index != index:
                raise RuntimeError("IDENTITY_MISMATCH: saved history cursor")
            backend.snapshots[index] = value
        else:
            if (not torch.is_tensor(value) or value.shape != (1, 16, 46, 40, 64)
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise RuntimeError("IDENTITY_MISMATCH: saved tensor shape/dtype/finite")
            if name == "off_terminal":
                backend.off_terminal = value
            else:
                backend.nodes.setdefault(int(name[1:]), {})[name[0]] = value


def run_independent_points(store, case_id, backend, directions, key, saved_case):
    """Each point resets z/history and the budget; errors retain every row."""
    for index in POINTS:
        arm = f"FIXED_T{index}"
        reference = next(p for p in saved_case["controls"]["MULTI46_47_48_49"]
                         if p["index"] == index)
        try:
            feedback_arm(store, case_id, arm, backend, directions, key,
                         independent_point=index, reference_point=reference)
        except Exception as exc:
            row = store.case(case_id)["controls"][arm][0]
            if row["outcome"] in ("PENDING", "ACCEPT", "ZERO_OPTIMUM",
                                   "NO_ACCEPTABLE_CANDIDATE"):
                row["prior_outcome"] = row["outcome"]
                row["outcome"] = "ENGINEERING_FAILURE"
            # Keep measured spent/response fields, but never count a verdict
            # as a completed point after a subsequent native-history failure.
            if arm in store.case(case_id)["budgets"]:
                store.case(case_id)["budgets"][arm].update(
                    status="FAILED", reason=row["outcome"])
            row["error"] = f"{type(exc).__name__}: {exc}"
            store.case(case_id)["failures"].append(dict(index=index, error=row["error"],
                                                        traceback=traceback.format_exc()))
            store.save()
            # A failed numerical solve leaves native state unchanged, but a
            # backend failure may leave GPU/VAE phase unsafe. Retain remaining.
            if row["outcome"] != "SOLVER_FAILURE":
                raise


def _mark_pending(store, case_id, reason):
    for rows in store.case(case_id)["controls"].values():
        for row in rows:
            if row["outcome"] == "PENDING":
                row.update(outcome="NOT_RUN_FAILURE", error=reason)
    store.save()


def recover_worker_failure(store, case_id, reason):
    """Invalidate provisional verdicts when a worker dies outside Python catch.

    A COMPLETE budget is the persisted point completion marker. Previously
    completed independent points remain valid; their case can still fail later.
    """
    case = store.case(case_id)
    case.update(status="FAILED", worker_error=reason)
    for arm, rows in case["controls"].items():
        budget = case["budgets"].get(arm, {})
        if budget.get("status") == "COMPLETE":
            continue
        for row in rows:
            if row["outcome"] in ("ACCEPT", "ZERO_OPTIMUM", "NO_ACCEPTABLE_CANDIDATE"):
                row.update(prior_outcome=row["outcome"],
                           outcome="ENGINEERING_FAILURE", error=reason)
                case["budgets"].setdefault(arm, {}).update(
                    status="FAILED", reason="WORKER_FAILURE_BEFORE_POINT_COMPLETION")
    _mark_pending(store, case_id, reason)

def run_case(store, case_id, saved, historical):
    from runtime.wan.rgb_dct_structured_feedback_backend import WanStructuredFeedbackBackend
    from runtime.wan.generation import prepare_generation
    from runtime.wan import trajectory
    case, saved_case = store.case(case_id), saved["cases"][case_id]
    store.active_case = case_id
    config = copy.deepcopy(historical)
    source = next(c for c in config["cases"] if c["id"] == case_id)
    config["generation"].update(prompt=source["prompt"], seed=source["seed"])
    backend = WanStructuredFeedbackBackend(config, store.count)
    started = time.perf_counter()
    try:
        case.update(status="RUNNING", stage="RESTORE_SAVED_STATE")
        store.save()
        case["environment_receipt"] = shared.environment_receipt(config)
        restore_backend(backend, saved_case, case_id, store.data["input_receipt"])
        case["stage"] = "MODEL_AND_CONDITIONING_SETUP_NO_PREFIX"
        store.count("model_setup", False)
        backend.pipe, initial, backend.prompt, backend.negative, backend.dtype = prepare_generation(config, load_vae=False)
        if trajectory.fingerprint(initial) != saved_case["generation"]["initial_noise_fingerprint"]:
            raise RuntimeError("IDENTITY_MISMATCH: recreated conditioning/noise seed")
        del initial
        backend.phase = "TRANSFORMER"
        store.count("model_setup", True)
        case["phase_receipts"].append(backend.to_vae_phase("SAVED_OFF_BASIS_VAE"))
        off_rgb, directions, single, basis = backend.build_basis(config["key_utf8"].encode())
        del off_rgb, single
        for name in ("raw_sha256", "masked_sha256"):
            if basis["single_geometry"][name] != saved_case["basis"]["single_geometry"][name]:
                raise RuntimeError(f"IDENTITY_MISMATCH: rebuilt basis {name}")
        basis_dir = Path(store.data["output_dir"]) / case_id / "basis"
        basis_dir.mkdir(parents=True, exist_ok=True)
        basis["full_byte_directions"] = []
        for j, direction in enumerate(directions):
            path = basis_dir / f"direction{j}.npy"
            np.save(path, direction, allow_pickle=False)
            basis["full_byte_directions"].append(dict(path=str(path),
                array_sha256=hashlib.sha256(np.ascontiguousarray(direction).tobytes()).hexdigest(),
                file_sha256=shared._sha(path), shape=list(direction.shape), dtype=str(direction.dtype)))
        basis["identity"] = "RAW_AND_MASKED_FULL_BYTE_MATCH"
        case["basis"] = basis
        store.save()
        case["phase_receipts"].append(backend.to_transformer_phase("SAVED_OFF_BASIS_RESUME"))
        run_independent_points(store, case_id, backend, directions,
                               config["key_utf8"].encode(), saved_case)
        case["status"] = "EVALUATED" if not case["failures"] else "FAILED"
    except Exception as exc:
        case.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        case["failures"].append(dict(stage=case["stage"], error=case["error"],
                                     traceback=traceback.format_exc()))
        _mark_pending(store, case_id, case["error"])
    finally:
        case["resources"] = backend.resources()
        try:
            backend.release()
            case["release_receipt"] = dict(status="COMPLETED")
        except Exception as exc:
            case["status"] = "FAILED"
            case["release_receipt"] = dict(status="FAILED", error=str(exc))
        case["elapsed_seconds"] = time.perf_counter() - started
        store.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-real-tail", action="store_true",
                        help="load model and execute the fixed six points; default is read-only preflight")
    args = parser.parse_args()
    candidate, historical = load_config()
    internal = os.environ.get("VIDEO_TRAJECTORY_FIXED_INTERNAL_CASE")
    if internal:
        if internal not in shared.CASE_IDS or not args.execute_real_tail:
            raise ValueError("invalid worker invocation")
        store = SixPointStore.open(args.output / "result.json")
        if (store.data["source_receipt"]["source_tree_sha256"] != source_receipt()["source_tree_sha256"]
                or store.data["config_sha256"] != shared._sha(CONFIG_PATH)
                or store.case(internal)["status"] != "PENDING"):
            raise ValueError("worker source/config/status changed")
        saved, inputs = preflight(args.saved_run)
        if inputs != store.data["input_receipt"]:
            raise ValueError("worker saved input changed")
        run_case(store, internal, saved, historical)
        return
    if args.output.exists():
        raise ValueError("fresh output directory required")
    args.output.mkdir(parents=True)
    # Persist six rows before preflight, so missing/corrupt inputs do not erase
    # the denominator. preflight itself never loads pickles or models.
    store = SixPointStore(args.output / "result.json", initial_result(
        args.output, args.saved_run, candidate, historical, source_receipt(), None))
    store.save()
    try:
        _, inputs = preflight(args.saved_run)
        store.data["input_receipt"] = inputs
    except Exception as exc:
        store.data.update(status="INPUT_FAILURE", error=f"{type(exc).__name__}: {exc}")
        for case_id in shared.CASE_IDS:
            _mark_pending(store, case_id, store.data["error"])
        store.save()
        return
    store.save()
    if not args.execute_real_tail:
        return
    store.data["status"] = "RUNNING"
    store.save()
    for case_id in shared.CASE_IDS:
        env = os.environ.copy()
        env["VIDEO_TRAJECTORY_FIXED_INTERNAL_CASE"] = case_id
        command = [sys.executable, "-m", MODULE, "--saved-run", str(args.saved_run),
                   "--output", str(args.output), "--execute-real-tail"]
        with (args.output / f"{case_id}_worker.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=historical["resources"]["case_timeout_seconds"], check=False)
                failure = None if result.returncode == 0 else f"WORKER_EXIT_{result.returncode}"
            except subprocess.TimeoutExpired:
                failure = "WORKER_TIMEOUT"
        store = SixPointStore.open(args.output / "result.json")
        if failure:
            recover_worker_failure(store, case_id, failure)
        store.save()
    store.data["status"] = ("SIX_POINT_EXECUTION_COMPLETE" if all(
        c["status"] == "EVALUATED" for c in store.data["cases"].values()) else "INCOMPLETE")
    store.save()


if __name__ == "__main__":
    main()
