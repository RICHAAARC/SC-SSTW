"""One consumed fresh-content attempt with isolated GPU/CPU worker processes."""
from pathlib import Path
import subprocess
import sys

from main.tube_state.b_joint_time_relative_validation_v1 import (ARMS, INNER_BUDGET, STAGES,
    construction_maps, denominators, operation_plan, stage_operations, units, validate)
from runtime.wan.native_baseline_storage_v1 import Journal, now, read_json, recover, reserve_local, write_json
from runtime.wan.b_joint_time_relative_evaluation_v1 import evaluate_saved

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "experiments/wan_state_clock/configs/b_joint_time_relative_validation_v1.json"


def recover_attempt(output):
    """Merge child journals in memory only; never resume a stage or evaluation."""
    output = Path(output)
    result = recover(output)
    rows = {r["id"]: r for r in result["operations"]}
    for stage in STAGES:
        try:
            rows.update({r["id"]: r for r in recover(output / "stages" / stage)["operations"]})
        except (OSError, ValueError, KeyError) as error:
            result["read_errors"].append(dict(stage=stage, error=str(error)))
    result["operations"] = list(rows.values())
    return result


class InnerEvents:
    def __init__(self, folder, stage):
        kinds = dict(prepare=("transformer_conditional", "transformer_unconditional", "native_step",
                              "payload_gradient", "D181", "baseline_head"), solve=("K", "KT"),
                     replay=("head_replay",), receive=())[stage]
        self.path, self.phase = Path(folder) / "internal_calls.json", None
        self.rows = {k: dict(limit=INNER_BUDGET[k], started=0, returned=0) for k in kinds}
        self.save()

    def save(self):
        write_json(self.path, self.rows)

    def __call__(self, kind, returned):
        row = self.rows[kind]
        if not returned:
            limit = 4200 if kind == "K" and self.phase == "solver" else row["limit"]
            if row["started"] >= limit:
                raise RuntimeError("Fixed internal call budget consumed: " + kind)
            row["started"] += 1
        else:
            if row["returned"] >= row["started"]:
                raise RuntimeError("Internal return without a reserved call: " + kind)
            row["returned"] += 1
        self.save()


def worker(stage, output, scratch, *, backend_factory=None):
    """Internal one-shot worker; per-stage receipt prevents an implicit restart."""
    output, scratch = Path(output), Path(scratch)
    if stage not in STAGES:
        raise ValueError("Unknown fixed stage")
    if read_json(output / "attempt.json").get("status") != "RUNNING":
        return dict(status="READ_ONLY", reason="No active parent reservation")
    folder = output / "stages" / stage
    if folder.exists() and any(folder.iterdir()):
        return dict(status="READ_ONLY", reason="Worker already consumed", saved=recover(folder))
    index = STAGES.index(stage)
    if index and read_json(output / "stages" / STAGES[index-1] / "stage_result.json")["status"] != "COMPLETED":
        raise ValueError("Prior stage is not complete; no refill or retry")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "stage_attempt.json", dict(status="STARTED", consumed=True, time=now()))
    config = read_json(output / "protocol.json")
    journal = Journal(folder, stage_operations(config, stage))
    events = InnerEvents(folder, stage)
    backend, failure, resources = None, None, {}
    try:
        if backend_factory is None:
            from runtime.wan.b_joint_time_relative_physical_v1 import Backend
            backend_factory = Backend
        backend = backend_factory(config, output, scratch, events, write_json)
        def call(key, function):
            events.phase = key
            return journal.call(key, function)
        if stage == "prepare":
            call("generation_load", backend.generation_load)
            call("generate_P0", backend.generate)
            call("capture_vae_load", backend.vae_load)
            call("capture", backend.capture)
            call("head_prepare", backend.head)
        elif stage == "solve":
            for key, function in (("q0", backend.q0), ("target", backend.target), ("solver", backend.solve),
                                  ("prediction", backend.predict), ("handoff", backend.handoff)):
                call(key, function)
        elif stage == "replay":
            call("replay_vae_load", backend.vae_load)
            receipt = call("native_replay", backend.replay)
            write_json(output / "native_receipt.json", receipt)
        else:
            from main.tube_state.b_joint_time_packet_a_relative_v1 import diagnose
            maps = construction_maps()  # Editor only; never passed to hard receive/compare.
            statuses = read_json(output / "unit_statuses.json")
            for arm in ARMS:
                preclip = backend.preclip(arm)
                pixels = backend.quantize(preclip)
                path = output / "media" / arm / "first.mp4"
                stem = arm + "/first/"
                call(stem+"encode", lambda: backend.encode(pixels, path))
                probe = call(stem+"probe", lambda: backend.probe(path, 181))
                write_json(path.with_suffix(".probe.json"), probe)
                first = call(stem+"decode_RGB", lambda: backend.decode(path, 181))
                del pixels, preclip
                for unit in units(config):
                    if unit["arm"] != arm:
                        continue
                    uid, n = unit["id"], unit["frames"]
                    path = output / "media" / uid / "received.mp4"
                    edited = first[maps[unit["view"]]]
                    call(uid+"/encode", lambda: backend.encode(edited, path))
                    probe = call(uid+"/probe", lambda: backend.probe(path, n))
                    write_json(path.with_suffix(".probe.json"), probe)
                    received = call(uid+"/decode_RGB", lambda: backend.decode(path, n))
                    # The RGB readback has a fixed physical slot. No E41/identity reads follow.
                    import numpy as np
                    cache = scratch / "received" / uid
                    cache.mkdir(parents=True, exist_ok=True)
                    np.save(cache / "RGB.npy", received, allow_pickle=False)
                    y = call(uid+"/decode_Y", lambda: backend.decode_y(path, n, probe))
                    np.save(cache / "Y.npy", y, allow_pickle=False)
                    blind_folder = output / "blind" / uid
                    packet = call(uid+"/hard_receive", lambda: backend.hard_receive(y, blind_folder))
                    if len(packet.get("nodes", [])) != n:
                        raise ValueError("Actual hard packet frame count differs from this unit")
                    relative = call(uid+"/relative_compare", lambda: diagnose(packet, uid))
                    write_json(blind_folder / "relative.json", relative)
                    statuses[uid].update(status="BLIND_SAVED", file=str((blind_folder / "relative.json").relative_to(output)))
                    write_json(output / "unit_statuses.json", statuses)
                    write_json(output / "progress.json", dict(stage=stage, unit=uid, status="BLIND_SAVED"))
                    print(uid, "blind saved", flush=True)
                    del edited, received, y, packet, relative
                del first
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        if backend is not None:
            try:
                resources = backend.close()
            except BaseException as error:
                if failure is None:
                    failure = dict(type=type(error).__name__, message="Resource close failed: " + str(error))
        snapshot = journal.snapshot()
        result = dict(stage=stage, status="COMPLETED" if failure is None else "STOPPED_FIXED_ATTEMPT",
                      failure=failure, internal_calls=events.rows, resources=resources, **snapshot)
        write_json(folder / "stage_result.json", result)
    return result


def subprocess_worker(stage, output, scratch):
    root = Path(__file__).resolve().parents[2]
    args = [sys.executable, "-u", str(root / "experiments/wan_state_clock/b_joint_time_relative_validation_v1_run.py"),
            "--worker-stage", stage, "--output", str(output), "--scratch", str(scratch)]
    process = None
    try:
        with (Path(output) / (stage + ".log")).open("a") as log:
            process = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(stage, "same consumed worker running; see", log.name, flush=True)
        receipt = read_json(Path(output) / "stages" / stage / "stage_result.json")
        if process.returncode != 0 or receipt.get("status") != "COMPLETED":
            raise RuntimeError(f"{stage} stopped (exit={process.returncode}): {receipt.get('failure')}")
        return receipt
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def execute_reserved(config, output, scratch, *, worker_runner=None, validator=validate, defer_post=False):
    output, scratch = Path(output), Path(scratch)
    attempt = read_json(output / "attempt.json")
    if attempt.get("status") != "RESERVED" or (output / "plan.json").exists():
        return recover_attempt(output)
    attempt.update(status="RUNNING", consumed=True, time=now())
    write_json(output / "attempt.json", attempt)
    write_json(output / "protocol.json", config)
    write_json(output / "plan.json", operation_plan(config))
    write_json(output / "unit_statuses.json", {u["id"]: u for u in units(config)})
    write_json(output / "private/attack_truth.json", construction_maps())
    failure, stage_results = None, {}
    try:
        validator(config)
        scratch.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            receipt = (worker_runner or subprocess_worker)(stage, output, scratch)
            stage_results[stage] = receipt
            if receipt.get("status") != "COMPLETED":
                raise RuntimeError(f"{stage} did not complete; fixed attempt stops")
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    ledger = {r["id"]: r for r in operation_plan(config)}
    accounting_errors = []
    for stage in STAGES:
        try:
            for row in recover(output / "stages" / stage)["operations"]:
                ledger[row["id"]] = row
        except (OSError, ValueError, KeyError) as error:
            accounting_errors.append(dict(stage=stage, error=str(error)))
    write_json(output / "ledger.json", list(ledger.values()))
    write_json(output / "blind_terminated.json", dict(status="BLIND_TERMINATED", time=now(), failure=failure))
    result = dict(status="BLIND_SAVED_EVALUATION_PENDING", failure=failure,
                  denominators=denominators(config), stage_results=stage_results,
                  accounting_errors=accounting_errors, scientific_pass=None, E41=0, identity_new_reads=0)
    write_json(output / "result.json", result)
    try:
        if not defer_post:
            evaluate_saved(output)
    except BaseException as error:
        result["evaluation_failure"] = dict(type=type(error).__name__, message=str(error))
        write_json(output / "evaluation_failure.json", result["evaluation_failure"])
    result["status"] = "STOPPED_FIXED_ATTEMPT" if failure or accounting_errors or result.get("evaluation_failure") else "COMPLETED"
    result["post_deferred"] = defer_post
    write_json(output / "result.json", result)
    attempt.update(status=result["status"], consumed=True, finished_at=now())
    write_json(output / "attempt.json", attempt)
    return result


def run(config, output, scratch, **kwargs):
    if not reserve_local(output):
        return recover_attempt(output)
    return execute_reserved(config, output, scratch, **kwargs)
