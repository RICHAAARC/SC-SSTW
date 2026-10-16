"""Fixed two-branch saved-input VJP; no scheduler/VAE/writer and no physical retry."""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import threading
import time

from main.tube_state import local_joint_p0_input_vjp_v1 as method
from runtime.wan.local_joint_p0_checkpoint_v1 import checkpointed_branch

KINDS = ("transformer_prepare", "dit_forward", "dit_input_vjp")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_tensor(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def tensor(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def read_state(output):
    output = Path(output)
    problems = []
    for name in ("result.json", "result.previous.json"):
        path = output / name
        if not path.exists():
            continue
        try:
            result = json.loads(path.read_text())
            if not isinstance(result.get("calls"), list) or set(result.get("branches", {})) != set(method.BRANCHES):
                raise ValueError("result must retain two branch slots and physical call rows")
            rows = {r["id"]: r for r in result["calls"]}
            for sidecar in (output / "calls").glob("*.json"):
                row = json.loads(sidecar.read_text())
                rows[row["id"]] = row
            result["calls"] = sorted(rows.values(), key=lambda r: r["id"])
            result.setdefault("recovery_notes", []).extend(problems)
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(dict(path=str(path), reason=repr(exc)))
    if problems or (output.exists() and any(output.iterdir())):
        raise OSError("existing P0 task has no readable state; no new physical work: " + str(problems))
    return None


def initial_result(config):
    return dict(status="RUNNING", selection_status="P0_ADOPTED_BY_USER_20261016",
                config=copy.deepcopy(config), calls=[], failures=[], resources=[], replay={},
                branches={b: dict(status="MISSING", reason="not_run", forward="NOT_RUN", vjp="NOT_RUN")
                          for b in method.BRANCHES}, expected_cost=method.EXPECTED,
                automatic_retry=False, scientific_pass=False, actual_model_execution="NOT_STARTED",
                ceiling=method.CEILING, created_at_unix=time.time())


def zero_physical_record(result, *, report=False):
    """Positive evidence of no entry, including stored counters before recount."""
    if not isinstance(result, dict):
        return False
    if not report and (result.get("actual_model_execution") != "NOT_STARTED" or result.get("calls") != []):
        return False
    if result.get("resources") != [] or result.get("replay") != {}:
        return False
    for field, kinds in (("counts", KINDS), ("block_counts", ("forward", "recompute"))):
        counts = result.get(field)
        if not isinstance(counts, dict) or not set(kinds).issubset(counts):
            return False
        for row in counts.values():
            if not isinstance(row, dict) or row.get("attempted") != 0 or any(type(v) is not int or v != 0 for v in row.values()):
                return False
    rows = result.get("branches", {})
    return isinstance(rows, dict) and set(rows) == set(method.BRANCHES) and all(
        isinstance(row, dict) and row.get("status") == "MISSING" and row.get("forward") == row.get("vjp") == "NOT_RUN"
        for row in rows.values())


def source_precheck_can_continue(output, result):
    """A failed precheck may precede the first physical attempt, never a retry."""
    if result.get("model_work_resume_blocked_reason") or result.get("recovery_notes"):
        return False
    failures = result.get("failures", [])
    if result.get("status") != "ENGINEERING_FAILURE" or not isinstance(failures, list) or not failures or any(not isinstance(f, dict) for f in failures):
        return False
    if not any(f.get("stage") == "saved_source" for f in failures) or any(
            f.get("stage") not in ("saved_source", "saved_input_or_exact_d") for f in failures):
        return False
    if not zero_physical_record(result):
        return False
    output = Path(output)
    try:
        # A receipt, partial branch output or unreadable prior record makes the
        # physical history uncertain. Do not erase that uncertainty by recounting.
        for directory in (output / "calls", *(output / b for b in method.BRANCHES)):
            if directory.exists() and any(directory.iterdir()):
                return False
        if list(output.glob("result*.tmp")):
            return False
        for name in ("result.json", "result.previous.json", "report.json"):
            path = output / name
            if path.exists() and not zero_physical_record(json.loads(path.read_text()), report=name == "report.json"):
                return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


def needs_model_work(output):
    """Shared notebook/runner decision; no torch/model dependency is imported."""
    result = read_state(output)
    return result is None or source_precheck_can_continue(output, result)


def resource_snapshot(device):
    import resource
    import torch
    row = dict(time_unix=time.time(), host_rss_bytes=None,
               host_process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               cuda_allocated_bytes=None, cuda_reserved_bytes=None,
               cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None)
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                row["host_rss_bytes"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    if str(device).startswith("cuda") and torch.cuda.is_initialized():
        row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                   cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return row


def recount(result):
    states = ("RETURNED", "FAILED", "INTERRUPTED_COMPLETION_UNKNOWN", "RUNNING")
    result["counts"] = {kind: dict(attempted=sum(r["kind"] == kind for r in result["calls"]),
                                 **{s: sum(r["kind"] == kind and r["status"] == s for r in result["calls"])
                                    for s in states}) for kind in KINDS}
    result["block_counts"] = {phase: dict(attempted=sum(r[phase] != "NOT_RUN"
                                             for b in result["replay"].values() for r in b.get("blocks", [])),
                                        **{s: sum(r[phase] == s for b in result["replay"].values()
                                                  for r in b.get("blocks", [])) for s in states})
                              for phase in ("forward", "recompute")}
    result["fixed_denominators"] = dict(branches=2, forward_units=2, input_vjp_units=2,
                                       new_video=0, receiver_windows=0, independent_new_sources=0)


class Journal:
    def __init__(self, output, result, device):
        self.output, self.result, self.device = Path(output), result, device
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.stage = None
        self.thread = None

    def save(self):
        with self.lock:
            recount(self.result)
            path = self.output / "result.json"
            if path.exists():
                # Keep the last readable state without replacing it with a torn copy.
                try:
                    previous = json.loads(path.read_text())
                except (OSError, ValueError):
                    previous = None
                if previous is not None:
                    write_json(self.output / "result.previous.json", previous)
            write_json(path, self.result)

    def sample(self):
        with self.lock:
            snapshot = resource_snapshot(self.device)
            if self.stage is not None:
                self.stage["latest"] = snapshot
                rss = snapshot["host_rss_bytes"]
                if rss is not None:
                    self.stage["sampled_host_rss_peak_bytes"] = max(self.stage.get("sampled_host_rss_peak_bytes", 0), rss)
            return snapshot

    def start_stage(self, name):
        import torch
        if str(self.device).startswith("cuda") and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.stage = dict(name=name, started_at_unix=time.time(), host_sample_interval_seconds=.5)
        self.result["resources"].append(self.stage)
        self.stage["start"] = self.sample()
        self.save()

    def end_stage(self):
        if self.stage is not None:
            self.stage["end"] = self.sample()
            self.stage["finished_at_unix"] = time.time()
            self.stage = None
        self.save()

    def __enter__(self):
        def monitor():
            checkpoint = time.monotonic()
            while not self.stop.wait(.5):
                try:
                    self.sample()
                    if time.monotonic() - checkpoint >= 60:
                        self.save()
                        print("P0 checkpoint", self.result["status"], self.result.get("counts"), flush=True)
                        checkpoint = time.monotonic()
                except Exception as exc:
                    self.result.setdefault("monitor_errors", []).append(repr(exc))
                    return
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        self.end_stage()


def validate_config(config):
    if config.get("branches") != list(method.BRANCHES) or config.get("index") != 49:
        raise ValueError("P0 requires the two fixed step49 branches")
    if config.get("transformer_dtype") != "bfloat16" or config.get("block_count") != 30:
        raise ValueError("P0 uses saved BF16 arithmetic and 30 native blocks")
    for name in ("source_run", "tail_run"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(name + " must locate actual saved files")
    # Source/schema/version labels are optional notes, never admission checks.


def load_source(config, *, shape=method.LATENT_SHAPE, rois=method.ROIS, time_range=(1, 45)):
    import torch
    source, tail = Path(config["source_run"]), Path(config["tail_run"])
    x = tensor(tail / "native/CONTROL_TAIL_after48.pt")
    predictions = tensor(tail / "native/CONTROL_TAIL_step49_predictions.pt")
    context = tensor(source / "native/context.pt")
    fork = tensor(source / "native/fork48.pt")
    unit = tensor(source / "units/gradient.pt")
    zero = tensor(source / "ZERO/terminal_latent.pt")
    method.finite_tensor(x, shape, "CONTROL_TAIL_after48", torch.float32)
    for name in ("input", *method.BRANCHES):
        method.finite_tensor(predictions[name], shape, "saved prediction " + name, torch.float32)
    if not torch.equal(predictions["input"], x):
        raise ValueError("saved step49 predictions belong to a different numerical input")
    method.finite_tensor(zero, shape, "saved ZERO", torch.float32)
    method.finite_tensor(unit["input_terminal"], shape, "saved gradient input", torch.float32)
    if not torch.equal(unit["input_terminal"], zero):
        raise ValueError("exact saved d gradient input differs from the original ZERO point")
    d = unit["direction"].detach()
    method.direction_contract(d, shape, rois, time_range)
    for name in ("prompt", "negative"):
        value = method.finite_tensor(context[name], name="saved " + name, dtype=torch.bfloat16)
        if value.ndim != 3 or value.shape[0] != x.shape[0] or not value.numel():
            raise ValueError("saved prompt/negative embedding shape invalid")
    timesteps = fork["scheduler"]["state"]["timesteps"]
    method.finite_tensor(timesteps, (50,), "saved 50-step timesteps")
    return dict(input=x, direction=d, prompt=context["prompt"], negative=context["negative"],
                timestep=timesteps[49].reshape(1), predictions={b: predictions[b] for b in method.BRANCHES},
                context=dict(saved_dtype_label=context.get("dtype"), actual_input_cast="bfloat16",
                             actual_embedding_dtypes={k: str(context[k].dtype) for k in ("prompt", "negative")},
                             saved_prediction_context=predictions.get("execution_context"),
                             source_run=str(source), tail_run=str(tail), direction_source="units/gradient.pt[direction]"))


def prepare_transformer(config, device):
    import torch
    from diffusers import WanTransformer3DModel
    # No pipeline, text encoder, scheduler or attention-processor replacement.
    model = WanTransformer3DModel.from_pretrained(config["model"]["id"], subfolder="transformer",
                                                torch_dtype=torch.bfloat16).eval()
    model.disable_gradient_checkpointing()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def validate_model_arithmetic(model):
    import torch
    keep = tuple(getattr(model, "_keep_in_fp32_modules", ()) or ())
    invalid = []
    for name, parameter in model.named_parameters():
        native_fp32 = any(token in name.split(".") for token in keep)
        allowed = (torch.bfloat16, torch.float32) if native_fp32 else (torch.bfloat16,)
        if parameter.dtype not in allowed:
            invalid.append(dict(name=name, dtype=str(parameter.dtype), native_fp32_exception=native_fp32))
    if invalid:
        raise ValueError("actual Transformer parameter arithmetic differs from BF16/native-FP32 path: " + str(invalid[:8]))
    return dict(native_keep_in_fp32_modules=list(keep),
                parameter_dtypes={str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
                                  for dtype in {p.dtype for p in model.parameters()}})


def terminate_phases(result, branch, fallback):
    row = result["branches"][branch]
    for phase, kind in (("forward", "dit_forward"), ("vjp", "dit_input_vjp")):
        if row[phase] != "RUNNING":
            continue
        calls = [c for c in result["calls"] if c["kind"] == kind and c["branch"] == branch]
        state = calls[-1]["status"] if calls else fallback
        row[phase] = state if state in ("FAILED", "INTERRUPTED_COMPLETION_UNKNOWN") else fallback


def save_report(output, result):
    recount(result)
    report = dict(status=result["status"], branches=result["branches"], counts=result["counts"],
                  block_counts=result["block_counts"], fixed_denominators=result["fixed_denominators"],
                  resources=result["resources"], replay=result["replay"], failures=result["failures"],
                  current_attempt=result.get("current_attempt"), preflight_history=result.get("preflight_history", []),
                  model_work_resume_blocked_reason=result.get("model_work_resume_blocked_reason"),
                  scientific_pass=False, ceiling=method.CEILING)
    write_json(Path(output) / "report.json", report)
    return report


def resume_report(output, result, *, reason="existing task: no repeated physical work"):
    """A new process cannot recreate a lost graph under the one-preparation budget."""
    import torch
    if not source_precheck_can_continue(output, result):
        result.setdefault("model_work_resume_blocked_reason", "physical work entered or zero-call precheck evidence is uncertain; report-only")
    for call in result["calls"]:
        if call["status"] == "RUNNING":
            call.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason=reason)
            write_json(Path(output) / "calls" / (call["id"] + ".json"), call)
    for ledger in result.get("replay", {}).values():
        for block in ledger.get("blocks", []):
            for phase in ("forward", "recompute"):
                if block[phase] == "RUNNING":
                    block[phase] = "INTERRUPTED_COMPLETION_UNKNOWN"
    root_error = None
    try:
        saved_input = tensor(Path(output) / "saved_input.pt")
        method.finite_tensor(saved_input, method.LATENT_SHAPE, "saved P0 input", torch.float32)
        saved_d = tensor(Path(output) / "saved_cotangent.pt")
        method.direction_contract(saved_d, method.LATENT_SHAPE, method.ROIS)
        result["saved_evidence_status"] = "READABLE_VALID_NUMERICAL_INPUTS"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError) as exc:
        root_error = repr(exc)
        result["saved_evidence_status"] = "MISSING_OR_INVALID"
        result["failures"].append(dict(stage="saved_input_or_exact_d", reason=root_error))
    for branch, row in result["branches"].items():
        terminate_phases(result, branch, "INTERRUPTED_COMPLETION_UNKNOWN")
        directory = Path(output) / branch
        prior_status = row["status"]
        try:
            if (directory / "forward.json").exists():
                output_value = tensor(directory / "output.pt")
                method.finite_tensor(output_value, method.LATENT_SHAPE, "saved branch output", torch.float32, allow_nonfinite=True)
                row["output_statistics"] = method.statistics(output_value)
                method.finite_tensor(output_value, method.LATENT_SHAPE, "saved branch output", torch.float32)
                row["forward"] = "SAVED"
                row["comparison"] = json.loads((directory / "forward.json").read_text())["comparison"]
            if (directory / "vjp.json").exists():
                gradient = tensor(directory / "input_cotangent.pt")
                method.finite_tensor(gradient, method.LATENT_SHAPE, "saved input cotangent", torch.float32, allow_nonfinite=True)
                row["cotangent_statistics"] = method.statistics(gradient)
                row["vjp"] = "SAVED"
                if prior_status not in ("FAILED", "NONFINITE"):
                    row["status"] = "COMPLETE" if row["cotangent_statistics"]["finite"] else "NONFINITE"
            elif row["status"] not in ("FAILED", "NONFINITE"):
                row.update(status="MISSING", reason="lost/unentered graph cannot be recreated within the fixed physical budget")
            if row.get("forward") == "SAVED" and not (directory / "forward.json").exists():
                raise OSError("saved forward receipt missing")
            if row.get("vjp") == "SAVED" and not (directory / "vjp.json").exists():
                raise OSError("saved VJP receipt missing")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError) as exc:
            row.update(status="MISSING", reason="saved unit unreadable; no replacement work: " + repr(exc))
            result["failures"].append(dict(stage="saved_" + branch, reason=repr(exc)))
        if root_error is not None:
            row.update(evidence_status_before_root_failure=row["status"], status="MISSING",
                       reason="saved input/exact d unavailable or invalid; no replacement work: " + root_error)
    if all(r["status"] == "COMPLETE" for r in result["branches"].values()):
        result["status"] = "COMPLETE"
    elif result["status"] == "RUNNING":
        result["status"] = "INTERRUPTED"
    elif result["status"] == "COMPLETE":
        result["status"] = "ENGINEERING_FAILURE"
    result["resume_policy"] = reason
    save_report(output, result)
    Journal(output, result, "cpu").save()
    return result


def finalize_interrupted(output, reason):
    result = read_state(output)
    return None if result is None else resume_report(output, result, reason=reason)


def run(config, output, *, preparer=None, source_loader=None, expected_blocks=30):
    import torch
    output = Path(output)
    existing = read_state(output)
    if existing is not None and not source_precheck_can_continue(output, existing):
        return resume_report(output, existing)
    validate_config(config)
    for name in ("source_run", "tail_run"):
        source = Path(config[name]).resolve()
        if source == output.resolve() or source in output.resolve().parents or output.resolve() in source.parents:
            raise ValueError("P0 output must be separate from both saved source directories")
    output.mkdir(parents=True, exist_ok=True)
    result = initial_result(config) if existing is None else existing
    if existing is not None:
        snapshot = copy.deepcopy(existing)
        snapshot.pop("preflight_history", None)
        result.setdefault("preflight_history", []).append(snapshot)
        result.update(status="RUNNING", config=copy.deepcopy(config), branches=initial_result(config)["branches"],
                      resume_policy="zero-call saved-source precheck repeated before first physical attempt")
        for key in ("finished_at_unix", "saved_evidence_status"):
            result.pop(key, None)
    device = config.get("device", "cuda")
    journal = Journal(output, result, device)
    model = None
    stage = "saved_source"
    attempt = str(time.time_ns())
    result["attempt_id"] = attempt
    failure_start = len(result["failures"])
    result["current_attempt"] = dict(id=attempt, failure_start_index=failure_start, status="RUNNING")

    def call(kind, branch, fn):
        limit = method.EXPECTED[kind]
        if sum(r["kind"] == kind for r in result["calls"]) >= limit:
            raise RuntimeError("fixed physical budget exhausted: " + kind)
        row = dict(id=attempt + "_" + str(len(result["calls"])).zfill(3), kind=kind,
                   branch=branch, status="RUNNING", started_at_unix=time.time())
        result["calls"].append(row)
        print("P0 enter", kind, branch, flush=True)
        sidecar = output / "calls" / (row["id"] + ".json")
        write_json(sidecar, row)
        journal.start_stage(kind + ("_" + branch if branch else ""))
        try:
            value = fn()
            if str(device).startswith("cuda") and torch.cuda.is_initialized():
                torch.cuda.synchronize(device)
            row["status"] = "RETURNED"
            return value
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED_COMPLETION_UNKNOWN", reason=repr(exc))
            raise
        finally:
            row["finished_at_unix"] = time.time()
            write_json(sidecar, row)
            journal.end_stage()
            print("P0 exit", kind, branch, row["status"], flush=True)

    def branch_unit(branch, source):
        directory = output / branch
        row = result["branches"][branch]
        embedding = source["prompt" if branch == "conditional" else "negative"].to(device)
        # FP32 leaf: its forward BF16 cast is exactly the original input conversion.
        x = source["input"].to(device).detach().requires_grad_(True)
        d = source["direction"].to(device)
        timestep = source["timestep"].to(device)
        prediction = gradient = None
        def replay_update(value):
            result["replay"][branch] = value
            journal.save()
        try:
            with checkpointed_branch(model, replay_update, root=config.get("spool_root")) as ledger:
                row["forward"] = "RUNNING"
                prediction = call("dit_forward", branch, lambda: model(
                    hidden_states=x.to(torch.bfloat16), timestep=timestep.expand(x.shape[0]),
                    encoder_hidden_states=embedding, attention_kwargs=None, return_dict=False)[0])
                actual_output_dtype = prediction.dtype
                prediction = prediction.float()
                # Save raw forward even if nonfinite; never label it as a completed VJP.
                saved = prediction.detach().cpu()
                save_tensor(directory / "output.pt", saved)
                row.update(forward="SAVED", model_output_dtype=str(actual_output_dtype),
                           embedding_dtype=str(embedding.dtype), output_statistics=method.statistics(saved),
                           comparison=method.difference(saved, source["predictions"][branch]))
                write_json(directory / "forward.json", dict(comparison=row["comparison"],
                           output_statistics=row["output_statistics"], context=source["context"],
                           model_output_dtype=row["model_output_dtype"], embedding_dtype=row["embedding_dtype"],
                           preparation_call_id=result["calls"][0]["id"], attempt_id=attempt))
                journal.save()
                if actual_output_dtype != torch.bfloat16:
                    raise ValueError("actual Transformer branch output is not native BF16; saved its FP32 values for diagnosis")
                method.finite_tensor(prediction, source["input"].shape, "new branch prediction", torch.float32)
                if any(r["forward"] != "RETURNED" for r in ledger.rows):
                    raise RuntimeError("native Transformer did not traverse every expected checkpoint block")
                row["vjp"] = "RUNNING"
                gradient = call("dit_input_vjp", branch, lambda: torch.autograd.grad(
                    prediction, x, grad_outputs=d, retain_graph=False, create_graph=False)[0])
                saved_gradient = gradient.detach().cpu()
                save_tensor(directory / "input_cotangent.pt", saved_gradient)
                row.update(vjp="SAVED", cotangent_statistics=method.statistics(saved_gradient))
                parameter_gradients_created = any(p.grad is not None or p.requires_grad for p in model.parameters())
                write_json(directory / "vjp.json", dict(cotangent_statistics=row["cotangent_statistics"],
                           output_cotangent="exact saved d, unscaled", input_coordinate="FP32 before native BF16 cast",
                           saved_direction_dtype=str(source["direction"].dtype), grad_outputs_dtype=str(d.dtype),
                           output_dtype=str(prediction.dtype), input_cotangent_dtype=str(gradient.dtype),
                           arithmetic_note="d is passed unchanged; native autograd converts cotangents along the FP32 output/BF16 path",
                           mathematical_discrete_derivative=False, parameter_gradients_created=parameter_gradients_created))
                row["status"] = "COMPLETE" if row["cotangent_statistics"]["finite"] else "NONFINITE"
                if parameter_gradients_created:
                    raise RuntimeError("P0 must not create parameter gradients")
                journal.save()
                if row["status"] == "NONFINITE":
                    raise FloatingPointError("nonfinite input cotangent saved; no retry")
        finally:
            prediction = gradient = x = d = embedding = timestep = None
            gc.collect()

    with journal:
        try:
            journal.save()
            source = (source_loader or load_source)(config)
            result["source_context"] = source["context"]
            # Persist actual inputs for same-task recovery/comparison without model work.
            save_tensor(output / "saved_input.pt", source["input"])
            save_tensor(output / "saved_cotangent.pt", source["direction"])
            result["saved_evidence_status"] = "READABLE_VALID_NUMERICAL_INPUTS"
            stage = "transformer_prepare"
            result["actual_model_execution"] = "STARTED"
            model = call("transformer_prepare", None, lambda: (preparer or prepare_transformer)(config, device))
            if len(model.blocks) != expected_blocks:
                raise ValueError("loaded Transformer block count differs from the fixed P0 budget")
            if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError("P0 requires eval/frozen Transformer with no parameter gradients")
            # from_pretrained preserves the model's native keep-in-FP32 exceptions.
            # Record these; do not force-cast norms/time embeddings to satisfy a label.
            result["arithmetic"] = validate_model_arithmetic(model)
            result["model_device_path"] = ["from_pretrained CPU", str(device), "release after serial branches"]
            for branch in method.BRANCHES:
                stage = branch
                try:
                    branch_unit(branch, source)
                except Exception as exc:
                    # Adopted P0 explicitly stops on OOM/nonfinite. Other branch-local
                    # errors do not cancel the other independently budgeted branch.
                    if isinstance(exc, (FloatingPointError, torch.OutOfMemoryError)):
                        raise
                    row = result["branches"][branch]
                    row.update(status="FAILED", reason=repr(exc))
                    terminate_phases(result, branch, "FAILED")
                    result["failures"].append(dict(stage=branch, reason=repr(exc), time_unix=time.time(), attempt_id=attempt))
                    journal.save()
            result["status"] = "ENGINEERING_FAILURE" if result["failures"][failure_start:] else "COMPLETE"
        except BaseException as exc:
            result["status"] = "ENGINEERING_FAILURE" if isinstance(exc, Exception) else "INTERRUPTED"
            result["failures"].append(dict(stage=stage, reason=repr(exc), time_unix=time.time(), attempt_id=attempt))
            if stage in result["branches"]:
                row = result["branches"][stage]
                if row["status"] != "NONFINITE":
                    row.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED_COMPLETION_UNKNOWN", reason=repr(exc))
                terminate_phases(result, stage, "FAILED" if isinstance(exc, Exception) else "INTERRUPTED_COMPLETION_UNKNOWN")
            for row in result["branches"].values():
                if row["status"] == "MISSING":
                    row["reason"] = "not entered after " + stage + ": " + repr(exc)
            journal.save()
            if not isinstance(exc, Exception):
                raise
        finally:
            model = None
            gc.collect()
            result["finished_at_unix"] = time.time()
            result["current_attempt"].update(status=result["status"], failure_end_index=len(result["failures"]),
                                             failures=copy.deepcopy(result["failures"][failure_start:]))
            result["storage_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            save_report(output, result)
            journal.save()
    return result
