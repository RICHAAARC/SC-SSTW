"""Fixed saved-JOINT terminal bridge: two posterior encodes, four decodes.

Model calls occur only at TrackedVAE boundaries; no generation/media path runs.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as diagnostic
from runtime.wan import generation, vae as vae_adapter
from runtime.wan.local_joint_state_payload_provider_v1 import (
    PUBLIC_LATENT_SUPPORT, WanPosteriorBackend, mask_and_cap,
)

RUN_ID = "20261009T132316545817Z"
KEY = "local-joint-state-payload-v1-first-mechanism"
MODEL = dict(id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", revision="0fad780a534b6463e45facd96134c9f345acfa5b")
LATENT_SHAPE = (1, 16, 46, 40, 64)
CALLS = (("encode_base", "encode"), ("encode_candidate", "encode"),
         *((name, "decode") for name in diagnostic.STAGES[2:]))
CEILING = ("One saved JOINT terminal bridge only; not original step25..49 conditional-clean, "
           "CFG/native causality, trajectory recovery, blind decoding, FPR or scientific PASS.")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def initial_result(config: dict) -> dict:
    return dict(schema="saved-joint-terminal-bridge-v1", status="RUNNING", config=config,
        scientific_pass=False, ceiling=CEILING, source_run=RUN_ID, source_arm="JOINT",
        expected_cost=dict(encode=2, decode=4, dit=0, native=0, codec=0, backward=0),
        counts=dict(encode_attempted=0, encode_completed=0, decode_attempted=0, decode_completed=0,
                    vae_load_attempted=0, vae_load_completed=0, dit=0, native=0, codec=0, backward=0),
        model_calls={name: dict(kind=kind, status="NOT_RUN") for name, kind in CALLS},
        stages={name: dict(status="NOT_RUN", views={view: dict(status="MISSING", reason="not_run",
                    expected_windows=88, expected_chips=1408, observed_windows=0)
                    for view in diagnostic.VIEWS}) for name in diagnostic.STAGES},
        failures=[], input_contract={}, cleanup_errors=[], actual_model_execution="NOT_STARTED")


def _write_comparisons(output, evaluations, raw_rows, *, key, message, protocol=carrier.PUBLIC):
    """One fixed report directory for normal completion and report-only takeover."""
    comparisons = [(f"{a}/postclip", f"{b}/postclip")
                   for a, b in zip(diagnostic.STAGES, diagnostic.STAGES[1:])]
    comparisons += [("base/postclip", f"{name}/postclip") for name in diagnostic.STAGES[2:]]
    comparisons += [(f"{name}/preclip", f"{name}/postclip") for name in diagnostic.STAGES[1:]]
    pairs = {}
    for left, right in comparisons:
        pair = diagnostic.paired(evaluations[left], evaluations[right], raw_rows[left], raw_rows[right],
                                 key=key, message=message, protocol=protocol)
        for metric in pair["state_correlations"] + pair["payload_signed_means"]:
            metric["status"] = "MISSING" if metric["value"] is None else "SCORED"
        pair["state_gap_status"] = "MISSING" if pair["state_gap"] is None else "SCORED"
        for chip in pair["chips"]:
            chip["delta_status"] = {name: "MISSING" if value is None else "SCORED"
                                    for name, value in chip["deltas"].items()}
            chip["signed_q_delta_status"] = "MISSING" if chip["signed_q_delta"] is None else "SCORED"
        pairs[left + " -> " + right] = pair
    write_json(output / "paired_comparisons.json", dict(direction="right_minus_left", comparisons=pairs))


def finalize_interrupted(output: Path, reason: str) -> dict | None:
    """After child reap, finish reports from saved evidence only; never resume execution."""
    output = Path(output)
    path = output / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "RUNNING":
        return result
    result["status"] = "INTERRUPTED"
    result["failures"].append(dict(stage="external_termination", reason=reason, retry=False))
    for call in result["model_calls"].values():
        if call["status"] == "RUNNING":
            call.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason=reason)
        elif call["status"] == "NOT_RUN":
            call.update(status="MISSING_DEPENDENCY", reason=reason)
    evaluations, raw_rows = {}, {}
    for name, stage in result["stages"].items():
        if stage["status"] in ("RUNNING", "NOT_RUN"):
            stage.update(status="INTERRUPTED" if stage["status"] == "RUNNING" else "MISSING_DEPENDENCY", reason=reason)
        for view_name, view in stage["views"].items():
            ident = f"{name}/{view_name}"
            missing_reason = view.get("reason")
            if (name, view_name) == ("base", "preclip"):
                missing_reason = "saved_base_preclip_unavailable"
            elif not missing_reason or missing_reason == "not_run":
                missing_reason = "interrupted before observation persistence completed: " + reason
            raw_path = output / name / f"{view_name}_raw.json"
            metric_path = output / name / f"{view_name}_metrics.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else None
            metric = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else None
            raw_saved = raw is not None and raw.get("status") == "OBSERVED"
            metric_saved = metric is not None and not metric.get("directory_placeholder", False)
            if raw_saved:
                raw_rows[ident] = raw["rows"]
                view.update(raw_status="SAVED", raw_path=str(raw_path.relative_to(output)),
                            observed_windows=len(raw["rows"]))
            else:
                raw_rows[ident] = []
                raw = dict(status="MISSING", reason=missing_reason, expected_windows=88,
                           expected_chips=1408, observed_windows=0, rows=[])
                write_json(raw_path, raw)
                view.update(raw_status="MISSING", observed_windows=0)
            if metric_saved:
                evaluations[ident] = metric
                # A metric file can precede its result.json checkpoint. Retain
                # that evidence, but do not infer model/stage completion from it.
                view.update(metrics_status="SAVED", metrics_path=str(metric_path.relative_to(output)),
                    state_gap=metric["state"]["c0_minus_max_other"],
                    payload_signed_means=[m["signed_mean"] for m in metric["payload"]["metrics"]])
                for field in ("weakest_bits", "descriptive_condition"):
                    if field in metric:
                        view[field] = metric[field]
            else:
                # Raw evidence may exist without a metric file. Preserve its
                # chip deltas; do not fabricate or recompute absent summary evidence.
                evaluations[ident] = posthoc.missing_evaluation(missing_reason)
                write_json(metric_path, evaluations[ident])
                view["metrics_status"] = "MISSING"
            view.update(status="OBSERVED" if raw_saved and metric_saved else "MISSING",
                        reason=None if raw_saved and metric_saved else missing_reason)
    _write_comparisons(output, evaluations, raw_rows, key=KEY, message=bytes.fromhex("8001a55a"))
    result["interruption_report"] = dict(status="COMPLETE", source="persisted_raw_and_metrics_only",
        comparisons=14, chips_per_comparison=1408, summary_slots_per_comparison=55,
        model_execution_resumed=False)
    # Write terminal status last: if report I/O is interrupted, the existing
    # RUNNING checkpoint can still be finalized by the live notebook parent.
    write_json(path, result)
    return result


def validate_config(config: dict) -> None:
    if (config.get("source_run") != RUN_ID or config.get("source_arm") != "JOINT"
            or config.get("model") != MODEL
            or config.get("carrier") != dict(key=KEY, message_hex="8001a55a", rho=.5, cap=1.0)):
        raise ValueError("fixed JOINT run/model/key/message/rho/cap semantics required")
    if not isinstance(config.get("inputs"), dict) or set(config["inputs"]) != {"terminal_latent", "float_rgb"}:
        raise ValueError("both saved JOINT tensor input paths required; no fallback decode")


def tensor_contract(value: Any, shape: tuple, *, bounded_rgb: bool = False) -> dict:
    import torch

    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or value.dtype != torch.float32:
        raise ValueError(f"expected finite FP32 tensor with shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite tensor input/output")
    if bounded_rgb and bool(((value < 0) | (value > 1)).any()):
        raise ValueError("saved RGB must already be clamped to [0,1]")
    return dict(status="VALID", shape=list(value.shape), dtype=str(value.dtype), finite=True,
                minimum=float(value.min()), maximum=float(value.max()), bounded_rgb=bounded_rgb)


class TrackedVAE:
    """Persist attempted before, and returned/failed immediately after, each model call."""

    def __init__(self, model, result: dict, save, *, execution_kind: str):
        self.model, self.result, self.save = model, result, save
        self.execution_kind = execution_kind
        self.current = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _call(self, kind, *args, **kwargs):
        row = self.result["model_calls"][self.current]
        if row["kind"] != kind or row["status"] != "NOT_RUN":
            raise RuntimeError("unexpected or repeated model call")
        self.result["counts"][kind + "_attempted"] += 1
        self.result["actual_model_execution"] = self.execution_kind
        row.update(status="RUNNING", started_utc=time.time())
        self.save()  # durable call intent immediately before entry
        print(f"MODEL {self.current}: {kind} ATTEMPTED", flush=True)
        started = time.perf_counter()
        try:
            value = getattr(self.model, kind)(*args, **kwargs)
        except BaseException as exc:
            row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter()-started)
            try:
                self.save()
            except BaseException as record_error:
                if hasattr(exc, "add_note"):
                    exc.add_note("secondary model failure persistence error: " + repr(record_error))
            raise
        self.result["counts"][kind + "_completed"] += 1
        row.update(status="RETURNED", elapsed_seconds=time.perf_counter()-started)
        self.save()  # adapter, reader or persistence failure cannot erase a returned call
        print(f"MODEL {self.current}: {kind} RETURNED", flush=True)
        return value

    def encode(self, *args, **kwargs):
        return self._call("encode", *args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._call("decode", *args, **kwargs)


def run(config: dict, output: Path, *, loader=None, protocol=carrier.PUBLIC,
        latent_shape=LATENT_SHAPE, support=PUBLIC_LATENT_SUPPORT, source_identity=None) -> dict:
    """CLI uses fixed public geometry; injected loader/geometry exist only for CPU tests."""
    import torch

    output = Path(output)
    # The old input run is read-only. Refuse outputs within either input directory.
    for value in config.get("inputs", {}).values():
        if output.resolve().is_relative_to(Path(value).resolve().parent):
            raise ValueError("diagnostic output must be outside the saved input directory")
    output.mkdir(parents=True, exist_ok=False)
    result = initial_result(config)
    result["source_identity"] = source_identity or dict(status="NOT_RECORDED", blocking=False)
    save = lambda: write_json(output / "result.json", result)
    save()
    model = tracked = backend = None
    stage_name = "input_contract"
    key, message, rho = KEY, bytes.fromhex("8001a55a"), .5
    evaluations, raw_rows = {}, {}
    for name in diagnostic.STAGES:
        for view in diagnostic.VIEWS:
            ident = f"{name}/{view}"
            reason = "saved_base_preclip_unavailable" if ident == "base/preclip" else "not_run"
            evaluations[ident] = posthoc.missing_evaluation(reason)
            raw_rows[ident] = []
            result["stages"][name]["views"][view]["reason"] = reason
            write_json(output / name / f"{view}_metrics.json", evaluations[ident])
            write_json(output / name / f"{view}_raw.json", dict(status="MISSING", reason=reason,
                expected_windows=88, expected_chips=1408, observed_windows=0, rows=[]))
    save()

    def fail(stage, exc):
        reason = f"{type(exc).__name__}: {exc}"
        result["failures"].append(dict(stage=stage, reason=reason, retry=False,
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
        return reason

    def persist_tensor(name, tensor):
        path = output / "tensors" / (name + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(tensor.detach().cpu(), temporary)
        os.replace(temporary, path)
        return str(path.relative_to(output))

    def observe(name, view, rgb, base):
        row = result["stages"][name]["views"][view]
        row.update(status="RUNNING", reason=None)
        save()
        print(f"READER {name}/{view}: 88 fixed windows", flush=True)
        rgb = rgb.detach().cpu()
        tensor_contract(rgb, protocol.video_shape, bounded_rgb=(view == "postclip"))
        ident = f"{name}/{view}"
        raw = diagnostic.read_rgb(rgb, key=key, rho=rho, protocol=protocol)
        raw_path = output / name / f"{view}_raw.json"
        write_json(raw_path, dict(status="OBSERVED", truth_used=False, rows=raw,
            expected_windows=88, expected_chips=1408, observed_windows=len(raw)))
        raw_rows[ident] = raw
        row.update(observed_windows=88, raw_path=str(raw_path.relative_to(output)), raw_status="SAVED")
        save()
        # Truth is applied only after the complete raw directory has been persisted.
        metric = diagnostic.evaluate_saved(raw, key=key, message=message, rho=rho, protocol=protocol)
        metric_path = output / name / f"{view}_metrics.json"
        write_json(metric_path, metric)
        raw_rows[ident], evaluations[ident] = raw, metric
        row.update(status="OBSERVED", metrics_path=str(metric_path.relative_to(output)),
            state_gap=metric["state"]["c0_minus_max_other"],
            payload_signed_means=[m["signed_mean"] for m in metric["payload"]["metrics"]],
            weakest_bits=metric["weakest_bits"], descriptive_condition=metric["descriptive_condition"])
        if metric["descriptive_condition"]["classification"] == "ENGINEERING_FAILURE":
            result["failures"].append(dict(stage=ident, reason="required reader evidence failed", retry=False))
        extra_errors = [r["tile_ratio_statistics_error"] for r in raw if "tile_ratio_statistics_error" in r]
        if extra_errors:
            row["tile_ratio_statistics_errors"] = extra_errors
            result["failures"].append(dict(stage=ident, reason="tile ratio statistic failure", errors=extra_errors, retry=False))
        save()
        # CPU per-frame reductions avoid creating full float64 video copies.
        squared, maximum, clamp_squared = 0.0, 0.0, 0.0
        for frame, reference in zip(rgb, base):
            difference = frame.double() - reference.double()
            squared += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            if view == "preclip":
                clamp_squared += float((frame.double()-frame.clamp(0, 1).double()).square().sum())
        row.update(status="OBSERVED", observed_windows=88, raw_path=str(raw_path.relative_to(output)),
            metrics_path=str(metric_path.relative_to(output)),
            state_gap=metric["state"]["c0_minus_max_other"],
            payload_signed_means=[m["signed_mean"] for m in metric["payload"]["metrics"]],
            weakest_bits=metric["weakest_bits"], descriptive_condition=metric["descriptive_condition"],
            rgb_vs_base=dict(rmse=(squared/rgb.numel())**.5, max_abs=maximum),
            clamp_adjustment_l2=clamp_squared**.5 if view == "preclip" else None,
            out_of_range_values=dict(low=int((rgb < 0).sum()), high=int((rgb > 1).sum())))
        save()

    try:
        validate_config(config)
        values = {}
        for name, shape in (("terminal_latent", latent_shape), ("float_rgb", protocol.video_shape)):
            path = Path(config["inputs"][name])
            result["input_contract"][name] = dict(status="READING", path=str(path))
            save()
            value = torch.load(path, map_location="cpu", weights_only=True)
            result["input_contract"][name].update(tensor_contract(value, shape, bounded_rgb=(name == "float_rgb")))
            values[name] = value
            save()
        z, base = values["terminal_latent"], values["float_rgb"]
        del values
        stage_name = "base"
        result["stages"][stage_name]["status"] = "RUNNING"
        observe("base", "postclip", base, base)
        result["stages"][stage_name]["status"] = "COMPLETE"
        stage_name = "candidate"
        result["stages"][stage_name]["status"] = "RUNNING"
        save()
        preclip = base.clone()
        def capture(frame, roi, pixels):
            y0, y1, x0, x1 = protocol.rois[roi]
            preclip[frame, y0:y1, x0:x1] = pixels
        candidate, receipt = carrier.apply_carrier_rgb(base, key=key, message=message, rho=rho,
            protocol=protocol, diagnostic_observer=capture)
        write_json(output / "candidate" / "carrier_receipt.json", receipt)
        observe("candidate", "preclip", preclip, base)
        observe("candidate", "postclip", candidate, base)
        del preclip
        result["stages"][stage_name]["status"] = "COMPLETE"
        save()
        stage_name = "vae_load"
        result["counts"]["vae_load_attempted"] += 1
        save()
        model = (loader or generation.load_frozen_vae)(config, device=config.get("device", "cuda"))
        result["counts"]["vae_load_completed"] += 1
        save()
        tracked = TrackedVAE(model, result, save, execution_kind="REAL" if loader is None else "INJECTED_TEST_DOUBLE")
        backend = WanPosteriorBackend(tracked)
        encoded = {}
        with torch.inference_mode():
            for name, pixels in (("encode_base", base), ("encode_candidate", candidate)):
                stage_name = tracked.current = name
                tensor = backend.encode_normalized(pixels).detach().cpu()
                tensor_contract(tensor, latent_shape)
                encoded[name] = tensor
                result["model_calls"][name]["normalized_tensor"] = persist_tensor(name, tensor)
                save()
            del candidate
            raw = encoded["encode_candidate"] - encoded["encode_base"]
            residual = z - encoded["encode_base"]
            capped, cap_receipt = mask_and_cap(raw, cap=1.0, support=support)
            masked = torch.zeros_like(raw)
            t0, t1 = support.time_range
            for y0, y1, x0, x1 in support.rois:
                masked[:, :, t0:t1, y0:y1, x0:x1] = raw[:, :, t0:t1, y0:y1, x0:x1]
            cap_receipt.update(coordinate="normalized_saved_terminal_latent",
                reconstruction_residual_l2=float(residual.double().norm()),
                norm_is_auxiliary_not_signal_retention=True)
            result["latent_diagnostics"] = dict(receipt=cap_receipt,
                tensors={name: persist_tensor(name, value) for name, value in
                         (("raw", raw), ("masked", masked), ("capped", capped), ("reconstruction_residual", residual))})
            save()
            inputs = (encoded["encode_candidate"], z+raw, z+masked, z+capped)
            for name, latent in zip(diagnostic.STAGES[2:], inputs):
                stage_name = tracked.current = name
                result["stages"][name]["status"] = "RUNNING"
                save()
                decoded = None
                try:
                    tensor_contract(latent, latent_shape)
                    decoded = vae_adapter.decode_normalized_latent(tracked,
                        latent.to(device=next(model.parameters()).device),
                        diagnostic_observer=lambda pixels, name=name: observe(name, "preclip", pixels, base))
                    observe(name, "postclip", decoded, base)
                    result["stages"][name]["status"] = "COMPLETE"
                except Exception as exc:
                    result["stages"][name].update(status="FAILED", reason=fail(name, exc))
                finally:
                    decoded = None
                save()
    except BaseException as exc:
        reason = fail(stage_name, exc)
        if stage_name in result["stages"]:
            result["stages"][stage_name].update(status="FAILED", reason=reason)
        for row in result["input_contract"].values():
            if row["status"] == "READING":
                row.update(status="FAILED", reason=reason)
    finally:
        if model is not None:
            try:
                vae_adapter._clear_cache(model)
            except BaseException as exc:
                result["cleanup_errors"].append(repr(exc))
        # Drop the proxy and backend as well as the model before releasing CUDA cache.
        model = tracked = backend = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except BaseException as exc:
                result["cleanup_errors"].append(repr(exc))
        missing_reason = result["failures"][-1]["reason"] if result["failures"] else "not_completed"
        for row in result["model_calls"].values():
            if row["status"] == "NOT_RUN":
                row.update(status="MISSING_DEPENDENCY", reason=missing_reason)
        for name, stage in result["stages"].items():
            if stage["status"] in ("NOT_RUN", "RUNNING"):
                stage.update(status="MISSING_DEPENDENCY", reason=missing_reason)
            for view, row in stage["views"].items():
                if row["status"] != "OBSERVED":
                    reason = "saved_base_preclip_unavailable" if (name, view) == ("base", "preclip") else stage.get("reason", missing_reason)
                    row.update(status="MISSING", reason=reason)
                    evaluations[f"{name}/{view}"] = posthoc.missing_evaluation(reason)
                    write_json(output / name / f"{view}_metrics.json", evaluations[f"{name}/{view}"])
                    # Keep an already saved raw read if later reduction/persistence failed.
                    raw_path = output / name / f"{view}_raw.json"
                    existing = json.loads(raw_path.read_text(encoding="utf-8"))
                    if existing.get("status") != "OBSERVED":
                        existing.update(reason=reason)
                        write_json(raw_path, existing)
        _write_comparisons(output, evaluations, raw_rows, key=key, message=message, protocol=protocol)
        result["status"] = "COMPLETE" if (not result["failures"] and not result["cleanup_errors"] and
            all(s["status"] == "COMPLETE" for s in result["stages"].values())) else "ENGINEERING_FAILURE"
        result["completed_utc"] = time.time()
        save()
    return result
