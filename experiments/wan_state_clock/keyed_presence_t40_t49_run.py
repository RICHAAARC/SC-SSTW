"""Fixed fresh-source keyed H0/H1 presence run; GPU execution belongs to the user."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from main.tube_state import fixed_key, fixed_key_split_receiver as split_receiver
from runtime.wan import keyed_presence_t40_t49 as writer
from runtime.wan import fixed_key_control, generation, io, media_channel_retention as retention, payload_control, trajectory, vae as vae_adapter
from runtime.wan.generation import load_frozen_vae
from runtime.wan.integrated_core import encode_four_phases
from runtime.wan.io import dump, encode_rgb, read_mp4
from runtime.wan.vae import decode_normalized_latent, quantize_rgb8_no_codec, reencode_rgb24_readback, _clear_cache

MANIFEST = Path(__file__).parent / "configs" / "keyed_presence_t40_t49_v1.json"
MODULE = "experiments.wan_state_clock.keyed_presence_t40_t49_run"
RECEIVERS = ("C2_STATE_CONFIRM", "ORIGINAL", "C1_MATCHED_CONFIRM")
LAYERS = ("TERMINAL", "FLOAT_RGB_REENCODE", "RGB8_NO_CODEC_REENCODE", "MP4_G0")
PARTITIONS = retention.PARTITIONS
PHASES = (0, 1, 2, 3)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def arms_for(case, config):
    return tuple(config["calibration_arms"] if case["role"] == "calibration_off" else config["evaluation_arms"])


def empty_layer():
    return {"status": "NOT_RUN", "partitions": {name: retention.empty_measurement() for name in PARTITIONS}, "error": None}


def empty_video(case, arm):
    return {"status": "NOT_RUN", "role": case["role"], "arm": arm,
            "layers": {layer: empty_layer() for layer in LAYERS},
            "phases": {str(g): {"status": "NOT_RUN"} for g in PHASES},
            "receivers": {name: {"status": "NOT_RUN", "statistic": None, "decision": None} for name in RECEIVERS},
            "failure": None}


def empty_case(case, config):
    return {"status": "NOT_RUN", "case_id": case["id"], "role": case["role"],
            "videos": {arm: empty_video(case, arm) for arm in arms_for(case, config)},
            "actual_calls": {}, "failures": [], "stage_logs": {}, "parent_failures": []}


def validate_manifest(config):
    if config != load(MANIFEST):
        raise ValueError("manifest differs from adopted fixed protocol")
    if config["adopted_stage1_draft_sha256"] != "bbe67da2f52bb4b466a7d602e2db268c3804e9c6cf7e3dfff45cb0f1f9d609b2":
        raise ValueError("wrong adopted stage-1 protocol")
    if len(config["cases"]) != 13 or len({c["id"] for c in config["cases"]}) != 13:
        raise ValueError("case roster mismatch")
    if tuple(config["evaluation_arms"]) != tuple(writer.ARMS) or tuple(config["calibration_arms"]) != ("OFF",):
        raise ValueError("arm roster mismatch")
    if config["control"]["R_star"] != writer.R_STAR or config["control"]["steps"] != {k: list(v) for k, v in writer.ARMS.items() if k != "OFF"}:
        raise ValueError("control schedule mismatch")
    if config["receiver"] != load(MANIFEST)["receiver"]:
        raise ValueError("receiver definition mismatch")
    if tuple(config["receiver"]["phases"]) != PHASES or config["receiver"]["paths"] != 4284:
        raise ValueError("receiver path family mismatch")


def case_config(config, case):
    out = {"model": copy.deepcopy(config["model"]), "generation": copy.deepcopy(config["generation"])}
    out["generation"].update(prompt=case["prompt"], seed=case["seed"])
    return out


def counter(result, progress_path):
    def count(kind, completed):
        key = kind + ("_completed" if completed else "_attempted")
        result["actual_calls"][key] = result["actual_calls"].get(key, 0) + 1
        dump(progress_path, {"status": result["status"], "progress": result.get("progress"),
                             "actual_calls": result["actual_calls"]})
        if kind == "transformer" and completed and result["actual_calls"][key] % 20 == 0:
            print(f"progress {result['case_id']} transformer={result['actual_calls'][key]}", flush=True)
    return count


def _artifact(value, path, item, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu(), path)
    item.setdefault("artifacts", {})[name] = {"path": str(path), "sha256": sha(path)}


def generate_case(case_id, config_path, output):
    config = load(config_path); validate_manifest(config)
    case = next(c for c in config["cases"] if c["id"] == case_id)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result = empty_case(case, config)
    result.update(status="GENERATION_RUNNING", config=case_config(config, case),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  manifest_sha256=sha(config_path), prompt=case["prompt"], seed=case["seed"],
                  source_files_sha256={str(p): sha(p) for p in (Path(__file__), MANIFEST, Path(writer.__file__),
                      Path(split_receiver.__file__), Path(fixed_key.__file__), Path(fixed_key_control.__file__),
                      Path(retention.__file__), Path(payload_control.__file__), Path(trajectory.__file__),
                      Path(generation.__file__), Path(io.__file__), Path(vae_adapter.__file__))})
    count = counter(result, output / "progress.json")
    def save(): dump(output / "generation.json", result)
    save()
    try:
        generated = writer.generate_terminals(result["config"], config["key_utf8"].encode(), arms_for(case, config), count)
        result["fresh_generation"] = {k: v for k, v in generated.items() if k != "arms"}
        for arm, branch in generated["arms"].items():
            item = result["videos"][arm]
            if branch["status"] != "GENERATED":
                item.update(status="FAILED_GENERATION", failure=branch.get("error"))
                result["failures"].append({"stage": "generate/" + arm, "error": branch.get("error")})
                save(); continue
            terminal = branch.pop("terminal")
            _artifact(terminal, output / "terminals" / f"{arm}.pt", item, "terminal")
            for index, arrays in branch.pop("control_tensors").items():
                tensor_path = output / "control_tensors" / arm / f"step{index}.pt"
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(arrays, tensor_path)
                item.setdefault("control_tensor_files", {})[index] = {"path": str(tensor_path), "sha256": sha(tensor_path)}
            item.update(branch)
            save()
    except Exception as exc:
        result["failures"].append({"stage": "generation_setup", "error": repr(exc), "traceback": traceback.format_exc()})
    result["status"] = "GENERATION_COMPLETE" if not result["failures"] and all(v["status"] == "GENERATED" for v in result["videos"].values()) else "WITH_RETAINED_FAILURES"
    result["resources_generation"] = {"cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save()
    return result


def score_layer(layer, tensor, book):
    try:
        scored = retention.score_identity_layer(tensor, book, fixed_key.REFERENCE_PATHS["IDENTITY"], list(fixed_key.carrier.SHAPE))
        layer.update(status="SCORED", partitions=scored["partitions"], error=None)
    except Exception as exc:
        layer.update(status="FAILED", error=repr(exc))
        for name in PARTITIONS:
            layer["partitions"][name].update(status="FAILED", failure=repr(exc))


def score_blind(observations, key, spec_sha):
    """Blind scorer sees only four MP4 observations, key and frozen spec identity."""
    if set(observations) != set(PHASES):
        return {name: {"status": "INVALID", "statistic": None} for name in RECEIVERS}, {}, {}
    book = fixed_key.codebook(key)
    original = fixed_key.read(observations, book)
    split = split_receiver.read(observations, key, spec_sha)
    rows = {
        "ORIGINAL": {"status": original["status"], "statistic": original.get("existence_statistic"),
                     "receiver_protocol_id": fixed_key.PROTOCOL_ID},
        "C1_MATCHED_CONFIRM": {"status": split["status"], "statistic": split.get("confirmation_scores", {}).get("C1_MATCHED_CONFIRM"),
                               "receiver_protocol_id": split_receiver.PROTOCOL_ID},
        "C2_STATE_CONFIRM": {"status": split["status"], "statistic": split.get("confirmation_scores", {}).get("C2_STATE_CONFIRM"),
                             "receiver_protocol_id": split_receiver.PROTOCOL_ID},
    }
    for row in rows.values():
        row.update(spec_sha256=spec_sha, key_id=book["key_id"], view="FULL")
        if row["status"] != "SCORED" or row["statistic"] is None or not math.isfinite(row["statistic"]):
            row.update(status="INVALID", statistic=None)
    return rows, original, split


def media_case(case_id, config_path, output):
    config = load(config_path); validate_manifest(config)
    case = next(c for c in config["cases"] if c["id"] == case_id)
    output = Path(output)
    result = load(output / "generation.json") if (output / "generation.json").exists() else empty_case(case, config)
    result["status"] = "MEDIA_RUNNING"
    count = counter(result, output / "progress.json")
    key, spec_sha = config["key_utf8"].encode(), sha(config_path)
    def save(): dump(output / "result.json", result)
    def fail(stage, exc):
        result["failures"].append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()})
        save()
    save()
    vae = None
    try:
        vae = load_frozen_vae(result.get("config") or case_config(config, case))
        book = fixed_key.codebook(key)
        for arm in arms_for(case, config):
            item = result["videos"][arm]
            result["progress"] = {"stage": "media", "case": case_id, "arm": arm}
            save()
            if item["status"] != "GENERATED":
                item["failure"] = item.get("failure") or "terminal unavailable"
                continue
            try:
                terminal = torch.load(output / "terminals" / f"{arm}.pt", map_location="cpu", weights_only=True)
                if sha(output / "terminals" / f"{arm}.pt") != item["artifacts"]["terminal"]["sha256"]:
                    raise ValueError("terminal artifact hash mismatch")
                if case["role"] == "evaluation":
                    score_layer(item["layers"]["TERMINAL"], terminal, book)
                count("vae_decode", False)
                rgb = decode_normalized_latent(vae, terminal.to(next(vae.parameters()).device)).detach().cpu()
                count("vae_decode", True)
                _artifact(rgb, output / "decoded_float_rgb" / f"{arm}.pt", item, "decoded_float_rgb")
                rgb8 = quantize_rgb8_no_codec(rgb)
                _artifact(rgb8, output / "rgb8_no_codec" / f"{arm}.pt", item, "rgb8_no_codec")
                if case["role"] == "evaluation":
                    for layer_name, pixels in (("FLOAT_RGB_REENCODE", rgb), ("RGB8_NO_CODEC_REENCODE", rgb8.float() / 255.0)):
                        try:
                            count("vae_encode", False)
                            encoded = reencode_rgb24_readback(vae, pixels).detach().cpu()
                            count("vae_encode", True)
                            _artifact(encoded, output / "layer_observations" / arm / f"{layer_name}.pt", item, layer_name)
                            score_layer(item["layers"][layer_name], encoded, book)
                        except Exception as exc:
                            item["layers"][layer_name].update(status="FAILED", error=repr(exc))
                            fail(f"layer/{arm}/{layer_name}", exc)
                path = output / "received_videos" / arm / "FULL.mp4"
                count("mp4_save", False)
                encode_rgb(rgb, path, config["generation"]["fps"], 18)
                count("mp4_save", True)
                item["mp4"] = {"path": str(path), "sha256": sha(path)}
                count("mp4_read", False)
                pixels = read_mp4(path)
                count("mp4_read", True)
                _artifact(pixels, output / "received_rgb" / f"{arm}.pt", item, "received_rgb")
                observations = {}
                def on_phase(phase, encoded, phase_row):
                    observations[phase] = encoded.detach().cpu().numpy()
                    obs_path = output / "observations" / arm / f"g{phase}.pt"
                    _artifact(encoded, obs_path, item, f"MP4_g{phase}")
                    phase_row["path"] = str(obs_path)
                    phase_row["sha256"] = sha(obs_path)
                    save()
                observations, phase_rows = encode_four_phases(pixels, vae, count, on_phase)
                item["phases"] = phase_rows
                if case["role"] == "evaluation" and 0 in observations:
                    score_layer(item["layers"]["MP4_G0"], torch.from_numpy(observations[0]), book)
                rows, original, split = score_blind(observations, key, spec_sha)
                item["receivers"] = rows
                dump(output / "detections" / arm / "original.json", original)
                dump(output / "detections" / arm / "split.json", split)
                item["status"] = "MEDIA_COMPLETE" if all(r["status"] == "SCORED" for r in rows.values()) and all(phase_rows.get(str(g), {}).get("status") == "COMPLETE" for g in PHASES) and (case["role"] != "evaluation" or all(layer["status"] == "SCORED" for layer in item["layers"].values())) else "PARTIAL_OR_FAILED"
                if item["status"] != "MEDIA_COMPLETE":
                    result["failures"].append({"stage": f"media/{arm}", "error": "one or more fixed layer/phase/receiver slots failed"})
            except Exception as exc:
                item.update(status="PARTIAL_OR_FAILED", failure=repr(exc))
                fail(f"media/{arm}", exc)
            finally:
                _clear_cache(vae)
                save()
    except Exception as exc:
        fail("media_setup", exc)
    finally:
        vae = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result["status"] = "EXECUTION_COMPLETE" if not result["failures"] and all(v["status"] == "MEDIA_COMPLETE" for v in result["videos"].values()) else "WITH_RETAINED_FAILURES"
    result["resources_media"] = {"cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save()
    return result


def freeze_calibration(cases, config, spec_sha):
    key_id = fixed_key.key_identifier(config["key_utf8"].encode())
    cal = {}
    for receiver in RECEIVERS:
        rows = {case["id"]: cases[case["id"]]["videos"]["OFF"]["receivers"][receiver] for case in config["cases"] if case["role"] == "calibration_off"}
        protocol_id = fixed_key.PROTOCOL_ID if receiver == "ORIGINAL" else split_receiver.PROTOCOL_ID
        valid = len(rows) == 9 and all(r["status"] == "SCORED" and r["statistic"] is not None and math.isfinite(r["statistic"])
            and r.get("spec_sha256") == spec_sha and r.get("key_id") == key_id and r.get("view") == "FULL"
            and r.get("receiver_protocol_id") == protocol_id for r in rows.values())
        cal[receiver] = {"status": "FROZEN" if valid else "UNCALIBRATED", "threshold": max(r["statistic"] for r in rows.values()) + config["receiver"]["calibration_guard"] if valid else None,
                         "sources": rows, "source_count": 9, "view": "FULL", "candidate_id": receiver,
                         "spec_sha256": spec_sha, "key_id": key_id, "receiver_protocol_id": protocol_id,
                         "guard": config["receiver"]["calibration_guard"]}
    return cal


def attach_decisions(item, calibration, *, case_role, arm):
    for receiver in RECEIVERS:
        row = item["receivers"][receiver]
        score, cal = row.get("statistic"), calibration[receiver]
        if row["status"] != "SCORED" or score is None or not math.isfinite(score):
            decision = {"status": "INVALID", "detected": None}
        elif cal["status"] != "FROZEN" or any(row.get(field) != cal.get(field) for field in
            ("spec_sha256", "key_id", "view", "receiver_protocol_id")):
            decision = {"status": "UNCALIBRATED", "detected": None}
        else:
            yes = score > cal["threshold"]
            decision = {"status": "DETECTED" if yes else "REJECTED", "detected": yes,
                        "statistic": score, "threshold": cal["threshold"], "margin": score - cal["threshold"]}
        row["decision"] = decision
    item["reporting_only"] = {"role": case_role, "arm": arm,
        "calibration_sample": case_role == "calibration_off",
        "truth": None if case_role == "calibration_off" else ("H0" if arm == "OFF" else "H1")}


def _child(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert child.stdout is not None
        for line in child.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        return child.wait()


def paired_diagnostics(cases, config):
    result = {}
    for case in config["cases"]:
        if case["role"] != "evaluation": continue
        videos = cases[case["id"]]["videos"]
        result[case["id"]] = {}
        for arm in config["evaluation_arms"][1:]:
            rows = {}
            for layer in LAYERS:
                rows[layer] = {}
                for part in PARTITIONS:
                    a = videos[arm]["layers"][layer]["partitions"][part]
                    b = videos["OFF"]["layers"][layer]["partitions"][part]
                    rows[layer][part] = retention.compare_measurements(a, b, 1e-12)
            changes = {}
            for before, after in zip(LAYERS, LAYERS[1:]):
                changes[f"{before}->{after}"] = {part: retention.compare_increments(rows[after][part], rows[before][part], 1e-12) for part in PARTITIONS}
            result[case["id"]][arm] = {"marked_minus_off": rows, "adjacent_paired_change": changes}
    return result


def raw_layer_changes(cases, config):
    """All 20 evaluation videos x three adjacent boundaries x three partitions."""
    result = {}
    for case in config["cases"]:
        if case["role"] != "evaluation": continue
        videos = cases[case["id"]]["videos"]
        result[case["id"]] = {}
        for arm in config["evaluation_arms"]:
            layers = videos[arm]["layers"]
            result[case["id"]][arm] = {
                f"{before}->{after}": {
                    part: retention.compare_measurements(
                        layers[after]["partitions"][part], layers[before]["partitions"][part], 1e-12,
                    ) for part in PARTITIONS
                } for before, after in zip(LAYERS, LAYERS[1:])
            }
    return result


def summarize_target(cases, config, calibration):
    tp = fp = invalid_marked = invalid_off = 0
    by_source = {}
    for case in config["cases"]:
        if case["role"] != "evaluation": continue
        videos = cases[case["id"]]["videos"]
        marked = videos["T40_49"]["receivers"]["C2_STATE_CONFIRM"]["decision"]
        off = videos["OFF"]["receivers"]["C2_STATE_CONFIRM"]["decision"]
        tp += marked["status"] == "DETECTED"
        fp += off["status"] == "DETECTED"
        invalid_marked += marked["status"] not in ("DETECTED", "REJECTED")
        invalid_off += off["status"] not in ("DETECTED", "REJECTED")
        by_source[case["id"]] = {"marked": marked, "off": off}
    frozen = calibration["C2_STATE_CONFIRM"]["status"] == "FROZEN"
    return {"status": "EVALUABLE" if frozen else "UNCALIBRATED", "TP": tp, "FP": fp,
            "INVALID_marked": invalid_marked, "INVALID_OFF": invalid_off,
            "TPR": tp / 4 if invalid_marked == 0 and frozen else None,
            "conservative_completion_detection_fraction": tp / 4,
            "TPR_identification_range": [tp / 4, (tp + invalid_marked) / 4],
            "FPR": fp / 4 if invalid_off == 0 and frozen else None,
            "FPR_identification_range": [fp / 4, (fp + invalid_off) / 4],
            "identification_range_is_confidence_interval": False,
            "target_pass": frozen and tp == 4 and fp == 0 and invalid_marked == 0 and invalid_off == 0,
            "by_source": by_source}


def run_all(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load(config_path); validate_manifest(config)
    if (output / "result.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("output already contains a run")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "manifest.json")
    result = {"status": "RUNNING", "fixed_denominator": config["fixed_denominator"],
              "fixed_calls": config["fixed_calls"],
              "cases": {case["id"]: empty_case(case, config) for case in config["cases"]},
              "calibration": {name: {"status": "NOT_RUN", "threshold": None} for name in RECEIVERS},
              "failures": [], "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "manifest_sha256": sha(config_path)}
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    dump(output / "environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": versions})
    dump(output / "result.json", result)

    def execute(case):
        case_id = case["id"]
        for stage in ("generate", "media"):
            log = output / f"{case_id}.{stage}.log"
            command = [sys.executable, "-u", "-m", MODULE, "--config", str(config_path),
                       "--output", str(output / case_id), "--case-id", case_id, "--stage", stage]
            result["cases"][case_id]["stage_logs"][stage] = str(log.relative_to(output))
            try: returncode = _child(command, log)
            except Exception as exc:
                returncode = None
                result["failures"].append({"case_id": case_id, "stage": stage, "error": repr(exc), "log": str(log)})
            path = output / case_id / ("generation.json" if stage == "generate" else "result.json")
            if path.exists():
                current = load(path)
                previous = result["cases"][case_id]
                current["stage_logs"] = previous["stage_logs"]
                current["parent_failures"] = previous["parent_failures"]
                for key in ("generate_exit_code", "media_exit_code"):
                    if key in previous: current[key] = previous[key]
                result["cases"][case_id] = current
            else:
                result["cases"][case_id]["status"] = "FAILED_CHILD_RESULT_MISSING"
                failure = {"case_id": case_id, "stage": stage, "error": "child result missing", "log": str(log)}
                result["cases"][case_id]["parent_failures"].append(failure)
                result["failures"].append(failure)
            progress_path = output / case_id / "progress.json"
            if progress_path.exists():
                try:
                    progress = load(progress_path)
                    saved = result["cases"][case_id].setdefault("actual_calls", {})
                    for key, value in progress.get("actual_calls", {}).items():
                        saved[key] = max(saved.get(key, 0), value)
                    result["cases"][case_id]["last_child_progress"] = progress.get("progress")
                    result["cases"][case_id]["progress_recovered_from"] = str(progress_path.relative_to(output))
                except Exception as exc:
                    failure = {"case_id": case_id, "stage": stage, "error": "progress read failed: " + repr(exc),
                               "log": str(log.relative_to(output))}
                    result["cases"][case_id]["parent_failures"].append(failure)
                    result["failures"].append(failure)
            if returncode != 0:
                failure = {"case_id": case_id, "stage": stage, "returncode": returncode,
                           "error": "child exited without successful completion", "log": str(log.relative_to(output))}
                result["cases"][case_id]["parent_failures"].append(failure)
                result["failures"].append(failure)
            result["cases"][case_id][stage + "_exit_code"] = returncode
            dump(output / "result.json", result)

    for case in config["cases"]:
        if case["role"] == "calibration_off": execute(case)
    calibration = freeze_calibration(result["cases"], config, sha(config_path))
    result["calibration"] = calibration
    dump(output / "calibration.json", calibration)
    for case in config["cases"]:
        if case["role"] != "calibration_off": continue
        item = result["cases"][case["id"]]["videos"]["OFF"]
        attach_decisions(item, calibration, case_role="calibration_off", arm="OFF")
    dump(output / "result.json", result)
    for case in config["cases"]:
        if case["role"] == "evaluation": execute(case)
    for case in config["cases"]:
        if case["role"] != "evaluation": continue
        for arm, item in result["cases"][case["id"]]["videos"].items():
            attach_decisions(item, calibration, case_role="evaluation", arm=arm)
    result["raw_layer_changes"] = raw_layer_changes(result["cases"], config)
    result["paired_diagnostics"] = paired_diagnostics(result["cases"], config)
    result["target_summary"] = summarize_target(result["cases"], config, calibration)
    result["actual_calls_observed"] = {kind + "_" + status: sum(c.get("actual_calls", {}).get(kind + "_" + status, 0) for c in result["cases"].values()) for kind in config["fixed_calls"] for status in ("attempted", "completed")}
    result["status"] = "EXECUTION_COMPLETE" if not result["failures"] and all(c.get("status") == "EXECUTION_COMPLETE" and c.get("generate_exit_code") == 0 and c.get("media_exit_code") == 0 for c in result["cases"].values()) and calibration["C2_STATE_CONFIRM"]["status"] == "FROZEN" else "WITH_RETAINED_FAILURES"
    dump(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(MANIFEST)); parser.add_argument("--output", required=True)
    parser.add_argument("--case-id"); parser.add_argument("--stage", choices=("generate", "media"))
    args = parser.parse_args()
    if args.case_id and args.stage:
        final = generate_case(args.case_id, args.config, args.output) if args.stage == "generate" else media_case(args.case_id, args.config, args.output)
    elif not args.case_id and not args.stage:
        final = run_all(args.config, args.output)
    else:
        parser.error("--case-id and --stage must be supplied together")
    if final["status"] not in ("GENERATION_COMPLETE", "EXECUTION_COMPLETE"):
        raise SystemExit(1)
