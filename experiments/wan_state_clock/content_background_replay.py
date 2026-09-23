"""Fixed 24-view CPU replay of Content-Background-Existence-V1."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from main.tube_state import content_background_existence as candidate
from main.tube_state import fixed_key

ROOT = Path(__file__).parents[2]
SPEC = ROOT / "docs" / "content_background_existence_v1_spec.md"
DEFAULT_INPUT = Path(
    "/home/richar/projects/Video-WM/diagnostics/project-status-20260921/"
    "flow_fixed_key_v1_20260922T014420975188Z/receiver_split_v1"
)
DEFAULT_OUTPUT = ROOT / "evidence" / "content_background_existence_v1_development"
CASES = ("cal_off_p0_s1", "cal_off_p1_s1", "eval_p2_s2", "eval_p3_s3")
ARMS = {
    "cal_off_p0_s1": ("OFF",),
    "cal_off_p1_s1": ("OFF",),
    "eval_p2_s2": ("OFF", "SINGLE46", "MULTI44_46"),
    "eval_p3_s3": ("OFF", "SINGLE46", "MULTI44_46"),
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def roster():
    return tuple((case, arm, view) for case in CASES for arm in ARMS[case] for view in fixed_key.FIXED_VIEWS)


def slot(case, arm, view):
    return "/".join((case, arm, view))


def load_observations(input_root: Path, original: dict, case: str, arm: str, view: str):
    observations, receipt = {}, {}
    rows = original["cases"][case]["videos"][arm]["views"][view]["observations"]
    for phase in range(4):
        phase_row = rows[str(phase)]
        relative = Path(case) / "observations" / arm / view / f"g{phase}.pt"
        path = input_root / "tensors" / relative
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        observations[phase] = tensor.numpy()
        receipt[str(relative)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "source_status": phase_row.get("status"),
        }
    return observations, receipt


def process_one(arguments):
    input_text, original, correct_key_hex, spec_sha, case, arm, view = arguments
    torch.set_num_threads(1)
    input_root = Path(input_text)
    started = time.monotonic()
    try:
        observations, receipt = load_observations(input_root, original, case, arm, view)
        row = candidate.read(observations, bytes.fromhex(correct_key_hex), spec_sha)
        expected = original["cases"][case]["videos"][arm]["views"][view]["detection"]
        correct = row.get("correct_detection")
        baseline = {
            "expected_score": expected.get("existence_statistic"),
            "observed_score": None if correct is None else correct.get("existence_statistic"),
            "expected_path": None if expected.get("best") is None else expected["best"].get("path"),
            "observed_path": None if correct is None or correct.get("best") is None else correct["best"].get("path"),
        }
        if baseline["observed_score"] is not None:
            baseline["absolute_score_difference"] = abs(baseline["observed_score"] - baseline["expected_score"])
        baseline["status"] = "PASS" if (
            baseline.get("absolute_score_difference", float("inf")) <= 1e-12
            and baseline["observed_path"] == baseline["expected_path"]
        ) else "MISMATCH"
        row.update(tensors=receipt, baseline_reproduction=baseline, seconds=time.monotonic() - started)
        return slot(case, arm, view), row
    except Exception as exc:
        return slot(case, arm, view), {
            "status": "INVALID", "error": repr(exc), "traceback": traceback.format_exc(),
            "seconds": time.monotonic() - started,
        }


def run(input_root: Path, output: Path, workers: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((input_root.parent / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_root.parent / "raw_implementation" / "manifest.json").read_text(encoding="utf-8"))
    correct_key = manifest["key_utf8"].encode("utf-8")
    spec_sha = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    rows = roster()
    result = {
        "status": "RUNNING",
        "started_utc": now(),
        "source_commit_base": "8aff4025fd0f8dd656091bf83af1311263f4d99c",
        "spec_sha256": spec_sha,
        "input_root": str(input_root),
        "fixed_denominator": {
            "sources": 4, "physical_source_arms": 8, "views": 24, "phases": 96,
            "keys_per_view": 17, "searches": 408,
            "expected_path_attempts": 1747872,
        },
        "workers": workers,
        "views": {slot(*row): {"status": "NOT_RUN"} for row in rows},
        "calibration": {"status": "NOT_RUN", "threshold": None},
        "failures": [],
    }
    dump(output / "result.json", result)
    jobs = [(str(input_root), original, correct_key.hex(), spec_sha, *row) for row in rows]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, job): job[-3:] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            identifier, row = future.result()
            result["views"][identifier] = row
            dump(output / "views" / Path(identifier + ".json"), row)
            if row.get("status") != "SCORED" or row.get("baseline_reproduction", {}).get("status") != "PASS":
                result["failures"].append({"view": identifier, "status": row.get("status"), "error": row.get("error")})
            dump(output / "result.json", result)
            print(json.dumps({"view": identifier, "status": row.get("status"), "seconds": row.get("seconds")}), flush=True)
    calibration_ids = ("cal_off_p0_s1", "cal_off_p1_s1")
    calibration_sources = {}
    for case in calibration_ids:
        values = {view: result["views"][slot(case, "OFF", view)] for view in fixed_key.FIXED_VIEWS}
        calibration_sources[case] = candidate.source_statistic(values, spec_sha, fixed_key.key_identifier(correct_key))
    calibration_ok = all(row.get("status") == "SCORED" for row in calibration_sources.values())
    result["calibration"] = {
        "status": "FROZEN_DEVELOPMENT_TWO_SOURCE" if calibration_ok else "UNCALIBRATED",
        "threshold": max(row["statistic"] for row in calibration_sources.values()) + 1e-6 if calibration_ok else None,
        "guard": 1e-6,
        "sources": calibration_sources,
        "claim": "same-batch two-source development diagnostic only; not the fixed four-source new-run calibration",
    }
    threshold = result["calibration"]["threshold"]
    for identifier, row in result["views"].items():
        statistic = row.get("statistic")
        row["development_decision"] = (
            {"status": "UNCALIBRATED", "detected": None, "statistic": statistic}
            if threshold is None or statistic is None
            else {
                "status": "DETECTED" if statistic > threshold else "REJECTED",
                "detected": statistic > threshold,
                "statistic": statistic,
                "threshold": threshold,
                "signed_margin": statistic - threshold,
            }
        )
    result["actual_search_count"] = sum(row.get("actual_search_count", 0) for row in result["views"].values())
    result["actual_path_attempts"] = sum(row.get("actual_path_attempts", 0) for row in result["views"].values())
    result["status"] = "EXECUTION_COMPLETE" if not result["failures"] and result["actual_search_count"] == 408 and result["actual_path_attempts"] == 1747872 else "COMPLETE_WITH_RETAINED_FAILURES"
    result["completed_utc"] = now()
    result["environment"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"}
    result["evidence_ceiling"] = "same-batch saved-tensor development diagnostic only; no independent FPR, generalization, quality, or causal background-removal claim"
    dump(output / "result.json", result)
    dump(output / "calibration.json", result["calibration"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    final = run(arguments.input, arguments.output, arguments.workers)
    print(final["status"], flush=True)
    if final["status"] != "EXECUTION_COMPLETE":
        raise SystemExit(1)
