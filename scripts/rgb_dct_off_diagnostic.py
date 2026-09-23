"""Read-only, fixed-eight old-OFF development diagnostic for the RGB/DCT receiver.

This is not a calibration, independent evaluation, or H0/H1 experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from main.tube_state.rgb_dct_presence import SPEC_ID, SPEC_SHA256, score_rgb
from runtime.wan.rgb_dct_presence_adapter import score_mp4


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "diagnostics/receiver-first-20260923/existing_off_media_manifest.json"
MANIFEST_SHA256 = "899829676206ba322d6e84a05eca1d593a149df3ddaff56cf5bf2d64b1ccd609"
RUN_ROOT = Path(
    "/content/drive/MyDrive/Video-WM/Content-Background-Existence-V1/"
    "content_background_existence_v1_20260923T024543022721Z"
)
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-Receiver-Diagnostic-V1")
KEY = b"WanProjection-first-validation-key-v1"
LEGACY_KEY_ID = "2deb9bf5842d9b2c"  # Old config/fixed-key provenance only.
NEW_KEY_ID = "785b91ae6b23bfc9"
CASE_IDS = (
    "new_cal_c0_s1", "new_cal_c1_s2", "new_cal_c2_s3", "new_cal_c3_s4",
    "new_eval_e0_s5", "new_eval_e1_s6", "new_eval_e2_s7", "new_eval_e3_s8",
)
FIXED_DENOMINATOR = 8


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("fixed manifest byte identity mismatch")
    manifest = json.loads(raw)
    if (manifest.get("run") != RUN_ROOT.name
            or manifest.get("drive_relative_run_root") != str(RUN_ROOT).removeprefix("/content/drive/MyDrive/")
            or manifest.get("physical_off_mp4_count") != FIXED_DENOMINATOR
            or [row.get("id") for row in manifest.get("rows", [])] != list(CASE_IDS)
            or any(row.get("path") != "received_videos/OFF/FULL.mp4" for row in manifest["rows"])):
        raise ValueError("fixed manifest roster or path mismatch")
    if score_rgb(None, KEY)["key_id"] != NEW_KEY_ID:
        raise ValueError("fixed new-receiver key identity mismatch")
    return manifest


def _write_result(path: Path, result: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _counts(result: dict) -> None:
    statuses = [row["status"] for row in result["rows"]]
    result["scored_count"] = statuses.count("SCORED")
    result["pending_count"] = statuses.count("PENDING")
    result["attempted_count"] = FIXED_DENOMINATOR - result["pending_count"]
    result["invalid_count"] = FIXED_DENOMINATOR - result["scored_count"] - result["pending_count"]


def run_diagnostic(
    run_root: Path,
    output: Path,
    manifest: dict,
    source_sha: str,
    *,
    score_fn: Callable[[Path, bytes], dict] = score_mp4,
    hash_fn: Callable[[Path], str] = _sha256_path,
) -> dict:
    """Internal test seam; the CLI fixes root, roster, key, and score function."""
    rows = manifest["rows"]
    if len(rows) != FIXED_DENOMINATOR or [r["id"] for r in rows] != list(CASE_IDS):
        raise ValueError("fixed denominator or case roster mismatch")
    if any(r["path"] != "received_videos/OFF/FULL.mp4" for r in rows):
        raise ValueError("OFF FULL path mismatch")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        raise FileExistsError(f"existing diagnostic result: {result_path}")
    result = dict(
        status="RUNNING", diagnostic_kind="OLD_OFF_DEVELOPMENT_ONLY",
        fixed_denominator=FIXED_DENOMINATOR, attempted_count=0, scored_count=0, invalid_count=0,
        pending_count=FIXED_DENOMINATOR, source_sha=source_sha,
        spec_id=SPEC_ID, spec_sha256=SPEC_SHA256,
        key_id=NEW_KEY_ID, historical_key_id=LEGACY_KEY_ID,
        key_utf8=KEY.decode("utf-8"), manifest_path=str(MANIFEST),
        manifest_sha256=MANIFEST_SHA256, input_root=str(run_root),
        output_dir=str(output), result_path=str(result_path),
        interpretation="All eight old OFF files are seen development OFF. Historical roles are labels only. Old marked videos are not H1 for this receiver. No new calibration, independent evaluation, threshold, PASS, FPR, or TPR.",
        rows=[dict(
            case_id=row["id"], historical_role=row["role"],
            diagnostic_role="SEEN_DEVELOPMENT_OFF",
            input_path=str(run_root / row["id"] / row["path"]),
            output_path=str(result_path), expected_sha256=row["sha256"],
            actual_sha256=None, status="PENDING", reason=None,
            score=None, frames_used=0,
        ) for row in rows],
    )
    # This is the first media-work side effect: all eight slots already exist.
    _write_result(result_path, result)
    for row in result["rows"]:
        path = Path(row["input_path"])
        try:
            if not path.is_file():
                row.update(status="MISSING", reason="INPUT_FILE_MISSING")
            else:
                row["actual_sha256"] = hash_fn(path)
                if row["actual_sha256"] != row["expected_sha256"]:
                    row.update(status="HASH_MISMATCH", reason="EXPECTED_SHA256_MISMATCH")
                else:
                    scored = score_fn(path, KEY)
                    row["frames_used"] = scored.get("frames_used", 0)
                    value = scored.get("score")
                    valid = (scored.get("status") == "SCORED"
                             and isinstance(value, (int, float)) and math.isfinite(value)
                             and row["frames_used"] == 181
                             and scored.get("spec_sha256") == SPEC_SHA256
                             and scored.get("key_id") == NEW_KEY_ID)
                    if valid:
                        row.update(status="SCORED", reason=None, score=float(value))
                    else:
                        reason = scored.get("reason") or "RECEIVER_OUTPUT_INVALID"
                        row.update(status="DECODE_INVALID", reason=reason, score=None)
        except Exception as exc:
            row.update(status="EXCEPTION", reason=f"{type(exc).__name__}: {exc}", score=None)
        _counts(result)
        _write_result(result_path, result)
    result["status"] = "COMPLETE" if result["invalid_count"] == 0 else "COMPLETE_WITH_INVALID"
    _write_result(result_path, result)
    return result


def _checked_output(path: Path) -> Path:
    if path.parent != OUTPUT_PARENT or not re.fullmatch(r"\d{8}T\d{12}Z", path.name):
        raise ValueError("output must be a fresh timestamp child of the fixed Drive output parent")
    if path.exists():
        if not path.is_dir() or not (path / "setup_receipt.json").is_file():
            raise FileExistsError("existing output must be the current notebook setup directory")
        allowed = {"setup_receipt.json", "setup.log", "environment_receipt.json", "source_receipt.json"}
        if any(child.name not in allowed for child in path.iterdir()):
            raise FileExistsError("output directory contains a prior run or unexpected file")
    elif not OUTPUT_PARENT.is_dir():
        raise FileNotFoundError("fixed Drive output parent is absent")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.run_root != RUN_ROOT:
        parser.error("run-root must be the fixed historical Drive run")
    output = _checked_output(args.output)
    manifest = _load_manifest()
    source_sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source SHA unavailable")
    result = run_diagnostic(RUN_ROOT, output, manifest, source_sha)
    for row in result["rows"]:
        print(row["case_id"], row["status"], row["score"], row["reason"], flush=True)
    print("result:", result["result_path"], flush=True)


if __name__ == "__main__":
    main()
