"""Fixed-manifest CPU observation runner. Outputs are engineering records only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from runtime.stage1.observation import (
    ObserverConfig, decode_video, observe_frames, summarize_observations, repeat_error,
)
from main.sc_sstw.aisb import BurstTemplate
from main.sc_sstw.public_scan import scan_public_q
from main.sc_sstw.public_candidates import read_frozen_bursts, candidate_digest


def validate_manifest(manifest):
    if set(manifest) not in ({"samples", "observer", "decode"},
                             {"samples", "observer", "decode", "public_acquisition"}):
        raise ValueError("manifest needs samples, observer, decode and optional public_acquisition")
    config = ObserverConfig(**manifest["observer"])
    samples = manifest["samples"]
    if not samples:
        raise ValueError("fixed nonempty sample list required")
    ids = []
    for sample in samples:
        if set(sample) != {"sample_id", "path", "split", "content_category"}:
            raise ValueError("unexpected or missing public sample fields")
        if not all(isinstance(v, str) and v for v in sample.values()):
            raise ValueError("sample metadata must be nonempty strings")
        if sample["split"] not in {"development", "validation"}:
            raise ValueError("split must be frozen as development or validation")
        ids.append(sample["sample_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs")
    if set(manifest["decode"]) != {"sample_hz", "max_frames", "max_pixels", "timeout_seconds"}:
        raise ValueError("all decode budgets must be explicit")
    return config


def run_manifest(manifest, output_dir, *, decoder=decode_video):
    config = validate_manifest(manifest)
    output = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[2]
    if output == repository or repository in output.parents:
        raise ValueError("run outputs must be outside Git checkout")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "input_manifest.json"
    if snapshot.exists() or (output / "results.json").exists():
        raise FileExistsError("use a fresh output directory; never replace a frozen run")
    write_json(snapshot, manifest)
    rows = []
    for sample in manifest["samples"]:
        row = {**sample, "terminal": "OPERATIONAL_BLOCKED", "evidence": "engineering_only",
               "public_coverage": "UNDETERMINED_NO_POSITIVE", "candidate_freeze": "NOT_RUN",
               "repeats": []}
        try:
            path = Path(sample["path"])
            if not path.is_absolute() or not path.is_file():
                raise FileNotFoundError("sample path must name an existing absolute video file")
            observations = []
            for repeat in range(2):
                width, height, frames = decoder(path, **manifest["decode"])
                observed = observe_frames(frames, width=width, height=height,
                                          sample_hz=manifest["decode"]["sample_hz"], config=config)
                observations.append(observed)
                row["repeats"].append({"repeat": repeat, "width": width, "height": height,
                                       "observations": [asdict(r) for r in observed],
                                       "summary": summarize_observations(observed)})
            row["stability"] = repeat_error(*observations)
            row["terminal"] = ("NO_GO_THIS_CONSTRUCTION" if not any(r.valid for r in observations[0])
                               else "OBSERVED_SCIENTIFIC_GATES_UNDETERMINED")
            if "public_acquisition" in manifest:
                acquisition = manifest["public_acquisition"]
                templates = tuple(BurstTemplate(t["template_id"], tuple(tuple(p) for p in t["points"]))
                                  for t in acquisition["templates"])
                scan = scan_public_q([r.q for r in observations[0]], templates=templates,
                                     missing_sets=tuple(tuple(m) for m in acquisition["missing_sets"]),
                                     max_evaluations=acquisition["max_evaluations"],
                                     max_retained=acquisition["max_retained"])
                frozen = scan.pop("frozen")
                scan["retained"] = [asdict(c) for c in scan["retained"]]
                row["public_acquisition"] = scan
                if frozen is None:
                    row["candidate_freeze"] = "INCOMPLETE_ALGORITHM_BUDGET"
                else:
                    read_frozen_bursts(frozen)
                    frozen_name = f"candidates_{len(rows):04d}.json"
                    (output / frozen_name).write_bytes(frozen)
                    row["candidate_freeze"] = "ENGINEERING_FROZEN_READBACK"
                    row["candidate_file"] = frozen_name
                    row["candidate_sha256"] = candidate_digest(frozen)
        except Exception as exc:
            row["terminal"] = "OPERATIONAL_BLOCKED"
            row["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        rows.append(row)
        write_json(output / "partial_results.json", {"fixed_denominator": len(manifest["samples"]), "rows": rows})
    result = {"fixed_denominator": len(manifest["samples"]), "terminal_count": len(rows),
              "all_ids_reconciled": [r["sample_id"] for r in rows] == [s["sample_id"] for s in manifest["samples"]],
              "science_denominator": 0, "rows": rows}
    write_json(output / "results.json", result)
    return result


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = run_manifest(manifest, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
