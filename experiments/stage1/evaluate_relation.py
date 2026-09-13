"""Development-only affine relation evaluator. Annotations never enter observer.

Fixed time split: even sampled t>=1 fit, odd t>=1 heldout. No offset search,
regularization, scientific PASS threshold, reindexing or sample replacement.
"""
import argparse
import json
import math
from pathlib import Path

SLOTS = ("horizontal", "vertical_or_diagonal", "two_dimensional_turn", "in_place_limb_motion")
COORDINATES = "normalized_xy_width_minus_1_height_minus_1"
RANK_ABS_TOL = 1e-15
RANK_REL_TOL = 1e-10


def finite_point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or not 0 <= x <= 1 for x in value)):
        raise ValueError("expected finite normalized 2-D point")
    return [float(x) for x in value]


def spread(points):
    n = len(points)
    if not n:
        return {"count": 0, "mean": None, "covariance": None,
                "lambda_min": None, "lambda_max": None, "rank": None,
                "rank_tolerance": None, "rms_radial_scale": None,
                "lambda_min_over_max": None, "minor_sd": None}
    mean = [sum(p[i] for p in points) / n for i in range(2)]
    xx = sum((p[0] - mean[0]) ** 2 for p in points) / n
    yy = sum((p[1] - mean[1]) ** 2 for p in points) / n
    xy = sum((p[0] - mean[0]) * (p[1] - mean[1]) for p in points) / n
    discriminant = math.hypot(xx - yy, 2 * xy)
    hi = max(0.0, (xx + yy + discriminant) / 2)
    lo = max(0.0, (xx + yy - discriminant) / 2)
    tol = max(RANK_ABS_TOL, hi * RANK_REL_TOL)
    return {"count": n, "mean": mean, "covariance": [[xx, xy], [xy, yy]],
            "lambda_min": lo, "lambda_max": hi, "rank": int(lo > tol) + int(hi > tol),
            "rank_tolerance": tol, "rms_radial_scale": math.sqrt(xx + yy),
            "lambda_min_over_max": lo / hi if hi > 0 else None, "minor_sd": math.sqrt(lo)}


def affine_fit(pairs):
    """Unregularized centered 2-D OLS; reject rank-deficient input explicitly."""
    source = spread([m for m, q in pairs])
    target = spread([q for m, q in pairs])
    if len(pairs) < 3 or source["rank"] != 2:
        return None, source, target
    (xx, xy), (_, yy) = source["covariance"]
    determinant = xx * yy - xy * xy
    if determinant <= 0:
        return None, source, target
    inverse = [[yy / determinant, -xy / determinant], [-xy / determinant, xx / determinant]]
    cross = [[sum((q[i] - target["mean"][i]) * (m[j] - source["mean"][j])
                  for m, q in pairs) / len(pairs) for j in range(2)] for i in range(2)]
    matrix = [[sum(cross[i][k] * inverse[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    bias = [target["mean"][i] - sum(matrix[i][j] * source["mean"][j] for j in range(2)) for i in range(2)]
    if not all(math.isfinite(v) for v in [*bias, *matrix[0], *matrix[1]]):
        raise ValueError("nonfinite unregularized fit")
    return {"A": matrix, "b": bias}, source, target


def squared_error(prediction, observed):
    return sum((a - b) ** 2 for a, b in zip(prediction, observed))


def evaluate_sample(observations, annotations, *, sample_id, position_definition):
    if annotations["sample_id"] != sample_id or annotations["position_definition"] != position_definition:
        raise ValueError("annotation identity or position definition mismatch")
    if annotations["coordinate_system"] != COORDINATES:
        raise ValueError("annotation coordinates must match observer normalization")
    width, height = annotations["frame_width"], annotations["frame_height"]
    if any(type(v) is not int or v < 2 for v in (width, height)):
        raise ValueError("annotation frame dimensions required")
    if not observations:
        raise ValueError("empty observation record")
    annotated = {}
    for entry in annotations["rows"]:
        index = entry["sample_index"]
        if type(index) is not int or index < 0 or index >= len(observations) or index in annotated:
            raise ValueError("duplicate or out-of-range annotation index")
        annotated[index] = entry
    rows = []
    for index, observation in enumerate(observations):
        if observation["sample_index"] != index or type(observation["sample_index"]) is not int:
            raise ValueError("observation rows must preserve contiguous sampled indices")
        time = observation["time_seconds"]
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
            raise ValueError("invalid observation time")
        if index and time <= observations[index - 1]["time_seconds"]:
            raise ValueError("nonincreasing observation time")
        entry = annotated.get(index)
        p, annotation_reason = None, "ANNOTATION_ROW_MISSING"
        p_pixel, uncertainty, normalized_uncertainty = None, None, None
        if entry is not None:
            annotated_time = entry["time_seconds"]
            if (isinstance(annotated_time, bool) or not isinstance(annotated_time, (int, float)) or
                    not math.isfinite(annotated_time) or abs(annotated_time - time) > 1e-9):
                raise ValueError("annotation timestamp mismatch; no offset search allowed")
            p = finite_point(entry["p"]) if entry["p"] is not None else None
            if p is None and (entry.get("p_pixel") is not None or entry.get("uncertainty_pixels") is not None):
                raise ValueError("missing normalized annotation must not carry pixel position or uncertainty")
            if p is not None:
                p_pixel = entry["p_pixel"]
                if (not isinstance(p_pixel, (list, tuple)) or len(p_pixel) != 2 or
                        any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                            for v in p_pixel)):
                    raise ValueError("finite pixel annotation required")
                if any(abs(p[i] - p_pixel[i] / (dimension - 1)) > 1e-9 for i, dimension in enumerate((width, height))):
                    raise ValueError("pixel and normalized annotation disagree")
                uncertainty = entry["uncertainty_pixels"]
                if (isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)) or
                        not math.isfinite(uncertainty) or uncertainty < 0):
                    raise ValueError("nonnegative manual uncertainty radius required")
                normalized_uncertainty = uncertainty * max(1 / (width - 1), 1 / (height - 1))
            annotation_reason = entry["reason"]
            if not isinstance(annotation_reason, str) or not annotation_reason:
                raise ValueError("annotation reason required even for missing positions")
        if type(observation["valid"]) is not bool:
            raise ValueError("observation validity must be boolean")
        q = finite_point(observation["q"]) if observation["valid"] else None
        if not observation["valid"] and observation["q"] is not None:
            raise ValueError("invalid q must be null, not filled")
        previous_p = rows[-1]["p"] if rows else None
        m = [(a + b) / 2 for a, b in zip(previous_p, p)] if previous_p is not None and p is not None else None
        m_uncertainty = (rows[-1]["normalized_uncertainty_radius"] + normalized_uncertainty) / 2 if m is not None else None
        role = "initial" if index == 0 else ("fit" if index % 2 == 0 else "heldout")
        reason = ("INITIAL_NO_MIDPOINT" if not index else "ANNOTATION_PAIR_MISSING" if m is None
                  else "OBSERVER_INVALID" if q is None else "ELIGIBLE")
        rows.append({"sample_index": index, "time_seconds": time, "role": role,
                     "p": p, "m": m, "q": q, "q_valid": observation["valid"],
                     "p_pixel": p_pixel, "uncertainty_pixels": uncertainty,
                     "normalized_uncertainty_radius": normalized_uncertainty,
                     "m_uncertainty_radius": m_uncertainty,
                     "q_reason": observation["reason"], "annotation_reason": annotation_reason,
                     "eligible": reason == "ELIGIBLE", "eligibility_reason": reason,
                     "prediction": None, "baseline": None, "squared_error": None,
                     "baseline_squared_error": None, "normalized_error": None,
                     "normalized_baseline_error": None})
    fit_rows = [r for r in rows if r["role"] == "fit" and r["eligible"]]
    heldout = [r for r in rows if r["role"] == "heldout" and r["eligible"]]
    model, fit_spread, fit_q_spread = affine_fit([(r["m"], r["q"]) for r in fit_rows])
    baseline = fit_q_spread["mean"]
    scale = fit_spread["rms_radial_scale"]
    for row in rows:
        if not row["eligible"]:
            continue
        if baseline is not None:
            row["baseline"] = baseline
            row["baseline_squared_error"] = squared_error(baseline, row["q"])
            if scale is not None and scale > 0:
                row["normalized_baseline_error"] = math.sqrt(row["baseline_squared_error"]) / scale
        if model is not None:
            row["prediction"] = [sum(model["A"][i][j] * row["m"][j] for j in range(2)) + model["b"][i] for i in range(2)]
            row["squared_error"] = squared_error(row["prediction"], row["q"])
            if scale is not None and scale > 0:
                row["normalized_error"] = math.sqrt(row["squared_error"]) / scale
    errors = [r["squared_error"] for r in heldout if r["squared_error"] is not None]
    baseline_errors = [r["baseline_squared_error"] for r in heldout if r["baseline_squared_error"] is not None]
    rmse = math.sqrt(sum(errors) / len(errors)) if errors else None
    baseline_rmse = math.sqrt(sum(baseline_errors) / len(baseline_errors)) if baseline_errors else None
    all_p_spread = spread([r["p"] for r in rows if r["p"] is not None])
    drift_q = spread([r["q"] for r in rows if r["q"] is not None])
    fit_uncertainty = math.sqrt(sum(r["m_uncertainty_radius"] ** 2 for r in fit_rows) / len(fit_rows)) if fit_rows else None
    return {"sample_id": sample_id, "terminal": "INPUT_RANK_INSUFFICIENT_FOR_2D_AFFINE" if model is None
            else "NO_ELIGIBLE_HELDOUT" if not heldout else "EVALUATED_DEVELOPMENT_NO_SCIENTIFIC_PASS",
            "rows": rows, "model": model, "baseline_fit_q_mean": baseline,
            "fit_indices": [r["sample_index"] for r in fit_rows],
            "heldout_indices": [r["sample_index"] for r in heldout],
            "fixed_time_counts": {role: sum(r["role"] == role for r in rows) for role in ("initial", "fit", "heldout")},
            "eligible_counts": {"fit": len(fit_rows), "heldout": len(heldout)},
            "spread": {"all_annotated_p": all_p_spread,
                       "all_available_m": spread([r["m"] for r in rows if r["m"] is not None]),
                       "all_valid_q": drift_q, "fit_eligible_m": fit_spread, "fit_eligible_q": fit_q_spread,
                       "heldout_eligible_m": spread([r["m"] for r in heldout]),
                       "heldout_eligible_q": spread([r["q"] for r in heldout])},
            "heldout_metrics": {"error_count": len(errors), "baseline_error_count": len(baseline_errors),
                                "rmse": rmse, "baseline_rmse": baseline_rmse,
                                "normalized_rmse": rmse / scale if rmse is not None and scale is not None and scale > 0 else None,
                                "normalized_baseline_rmse": baseline_rmse / scale if baseline_rmse is not None and scale is not None and scale > 0 else None,
                                "relative_rmse_improvement": 1 - rmse / baseline_rmse if rmse is not None and baseline_rmse is not None and baseline_rmse > 0 else None},
            "stationary_annotation_diagnostics": {"exact_zero_p_scale": all_p_spread["rms_radial_scale"] == 0,
                                                  "q_valid_count": drift_q["count"], "q_rms_drift": drift_q["rms_radial_scale"],
                                                  "false_validity_decision": "UNDETERMINED_NO_SCIENTIFIC_THRESHOLD"},
            "annotation_uncertainty": {"fit_midpoint_rms_radius_bound": fit_uncertainty,
                                       "fit_minor_sd_over_uncertainty": fit_spread["minor_sd"] / fit_uncertainty if fit_uncertainty is not None and fit_uncertainty > 0 else None,
                                       "normalization_scale_over_uncertainty": scale / fit_uncertainty if scale is not None and fit_uncertainty is not None and fit_uncertainty > 0 else None,
                                       "dimensional_readability": "UNDETERMINED_MECHANICAL_RANK_IS_NOT_SCIENTIFIC_2D_SUPPORT"},
            "uncertainty": "Manual uncertainty radii are subjective estimates, not statistical confidence bounds. Annotation noise can create numerical rank two. Temporally correlated heldout rows are not independent replication."}


def run_evaluation(manifest, output_dir):
    slots = manifest["slots"]
    if tuple(s["slot"] for s in slots) != SLOTS:
        raise ValueError("all four fixed slots required in protocol order")
    if not isinstance(manifest["position_definition"], str) or not manifest["position_definition"]:
        raise ValueError("frozen position definition required")
    output = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[2]
    if output == repository or repository in output.parents:
        raise ValueError("outputs must remain outside Git checkout")
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    results = []
    for slot in slots:
        if slot["status"] == "MISSING":
            results.append({"slot": slot["slot"], "sample_id": slot.get("sample_id"),
                            "terminal": "SAMPLE_MISSING", "reason": slot["reason"], "rows": []})
            continue
        try:
            if slot["status"] != "READY":
                raise ValueError("slot status must be READY or MISSING")
            source = json.loads(Path(slot["observation_results_path"]).read_text(encoding="utf-8"))
            matches = [r for r in source["rows"] if r["sample_id"] == slot["sample_id"]]
            if len(matches) != 1:
                raise ValueError("observation sample ID must match exactly once")
            if matches[0].get("terminal") == "OPERATIONAL_BLOCKED":
                raise ValueError("source observation run operationally blocked; partial repeat is not a completed input")
            annotations = json.loads(Path(slot["annotations_path"]).read_text(encoding="utf-8"))
            repeat = matches[0]["repeats"][0]
            if (annotations["frame_width"], annotations["frame_height"]) != (repeat["width"], repeat["height"]):
                raise ValueError("annotation dimensions differ from actual decoded observation frames")
            result = evaluate_sample(repeat["observations"], annotations,
                                     sample_id=slot["sample_id"], position_definition=manifest["position_definition"])
            result["slot"] = slot["slot"]
            results.append(result)
        except Exception as exc:
            results.append({"slot": slot["slot"], "sample_id": slot.get("sample_id"),
                            "terminal": "OPERATIONAL_BLOCKED", "reason": f"{type(exc).__name__}: {exc}", "rows": []})
    result = {"fixed_slot_denominator": 4, "rows": results,
              "claim_ceiling": "development relation diagnostics only; no scientific PASS, AISB truth or independent replication"}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_evaluation(json.loads(Path(args.manifest).read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
