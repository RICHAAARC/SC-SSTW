"""Bounded public enumeration; residuals do not establish valid AISB detection."""
import math
from .aisb import affine_burst_residual
from .public_candidates import PublicBurst, freeze_public_bursts


def scan_public_q(q, *, templates, missing_sets, max_evaluations, max_retained):
    """q contains public 2-D values or None for invalid observation positions.

    Never removes invalid positions, uses truth indices, fits a channel, or
    applies a residual acceptance threshold. Exact constant/collinear windows
    are retained as diagnostic scores, explicitly not valid AISB evidence.
    """
    if any(type(v) is not int or v < 1 for v in (max_evaluations, max_retained)):
        raise ValueError("positive explicit budgets required")
    templates = sorted(templates, key=lambda t: t.template_id)
    if not templates or len({t.template_id for t in templates}) != len(templates):
        raise ValueError("nonempty unique templates required")
    missing_sets = tuple(sorted(missing_sets))
    if not missing_sets or len(set(missing_sets)) != len(missing_sets):
        raise ValueError("explicit unique missing sets required")
    for template in templates:
        if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in template.points):
            raise ValueError("template points must be finite 2-D")
        for missing in missing_sets:
            PublicBurst(template.template_id, 0, template.length - len(missing),
                        template.length, missing, 0.0)
    for p in q:
        if p is not None and (len(p) != 2 or not all(math.isfinite(v) for v in p)):
            raise ValueError("nonfinite public q must be marked invalid, never zero-filled")
    scored, records, complete = [], [], True
    for template in templates:
        for missing in missing_sets:
            length = template.length - len(missing)
            for start in range(max(0, len(q) - length + 1)):
                if len(records) >= max_evaluations:
                    complete = False
                    break
                window = q[start:start + length]
                record = {"template_id": template.template_id, "start": start,
                          "observed_length": length, "missing_indices": missing}
                if any(p is None for p in window):
                    record["status"] = "INVALID_OBSERVATION_WINDOW"
                else:
                    n = len(window)
                    mx, my = (sum(p[i] for p in window) / n for i in range(2))
                    xx = sum((p[0] - mx) ** 2 for p in window) / n
                    yy = sum((p[1] - my) ** 2 for p in window) / n
                    xy = sum((p[0] - mx) * (p[1] - my) for p in window) / n
                    record.update(scatter=xx + yy, covariance_determinant=max(0.0, xx * yy - xy * xy),
                                  aisb_validity="UNDETERMINED_NO_FROZEN_VALIDITY_RULE")
                    try:
                        residual = affine_burst_residual(window, template, missing_template_index=missing or None)
                        candidate = PublicBurst(template.template_id, start, length, template.length, missing, residual)
                        scored.append(candidate)
                        record.update(status="SCORED_ONLY", residual=candidate.residual)
                    except (ValueError, ArithmeticError) as exc:
                        record.update(status="UNSCORABLE_PUBLIC_GEOMETRY", reason=str(exc))
                records.append(record)
            if not complete:
                break
        if not complete:
            break
    retained = tuple(sorted(scored, key=lambda c: c.sort_key())[:max_retained])
    return {"enumeration_complete": complete, "evaluated_count": len(records),
            "full_enumeration_count": len(records) if complete else None,
            "scored_count": len(scored), "retained_count": len(retained),
            "budget_dropped_count": len(scored) - len(retained),
            "stop_reason": "complete" if complete else "ALGORITHM_EVALUATION_BUDGET",
            "records": records, "retained": retained,
            "frozen": freeze_public_bursts(scored, budget=max_retained, enumeration_complete=True) if complete else None}
