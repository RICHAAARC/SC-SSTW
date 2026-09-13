"""Conditional q-only engineering certification; never a scientific AISB gate.

Keeps the original scan and residual unchanged. The bound is an external
assumption, not estimated from q. No p, identity labels or private inputs.
"""
from dataclasses import dataclass
import math

from .public_scan import scan_public_q
from .public_candidates import PublicBurst, freeze_public_bursts


@dataclass(frozen=True)
class ValidityConfig:
    epsilon: float = 1 / 1024
    condition_limit: float = 8.0

    def __post_init__(self):
        # This version implements the one frozen engineering proposal.
        if type(self.epsilon) not in (float, int) or self.epsilon != 1 / 1024:
            raise ValueError("v2 requires frozen epsilon=1/1024")
        if type(self.condition_limit) not in (float, int) or self.condition_limit != 8:
            raise ValueError("v2 requires frozen condition certificate limit=8")


def _spectrum(columns):
    """2xn singular values, avoiding small-eigenvalue subtraction cancellation."""
    xx = math.fsum(x*x for x, y in columns)
    yy = math.fsum(y*y for x, y in columns)
    xy = math.fsum(x*y for x, y in columns)
    large2 = (xx + yy + math.hypot(xx-yy, 2*xy)) / 2
    determinant = math.fsum((a*d-b*c)**2 for i, (a, b) in enumerate(columns)
                            for c, d in columns[i+1:])
    large = math.sqrt(large2)
    small = math.sqrt(determinant / large2) if large2 else 0.0
    return small, large, determinant


def geometry_decision(small, anchor_small, condition_certificate, config):
    """Strict margin boundaries; no numerical slack or fitted tolerances."""
    delta = 2 * math.sqrt(2) * config.epsilon
    reasons = []
    if not small > 2 * config.epsilon:
        reasons.append("FULL_WINDOW_MARGIN")
    if not anchor_small > 2 * delta:
        reasons.append("ANCHOR_MARGIN")
    if condition_certificate is None or not condition_certificate <= config.condition_limit:
        reasons.append("CONDITION_CERTIFICATE")
    return reasons


def _window_metrics(window, template, config):
    e = config.epsilon
    delta = 2 * math.sqrt(2) * e
    mean = [math.fsum(p[k] for p in window)/6 for k in (0, 1)]
    columns = [((p[0]-mean[0])/math.sqrt(6), (p[1]-mean[1])/math.sqrt(6))
               for p in window]
    small, large, determinant = _spectrum(columns)
    anchors = [(window[i][0]-window[0][0], window[i][1]-window[0][1])
               for i in (1, 2)]
    amin, amax, adet = _spectrum(anchors)
    cert = (amax+delta)/(amin-delta) if amin > delta else None
    reasons = geometry_decision(small, amin, cert, config)
    checks = []
    for j in (3, 4, 5):
        alpha, beta = template.points[j]
        weights = (1-alpha-beta, alpha, beta)
        prediction = [sum(weights[k]*window[k][axis] for k in range(3)) for axis in (0, 1)]
        error = math.hypot(*(window[j][k]-prediction[k] for k in (0, 1)))
        bound = (1+sum(abs(w) for w in weights))*e
        compatible = error <= bound
        checks.append(dict(index=j, weights=weights, prediction=prediction,
                           error=error, bound=bound, margin=bound-error,
                           compatible_necessary_bound=compatible))
        if not compatible:
            reasons.append("TEMPLATE_NECESSARY_BOUND_" + str(j))
    numeric_values = [small, large, determinant, amin, amax, adet]
    numeric_values += [v for check in checks for v in (check['error'], check['bound'], check['margin'])]
    if cert is not None:
        numeric_values.append(cert)
    if not all(math.isfinite(v) for v in numeric_values):
        raise ArithmeticError('nonfinite geometry metrics')
    return dict(full_window=dict(sigma_min=small, sigma_max=large,
                sub_status=('NO_CERTIFIED_2D' if small <= e else
                            'MARGINAL_2D' if small <= 2*e else 'MARGIN_CERTIFIED'),
                ratio=small/large if large else None, lambda_min=small*small,
                lambda_max=large*large, covariance_determinant=determinant,
                certified_true_minor_lower_bound=small-e, threshold=2*e,
                margin=small-2*e),
            anchors=dict(sigma_min=amin, sigma_max=amax, determinant_squared=adet,
                delta_bound=delta, true_sigma_min_lower_bound=amin-delta,
                threshold=2*delta, margin=amin-2*delta,
                condition_certificate=cert, condition_limit=config.condition_limit),
            template_checks=checks, rejection_reasons=reasons,
            certification_status=("CONDITIONALLY_CERTIFIED_ENGINEERING" if not reasons
                                  else "NOT_CERTIFIED_UNDER_NOISE_ASSUMPTION"))


def scan_public_q_v2(q, *, templates, max_evaluations, max_retained,
                     valid=None, config=ValidityConfig()):
    """Return unchanged original pool and conditional v2 pool, full axes intact.

    Missing/nonfinite observations become invalid *at their original index*.
    Malformed dimensions or nonboolean validity raise interface errors. The
    fixed six-point/no-deletion public template contract is explicit.
    """
    templates = tuple(templates)
    if not isinstance(config, ValidityConfig):
        raise ValueError("explicit ValidityConfig required")
    if any(t.length != 6 or tuple(t.points[:3]) != ((0, 0), (1, 0), (0, 1)) for t in templates):
        raise ValueError("v2 requires six points and canonical first three anchors")
    q = list(q)
    if valid is not None and (len(valid) != len(q) or any(type(v) is not bool for v in valid)):
        raise ValueError("validity must have one boolean per original index")
    normalized, inputs = [], []
    for index, p in enumerate(q):
        reason = None
        if p is None:
            reason = "MISSING_Q"
        else:
            if not isinstance(p, (tuple, list)) or len(p) != 2:
                raise ValueError("q must have two numeric coordinates")
            if any(type(v) not in (int, float) for v in p):
                raise ValueError("q coordinates must be numeric, not bool")
            if not all(math.isfinite(v) for v in p):
                reason = "NONFINITE_Q"
        if valid is not None and not valid[index]:
            reason = reason or "OBSERVER_INVALID"
        normalized.append(None if reason else tuple(float(v) for v in p))
        inputs.append(dict(index=index, q=normalized[-1], valid=reason is None, reason=reason))
    original = scan_public_q(normalized, templates=templates, missing_sets=((),),
                             max_evaluations=max_evaluations, max_retained=max_retained)
    by_id = {t.template_id: t for t in templates}
    original_candidates, certified, records = [], [], []
    for original_record in original["records"]:
        row = dict(original_record)
        window = normalized[row["start"]:row["start"]+6]
        row["original_rank"] = None
        row["v2_rank"] = None
        row["v2_retained"] = False
        if any(p is None for p in window):
            row.update(certification_status="INVALID_Q", rejection_reasons=["INVALID_Q"],
                       invalid_indices=[row["start"]+i for i, p in enumerate(window) if p is None])
        else:
            try:
                row.update(_window_metrics(window, by_id[row["template_id"]], config))
            except (ArithmeticError, ValueError) as exc:
                row.update(certification_status="NUMERIC_EVALUATION_FAILURE",
                           rejection_reasons=["NUMERIC_EVALUATION_FAILURE"], numeric_error=str(exc))
        if row["status"] == "SCORED_ONLY":
            candidate = PublicBurst(row["template_id"], row["start"], 6, 6, (), row["residual"])
            original_candidates.append(candidate)
            if row["certification_status"] == "CONDITIONALLY_CERTIFIED_ENGINEERING":
                certified.append(candidate)
        elif row["certification_status"] == "CONDITIONALLY_CERTIFIED_ENGINEERING":
            row["rejection_reasons"].append("ORIGINAL_RESIDUAL_UNSCORABLE")
            row["certification_status"] = "NOT_CERTIFIED_UNDER_NOISE_ASSUMPTION"
        records.append(row)
    ordered = sorted(certified, key=lambda c: c.sort_key())
    original_ranks = {c.identity(): i+1 for i, c in enumerate(sorted(original_candidates, key=lambda c: c.sort_key()))}
    ranks = {c.identity(): i+1 for i, c in enumerate(ordered)}
    for row in records:
        identity = (row["template_id"], row["start"], 6, 6, ())
        row["original_rank"] = original_ranks.get(identity)
        row["v2_rank"] = ranks.get(identity)
        row["v2_retained"] = row["v2_rank"] is not None and row["v2_rank"] <= max_retained
        row["pool_disposition"] = ("RETAINED" if row["v2_retained"] else
                                   "GLOBAL_BUDGET" if row["v2_rank"] else "NOT_CERTIFIED")
    complete = original["enumeration_complete"]
    return dict(schema="public-q-validity-v2-engineering", rule_version="public-q-validity-v2-engineering-1", evidence="conditional_engineering_only",
                noise_bound_status='UNVERIFIED_NOISE_BOUND_ASSUMED_ENGINEERING',
                epsilon=config.epsilon, allowed_total_mapping_singular_values=[0.02, 0.08],
                allowed_total_mapping_condition_limit=4,
                input_rows=inputs, original=original, records=records,
                enumeration_complete=complete, evaluated_count=len(records),
                full_enumeration_count=len(records) if complete else None,
                certified_count=len(ordered), retained_count=min(len(ordered), max_retained),
                budget_dropped_count=max(0, len(ordered)-max_retained),
                ranking_scope="complete" if complete else "partial_enumeration_only",
                retained=tuple(ordered[:max_retained]),
                frozen=freeze_public_bursts(ordered, budget=max_retained, enumeration_complete=True) if complete else None)


