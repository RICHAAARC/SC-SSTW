"""Adopted M3-151 geometry, complete local families and conditional paths.

No model, media, payload cache, private source or filesystem access. Q40 scoring
delegates to the existing pure method; the other APIs use only the stdlib.
"""
from collections.abc import Mapping
from numbers import Integral, Real
import math

SOURCE_FRAMES = 181
RECEIVED_FRAMES = 151
WINDOWS = {"L": 0, "M": 53, "R": 106}
H = tuple((p, s) for p in range(4) for s in range(1 if p == 0 else 2, 36))
H_SET = frozenset(H)
GAPS = {"LEFT_GAP": (45, 53), "RIGHT_GAP": (98, 106)}
PATH_CLASS_COUNT = 961
SEAM_HYPOTHESIS_COUNT = 8401
CALIBRATION_FAMILY_COUNT = 90
TAU_RULE = "max(0,max_S_of_90_complete_null_window_families)"


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def origin_to_state(origin):
    """Query source origin, not the whole received input's source origin."""
    if not _integer(origin) or not 0 <= origin <= 136:
        raise ValueError("query source origin must be an integer in 0..136")
    p = (-int(origin)) % 4
    return p, (int(origin) + p) // 4 + 1


def state_to_origin(phase, source_start):
    if not _integer(phase) or not _integer(source_start) or (phase, source_start) not in H_SET:
        raise ValueError("state must belong to H137")
    return 4 * (int(source_start) - 1) - int(phase)


def cell_mapping(received_query_start, phase, source_start, local_row):
    """Nominal four-frame mapping; never claims a VAE causal receptive field."""
    if not _integer(received_query_start) or received_query_start not in WINDOWS.values():
        raise ValueError("M3 received query start must be 0, 53 or 106")
    if not _integer(local_row) or not 1 <= local_row <= 10:
        raise ValueError("retained E41 local row must be 1..10")
    origin = state_to_origin(phase, source_start)
    w, p, r = int(received_query_start), int(phase), int(local_row)
    t = int(source_start) + r - 1
    local = [p + 4 * r - 3, p + 4 * r]
    received = [w + local[0], w + local[1]]
    source_frames = [origin + x - w for x in range(received[0], received[1] + 1)]
    return dict(local_row=r, phase=p, canonical_cell=t, group=(t - 1) // 11,
                query_local_interval=local, nominal_received_cell=received,
                source_frames_under_h=source_frames,
                source_support_under_h=dict(status="FULL_UNDER_H", canonical_cell=t,
                    group=(t - 1) // 11, source_frames=source_frames,
                    actual_support_known=False, vae_causal_support=False))


def public_protocol():
    return dict(source_frames=SOURCE_FRAMES, received_frames=RECEIVED_FRAMES,
                windows=WINDOWS.copy(), query_frames=45, encoder_frames=41,
                phases=list(range(4)), retained_local_rows=list(range(1, 11)),
                seam_intervals={k: list(v) for k, v in GAPS.items()},
                a_plus_d_max=30, local_domain_count=137,
                path_class_count=PATH_CLASS_COUNT, seam_hypothesis_count=SEAM_HYPOTHESIS_COUNT,
                calibration_null_window_families=90, evaluation_tasks=36,
                calibration_path_tasks=0, tau_rule=TAU_RULE,
                source_authenticated=False, exact_seam_certified=False,
                model_outside_scope_may_still_match=True)


def score_family(phase_bank, background_model):
    """Existing Q40 computation only. No fit, old threshold, FFT or payload."""
    if not isinstance(phase_bank, Mapping) or any(
            int(p in phase_bank) + int(str(p) in phase_bank) != 1 for p in range(4)):
        return dict(status="SCORE_UNRESOLVED", cause_set=["INCOMPLETE_OR_DUPLICATE_PHASES"],
                    candidates=[dict(phase=p, source_start=s, S=None,
                                     score_status="TECHNICAL_UNRESOLVED") for p, s in H])
    from . import b_background_normalized_match_v1 as score
    result = score.score_family(phase_bank, background_model)
    return {k: v for k, v in result.items() if k != "phase_local_bits"}


def _score_rows(family):
    """Canonicalize all137 planned slots without dropping bad/extra inputs."""
    family = family if isinstance(family, Mapping) else {}
    supplied = family.get("candidates", family.get("members", []))
    causes = []
    if not isinstance(supplied, (list, tuple)):
        supplied = []
        causes.append("INVALID_MEMBER_CONTAINER")
    if len(supplied) != 137:
        causes.append("MEMBER_COUNT_MISMATCH")
    upstream_status = family.get("score_status", family.get("status"))
    if upstream_status != "SCORES_COMPLETE":
        causes.append("UPSTREAM_SCORE_UNRESOLVED")
        if isinstance(upstream_status, str):
            causes.append("UPSTREAM:" + upstream_status)
    by_state, unexpected = {}, []
    for i, row in enumerate(supplied):
        if not isinstance(row, Mapping):
            unexpected.append(dict(index=i, reason="INVALID_MEMBER"))
            continue
        p, s = row.get("phase"), row.get("source_start")
        if not _integer(p) or not _integer(s) or (p, s) not in H_SET:
            unexpected.append(dict(index=i, reason="OUT_OF_DOMAIN_MEMBER"))
            continue
        by_state.setdefault((int(p), int(s)), []).append(row)
    if unexpected:
        causes.append("UNEXPECTED_MEMBERS")
    rows, missing = [], []
    for p, s in H:
        group = by_state.get((p, s), [])
        reason = []
        if not group:
            reason.append("MISSING_MEMBER")
        elif len(group) != 1:
            reason.append("DUPLICATE_MEMBER")
        raw = group[0] if len(group) == 1 else {}
        value = raw.get("S")
        if raw.get("score_status") != "RETURNED":
            reason.append("SCORE_NOT_RETURNED")
            if isinstance(raw.get("score_status"), str):
                reason.append("UPSTREAM:" + raw["score_status"])
        if not _finite(value) or not 0 <= value <= 1:
            reason.append("INVALID_SCORE")
        valid = not reason
        rows.append(dict(phase=p, source_start=s, S=float(value) if valid else None,
                         score_status="RETURNED" if valid else "TECHNICAL_UNRESOLVED",
                         upstream_score_status=raw.get("score_status") if isinstance(raw.get("score_status"), str) else None,
                         cause_set=reason))
        if reason:
            missing.append(dict(phase=p, source_start=s, cause_set=reason))
    if missing:
        causes.append("INCOMPLETE_H137")
    return rows, sorted(set(causes)), missing, unexpected, len(supplied)


def calibrate(null_families):
    """Exactly90 anonymous null families, selected by the outer fixed scheduler.

    Accept an id->raw_family mapping or [{family_id, family}, ...]. Positives and
    source/fixture/arm labels are not arguments. No paths are computed here.
    """
    if isinstance(null_families, Mapping):
        entries = [dict(family_id=k, family=v) for k, v in null_families.items()]
    elif isinstance(null_families, (list, tuple)):
        entries = list(null_families)
    else:
        entries = []
    causes = [] if len(entries) == 90 else ["NULL_FAMILY_COUNT_MISMATCH"]
    ids, maxima, missing = set(), [], []
    for i in range(90):
        entry = entries[i] if i < len(entries) and isinstance(entries[i], Mapping) else {}
        fid = entry.get("family_id")
        issues = []
        if not isinstance(fid, str) or not fid:
            issues.append("MISSING_FAMILY_ID")
            fid = None
        elif fid in ids:
            issues.append("DUPLICATE_FAMILY_ID")
        else:
            ids.add(fid)
        rows, row_issues, member_missing, unexpected, submitted = _score_rows(entry.get("family"))
        issues += row_issues
        item = dict(slot=i, family_id=fid, M=None if issues else max(r["S"] for r in rows),
                    status="TECHNICAL_UNRESOLVED" if issues else "RETURNED",
                    cause_set=sorted(set(issues)), missing_members=member_missing,
                    unexpected_members=unexpected, submitted_member_count=submitted)
        maxima.append(item)
        if issues:
            missing.append(dict(slot=i, family_id=fid, cause_set=item["cause_set"],
                                missing_members=member_missing))
    if missing:
        causes.append("INCOMPLETE_NULL_FAMILIES")
    complete = not causes
    return dict(status="FROZEN" if complete else "CALIBRATION_UNRESOLVED",
                tau=max(0.0, *(r["M"] for r in maxima)) if complete else None,
                tau_rule=TAU_RULE, maxima=maxima, missing_members=missing,
                unexpected_family_slots=list(range(90, len(entries))),
                planned_families=90, submitted_families=len(entries),
                planned_score_slots=12330, cause_set=causes,
                positive_used=False, payload_used=False, path_calculations=0,
                CFAR=False, population_FPR=None)


def prepare_local(family, threshold, query_id, received_query_start):
    """Freeze full local C and explicit four-frame geometry, never reading bits."""
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("nonempty opaque query_id required")
    # Validate even an empty/failed family; do not manufacture negative origins.
    cell_mapping(received_query_start, 0, 1, 1)
    rows, causes, missing, unexpected, submitted = _score_rows(family)
    numeric = not causes
    threshold = threshold if isinstance(threshold, Mapping) else {}
    tau = threshold.get("tau")
    good_tau = threshold.get("status") == "FROZEN" and _finite(tau) and 0 <= tau <= 1
    tau = float(tau) if good_tau else None
    maximum = max(r["S"] for r in rows) if numeric else None
    D = bool(maximum > tau) if numeric and good_tau else None
    for row in rows:
        p, s = row["phase"], row["source_start"]
        origin = state_to_origin(p, s)
        row.update(candidate_id=f"{query_id}/h{p}:{s}", s=s, O_h=origin,
                   query_source_origin_hypothesis=origin,
                   received_query_start=int(received_query_start),
                   local_received_to_source_map=[cell_mapping(received_query_start, p, s, r)
                                                 for r in range(1, 11)],
                   in_C=(row["S"] >= tau) if numeric and good_tau else None)
    C = [row for row in rows if row["in_C"]] if numeric and good_tau else None
    status = ("TECHNICAL_UNRESOLVED" if not numeric else "CALIBRATION_UNRESOLVED" if not good_tau
              else "DETECTED_PROVISIONAL" if D else "LOCAL_GATE_REJECT")
    result = dict(query_id=query_id, received_query_start=int(received_query_start),
                  status=status, score_status="SCORES_COMPLETE" if numeric else "SCORE_UNRESOLVED",
                  score_kind="Q40_background_normalized_S",
                  domain_count=137, submitted_member_count=submitted,
                  completeness="COMPLETE" if numeric else "UNRESOLVED", score_complete=numeric,
                  members=rows, C=C, D=D, detection=D, maximum=maximum, tau=tau,
                  threshold_status="FROZEN" if good_tau else "CALIBRATION_UNRESOLVED",
                  tau_rule=TAU_RULE, missing_members=missing, unexpected_members=unexpected,
                  cause_set=causes + ([] if good_tau else ["CALIBRATION_UNRESOLVED"]),
                  paths_pruned_local_C=False, payload_used=False, actual_support_known=False)
    # Only anonymous input/key metadata may pass through. Never copy other fields.
    if isinstance(family, Mapping):
        if isinstance(family.get("status"), str):
            result["upstream_status"] = family["status"]
        if isinstance(family.get("error"), str):
            result["upstream_error"] = family["error"]
        for key in ("input_id", "key_id"):
            if isinstance(family.get(key), str):
                result[key] = family[key]
    return result


def path_domain():
    """All961 classes. Each positive class keeps its nine seam aliases."""
    rows = []
    for a in range(31):
        options = [(0, "CONT")] + [(d, gap) for d in range(1, 31 - a) for gap in GAPS]
        for d, gap in options:
            origins = {"L": a, "M": a + 53 + (d if gap == "LEFT_GAP" else 0),
                       "R": a + 106 + d}
            interval = list(GAPS[gap]) if d else None
            rows.append(dict(path_id=f"{gap}:{a}:{d}", a=a, d=d, gap=gap,
                query_source_origins=origins,
                states={w: dict(phase=origin_to_state(o)[0], source_start=origin_to_state(o)[1])
                        for w, o in origins.items()},
                valid_received_domain=[0, 150], seam_alias_interval=interval,
                seam_multiplicity=9 if d else 1,
                mapping_under_h=dict(kind="SINGLE_GAP_CONDITIONAL" if d else "CONTINUOUS_CONDITIONAL",
                    a=a, d=d, seam_alias_interval=interval,
                    formula="a+u+d*I[u>=b]" if d else "a+u",
                    unobserved_frames_are_conditional=True),
                source_relation="SINGLE_INPUT_CONDITIONAL", exact_seam_certified=False))
    return rows


def map_received_frame(a, d, seam, received_index):
    """One explicit hypothesis, including one chosen seam alias, not truth."""
    if not all(_integer(x) for x in (a, d, received_index)):
        raise ValueError("integer hypothesis coordinates required")
    if a < 0 or d < 0 or a + d > 30 or not 0 <= received_index < 151:
        raise ValueError("outside M3 hypothesis domain")
    if d == 0:
        if seam is not None:
            raise ValueError("continuous hypothesis has no seam")
        return int(a + received_index)
    if not _integer(seam) or not any(lo <= seam <= hi for lo, hi in GAPS.values()):
        raise ValueError("seam outside the two declared safe intervals")
    return int(a + received_index + (d if received_index >= seam else 0))


def _checked_local(family, window):
    """Check semantic inputs; altered denominators/C cannot produce a small set."""
    if not isinstance(family, Mapping):
        return None, ["MISSING_LOCAL_FAMILY"]
    causes = list(family.get("cause_set", []))
    if family.get("domain_count") != 137:
        causes.append("LOCAL_DOMAIN_COUNT_MISMATCH")
    if family.get("received_query_start") != WINDOWS[window]:
        causes.append("RECEIVED_QUERY_START_MISMATCH")
    qid = family.get("query_id")
    if not isinstance(qid, str) or not qid:
        return None, sorted(set(causes + ["MISSING_QUERY_ID"]))
    clean = prepare_local(family, dict(status=family.get("threshold_status"), tau=family.get("tau")),
                          qid, WINDOWS[window])
    causes += clean["cause_set"]
    if not causes:
        supplied_C = family.get("C")
        states = [(r.get("phase"), r.get("source_start")) for r in supplied_C
                  if isinstance(r, Mapping)] if isinstance(supplied_C, list) else None
        expected = [(r["phase"], r["source_start"]) for r in clean["C"]]
        if states != expected or not isinstance(supplied_C, list) or len(states) != len(supplied_C):
            causes.append("LOCAL_C_MISMATCH")
        if type(family.get("D")) is not bool or family["D"] != clean["D"]:
            causes.append("LOCAL_DETECTION_MISMATCH")
        if family.get("maximum") != clean["maximum"]:
            causes.append("LOCAL_MAXIMUM_MISMATCH")
        for row in family["members"]:
            origin = state_to_origin(row["phase"], row["source_start"])
            if row.get("O_h") != origin or row.get("query_source_origin_hypothesis") != origin:
                causes.append("QUERY_SOURCE_ORIGIN_MISMATCH")
                break
    return clean, sorted(set(causes))


def paths(locals_by_window, mode):
    """Evaluate one E2/M3 task. The outer scheduler calls this for eval only.

    Calibration has no path slots or hidden execution branch. M is not accessed
    by E2, so an absent/failed M cannot erase a returned endpoint result.
    """
    if mode not in ("E2", "M3"):
        raise ValueError("mode must be E2 or M3")
    required = ("L", "R") if mode == "E2" else ("L", "M", "R")
    locals_by_window = locals_by_window if isinstance(locals_by_window, Mapping) else {}
    clean, issues = {}, {}
    for w in required:
        clean[w], issues[w] = _checked_local(locals_by_window.get(w), w)
    causes = [f"{w}:{cause}" for w in required for cause in issues[w]]
    if not causes:
        for field in ("input_id", "key_id", "tau"):
            values = {clean[w][field] for w in required if field in clean[w]}
            if len(values) > 1:
                causes.append(field.upper() + "_MISMATCH")
        if len({clean[w]["query_id"] for w in required}) != len(required):
            causes.append("QUERY_ID_REUSED_ACROSS_WINDOWS")
    defined = not causes
    detected = all(clean[w]["D"] for w in required) if defined else None
    sets = {w: {(r["phase"], r["source_start"]) for r in clean[w]["C"]}
            for w in required} if defined else {}
    rows = path_domain()
    compatible_ids, compatible_rows = [], []
    for row in rows:
        row["observed_windows"] = list(required)
        row["member_candidate_ids"] = {
            w: (f"{clean[w]['query_id']}/h{row['states'][w]['phase']}:{row['states'][w]['source_start']}"
                if clean.get(w) else None) for w in required}
        row["compatibility_mask"] = {
            w: ((row["states"][w]["phase"], row["states"][w]["source_start"]) in sets[w]
                if defined else None) for w in required}
        compatible = bool(detected and all(row["compatibility_mask"].values())) if defined else None
        row["compatibility"] = compatible
        row["cause_set"] = (list(causes) if not defined else
                            [f"{w}:LOCAL_GATE_REJECT" for w in required if not clean[w]["D"]] if not detected else
                            [f"{w}:MEMBER_NOT_IN_C" for w in required if not row["compatibility_mask"][w]])
        row["status"] = ("PATH_TECHNICAL_UNRESOLVED" if not defined else "PATH_LOCAL_GATE_REJECT"
                         if not detected else "COMPATIBLE" if compatible else "INCOMPATIBLE")
        if compatible:
            compatible_ids.append(row["path_id"])
            compatible_rows.append(row)
    status = ("CALIBRATION_UNRESOLVED" if any("CALIBRATION_UNRESOLVED" in c for c in causes)
              else "PATH_TECHNICAL_UNRESOLVED" if not defined else "PATH_LOCAL_GATE_REJECT" if not detected
              else "CONDITIONAL_PATH_SET" if compatible_rows else "PATH_INCOMPATIBLE")
    if status == "CALIBRATION_UNRESOLVED":
        for row in rows:
            row["status"] = status
    lengths = sorted({r["d"] for r in compatible_rows}) if defined else None
    location = (status if not compatible_rows else "CONTINUITY_NOT_EXCLUDED" if 0 in lengths
                else "POSITIVE_JUMP_WITHIN_DECLARED_MODEL")
    return dict(mode=mode, status=status, location_status=location, required_windows=list(required),
                all_required_detected=detected, paths=rows,
                compatible_path_ids=compatible_ids if defined else None,
                compatible_class_count=len(compatible_rows) if defined else None,
                compatible_seam_hypothesis_count=sum(r["seam_multiplicity"] for r in compatible_rows)
                    if defined else None,
                possible_a=sorted({r["a"] for r in compatible_rows}) if defined else None,
                possible_d=lengths,
                possible_gap=sorted({r["gap"] for r in compatible_rows}) if defined else None,
                domain_count=PATH_CLASS_COUNT, seam_hypothesis_count=SEAM_HYPOTHESIS_COUNT,
                planned=PATH_CLASS_COUNT, returned=PATH_CLASS_COUNT if defined else 0,
                failed=0 if defined else PATH_CLASS_COUNT, not_entered=0,
                cause_set=causes, local_C_preserved=True, payload_used=False,
                source_authenticated=False, exact_seam_certified=False,
                predicate_evaluations=PATH_CLASS_COUNT if detected else 0)
