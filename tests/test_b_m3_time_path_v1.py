"""Small synthetic engineering checks. No saved science, model or media calls."""
import ast
import copy
import inspect
import json
from unittest.mock import patch

import numpy as np
import pytest

from main.tube_state import b_m3_time_path_v1 as m
from runtime.wan import b_m3_observation_v1 as obs

pytestmark = pytest.mark.unit


def raw(value=0.2, peaks=()):
    selected = set(peaks)
    return dict(status="SCORES_COMPLETE", candidates=[dict(phase=p, source_start=s,
        S=0.8 if m.state_to_origin(p, s) in selected else value, score_status="RETURNED")
        for p, s in m.H])


def local(window, origins=(), value=0.2, tau=0.5):
    family = raw(value, origins)
    family.update(input_id="opaque-input", key_id="key00")
    return m.prepare_local(family, dict(status="FROZEN", tau=tau),
                           "opaque-query-" + window, m.WINDOWS[window])


def triple(a=17, d=8, gap="LEFT_GAP"):
    origins = {"L": a, "M": a + 53 + (d if gap == "LEFT_GAP" else 0), "R": a + 106 + d}
    return {w: local(w, [o]) for w, o in origins.items()}


def nulls():
    return [{"family_id": f"n{i:02}", "family": raw()} for i in range(90)]


def test_h137_origin_bijection_and_exact_boundaries():
    assert len(m.H) == 137
    assert {m.state_to_origin(*h) for h in m.H} == set(range(137))
    assert [m.state_to_origin(*m.origin_to_state(o)) for o in range(137)] == list(range(137))
    assert m.origin_to_state(0) == (0, 1)
    assert m.origin_to_state(136) == (0, 35)
    assert m.origin_to_state(1) == (3, 2)


@pytest.mark.parametrize("value", [-1, 137, 0.5, True, None])
def test_negative_outside_noninteger_origin_rejected(value):
    with pytest.raises(ValueError):
        m.origin_to_state(value)


def test_local_coordinates_add_w_exactly_once_and_source_origin_does_not_move():
    p, s = m.origin_to_state(78)
    row = m.cell_mapping(53, p, s, 1)
    assert row["query_local_interval"] == [3, 6]
    assert row["nominal_received_cell"] == [56, 59]
    assert row["source_frames_under_h"] == [81, 82, 83, 84]
    assert row["canonical_cell"] == 21
    assert not row["source_support_under_h"]["vae_causal_support"]
    for w in m.WINDOWS.values():
        for p, s in ((0, 1), (3, 2), (0, 35), (1, 35)):
            for r in (1, 10):
                x = m.cell_mapping(w, p, s, r)
                t = s + r - 1
                assert x["source_frames_under_h"] == list(range(4*t-3, 4*t+1))
    for args in ((-1, 0, 1, 1), (0, 1, 1, 1), (53, 0, 36, 1), (0, 0, 1, 0)):
        with pytest.raises(ValueError):
            m.cell_mapping(*args)


def test_fixed961_domain_and_seam_mapping_at_safe_boundaries():
    domain = m.path_domain()
    assert len(domain) == len({r["path_id"] for r in domain}) == 961
    assert sum(r["seam_multiplicity"] for r in domain) == 8401
    assert sum(r["d"] == 0 for r in domain) == 31
    for row in domain:
        assert row["a"] + 150 + row["d"] <= 180
        seams = row["seam_alias_interval"] or [None]
        for b in seams:
            for w, start in m.WINDOWS.items():
                assert m.map_received_frame(row["a"], row["d"], b, start) == row["query_source_origins"][w]
                # Each declared45-frame query remains contiguous, including edge seams.
                assert m.map_received_frame(row["a"], row["d"], b, start + 44) == row["query_source_origins"][w] + 44
    assert m.map_received_frame(17, 8, 49, 48) == 65
    assert m.map_received_frame(17, 8, 49, 49) == 74
    with pytest.raises(ValueError):
        m.map_received_frame(30, 1, 49, 0)
    with pytest.raises(ValueError):
        m.map_received_frame(0, 0, 49, 0)
    with pytest.raises(ValueError):
        m.map_received_frame(0, 1, 54, 0)


@pytest.mark.parametrize("gap", ["LEFT_GAP", "RIGHT_GAP"])
@pytest.mark.parametrize("a,d", [(17, 8), (0, 1), (0, 30), (29, 1)])
def test_new_middle_observation_resolves_only_gap_class_18_to9(a, d, gap):
    families = triple(a, d, gap)
    before = copy.deepcopy(families)
    e2, m3 = m.paths(families, "E2"), m.paths(families, "M3")
    assert e2["compatible_class_count"] == 2
    assert e2["compatible_seam_hypothesis_count"] == 18
    assert m3["compatible_class_count"] == 1
    assert m3["compatible_seam_hypothesis_count"] == 9
    assert m3["possible_gap"] == [gap]
    assert m3["possible_a"] == [a] and m3["possible_d"] == [d]
    assert set(m3["compatible_path_ids"]) <= set(e2["compatible_path_ids"])
    assert len(m3["paths"]) == len(e2["paths"]) == 961
    assert not m3["exact_seam_certified"] and not m3["source_authenticated"]
    assert families == before


def test_continuity_retained_without_fake_seam_and_no_successful_path_fallback():
    families = triple(17, 0)
    result = m.paths(families, "M3")
    assert result["compatible_class_count"] == result["compatible_seam_hypothesis_count"] == 1
    assert result["location_status"] == "CONTINUITY_NOT_EXCLUDED"
    assert next(r for r in result["paths"] if r["compatibility"])["seam_alias_interval"] is None
    families["M"] = local("M", [0])  # Strong but incompatible, not weak/missing.
    assert m.paths(families, "M3")["status"] == "PATH_INCOMPATIBLE"
    assert m.paths(families, "E2")["compatible_class_count"] == 1
    assert len(families["M"]["C"]) == 1


def test_strict_detection_inclusive_C_and_all_local_candidates_retained():
    boundary = local("M", value=0.5)
    assert boundary["D"] is False and len(boundary["C"]) == 137
    result = m.paths({w: local(w, value=0.5) for w in m.WINDOWS}, "M3")
    assert result["status"] == "PATH_LOCAL_GATE_REJECT"
    assert result["compatible_class_count"] == 0 and len(result["paths"]) == 961
    broad = {w: local(w, [0], value=0.5) for w in m.WINDOWS}
    result = m.paths(broad, "M3")
    assert result["compatible_class_count"] == 961
    assert all(len(f["C"]) == 137 for f in broad.values())


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "nan", "inf", "bool", "partial"])
def test_bad_local_family_keeps137_planned_slots_and_C_is_unknown(fault):
    family = raw()
    if fault == "missing":
        family["candidates"].pop()
    elif fault == "duplicate":
        family["candidates"][-1] = dict(family["candidates"][0])
    elif fault == "extra":
        family["candidates"].append(dict(family["candidates"][0]))
    elif fault == "partial":
        family["candidates"][10]["score_status"] = "PARTIAL"
    else:
        family["candidates"][10]["S"] = {"nan": float("nan"), "inf": float("inf"), "bool": True}[fault]
    prepared = m.prepare_local(family, dict(status="FROZEN", tau=0.5), "opaque", 53)
    assert len(prepared["members"]) == prepared["domain_count"] == 137
    assert prepared["C"] is None and prepared["D"] is None and prepared["maximum"] is None
    assert all(r["in_C"] is None for r in prepared["members"])
    assert prepared["cause_set"]
    json.dumps(prepared, allow_nan=False)


def test_E2_does_not_access_M_and_M3_failure_is_not_empty_candidate_set():
    class EndpointsOnly(dict):
        def get(self, key, default=None):
            if key == "M":
                raise AssertionError("E2 attempted to consume M")
            return super().get(key, default)
    full = triple()
    ends = EndpointsOnly(L=full["L"], R=full["R"])
    assert m.paths(ends, "E2")["compatible_class_count"] == 2
    failed = m.paths(dict(ends), "M3")
    assert failed["compatible_path_ids"] is None
    assert failed["compatible_class_count"] is None
    assert all(row["compatibility"] is None for row in failed["paths"])
    assert failed["planned"] == failed["failed"] == 961


@pytest.mark.parametrize("upstream", ["PARTIAL", "UNSUPPORTED", "NOT_ENTERED"])
def test_upstream_failure_label_remains_visible(upstream):
    family = raw()
    family["candidates"][7]["score_status"] = upstream
    family["status"] = upstream
    family["error"] = "synthetic upstream failure"
    prepared = m.prepare_local(family, dict(status="FROZEN", tau=0.5), "q", 0)
    assert prepared["upstream_status"] == upstream
    assert prepared["upstream_error"] == "synthetic upstream failure"
    assert prepared["members"][7]["upstream_score_status"] == upstream
    assert "UPSTREAM:" + upstream in prepared["members"][7]["cause_set"]
    assert prepared["C"] is None


@pytest.mark.parametrize("fault", ["denominator", "C", "origin", "coordinate", "D", "key"])
def test_semantically_broken_local_record_cannot_select_paths(fault):
    families = triple()
    broken = families["M"]
    if fault == "denominator":
        broken["domain_count"] = 136
    elif fault == "C":
        broken["C"] = []
    elif fault == "origin":
        broken["members"][0]["O_h"] = -1
    elif fault == "coordinate":
        broken["received_query_start"] = 0
    elif fault == "D":
        broken["D"] = False
    else:
        broken["key_id"] = "key01"
    assert m.paths(families, "M3")["compatible_class_count"] is None
    assert m.paths(families, "E2")["compatible_class_count"] == 2


def test90_null_max_calibration_is_path_free_and_uses_all_members():
    families = nulls()
    families[-1]["family"]["candidates"][-1]["S"] = 0.6
    with patch.object(m, "path_domain", side_effect=AssertionError("calibration path computation")):
        threshold = m.calibrate(families)
    assert threshold["status"] == "FROZEN" and threshold["tau"] == 0.6
    assert threshold["path_calculations"] == 0 and threshold["planned_score_slots"] == 12330
    assert len(threshold["maxima"]) == 90
    assert m.calibrate({r["family_id"]: r["family"] for r in families})["tau"] == 0.6


@pytest.mark.parametrize("fault", ["89", "91", "duplicate_id", "missing_score", "bad_family"])
def test_incomplete_calibration_never_shrinks_denominator_or_inherits_tau(fault):
    families = nulls()
    if fault == "89":
        families.pop()
    elif fault == "91":
        families.append(dict(family_id="extra", family=raw(0.9)))
    elif fault == "duplicate_id":
        families[1]["family_id"] = families[0]["family_id"]
    elif fault == "bad_family":
        families[8]["family"] = None
    else:
        families[8]["family"]["candidates"][42]["S"] = None
    threshold = m.calibrate(families)
    assert threshold["status"] == "CALIBRATION_UNRESOLVED" and threshold["tau"] is None
    assert len(threshold["maxima"]) == threshold["planned_families"] == 90
    assert threshold["path_calculations"] == 0
    unavailable = {w: m.prepare_local(raw(), threshold, "q"+w, start) for w, start in m.WINDOWS.items()}
    assert m.paths(unavailable, "M3")["status"] == "CALIBRATION_UNRESOLVED"


def test_score_wrapper_uses_Q40_and_no_payload_or_old_threshold():
    background = dict(status="BACKGROUND_FROZEN", mean=[0.0]*40, factor=np.eye(40).tolist())
    banks = {str(p): dict(roi_q=np.ones((10, 4, 44))) for p in range(4)}
    result = m.score_family(banks, background)
    assert result["status"] == "SCORES_COMPLETE" and len(result["candidates"]) == 137
    assert all(abs(r["S"] - 1) < 1e-12 for r in result["candidates"])
    assert "phase_local_bits" not in result and "tau" not in result
    banks["1"]["roi_q"][2, 1, 5] = np.nan
    result = m.score_family(banks, background)
    assert result["status"] == "SCORE_UNRESOLVED" and len(result["candidates"]) == 137
    assert m.score_family({}, background)["cause_set"] == ["INCOMPLETE_OR_DUPLICATE_PHASES"]


def test_payload_poison_ignored_and_pure_import_boundaries():
    class Poison:
        def __iter__(self):
            raise AssertionError("payload read")
        def __getitem__(self, key):
            raise AssertionError("payload read")
    family = raw()
    family["phase_local_bits"] = Poison()
    family["writer_bits"] = Poison()
    out = m.prepare_local(family, dict(status="FROZEN", tau=0.5), "q", 0)
    assert "phase_local_bits" not in out and "writer_bits" not in out
    tree = ast.parse(inspect.getsource(m))
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("runtime", "experiments", "governance")) for n in ast.walk(tree))
    assert list(inspect.signature(m.paths).parameters) == ["locals_by_window", "mode"]
    assert list(inspect.signature(m.prepare_local).parameters) == ["family", "threshold", "query_id", "received_query_start"]


def test_view_schedule_is12_physical_views_and_uint8_extraction_only():
    plan = obs.view_plan("opaque")
    assert len(plan) == len({r["view_id"] for r in plan}) == 12
    assert all("key_id" not in r and r["status"] == "PLANNED" for r in plan)
    assert plan[-1]["encoder_input_support"]["received_interval_inclusive"] == [109, 149]
    # Broadcast avoids allocating a full151-frame fixture; no VAE/model involved.
    received = np.broadcast_to(np.arange(151, dtype=np.uint8)[:, None, None, None], (151, 320, 512, 3))
    selected = obs.extract_e41(received, 106, 3)
    assert selected.shape == (41, 320, 512, 3) and selected.dtype == np.uint8
    assert selected[:, 0, 0, 0].tolist() == list(range(109, 150))
    assert not np.shares_memory(received, selected)
    with pytest.raises(ValueError):
        obs.extract_e41(received[:121], 0, 0)
    with pytest.raises(ValueError):
        obs.extract_e41(received, 38, 0)
    with pytest.raises(ValueError):
        obs.extract_e41(received, 0, 4)
