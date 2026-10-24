"""CPU small constructions and fake dispatch; no real model/media or saved replay."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect
import json

import pytest
import torch

from main.tube_state import b_local_identity_dct_v1 as m
from runtime.wan import b_local_identity_dct_v1 as runtime

pytestmark = pytest.mark.unit
BITS = [int(b) for byte in bytes.fromhex("a6d39c5e") for b in f"{byte:08b}"]
KEY = "watermark"


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def no_legacy_fft(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("DCT2 must not execute the old FFT payload")
    monkeypatch.setattr(torch.fft, "fft2", forbidden)
    monkeypatch.setattr(torch.fft, "ifft2", forbidden)
    monkeypatch.setattr(runtime.native_payload.payload, "local_delta", forbidden)


def synthesize(target):
    z = torch.zeros(m.LATENT_SHAPE)
    directions = m.basis()
    for rho, region in enumerate(m.PAYLOAD_REGIONS):
        y0, y1, x0, x1 = m.REGIONS[region]
        z[0, :4, :, y0:y1, x0:x1] = torch.einsum("ctm,mij->ctij", target[rho], directions)
    return z


def test_unit_basis_and_key_layout_are_orthogonal_not_statistical_claims():
    d = m.basis(dtype=torch.float64).flatten(1)
    torch.testing.assert_close(d @ d.T, torch.eye(16, dtype=torch.float64), atol=3e-15, rtol=0)
    assert m.coordinates(KEY, "R0") == m.coordinates(KEY, "R0")
    for region in m.PAYLOAD_REGIONS:
        assert set(m.coordinates(KEY, region)) == set(m.MODES)
    assert m.coordinates(KEY, "R0") != m.coordinates(KEY, "R1")
    # A caller cannot corrupt the reusable canonical basis.
    changed = m.basis(dtype=torch.float64)
    changed.zero_()
    assert bool((m.basis(dtype=torch.float64) != 0).any())


def test_writer_target_guards_groups_and_native_round_trip_without_group_assignment():
    target = m.build_target(torch.zeros(m.LATENT_SHAPE), KEY, BITS)
    assert target.shape == (2, 4, 46, 16) and target.numel() == 5888
    assert torch.equal(target[:, :, 0], target[:, :, 1])
    assert torch.equal(target[:, :, 45], target[:, :, 44])
    assert not torch.equal(target[:, :, 11], target[:, :, 12])
    z = synthesize(target)
    projected = m.project(z, ("R0", "R1"), range(1, 45))
    for rho, region in enumerate(m.PAYLOAD_REGIONS):
        # FP32 synthesis + 256-term projection; this tolerance is engineering
        # roundoff, not a reader threshold or a change to beta/eta.
        torch.testing.assert_close(projected["coefficients"][region], target[rho, :, 1:45],
                                   atol=8*torch.finfo(torch.float32).eps, rtol=0)
    cache = m.decode(projected, KEY, view_id="native-view", key_id="key00")
    assert cache["planned_bits"] == cache["returned_bits"] == 704
    assert cache["group_assigned"] is False and not cache["guard_read"]
    for region in m.PAYLOAD_REGIONS:
        for cell in cache["regions"][region]["local_cells"]:
            t = cell["local_row"]
            assert [b["value"] for b in cell["bits"]] == BITS[8*((t-1)//11):8*((t-1)//11)+8]
            assert "group" not in cell and "source_time" not in cell
            assert all(b["count"] == 8 and b["returned_count"] == 8 for b in cell["bits"])
    with pytest.raises(ValueError, match="guard"):
        m.project(z, m.PAYLOAD_REGIONS, range(46))


def test_local_gradient_has_one_eighth_unit_scale_exact_zero_outside_and_fresh_leaf():
    target = m.build_target(torch.zeros(m.LATENT_SHAPE), KEY, BITS)
    clean = synthesize(target)
    direction = m.basis()[3]
    clean[0, 2, 12, 8:24, 8:24] += 2 * direction
    clean.requires_grad_(True)
    delta, receipt = m.local_delta(clean, target)
    expected = torch.zeros_like(clean)
    expected[0, 2, 12, 8:24, 8:24] = -0.25 * direction
    torch.testing.assert_close(delta, expected, atol=1e-7, rtol=0)
    mask = torch.zeros_like(delta, dtype=torch.bool)
    for region in m.PAYLOAD_REGIONS:
        y0, y1, x0, x1 = m.REGIONS[region]
        mask[0, :4, :, y0:y1, x0:x1] = True
    assert torch.count_nonzero(delta[~mask]) == 0
    assert clean.grad is None and not delta.requires_grad
    assert (receipt["target_count"], receipt["eta"], receipt["local_backwards"]) == (5888, 368, 1)
    assert receipt["effective_target_count"] + receipt["guard_target_count"] == 5888
    assert receipt["unit_direction_update"] == 0.125 and receipt["writer_fft_calls"] == 0
    assert len(receipt["regions"]) == 2


def test_e41_regions_stay_separate_strict_zero_and_counter_first_tie():
    a = torch.full((4, 10, 16), -1.0)
    a[:2] = 1
    b = -a
    mode = m.MODES.index(m.coordinates(KEY, "R0")[0])
    a[2, 0, mode] = 0
    packet = dict(status="PROJECTED", cause_set=[], kind="E41", local_rows=list(range(1, 11)),
                  include_control=False, coefficients={"R0": a, "R1": b})
    cache = m.decode(packet, KEY, view_id="actual-E")
    first = cache["regions"]["R0"]["local_cells"][0]["bits"][0]
    assert (first["value"], first["tie"], first["ones"], first["zero_coefficients"]) == (1, True, 4, 1)
    second = cache["regions"]["R1"]["local_cells"][0]["bits"][0]
    assert second["value"] == 0 and second["tie"]
    zero = m.read_coefficients(torch.zeros(4, 10, 16), KEY, "R0", range(1, 11))
    bit = zero["local_cells"][0]["bits"][0]
    assert (bit["value"], bit["ones"], bit["zero_coefficients"], bit["tie"]) == (0, 0, 8, False)
    assert bit["evidence_id"] is None  # thin API leaves physical IDs to runtime
    assert [(r["channel"], r["keyed_index"]) for r in first["coefficient_refs"]] == [
        (c, j) for c in range(4) for j in (0, 8)]


def test_one_projection_shared_keys_and_control_templates_have_same_physical_id(monkeypatch):
    calls = []
    original = m._project_patch
    def counted(*args):
        calls.append(args[1])
        return original(*args)
    monkeypatch.setattr(m, "_project_patch", counted)
    packet = m.project(torch.zeros(m.LOCAL_SHAPE), iter(("R0", "R1", "Rc")), range(1, 11))
    right = m.decode(packet, KEY, view_id="view", key_id="key00")
    wrong = m.decode(packet, "watermark-wrong", view_id="view", key_id="key01")
    assert calls == ["R0", "R1", "Rc"]
    assert packet["coefficient_slots"] == 1920
    assert right["planned_bits"] == right["diagnostic_planned_bits"] == 160
    for region in m.PAYLOAD_REGIONS:
        a = right["regions"][region]["local_cells"][0]
        b = wrong["regions"][region]["local_cells"][0]
        assert a["evidence_id"] == b["evidence_id"]
    controls = right["control_diagnostics"]
    assert controls["R0"]["local_cells"][0]["evidence_id"] == controls["R1"]["local_cells"][0]["evidence_id"]
    assert all(p["diagnostic_only"] for p in controls.values())
    assert right["required_replica_ids"] == ["R0", "R1"]
    raw = m.read_coefficients(packet["coefficients"]["Rc"], KEY, "R1", range(1, 11), observed_region_id="Rc")
    assert raw["diagnostic_only"] and raw["observed_region_id"] == "Rc"
    with pytest.raises(ValueError, match="renamed"):
        m.read_coefficients(packet["coefficients"]["R1"], KEY, "R0", range(1, 11), observed_region_id="R1")


def test_nonfinite_row_and_missing_region_retain_fixed_slots_and_json_nulls():
    packet = m.project(torch.zeros(m.LOCAL_SHAPE))
    packet["coefficients"]["R0"][0, 2, 0] = float("nan")
    cache = m.decode(packet, KEY, view_id="failed-row")
    assert cache["planned_bits"] == 160 and cache["returned_bits"] == 152
    bad = cache["regions"]["R0"]["local_cells"][2]
    assert len(bad["bits"]) == 8 and all(b["value"] is None for b in bad["bits"])
    assert "NONFINITE_COEFFICIENT_ROW" in bad["cause_set"]
    assert cache["regions"]["R1"]["returned_bits"] == 80
    json.dumps(cache, allow_nan=False)
    packet["coefficients"]["R1"] = None
    cache = m.decode(packet, KEY, view_id="missing-region")
    assert cache["planned_bits"] == 160 and cache["returned_bits"] == 72
    assert all(b["returned_count"] == 0 and b["count"] == 8
               for row in cache["regions"]["R1"]["local_cells"] for b in row["bits"])


@pytest.mark.parametrize("status", ["FAILED", "NOT_ENTERED", "UNSUPPORTED", "INTERRUPTED", "UNRESOLVED"])
def test_upstream_unavailable_keeps_effective_rows_and_status(status):
    packet = m.unavailable_projection(native=True, status=status, cause="TEST_UPSTREAM")
    cache = m.decode(packet, KEY, view_id="native")
    assert cache["planned_bits"] == 704 and cache["returned_bits"] == 0
    for region in cache["regions"].values():
        assert len(region["local_cells"]) == 44
        assert all(b["status"] == status and b["value"] is None
                   for row in region["local_cells"] for b in row["bits"])
    json.dumps(cache, allow_nan=False)


def test_projection_and_writer_validate_actual_shape_dtype_finite():
    with pytest.raises(ValueError, match="shape"):
        m.project(torch.zeros(1, 16, 10, 40, 64))
    with pytest.raises(ValueError, match="FP32"):
        m.project(torch.zeros(m.LOCAL_SHAPE, dtype=torch.float64))
    bad = torch.zeros(m.LOCAL_SHAPE)
    bad[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        m.project(bad)
    with pytest.raises(ValueError, match="32 binary"):
        m.build_target(torch.zeros(m.LATENT_SHAPE), KEY, [False] * 32)


def test_cfg_clean_seam_inactive_and_active_with_real_small_tensor_gradient():
    z = torch.zeros(m.LATENT_SHAPE)
    target = m.build_target(z, KEY, BITS)
    c, u = torch.full_like(z, 0.01), torch.full_like(z, -0.02)
    events = []
    inactive, receipt = runtime.guided_velocity(z, c, u, 0.5, target, 24,
                                                lambda *event: events.append(event))
    assert torch.equal(inactive, u + 5 * (c-u)) and not receipt["enabled"] and events == []
    delta, _ = m.local_delta(z - 0.5 * c, target)
    active, receipt = runtime.guided_velocity(z, c, u, 0.5, target, 25,
                                             lambda *event: events.append(event))
    torch.testing.assert_close((z-0.5*active)-(z-0.5*inactive), 5*delta, atol=4e-8, rtol=1e-5)
    assert events == [("payload_gradient", False), ("payload_gradient", True)]
    assert receipt["payload"]["target_count"] == 5888


class FakeScheduler:
    def __init__(self):
        self.step_index = None
        self.predict_x0 = True
        self.timesteps = torch.arange(50)
        self.sigmas = torch.cat([torch.ones(50), torch.zeros(1)])
        self.config = SimpleNamespace(prediction_type="flow_prediction", thresholding=False,
                                      predict_x0=True, lower_order_final=True)
        self.calls = []
    def step(self, velocity, timestep, z, return_dict=False):
        self.calls.append(int(timestep))
        self.step_index = len(self.calls)
        return (z,)


def test_native50_dispatch_counts_and_gradient_failure_has_no_retry(monkeypatch):
    initial = torch.zeros(m.LATENT_SHAPE)
    zero = torch.zeros_like(initial)
    calls = []
    def transformer(**kwargs):
        calls.append(int(kwargs["timestep"][0]))
        return (zero,)
    pipe = SimpleNamespace(transformer=transformer)
    local_calls = []
    def local(clean, target):
        local_calls.append(tuple(target.shape))
        return zero, dict(conditional_clean_delta_rms=0.0, target_count=5888)
    monkeypatch.setattr(m, "local_delta", local)
    scheduler = FakeScheduler()
    traces, events = [], []
    terminal, receipt = runtime.run_trajectory(pipe, initial, scheduler, None, None,
        torch.float32, KEY, BITS, lambda *e: events.append(e), traces.append)
    assert len(calls) == 100 and scheduler.calls == list(range(50))
    assert len(local_calls) == 25 and len(traces) == 50
    assert [t["index"] for t in traces if t["enabled"]] == list(range(25, 50))
    assert Counter(events)[("payload_gradient", False)] == 25
    assert Counter(events)[("payload_gradient", True)] == 25
    assert Counter(events)[("native_step", True)] == 50
    assert terminal.dtype == torch.float32 and terminal.device.type == "cpu"
    assert receipt["writer_fft_calls"] == 0 and not receipt["state_writer_called"]
    def fail(*args):
        raise RuntimeError("synthetic local failure")
    monkeypatch.setattr(m, "local_delta", fail)
    scheduler, events, traces = FakeScheduler(), [], []
    with pytest.raises(RuntimeError, match="synthetic local"):
        runtime.run_trajectory(pipe, initial, scheduler, None, None, torch.float32,
                               KEY, BITS, lambda *e: events.append(e), traces.append)
    assert scheduler.step_index == 25 and len(traces) == 25
    assert Counter(events)[("payload_gradient", False)] == 1
    assert Counter(events)[("payload_gradient", True)] == 0


def test_reader_has_no_truth_or_clock_arguments_and_core_has_no_outer_imports():
    for function in (m.project, m.read_coefficients, m.decode):
        params = set(inspect.signature(function).parameters)
        assert not params & {"bits", "source_id", "origin", "source_clock", "path", "truth"}
    source = Path(inspect.getfile(m)).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("runtime", "experiments", "governance"))
        elif isinstance(node, ast.Import):
            assert all(not n.name.startswith(("runtime", "experiments", "governance")) for n in node.names)
