"""Small CPU fixtures for the adopted paired-energy carrier arithmetic."""
from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier


pytestmark = pytest.mark.unit


SMALL = carrier.CarrierProtocol(
    video_shape=(17, 24, 24, 3),
    segment_start=1,
    segment_frames=8,
    segment_count=2,
    rois=((0, 8, 0, 8), (0, 8, 16, 24), (16, 24, 0, 8), (16, 24, 16, 24)),
)


def _small_rgb() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261009)
    return 0.45 + 0.10 * torch.rand(SMALL.video_shape, generator=generator)


def _spec(phase: int, slot: int, roi: int, *, received: int = 17):
    return next(
        row for row in carrier.phase_window_catalog(SMALL, received_frame_count=received)
        if (row.phase, row.slot, row.roi_index) == (phase, slot, roi)
    )


def test_key_pair_rm_and_fragment_derivation_are_exact_and_ordered():
    key = "e\u0301/密钥"
    expected = hashlib.sha256(json.dumps(
        ["LJSP1/PAIR", key, 2, 7], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).digest()
    assert carrier.public_hash("LJSP1/PAIR", key, 2, 7) == expected
    assert carrier.public_hash("domain", "é", 0) != carrier.public_hash("domain", "e\u0301", 0)

    coords = carrier.dct_coordinates()
    assert len(coords) == 32 and (0, 0) not in coords
    assert coords == tuple(sorted(coords, key=lambda value: (sum(value), *value)))
    pairs = carrier.coefficient_pairs(key, 2)
    assert len(pairs) == 16
    assert len({coord for pair in pairs for coord in pair}) == 32

    state = carrier.state_matrix(key)
    assert len(state) == 22 and all(len(row) == 32 for row in state)
    assert len(set(state)) == 22
    assert all(row.count(1) == row.count(-1) == 16 for row in state)

    fragments = carrier.message_fragments(bytes((0x80, 0x01, 0xA5, 0x5A)))
    assert fragments[0] == (1, 0, 0, 0, 0, 0, 0, 0)
    assert fragments[1] == (0, 0, 0, 0, 0, 0, 0, 1)
    assert fragments[2] == (1, 0, 1, 0, 0, 1, 0, 1)
    assert carrier.message_fragment_collision_classes(bytes((0xAA, 0xAA, 0x55, 0xAA))) == (
        (0, 1, 3),
    )
    _, payload0 = carrier.segment_targets(key, bytes((0x80, 0x01, 0xA5, 0x5A)), 0)
    _, payload1 = carrier.segment_targets(key, bytes((0x80, 0x01, 0xA5, 0x5A)), 1)
    assert payload0 == (1, -1, -1, -1, -1, -1, -1, -1)
    assert payload1 == (-1, -1, -1, -1, -1, -1, -1, 1)
    with pytest.raises(ValueError, match="exactly four bytes"):
        carrier.message_fragments(b"abc")


def test_fp64_pair_redistribution_preserves_energy_ratio_sign_and_zero_support():
    key = "pair-fixture"
    coefficients = torch.zeros((1, 8, 8), dtype=torch.float64)
    pairs = carrier.coefficient_pairs(key, 0)
    for index, (plus, minus) in enumerate(pairs):
        coefficients[0, plus[0], plus[1]] = index + 1.0
        coefficients[0, minus[0], minus[1]] = -(index + 2.0)
    coefficients[0, pairs[1][0][0], pairs[1][0][1]] = 0.0
    coefficients[0, pairs[2][0][0], pairs[2][0][1]] = 0.0
    coefficients[0, pairs[2][1][0], pairs[2][1][1]] = 0.0
    targets = tuple(1 if index % 2 == 0 else -1 for index in range(16))
    changed, rows = carrier.redistribute_coefficients(
        coefficients, key=key, frame=3, roi_index=0,
        targets=targets, rho=0.4, tiles_x=1,
    )
    for index, (plus, minus) in enumerate(pairs):
        before = coefficients[0, plus[0], plus[1]].square() + coefficients[0, minus[0], minus[1]].square()
        after_plus = changed[0, plus[0], plus[1]]
        after_minus = changed[0, minus[0], minus[1]]
        after = after_plus.square() + after_minus.square()
        torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)
        assert rows[index]["energy_after"] == pytest.approx(rows[index]["energy_before"], abs=1e-12)
        if float(before) == 0:
            assert after_plus == 0 and after_minus == 0
            assert rows[index]["zero_support_tiles"] == 1
        else:
            q = float((after_plus.square() - after_minus.square()) / after)
            assert q == pytest.approx(0.4 * targets[index], abs=1e-12)
    assert torch.sign(changed[0, pairs[0][0][0], pairs[0][0][1]]) == 1
    assert torch.sign(changed[0, pairs[0][1][0], pairs[0][1][1]]) == -1
    expected_zero_sign = 2 * (
        carrier.public_hash("LJSP1/ZERO_SIGN", key, 3, 0, 0, 0, 1, 0)[0] & 1
    ) - 1
    assert int(torch.sign(changed[0, pairs[1][0][0], pairs[1][0][1]])) == expected_zero_sign

    tiles = torch.randn((3, 8, 8), dtype=torch.float64)
    torch.testing.assert_close(
        carrier.idct2_ortho(carrier.dct2_ortho(tiles)), tiles,
        rtol=1e-12, atol=1e-12,
    )

    rho_zero_input = torch.zeros((1, 8, 8), dtype=torch.float64)
    plus, minus = pairs[0]
    rho_zero_input[0, plus[0], plus[1]] = 3.0
    rho_zero_input[0, minus[0], minus[1]] = 4.0
    rho_zero, _ = carrier.redistribute_coefficients(
        rho_zero_input, key=key, frame=3, roi_index=0,
        targets=(1,) * 16, rho=0.0, tiles_x=1,
    )
    assert not torch.equal(rho_zero, rho_zero_input)
    assert rho_zero[0, plus[0], plus[1]].square() == pytest.approx(12.5)
    assert rho_zero[0, minus[0], minus[1]].square() == pytest.approx(12.5)


def test_small_rgb_writer_is_local_and_reader_retains_raw_energy_q_and_support():
    key = "writer-reader-fixture"
    message = bytes((0x80, 0x01, 0xA5, 0x5A))
    original = _small_rgb()
    written, receipt = carrier.apply_carrier_rgb(
        original, key=key, message=message, rho=0.35, protocol=SMALL
    )
    assert receipt["rho"] == 0.35 and receipt["adaptive_retry"] is False
    assert receipt["outside_declared_roi_written_values"] == 0
    assert receipt["clipped_low_values"] == receipt["clipped_high_values"] == 0
    assert receipt["pair_energy_constraint_stage"] == "pre_clipping_fp64_coefficients"
    assert receipt["clip_adjustment_l2"] == 0
    assert receipt["fp32_rounding_l2"] >= 0
    recorded_errors = [row["post_clip_abs_ratio_error_max"] for row in receipt["pair_totals"]]
    assert all(math.isfinite(value) and value >= 0 for value in recorded_errors)
    assert max(recorded_errors) > 0
    torch.testing.assert_close(written[0], original[0], rtol=0, atol=0)
    support = torch.zeros(SMALL.video_shape[:-1], dtype=torch.bool)
    for frame in range(1, 17):
        for y0, y1, x0, x1 in SMALL.rois:
            support[frame, y0:y1, x0:x1] = True
    difference = written - original.float()
    assert int(torch.count_nonzero(difference[~support])) == 0

    row = carrier.observe_window(written, key, _spec(1, 0, 0), protocol=SMALL)
    assert row.state_status == row.payload_status == "SCORED"
    assert [chip.component_index for chip in row.state_chips] == list(range(8))
    assert [chip.component_index for chip in row.payload_chips] == list(range(8))
    state, payload = carrier.segment_targets(key, message, 0, segment_count=2)
    for chip, target in zip(row.state_chips, state[:8]):
        assert chip.q == pytest.approx(0.35 * target, abs=2e-6)
        assert chip.expected_samples == chip.available_samples == 8
        assert chip.failed_samples == 0 and chip.energy_plus is not None
    for chip, target in zip(row.payload_chips, payload):
        assert chip.q == pytest.approx(0.35 * target, abs=2e-6)
    json.dumps(row.to_dict(), allow_nan=False)

    # Independently re-extract all final-FP32 writer windows and reconcile the
    # serialized post-clip E+/E-/pooled-q receipt without using writer targets.
    independent = [dict(plus=0.0, minus=0.0) for _ in range(16)]
    for segment in range(2):
        for roi in range(4):
            observed = carrier.observe_window(
                written, key, _spec(1, segment, roi), protocol=SMALL
            )
            for chip in observed.state_chips + observed.payload_chips:
                independent[chip.pair_index]["plus"] += chip.energy_plus
                independent[chip.pair_index]["minus"] += chip.energy_minus
    for index, values in enumerate(independent):
        recorded = receipt["pair_totals"][index]
        assert recorded["post_clip_energy_plus"] == pytest.approx(values["plus"], rel=1e-12)
        assert recorded["post_clip_energy_minus"] == pytest.approx(values["minus"], rel=1e-12)
        pooled_q = (values["plus"] - values["minus"]) / (values["plus"] + values["minus"])
        recorded_q = (
            (recorded["post_clip_energy_plus"] - recorded["post_clip_energy_minus"])
            / (recorded["post_clip_energy_plus"] + recorded["post_clip_energy_minus"])
        )
        assert recorded_q == pytest.approx(pooled_q, rel=1e-12)


def test_fixed_phase_catalog_keeps_signed_boundaries_short_input_and_partial_chips():
    catalog = carrier.phase_window_catalog()
    assert len(catalog) == 8 * 24 * 4 == 768
    first = next(row for row in catalog if (row.phase, row.slot, row.roi_index) == (0, -1, 0))
    tail = next(row for row in catalog if (row.phase, row.slot, row.roi_index) == (0, 22, 0))
    reference = next(row for row in catalog if (row.phase, row.slot, row.roi_index) == (1, 0, 0))
    assert first.requested_frames == tuple(range(-8, 0)) and first.availability == "MISSING"
    assert tail.requested_frames == tuple(range(176, 184))
    assert tail.received_frames == tuple(range(176, 181)) and tail.availability == "PARTIAL"
    assert reference.requested_frames == tuple(range(1, 9))
    assert not hasattr(reference, "writer_nominal_segment")

    short = _small_rgb()[:5]
    partial_spec = _spec(1, 0, 0, received=5)
    assert partial_spec.received_frames == (1, 2, 3, 4) and partial_spec.availability == "PARTIAL"
    partial = carrier.observe_window(short, "short-key", partial_spec, protocol=SMALL)
    assert partial.state_status == partial.payload_status == "PARTIAL"
    assert all(chip.status == "SCORED" for chip in partial.state_chips)
    assert all((chip.expected_samples, chip.available_samples) == (8, 4) for chip in partial.state_chips)

    missing = carrier.observe_window(
        short, "short-key", _spec(0, -1, 0, received=5), protocol=SMALL
    )
    assert missing.state_status == missing.payload_status == "MISSING"
    assert all(chip.q is None and chip.available_samples == 0 for chip in missing.state_chips)

    damaged = short.clone()
    damaged[1, 0, 0, 0] = float("nan")
    damaged_row = carrier.observe_window(damaged, "short-key", partial_spec, protocol=SMALL)
    assert damaged_row.state_status == damaged_row.payload_status == "PARTIAL"
    assert all(chip.status == "PARTIAL" and chip.failed_samples == 1 for chip in damaged_row.state_chips)
    assert all(chip.q is not None for chip in damaged_row.state_chips)
    json.dumps(damaged_row.to_dict(), allow_nan=False)

    overlong = torch.zeros((18, 24, 24, 3), dtype=torch.float32)
    failed = carrier.observe_catalog(overlong, "short-key", protocol=SMALL)
    assert len(failed) == 8 * 4 * 4
    assert all(row.state_status == row.payload_status == "FAILED" for row in failed)


def test_fragment_routes_preserve_raw_entries_collisions_erasures_and_declared_equivalence():
    key = "route-key"
    rgb = _small_rgb()
    first = carrier.observe_window(rgb, key, _spec(1, 0, 0), protocol=SMALL)
    second = carrier.observe_window(rgb, key, _spec(1, 0, 1), protocol=SMALL)
    routes = (
        carrier.SourceSlotCorrespondence("m0", "candidate-a", first.spec.observation_id, 5, 1, "eq-a"),
        carrier.SourceSlotCorrespondence("m1", "candidate-a", first.spec.observation_id, 6, 2, "eq-a"),
        carrier.SourceSlotCorrespondence("m2", "candidate-a", second.spec.observation_id, 9, 1, "eq-a"),
        carrier.SourceSlotCorrespondence("m3", "candidate-b", second.spec.observation_id, 0, 3, "eq-b"),
    )
    result = carrier.route_fragment_evidence((first, second), routes)
    first_entry = result["candidates"]["candidate-a"][1][0]
    assert first_entry["received_slot"] == 0 and first_entry["source_slot"] == 5
    assert first_entry["mod4_equivalent"] is True
    assert first_entry["aggregation"] == "none"
    assert len(first_entry["payload_chips"]) == 8
    assert result["erasures"]["candidate-a"] == [0, 3]
    assert result["conflicts"] == [{
        "candidate_id": "candidate-a", "observation_id": first.spec.observation_id,
        "fragment_slots": [1, 2],
    }]
    assert result["reused_observations"][0]["route_count"] == 2
    assert result["many_to_one_routes"][0]["source_slots"] == [5, 9]
    assert result["caller_declared_equivalence_classes"]["eq-a"] == ["m0", "m1", "m2"]
    assert result["equivalence_verified"] is False
    assert result["decoded_message"] is None and result["accepted"] is None
    assert result["truth_used"] is False
    json.dumps(result, allow_nan=False)
