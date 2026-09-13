"""Run a CPU-only AISB synthetic acquisition probe."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.sc_sstw.aisb import (
    BurstCandidate,
    best_non_overlapping_sequence,
    make_default_templates,
    make_redundant_templates,
    make_double_redundant_templates,
    scan_burst_candidates,
    template_observation_pairs,
)
from main.sc_sstw.calibration import calibrate_from_pilot_pairs, equalize_observations
from main.sc_sstw.sync import dynamic_time_sync
from experiments.feasibility.synthetic.channel import (
    SyntheticChannel,
    generate_observations,
    make_random_channel,
)


def _unit_state(key: str, index: int) -> tuple[float, float]:
    digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).digest()
    angle_seed = int.from_bytes(digest[:8], "big") / float(1 << 64)
    step_seed = 1 if digest[8] & 1 else -1
    angle = 2.0 * math.pi * angle_seed + 0.41 * step_seed
    return (math.cos(angle), math.sin(angle))


def _template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_default_templates()}


def _build_sequence(case_index: int) -> tuple[list[tuple[float, float]], list[tuple[int, str]]]:
    templates = _template_by_id()
    template_ids = ["burst_alpha", "burst_beta", "burst_gamma", "burst_alpha", "burst_gamma"]
    states: list[tuple[float, float]] = []
    truth: list[tuple[int, str]] = []
    rng = random.Random(3000 + case_index)
    for template_id in template_ids:
        for _ in range(2 + rng.randrange(3)):
            angle = rng.random() * 2.0 * math.pi
            radius = 0.6 * rng.random()
            states.append((radius * math.cos(angle), radius * math.sin(angle)))
        start = len(states)
        states.extend(templates[template_id].points)
        truth.append((start, template_id))
    for _ in range(4):
        angle = rng.random() * 2.0 * math.pi
        radius = 0.4 * rng.random()
        states.append((radius * math.cos(angle), radius * math.sin(angle)))
    return states, truth


def _edit_observations[T](
    values: list[T],
    truth: list[tuple[int, str]],
    *,
    case_index: int,
    delete_inside_bursts: bool,
) -> tuple[list[T], list[int], list[tuple[int, str, int | None]]]:
    crop_start = 1 + (case_index % 2)
    crop_end = len(values) - (case_index % 3)
    protected = set()
    missing_by_start: dict[int, int] = {}
    for burst_ordinal, (start, _) in enumerate(truth):
        missing_index = 3 + ((case_index + burst_ordinal) % 3) if delete_inside_bursts else None
        if missing_index is not None:
            missing_by_start[start] = missing_index
        for offset in range(6):
            if missing_index is None or offset != missing_index:
                protected.add(start + offset)
    edited: list[T] = []
    source_to_observed: dict[int, int] = {}
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        missing_index = next(
            (
                missing
                for burst_start, missing in missing_by_start.items()
                if source_index == burst_start + missing
            ),
            None,
        )
        if missing_index is not None:
            continue
        if source_index not in protected and (source_index + case_index) % 11 == 0:
            continue
        source_to_observed[source_index] = len(edited)
        edited.append(value)
    edited_truth: list[tuple[int, str, int | None]] = []
    for start, template_id in truth:
        missing_index = missing_by_start.get(start)
        present_offsets = [offset for offset in range(6) if offset != missing_index]
        if all((start + offset) in source_to_observed for offset in present_offsets):
            edited_truth.append((source_to_observed[start + present_offsets[0]], template_id, missing_index))
    source_indices = [
        source_index
        for source_index in range(crop_start, crop_end)
        if source_index in source_to_observed
    ]
    return edited, source_indices, edited_truth


def _candidate_key(candidate: BurstCandidate) -> tuple[int, str, int | tuple[int, ...] | None]:
    return (candidate.start_index, candidate.template_id, candidate.missing_template_index)


def _calibrate_from_aisb_candidates(observations: list[list[float]], accepted: list[BurstCandidate]):
    return _calibrate_from_candidates(observations, accepted, _template_by_id())


def _case(case_index: int, *, delete_inside_bursts: bool) -> dict[str, object]:
    templates = make_default_templates()
    states, truth = _build_sequence(case_index)
    channel: SyntheticChannel = make_random_channel(4000 + case_index, relation_count=12, noise_std=0.015)
    observations = generate_observations(states, channel, seed=5000 + case_index)
    edited, source_indices, edited_truth = _edit_observations(
        observations,
        truth,
        case_index=case_index,
        delete_inside_bursts=delete_inside_bursts,
    )
    edited_states = [states[index] for index in source_indices]
    candidates = scan_burst_candidates(edited, templates, allow_single_deletion=delete_inside_bursts)
    accepted = best_non_overlapping_sequence(
        candidates,
        burst_length=templates[0].length,
        residual_threshold=0.006,
    )
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    alignment_accuracy = true_positive / max(1, len(truth_set))
    calibration = _calibrate_from_aisb_candidates(edited, accepted)
    equalized = equalize_observations(edited, calibration)
    state_reconstruction_mse = sum(
        (estimated[0] - expected[0]) ** 2 + (estimated[1] - expected[1]) ** 2
        for estimated, expected in zip(equalized, edited_states, strict=True)
    ) / len(edited_states)
    return {
        "case_index": case_index,
        "delete_inside_bursts": delete_inside_bursts,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "alignment_accuracy": alignment_accuracy,
        "calibration_condition_number": calibration.condition_number,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "state_reconstruction_mse": state_reconstruction_mse,
        "best_residual": min((candidate.residual for candidate in candidates), default=None),
        "pass": (
            alignment_accuracy >= 0.9
            and false_positive == 0
            and false_negative == 0
            and state_reconstruction_mse < 0.01
        ),
    }


def _false_case(case_index: int, *, allow_single_deletion: bool) -> dict[str, object]:
    templates = make_default_templates()
    rng = random.Random(9000 + case_index)
    states = [
        (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        for _ in range(44)
    ]
    channel = make_random_channel(9100 + case_index, relation_count=12, noise_std=0.015)
    observations = generate_observations(states, channel, seed=9200 + case_index)
    candidates = scan_burst_candidates(observations, templates, allow_single_deletion=allow_single_deletion)
    accepted = best_non_overlapping_sequence(
        candidates,
        burst_length=templates[0].length,
        residual_threshold=0.006,
    )
    return {
        "case_index": case_index,
        "allow_single_deletion": allow_single_deletion,
        "accepted_count": len(accepted),
        "best_residual": min((candidate.residual for candidate in candidates), default=None),
        "pass": len(accepted) == 0,
    }


def _mixed_states(key: str, length: int, burst_plan: list[tuple[int, str]]) -> list[tuple[float, float]]:
    templates = _template_by_id()
    states = [_unit_state(key, index) for index in range(length)]
    for start, template_id in burst_plan:
        for offset, point in enumerate(templates[template_id].points):
            states[start + offset] = point
    return states


def _mixed_sequence_case(case_index: int) -> dict[str, object]:
    owner_key = f"owner_mixed_{case_index}"
    wrong_keys = [f"wrong_mixed_{case_index}_{index}" for index in range(10)]
    burst_plan = [(8, "burst_alpha"), (27, "burst_beta"), (49, "burst_gamma"), (70, "burst_alpha")]
    states = _mixed_states(owner_key, 88, burst_plan)
    channel = make_random_channel(12000 + case_index, relation_count=14, noise_std=0.018)
    observations = generate_observations(states, channel, seed=12100 + case_index)
    edited, source_indices, edited_truth = _edit_observations(
        observations,
        burst_plan,
        case_index=case_index,
        delete_inside_bursts=True,
    )
    candidates = scan_burst_candidates(edited, make_default_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=6, residual_threshold=0.006)
    calibration = _calibrate_from_aisb_candidates(edited, accepted)
    equalized = equalize_observations(edited, calibration)
    owner_candidate = _mixed_states(owner_key, len(states), burst_plan)
    owner_sync = dynamic_time_sync(equalized, owner_candidate)
    wrong_scores = [
        dynamic_time_sync(equalized, _mixed_states(wrong_key, len(states), burst_plan)).score
        for wrong_key in wrong_keys
    ]
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "owner_score": owner_sync.score,
        "best_wrong_score": max(wrong_scores),
        "score_margin": owner_sync.score - max(wrong_scores),
        "calibration_condition_number": calibration.condition_number,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "pass": (
            false_positive == 0
            and false_negative == 0
            and owner_sync.score > max(wrong_scores)
            and owner_sync.score - max(wrong_scores) > 0.02
            and calibration.pilot_reconstruction_mse < 0.01
        ),
    }


def _redundant_template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_redundant_templates()}


def _double_redundant_template_by_id() -> dict[str, object]:
    return {template.template_id: template for template in make_double_redundant_templates()}


def _build_double_redundant_sequence(case_index: int) -> tuple[list[tuple[float, float]], list[tuple[int, str]]]:
    templates = _double_redundant_template_by_id()
    template_ids = ["double_redundant_alpha", "double_redundant_beta", "double_redundant_gamma", "double_redundant_alpha"]
    states: list[tuple[float, float]] = []
    truth: list[tuple[int, str]] = []
    rng = random.Random(34000 + case_index)
    for template_id in template_ids:
        for _ in range(2 + rng.randrange(3)):
            angle = rng.random() * 2.0 * math.pi
            radius = 0.5 * rng.random()
            states.append((radius * math.cos(angle), radius * math.sin(angle)))
        start = len(states)
        states.extend(templates[template_id].points)
        truth.append((start, template_id))
    for _ in range(5):
        angle = rng.random() * 2.0 * math.pi
        radius = 0.45 * rng.random()
        states.append((radius * math.cos(angle), radius * math.sin(angle)))
    return states, truth


def _build_redundant_sequence(case_index: int) -> tuple[list[tuple[float, float]], list[tuple[int, str]]]:
    templates = _redundant_template_by_id()
    template_ids = ["redundant_alpha", "redundant_beta", "redundant_gamma", "redundant_alpha"]
    states: list[tuple[float, float]] = []
    truth: list[tuple[int, str]] = []
    rng = random.Random(7000 + case_index)
    for template_id in template_ids:
        for _ in range(2 + rng.randrange(3)):
            angle = rng.random() * 2.0 * math.pi
            radius = 0.5 * rng.random()
            states.append((radius * math.cos(angle), radius * math.sin(angle)))
        start = len(states)
        states.extend(templates[template_id].points)
        truth.append((start, template_id))
    for _ in range(5):
        angle = rng.random() * 2.0 * math.pi
        radius = 0.45 * rng.random()
        states.append((radius * math.cos(angle), radius * math.sin(angle)))
    return states, truth


def _edit_redundant_observations[T](
    values: list[T],
    truth: list[tuple[int, str]],
    *,
    case_index: int,
    missing_template_index: int,
) -> tuple[list[T], list[int], list[tuple[int, str, int]]]:
    crop_start = case_index % 2
    crop_end = len(values) - (case_index % 3)
    protected = set()
    for start, _ in truth:
        for offset in range(9):
            if offset != missing_template_index:
                protected.add(start + offset)
    edited: list[T] = []
    source_to_observed: dict[int, int] = {}
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        if any(source_index == start + missing_template_index for start, _ in truth):
            continue
        if source_index not in protected and (source_index + case_index) % 13 == 0:
            continue
        source_to_observed[source_index] = len(edited)
        edited.append(value)
    edited_truth: list[tuple[int, str, int]] = []
    for start, template_id in truth:
        present_offsets = [offset for offset in range(9) if offset != missing_template_index]
        if all((start + offset) in source_to_observed for offset in present_offsets):
            edited_truth.append((source_to_observed[start + present_offsets[0]], template_id, missing_template_index))
    source_indices = [
        source_index
        for source_index in range(crop_start, crop_end)
        if source_index in source_to_observed
    ]
    return edited, source_indices, edited_truth


def _edit_redundant_observations_double[T](
    values: list[T],
    truth: list[tuple[int, str]],
    *,
    case_index: int,
    missing_template_indices: tuple[int, int],
    template_length: int,
) -> tuple[list[T], list[int], list[tuple[int, str, tuple[int, int]]]]:
    crop_start = case_index % 2
    crop_end = len(values) - (case_index % 3)
    missing_set = set(missing_template_indices)
    protected = set()
    for start, _ in truth:
        for offset in range(template_length):
            if offset not in missing_set:
                protected.add(start + offset)
    edited: list[T] = []
    source_to_observed: dict[int, int] = {}
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        if any(source_index == start + missing for start, _ in truth for missing in missing_template_indices):
            continue
        if source_index not in protected and (source_index + case_index) % 13 == 0:
            continue
        source_to_observed[source_index] = len(edited)
        edited.append(value)
    edited_truth: list[tuple[int, str, tuple[int, int]]] = []
    for start, template_id in truth:
        present_offsets = [offset for offset in range(template_length) if offset not in missing_set]
        if all((start + offset) in source_to_observed for offset in present_offsets):
            edited_truth.append((source_to_observed[start + present_offsets[0]], template_id, missing_template_indices))
    source_indices = [
        source_index
        for source_index in range(crop_start, crop_end)
        if source_index in source_to_observed
    ]
    return edited, source_indices, edited_truth


def _redundant_any_deletion_case(case_index: int) -> dict[str, object]:
    missing_template_index = case_index % 9
    templates = make_redundant_templates()
    states, truth = _build_redundant_sequence(case_index)
    channel = make_random_channel(8000 + case_index, relation_count=12, noise_std=0.012)
    observations = generate_observations(states, channel, seed=8100 + case_index)
    edited, source_indices, edited_truth = _edit_redundant_observations(
        observations,
        truth,
        case_index=case_index,
        missing_template_index=missing_template_index,
    )
    edited_states = [states[index] for index in source_indices]
    candidates = scan_burst_candidates(edited, templates, allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=templates[0].length, residual_threshold=0.006)
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    calibration = _calibrate_from_candidates(edited, accepted, _redundant_template_by_id())
    equalized = equalize_observations(edited, calibration)
    state_reconstruction_mse = sum(
        (estimated[0] - expected[0]) ** 2 + (estimated[1] - expected[1]) ** 2
        for estimated, expected in zip(equalized, edited_states, strict=True)
    ) / len(edited_states)
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "missing_template_index": missing_template_index,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "calibration_condition_number": calibration.condition_number,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "state_reconstruction_mse": state_reconstruction_mse,
        "best_residual": min((candidate.residual for candidate in candidates), default=None),
        "pass": (
            false_positive == 0
            and false_negative == 0
            and state_reconstruction_mse < 0.01
        ),
    }


def _calibrate_from_candidates(
    observations: list[list[float]],
    accepted: list[BurstCandidate],
    template_by_id: dict[str, object],
):
    pilot_pairs = []
    for candidate in accepted:
        template = template_by_id[candidate.template_id]
        pilot_pairs.extend(template_observation_pairs(candidate, observations, template))
    return calibrate_from_pilot_pairs(pilot_pairs)


def _redundant_two_deletion_case(case_index: int) -> dict[str, object]:
    first_missing = case_index % 12
    second_missing = (first_missing + 4 + case_index // 3) % 12
    if second_missing == first_missing:
        second_missing = (second_missing + 1) % 12
    missing_template_indices = tuple(sorted((first_missing, second_missing)))
    templates = make_double_redundant_templates()
    states, truth = _build_double_redundant_sequence(case_index)
    channel = make_random_channel(31000 + case_index, relation_count=12, noise_std=0.012)
    observations = generate_observations(states, channel, seed=32000 + case_index)
    edited, source_indices, edited_truth = _edit_redundant_observations_double(
        observations,
        truth,
        case_index=case_index,
        missing_template_indices=missing_template_indices,
        template_length=templates[0].length,
    )
    edited_states = [states[index] for index in source_indices]
    candidates = scan_burst_candidates(edited, templates, allow_double_deletion=True, top_k_per_start=2)
    accepted = best_non_overlapping_sequence(candidates, burst_length=templates[0].length, residual_threshold=0.006, maximize_count=True)
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    calibration = _calibrate_from_candidates(edited, accepted, _double_redundant_template_by_id())
    equalized = equalize_observations(edited, calibration)
    state_reconstruction_mse = sum(
        (estimated[0] - expected[0]) ** 2 + (estimated[1] - expected[1]) ** 2
        for estimated, expected in zip(equalized, edited_states, strict=True)
    ) / len(edited_states)
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "missing_template_indices": missing_template_indices,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "calibration_condition_number": calibration.condition_number,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "state_reconstruction_mse": state_reconstruction_mse,
        "best_residual": min((candidate.residual for candidate in candidates), default=None),
        "pass": (
            false_positive == 0
            and false_negative == 0
            and state_reconstruction_mse < 0.02
        ),
    }


def _redundant_mixed_states(key: str, length: int, burst_plan: list[tuple[int, str]]) -> list[tuple[float, float]]:
    templates = _redundant_template_by_id()
    states = [_unit_state(key, index) for index in range(length)]
    for start, template_id in burst_plan:
        for offset, point in enumerate(templates[template_id].points):
            states[start + offset] = point
    return states


def _double_redundant_mixed_states(key: str, length: int, burst_plan: list[tuple[int, str]]) -> list[tuple[float, float]]:
    templates = _double_redundant_template_by_id()
    states = [_unit_state(key, index) for index in range(length)]
    for start, template_id in burst_plan:
        for offset, point in enumerate(templates[template_id].points):
            states[start + offset] = point
    return states


def _redundant_mixed_sequence_two_deletion_case(case_index: int) -> dict[str, object]:
    owner_key = f"owner_redundant_mixed_double_{case_index}"
    wrong_keys = [f"wrong_redundant_mixed_double_{case_index}_{index}" for index in range(10)]
    first_missing = case_index % 12
    second_missing = (first_missing + 5 + case_index // 3) % 12
    if second_missing == first_missing:
        second_missing = (second_missing + 1) % 12
    missing_template_indices = tuple(sorted((first_missing, second_missing)))
    burst_plan = [(9, "double_redundant_alpha"), (34, "double_redundant_beta"), (61, "double_redundant_gamma"), (88, "double_redundant_alpha")]
    states = _double_redundant_mixed_states(owner_key, 115, burst_plan)
    channel = make_random_channel(33000 + case_index, relation_count=14, noise_std=0.014)
    observations = generate_observations(states, channel, seed=33100 + case_index)
    edited, _, edited_truth = _edit_redundant_observations_double(
        observations,
        burst_plan,
        case_index=case_index,
        missing_template_indices=missing_template_indices,
        template_length=12,
    )
    candidates = scan_burst_candidates(edited, make_double_redundant_templates(), allow_double_deletion=True, top_k_per_start=2)
    accepted = best_non_overlapping_sequence(candidates, burst_length=12, residual_threshold=0.006, maximize_count=True)
    calibration = _calibrate_from_candidates(edited, accepted, _double_redundant_template_by_id())
    equalized = equalize_observations(edited, calibration)
    owner_sync = dynamic_time_sync(equalized, _double_redundant_mixed_states(owner_key, len(states), burst_plan))
    wrong_scores = [
        dynamic_time_sync(equalized, _double_redundant_mixed_states(wrong_key, len(states), burst_plan)).score
        for wrong_key in wrong_keys
    ]
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "missing_template_indices": missing_template_indices,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "owner_score": owner_sync.score,
        "best_wrong_score": max(wrong_scores),
        "score_margin": owner_sync.score - max(wrong_scores),
        "calibration_condition_number": calibration.condition_number,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "pass": (
            false_positive == 0
            and false_negative == 0
            and owner_sync.score > max(wrong_scores)
            and owner_sync.score - max(wrong_scores) > 0.02
            and calibration.pilot_reconstruction_mse < 0.02
        ),
    }


def _redundant_mixed_sequence_case(case_index: int) -> dict[str, object]:
    owner_key = f"owner_redundant_mixed_{case_index}"
    wrong_keys = [f"wrong_redundant_mixed_{case_index}_{index}" for index in range(10)]
    missing_template_index = case_index % 9
    burst_plan = [(9, "redundant_alpha"), (31, "redundant_beta"), (55, "redundant_gamma"), (78, "redundant_alpha")]
    states = _redundant_mixed_states(owner_key, 100, burst_plan)
    channel = make_random_channel(18000 + case_index, relation_count=14, noise_std=0.014)
    observations = generate_observations(states, channel, seed=18100 + case_index)
    edited, source_indices, edited_truth = _edit_redundant_observations(
        observations,
        burst_plan,
        case_index=case_index,
        missing_template_index=missing_template_index,
    )
    candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
    accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
    calibration = _calibrate_from_candidates(edited, accepted, _redundant_template_by_id())
    equalized = equalize_observations(edited, calibration)
    owner_sync = dynamic_time_sync(equalized, _redundant_mixed_states(owner_key, len(states), burst_plan))
    wrong_scores = [
        dynamic_time_sync(equalized, _redundant_mixed_states(wrong_key, len(states), burst_plan)).score
        for wrong_key in wrong_keys
    ]
    truth_set = set(edited_truth)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    true_positive = len(truth_set & accepted_set)
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "missing_template_index": missing_template_index,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_accuracy": true_positive / max(1, len(truth_set)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "owner_score": owner_sync.score,
        "best_wrong_score": max(wrong_scores),
        "score_margin": owner_sync.score - max(wrong_scores),
        "calibration_condition_number": calibration.condition_number,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "pass": (
            false_positive == 0
            and false_negative == 0
            and owner_sync.score > max(wrong_scores)
            and owner_sync.score - max(wrong_scores) > 0.02
            and calibration.pilot_reconstruction_mse < 0.01
        ),
    }


def _threshold_diagnostic(case_count: int = 128) -> dict[str, object]:
    best_residuals: list[float] = []
    accepted_total = 0
    for case_index in range(case_count):
        rng = random.Random(15000 + case_index)
        states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(72)]
        channel = make_random_channel(15100 + case_index, relation_count=12, noise_std=0.012)
        observations = generate_observations(states, channel, seed=15200 + case_index)
        candidates = scan_burst_candidates(observations, make_redundant_templates(), allow_single_deletion=True)
        best_residuals.append(min(candidate.residual for candidate in candidates))
        accepted_total += len(best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006))
    ordered = sorted(best_residuals)
    def quantile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]
    return {
        "case_count": case_count,
        "diagnostic_threshold": 0.006,
        "accepted_count_total": accepted_total,
        "best_residual_min": ordered[0],
        "best_residual_p05": quantile(0.05),
        "best_residual_median": quantile(0.50),
        "best_residual_p95": quantile(0.95),
        "best_residual_max": ordered[-1],
        "diagnostic_pass": accepted_total == 0 and ordered[0] > 0.006,
        "fixed_fpr_claim": False,
    }


def _redundant_truth_and_candidates(
    case_index: int,
    *,
    noise_std: float,
    seed_offset: int,
) -> tuple[set[tuple[int, str, int | None]], list[BurstCandidate]]:
    missing_template_index = case_index % 9
    states, truth = _build_redundant_sequence(seed_offset + case_index)
    channel = make_random_channel(24000 + seed_offset + case_index, relation_count=12, noise_std=noise_std)
    observations = generate_observations(states, channel, seed=25000 + seed_offset + case_index)
    edited, _, edited_truth = _edit_redundant_observations(
        observations,
        truth,
        case_index=seed_offset + case_index,
        missing_template_index=missing_template_index,
    )
    return set(edited_truth), scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)


def _redundant_positive_residuals(
    *,
    case_count: int,
    noise_std: float,
    seed_offset: int,
) -> list[float]:
    residuals: list[float] = []
    for case_index in range(case_count):
        truth_set, candidates = _redundant_truth_and_candidates(case_index, noise_std=noise_std, seed_offset=seed_offset)
        candidate_by_key = {_candidate_key(candidate): candidate for candidate in candidates}
        residuals.extend(candidate_by_key[key].residual for key in truth_set if key in candidate_by_key)
    return residuals


def _redundant_negative_residuals(
    *,
    case_count: int,
    noise_std: float,
    seed_offset: int,
) -> list[float]:
    residuals: list[float] = []
    for case_index in range(case_count):
        rng = random.Random(26000 + seed_offset + case_index)
        states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(72)]
        channel = make_random_channel(27000 + seed_offset + case_index, relation_count=12, noise_std=noise_std)
        observations = generate_observations(states, channel, seed=28000 + seed_offset + case_index)
        candidates = scan_burst_candidates(observations, make_redundant_templates(), allow_single_deletion=True)
        residuals.append(min(candidate.residual for candidate in candidates))
    return residuals


def _redundant_threshold_development_diagnostic() -> dict[str, object]:
    noise_std = 0.03
    dev_positive = _redundant_positive_residuals(case_count=18, noise_std=noise_std, seed_offset=0)
    dev_negative = _redundant_negative_residuals(case_count=128, noise_std=noise_std, seed_offset=0)
    dev_positive_max = max(dev_positive)
    dev_negative_min = min(dev_negative)
    if dev_positive_max >= dev_negative_min:
        threshold = dev_positive_max
        separable = False
    else:
        threshold = (dev_positive_max + dev_negative_min) / 2.0
        separable = True
    test_positive_pass = 0
    test_positive_total = 0
    for case_index in range(18):
        truth_set, candidates = _redundant_truth_and_candidates(case_index, noise_std=noise_std, seed_offset=1000)
        accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=threshold)
        accepted_set = {_candidate_key(candidate) for candidate in accepted}
        test_positive_pass += int(accepted_set == truth_set)
        test_positive_total += 1
    test_negative = _redundant_negative_residuals(case_count=128, noise_std=noise_std, seed_offset=1000)
    test_negative_accepted = sum(1 for residual in test_negative if residual <= threshold)
    return {
        "diagnostic_kind": "development_test_threshold_margin_not_fixed_fpr",
        "noise_std": noise_std,
        "development_positive_count": len(dev_positive),
        "development_negative_count": len(dev_negative),
        "development_positive_max": dev_positive_max,
        "development_negative_min": dev_negative_min,
        "selected_threshold": threshold,
        "development_separable": separable,
        "test_positive_case_count": test_positive_total,
        "test_positive_pass_count": test_positive_pass,
        "test_negative_case_count": len(test_negative),
        "test_negative_accepted_count": test_negative_accepted,
        "diagnostic_pass": separable and test_positive_pass == test_positive_total and test_negative_accepted == 0,
        "fixed_fpr_claim": False,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summarize_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_accuracy_mean": _mean([float(case["alignment_accuracy"]) for case in cases]),
        "false_positive": sum(int(case["false_positive"]) for case in cases),
        "false_negative": sum(int(case["false_negative"]) for case in cases),
        "state_reconstruction_mse_mean": _mean([
            float(case.get("state_reconstruction_mse", 0.0)) for case in cases
        ]),
        "cases": cases,
    }


def main() -> None:
    complete_cases = [_case(index, delete_inside_bursts=False) for index in range(8)]
    deletion_cases = [_case(index, delete_inside_bursts=True) for index in range(8)]
    false_cases = [_false_case(index, allow_single_deletion=True) for index in range(8)]
    mixed_cases = [_mixed_sequence_case(index) for index in range(6)]
    redundant_any_deletion_cases = [_redundant_any_deletion_case(index) for index in range(18)]
    redundant_two_deletion_cases = [_redundant_two_deletion_case(index) for index in range(18)]
    redundant_mixed_cases = [_redundant_mixed_sequence_case(index) for index in range(9)]
    redundant_mixed_two_deletion_cases = [_redundant_mixed_sequence_two_deletion_case(index) for index in range(9)]
    threshold_diagnostic = _threshold_diagnostic()
    threshold_development_diagnostic = _redundant_threshold_development_diagnostic()
    report = {
        "status": "aisb_synthetic_only_no_video_no_gpu_no_claim",
        "complete_burst_cases": _summarize_cases(complete_cases),
        "single_checksum_deletion_burst_cases": _summarize_cases(deletion_cases),
        "random_non_burst_cases": {
            "case_count": len(false_cases),
            "false_pass_count": sum(1 for case in false_cases if case["pass"]),
            "accepted_count_total": sum(int(case["accepted_count"]) for case in false_cases),
            "cases": false_cases,
        },
        "redundant_any_single_deletion_cases": _summarize_cases(redundant_any_deletion_cases),
        "redundant_any_double_deletion_cases": _summarize_cases(redundant_two_deletion_cases),
        "redundant_random_non_burst_threshold_diagnostic": threshold_diagnostic,
        "redundant_threshold_development_diagnostic": threshold_development_diagnostic,
        "redundant_mixed_sequence_owner_wrong_key_cases": {
            "case_count": len(redundant_mixed_cases),
            "pass_count": sum(1 for case in redundant_mixed_cases if case["pass"]),
            "score_margin_mean": _mean([float(case["score_margin"]) for case in redundant_mixed_cases]),
            "alignment_accuracy_mean": _mean([float(case["alignment_accuracy"]) for case in redundant_mixed_cases]),
            "false_positive": sum(int(case["false_positive"]) for case in redundant_mixed_cases),
            "false_negative": sum(int(case["false_negative"]) for case in redundant_mixed_cases),
            "cases": redundant_mixed_cases,
        },
        "redundant_mixed_sequence_double_deletion_owner_wrong_key_cases": {
            "case_count": len(redundant_mixed_two_deletion_cases),
            "pass_count": sum(1 for case in redundant_mixed_two_deletion_cases if case["pass"]),
            "score_margin_mean": _mean([float(case["score_margin"]) for case in redundant_mixed_two_deletion_cases]),
            "alignment_accuracy_mean": _mean([float(case["alignment_accuracy"]) for case in redundant_mixed_two_deletion_cases]),
            "false_positive": sum(int(case["false_positive"]) for case in redundant_mixed_two_deletion_cases),
            "false_negative": sum(int(case["false_negative"]) for case in redundant_mixed_two_deletion_cases),
            "cases": redundant_mixed_two_deletion_cases,
        },
        "mixed_sequence_owner_wrong_key_cases": {
            "case_count": len(mixed_cases),
            "pass_count": sum(1 for case in mixed_cases if case["pass"]),
            "score_margin_mean": _mean([float(case["score_margin"]) for case in mixed_cases]),
            "alignment_accuracy_mean": _mean([float(case["alignment_accuracy"]) for case in mixed_cases]),
            "false_positive": sum(int(case["false_positive"]) for case in mixed_cases),
            "false_negative": sum(int(case["false_negative"]) for case in mixed_cases),
            "cases": mixed_cases,
        },
        "synthetic_construction_pass": (
            all(case["pass"] for case in complete_cases)
            and all(case["pass"] for case in deletion_cases)
            and all(case["pass"] for case in false_cases)
            and all(case["pass"] for case in mixed_cases)
            and all(case["pass"] for case in redundant_any_deletion_cases)
            and all(case["pass"] for case in redundant_two_deletion_cases)
            and all(case["pass"] for case in redundant_mixed_cases)
            and all(case["pass"] for case in redundant_mixed_two_deletion_cases)
            and threshold_diagnostic["diagnostic_pass"]
            and threshold_development_diagnostic["diagnostic_pass"]
        ),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
