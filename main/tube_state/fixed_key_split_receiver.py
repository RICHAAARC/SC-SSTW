"""Frozen A-select/B-confirm fixed-key receivers; no runtime, writer, or truth inputs."""
from __future__ import annotations

import math

import numpy as np

from . import fixed_key, projection_margin as carrier, state_clock

A = np.array([i for i in range(160) if (i // 16 + (i % 16) // 2) % 2 == 0])
B = np.array([i for i in range(160) if (i // 16 + (i % 16) // 2) % 2 == 1])
CANDIDATES = ("C1_MATCHED_CONFIRM", "C2_STATE_CONFIRM")
PROTOCOL_ID = "SC-SSTW-Receiver-Split-V1"


class PartitionEvidence:
    """A view of only one fixed spatial partition; no cross-partition cache."""

    def __init__(self, observations, book, indices, name):
        self.book, self.indices, self.name = book, np.asarray(indices), name
        self.counts = {g: z.shape[2] - 1 for g, z in observations.items()}
        self.parts = {}
        for g, z in observations.items():
            groups = z[0, :, 1:].transpose(1, 0, 2, 3)
            data = groups.reshape(len(groups), 16, 10, 4, 16, 4)
            data = data.transpose(0, 2, 4, 1, 3, 5).reshape(len(groups), 160, 256)
            self.parts[g] = data[:, self.indices].astype(np.float64)
        self.directions = book["directions"].reshape(11, 160, 4, 256)[:, self.indices].astype(np.float64)
        self.cache = {}
        self.allocations = {}
        self.path_evaluations = 0

    def emission(self, n, g, num, den, offset):
        key = (n, g, num, den, offset)
        if key in self.cache:
            return self.cache[key]
        allocation_key = (g, num, den, offset)
        if allocation_key not in self.allocations:
            self.allocations[allocation_key] = carrier.allocation(self.counts[g], g, num, den, offset)
        selected = self.allocations[allocation_key][4 * n:4 * n + 4]
        groups = sum(j is not None for j in selected)
        row = dict(window=n, phase=g, selected=selected, group_count=groups,
                   valid=groups >= 3, observed_components=80 * groups if groups >= 3 else 0)
        if groups < 3:
            row.update(support_kind="INVALID", signed_evidence=None, q=[0.0, 0.0], q_norm=0.0)
        else:
            projection = np.zeros(80, dtype=np.float64)
            for slot, j in enumerate(selected):
                if j is not None:
                    projection += np.einsum(
                        "ij,ij->i", self.parts[g][j], self.directions[n, :, slot],
                    )
            clipped = np.clip(projection, -1.0, 1.0)
            ids = 160 * n + self.indices
            decoded = clipped * self.book["sync"][ids] * self.book["polarity"][ids]
            q = [float(decoded[self.indices % 2 == axis].mean()) for axis in (0, 1)]
            row.update(
                support_kind="FULL" if groups == 4 else "PARTIAL3",
                signed_evidence=float(np.mean(clipped * self.book["code"][ids])),
                q=q,
                q_norm=float(np.linalg.norm(q)),
                projection_rms=float(np.sqrt(np.mean(projection ** 2))),
                clipped_fraction=float(np.mean(np.abs(projection) >= 1)),
            )
        self.cache[key] = row
        return row

    def score_path(self, path):
        self.path_evaluations += 1
        windows = []
        for n in range(11):
            delta = path["delta"] if n >= path["boundary"] else 0
            windows.append(self.emission(
                n, (path["g"] - delta) % 4, *path["scale"], path["offset"] + delta,
            ))
        valid = [window["valid"] for window in windows]
        weights = [window["observed_components"] for window in windows if window["valid"]]
        if not weights:
            return dict(status="INVALID", path=dict(path), windows=windows, score=None)
        matched = float(np.average(
            [window["signed_evidence"] for window in windows if window["valid"]], weights=weights,
        ))
        observer = state_clock.observe(
            np.asarray([window["q"] for window in windows]), valid,
            self.book["states"], self.book["steps"],
        )
        event = state_clock.EDIT_COST if path["delta"] else 0.0
        return dict(
            status="SCORED", path=dict(path), score=matched - 0.05 * observer["innovation_mean"] - event,
            matched_score=matched, state_innovation_mean=observer["innovation_mean"],
            innovation_by_window=observer["innovation_by_window"], event_cost=event,
            windows=windows, matched_windows=[i for i, value in enumerate(valid) if value],
            matched_components=sum(weights),
            full_blocks=80 * sum(window["group_count"] == 4 for window in windows),
            partial_blocks=80 * sum(window["group_count"] == 3 for window in windows),
            partition=self.name,
        )


def select_path(locator):
    """Select from locator A only; no B evidence or confirmation callback is accepted."""
    assert locator.name == "A"
    best, attempted, scored = None, 0, 0
    for path in state_clock.clock_paths():
        attempted += 1
        row = locator.score_path(path)
        if row["status"] == "SCORED" and math.isfinite(row["score"]):
            scored += 1
            if best is None or fixed_key._rank(row) < fixed_key._rank(best):
                best = row
    return dict(best=best, attempted_path_count=attempted, scored_path_count=scored)


def confirmation_scores(row):
    if row["status"] != "SCORED":
        return {candidate: None for candidate in CANDIDATES}
    return {
        "C1_MATCHED_CONFIRM": row["matched_score"] - row["event_cost"],
        "C2_STATE_CONFIRM": row["matched_score"] - 0.05 * row["state_innovation_mean"] - row["event_cost"],
    }


def read(observations, key, spec_sha):
    if (
        set(observations) != set(range(4))
        or any(
            value.ndim != 5
            or value.shape[:2] != (1, 16)
            or value.shape[3:] != (40, 64)
            or not np.isfinite(value).all()
            for value in observations.values()
        )
    ):
        return dict(status="INVALID", reason="four finite valid phase tensors required")
    book = fixed_key.codebook(key)
    locator = PartitionEvidence(observations, book, A, "A")
    selection = select_path(locator)
    if selection["best"] is None:
        return dict(status="INVALID", reason="no usable locator path", selection=selection)
    selected_path = dict(selection["best"]["path"])
    confirmation = PartitionEvidence(observations, book, B, "B")
    heldout = confirmation.score_path(selected_path)
    values = confirmation_scores(heldout)
    diagnostics = {}
    for name, path in fixed_key.REFERENCE_PATHS.items():
        locator_row, confirmation_row = locator.score_path(path), confirmation.score_path(path)
        diagnostics[name] = dict(
            locator=locator_row,
            confirmation=confirmation_row,
            scores=confirmation_scores(confirmation_row),
        )
    assert selection["attempted_path_count"] == 4284
    assert confirmation.path_evaluations == 4
    alignment_input = dict(
        best=selection["best"],
        fixed_path_diagnostics={name: value["locator"] for name, value in diagnostics.items()},
    )
    return dict(
        status="SCORED", spec_sha256=spec_sha, key_id=book["key_id"],
        receiver_protocol_id=PROTOCOL_ID, selection=selection,
        confirmation=heldout, confirmation_scores=values,
        fixed_path_diagnostics=diagnostics,
        confirmation_path_evaluations=confirmation.path_evaluations,
        alignment_reporting_only={
            view: fixed_key.alignment_report(alignment_input, view) for view in fixed_key.FIXED_VIEWS
        },
    )


def source_statistic(views, candidate, spec_sha, key_id):
    expected = dict(
        spec_sha256=spec_sha, key_id=key_id, receiver_protocol_id=PROTOCOL_ID,
    )
    if (
        set(views) != set(fixed_key.FIXED_VIEWS)
        or any(
            value.get("status") != "SCORED"
            or any(value.get(key) != expected_value for key, expected_value in expected.items())
            for value in views.values()
        )
    ):
        return dict(status="INVALID", statistic=None, candidate_id=candidate, **expected)
    scores = {name: views[name]["confirmation_scores"][candidate] for name in fixed_key.FIXED_VIEWS}
    if any(score is None or not math.isfinite(score) for score in scores.values()):
        return dict(status="INVALID", statistic=None, candidate_id=candidate, **expected)
    winner = max(fixed_key.FIXED_VIEWS, key=lambda view: (scores[view], view))
    return dict(
        status="SCORED", statistic=scores[winner], winning_view=winner,
        candidate_id=candidate, **expected,
    )


def calibrate(calibration_sources, candidate, spec_sha, key_id):
    ids = ("cal_off_p0_s1", "cal_off_p1_s1")
    binding = dict(
        candidate_id=candidate, spec_sha256=spec_sha, key_id=key_id,
        receiver_protocol_id=PROTOCOL_ID,
    )
    ok = set(calibration_sources) == set(ids) and all(
        row.get("status") == "SCORED"
        and all(row.get(key) == value for key, value in binding.items())
        for row in calibration_sources.values()
    )
    return dict(
        status="FROZEN" if ok else "UNCALIBRATED",
        threshold=max(row["statistic"] for row in calibration_sources.values()) + 1e-6 if ok else None,
        guard=1e-6, sources=calibration_sources, source_count=2, **binding,
    )


def decide(score, status, calibration, candidate, spec_sha, key_id):
    if status != "SCORED" or score is None or not math.isfinite(score):
        return dict(status="INVALID", detected=None, statistic=score)
    binding = dict(
        candidate_id=candidate, spec_sha256=spec_sha, key_id=key_id,
        receiver_protocol_id=PROTOCOL_ID,
    )
    if calibration.get("status") != "FROZEN" or any(
        calibration.get(key) != value for key, value in binding.items()
    ):
        return dict(status="UNCALIBRATED", detected=None, statistic=score)
    threshold = calibration["threshold"]
    return dict(
        status="DETECTED" if score > threshold else "REJECTED",
        detected=score > threshold, statistic=score, threshold=threshold,
    )
