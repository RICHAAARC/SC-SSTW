"""Affine-invariant synchronization burst primitives."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from .linalg import Vector, squared_norm


Vector2 = tuple[float, float]


@dataclass(frozen=True)
class BurstTemplate:
    """A public AISB template with affine checksum geometry."""

    template_id: str
    points: tuple[Vector2, ...]

    @property
    def length(self) -> int:
        return len(self.points)


MissingTemplateIndex = int | tuple[int, ...] | None


def _missing_indices(missing_template_index: MissingTemplateIndex) -> tuple[int, ...]:
    if missing_template_index is None:
        return ()
    if isinstance(missing_template_index, int):
        return (missing_template_index,)
    return tuple(missing_template_index)


def _missing_sort_key(missing_template_index: MissingTemplateIndex) -> tuple[int, ...]:
    indices = _missing_indices(missing_template_index)
    return indices if indices else (-1,)


@dataclass(frozen=True)
class BurstCandidate:
    start_index: int
    template_id: str
    residual: float
    observed_length: int
    missing_template_index: MissingTemplateIndex = None

    @property
    def source_span_length(self) -> int:
        """Template span covered by this observed candidate."""

        return self.observed_length + len(_missing_indices(self.missing_template_index))


def barycentric_weights(point: Vector2, anchors: tuple[Vector2, Vector2, Vector2]) -> tuple[float, float, float]:
    """Return barycentric coordinates of `point` with respect to 3 anchors."""

    (x1, y1), (x2, y2), (x3, y3) = anchors
    x, y = point
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(det) < 1e-12:
        raise ValueError("degenerate affine anchors")
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    w3 = 1.0 - w1 - w2
    return w1, w2, w3


def _combine(weights: tuple[float, float, float], vectors: tuple[Vector, Vector, Vector]) -> Vector:
    return [
        weights[0] * vectors[0][index]
        + weights[1] * vectors[1][index]
        + weights[2] * vectors[2][index]
        for index in range(len(vectors[0]))
    ]


def _scatter(observations: list[Vector]) -> float:
    dimension = len(observations[0])
    mean = [
        sum(observation[index] for observation in observations) / len(observations)
        for index in range(dimension)
    ]
    return sum(squared_norm([value - mean[index] for index, value in enumerate(observation)]) for observation in observations)


def _non_collinear_anchor_sets(template_indices: tuple[int, ...], template: BurstTemplate) -> list[tuple[int, int, int]]:
    """Return public anchor triples for a deletion-tolerant AISB window.

    The complete and one-deletion contracts preserve the original 0,1,2 anchor
    preference whenever possible. For two missing points, redundant templates may
    lose too much of that preferred anchor set, so the scanner enumerates public
    non-collinear triples from retained template points and averages their public
    checksum residuals. This still does not estimate A,b or choose geometry from
    secret state; it only enumerates template-declared public alternatives.
    """

    present = set(template_indices)
    original_anchors = (0, 1, 2)
    if all(index in present for index in original_anchors):
        return [original_anchors]
    missing_original = [index for index in original_anchors if index not in present]
    if len(missing_original) == 1 and template.length >= 9:
        redundant_anchor_index = 6 + missing_original[0]
        if redundant_anchor_index in present:
            return [tuple(sorted((*(index for index in original_anchors if index in present), redundant_anchor_index)))]

    if template.length >= 12:
        anchor_groups = (
            tuple(index for index in (0, 6, 9) if index in present),
            tuple(index for index in (1, 7, 10) if index in present),
            tuple(index for index in (2, 8, 11) if index in present),
        )
        if all(anchor_groups):
            return [tuple(sorted(indices)) for indices in product(*anchor_groups)]

    anchor_sets: list[tuple[int, int, int]] = []
    for indices in combinations(template_indices, 3):
        try:
            barycentric_weights(template.points[template_indices[0]], tuple(template.points[index] for index in indices))
        except ValueError:
            continue
        anchor_sets.append(tuple(indices))
    return anchor_sets


def _mapped_observations(
    observation_window: list[Vector],
    template: BurstTemplate,
    *,
    missing_template_index: MissingTemplateIndex,
) -> list[tuple[int, Vector]]:
    missing_indices = _missing_indices(missing_template_index)
    if len(set(missing_indices)) != len(missing_indices):
        raise ValueError("missing_template_index entries must be unique")
    if any(index < 0 or index >= template.length for index in missing_indices):
        raise ValueError("missing_template_index must be within the template")
    expected_length = template.length - len(missing_indices)
    if len(observation_window) != expected_length:
        raise ValueError("observation window length must match template length minus missing points")
    missing_set = set(missing_indices)
    present_indices = tuple(index for index in range(template.length) if index not in missing_set)
    return list(zip(present_indices, observation_window, strict=True))


def affine_burst_residual(
    observation_window: list[Vector],
    template: BurstTemplate,
    *,
    missing_template_index: MissingTemplateIndex = None,
) -> float:
    """Score a window against public affine geometry without fitting A,b.

    If burst points are missing, the caller supplies the missing public-template
    index or index tuple. The residual averages all affine constraints induced by
    public non-collinear anchor triples among observed template points. This
    preserves the AISB acquisition boundary: no video-specific affine channel is
    estimated while locating the burst.
    """

    if template.length < 6:
        raise ValueError("AISB template length must be at least 6")
    if missing_template_index is None:
        if len(observation_window) != template.length:
            raise ValueError("observation window length must match template length")
        anchors = (template.points[0], template.points[1], template.points[2])
        observed_anchors = (observation_window[0], observation_window[1], observation_window[2])
        residual = 0.0
        for point, observed in zip(template.points[3:], observation_window[3:], strict=True):
            weights = barycentric_weights(point, anchors)
            predicted = _combine(weights, observed_anchors)
            residual += squared_norm([
                value - predicted[index]
                for index, value in enumerate(observed)
            ])
        return residual / (_scatter(observation_window) + 1e-9)

    mapped = _mapped_observations(observation_window, template, missing_template_index=missing_template_index)
    template_to_observed = {template_index: observed for template_index, observed in mapped}
    present_indices = tuple(template_index for template_index, _ in mapped)
    anchor_sets = _non_collinear_anchor_sets(present_indices, template)
    if not anchor_sets:
        raise ValueError("deleted AISB window must retain non-collinear anchors")
    normalized_residuals: list[float] = []
    scatter = _scatter([observed for _, observed in mapped]) + 1e-9
    for anchor_indices in anchor_sets:
        residual = 0.0
        scored = 0
        anchors = tuple(template.points[index] for index in anchor_indices)
        observed_anchors = tuple(template_to_observed[index] for index in anchor_indices)
        for template_index in present_indices:
            if template_index in anchor_indices:
                continue
            weights = barycentric_weights(template.points[template_index], anchors)
            predicted = _combine(weights, observed_anchors)
            observed = template_to_observed[template_index]
            residual += squared_norm([
                value - predicted[index]
                for index, value in enumerate(observed)
            ])
            scored += 1
        if scored > 0:
            normalized_residuals.append(residual / scatter)
    if not normalized_residuals:
        raise ValueError("deleted AISB window has no checksum point to score")
    return sum(normalized_residuals) / len(normalized_residuals)


def scan_burst_candidates(
    observations: list[Vector],
    templates: tuple[BurstTemplate, ...],
    *,
    top_k_per_start: int = 1,
    allow_single_deletion: bool = False,
    allow_double_deletion: bool = False,
) -> list[BurstCandidate]:
    """Scan contiguous windows and return affine-invariant burst candidates."""

    if not templates:
        raise ValueError("at least one template is required")
    length = templates[0].length
    if any(template.length != length for template in templates):
        raise ValueError("all templates must share length")
    candidates: list[BurstCandidate] = []
    window_lengths = [length]
    if allow_single_deletion:
        window_lengths.append(length - 1)
    if allow_double_deletion:
        window_lengths.append(length - 2)
    for window_length in window_lengths:
        for start in range(0, len(observations) - window_length + 1):
            scored: list[BurstCandidate] = []
            for template in templates:
                if window_length == length:
                    scored.append(
                        BurstCandidate(
                            start_index=start,
                            template_id=template.template_id,
                            residual=affine_burst_residual(observations[start : start + window_length], template),
                            observed_length=window_length,
                            missing_template_index=None,
                        )
                    )
                else:
                    missing_count = length - window_length
                    for missing_template_index in combinations(range(length), missing_count):
                        candidate_missing: MissingTemplateIndex = (
                            missing_template_index[0]
                            if len(missing_template_index) == 1
                            else missing_template_index
                        )
                        scored.append(
                            BurstCandidate(
                                start_index=start,
                                template_id=template.template_id,
                                residual=affine_burst_residual(
                                    observations[start : start + window_length],
                                    template,
                                    missing_template_index=candidate_missing,
                                ),
                                observed_length=window_length,
                                missing_template_index=candidate_missing,
                            )
                        )
            scored.sort(key=lambda candidate: candidate.residual)
            candidates.extend(scored[:top_k_per_start])
    return candidates


def make_default_templates() -> tuple[BurstTemplate, ...]:
    """Return small public AISB templates with distinct affine checksum geometry."""

    return (
        BurstTemplate(
            "burst_alpha",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.25, 0.35),
                (0.75, 0.20),
                (0.20, 0.80),
            ),
        ),
        BurstTemplate(
            "burst_beta",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.40, 0.15),
                (0.62, 0.58),
                (0.12, 0.48),
            ),
        ),
        BurstTemplate(
            "burst_gamma",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.18, 0.22),
                (0.52, 0.76),
                (0.86, 0.10),
            ),
        ),
    )



def make_redundant_templates() -> tuple[BurstTemplate, ...]:
    """Return AISB templates with redundant anchors for any one deletion.

    Indices 0,1,2 are the primary anchors, 3,4,5 are checksum points, and
    6,7,8 are exact public redundant copies of anchors 0,1,2. This synthetic
    construction tests whether AISB can be made identifiable under one arbitrary
    missing burst sample; it is not a claim that the shorter default template has
    this property.
    """

    return (
        BurstTemplate(
            "redundant_alpha",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.23, 0.31),
                (0.68, 0.17),
                (0.16, 0.74),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
        ),
        BurstTemplate(
            "redundant_beta",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.34, 0.18),
                (0.61, 0.53),
                (0.11, 0.42),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
        ),
        BurstTemplate(
            "redundant_gamma",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.19, 0.24),
                (0.55, 0.72),
                (0.88, 0.12),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
        ),
    )


def make_double_redundant_templates() -> tuple[BurstTemplate, ...]:
    """Return AISB templates with enough public anchor redundancy for two deletions.

    Indices 0,1,2 are primary anchors, 3,4,5 are checksum points, 6,7,8 are one
    redundant anchor copy, and 9,10,11 are a second redundant anchor copy. With
    any two missing template points, at least one representative of each anchor
    class remains public. This is a stronger synthetic construction than
    ``make_redundant_templates`` and is used only for double-missing temporal
    robustness feasibility.
    """

    return (
        BurstTemplate(
            "double_redundant_alpha",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.23, 0.31),
                (0.68, 0.17),
                (0.16, 0.74),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
        ),
        BurstTemplate(
            "double_redundant_beta",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.34, 0.18),
                (0.61, 0.53),
                (0.11, 0.42),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
        ),
        BurstTemplate(
            "double_redundant_gamma",
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.19, 0.24),
                (0.55, 0.72),
                (0.88, 0.12),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
        ),
    )


def template_observation_pairs(
    candidate: BurstCandidate,
    observations: list[Vector],
    template: BurstTemplate,
) -> list[tuple[Vector2, Vector]]:
    """Return public template points paired with observed burst vectors."""

    window = observations[candidate.start_index : candidate.start_index + candidate.observed_length]
    return [
        (template.points[template_index], observed)
        for template_index, observed in _mapped_observations(
            window,
            template,
            missing_template_index=candidate.missing_template_index,
        )
    ]


def best_non_overlapping_sequence(
    candidates: list[BurstCandidate],
    *,
    burst_length: int,
    residual_threshold: float,
    maximize_count: bool = False,
) -> list[BurstCandidate]:
    """Select a deterministic low-residual non-overlapping candidate sequence.

    Default selection preserves the original cluster near-tie behavior used for
    one-deletion AISB acquisition. ``maximize_count=True`` is reserved for the
    double-deletion temporal robustness construction, where low-residual bridge
    windows can otherwise merge two real bursts into one cluster.
    """

    if maximize_count:
        eligible = sorted(
            (candidate for candidate in candidates if candidate.residual <= residual_threshold),
            key=lambda item: (item.start_index + item.observed_length, item.start_index, item.residual),
        )
        if not eligible:
            return []
        ends = [candidate.start_index + candidate.observed_length for candidate in eligible]
        previous: list[int] = []
        for index, candidate in enumerate(eligible):
            prev = -1
            for other in range(index - 1, -1, -1):
                if ends[other] <= candidate.start_index:
                    prev = other
                    break
            previous.append(prev)
        states: list[tuple[tuple[int, float, int], tuple[int, ...]]] = [((0, 0.0, 0), ())]
        for index, candidate in enumerate(eligible):
            skip_score, skip_indices = states[-1]
            prev_score, prev_indices = states[previous[index] + 1]
            take_score = (
                prev_score[0] + 1,
                prev_score[1] - candidate.residual,
                prev_score[2] - candidate.start_index,
            )
            take_indices = (*prev_indices, index)
            if take_score > skip_score:
                states.append((take_score, take_indices))
            else:
                states.append((skip_score, skip_indices))
        return [eligible[index] for index in states[-1][1]]

    eligible = sorted(
        (candidate for candidate in candidates if candidate.residual <= residual_threshold),
        key=lambda item: (item.start_index, item.start_index + item.observed_length, item.residual),
    )
    accepted: list[BurstCandidate] = []
    cluster: list[BurstCandidate] = []
    cluster_end = -1

    def choose_cluster(items: list[BurstCandidate]) -> BurstCandidate | None:
        if not items:
            return None
        min_residual = min(candidate.residual for candidate in items)
        near_tie_limit = min_residual * 1.25 + 1e-12
        near_tie = [candidate for candidate in items if candidate.residual <= near_tie_limit]
        return min(
            near_tie,
            key=lambda item: (
                item.start_index,
                item.missing_template_index is None,
                _missing_sort_key(item.missing_template_index),
                item.residual,
                item.template_id,
            ),
        )

    for candidate in eligible:
        candidate_end = candidate.start_index + candidate.observed_length
        if not cluster:
            cluster = [candidate]
            cluster_end = candidate_end
            continue
        if candidate.start_index < cluster_end:
            cluster.append(candidate)
            cluster_end = max(cluster_end, candidate_end)
            continue
        chosen = choose_cluster(cluster)
        if chosen is not None:
            accepted.append(chosen)
        cluster = [candidate]
        cluster_end = candidate_end
    chosen = choose_cluster(cluster)
    if chosen is not None:
        accepted.append(chosen)
    return accepted
