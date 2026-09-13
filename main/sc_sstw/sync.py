"""State-constrained dynamic temporal synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .linalg import squared_norm


@dataclass(frozen=True)
class SyncResult:
    score: float
    path: tuple[tuple[int, int], ...]
    average_cost: float


def _state_cost(observed: tuple[float, float], expected: tuple[float, float]) -> float:
    return squared_norm((observed[0] - expected[0], observed[1] - expected[1]))


def dynamic_time_sync(
    observed_states: list[tuple[float, float]],
    candidate_states: list[tuple[float, float]],
    *,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> SyncResult:
    """Align observed states to candidate states with deletion/repeat tolerance."""

    rows = len(observed_states)
    cols = len(candidate_states)
    dp = [[math.inf] * (cols + 1) for _ in range(rows + 1)]
    parent: list[list[tuple[int, int] | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    dp[0][0] = 0.0
    for col in range(1, cols + 1):
        dp[0][col] = dp[0][col - 1] + skip_penalty
        parent[0][col] = (0, col - 1)
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            match_cost = _state_cost(observed_states[row - 1], candidate_states[col - 1])
            candidates = (
                (dp[row - 1][col - 1] + match_cost, (row - 1, col - 1)),
                (dp[row][col - 1] + skip_penalty, (row, col - 1)),
                (dp[row - 1][col] + repeat_penalty + match_cost, (row - 1, col)),
            )
            best_cost, best_parent = min(candidates, key=lambda item: item[0])
            dp[row][col] = best_cost
            parent[row][col] = best_parent
    path: list[tuple[int, int]] = []
    row, col = rows, cols
    while row > 0 or col > 0:
        previous = parent[row][col]
        if previous is None:
            break
        prev_row, prev_col = previous
        if row > prev_row and col > prev_col:
            path.append((row - 1, col - 1))
        row, col = prev_row, prev_col
    path.reverse()
    average_cost = dp[rows][cols] / max(1, len(path))
    return SyncResult(score=-average_cost, path=tuple(path), average_cost=average_cost)


def dynamic_time_sync_score(
    observed_states: list[tuple[float, float]],
    candidate_states: list[tuple[float, float]],
    *,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> float:
    """Return the exact `dynamic_time_sync(...).score` without storing a path.

    The existing scorer normalizes the final DP cost by the number of diagonal
    match steps recovered during backtracking. This helper keeps that same
    match-count alongside the rolling DP rows, preserving the original
    transition order and tie-breaking while avoiding the full parent matrix.
    """

    cols = len(candidate_states)
    previous_costs = [math.inf] * (cols + 1)
    previous_counts = [0] * (cols + 1)
    previous_costs[0] = 0.0
    for col in range(1, cols + 1):
        previous_costs[col] = previous_costs[col - 1] + skip_penalty
        previous_counts[col] = previous_counts[col - 1]

    for observed in observed_states:
        observed_x, observed_y = observed
        current_costs = [math.inf] * (cols + 1)
        current_counts = [0] * (cols + 1)
        for col in range(1, cols + 1):
            expected_x, expected_y = candidate_states[col - 1]
            delta_x = observed_x - expected_x
            delta_y = observed_y - expected_y
            match_cost = delta_x * delta_x + delta_y * delta_y
            best_cost = previous_costs[col - 1] + match_cost
            best_count = previous_counts[col - 1] + 1
            skip_cost = current_costs[col - 1] + skip_penalty
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_count = current_counts[col - 1]
            repeat_cost = previous_costs[col] + repeat_penalty + match_cost
            if repeat_cost < best_cost:
                best_cost = repeat_cost
                best_count = previous_counts[col]
            current_costs[col] = best_cost
            current_counts[col] = best_count
        previous_costs = current_costs
        previous_counts = current_counts

    average_cost = previous_costs[cols] / max(1, previous_counts[cols])
    return -average_cost


def dynamic_time_sync_score_bounded(
    observed_states: list[tuple[float, float]],
    candidate_states: list[tuple[float, float]],
    *,
    min_score_to_beat: float,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> tuple[float, bool]:
    """Return exact score or safely abandon when it cannot beat a score.

    The original score is `-final_cost / diagonal_match_count`. During the DP,
    all costs are non-negative and can only increase, while the final diagonal
    match count cannot exceed `len(observed_states)`. Therefore
    `min_current_cost / len(observed_states)` is a conservative lower bound on
    the final average cost. If that lower bound is already worse than the
    current best average cost, this candidate cannot beat `min_score_to_beat`.

    Returns `(score, abandoned)`. When `abandoned` is false, `score` is exactly
    `dynamic_time_sync_score(...)`. When true, `score` is a safe value less than
    or equal to `min_score_to_beat`; callers must not use it as an exact score.
    """

    if min_score_to_beat == -math.inf:
        return dynamic_time_sync_score(
            observed_states,
            candidate_states,
            skip_penalty=skip_penalty,
            repeat_penalty=repeat_penalty,
        ), False

    rows = len(observed_states)
    if rows == 0:
        return dynamic_time_sync_score(
            observed_states,
            candidate_states,
            skip_penalty=skip_penalty,
            repeat_penalty=repeat_penalty,
        ), False

    best_average_cost_to_beat = -min_score_to_beat
    cols = len(candidate_states)
    previous_costs = [math.inf] * (cols + 1)
    previous_counts = [0] * (cols + 1)
    previous_costs[0] = 0.0
    for col in range(1, cols + 1):
        previous_costs[col] = previous_costs[col - 1] + skip_penalty
        previous_counts[col] = previous_counts[col - 1]
    if min(previous_costs) / rows > best_average_cost_to_beat:
        return min_score_to_beat, True

    for observed in observed_states:
        observed_x, observed_y = observed
        current_costs = [math.inf] * (cols + 1)
        current_counts = [0] * (cols + 1)
        for col in range(1, cols + 1):
            expected_x, expected_y = candidate_states[col - 1]
            delta_x = observed_x - expected_x
            delta_y = observed_y - expected_y
            match_cost = delta_x * delta_x + delta_y * delta_y
            best_cost = previous_costs[col - 1] + match_cost
            best_count = previous_counts[col - 1] + 1
            skip_cost = current_costs[col - 1] + skip_penalty
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_count = current_counts[col - 1]
            repeat_cost = previous_costs[col] + repeat_penalty + match_cost
            if repeat_cost < best_cost:
                best_cost = repeat_cost
                best_count = previous_counts[col]
            current_costs[col] = best_cost
            current_counts[col] = best_count
        previous_costs = current_costs
        previous_counts = current_counts
        if min(previous_costs) / rows > best_average_cost_to_beat:
            return min_score_to_beat, True

    average_cost = previous_costs[cols] / max(1, previous_counts[cols])
    return -average_cost, False


def flatten_state_pairs(states: list[tuple[float, float]]) -> list[float]:
    """Return `[x0, y0, x1, y1, ...]` for scorer hot paths."""

    flattened: list[float] = []
    for state_x, state_y in states:
        flattened.append(state_x)
        flattened.append(state_y)
    return flattened


def dynamic_time_sync_score_flat(
    observed_xy: list[float],
    candidate_xy: list[float],
    *,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> float:
    """Exact score-only DTW over flattened 2D states.

    This is the same recurrence and tie-break order as
    `dynamic_time_sync_score`; it only avoids tuple unpacking in large
    exhaustive-search diagnostics.
    """

    cols = len(candidate_xy) // 2
    previous_costs = [math.inf] * (cols + 1)
    previous_counts = [0] * (cols + 1)
    previous_costs[0] = 0.0
    for col in range(1, cols + 1):
        previous_costs[col] = previous_costs[col - 1] + skip_penalty
        previous_counts[col] = previous_counts[col - 1]

    for observed_index in range(0, len(observed_xy), 2):
        observed_x = observed_xy[observed_index]
        observed_y = observed_xy[observed_index + 1]
        current_costs = [math.inf] * (cols + 1)
        current_counts = [0] * (cols + 1)
        candidate_index = 0
        for col in range(1, cols + 1):
            delta_x = observed_x - candidate_xy[candidate_index]
            delta_y = observed_y - candidate_xy[candidate_index + 1]
            candidate_index += 2
            match_cost = delta_x * delta_x + delta_y * delta_y
            best_cost = previous_costs[col - 1] + match_cost
            best_count = previous_counts[col - 1] + 1
            skip_cost = current_costs[col - 1] + skip_penalty
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_count = current_counts[col - 1]
            repeat_cost = previous_costs[col] + repeat_penalty + match_cost
            if repeat_cost < best_cost:
                best_cost = repeat_cost
                best_count = previous_counts[col]
            current_costs[col] = best_cost
            current_counts[col] = best_count
        previous_costs = current_costs
        previous_counts = current_counts

    average_cost = previous_costs[cols] / max(1, previous_counts[cols])
    return -average_cost


def dynamic_time_sync_score_bounded_flat(
    observed_xy: list[float],
    candidate_xy: list[float],
    *,
    min_score_to_beat: float,
    skip_penalty: float = 0.22,
    repeat_penalty: float = 0.14,
) -> tuple[float, bool]:
    """Exact bounded score-only DTW over flattened 2D states."""

    if min_score_to_beat == -math.inf:
        return dynamic_time_sync_score_flat(
            observed_xy,
            candidate_xy,
            skip_penalty=skip_penalty,
            repeat_penalty=repeat_penalty,
        ), False

    rows = len(observed_xy) // 2
    if rows == 0:
        return dynamic_time_sync_score_flat(
            observed_xy,
            candidate_xy,
            skip_penalty=skip_penalty,
            repeat_penalty=repeat_penalty,
        ), False

    best_average_cost_to_beat = -min_score_to_beat
    cols = len(candidate_xy) // 2

    def best_possible_average_cost_lower_bound(
        costs: list[float],
        counts: list[int],
        *,
        processed_rows: int,
    ) -> float:
        remaining_rows = rows - processed_rows
        lower_bound = math.inf
        for col, cost in enumerate(costs):
            required_skip_count = max(0, cols - col - remaining_rows)
            minimum_final_cost = cost + required_skip_count * skip_penalty
            maximum_final_match_count = counts[col] + min(remaining_rows, cols - col)
            average_lower_bound = minimum_final_cost / max(1, maximum_final_match_count)
            if average_lower_bound < lower_bound:
                lower_bound = average_lower_bound
        return lower_bound

    previous_costs = [math.inf] * (cols + 1)
    previous_counts = [0] * (cols + 1)
    previous_costs[0] = 0.0
    for col in range(1, cols + 1):
        previous_costs[col] = previous_costs[col - 1] + skip_penalty
        previous_counts[col] = previous_counts[col - 1]
    if best_possible_average_cost_lower_bound(
        previous_costs,
        previous_counts,
        processed_rows=0,
    ) > best_average_cost_to_beat:
        return min_score_to_beat, True

    for row_index, observed_index in enumerate(range(0, len(observed_xy), 2), start=1):
        observed_x = observed_xy[observed_index]
        observed_y = observed_xy[observed_index + 1]
        current_costs = [math.inf] * (cols + 1)
        current_counts = [0] * (cols + 1)
        candidate_index = 0
        for col in range(1, cols + 1):
            delta_x = observed_x - candidate_xy[candidate_index]
            delta_y = observed_y - candidate_xy[candidate_index + 1]
            candidate_index += 2
            match_cost = delta_x * delta_x + delta_y * delta_y
            best_cost = previous_costs[col - 1] + match_cost
            best_count = previous_counts[col - 1] + 1
            skip_cost = current_costs[col - 1] + skip_penalty
            if skip_cost < best_cost:
                best_cost = skip_cost
                best_count = current_counts[col - 1]
            repeat_cost = previous_costs[col] + repeat_penalty + match_cost
            if repeat_cost < best_cost:
                best_cost = repeat_cost
                best_count = previous_counts[col]
            current_costs[col] = best_cost
            current_counts[col] = best_count
        previous_costs = current_costs
        previous_counts = current_counts
        if best_possible_average_cost_lower_bound(
            previous_costs,
            previous_counts,
            processed_rows=row_index,
        ) > best_average_cost_to_beat:
            return min_score_to_beat, True

    average_cost = previous_costs[cols] / max(1, previous_counts[cols])
    return -average_cost, False
