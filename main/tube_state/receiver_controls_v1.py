"""Fixed received-score controls: median column centering and duration-four DP.

Only the received matrix enters the selectors. No attack, truth, writer, RECON,
message, model or media dependency is accepted. This is same-batch development.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import solve_monotone

SOURCE_FRAMES = 181
MAX_DWELL = 4
ARMS = ("RAW_U", "CENTERED_U", "RAW_D4", "CENTERED_D4")


def received_scores(value) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != SOURCE_FRAMES:
        raise ValueError("received score must have shape [N>=1,181]")
    if not bool(np.isfinite(q).all()):
        raise ValueError("received score contains nonfinite entries")
    return q


def center_received_scores(value) -> tuple[np.ndarray, np.ndarray]:
    q = received_scores(value)
    background = np.median(q, axis=0)
    centered = q - background
    if not bool(np.isfinite(centered).all()):
        raise FloatingPointError("column subtraction produced a nonfinite score")
    return background, centered


def path_statistics(path: list[int] | None) -> dict[str, Any] | None:
    if path is None:
        return None
    lengths = []
    current = 1
    for left, right in zip(path, path[1:]):
        if right == left:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    changes = [b-a for a, b in zip(path, path[1:])]
    return dict(source_start=path[0], source_end=path[-1], max_dwell=max(lengths),
                distinct_sources=len(set(path)),
                transition_counts=dict(stay=sum(d == 0 for d in changes),
                                       advance_one=sum(d == 1 for d in changes),
                                       skip=sum(d > 1 for d in changes)))


def _duration_top_two(local: np.ndarray, max_dwell: int = MAX_DWELL) -> dict[str, Any]:
    """Internal small-width-capable recurrence; public entry fixes 181 and D4.

    State (source, duration, rank) contains two distinct prefix paths. A strict
    source advance resets duration; a stay extends it. Free start/end and no
    soft transition penalty. State paths cannot alias across duration states.
    """
    n, width = local.shape
    if n > width * max_dwell:
        return dict(status="UNSUPPORTED", reason="NO_COMPLETE_DURATION_PATH",
                    path=None, best_path=None, runner_up_path=None,
                    best_score=None, runner_up_score=None, score_gap=None,
                    exact_tie=False)
    prior = np.full((width, max_dwell, 2), -np.inf, dtype=np.float64)
    prior[:, 0, 0] = local[0]
    back = np.full((n, width, max_dwell, 2, 3), -1, dtype=np.int32)
    for t in range(1, n):
        current = np.full_like(prior, -np.inf)
        prefix: list[tuple[float, int, int, int]] = []
        for s in range(width):
            # Only p<s is present here; s itself enters the prefix afterward.
            for rank, (score, p, duration, old_rank) in enumerate(prefix):
                current[s, 0, rank] = score + local[t, s]
                back[t, s, 0, rank] = p, duration, old_rank
            for duration in range(1, max_dwell):
                for rank in range(2):
                    score = prior[s, duration-1, rank]
                    if score != -np.inf:
                        current[s, duration, rank] = score + local[t, s]
                        back[t, s, duration, rank] = s, duration-1, rank
            for duration in range(max_dwell):
                for rank in range(2):
                    score = prior[s, duration, rank]
                    if score != -np.inf:
                        prefix.append((float(score), s, duration, rank))
            prefix.sort(key=lambda item: item[0], reverse=True)
            del prefix[2:]
        if np.isnan(current).any() or np.isposinf(current).any():
            raise FloatingPointError("nonfinite duration path objective")
        prior = current
    finals = [(float(prior[s, d, r]), s, d, r)
              for s in range(width) for d in range(max_dwell) for r in range(2)
              if prior[s, d, r] != -np.inf]
    finals.sort(key=lambda item: item[0], reverse=True)
    if not finals:
        raise RuntimeError("duration DP lost a legal complete path")

    def reconstruct(final):
        _, s, duration, rank = final
        path = [s]
        for t in range(n-1, 0, -1):
            s, duration, rank = map(int, back[t, s, duration, rank])
            if min(s, duration, rank) < 0:
                raise RuntimeError("duration DP missing backpointer")
            path.append(s)
        return list(reversed(path))

    best = reconstruct(finals[0])
    runner = reconstruct(finals[1]) if len(finals) > 1 else None
    if runner == best:
        raise RuntimeError("duration runner-up duplicated the best complete path")
    best_score = finals[0][0]
    second = finals[1][0] if runner is not None else None
    tie = second is not None and second == best_score
    return dict(status="UNRESOLVED" if tie else "ESTIMATED",
                reason="EXACT_BEST_PATH_TIE" if tie else None,
                path=None if tie else best, best_path=best, runner_up_path=runner,
                best_score=best_score, runner_up_score=second,
                score_gap=None if second is None else best_score-second, exact_tie=tie)


def select_score_path(value, grammar: str) -> dict[str, Any]:
    """Select solely from the supplied local matrix and a fixed public grammar."""
    local = received_scores(value)
    if grammar == "U":
        answer = solve_monotone(local, np.ones_like(local))
    elif grammar == "D4":
        answer = _duration_top_two(local)
    else:
        raise ValueError("grammar must be U or D4")
    for key in ("best_score", "runner_up_score", "score_gap"):
        if answer.get(key) is not None and not math.isfinite(answer[key]):
            raise FloatingPointError(f"nonfinite {key}")
    return dict(status=answer["status"], reason=answer.get("reason"),
                path=answer.get("path"),
                diagnostic_best_path=answer.get("best_path"),
                diagnostic_runner_up_path=answer.get("runner_up_path"),
                best_score=answer.get("best_score"),
                runner_up_score=answer.get("runner_up_score"),
                score_gap=answer.get("score_gap"), exact_tie=answer.get("exact_tie", False),
                received_frames=len(local), source_frames=SOURCE_FRAMES, grammar=grammar,
                max_dwell_limit=MAX_DWELL if grammar == "D4" else None,
                path_statistics=path_statistics(answer.get("path")),
                diagnostic_best_statistics=path_statistics(answer.get("best_path")),
                selector_inputs="received_score_matrix + fixed_public_grammar_only",
                truth_inputs=False, attack_inputs=False, message_inputs=False,
                recon_inputs=False)


def path_sum(q: np.ndarray, path: list[int] | None) -> float | None:
    return None if path is None else math.fsum(float(q[t, s]) for t, s in enumerate(path))


def receive_scores(signed_projection, rho, receiver="RAW_U"):
    """Shared blind dispatch; input contains only observations and fixed receiver name."""
    numerator = received_scores(signed_projection)
    denominator = np.asarray(rho, dtype=np.float64)
    if denominator.shape != numerator.shape or not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("rho must be same-shaped finite positive scores")
    if receiver not in ARMS:
        raise ValueError("unknown fixed receiver")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        q = received_scores(numerator / denominator)
    background, centered = center_received_scores(q)
    estimate = select_score_path(centered if receiver.startswith("CENTERED") else q,
                                 "D4" if receiver.endswith("D4") else "U")
    # Preserve legacy estimate field names used by clock sidecars and evaluation.
    estimate["best_path"] = estimate["diagnostic_best_path"]
    estimate["runner_up_path"] = estimate["diagnostic_runner_up_path"]
    from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span
    operation = decode_visible_span(estimate["path"]) if estimate["status"] == "ESTIMATED" else None
    return dict(receiver=receiver, estimate=estimate, operation=operation,
                status=operation["status"] if operation is not None else estimate["status"],
                reason=operation.get("reason") if operation else estimate.get("reason"),
                background=background.tolist(), raw_q_path_sum=path_sum(q, estimate["path"]),
                centered_q_path_sum=path_sum(centered, estimate["path"]), truth_inputs=False)
