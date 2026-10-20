
from __future__ import annotations

from collections.abc import Callable


EVIDENCE_ENABLEMENT_ORDER = ("tubelet", "sync")


def build_disabled_evidence(enabled_evidence: dict[str, bool]) -> list[str]:
    if not isinstance(enabled_evidence, dict):
        raise TypeError("enabled_evidence must be a dictionary")
    disabled_evidence = [
        evidence_name
        for evidence_name in EVIDENCE_ENABLEMENT_ORDER
        if not bool(enabled_evidence.get(evidence_name, False))
    ]
    return disabled_evidence


def constant_zero_fusion_placeholder(evidence_scores: dict[str, float | None]) -> float:
    if not isinstance(evidence_scores, dict):
        raise TypeError("evidence_scores must be a dictionary")
    return 0.0


def random_score_linear_fusion_random(evidence_scores: dict[str, float | None]) -> float:
    if not isinstance(evidence_scores, dict):
        raise TypeError("evidence_scores must be a dictionary")
    available_scores = [
        float(evidence_scores[score_name])
        for score_name in ("S_tubelet", "S_sync")
        if evidence_scores.get(score_name) is not None
    ]
    if not available_scores:
        # 当没有有效 evidence 时，协议仍需返回稳定分数而非抛弃字段。
        return 0.0
    return round(sum(available_scores) / len(available_scores), 6)


def tubelet_score_only(evidence_scores: dict[str, float | None]) -> float:
    if not isinstance(evidence_scores, dict):
        raise TypeError("evidence_scores must be a dictionary")
    tubelet_score = evidence_scores.get("S_tubelet")
    if tubelet_score is None:
        return 0.0
    return round(float(tubelet_score), 6)


def calibrated_tubelet_sync(evidence_scores: dict[str, float | None]) -> float:
    if not isinstance(evidence_scores, dict):
        raise TypeError("evidence_scores must be a dictionary")
    tubelet_score = evidence_scores.get("S_tubelet")
    sync_score = evidence_scores.get("S_sync")
    if tubelet_score is None and sync_score is None:
        return 0.0
    if tubelet_score is None:
        return round(float(sync_score), 6)
    if sync_score is None or float(sync_score) <= 0.0:
        return round(float(tubelet_score), 6)
    return round((0.65 * float(tubelet_score)) + (0.35 * float(sync_score)), 6)


def sync_rescue_fusion(
    evidence_scores: dict[str, float | None],
    payload_rescue_gain: float = 0.0,
    lambda_sync: float = 0.1,
    gate_sync: bool = False,
) -> float:
    if not isinstance(evidence_scores, dict):
        raise TypeError("evidence_scores must be a dictionary")
    if not isinstance(payload_rescue_gain, (int, float)):
        raise TypeError("payload_rescue_gain must be numeric")
    if not isinstance(lambda_sync, (int, float)):
        raise TypeError("lambda_sync must be numeric")
    tubelet_score = evidence_scores.get("S_tubelet")
    base_score = 0.0 if tubelet_score is None else float(tubelet_score)
    gated_rescue_gain = max(0.0, float(payload_rescue_gain)) if gate_sync else 0.0
    return round(
        base_score + gated_rescue_gain,
        6,
    )


def get_fusion_rule(fusion_rule: str) -> Callable[[dict[str, float | None]], float]:
    if not isinstance(fusion_rule, str) or not fusion_rule:
        raise ValueError("fusion_rule must be a non-empty string")
    if fusion_rule == "constant_zero_fusion_placeholder":
        return constant_zero_fusion_placeholder
    if fusion_rule == "random_score_linear_fusion_random":
        return random_score_linear_fusion_random
    if fusion_rule == "tubelet_score_only":
        return tubelet_score_only
    if fusion_rule == "calibrated_tubelet_sync":
        return calibrated_tubelet_sync
    if fusion_rule == "sync_rescue_fusion":
        return sync_rescue_fusion
    raise ValueError(f"unsupported fusion_rule: {fusion_rule}")
