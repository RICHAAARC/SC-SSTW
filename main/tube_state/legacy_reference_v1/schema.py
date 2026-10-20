from dataclasses import dataclass
from typing import Any
EVIDENCE_SCORE_ORDER = ("S_tubelet", "S_sync", "S_final")
EVIDENCE_SCORE_NAMES=set(EVIDENCE_SCORE_ORDER)

@dataclass(frozen=True)
class LatentSample:

    sample_id: str
    split: str
    sample_role: str
    latent_shape: tuple[int, int, int, int]
    latent_tensor_digest_random: str
    latent_generation_seed_random: int
    latent_backend_name: str
    latent_backend_status: str
    latent_artifact_relpath: str | None = None
    latent_artifact_path: str | None = None
    latent_artifact_digest: str | None = None
    run_root_path: str | None = None
    mechanism_trace: dict[str, Any] | None = None
    applied_attack_params: dict[str, Any] | None = None

def build_empty_evidence_scores(final_score: float = 0.0) -> dict[str, float | None]:
    if not isinstance(final_score, (int, float)):
        # 最终分数必须可转换为浮点数，否则无法进入后续阈值逻辑。
        raise TypeError("final_score must be numeric")
    return {
        "S_tubelet": None,
        "S_sync": None,
        "S_final": round(float(final_score), 6),
    }

def validate_evidence_scores(evidence_scores: dict[str, float | None]) -> None:
    if not isinstance(evidence_scores, dict):
        # evidence score 容器必须是字典，便于后续稳定序列化。
        raise TypeError("evidence_scores must be a dictionary")

    missing_fields = EVIDENCE_SCORE_NAMES.difference(evidence_scores.keys())
    if missing_fields:
        # 缺失任一 evidence 字段都会破坏 record schema 冻结口径。
        raise ValueError(f"missing evidence score fields: {sorted(missing_fields)}")

    for score_name in EVIDENCE_SCORE_ORDER:
        score_value = evidence_scores[score_name]
        if score_value is not None and not isinstance(score_value, (int, float)):
            # evidence 值只允许浮点数或显式空值编码。
            raise ValueError(f"invalid evidence score value for {score_name}")
