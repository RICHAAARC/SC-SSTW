from __future__ import annotations
from statistics import median
from .digest import compute_object_digest
from .schema import LatentSample, build_empty_evidence_scores, validate_evidence_scores
from .tensor_artifact import read_float_tensor_npy
from .codebook import (
    build_codebook_config,
    build_tubelet_codebook,
)
from .embedding import (
    build_partition_config_from_method_config,
)
from .fusion import get_fusion_rule, sync_rescue_fusion
from .interfaces import EvidenceExtractor
from .synchronization import (
    build_offset_scale_search_result,
    build_offset_search_result,
)
from .tubelet_partition import (
    build_tubelet_descriptors,
    compute_tubelet_partition_digest,
    dot_tubelet_direction,
)

class SyntheticProbeEvidenceExtractor(EvidenceExtractor):

    def __init__(
        self,
        method_variant: str,
        method_config: dict[str, object],
        enabled_evidence: dict[str, bool],
        fusion_rule: str,
    ) -> None:
        if not isinstance(method_variant, str) or not method_variant:
            raise ValueError("method_variant must be a non-empty string")
        if not isinstance(method_config, dict):
            raise TypeError("method_config must be a dictionary")
        if not isinstance(enabled_evidence, dict):
            raise TypeError("enabled_evidence must be a dictionary")
        self._method_variant = method_variant
        self._method_config = method_config
        self._enabled_evidence = enabled_evidence
        self._fusion_rule = fusion_rule
        self._fusion_callable = get_fusion_rule(fusion_rule)
        self._codebook_config = build_codebook_config()

    def extract(self, sample: LatentSample) -> tuple[dict[str, float | None], dict[str, object]]:
        if not isinstance(sample, LatentSample):
            raise TypeError("sample must be a LatentSample instance")
        if sample.latent_artifact_path is None:
            raise ValueError("sample must carry latent_artifact_path")

        tensor_artifact = read_float_tensor_npy(sample.latent_artifact_path)
        partition_config = build_partition_config_from_method_config(self._method_config)
        descriptors = build_tubelet_descriptors(sample.latent_shape, partition_config)
        evidence_scores = build_empty_evidence_scores(0.0)
        (
            payload_coded_projections,
            codebook,
            reference_descriptor_map,
            reference_shape_trace,
        ) = self._build_payload_coded_projections(
            sample,
            tensor_artifact,
            descriptors,
            partition_config,
        )
        mechanism_trace = dict(sample.mechanism_trace or {})
        mechanism_trace.update(
            {
                "tubelet_length": partition_config.tubelet_length,
                "spatial_patch_size": list(partition_config.spatial_patch_size),
                "partition_digest": compute_tubelet_partition_digest(
                    sample.latent_shape,
                    partition_config,
                ),
                "codebook_digest": codebook.codebook_digest,
                "sync_code_digest": codebook.sync_code_digest,
                "payload_digest": codebook.payload_digest,
            }
        )
        mechanism_trace.update(reference_shape_trace)
        embedding_support = self._resolve_embedding_support(sample)
        attack_strength = self._resolve_attack_strength(sample)
        tubelet_projection_coverage_ratio = float(
            reference_shape_trace["tubelet_projection_coverage_ratio"]
        )

        S_payload_unaligned = self._build_tubelet_score(
            payload_coded_projections,
            embedding_support,
            attack_strength,
            tubelet_projection_coverage_ratio,
            apply_coverage_penalty=True,
        )
        S_payload_aligned = S_payload_unaligned
        S_payload_rescue_gain = 0.0
        sync_rescue_applied = False
        if self._enabled_evidence.get("sync", False):
            (
                alignment_scores,
                alignment_candidate_metrics,
                search_score_rule,
            ) = self._build_alignment_candidate_scores(
                descriptors,
                tensor_artifact,
                reference_descriptor_map,
                codebook,
                sample,
            )
            sync_result = self._build_sync_search_result(
                alignment_scores,
                mechanism_trace,
                sample,
                search_score_rule,
            )
            sync_result.update(
                self._build_best_alignment_candidate_trace(
                    sync_result,
                    alignment_candidate_metrics,
                )
            )
            sync_result["sync_search_score_rule"] = search_score_rule
            mechanism_trace.update(sync_result)
            aligned_tubelet_projections = self._align_tubelet_projections(
                descriptors,
                tensor_artifact,
                reference_descriptor_map,
                codebook,
                int(sync_result["sync_estimated_offset"]),
                float(sync_result["sync_estimated_scale"]),
            )
            aligned_projection_coverage_ratio = float(
                sync_result.get("sync_alignment_coverage_ratio")
                or tubelet_projection_coverage_ratio
            )
            S_payload_aligned = self._build_tubelet_score(
                aligned_tubelet_projections,
                embedding_support,
                attack_strength,
                aligned_projection_coverage_ratio,
                apply_coverage_penalty=True,
            )
            S_payload_rescue_gain = round(
                max(0.0, S_payload_aligned - S_payload_unaligned),
                6,
            )
            evidence_scores["S_sync"] = self._build_sync_support_score(
                float(sync_result["S_sync_positive_margin"]),
            )
            sync_confidence_trace = self._build_sync_confidence_trace_for_gate_rule(
                sync_result=sync_result,
                S_payload_aligned=S_payload_aligned,
                S_payload_rescue_gain=S_payload_rescue_gain,
            )
            mechanism_trace.update(sync_confidence_trace)
            sync_rescue_applied = bool(sync_confidence_trace["sync_confident"])
        else:
            mechanism_trace.update(
                {
                    "sync_search_enabled": False,
                    "sync_estimated_offset": None,
                    "sync_ground_truth_offset": mechanism_trace.get(
                        "sync_ground_truth_offset"
                    ),
                    "sync_alignment_error": None,
                    "sync_peak_rank": None,
                    "sync_search_space_size": None,
                    "sync_search_space_digest": None,
                    "sync_estimated_scale": None,
                    "sync_ground_truth_scale": self._resolve_ground_truth_scale(sample),
                    "sync_scale_error": None,
                    "sync_alignment_mode": None,
                    "S_sync_peak_best": None,
                    "S_sync_peak_second_or_median": None,
                    "S_sync_peak_margin": None,
                    "S_sync_positive_margin": None,
                    "sync_alignment_matched_count": None,
                    "sync_alignment_candidate_count": None,
                    "sync_alignment_coverage_ratio": None,
                    "sync_candidate_score_raw": None,
                    "sync_candidate_score_penalized": None,
                    "sync_candidate_score_hybrid": None,
                    "sync_confident": False,
                    "sync_confidence_failure_reason": "sync_disabled",
                    "sync_confidence_gate_rule": None,
                    "sync_confidence_min_margin": None,
                    "sync_confidence_min_coverage_ratio": None,
                    "sync_confidence_min_matched_count": None,
                    "sync_confidence_min_candidate_score": None,
                    "sync_confidence_min_payload_rescue_gain": None,
                    "sync_confidence_min_aligned_payload_score": None,
                    "sync_confidence_score_field": None,
                }
            )
        if self._enabled_evidence.get("tubelet", False):
            evidence_scores["S_tubelet"] = S_payload_unaligned
        if self._fusion_rule == "sync_rescue_fusion":
            S_final_before_rescue = round(float(evidence_scores["S_tubelet"] or 0.0), 6)
            evidence_scores["S_final"] = sync_rescue_fusion(
                evidence_scores,
                payload_rescue_gain=S_payload_rescue_gain,
                lambda_sync=self._resolve_lambda_sync(),
                gate_sync=sync_rescue_applied,
            )
        else:
            S_final_before_rescue = round(float(evidence_scores["S_tubelet"] or 0.0), 6)
            evidence_scores["S_final"] = self._fusion_callable(evidence_scores)
        S_final_after_rescue = round(float(evidence_scores["S_final"] or 0.0), 6)
        mechanism_trace.update(
            {
                "S_payload_unaligned": S_payload_unaligned,
                "S_payload_aligned": S_payload_aligned,
                "S_payload_rescue_gain": S_payload_rescue_gain,
                "S_final_before_rescue": S_final_before_rescue,
                "S_final_after_rescue": S_final_after_rescue,
                "rescue_gain_clipped": False,
                "sync_rescue_applied": sync_rescue_applied,
                "fusion_rule": self._fusion_rule,
                "lambda_sync": self._resolve_lambda_sync(),
            }
        )
        validate_evidence_scores(evidence_scores)
        return evidence_scores, mechanism_trace

    def build_sync_candidate_surface(self, sample: LatentSample) -> dict[str, object]:
        if not isinstance(sample, LatentSample):
            raise TypeError("sample must be a LatentSample instance")
        if not self._enabled_evidence.get("sync", False):
            raise ValueError("sync candidate surface requires sync evidence to be enabled")
        if sample.latent_artifact_path is None:
            raise ValueError("sample must carry latent_artifact_path")

        tensor_artifact = read_float_tensor_npy(sample.latent_artifact_path)
        partition_config = build_partition_config_from_method_config(self._method_config)
        descriptors = build_tubelet_descriptors(sample.latent_shape, partition_config)
        (
            _payload_coded_projections,
            codebook,
            reference_descriptor_map,
            reference_shape_trace,
        ) = self._build_payload_coded_projections(
            sample,
            tensor_artifact,
            descriptors,
            partition_config,
        )
        mechanism_trace = dict(sample.mechanism_trace or {})
        mechanism_trace.update(reference_shape_trace)
        (
            alignment_scores,
            alignment_candidate_metrics,
            search_score_rule,
        ) = self._build_alignment_candidate_scores(
            descriptors,
            tensor_artifact,
            reference_descriptor_map,
            codebook,
            sample,
        )
        sync_result = self._build_sync_search_result(
            alignment_scores,
            mechanism_trace,
            sample,
            search_score_rule,
        )
        current_selected_key = (
            int(sync_result["sync_estimated_offset"]),
            round(float(sync_result["sync_estimated_scale"]), 6),
        )
        ground_truth_key = self._resolve_ground_truth_candidate_key(
            mechanism_trace,
            sample,
        )
        candidate_rows = self._build_sync_candidate_surface_rows(
            alignment_candidate_metrics,
            current_selected_key=current_selected_key,
            ground_truth_key=ground_truth_key,
        )

        ranking_specs = (
            ("penalized_prior", "sync_candidate_score_penalized", True),
            ("penalized_no_prior", "sync_candidate_score_penalized", False),
            ("hybrid_prior", "sync_candidate_score_hybrid", True),
            ("hybrid_no_prior", "sync_candidate_score_hybrid", False),
            ("raw_prior", "sync_candidate_score_raw", True),
            ("raw_no_prior", "sync_candidate_score_raw", False),
        )
        ranking_summaries: dict[str, dict[str, object]] = {}
        for rule_name, score_field, center_prior_enabled in ranking_specs:
            ranking_summary, rank_map, winner_key = self._build_alignment_candidate_ranking(
                candidate_rows,
                score_field=score_field,
                center_prior_enabled=center_prior_enabled,
                ground_truth_key=ground_truth_key,
            )
            ranking_summaries[rule_name] = ranking_summary
            for candidate_row in candidate_rows:
                candidate_key = (
                    int(candidate_row["offset_candidate"]),
                    round(float(candidate_row["scale_candidate"]), 6),
                )
                candidate_row[f"rank_{rule_name}"] = rank_map[candidate_key]
                candidate_row[f"selected_{rule_name}"] = candidate_key == winner_key

        selected_candidate_row = next(
            row for row in candidate_rows if bool(row["is_current_selected_candidate"])
        )
        ground_truth_candidate_row = None
        if ground_truth_key is not None:
            ground_truth_candidate_row = next(
                (
                    row
                    for row in candidate_rows
                    if bool(row["is_ground_truth_candidate"])
                ),
                None,
            )

        return {
            "sync_result": dict(sync_result),
            "search_score_rule": search_score_rule,
            "coverage_penalty_enabled": self._coverage_penalty_enabled(),
            "candidate_rows": candidate_rows,
            "selected_candidate": dict(selected_candidate_row),
            "ground_truth_candidate": (
                None if ground_truth_candidate_row is None else dict(ground_truth_candidate_row)
            ),
            "ranking_summaries": ranking_summaries,
        }

    def build_tubelet_anchor_forensics(self, sample: LatentSample) -> dict[str, object]:
        if not isinstance(sample, LatentSample):
            raise TypeError("sample must be a LatentSample instance")
        if sample.latent_artifact_path is None:
            raise ValueError("sample must carry latent_artifact_path")

        tensor_artifact = read_float_tensor_npy(sample.latent_artifact_path)
        partition_config = build_partition_config_from_method_config(self._method_config)
        descriptors = build_tubelet_descriptors(sample.latent_shape, partition_config)
        (
            payload_coded_projections,
            codebook,
            _reference_descriptor_map,
            reference_shape_trace,
        ) = self._build_payload_coded_projections(
            sample,
            tensor_artifact,
            descriptors,
            partition_config,
        )
        embedding_support = self._resolve_embedding_support(sample)
        attack_strength = self._resolve_attack_strength(sample)
        coverage_ratio = float(reference_shape_trace["tubelet_projection_coverage_ratio"])
        score_components = self._build_tubelet_score_components(
            payload_coded_projections,
            embedding_support,
            attack_strength,
            coverage_ratio,
            apply_coverage_penalty=True,
        )
        projection_stats = self._build_payload_projection_stats(
            payload_coded_projections
        )

        return {
            "method_variant": self._method_variant,
            "tubelet_length": partition_config.tubelet_length,
            "spatial_patch_size": list(partition_config.spatial_patch_size),
            "partition_digest": compute_tubelet_partition_digest(
                sample.latent_shape,
                partition_config,
            ),
            "codebook_digest": codebook.codebook_digest,
            "payload_digest": codebook.payload_digest,
            "payload_coded_projection_digest": compute_object_digest(
                [round(float(value), 6) for value in payload_coded_projections]
            ),
            **reference_shape_trace,
            **projection_stats,
            **score_components,
        }

    def _build_payload_coded_projections(
        self,
        sample: LatentSample,
        tensor_artifact: object,
        descriptors: list[object],
        partition_config: object,
    ) -> tuple[list[float], object, dict[tuple[int, int, int], object], dict[str, object]]:
        mechanism_trace = sample.mechanism_trace or {}
        fallback_used = "reference_latent_shape" not in mechanism_trace
        reference_latent_shape = tuple(
            mechanism_trace.get("reference_latent_shape", sample.latent_shape)
        )
        reference_shape_trace = {
            "reference_latent_shape": list(reference_latent_shape),
            "reference_latent_shape_source": (
                "sample_latent_shape_fallback"
                if fallback_used
                else "mechanism_trace"
            ),
            "reference_latent_shape_fallback_used": fallback_used,
        }
        reference_descriptors = build_tubelet_descriptors(
            reference_latent_shape,
            partition_config,
        )
        reference_descriptor_map = self._build_reference_descriptor_map(reference_descriptors)
        codebook = build_tubelet_codebook(
            sample.sample_id,
            reference_descriptors,
            len(reference_descriptors[0].flat_indices),
            self._codebook_config,
            enable_sync=self._enabled_evidence.get("sync", False),
        )
        payload_coded_projections: list[float] = []
        for descriptor in descriptors:
            reference_descriptor = self._resolve_reference_descriptor(
                reference_descriptor_map,
                descriptor,
                estimated_offset=0,
                estimated_scale=1.0,
            )
            if reference_descriptor is None:
                continue
            direction = codebook.directions[reference_descriptor.tubelet_index]
            raw_projection = self._dot_observed_tubelet_direction(
                tensor_artifact,
                descriptor,
                direction,
            )
            if raw_projection is None:
                continue
            payload_coded_projections.append(
                self._clip_score(
                    codebook.payload_codes[reference_descriptor.tubelet_index]
                    * raw_projection
                )
            )
        if not payload_coded_projections:
            raise ValueError("projection extraction produced no valid tubelets")
        tubelet_candidate_count = max(len(reference_descriptor_map), 1)
        tubelet_matched_count = len(payload_coded_projections)
        tubelet_coverage_ratio = min(
            1.0,
            max(0.0, float(tubelet_matched_count) / float(tubelet_candidate_count)),
        )
        reference_shape_trace.update(
            {
                "tubelet_projection_matched_count": tubelet_matched_count,
                "tubelet_projection_candidate_count": tubelet_candidate_count,
                "tubelet_projection_coverage_ratio": round(tubelet_coverage_ratio, 6),
                "tubelet_coverage_penalty_enabled": self._coverage_penalty_enabled(),
            }
        )
        return (
            payload_coded_projections,
            codebook,
            reference_descriptor_map,
            reference_shape_trace,
        )

    def _build_sync_candidate_surface_rows(
        self,
        alignment_candidate_metrics: dict[tuple[int, float], dict[str, float | int]],
        *,
        current_selected_key: tuple[int, float],
        ground_truth_key: tuple[int, float] | None,
    ) -> list[dict[str, float | int | bool]]:
        candidate_rows: list[dict[str, float | int | bool]] = []
        for candidate_key, candidate_metrics in sorted(
            alignment_candidate_metrics.items(),
            key=lambda item: (int(item[0][0]), float(item[0][1])),
        ):
            offset_candidate = int(candidate_key[0])
            scale_candidate = round(float(candidate_key[1]), 6)
            candidate_rows.append(
                {
                    "offset_candidate": offset_candidate,
                    "scale_candidate": scale_candidate,
                    "sync_alignment_matched_count": int(
                        candidate_metrics["sync_alignment_matched_count"]
                    ),
                    "sync_alignment_candidate_count": int(
                        candidate_metrics["sync_alignment_candidate_count"]
                    ),
                    "sync_alignment_coverage_ratio": round(
                        float(candidate_metrics["sync_alignment_coverage_ratio"]),
                        6,
                    ),
                    "sync_candidate_score_raw": round(
                        float(candidate_metrics["sync_candidate_score_raw"]),
                        6,
                    ),
                    "sync_candidate_score_penalized": round(
                        float(candidate_metrics["sync_candidate_score_penalized"]),
                        6,
                    ),
                    "sync_candidate_score_hybrid": round(
                        float(candidate_metrics["sync_candidate_score_hybrid"]),
                        6,
                    ),
                    "is_current_selected_candidate": (
                        offset_candidate == int(current_selected_key[0])
                        and abs(scale_candidate - float(current_selected_key[1])) <= 1e-6
                    ),
                    "is_ground_truth_candidate": (
                        ground_truth_key is not None
                        and offset_candidate == int(ground_truth_key[0])
                        and abs(scale_candidate - float(ground_truth_key[1])) <= 1e-6
                    ),
                }
            )
        return candidate_rows

    def _build_alignment_candidate_ranking(
        self,
        candidate_rows: list[dict[str, float | int | bool]],
        *,
        score_field: str,
        center_prior_enabled: bool,
        ground_truth_key: tuple[int, float] | None,
    ) -> tuple[dict[str, object], dict[tuple[int, float], int], tuple[int, float]]:
        ranked_rows = sorted(
            candidate_rows,
            key=lambda row: (
                -float(row[score_field]),
                abs(int(row["offset_candidate"])) if center_prior_enabled else 0,
                (
                    abs(float(row["scale_candidate"]) - 1.0)
                    if center_prior_enabled
                    else 0.0
                ),
                int(row["offset_candidate"]),
                float(row["scale_candidate"]),
            ),
        )
        rank_map = {
            (
                int(candidate_row["offset_candidate"]),
                round(float(candidate_row["scale_candidate"]), 6),
            ): rank
            for rank, candidate_row in enumerate(ranked_rows, start=1)
        }
        winner_row = ranked_rows[0]
        winner_key = (
            int(winner_row["offset_candidate"]),
            round(float(winner_row["scale_candidate"]), 6),
        )
        winner_score = round(float(winner_row[score_field]), 6)
        second_score = winner_score
        if len(ranked_rows) > 1:
            second_score = round(float(ranked_rows[1][score_field]), 6)
        winner_tie_count = sum(
            1
            for candidate_row in ranked_rows
            if abs(float(candidate_row[score_field]) - winner_score) <= 1e-9
        )
        summary = {
            "score_field": score_field,
            "center_prior_enabled": center_prior_enabled,
            "winner": dict(winner_row),
            "winner_margin_to_second": round(winner_score - second_score, 6),
            "winner_tie_count": winner_tie_count,
            "ground_truth_rank": None,
        }
        if ground_truth_key is not None:
            summary["ground_truth_rank"] = rank_map.get(ground_truth_key)
        return summary, rank_map, winner_key

    def _resolve_ground_truth_candidate_key(
        self,
        mechanism_trace: dict[str, object],
        sample: LatentSample,
    ) -> tuple[int, float] | None:
        ground_truth_offset = mechanism_trace.get("sync_ground_truth_offset")
        if not isinstance(ground_truth_offset, int):
            return None
        if self._scale_search_enabled(sample):
            ground_truth_scale = self._resolve_ground_truth_scale(sample)
            if ground_truth_scale is None:
                return None
            return (int(ground_truth_offset), round(float(ground_truth_scale), 6))
        return (int(ground_truth_offset), 1.0)

    def _build_reference_descriptor_map(
        self,
        reference_descriptors: list[object],
    ) -> dict[tuple[int, int, int], object]:
        return {
            (
                int(descriptor.frame_start),
                int(descriptor.height_start),
                int(descriptor.width_start),
            ): descriptor
            for descriptor in reference_descriptors
        }

    def _resolve_reference_descriptor(
        self,
        reference_descriptor_map: dict[tuple[int, int, int], object],
        descriptor: object,
        estimated_offset: int,
        estimated_scale: float = 1.0,
    ) -> object | None:
        reference_frame_start = int(round(int(descriptor.frame_start) * float(estimated_scale))) - int(
            estimated_offset
        )
        spatial_key = (
            int(descriptor.height_start),
            int(descriptor.width_start),
        )
        exact_key = (
            reference_frame_start,
            spatial_key[0],
            spatial_key[1],
        )
        exact_descriptor = reference_descriptor_map.get(exact_key)
        if exact_descriptor is not None or abs(float(estimated_scale) - 1.0) <= 1e-9:
            return exact_descriptor

        for frame_delta in range(1, self._resolve_scale_search_snap_radius() + 1):
            for candidate_frame_start in (
                reference_frame_start - frame_delta,
                reference_frame_start + frame_delta,
            ):
                candidate_descriptor = reference_descriptor_map.get(
                    (
                        candidate_frame_start,
                        spatial_key[0],
                        spatial_key[1],
                    )
                )
                if candidate_descriptor is not None:
                    return candidate_descriptor
        return None

    def _build_alignment_candidate_scores(
        self,
        descriptors: list[object],
        tensor_artifact: object,
        reference_descriptor_map: dict[tuple[int, int, int], object],
        codebook: object,
        sample: LatentSample,
    ) -> tuple[
        dict[tuple[int, float], float],
        dict[tuple[int, float], dict[str, float | int]],
        str,
    ]:
        search_score_rule = self._resolve_sync_search_score_rule(sample)
        search_score_field = self._resolve_sync_search_score_field(search_score_rule)
        alignment_scores: dict[tuple[int, float], float] = {}
        alignment_candidate_metrics: dict[tuple[int, float], dict[str, float | int]] = {}
        for offset_candidate in self._resolve_offset_candidates(sample):
            for scale_candidate in self._resolve_scale_candidates(sample):
                candidate_key = (offset_candidate, scale_candidate)
                candidate_metrics = self._score_alignment_candidate(
                    descriptors,
                    tensor_artifact,
                    reference_descriptor_map,
                    codebook,
                    offset_candidate,
                    scale_candidate,
                )
                alignment_candidate_metrics[candidate_key] = candidate_metrics
                alignment_scores[candidate_key] = float(candidate_metrics[search_score_field])
        return alignment_scores, alignment_candidate_metrics, search_score_rule

    def _score_alignment_candidate(
        self,
        descriptors: list[object],
        tensor_artifact: object,
        reference_descriptor_map: dict[tuple[int, int, int], object],
        codebook: object,
        offset_candidate: int,
        scale_candidate: float,
    ) -> dict[str, float | int]:
        candidate_payload_projections: list[float] = []
        candidate_count = max(len(reference_descriptor_map), 1)
        for descriptor in descriptors:
            reference_descriptor = self._resolve_reference_descriptor(
                reference_descriptor_map,
                descriptor,
                offset_candidate,
                scale_candidate,
            )
            if reference_descriptor is None:
                continue
            direction = codebook.directions[reference_descriptor.tubelet_index]
            raw_projection = self._dot_observed_tubelet_direction(
                tensor_artifact,
                descriptor,
                direction,
            )
            if raw_projection is None:
                continue
            payload_projection = self._clip_score(
                codebook.payload_codes[reference_descriptor.tubelet_index]
                * raw_projection
            )
            candidate_payload_projections.append(payload_projection)
        if not candidate_payload_projections:
            return {
                "sync_alignment_matched_count": 0,
                "sync_alignment_candidate_count": candidate_count,
                "sync_alignment_coverage_ratio": 0.0,
                "sync_candidate_score_raw": -1.0,
                "sync_candidate_score_penalized": -1.0,
                "sync_candidate_score_hybrid": -1.0,
            }
        projection_support = sum(candidate_payload_projections) / len(
            candidate_payload_projections
        )
        matched_count = len(candidate_payload_projections)
        coverage_ratio = min(1.0, max(0.0, float(matched_count) / float(candidate_count)))
        if self._coverage_penalty_enabled() and projection_support > 0.0:
            penalized_score = projection_support * coverage_ratio
        else:
            penalized_score = projection_support
        hybrid_score = penalized_score
        if self._candidate_supports_hybrid_search(
            matched_count=matched_count,
            coverage_ratio=coverage_ratio,
            projection_support=projection_support,
        ):
            hybrid_score = projection_support
        return {
            "sync_alignment_matched_count": matched_count,
            "sync_alignment_candidate_count": candidate_count,
            "sync_alignment_coverage_ratio": round(coverage_ratio, 6),
            "sync_candidate_score_raw": round(projection_support, 6),
            "sync_candidate_score_penalized": round(penalized_score, 6),
            "sync_candidate_score_hybrid": round(hybrid_score, 6),
        }

    def _candidate_supports_hybrid_search(
        self,
        *,
        matched_count: int,
        coverage_ratio: float,
        projection_support: float,
    ) -> bool:
        if projection_support <= 0.0:
            return False
        min_coverage_ratio = self._resolve_sync_confidence_value(
            "min_sync_alignment_coverage_ratio",
            0.5,
        )
        min_matched_count = int(
            self._resolve_sync_confidence_value("min_sync_alignment_matched_count", 1.0)
        )
        return coverage_ratio >= float(min_coverage_ratio) and matched_count >= min_matched_count

    def _resolve_sync_search_score_rule(self, sample: LatentSample) -> str:
        sync_search_config = self._method_config.get("sync_search", {})
        configured_rule = None
        if isinstance(sync_search_config, dict):
            configured_rule = sync_search_config.get("search_score_rule")
        if configured_rule is not None:
            normalized_rule = str(configured_rule).strip()
            if normalized_rule not in {
                "penalized_prior",
                "penalized_no_prior",
                "raw_prior",
                "raw_no_prior",
                "hybrid_prior",
                "hybrid_no_prior",
            }:
                raise ValueError(
                    "sync_search.search_score_rule must be one of: penalized_prior, penalized_no_prior, raw_prior, raw_no_prior, hybrid_prior, hybrid_no_prior"
                )
            return normalized_rule
        if self._coverage_penalty_enabled() and self._is_local_clip_sample(sample):
            return "hybrid_no_prior"
        return "penalized_prior"

    def _resolve_sync_search_score_field(self, search_score_rule: str) -> str:
        return {
            "penalized_prior": "sync_candidate_score_penalized",
            "penalized_no_prior": "sync_candidate_score_penalized",
            "raw_prior": "sync_candidate_score_raw",
            "raw_no_prior": "sync_candidate_score_raw",
            "hybrid_prior": "sync_candidate_score_hybrid",
            "hybrid_no_prior": "sync_candidate_score_hybrid",
        }[search_score_rule]

    def _sync_search_uses_center_prior(self, search_score_rule: str) -> bool:
        return not str(search_score_rule).endswith("_no_prior")

    def _is_local_clip_sample(self, sample: LatentSample) -> bool:
        applied_attack_params = sample.applied_attack_params or {}
        if isinstance(applied_attack_params, dict) and "clip_length" in applied_attack_params:
            return True
        mechanism_trace = sample.mechanism_trace or {}
        return isinstance(mechanism_trace, dict) and mechanism_trace.get("clip_length") is not None

    def _build_best_alignment_candidate_trace(
        self,
        sync_result: dict[str, object],
        alignment_candidate_metrics: dict[tuple[int, float], dict[str, float | int]],
    ) -> dict[str, float | int | None]:
        best_key = (
            int(sync_result["sync_estimated_offset"]),
            float(sync_result["sync_estimated_scale"]),
        )
        best_metrics = alignment_candidate_metrics.get(best_key)
        if best_metrics is None:
            return {
                "sync_alignment_matched_count": None,
                "sync_alignment_candidate_count": None,
                "sync_alignment_coverage_ratio": None,
                "sync_candidate_score_raw": None,
                "sync_candidate_score_penalized": None,
                "sync_candidate_score_hybrid": None,
            }
        return dict(best_metrics)

    def _build_sync_confidence_trace_for_gate_rule(
        self,
        *,
        sync_result: dict[str, object],
        S_payload_aligned: float,
        S_payload_rescue_gain: float,
    ) -> dict[str, object]:
        self._resolve_sync_confidence_gate_rule()
        return self._build_aligned_payload_safety_confidence_trace(
            sync_result=sync_result,
            S_payload_aligned=S_payload_aligned,
            S_payload_rescue_gain=S_payload_rescue_gain,
        )

    def _resolve_sync_confidence_gate_rule(self) -> str:
        sync_search_config = self._method_config.get("sync_search", {})
        gate_rule = "aligned_payload_safety_gate"
        if isinstance(sync_search_config, dict):
            configured_gate_rule = sync_search_config.get("sync_confidence_gate_rule")
            if configured_gate_rule is not None:
                gate_rule = str(configured_gate_rule).strip()
        if gate_rule != "aligned_payload_safety_gate":
            raise ValueError(
                "sync_search.sync_confidence_gate_rule must be aligned_payload_safety_gate"
            )
        return gate_rule

    def _build_aligned_payload_safety_confidence_trace(
        self,
        *,
        sync_result: dict[str, object],
        S_payload_aligned: float,
        S_payload_rescue_gain: float,
    ) -> dict[str, object]:
        min_rescue_gain = self._resolve_sync_confidence_value(
            "min_payload_rescue_gain",
            0.01,
        )
        min_aligned_payload_score = self._resolve_sync_confidence_value(
            "min_aligned_payload_score",
            0.1,
        )
        min_coverage_ratio = self._resolve_sync_confidence_value(
            "min_sync_alignment_coverage_ratio",
            0.125,
        )
        min_matched_count = int(
            self._resolve_sync_confidence_value(
                "min_sync_alignment_matched_count",
                64.0,
            )
        )
        coverage_ratio = float(sync_result.get("sync_alignment_coverage_ratio") or 0.0)
        matched_count = int(sync_result.get("sync_alignment_matched_count") or 0)
        failure_reasons: list[str] = []
        if float(S_payload_rescue_gain) < float(min_rescue_gain):
            failure_reasons.append("payload_rescue_gain_below_gate")
        if float(S_payload_aligned) < float(min_aligned_payload_score):
            failure_reasons.append("aligned_payload_score_below_gate")
        if coverage_ratio < float(min_coverage_ratio):
            failure_reasons.append("sync_coverage_below_gate")
        if matched_count < min_matched_count:
            failure_reasons.append("sync_matched_count_below_gate")
        return {
            "sync_confident": not failure_reasons,
            "sync_confidence_failure_reason": ";".join(failure_reasons) or None,
            "sync_confidence_gate_rule": "aligned_payload_safety_gate",
            "sync_confidence_min_margin": None,
            "sync_confidence_min_coverage_ratio": round(float(min_coverage_ratio), 6),
            "sync_confidence_min_matched_count": min_matched_count,
            "sync_confidence_min_payload_rescue_gain": round(float(min_rescue_gain), 6),
            "sync_confidence_min_aligned_payload_score": round(
                float(min_aligned_payload_score),
                6,
            ),
            "sync_confidence_score_field": "S_payload_aligned",
        }

    def _build_sync_search_result(
        self,
        alignment_scores: dict[tuple[int, float], float],
        mechanism_trace: dict[str, object],
        sample: LatentSample,
        search_score_rule: str,
    ) -> dict[str, float | int | str | None | bool]:
        ground_truth_offset = mechanism_trace.get("sync_ground_truth_offset")
        resolved_ground_truth_offset = (
            int(ground_truth_offset) if isinstance(ground_truth_offset, int) else None
        )
        ground_truth_scale = self._resolve_ground_truth_scale_from_trace(mechanism_trace)
        center_prior_enabled = self._sync_search_uses_center_prior(search_score_rule)
        if self._scale_search_enabled(sample):
            return build_offset_scale_search_result(
                alignment_scores,
                ground_truth_offset=resolved_ground_truth_offset,
                ground_truth_scale=ground_truth_scale,
                center_prior_enabled=center_prior_enabled,
            )

        offset_scores = {
            int(offset_candidate): float(candidate_score)
            for (offset_candidate, scale_candidate), candidate_score in alignment_scores.items()
            if abs(float(scale_candidate) - 1.0) <= 1e-9
        }
        return build_offset_search_result(
            offset_scores,
            ground_truth_offset=resolved_ground_truth_offset,
            center_prior_enabled=center_prior_enabled,
        )

    def _build_auxiliary_sync_score(self, coded_projections: list[float]) -> float:
        return self._clip_score(median(coded_projections))

    def _resolve_embedding_support(self, sample: LatentSample) -> float:
        mechanism_trace = sample.mechanism_trace or {}
        projection_before = mechanism_trace.get("mean_projection_before")
        projection_after = mechanism_trace.get("mean_projection_after")
        if projection_before is None or projection_after is None:
            return 0.0
        return max(
            0.0,
            min(1.0, float(projection_after) - float(projection_before)),
        )

    def _resolve_attack_strength(self, sample: LatentSample) -> float:
        applied_attack_params = sample.applied_attack_params or {}
        original_frame_count = applied_attack_params.get("original_frame_count")
        observed_frame_count = applied_attack_params.get("observed_frame_count")
        if (
            not isinstance(original_frame_count, int)
            or original_frame_count < 1
            or not isinstance(observed_frame_count, int)
            or observed_frame_count < 0
        ):
            return 0.0
        return max(
            0.0,
            min(1.0, 1.0 - (float(observed_frame_count) / float(original_frame_count))),
        )

    def _build_tubelet_score(
        self,
        aligned_tubelet_projections: list[float],
        embedding_support: float,
        attack_strength: float,
        coverage_ratio: float,
        *,
        apply_coverage_penalty: bool,
    ) -> float:
        return float(
            self._build_tubelet_score_components(
                aligned_tubelet_projections,
                embedding_support,
                attack_strength,
                coverage_ratio,
                apply_coverage_penalty=apply_coverage_penalty,
            )["S_tubelet_recomputed"]
        )

    def _build_tubelet_score_components(
        self,
        aligned_tubelet_projections: list[float],
        embedding_support: float,
        attack_strength: float,
        coverage_ratio: float,
        *,
        apply_coverage_penalty: bool,
    ) -> dict[str, float | bool]:
        del attack_strength
        base_score = sum(aligned_tubelet_projections) / len(aligned_tubelet_projections)
        coverage_multiplier = 1.0
        if apply_coverage_penalty and self._coverage_penalty_enabled() and base_score > 0.0:
            coverage_multiplier = max(0.0, min(1.0, float(coverage_ratio)))
        base_score_after_coverage = base_score * coverage_multiplier
        projection_support_weight = self._resolve_score_calibration_value(
            "embedding_projection_support_weight",
        )
        support_score = float(embedding_support) * projection_support_weight
        return {
            "tubelet_score_base_before_coverage": round(float(base_score), 6),
            "tubelet_score_coverage_multiplier": round(float(coverage_multiplier), 6),
            "tubelet_score_base_after_coverage": round(
                float(base_score_after_coverage),
                6,
            ),
            "embedding_support": round(float(embedding_support), 6),
            "embedding_projection_support_weight": round(
                float(projection_support_weight),
                6,
            ),
            "embedding_support_score": round(float(support_score), 6),
            "tubelet_score_coverage_penalty_applied": bool(
                apply_coverage_penalty
                and self._coverage_penalty_enabled()
                and base_score > 0.0
            ),
            "S_tubelet_recomputed": self._clip_score(
                base_score_after_coverage + support_score
            ),
        }

    def _build_payload_projection_stats(
        self,
        payload_coded_projections: list[float],
    ) -> dict[str, float | int]:
        sorted_projections = sorted(float(value) for value in payload_coded_projections)
        projection_count = len(sorted_projections)
        positive_count = sum(1 for value in sorted_projections if value > 0.0)
        negative_count = sum(1 for value in sorted_projections if value < 0.0)
        zero_count = projection_count - positive_count - negative_count
        return {
            "payload_projection_count": projection_count,
            "payload_projection_mean": round(
                sum(sorted_projections) / projection_count,
                6,
            ),
            "payload_projection_median": round(float(median(sorted_projections)), 6),
            "payload_projection_min": round(float(sorted_projections[0]), 6),
            "payload_projection_max": round(float(sorted_projections[-1]), 6),
            "payload_projection_positive_count": positive_count,
            "payload_projection_negative_count": negative_count,
            "payload_projection_zero_count": zero_count,
            "payload_projection_positive_fraction": round(
                float(positive_count) / float(projection_count),
                6,
            ),
            "payload_projection_negative_fraction": round(
                float(negative_count) / float(projection_count),
                6,
            ),
        }

    def _resolve_score_calibration_value(self, field_name: str) -> float:
        score_calibration = self._method_config.get("score_calibration", {})
        if not isinstance(score_calibration, dict):
            return 0.0
        field_value = score_calibration.get(field_name, 0.0)
        if not isinstance(field_value, (int, float)):
            return 0.0
        return float(field_value)

    def _build_sync_support_score(
        self,
        sync_positive_margin: float,
    ) -> float | None:
        return self._clip_score(max(0.0, float(sync_positive_margin)))

    def _align_tubelet_projections(
        self,
        descriptors: list[object],
        tensor_artifact: object,
        reference_descriptor_map: dict[tuple[int, int, int], object],
        codebook: object,
        estimated_offset: int,
        estimated_scale: float,
    ) -> list[float]:
        aligned_projections: list[float] = []
        for descriptor in descriptors:
            reference_descriptor = self._resolve_reference_descriptor(
                reference_descriptor_map,
                descriptor,
                estimated_offset,
                estimated_scale,
            )
            if reference_descriptor is None:
                continue
            direction = codebook.directions[reference_descriptor.tubelet_index]
            raw_projection = self._dot_observed_tubelet_direction(
                tensor_artifact,
                descriptor,
                direction,
            )
            if raw_projection is None:
                continue
            aligned_projections.append(
                self._clip_score(
                    codebook.payload_codes[reference_descriptor.tubelet_index]
                    * raw_projection
                )
            )
        if not aligned_projections:
            return [-1.0]
        return aligned_projections

    def _resolve_offset_candidates(self, sample: LatentSample | None = None) -> list[int]:
        sync_search_config = self._resolve_sync_search_config()
        offset_min = int(sync_search_config.get("offset_search_min", -16))
        offset_max = int(sync_search_config.get("offset_search_max", 16))
        if offset_min > offset_max:
            raise ValueError("sync_search offset_search_min must not exceed offset_search_max")
        if sample is not None and not self._scale_search_enabled(sample):
            reference_frame_count = self._resolve_reference_frame_count_for_sync_search(sample)
            observed_frame_count = self._resolve_observed_frame_count_for_sync_search(sample)
            if (
                reference_frame_count is not None
                and observed_frame_count is not None
                and reference_frame_count > observed_frame_count
            ):
                offset_min = min(
                    offset_min,
                    int(observed_frame_count) - int(reference_frame_count),
                )
        return list(range(offset_min, offset_max + 1))

    def _resolve_reference_frame_count_for_sync_search(
        self,
        sample: LatentSample,
    ) -> int | None:
        mechanism_trace = sample.mechanism_trace or {}
        reference_latent_shape = mechanism_trace.get("reference_latent_shape")
        if (
            isinstance(reference_latent_shape, (list, tuple))
            and reference_latent_shape
            and isinstance(reference_latent_shape[0], int)
            and int(reference_latent_shape[0]) > 0
        ):
            return int(reference_latent_shape[0])
        applied_attack_params = sample.applied_attack_params or {}
        original_frame_count = applied_attack_params.get("original_frame_count")
        if isinstance(original_frame_count, int) and original_frame_count > 0:
            return int(original_frame_count)
        return self._resolve_observed_frame_count_for_sync_search(sample)

    def _resolve_observed_frame_count_for_sync_search(
        self,
        sample: LatentSample,
    ) -> int | None:
        if sample.latent_shape and isinstance(sample.latent_shape[0], int):
            observed_frame_count = int(sample.latent_shape[0])
            if observed_frame_count > 0:
                return observed_frame_count
        applied_attack_params = sample.applied_attack_params or {}
        observed_frame_count = applied_attack_params.get("observed_frame_count")
        if isinstance(observed_frame_count, int) and observed_frame_count > 0:
            return int(observed_frame_count)
        return None

    def _resolve_scale_candidates(self, sample: LatentSample | None = None) -> list[float]:
        if not self._scale_search_enabled(sample):
            return [1.0]
        sync_search_config = self._resolve_sync_search_config()
        scale_candidates = sync_search_config.get("scale_candidates", [1.0])
        if not isinstance(scale_candidates, list) or not scale_candidates:
            raise ValueError("sync_search scale_candidates must be a non-empty list")
        normalized_candidates = sorted(
            {
                round(float(scale_candidate), 6)
                for scale_candidate in scale_candidates
                if isinstance(scale_candidate, (int, float))
                and float(scale_candidate) > 0.0
            }
        )
        if 1.0 not in normalized_candidates:
            normalized_candidates.append(1.0)
            normalized_candidates.sort()
        normalized_candidates = self._filter_scale_candidates_for_speed_change(
            normalized_candidates,
            sample,
        )
        if not normalized_candidates:
            raise ValueError("sync_search scale_candidates must contain positive numbers")
        return normalized_candidates

    def _filter_scale_candidates_for_speed_change(
        self,
        scale_candidates: list[float],
        sample: LatentSample | None,
    ) -> list[float]:
        if sample is None:
            return scale_candidates
        attack_params = sample.applied_attack_params or {}
        speed_ratio = attack_params.get("speed_ratio")
        if not isinstance(speed_ratio, (int, float)) or float(speed_ratio) <= 0.0:
            return scale_candidates
        normalized_speed_ratio = round(float(speed_ratio), 6)
        if abs(normalized_speed_ratio - 1.0) <= 1e-9:
            return [1.0]
        if normalized_speed_ratio > 1.0:
            filtered_candidates = [
                scale_candidate
                for scale_candidate in scale_candidates
                if float(scale_candidate) >= 1.0
            ]
        else:
            filtered_candidates = [
                scale_candidate
                for scale_candidate in scale_candidates
                if float(scale_candidate) <= 1.0
            ]
        return filtered_candidates or [1.0]

    def _scale_search_enabled(self, sample: LatentSample | None = None) -> bool:
        sync_search_config = self._resolve_sync_search_config()
        if not bool(sync_search_config.get("enable_scale_search", False)):
            return False
        if sample is None:
            return True
        attack_params = sample.applied_attack_params or {}
        return "speed_ratio" in attack_params

    def _resolve_sync_search_config(self) -> dict[str, object]:
        sync_search_config = self._method_config.get("sync_search", {})
        if not isinstance(sync_search_config, dict):
            return {}
        return sync_search_config

    def _coverage_penalty_enabled(self) -> bool:
        sync_search_config = self._resolve_sync_search_config()
        return bool(sync_search_config.get("coverage_penalty_enabled", True))

    def _resolve_sync_confidence_value(self, field_name: str, default_value: float) -> float:
        sync_search_config = self._resolve_sync_search_config()
        field_value = sync_search_config.get(field_name, default_value)
        if not isinstance(field_value, (int, float)):
            return float(default_value)
        return float(field_value)

    def _resolve_scale_search_snap_radius(self) -> int:
        sync_search_config = self._resolve_sync_search_config()
        snap_radius = sync_search_config.get("scale_search_snap_radius", 3)
        if not isinstance(snap_radius, int) or snap_radius < 0:
            return 3
        return snap_radius

    def _resolve_ground_truth_scale(self, sample: LatentSample) -> float | None:
        mechanism_trace = sample.mechanism_trace or {}
        return self._resolve_ground_truth_scale_from_trace(mechanism_trace)

    def _resolve_ground_truth_scale_from_trace(
        self,
        mechanism_trace: dict[str, object],
    ) -> float | None:
        if "sync_ground_truth_scale" not in mechanism_trace:
            return 1.0
        ground_truth_scale = mechanism_trace.get("sync_ground_truth_scale")
        if isinstance(ground_truth_scale, (int, float)):
            return round(float(ground_truth_scale), 6)
        return None

    def _resolve_lambda_sync(self) -> float:
        lambda_sync = self._method_config.get("lambda_sync", 0.1)
        if not isinstance(lambda_sync, (int, float)):
            return 0.1
        return round(float(lambda_sync), 6)

    def _dot_observed_tubelet_direction(
        self,
        tensor_artifact: object,
        descriptor: object,
        direction: tuple[float, ...],
    ) -> float | None:
        if len(descriptor.flat_indices) == len(direction):
            return dot_tubelet_direction(tensor_artifact, descriptor, direction)
        if len(descriptor.flat_indices) > len(direction):
            return None
        return sum(
            float(tensor_artifact.values[flat_index]) * float(direction_value)
            for flat_index, direction_value in zip(
                descriptor.flat_indices,
                direction[: len(descriptor.flat_indices)],
            )
        )

    def _clip_score(self, score: float) -> float:
        return round(max(-1.0, min(1.0, score)), 6)
