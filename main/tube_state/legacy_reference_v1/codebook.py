
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

from .digest import compute_object_digest
from .tubelet_partition import TubeletDescriptor


@dataclass(frozen=True)
class CodebookConfig:

    direction_key: str = "direction_key_synthetic_probe"
    payload_key: str = "payload_key_synthetic_probe"
    sync_key: str = "sync_key_synthetic_probe"
    payload_bits: int = 64
    sync_code_family: str = "keyed_binary_low_correlation"
    direction_normalization: str = "l2"


@dataclass(frozen=True)
class TubeletCodebook:

    directions: dict[int, tuple[float, ...]]
    payload_codes: dict[int, int]
    sync_codes: dict[int, int]
    combined_codes: dict[int, int]
    direction_norm_squares: dict[int, float]
    codebook_digest: str
    sync_code_digest: str
    payload_digest: str


@lru_cache(maxsize=1)
def build_codebook_config() -> CodebookConfig:
    return CodebookConfig()


def build_tubelet_codebook(
    sample_id: str,
    tubelet_descriptors: list[TubeletDescriptor],
    vector_length: int | None,
    codebook_config: CodebookConfig,
    enable_sync: bool,
    reference_temporal_indices: list[int] | None = None,
) -> TubeletCodebook:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(tubelet_descriptors, list) or not tubelet_descriptors:
        raise ValueError("tubelet_descriptors must be a non-empty list")
    if vector_length is not None and (
        not isinstance(vector_length, int) or vector_length < 1
    ):
        raise ValueError("vector_length must be a positive integer")
    if not isinstance(codebook_config, CodebookConfig):
        raise TypeError("codebook_config must be a CodebookConfig instance")

    normalized_reference_temporal_indices = (
        tuple(sorted({int(temporal_index) for temporal_index in reference_temporal_indices}))
        if reference_temporal_indices is not None
        else None
    )
    descriptor_signature = _build_descriptor_signature(tubelet_descriptors)
    return _build_cached_tubelet_codebook(
        sample_id,
        descriptor_signature,
        codebook_config,
        bool(enable_sync),
        normalized_reference_temporal_indices,
    )


def _build_descriptor_signature(
    tubelet_descriptors: list[TubeletDescriptor],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            descriptor.temporal_index,
            descriptor.tubelet_index,
            len(descriptor.flat_indices),
        )
        for descriptor in tubelet_descriptors
    )


@lru_cache(maxsize=128)
def _build_cached_tubelet_codebook(
    sample_id: str,
    descriptor_signature: tuple[tuple[int, int, int], ...],
    codebook_config: CodebookConfig,
    enable_sync: bool,
    reference_temporal_indices: tuple[int, ...] | None,
) -> TubeletCodebook:
    temporal_indices = (
        list(reference_temporal_indices)
        if reference_temporal_indices is not None
        else sorted({temporal_index for temporal_index, _, _ in descriptor_signature})
    )
    sync_codes = {
        temporal_index: _build_binary_code(
            codebook_config.sync_key,
            sample_id,
            temporal_index,
        )
        for temporal_index in temporal_indices
    }
    payload_codes: dict[int, int] = {}
    combined_codes: dict[int, int] = {}
    directions: dict[int, tuple[float, ...]] = {}
    direction_norm_squares: dict[int, float] = {}
    direction_seed_payload: list[dict[str, int]] = []

    for temporal_index, tubelet_index, descriptor_vector_length in descriptor_signature:
        payload_code = _build_binary_code(
            codebook_config.payload_key,
            sample_id,
            tubelet_index,
        )
        direction_seed = _build_integer_seed(
            codebook_config.direction_key,
            sample_id,
            tubelet_index,
        )
        payload_codes[tubelet_index] = payload_code
        combined_codes[tubelet_index] = payload_code * (
            sync_codes[temporal_index] if enable_sync else 1
        )
        direction = _build_normalized_direction(
            direction_seed,
            descriptor_vector_length,
            codebook_config.direction_normalization,
        )
        directions[tubelet_index] = direction
        direction_norm_squares[tubelet_index] = round(
            sum(direction_value * direction_value for direction_value in direction),
            8,
        )
        direction_seed_payload.append(
            {
                "tubelet_index": tubelet_index,
                "temporal_index": temporal_index,
                "direction_seed": direction_seed,
                "vector_length": descriptor_vector_length,
            }
        )

    return TubeletCodebook(
        directions=directions,
        payload_codes=payload_codes,
        sync_codes=sync_codes,
        combined_codes=combined_codes,
        direction_norm_squares=direction_norm_squares,
        codebook_digest=compute_object_digest(direction_seed_payload),
        sync_code_digest=compute_object_digest(sync_codes),
        payload_digest=compute_object_digest(payload_codes),
    )


def _build_integer_seed(key: str, sample_id: str, index_value: int) -> int:
    digest = compute_object_digest(
        {
            "key": key,
            "sample_id": sample_id,
            "index_value": index_value,
        }
    )
    return int(digest[:16], 16)


def _build_binary_code(key: str, sample_id: str, index_value: int) -> int:
    seed_value = _build_integer_seed(key, sample_id, index_value)
    return 1 if seed_value % 2 == 0 else -1


def _build_normalized_direction(
    seed_value: int,
    vector_length: int,
    normalization_rule: str,
) -> tuple[float, ...]:
    if normalization_rule != "l2":
        raise ValueError(f"unsupported direction_normalization: {normalization_rule}")
    generator = random.Random(seed_value)
    values = [generator.uniform(-1.0, 1.0) for _ in range(vector_length)]
    l2_norm = math.sqrt(sum(value * value for value in values))
    if l2_norm == 0.0:
        raise ValueError("direction generation produced a zero vector")
    return tuple(round(value / l2_norm, 8) for value in values)