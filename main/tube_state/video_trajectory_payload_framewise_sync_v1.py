"""Public-coordinate framewise latent trajectory sync candidate.

The receiver accepts only a received framewise latent, a key, and this public
protocol. Experiment truth (including the fixed crop start) is deliberately
absent. Every public offset and every intersecting source tubelet is retained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PublicProtocol:
    method_version: str = "trajectory-payload-framewise-sync-v1"
    source_frames: int = 181
    received_frame_counts: tuple[int, ...] = (181, 177)
    source_height: int = 320
    source_width: int = 512
    fps: int = 8
    framewise_downsample: int = 8
    latent_channels: int = 4
    tubelet_length: int = 4
    patch_height: int = 4
    patch_width: int = 4
    scaled_latent_factor: float = 0.18215
    embedding_margin: float = 1.0
    tie_atol: float = 1e-12

    @property
    def latent_height(self) -> int:
        return self.source_height // self.framewise_downsample

    @property
    def latent_width(self) -> int:
        return self.source_width // self.framewise_downsample


PUBLIC = PublicProtocol()


def _require_public(public: PublicProtocol) -> None:
    if not isinstance(public, PublicProtocol):
        raise TypeError("public must be a PublicProtocol")
    if public.source_frames < 1 or public.source_frames not in public.received_frame_counts:
        raise ValueError("public source frame count must be an allowed received length")
    if not public.received_frame_counts or any(
        value < 1 or value > public.source_frames for value in public.received_frame_counts
    ):
        raise ValueError("invalid public received frame counts")
    if any((value - 1) % 4 for value in public.received_frame_counts):
        raise ValueError("received lengths must preserve the original Wan 1+4k support")
    if public.source_height % public.framewise_downsample:
        raise ValueError("source height must match framewise VAE downsample")
    if public.source_width % public.framewise_downsample:
        raise ValueError("source width must match framewise VAE downsample")
    if public.latent_height % public.patch_height:
        raise ValueError("latent height must tile into fixed spatial patches")
    if public.latent_width % public.patch_width:
        raise ValueError("latent width must tile into fixed spatial patches")
    if public.latent_channels != 4 or public.tubelet_length != 4:
        raise ValueError("fixed channel/tubelet geometry changed")
    if public.scaled_latent_factor != 0.18215 or public.embedding_margin != 1.0:
        raise ValueError("fixed scaled-latent factor or margin changed")
    if public.tie_atol < 0.0:
        raise ValueError("tie_atol must be nonnegative")


def public_receipt(public: PublicProtocol = PUBLIC) -> dict[str, Any]:
    _require_public(public)
    row = asdict(public)
    row.update(
        framewise_source_latent_shape=[
            public.source_frames,
            public.latent_channels,
            public.latent_height,
            public.latent_width,
        ],
        candidate_rule="range(source_frames - received_frames + 1)",
        full_singleton_interpretation="public geometry only; not synchronization evidence",
        score_rule=(
            "S(o)=sum_b,p s_b*<z_A,d_A>/sum_b,p ||d_A||^2; "
            "full public direction is sliced but never renormalized"
        ),
        codebook_inputs=(
            "key and fixed public source coordinates only; no sample_id, message, "
            "truth, true crop start, writer trace, or paired control"
        ),
    )
    return row


def key_identifier(key: str) -> str:
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def candidate_offsets(received_frames: int, public: PublicProtocol = PUBLIC) -> tuple[int, ...]:
    _require_public(public)
    if int(received_frames) not in public.received_frame_counts:
        raise ValueError("received frame count is outside the fixed public protocol")
    return tuple(range(public.source_frames - int(received_frames) + 1))


def source_frame_indices(
    received_frames: int,
    source_offset: int,
    public: PublicProtocol = PUBLIC,
) -> tuple[int, ...]:
    offsets = candidate_offsets(received_frames, public)
    if source_offset not in offsets:
        raise ValueError("source offset is outside the public candidate domain")
    return tuple(range(source_offset, source_offset + received_frames))


def source_tubelet_frame_count(tubelet_index: int, public: PublicProtocol = PUBLIC) -> int:
    _require_public(public)
    start = int(tubelet_index) * public.tubelet_length
    if start < 0 or start >= public.source_frames:
        raise ValueError("source tubelet index is outside the public source")
    return min(public.tubelet_length, public.source_frames - start)


def tubelet_support_rows(
    received_frames: int,
    source_offset: int,
    public: PublicProtocol = PUBLIC,
) -> list[dict[str, Any]]:
    source_indices = source_frame_indices(received_frames, source_offset, public)
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for received_index, source_index in enumerate(source_indices):
        tubelet_index = source_index // public.tubelet_length
        grouped.setdefault(tubelet_index, []).append(
            (received_index, source_index, source_index % public.tubelet_length)
        )
    rows: list[dict[str, Any]] = []
    for tubelet_index in sorted(grouped):
        triples = grouped[tubelet_index]
        full_frames = source_tubelet_frame_count(tubelet_index, public)
        rows.append(
            {
                "source_tubelet": tubelet_index,
                "source_start": tubelet_index * public.tubelet_length,
                "source_frame_count": full_frames,
                "received_indices": [value[0] for value in triples],
                "source_indices": [value[1] for value in triples],
                "source_ages": [value[2] for value in triples],
                "observed_frames": len(triples),
                "support_fraction": len(triples) / full_frames,
                "complete_source_support": len(triples) == full_frames,
            }
        )
    return rows


def spatial_patch_coordinates(
    latent_height: int,
    latent_width: int,
    public: PublicProtocol = PUBLIC,
) -> tuple[tuple[int, int], ...]:
    _require_public(public)
    if latent_height != public.latent_height or latent_width != public.latent_width:
        raise ValueError("framewise latent spatial shape does not match public protocol")
    return tuple(
        (y, x)
        for y in range(0, latent_height, public.patch_height)
        for x in range(0, latent_width, public.patch_width)
    )


def _stream_seed(key: str, role: str, coordinates: tuple[int, ...], public: PublicProtocol) -> bytes:
    key_identifier(key)
    fields = [public.method_version, role, key, *(str(value) for value in coordinates)]
    return hashlib.sha256(b"\x00".join(value.encode("utf-8") for value in fields)).digest()


def sync_sign(key: str, source_tubelet: int, public: PublicProtocol = PUBLIC) -> int:
    source_tubelet_frame_count(source_tubelet, public)
    return 1 if _stream_seed(key, "sync-sign", (source_tubelet,), public)[0] & 1 else -1


@lru_cache(maxsize=32768)
def _cached_patch_direction(
    key: str,
    source_tubelet: int,
    patch_y: int,
    patch_x: int,
    public: PublicProtocol,
) -> np.ndarray:
    frame_count = source_tubelet_frame_count(source_tubelet, public)
    element_count = frame_count * public.latent_channels * public.patch_height * public.patch_width
    seed = _stream_seed(
        key,
        "direction-stream",
        (source_tubelet, patch_y, patch_x, frame_count),
        public,
    )
    raw = bytearray()
    counter = 0
    while len(raw) < element_count:
        raw.extend(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest())
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(raw), dtype=np.uint8), bitorder="little")[:element_count]
    values = np.where(bits, np.float32(1.0), np.float32(-1.0)).reshape(
        frame_count,
        public.latent_channels,
        public.patch_height,
        public.patch_width,
    )
    values = np.asarray(values / np.float32(math.sqrt(element_count)), dtype=np.float32)
    values.setflags(write=False)
    return values


def patch_direction(
    key: str,
    source_tubelet: int,
    patch_y: int,
    patch_x: int,
    public: PublicProtocol = PUBLIC,
) -> np.ndarray:
    _require_public(public)
    if (patch_y, patch_x) not in spatial_patch_coordinates(
        public.latent_height, public.latent_width, public
    ):
        raise ValueError("patch coordinate is outside the public latent grid")
    return _cached_patch_direction(key, source_tubelet, patch_y, patch_x, public)


def validate_source_latent(latent: np.ndarray, public: PublicProtocol = PUBLIC) -> np.ndarray:
    _require_public(public)
    value = np.asarray(latent)
    expected = (
        public.source_frames,
        public.latent_channels,
        public.latent_height,
        public.latent_width,
    )
    if value.shape != expected:
        raise ValueError(f"source framewise latent must have shape {expected}")
    if value.dtype != np.float32:
        raise TypeError("method source latent must use float32 scaled-latent coordinates")
    if not bool(np.isfinite(value).all()):
        raise ValueError("source framewise latent contains nonfinite values")
    return value


def validate_received_latent(latent: np.ndarray, public: PublicProtocol = PUBLIC) -> np.ndarray:
    _require_public(public)
    value = np.asarray(latent)
    if value.dtype != np.float32:
        raise TypeError("receiver latent must use float32 scaled-latent coordinates")
    if value.ndim != 4:
        raise ValueError("receiver latent must be [T,C,H,W]")
    if int(value.shape[0]) not in public.received_frame_counts:
        raise ValueError("receiver frame count is outside the public protocol")
    if tuple(value.shape[1:]) != (
        public.latent_channels,
        public.latent_height,
        public.latent_width,
    ):
        raise ValueError("receiver latent channel/spatial geometry mismatch")
    return value


def _projection(block: np.ndarray, direction: np.ndarray, sign: int) -> tuple[float, float]:
    raw = float(np.sum(block.astype(np.float64) * direction.astype(np.float64)))
    return raw, float(sign * raw)


def apply_projection_margin(
    latent: np.ndarray,
    key: str,
    public: PublicProtocol = PUBLIC,
    *,
    target_margin: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one adopted writer-only projection target and measure float32 delta."""

    target_margin = float(target_margin)
    if target_margin not in (0.5, 1.0):
        raise ValueError("target_margin must be one of the two adopted fixed targets")
    source = validate_source_latent(latent, public)
    output = source.copy()
    rows: list[dict[str, Any]] = []
    raw_energy = 0.0
    applied_energy = 0.0
    patch_coordinates = spatial_patch_coordinates(public.latent_height, public.latent_width, public)
    tubelet_count = math.ceil(public.source_frames / public.tubelet_length)
    for tubelet_index in range(tubelet_count):
        start = tubelet_index * public.tubelet_length
        stop = min(start + public.tubelet_length, public.source_frames)
        sign = sync_sign(key, tubelet_index, public)
        for patch_y, patch_x in patch_coordinates:
            direction = patch_direction(key, tubelet_index, patch_y, patch_x, public)
            target = output[
                start:stop,
                :,
                patch_y : patch_y + public.patch_height,
                patch_x : patch_x + public.patch_width,
            ]
            before = target.copy()
            _, signed_before = _projection(before, direction, sign)
            raw_delta = max(0.0, target_margin - signed_before)
            if raw_delta:
                target[...] = target + np.float32(raw_delta * sign) * direction
            raw_after, signed_after = _projection(target, direction, sign)
            applied = target.astype(np.float64) - before.astype(np.float64)
            applied_l2 = float(np.linalg.norm(applied.ravel()))
            raw_energy += raw_delta * raw_delta
            applied_energy += applied_l2 * applied_l2
            rows.append(
                {
                    "source_tubelet": tubelet_index,
                    "source_start": start,
                    "source_stop": stop,
                    "patch_y": patch_y,
                    "patch_x": patch_x,
                    "sync_sign": sign,
                    "signed_projection_before": signed_before,
                    "raw_projection_after": raw_after,
                    "signed_projection_after": signed_after,
                    "raw_target_delta_l2": raw_delta,
                    "applied_float32_delta_l2": applied_l2,
                    "target_margin": target_margin,
                }
            )
    receipt = {
        "status": "COMPLETE",
        "method_version": public.method_version,
        "key_id": key_identifier(key),
        "source_shape": list(source.shape),
        "method_coordinate_dtype": "float32",
        "tubelet_rows": tubelet_count,
        "spatial_patches_per_tubelet": len(patch_coordinates),
        "projection_rows": len(rows),
        "active_projection_rows": sum(row["raw_target_delta_l2"] > 0.0 for row in rows),
        "raw_target_delta_l2": math.sqrt(raw_energy),
        "applied_float32_delta_l2": math.sqrt(applied_energy),
        "target_margin": target_margin,
        "embedding_margin_is_projection_target_not_cap": True,
        "target_is_writer_only_not_receiver_protocol": True,
        "rows": rows,
        "truth_used": False,
        "message_used": False,
    }
    return output, receipt


def score_received_latent(
    received_latent: np.ndarray,
    key: str,
    public: PublicProtocol = PUBLIC,
) -> dict[str, Any]:
    """Score every public offset without truth, gates, or writer-side state."""

    received = validate_received_latent(received_latent, public)
    offsets = candidate_offsets(int(received.shape[0]), public)
    patches = spatial_patch_coordinates(public.latent_height, public.latent_width, public)
    candidate_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for offset in offsets:
        numerator = 0.0
        denominator = 0.0
        candidate_status = "SCORED"
        supports = tubelet_support_rows(int(received.shape[0]), offset, public)
        for support in supports:
            tubelet = int(support["source_tubelet"])
            sign = sync_sign(key, tubelet, public)
            ages = np.asarray(support["source_ages"], dtype=np.int64)
            received_indices = np.asarray(support["received_indices"], dtype=np.int64)
            local_raw_dot = 0.0
            local_signed_projection = 0.0
            local_rho = 0.0
            local_status = "SCORED"
            for patch_y, patch_x in patches:
                full_direction = patch_direction(key, tubelet, patch_y, patch_x, public)
                direction = full_direction[ages]
                block = received[
                    received_indices,
                    :,
                    patch_y : patch_y + public.patch_height,
                    patch_x : patch_x + public.patch_width,
                ]
                if not bool(np.isfinite(block).all()):
                    local_status = "NONFINITE"
                    continue
                raw_dot = float(np.sum(block.astype(np.float64) * direction.astype(np.float64)))
                rho = float(np.sum(direction.astype(np.float64) ** 2))
                local_raw_dot += raw_dot
                local_signed_projection += sign * raw_dot
                local_rho += rho
            if local_status == "NONFINITE":
                candidate_status = "NONFINITE"
                q = None
            elif local_rho == 0.0:
                local_status = "ERASURE"
                candidate_status = "ERASURE" if candidate_status == "SCORED" else candidate_status
                q = None
            else:
                q = local_signed_projection / local_rho
                numerator += local_signed_projection
                denominator += local_rho
            local_rows.append(
                {
                    "source_offset": offset,
                    **support,
                    "status": local_status,
                    "sync_sign": sign,
                    "raw_dot": local_raw_dot if local_status != "NONFINITE" else None,
                    "signed_projection": local_signed_projection if local_status != "NONFINITE" else None,
                    "rho": local_rho if local_status != "NONFINITE" else None,
                    "q": q,
                }
            )
        score = numerator / denominator if candidate_status == "SCORED" and denominator > 0.0 else None
        candidate_rows.append(
            {
                "source_offset": offset,
                "status": candidate_status if score is not None else (
                    candidate_status if candidate_status != "SCORED" else "ERASURE"
                ),
                "numerator": numerator if score is not None else None,
                "denominator": denominator if score is not None else None,
                "score": score,
                "tubelet_rows": len(supports),
            }
        )
    finite = [row for row in candidate_rows if row["score"] is not None and math.isfinite(row["score"])]
    if finite:
        best = max(float(row["score"]) for row in finite)
        top = [int(row["source_offset"]) for row in finite if abs(float(row["score"]) - best) <= public.tie_atol]
    else:
        best = None
        top = []
    return {
        "status": "COMPLETE" if len(finite) == len(candidate_rows) else "RETAINED_FAILURES",
        "method_version": public.method_version,
        "key_id": key_identifier(key),
        "received_shape": list(received.shape),
        "candidate_offsets": list(offsets),
        "candidate_rows": candidate_rows,
        "local_rows": local_rows,
        "summary": {
            "best_score": best,
            "top_offsets": top,
            "canonical_offset": top[0] if len(top) == 1 else None,
            "unique": len(top) == 1,
            "sync_accepted": False,
            "full_singleton_geometry_only": len(offsets) == 1,
        },
        "counts": {
            "candidate_scores": len(candidate_rows),
            "candidate_tubelet_rows": len(local_rows),
            "failed_candidates": len(candidate_rows) - len(finite),
        },
        "truth_inputs": False,
    }
