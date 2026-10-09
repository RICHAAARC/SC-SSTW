"""Adopted local paired-energy carrier arithmetic and public RGB evidence.

This module contains deterministic carrier math only.  It deliberately has no
blind path selector, threshold, payload decoder, experiment acceptance flag,
or model/runtime dependency.  Scientific parameters such as ``rho`` are
always explicit call arguments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


CHIP_STATUSES = ("SCORED", "PARTIAL", "MISSING", "FAILED")
COMPONENT_STATUSES = ("SCORED", "PARTIAL", "MISSING", "FAILED")


@dataclass(frozen=True)
class CarrierProtocol:
    video_shape: tuple[int, int, int, int]
    segment_start: int
    segment_frames: int
    segment_count: int
    rois: tuple[tuple[int, int, int, int], ...]
    tile_size: int = 8

    def __post_init__(self) -> None:
        t, h, w, c = self.video_shape
        if c != 3 or min(t, h, w) <= 0:
            raise ValueError("carrier requires positive [T,H,W,3] RGB geometry")
        if self.segment_frames != 8 or self.tile_size != 8:
            raise ValueError("V1 uses fixed eight-frame segments and 8x8 DCT tiles")
        if self.segment_start < 0 or self.segment_count <= 0:
            raise ValueError("invalid carrier segment range")
        if self.segment_start + self.segment_frames * self.segment_count > t:
            raise ValueError("carrier segments exceed the public video")
        if len(self.rois) != 4:
            raise ValueError("V1 requires exactly four RGB ROIs")
        occupied: set[tuple[int, int]] = set()
        for y0, y1, x0, x1 in self.rois:
            if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
                raise ValueError("ROI is outside the public video")
            if (y1 - y0) % 8 or (x1 - x0) % 8:
                raise ValueError("each ROI must contain complete 8x8 tiles")
            cells = {(y, x) for y in range(y0, y1) for x in range(x0, x1)}
            if occupied.intersection(cells):
                raise ValueError("V1 ROIs must be disjoint")
            occupied.update(cells)


PUBLIC = CarrierProtocol(
    video_shape=(181, 320, 512, 3),
    segment_start=1,
    segment_frames=8,
    segment_count=22,
    rois=(
        (64, 128, 96, 160),
        (64, 128, 352, 416),
        (224, 288, 96, 160),
        (224, 288, 352, 416),
    ),
)


def public_hash(domain: str, key: str, *indices: int) -> bytes:
    """Exact public SHA256 derivation; it is not an authentication primitive."""
    if not isinstance(domain, str) or not isinstance(key, str):
        raise TypeError("hash domain and key must be strings")
    if any(type(value) is not int or value < 0 for value in indices):
        raise ValueError("public hash indices must be nonnegative integers")
    encoded = json.dumps(
        [domain, key, *indices], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def dct_coordinates() -> tuple[tuple[int, int], ...]:
    coords = sorted(
        ((u, v) for u in range(8) for v in range(8) if (u, v) != (0, 0)),
        key=lambda value: (value[0] + value[1], value[0], value[1]),
    )
    return tuple(coords[:32])


def coefficient_pairs(key: str, roi_index: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    if type(roi_index) is not int or not 0 <= roi_index < 4:
        raise ValueError("ROI index must be in 0..3")
    coords = dct_coordinates()
    order = sorted(
        range(32), key=lambda n: (public_hash("LJSP1/PAIR", key, roi_index, n), n)
    )
    return tuple((coords[order[2 * k]], coords[order[2 * k + 1]]) for k in range(16))


def state_matrix(key: str, segment_count: int = 22) -> tuple[tuple[int, ...], ...]:
    if type(segment_count) is not int or not 0 < segment_count <= 62:
        raise ValueError("state word count must be in 1..62")
    words = [a for a in range(64) if (a & 31) != 0]
    words.sort(key=lambda a: (public_hash("LJSP1/STATE_WORD", key, a), a))
    columns = sorted(
        range(32), key=lambda x: (public_hash("LJSP1/STATE_CHIP", key, x), x)
    )
    rows = []
    for a in words[:segment_count]:
        rows.append(tuple(
            -1 if ((a >> 5) + ((a & 31) & x).bit_count()) % 2 else 1
            for x in columns
        ))
    return tuple(rows)


def message_fragments(message: bytes) -> tuple[tuple[int, ...], ...]:
    if not isinstance(message, bytes) or len(message) != 4:
        raise ValueError("payload message must be exactly four bytes")
    return tuple(tuple((byte >> shift) & 1 for shift in range(7, -1, -1)) for byte in message)


def message_fragment_collision_classes(message: bytes) -> tuple[tuple[int, ...], ...]:
    """Pre-registration metadata for identical source fragments, not a receiver rule."""
    fragments = message_fragments(message)
    groups = []
    for index, fragment in enumerate(fragments):
        matching = tuple(i for i, value in enumerate(fragments) if value == fragment)
        if len(matching) > 1 and matching[0] == index:
            groups.append(matching)
    return tuple(groups)


def segment_targets(
    key: str, message: bytes, segment_index: int, *, segment_count: int = 22
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(segment_index) is not int or not 0 <= segment_index < segment_count:
        raise ValueError("segment index outside the declared carrier")
    state = state_matrix(key, segment_count)[segment_index]
    fragment = message_fragments(message)[segment_index % 4]
    return state, tuple(2 * bit - 1 for bit in fragment)


def _dct_matrix(reference: Any) -> Any:
    import torch

    n = torch.arange(8, device=reference.device, dtype=torch.float64)
    k = torch.arange(8, device=reference.device, dtype=torch.float64).reshape(-1, 1)
    matrix = torch.cos(math.pi * (n + 0.5) * k / 8.0) * math.sqrt(2.0 / 8.0)
    matrix[0] = math.sqrt(1.0 / 8.0)
    return matrix


def dct2_ortho(tiles: Any) -> Any:
    """FP64 orthonormal DCT-II over the last two 8x8 dimensions."""
    import torch

    if not torch.is_tensor(tiles) or tuple(tiles.shape[-2:]) != (8, 8):
        raise ValueError("DCT input must end in one or more 8x8 tiles")
    values = tiles.to(dtype=torch.float64)
    matrix = _dct_matrix(values)
    return torch.matmul(torch.matmul(matrix, values), matrix.transpose(0, 1))


def idct2_ortho(coefficients: Any) -> Any:
    """FP64 inverse matching :func:`dct2_ortho`."""
    import torch

    if not torch.is_tensor(coefficients) or tuple(coefficients.shape[-2:]) != (8, 8):
        raise ValueError("inverse DCT input must end in 8x8 coefficients")
    values = coefficients.to(dtype=torch.float64)
    matrix = _dct_matrix(values)
    return torch.matmul(torch.matmul(matrix.transpose(0, 1), values), matrix)


def _zero_signs(
    key: str, frame: int, roi_index: int, tile_count: int, pair_index: int,
    side: int, tiles_x: int, reference: Any,
) -> Any:
    import torch

    signs = []
    for flat in range(tile_count):
        ty, tx = divmod(flat, tiles_x)
        bit = public_hash(
            "LJSP1/ZERO_SIGN", key, frame, roi_index, ty, tx, pair_index, side
        )[0] & 1
        signs.append(2 * bit - 1)
    return torch.tensor(signs, device=reference.device, dtype=torch.float64)


def redistribute_coefficients(
    coefficients: Any, *, key: str, frame: int, roi_index: int,
    targets: tuple[int, ...], rho: float, tiles_x: int,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Redistribute each tile's 16 pair energies without adding zero support."""
    import torch

    if not torch.is_tensor(coefficients) or coefficients.ndim != 3 or tuple(coefficients.shape[-2:]) != (8, 8):
        raise ValueError("coefficients must be [tiles,8,8]")
    if len(targets) != 16 or any(value not in (-1, 1) for value in targets):
        raise ValueError("one signed target is required for every coefficient pair")
    if not math.isfinite(float(rho)) or not 0 <= float(rho) <= 1:
        raise ValueError("rho must be explicit, finite, and in [0,1]")
    if tiles_x <= 0 or coefficients.shape[0] % tiles_x:
        raise ValueError("tiles_x must divide the tile batch")
    changed = coefficients.to(dtype=torch.float64).clone()
    pair_rows = []
    for pair_index, ((up, vp), (um, vm)) in enumerate(coefficient_pairs(key, roi_index)):
        plus = changed[:, up, vp].clone()
        minus = changed[:, um, vm].clone()
        energy = plus.square() + minus.square()
        supported = energy > 0
        plus_sign = torch.sign(plus)
        minus_sign = torch.sign(minus)
        plus_sign = torch.where(
            (plus_sign == 0) & supported,
            _zero_signs(key, frame, roi_index, len(energy), pair_index, 0, tiles_x, plus),
            plus_sign,
        )
        minus_sign = torch.where(
            (minus_sign == 0) & supported,
            _zero_signs(key, frame, roi_index, len(energy), pair_index, 1, tiles_x, minus),
            minus_sign,
        )
        target = float(targets[pair_index])
        plus_new = plus_sign * torch.sqrt(energy * (1.0 + float(rho) * target) / 2.0)
        minus_new = minus_sign * torch.sqrt(energy * (1.0 - float(rho) * target) / 2.0)
        plus_new = torch.where(supported, plus_new, torch.zeros_like(plus_new))
        minus_new = torch.where(supported, minus_new, torch.zeros_like(minus_new))
        changed[:, up, vp] = plus_new
        changed[:, um, vm] = minus_new
        after_energy = plus_new.square() + minus_new.square()
        pair_rows.append(dict(
            pair_index=pair_index,
            component="state" if pair_index < 8 else "payload",
            target=int(target),
            plus_coord=[up, vp], minus_coord=[um, vm],
            supported_tiles=int(supported.sum()),
            zero_support_tiles=int((~supported).sum()),
            energy_before=float(energy.sum()),
            energy_after=float(after_energy.sum()),
        ))
    if not bool(torch.isfinite(changed).all()):
        raise FloatingPointError("nonfinite paired-energy coefficient update")
    return changed, tuple(pair_rows)


def _roi_tiles(rgb_roi: Any) -> tuple[Any, int, int]:
    y = (
        rgb_roi[..., 0].double() * 0.299
        + rgb_roi[..., 1].double() * 0.587
        + rgb_roi[..., 2].double() * 0.114
    )
    height, width = map(int, y.shape)
    tiles_y, tiles_x = height // 8, width // 8
    tiles = y.reshape(tiles_y, 8, tiles_x, 8).permute(0, 2, 1, 3).reshape(-1, 8, 8)
    return tiles, tiles_y, tiles_x


def apply_carrier_rgb(
    rgb: Any, *, key: str, message: bytes, rho: float,
    protocol: CarrierProtocol = PUBLIC,
) -> tuple[Any, dict[str, Any]]:
    """Apply the complete state+fragment carrier to finite RGB ``[0,1]``.

    Only declared segment/ROI pixels are assigned.  ``rho`` has no default and
    clipping is measured, not used to trigger an adaptive retry.
    """
    import torch

    if not torch.is_tensor(rgb) or tuple(rgb.shape) != protocol.video_shape:
        raise ValueError("RGB input does not match the declared carrier protocol")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()):
        raise FloatingPointError("carrier input must be finite floating RGB")
    if bool(((rgb < 0) | (rgb > 1)).any()):
        raise ValueError("carrier input must already be in [0,1]")
    if not math.isfinite(float(rho)) or not 0 <= float(rho) <= 1:
        raise ValueError("rho must be explicit, finite, and in [0,1]")
    output = rgb.detach().to(dtype=torch.float32).clone()
    pair_totals = [dict(
        energy_before=0.0, energy_after=0.0, zero_support_tiles=0,
        post_clip_energy_plus=0.0, post_clip_energy_minus=0.0,
        post_clip_supported_tiles=0, post_clip_abs_ratio_error_sum=0.0,
        post_clip_abs_ratio_error_max=0.0,
    )
                   for _ in range(16)]
    clipped_low = clipped_high = tile_count = 0
    clip_adjustment_sq = fp32_rounding_sq = 0.0
    fragments = message_fragments(message)
    states = state_matrix(key, protocol.segment_count)
    for segment_index in range(protocol.segment_count):
        state = states[segment_index]
        payload = tuple(2 * bit - 1 for bit in fragments[segment_index % 4])
        for offset in range(protocol.segment_frames):
            frame = protocol.segment_start + protocol.segment_frames * segment_index + offset
            for roi_index, (y0, y1, x0, x1) in enumerate(protocol.rois):
                original = output[frame, y0:y1, x0:x1].to(dtype=torch.float64)
                tiles, tiles_y, tiles_x = _roi_tiles(original)
                coefficients = dct2_ortho(tiles)
                targets = tuple(state[8 * roi_index:8 * (roi_index + 1)]) + payload
                modified_coefficients, rows = redistribute_coefficients(
                    coefficients, key=key, frame=frame, roi_index=roi_index,
                    targets=targets, rho=rho, tiles_x=tiles_x,
                )
                modified_y_tiles = idct2_ortho(modified_coefficients)
                modified_y = modified_y_tiles.reshape(tiles_y, tiles_x, 8, 8).permute(
                    0, 2, 1, 3
                ).reshape(y1 - y0, x1 - x0)
                original_y = tiles.reshape(tiles_y, tiles_x, 8, 8).permute(
                    0, 2, 1, 3
                ).reshape(y1 - y0, x1 - x0)
                candidate = original + (modified_y - original_y).unsqueeze(-1)
                clipped_low += int((candidate < 0).sum())
                clipped_high += int((candidate > 1).sum())
                clipped = candidate.clamp(0, 1)
                stored = clipped.to(dtype=torch.float32)
                clip_adjustment_sq += float((clipped - candidate).square().sum())
                fp32_rounding_sq += float((stored.double() - clipped).square().sum())
                output[frame, y0:y1, x0:x1] = stored
                post_tiles, _, _ = _roi_tiles(stored.double())
                post_coefficients = dct2_ortho(post_tiles)
                post_pairs = coefficient_pairs(key, roi_index)
                tile_count += len(tiles)
                for pair_index, row in enumerate(rows):
                    pair_totals[pair_index]["energy_before"] += row["energy_before"]
                    pair_totals[pair_index]["energy_after"] += row["energy_after"]
                    pair_totals[pair_index]["zero_support_tiles"] += row["zero_support_tiles"]
                    plus, minus = post_pairs[pair_index]
                    pair_totals[pair_index]["post_clip_energy_plus"] += float(
                        post_coefficients[:, plus[0], plus[1]].square().sum()
                    )
                    pair_totals[pair_index]["post_clip_energy_minus"] += float(
                        post_coefficients[:, minus[0], minus[1]].square().sum()
                    )
                    post_plus = post_coefficients[:, plus[0], plus[1]].square()
                    post_minus = post_coefficients[:, minus[0], minus[1]].square()
                    denominator = post_plus + post_minus
                    supported = denominator > 0
                    if bool(supported.any()):
                        post_q = (post_plus[supported] - post_minus[supported]) / denominator[supported]
                        errors = (post_q - float(rho) * rows[pair_index]["target"]).abs()
                        pair_totals[pair_index]["post_clip_supported_tiles"] += int(supported.sum())
                        pair_totals[pair_index]["post_clip_abs_ratio_error_sum"] += float(errors.sum())
                        pair_totals[pair_index]["post_clip_abs_ratio_error_max"] = max(
                            pair_totals[pair_index]["post_clip_abs_ratio_error_max"],
                            float(errors.max()),
                        )
    for row in pair_totals:
        row["post_clip_abs_ratio_error_mean"] = (
            row["post_clip_abs_ratio_error_sum"] / row["post_clip_supported_tiles"]
            if row["post_clip_supported_tiles"] else None
        )
    return output, dict(
        coordinate="received_rgb_float32",
        rho=float(rho),
        segment_count=protocol.segment_count,
        frames_written=protocol.segment_frames * protocol.segment_count,
        roi_count=len(protocol.rois), tile_instances=tile_count,
        clipped_low_values=clipped_low, clipped_high_values=clipped_high,
        clip_adjustment_l2=math.sqrt(clip_adjustment_sq),
        fp32_rounding_l2=math.sqrt(fp32_rounding_sq),
        outside_declared_roi_written_values=0,
        pair_energy_constraint_stage="pre_clipping_fp64_coefficients",
        pair_totals=pair_totals,
        adaptive_retry=False,
    )


@dataclass(frozen=True)
class RawWindowSpec:
    observation_id: str
    phase: int
    slot: int
    roi_index: int
    requested_frames: tuple[int, ...]
    received_frames: tuple[int, ...]
    availability: str


def phase_window_catalog(
    protocol: CarrierProtocol = PUBLIC, *, received_frame_count: int | None = None,
) -> tuple[RawWindowSpec, ...]:
    """Return the fixed raw directory without filtering boundary rows.

    For production this is 8 phases x 24 slots x 4 ROIs = 768 rows.  Signed
    requested coordinates are retained even when no received frame exists.
    """
    rows = []
    frame_count = protocol.video_shape[0] if received_frame_count is None else received_frame_count
    if type(frame_count) is not int or not 0 <= frame_count <= protocol.video_shape[0]:
        raise ValueError("received frame count must be in the supported public range")
    for phase in range(protocol.segment_frames):
        for slot in range(-1, protocol.segment_count + 1):
            requested = tuple(phase + protocol.segment_frames * slot + i for i in range(8))
            received = tuple(value for value in requested if 0 <= value < frame_count)
            availability = "FULL" if len(received) == 8 else "PARTIAL" if received else "MISSING"
            for roi_index in range(4):
                rows.append(RawWindowSpec(
                    observation_id=f"phase{phase}:slot{slot}:roi{roi_index}",
                    phase=phase, slot=slot, roi_index=roi_index,
                    requested_frames=requested, received_frames=received,
                    availability=availability,
                ))
    return tuple(rows)


@dataclass(frozen=True)
class ChipEvidence:
    component: str
    component_index: int
    pair_index: int
    plus_coord: tuple[int, int]
    minus_coord: tuple[int, int]
    status: str
    energy_plus: float | None
    energy_minus: float | None
    q: float | None
    expected_frames: int
    available_frames: int
    expected_samples: int
    available_samples: int
    supported_samples: int
    zero_energy_samples: int
    failed_samples: int
    error: str | None = None


@dataclass(frozen=True)
class RawWindowObservation:
    spec: RawWindowSpec
    state_status: str
    payload_status: str
    state_chips: tuple[ChipEvidence, ...]
    payload_chips: tuple[ChipEvidence, ...]
    truth_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _component_status(chips: tuple[ChipEvidence, ...]) -> str:
    statuses = [chip.status for chip in chips]
    if all(value == "SCORED" for value in statuses):
        return "SCORED"
    if any(value in ("SCORED", "PARTIAL") for value in statuses):
        return "PARTIAL"
    if any(value == "FAILED" for value in statuses):
        return "FAILED"
    return "MISSING"


def _empty_chips(
    key: str, spec: RawWindowSpec, status: str, protocol: CarrierProtocol,
    error: str | None = None,
) -> tuple[ChipEvidence, ...]:
    y0, y1, x0, x1 = protocol.rois[spec.roi_index]
    tiles_per_frame = ((y1 - y0) // 8) * ((x1 - x0) // 8)
    available_samples = len(spec.received_frames) * tiles_per_frame
    rows = []
    for pair_index, (plus, minus) in enumerate(coefficient_pairs(key, spec.roi_index)):
        rows.append(ChipEvidence(
            component="state" if pair_index < 8 else "payload",
            component_index=(8 * spec.roi_index + pair_index) if pair_index < 8 else pair_index - 8,
            pair_index=pair_index, plus_coord=plus, minus_coord=minus,
            status=status, energy_plus=None, energy_minus=None, q=None,
            expected_frames=8, available_frames=len(spec.received_frames),
            expected_samples=8 * tiles_per_frame, available_samples=available_samples,
            supported_samples=0, zero_energy_samples=0,
            failed_samples=available_samples if status == "FAILED" else 0,
            error=error,
        ))
    return tuple(rows)


def observe_window(
    received_rgb: Any, key: str, spec: RawWindowSpec, *,
    protocol: CarrierProtocol = PUBLIC,
) -> RawWindowObservation:
    """Extract truth-free per-chip raw energy and q from one declared row."""
    import torch

    if not isinstance(spec, RawWindowSpec):
        raise TypeError("raw window specification required")
    if (not torch.is_tensor(received_rgb) or received_rgb.ndim != 4 or
            tuple(received_rgb.shape[1:]) != protocol.video_shape[1:] or
            not 0 <= received_rgb.shape[0] <= protocol.video_shape[0]):
        raise ValueError("received RGB must keep public H,W,3 with supported T<=public T")
    if any(value >= received_rgb.shape[0] for value in spec.received_frames):
        raise ValueError("window availability exceeds the supplied received RGB")
    if not spec.received_frames:
        chips = _empty_chips(key, spec, "MISSING", protocol, "no received frame support")
        return RawWindowObservation(
            spec, "MISSING", "MISSING", chips[:8], chips[8:]
        )
    y0, y1, x0, x1 = protocol.rois[spec.roi_index]
    selected = received_rgb[list(spec.received_frames), y0:y1, x0:x1]
    if selected.shape[-1] != 3:
        raise ValueError("received observation is not RGB")
    selected = selected.to(dtype=torch.float64)
    luminance = selected[..., 0] * 0.299 + selected[..., 1] * 0.587 + selected[..., 2] * 0.114
    frame_count, height, width = map(int, luminance.shape)
    tiles_y, tiles_x = height // 8, width // 8
    tiles = luminance.reshape(frame_count, tiles_y, 8, tiles_x, 8).permute(
        0, 1, 3, 2, 4
    ).reshape(-1, 8, 8)
    coefficients = dct2_ortho(tiles)
    rows = []
    for pair_index, (plus_coord, minus_coord) in enumerate(coefficient_pairs(key, spec.roi_index)):
        plus = coefficients[:, plus_coord[0], plus_coord[1]]
        minus = coefficients[:, minus_coord[0], minus_coord[1]]
        component = "state" if pair_index < 8 else "payload"
        component_index = 8 * spec.roi_index + pair_index if pair_index < 8 else pair_index - 8
        finite = torch.isfinite(plus) & torch.isfinite(minus)
        failed_samples = int((~finite).sum())
        finite_plus, finite_minus = plus[finite], minus[finite]
        pair_energy = finite_plus.square() + finite_minus.square()
        supported_samples = int((pair_energy > 0).sum())
        zero_samples = int((pair_energy == 0).sum())
        if supported_samples:
            energy_plus = float(finite_plus.square().sum())
            energy_minus = float(finite_minus.square().sum())
            denominator = energy_plus + energy_minus
            row = ChipEvidence(
                component, component_index, pair_index, plus_coord, minus_coord,
                "PARTIAL" if failed_samples else "SCORED",
                energy_plus, energy_minus,
                (energy_plus - energy_minus) / denominator,
                8, frame_count, 8 * tiles_y * tiles_x, len(tiles),
                supported_samples, zero_samples, failed_samples,
                f"{failed_samples} nonfinite coefficient samples" if failed_samples else None,
            )
        else:
            status = "FAILED" if failed_samples else "MISSING"
            row = ChipEvidence(
                component, component_index, pair_index, plus_coord, minus_coord,
                status, None if failed_samples else 0.0, None if failed_samples else 0.0,
                None, 8, frame_count, 8 * tiles_y * tiles_x, len(tiles),
                0, zero_samples, failed_samples,
                "no finite supported pair energy" if failed_samples else "zero pair energy",
            )
        rows.append(row)
    state, payload = tuple(rows[:8]), tuple(rows[8:])
    state_status, payload_status = _component_status(state), _component_status(payload)
    if spec.availability == "PARTIAL":
        if state_status == "SCORED":
            state_status = "PARTIAL"
        if payload_status == "SCORED":
            payload_status = "PARTIAL"
    return RawWindowObservation(spec, state_status, payload_status, state, payload)


def observe_catalog(
    received_rgb: Any, key: str, *, protocol: CarrierProtocol = PUBLIC,
    catalog: Iterable[RawWindowSpec] | None = None,
) -> tuple[RawWindowObservation, ...]:
    """Retain every declared raw row, converting extraction errors to FAILED."""
    output = []
    if catalog is None:
        try:
            received_count = int(received_rgb.shape[0])
        except Exception:
            received_count = protocol.video_shape[0]
        if 0 <= received_count <= protocol.video_shape[0]:
            declared = phase_window_catalog(protocol, received_frame_count=received_count)
        else:
            # Keep the fixed denominator; each row below records the unsupported
            # received geometry as FAILED instead of dropping the catalog.
            declared = phase_window_catalog(protocol)
    else:
        declared = tuple(catalog)
    for spec in declared:
        try:
            row = observe_window(received_rgb, key, spec, protocol=protocol)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            chips = _empty_chips(key, spec, "FAILED", protocol, error)
            row = RawWindowObservation(spec, "FAILED", "FAILED", chips[:8], chips[8:])
        output.append(row)
    return tuple(output)


@dataclass(frozen=True)
class SourceSlotCorrespondence:
    mapping_id: str
    candidate_id: str
    observation_id: str
    source_slot: int
    fragment_slot: int
    equivalence_id: str | None = None

    def __post_init__(self) -> None:
        if not self.mapping_id or not self.candidate_id or not self.observation_id:
            raise ValueError("explicit mapping, candidate, and observation ids are required")
        if type(self.source_slot) is not int or type(self.fragment_slot) is not int:
            raise TypeError("source and fragment slots must be integers")
        if not 0 <= self.source_slot < PUBLIC.segment_count:
            raise ValueError("source slot must be in the fixed public 0..21 range")
        if not 0 <= self.fragment_slot < 4:
            raise ValueError("fragment slot must be in 0..3")
        if self.equivalence_id is not None and (
                not isinstance(self.equivalence_id, str) or not self.equivalence_id):
            raise ValueError("declared equivalence id must be a nonempty string")


def route_fragment_evidence(
    observations: Iterable[RawWindowObservation],
    correspondences: Iterable[SourceSlotCorrespondence],
) -> dict[str, Any]:
    """Route raw payload evidence by explicit correspondence without aggregation."""
    observed = tuple(observations)
    by_id = {row.spec.observation_id: row for row in observed}
    if len(by_id) != len(observed):
        raise ValueError("raw observation ids must be unique")
    routes = tuple(correspondences)
    if any(not isinstance(route, SourceSlotCorrespondence) for route in routes):
        raise TypeError("explicit source-slot correspondence required")
    if len({route.mapping_id for route in routes}) != len(routes):
        raise ValueError("explicit mapping ids must be unique")
    buckets: dict[str, dict[int, list[dict[str, Any]]]] = {}
    used: dict[tuple[str, str], list[int]] = {}
    equivalence: dict[str, list[str]] = {}
    for route in routes:
        row = by_id.get(route.observation_id)
        if row is None:
            raise ValueError(f"unknown observation id: {route.observation_id}")
        entry = dict(
            mapping_id=route.mapping_id, candidate_id=route.candidate_id,
            observation_id=route.observation_id, source_slot=route.source_slot,
            received_phase=row.spec.phase, received_slot=row.spec.slot,
            fragment_slot=route.fragment_slot,
            source_mod4=route.source_slot % 4,
            mod4_equivalent=(route.source_slot % 4 == route.fragment_slot),
            equivalence_id=route.equivalence_id,
            payload_status=row.payload_status,
            payload_chips=[asdict(chip) for chip in row.payload_chips],
            aggregation="none",
        )
        buckets.setdefault(route.candidate_id, {slot: [] for slot in range(4)})[
            route.fragment_slot
        ].append(entry)
        used.setdefault((route.candidate_id, route.observation_id), []).append(route.fragment_slot)
        if route.equivalence_id is not None:
            equivalence.setdefault(route.equivalence_id, []).append(route.mapping_id)
    conflicts = [
        dict(candidate_id=candidate, observation_id=observation, fragment_slots=sorted(set(slots)))
        for (candidate, observation), slots in used.items() if len(set(slots)) > 1
    ]
    reused = [
        dict(candidate_id=candidate, observation_id=observation, route_count=len(slots))
        for (candidate, observation), slots in used.items() if len(slots) > 1
    ]
    erasures = {
        candidate: [slot for slot, entries in slots.items() if not entries]
        for candidate, slots in buckets.items()
    }
    many_to_one = []
    for candidate, slots in buckets.items():
        for fragment_slot, entries in slots.items():
            source_slots = sorted({entry["source_slot"] for entry in entries})
            if len(source_slots) > 1:
                many_to_one.append(dict(
                    candidate_id=candidate, fragment_slot=fragment_slot,
                    source_slots=source_slots,
                    observation_ids=[entry["observation_id"] for entry in entries],
                ))
    return dict(
        candidates=buckets,
        erasures=erasures,
        conflicts=conflicts,
        reused_observations=reused,
        many_to_one_routes=many_to_one,
        caller_declared_equivalence_classes=equivalence,
        equivalence_verified=False,
        aggregation="none",
        decoded_message=None,
        accepted=None,
        truth_used=False,
    )
