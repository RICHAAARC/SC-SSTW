"""Four-bit payload code, explicit sync pilots, blind search, and calibration.

The payload is one arbitrary nibble (0..15).  RM(1,3) supplies eight coded
data windows with minimum Hamming distance four.  Three other windows carry a
common keyed pilot through an explicit loss term; a common pilot is never
counted as target-versus-rival evidence.
"""
from __future__ import annotations

import itertools
import hashlib
import math
from typing import Iterable

import numpy as np

from . import projection_margin as carrier
from . import state_clock

PAYLOAD_BITS = 4
PAYLOAD_COUNT = 1 << PAYLOAD_BITS
WINDOWS = 11
SUPPORTS_PER_WINDOW = 160
DATA_WINDOWS = (1, 2, 3, 4, 6, 7, 8, 9)
PILOT_WINDOWS = (0, 5, 10)
PILOT_LOSS_WEIGHT = 0.25
EVENT_COST = state_clock.EDIT_COST
FIXED_VIEWS = ("FULL", "CROP0_129", "CROP4_129", "CROP8_129", "DELETE90", "SPEED5_4", "REENCODE")
RECEIVER_PROTOCOL_ID = "SC-SSTW-Payload-RM13-Partial3-V2"


def key_identifier(key: bytes) -> str:
    """Non-secret binding used to prevent applying thresholds from another key."""
    return hashlib.sha256(b"SC-SSTW/receiver-key-id/v1\0" + key).hexdigest()[:16]


def parse_payload(value: int | str) -> int:
    """Parse one net four-bit payload without A/B aliases."""
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            result = int(text, 16)
        elif len(text) == 4 and set(text) <= {"0", "1"}:
            result = int(text, 2)
        else:
            result = int(text, 10)
    else:
        raise TypeError("payload must be an integer, hex nibble, or four-bit string")
    if not 0 <= result < PAYLOAD_COUNT:
        raise ValueError("payload is outside the four-bit range 0..15")
    return result


def payload_bits(payload: int | str) -> np.ndarray:
    value = parse_payload(payload)
    return np.asarray([(value >> shift) & 1 for shift in (3, 2, 1, 0)], dtype=np.int8)


def rm_encode(payload: int | str) -> np.ndarray:
    """Encode affine Boolean f(x)=a0+a1*x1+a2*x2+a3*x3 over F2."""
    a = payload_bits(payload)
    words = []
    for x1, x2, x3 in itertools.product((0, 1), repeat=3):
        words.append(int(a[0] ^ (a[1] & x1) ^ (a[2] & x2) ^ (a[3] & x3)))
    return np.asarray(words, dtype=np.int8)


RM_CODEWORDS = np.stack([rm_encode(payload) for payload in range(PAYLOAD_COUNT)])
RM_MIN_DISTANCE = min(
    int(np.count_nonzero(RM_CODEWORDS[a] != RM_CODEWORDS[b]))
    for a in range(PAYLOAD_COUNT)
    for b in range(a + 1, PAYLOAD_COUNT)
)


def rm_decode(bits: Iterable[int | bool | None]) -> dict:
    received = list(bits)
    if len(received) != 8 or any(value not in (0, 1, False, True, None) for value in received):
        raise ValueError("RM decoder requires eight binary or erased data-window decisions")
    visible = np.asarray([value is not None for value in received], dtype=bool)
    values = np.asarray([0 if value is None else int(value) for value in received], dtype=np.int8)
    distances = np.count_nonzero((RM_CODEWORDS != values[None, :]) & visible[None, :], axis=1)
    minimum = int(distances.min())
    erasures = int((~visible).sum())
    winners = np.flatnonzero(distances == minimum).tolist()
    unique = len(winners) == 1
    within_bound = 2 * minimum + erasures < RM_MIN_DISTANCE
    return {
        "status": "EXACT" if unique and minimum == 0 and erasures == 0 else ("BOUNDED_RECOVERY" if unique and within_bound else "OUTSIDE_BOUNDED_REGION"),
        "payload": winners[0] if unique and within_bound else None,
        "visible_mismatches": minimum,
        "erasures": erasures,
        "nearest_payloads": winners,
        "guarantee": "unique bounded hard-window decoding only when 2*errors+erasures<4; no video-frame or soft-channel guarantee",
    }


def codebook(key: bytes) -> dict:
    base = carrier.codebook(key)
    polarity = np.asarray(
        [carrier.sign(key, "payload_codec/polarity", index) for index in range(carrier.SUPPORT_COUNT)],
        dtype=np.int8,
    )
    base_phases = np.asarray([
        carrier.digest(key, "payload_codec/base_phase", window)[0] % 4 for window in range(WINDOWS)
    ], dtype=np.int8)
    base_states = state_clock.PHASES[base_phases].astype(np.int8)
    axes = np.arange(carrier.SUPPORT_COUNT) % 2
    window_index = np.arange(carrier.SUPPORT_COUNT) // SUPPORTS_PER_WINDOW
    common = base_states[window_index, axes] * base["sync"].astype(np.int8) * polarity
    codes = np.repeat(common[None, :], PAYLOAD_COUNT, axis=0)
    states = np.repeat(base_states[None, :, :], PAYLOAD_COUNT, axis=0)
    for payload in range(PAYLOAD_COUNT):
        signs = 1 - 2 * RM_CODEWORDS[payload]
        for bit_index, window in enumerate(DATA_WINDOWS):
            sl = slice(window * SUPPORTS_PER_WINDOW, (window + 1) * SUPPORTS_PER_WINDOW)
            codes[payload, sl] *= signs[bit_index]
            states[payload, window] *= signs[bit_index]
    phase_by_payload = np.empty((PAYLOAD_COUNT, WINDOWS), dtype=np.int8)
    for payload in range(PAYLOAD_COUNT):
        for window in range(WINDOWS):
            phase_by_payload[payload, window] = next(
                index for index, phase in enumerate(state_clock.PHASES) if np.array_equal(phase, states[payload, window])
            )
    steps = (phase_by_payload[:, 1:] - phase_by_payload[:, :-1]) % 4
    pilot_mask = np.zeros(carrier.SUPPORT_COUNT, dtype=bool)
    data_mask = np.zeros(carrier.SUPPORT_COUNT, dtype=bool)
    for window in PILOT_WINDOWS:
        pilot_mask[window * SUPPORTS_PER_WINDOW:(window + 1) * SUPPORTS_PER_WINDOW] = True
    for window in DATA_WINDOWS:
        data_mask[window * SUPPORTS_PER_WINDOW:(window + 1) * SUPPORTS_PER_WINDOW] = True
    return {
        "receiver_protocol_id": RECEIVER_PROTOCOL_ID,
        "key_id": key_identifier(key),
        "directions": base["directions"],
        "sync": base["sync"],
        "polarity": polarity,
        "common": common,
        "codes": codes.astype(np.int8),
        "data_mask": data_mask,
        "pilot_mask": pilot_mask,
        "rm_codewords": RM_CODEWORDS.copy(),
        "base_phases": base_phases,
        "states": states,
        "steps": steps,
    }


def torch_blocks(z):
    if tuple(z.shape) != carrier.SHAPE:
        raise ValueError(f"payload latent must have shape {carrier.SHAPE}")
    x = z[0, :, 1:45].permute(1, 0, 2, 3)
    return x.reshape(11, 4, 16, 10, 4, 16, 4).permute(0, 3, 5, 1, 2, 4, 6).reshape(carrier.SUPPORT_COUNT, 1024)


def torch_projections(z, directions):
    return (torch_blocks(z).double() * directions.double()).sum(dim=1)


def competitive_tanh_loss(projections, codes, payload: int):
    """Mean-rival competition; with two templates it is the legacy A/B loss."""
    payload = parse_payload(payload)
    if codes.ndim != 2 or not 0 <= payload < codes.shape[0] or codes.shape[0] < 2:
        raise ValueError("competitive loss requires at least two payload templates")
    rivals = (codes.sum(dim=0) - codes[payload]) / (codes.shape[0] - 1)
    return -(projections.tanh() * (codes[payload] - rivals)).mean()


def payload_sync_loss(z, book: dict, payload: int):
    """Temperature-one data competition plus a nonzero explicit pilot term."""
    import torch

    directions = torch.as_tensor(book["directions"], dtype=torch.float64, device=z.device)
    codes = torch.as_tensor(book["codes"], dtype=torch.float64, device=z.device)
    common = torch.as_tensor(book["common"], dtype=torch.float64, device=z.device)
    data_mask = torch.as_tensor(book["data_mask"], dtype=torch.bool, device=z.device)
    pilot_mask = torch.as_tensor(book["pilot_mask"], dtype=torch.bool, device=z.device)
    p = torch_projections(z, directions)
    data = competitive_tanh_loss(p[data_mask], codes[:, data_mask], payload)
    pilot = -(p[pilot_mask].tanh() * common[pilot_mask]).mean()
    total = (data + PILOT_LOSS_WEIGHT * pilot) / (1.0 + PILOT_LOSS_WEIGHT)
    return total, {"data_loss": data, "pilot_loss": pilot, "total_loss": total}


def _emission(observations: dict, book: dict, window: int, g: int, num: int, den: int, offset: int) -> dict:
    z = observations.get(g)
    selected = [None] * 4 if z is None else carrier.allocation(z.shape[2] - 1, g, num, den, offset)[4 * window:4 * window + 4]
    group_count = sum(value is not None for value in selected)
    if group_count < 3:
        return {"valid": False, "selected": selected, "group_count": group_count,
                "support_kind": "INVALID", "observed_components": 0}
    sl = slice(window * SUPPORTS_PER_WINDOW, (window + 1) * SUPPORTS_PER_WINDOW)
    if group_count == 4:
        # Preserve the successful complete-window numerical path exactly.
        data = np.take(z[0], [value + 1 for value in selected], axis=1).transpose(1, 0, 2, 3)
        data = data.reshape(4, 16, 10, 4, 16, 4).transpose(2, 4, 0, 1, 3, 5).reshape(SUPPORTS_PER_WINDOW, 1024)
        values = np.einsum("ij,ij->i", data.astype(np.float64), book["directions"][sl].astype(np.float64))
    else:
        directions = book["directions"][sl].astype(np.float64).reshape(SUPPORTS_PER_WINDOW, 4, 256)
        values = np.zeros(SUPPORTS_PER_WINDOW, dtype=np.float64)
        # Sum only the three physical temporal components and clip once.  There
        # is no zero imputation, 4/3 multiplier, or energy rescaling.
        for slot, value in enumerate(selected):
            if value is None:
                continue
            component = z[0, :, value + 1].reshape(16, 10, 4, 16, 4).transpose(1, 3, 0, 2, 4).reshape(SUPPORTS_PER_WINDOW, 256)
            values += np.einsum("ij,ij->i", component.astype(np.float64), directions[:, slot])
    clipped = np.clip(values, -1.0, 1.0)
    common = book["common"][sl]
    common_corr = float(np.mean(clipped * common))
    decoded = clipped * book["sync"][sl] * book["polarity"][sl]
    return {
        "valid": True,
        "selected": selected,
        "group_count": group_count,
        "support_kind": "FULL" if group_count == 4 else "PARTIAL3",
        "full_blocks": SUPPORTS_PER_WINDOW if group_count == 4 else 0,
        "partial_blocks": SUPPORTS_PER_WINDOW if group_count == 3 else 0,
        "observed_components": SUPPORTS_PER_WINDOW * group_count,
        "common_correlation": common_corr,
        "q": [float(decoded[::2].mean()), float(decoded[1::2].mean())],
        "hard_bit": None if common_corr == 0.0 else int(common_corr < 0),
        "payload_scores": [float(np.mean(clipped * book["codes"][payload, sl])) for payload in range(PAYLOAD_COUNT)],
    }


def read(observations: dict, book: dict) -> dict:
    """Blind 16-payload/time search; no truth, prompt, or writer state input."""
    cache: dict[tuple, dict] = {}
    compact_candidates = []
    best_by_payload = [None] * PAYLOAD_COUNT
    best_overall = None
    for path in state_clock.clock_paths():
        windows = []
        origins = []
        for window in range(WINDOWS):
            delta = path["delta"] if window >= path["boundary"] else 0
            g = (path["g"] - delta) % 4
            offset = path["offset"] + delta
            key = (window, g, *path["scale"], offset)
            if key not in cache:
                cache[key] = _emission(observations, book, window, g, *path["scale"], offset)
            windows.append(cache[key])
            origins.append(g)
        valid_data = [window for window in DATA_WINDOWS if windows[window]["valid"]]
        valid_pilot = [window for window in PILOT_WINDOWS if windows[window]["valid"]]
        if not valid_data:
            continue
        pilot_weights = np.asarray([windows[window]["observed_components"] for window in valid_pilot], dtype=float)
        pilot_score = float(np.average([windows[window]["common_correlation"] for window in valid_pilot], weights=pilot_weights)) if valid_pilot else -1.0
        penalty = EVENT_COST if path["delta"] else 0.0
        row_path = dict(path, origins=origins, valid_data_windows=valid_data, valid_pilot_windows=valid_pilot)
        qs = np.asarray([window.get("q", [0.0, 0.0]) for window in windows], dtype=float)
        valid = [bool(window["valid"]) for window in windows]
        path_best = None
        for payload in range(PAYLOAD_COUNT):
            data_weights = np.asarray([windows[window]["observed_components"] for window in valid_data], dtype=float)
            data_score = float(np.average([windows[window]["payload_scores"][payload] for window in valid_data], weights=data_weights))
            matched_score = (data_score + PILOT_LOSS_WEIGHT * pilot_score) / (1.0 + PILOT_LOSS_WEIGHT)
            observer = state_clock.observe(qs, valid, book["states"][payload], book["steps"][payload])
            score = matched_score - state_clock.INNOVATION_WEIGHT * observer["innovation_mean"] - penalty
            hard = [windows[window].get("hard_bit") for window in DATA_WINDOWS]
            evidence = [windows[window].get("common_correlation") for window in DATA_WINDOWS]
            evidence_weights = [windows[window].get("observed_components", 0) for window in DATA_WINDOWS]
            row = row_path | {"payload": payload, "score": score, "matched_score": matched_score,
                              "state_innovation_mean": observer["innovation_mean"],
                              "data_score": data_score, "pilot_score": pilot_score,
                              "matched_data_blocks": SUPPORTS_PER_WINDOW * len(valid_data),
                              "matched_pilot_blocks": SUPPORTS_PER_WINDOW * len(valid_pilot),
                              "matched_data_components": int(data_weights.sum()),
                              "matched_pilot_components": int(pilot_weights.sum()),
                              "hard_data_bits": hard, "hard_window_evidence": evidence,
                              "hard_window_component_weights": evidence_weights,
                              "decoder": rm_decode(hard),
                              "full_data_blocks": sum(windows[window].get("full_blocks", 0) for window in DATA_WINDOWS),
                              "partial_data_blocks": sum(windows[window].get("partial_blocks", 0) for window in DATA_WINDOWS)}
            if best_by_payload[payload] is None or _rank(row) < _rank(best_by_payload[payload]):
                best_by_payload[payload] = row
            if best_overall is None or _rank(row) < _rank(best_overall):
                best_overall = row
            if path_best is None or _rank(row) < _rank(path_best):
                path_best = row
        compact_candidates.append(path_best)
    ranked = sorted((row for row in best_by_payload if row is not None), key=_rank)
    if best_overall is None:
        return {"status": "INVALID", "reason": "NO_USABLE_DATA_WINDOW", "best": None, "best_by_payload": {}, "decoder": None,
                "receiver_protocol_id": book.get("receiver_protocol_id", RECEIVER_PROTOCOL_ID), "key_id": book.get("key_id")}
    top_tie = len(ranked) > 1 and abs(ranked[0]["score"] - ranked[1]["score"]) <= 1e-12
    return {
        "status": "SCORED",
        "candidate_path_count": len(compact_candidates),
        "message_path_search_count": len(compact_candidates) * PAYLOAD_COUNT,
        "best": best_overall,
        "best_by_payload": {str(row["payload"]): row for row in ranked},
        "top_payload_unique": not top_tie,
        "hard_data_bits": best_overall["hard_data_bits"],
        "hard_window_evidence": best_overall["hard_window_evidence"],
        "hard_window_component_weights": best_overall["hard_window_component_weights"],
        "decoder": best_overall["decoder"],
        "receiver_protocol_id": book.get("receiver_protocol_id", RECEIVER_PROTOCOL_ID),
        "key_id": book.get("key_id"),
        "existence_statistic": best_overall["score"],
        "statistic_definition": "max over 16 payloads and all fixed clock paths of normalized data+0.25*pilot matched score minus 0.05 fixed-gain state innovation and event cost",
    }


def _rank(row: dict) -> tuple:
    return (
        -row["score"], abs(row["delta"]), abs(row["offset"]),
        abs(row["scale"][0] / row["scale"][1] - 1.0), row["g"], row["boundary"], row["payload"],
    )


def aggregate_crop_views(detections: dict[str, dict]) -> dict:
    names = ("CROP0_129", "CROP4_129", "CROP8_129")
    present = {name: detections.get(name) for name in names if detections.get(name, {}).get("status") == "SCORED"}
    if len(present) != len(names):
        return {"status": "INVALID", "required_views": list(names), "present_views": sorted(present)}
    scores = []
    for payload in range(PAYLOAD_COUNT):
        rows = [present[name]["best_by_payload"].get(str(payload)) for name in names]
        if any(row is None for row in rows):
            continue
        weights = np.asarray([
            row.get("matched_data_components", row.get("matched_data_blocks", row.get("matched_data_supports", 0)) * 4) +
            row.get("matched_pilot_components", row.get("matched_pilot_blocks", row.get("matched_pilot_supports", 0)) * 4)
            for row in rows
        ], dtype=float)
        score = float(np.average([row["score"] for row in rows], weights=weights))
        combined_evidence = []
        combined_weights = []
        for bit_index in range(8):
            pairs = []
            for row in rows:
                value = row["hard_window_evidence"][bit_index]
                weight = row.get("hard_window_component_weights", [640] * 8)[bit_index]
                if value is not None and math.isfinite(value) and value != 0.0 and weight > 0:
                    pairs.append((value, weight))
            combined_evidence.append(None if not pairs else float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs])))
            combined_weights.append(sum(p[1] for p in pairs))
        hard = [None if value is None or value == 0.0 else int(value < 0) for value in combined_evidence]
        scores.append({"payload": payload, "score": score,
                       "view_scores": {name: rows[index]["score"] for index, name in enumerate(names)},
                       "hard_data_bits": hard, "hard_window_evidence": combined_evidence,
                       "hard_window_component_weights": combined_weights, "decoder": rm_decode(hard)})
    scores.sort(key=lambda row: (-row["score"], row["payload"]))
    winner = scores[0] if scores else None
    protocol_ids = {present[name].get("receiver_protocol_id") for name in names}
    key_ids = {present[name].get("key_id") for name in names}
    binding_ok = len(protocol_ids) == 1 and len(key_ids) == 1
    return {
        "status": "SCORED" if scores and binding_ok else "INVALID",
        "views": list(names),
        "best": winner,
        "best_by_payload": {str(row["payload"]): row for row in scores},
        "top_payload_unique": len(scores) < 2 or abs(scores[0]["score"] - scores[1]["score"]) > 1e-12,
        "hard_data_bits": None if winner is None else winner["hard_data_bits"],
        "hard_window_evidence": None if winner is None else winner["hard_window_evidence"],
        "hard_window_component_weights": None if winner is None else winner["hard_window_component_weights"],
        "decoder": None if winner is None else winner["decoder"],
        "existence_statistic": winner["score"] if winner else None,
        "receiver_protocol_id": next(iter(protocol_ids)) if len(protocol_ids) == 1 else None,
        "key_id": next(iter(key_ids)) if len(key_ids) == 1 else None,
        "reason": None if binding_ok else "VIEW_PROTOCOL_OR_KEY_BINDING_MISMATCH",
        "statistic_definition": "support-weighted aggregation of three independently saved and VAE-encoded 129-frame crop views",
    }


def source_max_statistic(detections: dict[str, dict], crop_aggregate: dict) -> dict:
    missing = [name for name in FIXED_VIEWS if detections.get(name, {}).get("status") != "SCORED"]
    if missing or crop_aggregate.get("status") != "SCORED":
        return {"status": "INVALID", "statistic": None, "winning_view": None,
                "missing_or_failed_views": missing,
                "crop_aggregate_status": crop_aggregate.get("status"),
                "reason": "fixed max-search family must be complete; no denominator shrinking"}
    rows = [(name, detections[name]["existence_statistic"]) for name in FIXED_VIEWS]
    rows.append(("CROP_SEQUENCE_AGGREGATE", crop_aggregate["existence_statistic"]))
    if any(not math.isfinite(value) for _, value in rows):
        return {"status": "INVALID", "statistic": None, "winning_view": None, "reason": "nonfinite fixed-view statistic"}
    name, value = max(rows, key=lambda item: (item[1], item[0]))
    bindings = {(detections[view].get("receiver_protocol_id"), detections[view].get("key_id")) for view in FIXED_VIEWS}
    bindings.add((crop_aggregate.get("receiver_protocol_id"), crop_aggregate.get("key_id")))
    if len(bindings) != 1:
        return {"status": "INVALID", "statistic": None, "winning_view": None,
                "reason": "fixed search family protocol/key binding mismatch"}
    protocol_id, key_id = next(iter(bindings))
    return {"status": "SCORED", "statistic": float(value), "winning_view": name,
            "receiver_protocol_id": protocol_id, "key_id": key_id,
            "definition": "source-level max over fixed saved views and the fixed crop-sequence aggregate after payload/path maximization"}


def freeze_calibration(source_rows: list[dict], guard: float = 1e-6, *, protocol_id: str | None = None,
                       key_id: str | None = None) -> dict:
    if not math.isfinite(guard) or guard <= 0:
        raise ValueError("calibration guard must be positive and finite")
    protocol_id = protocol_id or RECEIVER_PROTOCOL_ID
    bindings_match = all(row.get("receiver_protocol_id") == protocol_id and row.get("key_id") == key_id for row in source_rows)
    if len(source_rows) != 2 or not bindings_match or any(row.get("status") != "SCORED" or not math.isfinite(row.get("statistic", math.nan)) for row in source_rows):
        return {"status": "UNCALIBRATED", "threshold": None, "sources": source_rows,
                "receiver_protocol_id": protocol_id, "key_id": key_id,
                "reason": "two complete independent OFF calibration sources with matching protocol and key binding required"}
    maximum = max(row["statistic"] for row in source_rows)
    return {"status": "FROZEN", "threshold": float(maximum + guard), "guard": guard, "sources": source_rows,
            "receiver_protocol_id": protocol_id, "key_id": key_id,
            "source_count": 2, "empirical_rank_resolution": "1/3", "claim": "functional independent-OFF calibration only; no low-FPR estimate"}


def decide(detection: dict, calibration: dict | None) -> dict:
    """Gate existence before returning a payload; ranking remains separately visible."""
    ranking = detection.get("best")
    if detection.get("status") != "SCORED" or ranking is None:
        return {"status": "INVALID", "payload": None}
    if not calibration or calibration.get("status") != "FROZEN":
        return {"status": "UNCALIBRATED", "payload": None, "ranked_payload": ranking["payload"]}
    if calibration.get("receiver_protocol_id") != detection.get("receiver_protocol_id") or calibration.get("key_id") != detection.get("key_id"):
        return {"status": "UNCALIBRATED", "payload": None, "ranked_payload": ranking["payload"],
                "reason": "calibration protocol/key binding mismatch"}
    threshold = calibration.get("threshold")
    statistic = detection.get("existence_statistic")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not isinstance(statistic, (int, float)) or not math.isfinite(statistic):
        return {"status": "INVALID", "payload": None, "reason": "nonfinite existence boundary"}
    if statistic <= threshold:
        return {"status": "REJECTED", "payload": None, "ranked_payload": ranking["payload"],
                "statistic": detection["existence_statistic"], "threshold": calibration["threshold"]}
    decoder = detection.get("decoder")
    if not detection.get("top_payload_unique") or not decoder or decoder.get("payload") is None or decoder["payload"] != ranking["payload"]:
        return {"status": "AMBIGUOUS", "payload": None, "ranked_payload": ranking["payload"], "decoder": decoder}
    return {"status": "DETECTED", "payload": decoder["payload"], "ranked_payload": ranking["payload"],
            "statistic": detection["existence_statistic"], "threshold": calibration["threshold"], "decoder": decoder}


assert RM_MIN_DISTANCE == 4
assert set(DATA_WINDOWS).isdisjoint(PILOT_WINDOWS)
assert sorted(DATA_WINDOWS + PILOT_WINDOWS) == list(range(WINDOWS))
