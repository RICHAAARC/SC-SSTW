"""Pure NumPy Wan terminal projection candidate. No source/attack metadata in read."""
from __future__ import annotations
import hashlib
import hmac
import random
from fractions import Fraction
import numpy as np

NAMESPACE = b"SC-SSTW/WanProjectionMargin/v2"
SHAPE = (1, 16, 46, 40, 64)
LENGTH, PATCH, MARGIN = 4, 4, 1.0
SUPPORT_COUNT = 1760
MESSAGES = (0, 1)
SCALES = ((4, 5), (1, 1), (5, 4))
OFFSETS = tuple(range(-8, 9))  # Source RGB frames, not latent groups.

def digest(key: bytes, role: str, index: int) -> bytes:
    return hmac.new(key, NAMESPACE + b"/" + role.encode() + b"/" + str(index).encode(), hashlib.sha256).digest()

def sign(key: bytes, role: str, index: int) -> int:
    return 1 if digest(key, role, index)[0] & 1 else -1

def codebook(key: bytes) -> dict:
    directions = []
    for index in range(SUPPORT_COUNT):
        rng = random.Random(int.from_bytes(digest(key, "direction", index)[:8], "big"))
        vector = np.array([rng.uniform(-1, 1) for _ in range(1024)], dtype=np.float64)
        directions.append((vector / np.linalg.norm(vector)).astype(np.float32))
    sync = np.array([sign(key, "sync", 1 + 4 * (i // 160)) for i in range(SUPPORT_COUNT)])
    payload = np.array([[sign(key, f"payload/message{m}", i) for i in range(SUPPORT_COUNT)] for m in MESSAGES])
    return {"directions": np.stack(directions), "sync": sync, "payload": payload, "codes": payload * sync[None, :]}

def blocks(z: np.ndarray) -> np.ndarray:
    """Protocol array -> [11,160,1024], each support flattened T,C,H,W."""
    if tuple(z.shape) != SHAPE:
        raise ValueError(f"protocol terminal shape must be {SHAPE}, got {z.shape}")
    x = z[0, :, 1:45].transpose(1, 0, 2, 3)
    return x.reshape(11, 4, 16, 10, 4, 16, 4).transpose(0, 3, 5, 1, 2, 4, 6).reshape(11, 160, 1024)

def put_blocks(z: np.ndarray, values: np.ndarray) -> None:
    x = values.reshape(11, 10, 16, 4, 16, 4, 4).transpose(0, 3, 4, 1, 5, 2, 6).reshape(44, 16, 40, 64)
    z[0, :, 1:45] = x.transpose(1, 0, 2, 3)

def write(z: np.ndarray, book: dict, message: int) -> tuple[np.ndarray, dict]:
    if message not in MESSAGES:
        raise ValueError("only fixed message candidates 0 and 1")
    out = np.array(z, dtype=np.float32, copy=True)
    x = blocks(out).reshape(SUPPORT_COUNT, -1).astype(np.float64)
    d = book["directions"].astype(np.float64)
    c = book["codes"][message]
    before = np.einsum("ij,ij->i", x, d)
    norm2 = np.einsum("ij,ij->i", d, d)
    step = np.maximum(0, MARGIN - c * before) / norm2
    x += (step * c)[:, None] * d
    put_blocks(out, x.astype(np.float32))
    after = np.einsum("ij,ij->i", blocks(out).reshape(SUPPORT_COUNT, -1).astype(np.float64), d)
    return out, {"projection_before": before.tolist(), "projection_after": after.tolist(),
                 "signed_projection_after": (c * after).tolist(), "delta_l2": step.tolist(),
                 "minimum_signed_projection_after": float(np.min(c * after)),
                 "support_count": SUPPORT_COUNT, "margin": MARGIN}

def allocation(count: int, g: int, numerator: int, denominator: int, offset: int) -> list:
    """Center-nearest, ties earlier, no observation reuse; max error two RGB frames.

    Reference group r center=2.5+4*r. Affine map acts on received RGB centers.
    This is a group approximation, not an inverse temporal VAE or deletion path.
    """
    scale = Fraction(numerator, denominator)
    centers = [scale * Fraction(2 * g + 5 + 8 * j, 2) + offset for j in range(count)]
    used, selected = set(), []
    for r in range(44):
        target = Fraction(5 + 8 * r, 2)
        choices = [(abs(center - target), j) for j, center in enumerate(centers)
                   if j not in used and abs(center - target) <= 2]
        if choices:
            _, j = min(choices)
            used.add(j)
            selected.append(j)
        else:
            selected.append(None)
    return selected

def projections(z: np.ndarray, selected: list, book: dict) -> tuple:
    values = np.zeros(SUPPORT_COUNT, dtype=np.float64)
    valid = np.zeros(SUPPORT_COUNT, dtype=bool)
    directions = book["directions"].reshape(11, 160, 1024).astype(np.float64)
    for temporal in range(11):
        js = selected[4 * temporal:4 * temporal + 4]
        if any(j is None for j in js):
            continue
        block = np.take(z[0], [1 + j for j in js], axis=1).transpose(1, 0, 2, 3)
        block = block.reshape(4, 16, 10, 4, 16, 4).transpose(2, 4, 0, 1, 3, 5).reshape(160, 1024)
        sl = slice(160 * temporal, 160 * (temporal + 1))
        values[sl] = np.einsum("ij,ij->i", block.astype(np.float64), directions[temporal])
        valid[sl] = True
    return values, valid

def score(values: np.ndarray, valid: np.ndarray, book: dict) -> list:
    result = []
    for m in MESSAGES:
        aligned = book["codes"][m] * values
        result.append({"message": m, "score": float(np.clip(aligned[valid], -1, 1).sum() / SUPPORT_COUNT) if valid.any() else None,
                       "matched_supports": int(valid.sum()), "coverage": float(valid.mean()),
                       "aligned_payload_agreement": float((aligned[valid] > 0).mean()) if valid.any() else None,
                       "mean_signed_projection": float(aligned[valid].mean()) if valid.any() else None})
    return result

def read(observations: dict, book: dict) -> dict:
    """Identical fixed 204 time candidates for every video, two message templates."""
    rows, classes, cache = [], {}, {}
    for g in range(4):
        for num, den in SCALES:
            for b in OFFSETS:
                if g not in observations:
                    rows.append({"g": g, "scale": [num, den], "offset": b, "status": "MISSING_OBSERVATION", "scores": []})
                    continue
                z = observations[g]
                selected = allocation(z.shape[2] - 1, g, num, den, b)
                signature = (g, tuple(selected))
                if signature not in classes:
                    classes[signature] = len(classes)
                    values, valid = projections(z, selected, book)
                    cache[signature] = score(values, valid, book)
                rows.append({"g": g, "scale": [num, den], "offset": b,
                             "status": "SCORED" if cache[signature][0]["score"] is not None else "NO_COMPLETE_TUBELETS",
                             "class": classes[signature], "selected_ordinary_groups": selected, "scores": cache[signature]})
    def rank_key(pair):
        r, s = pair
        return (-s["score"], abs(r["offset"]), abs(r["scale"][0] / r["scale"][1] - 1), r["g"], r["offset"], r["scale"], s["message"])
    pairs = sorted([(r, s) for r in rows for s in r["scores"] if s["score"] is not None], key=rank_key)
    def compact(pair):
        r, s = pair
        return {k: r[k] for k in ("g", "scale", "offset", "class")} | s
    best_by_message = {str(m): next((compact(p) for p in pairs if p[1]["message"] == m), None) for m in MESSAGES}
    identity = next(r for r in rows if r["g"] == 0 and r["scale"] == [1, 1] and r["offset"] == 0)
    best_score = pairs[0][1]["score"] if pairs else None
    tied = [compact(pair) for pair in pairs if abs(pair[1]["score"] - best_score) <= 1e-12]
    identity_class = identity.get("class")
    other_time = {str(m): next((compact(pair) for pair in pairs if pair[1]["message"] == m and pair[0]["class"] != identity_class), None) for m in MESSAGES}
    return {"candidate_count": len(rows), "effective_class_count": len(classes), "candidates": rows,
            "best": compact(pairs[0]) if pairs else None, "best_by_message": best_by_message,
            "top_score_ties": tied, "top_message_unique": len({r['message'] for r in tied}) == 1,
            "best_outside_identity_class": other_time,
            "identity_path": identity, "decision": "RANKING_ONLY_NO_ACCEPTANCE_THRESHOLD"}
