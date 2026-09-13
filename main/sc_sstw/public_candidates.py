"""Engineering-only public burst container; does not implement AISB acquisition.

Callers must supply a complete enumerated set, not legacy local top-k output.
No sequence search, calibration, truth evaluation or private scoring occurs here.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math


@dataclass(frozen=True)
class PublicBurst:
    template_id: str
    start: int
    observed_length: int
    template_length: int
    missing_indices: tuple[int, ...]
    residual: float

    def __post_init__(self):
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be nonempty")
        ints = (self.start, self.observed_length, self.template_length, *self.missing_indices)
        if any(type(v) is not int for v in ints):
            raise ValueError("indices and lengths must be integers")
        if self.start < 0 or self.observed_length < 1 or self.template_length < 1:
            raise ValueError("invalid span")
        if tuple(sorted(set(self.missing_indices))) != self.missing_indices:
            raise ValueError("missing indices must be sorted and unique")
        if any(i < 0 or i >= self.template_length for i in self.missing_indices):
            raise ValueError("missing index outside template")
        if self.observed_length + len(self.missing_indices) != self.template_length:
            raise ValueError("span does not reconcile")
        if isinstance(self.residual, bool) or not math.isfinite(self.residual) or self.residual < 0:
            raise ValueError("residual must be finite and nonnegative")
        object.__setattr__(self, "residual", float(self.residual) if self.residual else 0.0)

    def sort_key(self):
        return (self.residual, self.template_id, self.start, self.observed_length,
                self.template_length, self.missing_indices)

    def identity(self):
        return (self.template_id, self.start, self.observed_length,
                self.template_length, self.missing_indices)


def canonical_bytes(payload):
    """stage1-json-v1: UTF-8, sorted keys, compact JSON, finite binary64 numbers.

    Python JSON shortest-roundtrip number encoding; no rounding/quantization.
    This protocol is deliberately not a cross-language RFC8785 claim.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def freeze_public_bursts(candidates, *, budget: int, enumeration_complete: bool):
    """Globally rank a finite complete burst enumeration, then truncate once.

    Budget and rank are an explicit opt-in engineering policy, not a selected
    scientific acquisition protocol. Duplicate identities are rejected.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("positive explicit budget required")
    if enumeration_complete is not True:
        raise ValueError("cannot freeze unknown or locally prepruned enumeration")
    ordered = sorted(candidates, key=lambda c: c.sort_key())
    if len({c.identity() for c in ordered}) != len(ordered):
        raise ValueError("duplicate candidate identity")
    retained = ordered[:budget]
    payload = {
        "schema": "stage1-public-bursts-v1", "encoding": "stage1-json-v1",
        "evidence": "engineering_only", "semantics": "burst_set_only",
        "sequence_coverage": "UNDETERMINED",
        "compatibility": "half_open_observed_intervals_disjoint_only_not_a_sequence",
        "ranking": "residual,template_id,start,observed_length,template_length,missing_indices",
        "enumeration_complete": True, "budget": budget,
        "before_count": len(ordered), "after_count": len(retained),
        "dropped_count": len(ordered) - len(retained),
        "truncation_reason": "global_budget" if len(ordered) > budget else "none",
        "candidates": [asdict(c) for c in retained],
    }
    return canonical_bytes(payload)


def candidate_digest(frozen: bytes):
    read_frozen_bursts(frozen)
    return hashlib.sha256(frozen).hexdigest()


def read_frozen_bursts(frozen: bytes):
    """Strict readback: reject noncanonical encoding or inconsistent metadata."""
    payload = json.loads(frozen)
    candidates = [PublicBurst(**dict(c, missing_indices=tuple(c["missing_indices"])))
                  for c in payload["candidates"]]
    expected = json.loads(freeze_public_bursts(candidates, budget=payload["budget"],
                                             enumeration_complete=True))
    before = payload["before_count"]
    if type(before) is not int or before < len(candidates):
        raise ValueError("invalid before_count")
    if len(candidates) != min(before, payload["budget"]):
        raise ValueError("budget/count mismatch")
    expected.update(before_count=before, dropped_count=before - len(candidates),
                    truncation_reason="global_budget" if before > len(candidates) else "none")
    if payload != expected or canonical_bytes(payload) != frozen:
        raise ValueError("noncanonical or inconsistent frozen object")
    return tuple(candidates)


def non_overlapping(left: PublicBurst, right: PublicBurst):
    """Only temporal compatibility; does not construct or rank AISB sequences."""
    return (left.start + left.observed_length <= right.start or
            right.start + right.observed_length <= left.start)
