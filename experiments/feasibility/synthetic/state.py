"""Key-conditioned 2D state trajectory and public pilot schedule."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random


Vector2 = tuple[float, float]


PILOT_DIRECTIONS: tuple[Vector2, ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
)


PUBLIC_SYNC_DIRECTIONS: tuple[Vector2, ...] = tuple(
    (math.cos(angle), math.sin(angle))
    for angle in (
        0.00,
        1.17,
        2.71,
        4.38,
        0.62,
        3.44,
        5.41,
        2.03,
        4.91,
        1.81,
        3.93,
        0.29,
        5.77,
        2.36,
        4.64,
        1.02,
    )
)


@dataclass(frozen=True)
class PilotPattern:
    """Periodic public pilot pattern.

    `period` controls pilot spacing. Pilot windows use the four public cardinal
    directions in a fixed cycle. Non-pilot windows carry the secret state.
    """

    period: int = 8
    directions: tuple[Vector2, ...] = PILOT_DIRECTIONS
    cyclic: bool = True

    def pilot_at(self, index: int) -> Vector2 | None:
        if self.period <= 0:
            raise ValueError("pilot period must be positive")
        if not self.directions:
            raise ValueError("pilot directions must not be empty")
        if index % self.period != 0:
            return None
        pilot_ordinal = index // self.period
        if self.cyclic:
            return self.directions[pilot_ordinal % len(self.directions)]
        if pilot_ordinal >= len(self.directions):
            return None
        return self.directions[pilot_ordinal]


def _key_bit(key: str, index: int) -> int:
    digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).digest()
    return 1 if digest[0] & 1 else -1


def _random_initial_phase(key: str) -> float:
    digest = hashlib.sha256(f"{key}:phase".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed).random() * 2.0 * math.pi


def generate_state_sequence(
    key: str,
    length: int,
    *,
    pilot_pattern: PilotPattern | None = None,
    base_frequency: float = 0.37,
    phase_step: float = 0.41,
) -> list[Vector2]:
    """Generate public-pilot plus key-conditioned 2D unit-state sequence."""

    if length <= 0:
        raise ValueError("length must be positive")
    pattern = pilot_pattern or PilotPattern()
    phase = _random_initial_phase(key)
    states: list[Vector2] = []
    for index in range(length):
        pilot = pattern.pilot_at(index)
        if pilot is not None:
            states.append(pilot)
        else:
            phase += base_frequency + phase_step * _key_bit(key, index)
            states.append((math.cos(phase), math.sin(phase)))
    return states


def pilot_indices(length: int, pattern: PilotPattern) -> list[int]:
    """Return indices that carry public pilots."""

    return [index for index in range(length) if pattern.pilot_at(index) is not None]
