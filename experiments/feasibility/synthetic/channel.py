"""Synthetic video-specific relation channel."""

from __future__ import annotations

from dataclasses import dataclass
import random

from main.sc_sstw.linalg import Matrix, Vector, matvec


@dataclass(frozen=True)
class SyntheticChannel:
    """A synthetic q = A u + b + noise relation channel."""

    matrix: Matrix
    bias: Vector
    noise_std: float


def make_random_channel(
    seed: int,
    *,
    relation_count: int = 12,
    noise_std: float = 0.035,
) -> SyntheticChannel:
    if relation_count < 3:
        raise ValueError("relation_count must be at least 3")
    rng = random.Random(seed)
    first = [rng.uniform(-1.0, 1.0) for _ in range(relation_count)]
    second = [rng.uniform(-1.0, 1.0) for _ in range(relation_count)]
    matrix = [[first[index], second[index]] for index in range(relation_count)]
    bias = [rng.uniform(-0.25, 0.25) for _ in range(relation_count)]
    return SyntheticChannel(matrix=matrix, bias=bias, noise_std=noise_std)


def generate_observations(
    states: list[tuple[float, float]],
    channel: SyntheticChannel,
    *,
    seed: int,
) -> list[Vector]:
    rng = random.Random(seed)
    observations: list[Vector] = []
    for state in states:
        clean = matvec(channel.matrix, [state[0], state[1]])
        noisy = [
            value + bias + rng.gauss(0.0, channel.noise_std)
            for value, bias in zip(clean, channel.bias, strict=True)
        ]
        observations.append(noisy)
    return observations
