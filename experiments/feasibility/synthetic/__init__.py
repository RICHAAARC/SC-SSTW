"""Constructed-channel support for synthetic-only feasibility experiments."""

from .channel import SyntheticChannel, generate_observations, make_random_channel
from .state import PUBLIC_SYNC_DIRECTIONS, PilotPattern, generate_state_sequence

__all__ = [
    "PUBLIC_SYNC_DIRECTIONS",
    "PilotPattern",
    "SyntheticChannel",
    "generate_observations",
    "generate_state_sequence",
    "make_random_channel",
]
