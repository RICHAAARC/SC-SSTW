"""Window/axis state-MSE target for the fixed-key writer.

This writer-side surrogate uses the existing 1,760 fixed supports.  It does not
use receiver clipping, receiver search, or any A/B receiver partition.
"""
from __future__ import annotations

import torch

from main.tube_state import fixed_key

OBJECTIVE_ID = "window-state-mse-v1"
WINDOWS = 11
SUPPORTS_PER_WINDOW = 160
BLOCKS_PER_AXIS = 80


def state_means_from_projections(projections, book):
    """Return t[w, a] = mean_axis(tanh(p_i) * sync_i * polarity_i)."""
    if projections.numel() != WINDOWS * SUPPORTS_PER_WINDOW:
        raise ValueError("expected the fixed 11-by-160 projection family")
    sync = torch.as_tensor(book["sync"], dtype=torch.float64, device=projections.device)
    polarity = torch.as_tensor(book["polarity"], dtype=torch.float64, device=projections.device)
    signed = projections.reshape(WINDOWS, SUPPORTS_PER_WINDOW).tanh()
    signed = signed * (sync * polarity).reshape(WINDOWS, SUPPORTS_PER_WINDOW)
    return torch.stack((signed[:, 0::2].mean(dim=1), signed[:, 1::2].mean(dim=1)), dim=1)


def state_means(z, book):
    directions = torch.as_tensor(book["directions"], device=z.device)
    return state_means_from_projections(fixed_key.torch_projections(z, directions), book)


def loss_from_projections(projections, book):
    target = torch.as_tensor(book["states"], dtype=torch.float64, device=projections.device)
    values = state_means_from_projections(projections, book)
    return (values - target).square().mean(), values


def loss(z, book):
    directions = torch.as_tensor(book["directions"], device=z.device)
    projections = fixed_key.torch_projections(z, directions)
    return loss_from_projections(projections, book)


def identity():
    return {
        "id": OBJECTIVE_ID,
        "formula": "mean_22((mean_80(tanh(p)*sync*polarity)-states)^2)",
        "temperature": 1,
        "support_count": WINDOWS * SUPPORTS_PER_WINDOW,
        "states": "existing +/-1 fixed codebook states",
        "receiver_clipping_used": False,
    }
