"""Fixed C2 2A covariance carrier and non-model evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from typing import Any


ARM_STATES: dict[str, tuple[float, float] | None] = {
    "OFF": None,
    "ZERO": (0.0, 0.0),
    "PLUS_E1": (0.5, 0.0),
    "MINUS_E1": (-0.5, 0.0),
    "PLUS_E2": (0.0, 0.5),
    "MINUS_E2": (0.0, -0.5),
}


@dataclass(frozen=True)
class C2AConfig:
    """Predeclared 2A choices in normalized Wan-latent coordinates."""

    channels: tuple[int, int] = (0, 1)
    block_size: int = 8
    beta: float = 0.25
    rho: float = 0.5
    # Index zero is the VAE special group.  The three ordinary groups are
    # written identically and index two is the sole C2A read location.
    ordinary_groups: tuple[int, int, int] = (1, 2, 3)
    read_group: int = 2
    # No regularization or pseudo-inverse is permitted.  This is a numerical
    # support test, not a result-selected tuning value.
    singular_relative_eigenvalue: float = 1e-8

    def validate(self) -> None:
        if self.channels != (0, 1):
            raise ValueError("C2A fixes zero-based normalized-latent channels (0, 1)")
        if self.block_size != 8 or self.block_size % 2:
            raise ValueError("C2A fixes an even 8x8 spatial block")
        if self.beta != 0.25 or self.rho != 0.5:
            raise ValueError("C2A fixes beta=1/4 and rho=1/2")
        if self.ordinary_groups != (1, 2, 3) or self.read_group != 2:
            raise ValueError("C2A fixes ordinary groups (1,2,3) and reads the middle group")
        if self.singular_relative_eigenvalue <= 0:
            raise ValueError("singular support threshold must be positive")


def central_block_slices(height: int, width: int, block_size: int = 8) -> tuple[slice, slice]:
    """Use floor((size-8)/2), fixed before any observation is known."""

    if height < block_size or width < block_size:
        raise ValueError("latent spatial geometry cannot contain the fixed central 8x8 support")
    top = (height - block_size) // 2
    left = (width - block_size) // 2
    return slice(top, top + block_size), slice(left, left + block_size)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("C2A write/read primitives require the optional torch runtime") from exc
    return torch


def _symmetric_sqrt(matrix: Any, *, inverse: bool = False, config: C2AConfig) -> Any:
    torch = _torch()
    values, vectors = torch.linalg.eigh(matrix.to(dtype=torch.float64))
    if not bool(torch.isfinite(values).all()):
        raise ValueError("NONFINITE_COVARIANCE")
    largest = float(values[-1].item())
    smallest = float(values[0].item())
    if smallest <= config.singular_relative_eigenvalue * max(largest, 1.0):
        raise ValueError("SINGULAR_OR_NEAR_SINGULAR_COVARIANCE")
    factors = values.rsqrt() if inverse else values.sqrt()
    return (vectors * factors.unsqueeze(0)) @ vectors.transpose(-1, -2)


def _support(normalized_latent: Any, group: int, config: C2AConfig) -> Any:
    if normalized_latent.ndim != 5 or normalized_latent.shape[0] != 1:
        raise ValueError("C2A requires one normalized latent with layout [1,C,T,H,W]")
    _, channels, groups, height, width = normalized_latent.shape
    if channels < 2 or group >= groups:
        raise ValueError("C2A latent lacks the fixed channel or ordinary-group support")
    rows, cols = central_block_slices(int(height), int(width), config.block_size)
    # A slice, rather than advanced channel indexing, is required here: the
    # writer must update the cloned latent in place.
    return normalized_latent[0, 0:2, group, rows, cols]


def write_covariance_state(normalized_latent: Any, state: tuple[float, float], *, config: C2AConfig = C2AConfig()) -> tuple[Any, list[dict[str, float | int]]]:
    """Apply the documented SPD OT mapping to all three hold groups.

    The input is never mutated.  Nonpositive/near-singular empirical covariance
    is a retained unsupported arm, not an invitation to add a regularizer.
    """

    torch = _torch()
    config.validate()
    if max(abs(float(value)) for value in state) > config.rho:
        raise ValueError("state is outside the fixed rho ball")
    written = normalized_latent.detach().clone()
    records: list[dict[str, float | int]] = []
    sigma = torch.tensor(
        ((1.0 + config.beta * state[0], config.beta * state[1]),
         (config.beta * state[1], 1.0 - config.beta * state[0])),
        device=written.device,
        dtype=torch.float64,
    )
    for group in config.ordinary_groups:
        block = _support(written, group, config)
        original_dtype = block.dtype
        x = block.reshape(2, -1).to(dtype=torch.float64)
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        covariance = centered @ centered.transpose(0, 1) / 64.0
        root = _symmetric_sqrt(covariance, config=config)
        inverse_root = _symmetric_sqrt(covariance, inverse=True, config=config)
        middle = root @ sigma @ root
        transform = inverse_root @ _symmetric_sqrt(middle, config=config) @ inverse_root
        updated = (transform @ centered + mean).reshape_as(block).to(dtype=original_dtype)
        block.copy_(updated)
        check = (transform @ covariance @ transform.transpose(0, 1)).detach()
        eigenvalues = torch.linalg.eigvalsh(covariance)
        records.append({
            "group": group,
            "covariance_lambda_min": float(eigenvalues[0].item()),
            "covariance_lambda_max": float(eigenvalues[-1].item()),
            "post_target_max_abs_error": float((check - sigma).abs().max().item()),
        })
    return written, records


def read_q(normalized_latent: Any, *, config: C2AConfig = C2AConfig()) -> Any:
    """Read the fixed traceless covariance coordinates from the middle group."""

    torch = _torch()
    config.validate()
    block = _support(normalized_latent, config.read_group, config)
    x = block.reshape(2, -1).to(dtype=torch.float64)
    centered = x - x.mean(dim=1, keepdim=True)
    covariance = centered @ centered.transpose(0, 1) / 64.0
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("NONFINITE_READ_COVARIANCE")
    return torch.stack(((covariance[0, 0] - covariance[1, 1]) / 2.0, covariance[0, 1]))


def latent_change_metrics(reference: Any, changed: Any, *, config: C2AConfig = C2AConfig()) -> dict[str, float | None]:
    """Report support and whole-latent relative changes with explicit zero rules."""

    torch = _torch()
    before_support = torch.cat([_support(reference, group, config).reshape(-1) for group in config.ordinary_groups])
    after_support = torch.cat([_support(changed, group, config).reshape(-1) for group in config.ordinary_groups])

    def relative(before: Any, after: Any) -> float | None:
        denominator = float(before.detach().float().norm().item())
        return None if denominator == 0.0 else float((after - before).detach().float().norm().item() / denominator)

    return {
        "support_relative_l2": relative(before_support, after_support),
        "whole_latent_relative_l2": relative(reference.reshape(-1), changed.reshape(-1)),
    }


def rgb_quality_metrics(reference: Any, candidate: Any) -> dict[str, float | None]:
    """Fixed full-frame [0,1] RGB metrics; no candidate numerical gate is applied."""

    torch = _torch()
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError("RGB comparison requires equal [T,H,W,3] clips")
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    mse = float(difference.square().mean().item())
    psnr = None if mse == 0.0 else 10.0 * log10(1.0 / mse)
    if reference.shape[0] < 2:
        temporal_ratio = None
    else:
        base = reference[1:].to(torch.float64) - reference[:-1].to(torch.float64)
        altered = candidate[1:].to(torch.float64) - candidate[:-1].to(torch.float64)
        baseline_energy = float(base.square().mean().item())
        temporal_ratio = None if baseline_energy == 0.0 else float(altered.square().mean().item() / baseline_energy)
    return {"rgb_mse": mse, "rgb_psnr_db": psnr, "temporal_difference_energy_ratio": temporal_ratio}


def axis_response_summary(q_by_arm: dict[str, Any], *, rho: float = 0.5) -> dict[str, Any]:
    """Fit only ZERO/+e1/+e2; keep -e1/-e2 as genuine signed holdouts."""

    torch = _torch()
    if set(q_by_arm) != set(ARM_STATES):
        raise ValueError("axis response summary requires every fixed C2A arm")
    q0, plus1, plus2 = (q_by_arm[name].to(torch.float64) for name in ("ZERO", "PLUS_E1", "PLUS_E2"))
    response = torch.stack(((plus1 - q0) / rho, (plus2 - q0) / rho), dim=1)
    singular_values = torch.linalg.svdvals(response)
    holdouts: dict[str, float] = {}
    for minus, plus in (("MINUS_E1", "PLUS_E1"), ("MINUS_E2", "PLUS_E2")):
        predicted = 2.0 * q0 - q_by_arm[plus].to(torch.float64)
        holdouts[minus] = float((q_by_arm[minus].to(torch.float64) - predicted).norm().item())
    return {
        "fit_arms": ["ZERO", "PLUS_E1", "PLUS_E2"],
        "holdout_arms": ["MINUS_E1", "MINUS_E2"],
        "response_matrix": response.detach().cpu().tolist(),
        "response_singular_values": singular_values.detach().cpu().tolist(),
        "response_rank_numeric": int(torch.linalg.matrix_rank(response).item()),
        "holdout_prediction_l2": holdouts,
        "formal_acceptance": "UNSET_NO_NOISE_REPLICATES_OR_USER_APPROVED_THRESHOLD",
    }
