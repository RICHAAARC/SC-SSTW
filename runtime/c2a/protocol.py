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
    _, _, _, height, width = normalized_latent.shape
    top, left = (int(height) - config.block_size) // 2, (int(width) - config.block_size) // 2
    return _support_at(normalized_latent, group, top_left=(top, left), config=config)


def _support_at(normalized_latent: Any, group: int, *, top_left: tuple[int, int], config: C2AConfig) -> Any:
    """Return one named 8x8 support without changing the central-block API."""

    if normalized_latent.ndim != 5 or normalized_latent.shape[0] != 1:
        raise ValueError("C2A requires one normalized latent with layout [1,C,T,H,W]")
    _, channels, groups, height, width = normalized_latent.shape
    if channels < 2 or group >= groups:
        raise ValueError("C2A latent lacks the fixed channel or ordinary-group support")
    top, left = (int(top_left[0]), int(top_left[1]))
    if top < 0 or left < 0 or top + config.block_size > height or left + config.block_size > width:
        raise ValueError("C2A block top-left lies outside the latent spatial geometry")
    rows, cols = slice(top, top + config.block_size), slice(left, left + config.block_size)
    # A slice, rather than advanced channel indexing, is required here: the
    # writer must update the cloned latent in place.
    return normalized_latent[0, 0:2, group, rows, cols]


def _covariance_and_q(block: Any) -> tuple[Any, Any]:
    """Measure the empirical 2x2 covariance of an *actual* support block.

    The block may be FP32 (as it is after ``copy_`` in the writer).  Values are
    promoted only for the measurement; no analytic transport identity is used
    as a substitute for this post-write calculation.
    """

    torch = _torch()
    x = block.reshape(2, -1).to(dtype=torch.float64)
    centered = x - x.mean(dim=1, keepdim=True)
    covariance = centered @ centered.transpose(0, 1) / 64.0
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("NONFINITE_READ_COVARIANCE")
    q = torch.stack(((covariance[0, 0] - covariance[1, 1]) / 2.0, covariance[0, 1]))
    return covariance, q


def _target_covariance(state: tuple[float, float], *, config: C2AConfig) -> Any:
    torch = _torch()
    return torch.tensor(
        ((1.0 + config.beta * state[0], config.beta * state[1]),
         (config.beta * state[1], 1.0 - config.beta * state[0])),
        dtype=torch.float64,
    )


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
    sigma = _target_covariance(state, config=config).to(device=written.device)
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
            "pre_write_covariance_lambda_min": float(eigenvalues[0].item()),
            "pre_write_covariance_lambda_max": float(eigenvalues[-1].item()),
            "analytic_transport_target_max_abs_error_float64": float((check - sigma).abs().max().item()),
        })
    return written, records


def read_q(normalized_latent: Any, *, config: C2AConfig = C2AConfig()) -> Any:
    """Read the fixed traceless covariance coordinates from the middle group."""

    config.validate()
    block = _support(normalized_latent, config.read_group, config)
    return _covariance_and_q(block)[1]


def actual_write_measurement(normalized_latent: Any, state: tuple[float, float], *, config: C2AConfig = C2AConfig()) -> dict[str, Any]:
    """Measure each written FP32 block after the writer has actually copied it.

    This deliberately records the covariance and q of groups 1, 2, and 3 from
    the tensor that will be decoded.  ``analytic_transport_*`` values from the
    writer are separate diagnostics and must not be read as this measurement.
    """

    config.validate()
    target = _target_covariance(state, config=config)
    target_q = ((target[0, 0] - target[1, 1]) / 2.0, target[0, 1])
    groups: list[dict[str, Any]] = []
    for group in config.ordinary_groups:
        covariance, q = _covariance_and_q(_support(normalized_latent, group, config))
        target_on_measurement_device = target.to(device=covariance.device)
        groups.append({
            "group": group,
            "actual_covariance_after_fp32_write": covariance.detach().cpu().tolist(),
            "actual_q_after_fp32_write": q.detach().cpu().tolist(),
            "actual_target_max_abs_error": float((covariance - target_on_measurement_device).abs().max().item()),
        })
    return {
        "target_covariance": target.detach().cpu().tolist(),
        "target_q": [float(target_q[0].item()), float(target_q[1].item())],
        "groups": groups,
        "read_group": config.read_group,
        "middle_group_q_after_fp32_write": next(row["actual_q_after_fp32_write"] for row in groups if row["group"] == config.read_group),
    }


def actual_written_block_snapshots(normalized_latent: Any, *, config: C2AConfig = C2AConfig()) -> dict[str, Any]:
    """Copy exactly the decoded support blocks to CPU without changing dtype."""

    config.validate()
    return {
        f"group_{group}": _support(normalized_latent, group, config).detach().to(device="cpu").contiguous().clone()
        for group in config.ordinary_groups
    }


def read_q_at_block(normalized_latent: Any, *, top_left: tuple[int, int], config: C2AConfig = C2AConfig()) -> Any:
    """Read q from a fixed non-central support using the same middle group."""

    config.validate()
    return _covariance_and_q(_support_at(normalized_latent, config.read_group, top_left=top_left, config=config))[1]


def write_covariance_state_at_blocks(normalized_latent: Any, state: tuple[float, float], *, blocks: tuple[tuple[int, int], ...], config: C2AConfig = C2AConfig()) -> tuple[Any, list[dict[str, Any]]]:
    """Write the same fixed state independently in each predeclared support."""

    torch = _torch()
    config.validate()
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("C2A multiblock layout requires unique nonempty block locations")
    if max(abs(float(value)) for value in state) > config.rho:
        raise ValueError("state is outside the fixed rho ball")
    written = normalized_latent.detach().clone()
    sigma = _target_covariance(state, config=config).to(device=written.device)
    records: list[dict[str, Any]] = []
    for block_index, top_left in enumerate(blocks):
        for group in config.ordinary_groups:
            block = _support_at(written, group, top_left=top_left, config=config)
            original_dtype = block.dtype
            x = block.reshape(2, -1).to(dtype=torch.float64)
            centered = x - x.mean(dim=1, keepdim=True)
            covariance = centered @ centered.transpose(0, 1) / 64.0
            root = _symmetric_sqrt(covariance, config=config)
            inverse_root = _symmetric_sqrt(covariance, inverse=True, config=config)
            transform = inverse_root @ _symmetric_sqrt(root @ sigma @ root, config=config) @ inverse_root
            block.copy_((transform @ centered + x.mean(dim=1, keepdim=True)).reshape_as(block).to(dtype=original_dtype))
            analytic = transform @ covariance @ transform.transpose(0, 1)
            actual_covariance, actual_q = _covariance_and_q(block)
            eigenvalues = torch.linalg.eigvalsh(covariance)
            records.append({
                "block_index": block_index,
                "top_left": [int(top_left[0]), int(top_left[1])],
                "group": group,
                "pre_write_covariance_lambda_min": float(eigenvalues[0].item()),
                "pre_write_covariance_lambda_max": float(eigenvalues[-1].item()),
                "analytic_transport_target_max_abs_error_float64": float((analytic - sigma).abs().max().item()),
                "actual_covariance_after_fp32_write": actual_covariance.detach().cpu().tolist(),
                "actual_q_after_fp32_write": actual_q.detach().cpu().tolist(),
                "actual_target_max_abs_error": float((actual_covariance - sigma).abs().max().item()),
            })
    return written, records


def written_block_snapshots_at_blocks(normalized_latent: Any, *, blocks: tuple[tuple[int, int], ...], config: C2AConfig = C2AConfig()) -> dict[str, Any]:
    """Persist exactly the blocks that are later decoded, grouped by location."""

    config.validate()
    return {
        f"block_{index}_{top}_{left}": {
            f"group_{group}": _support_at(normalized_latent, group, top_left=(top, left), config=config).detach().to(device="cpu").contiguous().clone()
            for group in config.ordinary_groups
        }
        for index, (top, left) in enumerate(blocks)
    }


def latent_change_metrics_at_blocks(reference: Any, changed: Any, *, blocks: tuple[tuple[int, int], ...], config: C2AConfig = C2AConfig()) -> dict[str, Any]:
    """Report dimensionless normalized-latent L2 costs for a fixed layout."""

    torch = _torch()
    before = torch.cat([_support_at(reference, group, top_left=top_left, config=config).reshape(-1) for top_left in blocks for group in config.ordinary_groups])
    after = torch.cat([_support_at(changed, group, top_left=top_left, config=config).reshape(-1) for top_left in blocks for group in config.ordinary_groups])

    def metrics(left: Any, right: Any) -> tuple[float, float | None]:
        delta_l2 = float((right - left).detach().float().norm().item())
        baseline_l2 = float(left.detach().float().norm().item())
        return delta_l2, None if baseline_l2 == 0.0 else delta_l2 / baseline_l2

    local_l2, local_relative = metrics(before, after)
    global_l2, global_relative = metrics(reference.reshape(-1), changed.reshape(-1))
    return {
        "coordinate_system": "normalized_latent",
        "local_support_delta_l2": local_l2,
        "local_support_relative_l2_dimensionless": local_relative,
        "whole_latent_delta_l2": global_l2,
        "whole_latent_relative_l2_dimensionless": global_relative,
        "spatial_blocks": len(blocks),
    }


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
