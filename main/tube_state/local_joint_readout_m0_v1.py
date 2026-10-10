"""Adopted M0: differentiable original pooled-energy readout and five gradients.

The final observer remains local_joint_terminal_bridge_v1; this module supplies
the same mathematics without converting q or objective tensors to Python floats.
"""
from __future__ import annotations

from itertools import combinations

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier

VIEWS = ("BASE", "BRIDGE", "MEAN", "COMMON", "FD_MINUS", "FD_PLUS")
OBJECTIVES = ("state_gap", "fragment_0_min", "fragment_1_min", "fragment_2_min", "fragment_3_min")
H = 1 / 64


class UndefinedActiveTie(ValueError):
    """Adopted construction is undefined, rather than a decoder execution error."""


def tensor_readout(rgb, *, key, message, protocol=carrier.PUBLIC):
    """88 windows, each pooling all frame/tile energies before division; no epsilon.

    Nonfinite or zero pooled support is undefined, just as in the original reader.
    Equality tests select the adopted unique active branches, never a surrogate.
    """
    import torch

    if tuple(rgb.shape) != tuple(protocol.video_shape):
        raise ValueError("RGB shape differs from carrier protocol")
    rows = []
    for slot in range(22):
        roi_rows = []
        start = protocol.segment_start + slot * protocol.segment_frames
        for roi_index, (y0, y1, x0, x1) in enumerate(protocol.rois):
            pixels = rgb[start:start + protocol.segment_frames, y0:y1, x0:x1].double()
            y = .299*pixels[..., 0] + .587*pixels[..., 1] + .114*pixels[..., 2]
            tiles = y.reshape(len(y), (y1-y0)//8, 8, (x1-x0)//8, 8).permute(0, 1, 3, 2, 4).reshape(-1, 8, 8)
            coefficients = carrier.dct2_ortho(tiles)
            chips = []
            for p, m in carrier.coefficient_pairs(key, roi_index):
                a, b = coefficients[:, p[0], p[1]], coefficients[:, m[0], m[1]]
                ep, em = a.square().sum(), b.square().sum()
                total = ep + em
                if not bool(torch.isfinite(total)) or not bool(total > 0):
                    raise FloatingPointError("undefined pooled-energy chip")
                chips.append((ep-em)/total)
            roi_rows.append(torch.stack(chips))
        rows.append(torch.stack(roi_rows))
    q = torch.stack(rows)
    signs = torch.tensor(carrier.state_matrix(key, 22), dtype=q.dtype, device=q.device)
    state = q[:, :, :8].reshape(22, 32)
    correlations = torch.stack([(state * signs.roll(-offset, 0)).mean() for offset in range(22)])
    bits = torch.tensor(carrier.message_fragments(message), dtype=q.dtype, device=q.device)*2-1
    margins = torch.stack([q[f::4, :, 8:].mean((0, 1))*bits[f] for f in range(4)])
    max_other = correlations[1:].max()
    minimums = margins.min(dim=1).values
    active = dict(other_offsets=(torch.nonzero(correlations[1:] == max_other).flatten()+1).tolist(),
                  fragment_bits=[torch.nonzero(margins[f] == minimums[f]).flatten().tolist() for f in range(4)])
    tied = len(active["other_offsets"]) != 1 or any(len(x) != 1 for x in active["fragment_bits"])
    objectives = torch.cat(((correlations[0]-max_other).reshape(1), minimums))
    return dict(q=q, correlations=correlations, margins=margins, objectives=objectives,
                active=active, tied=tied)


def simplex_minimum(gram):
    """Solve the five-variable PSD simplex QP by its 31 faces, with no ridge.

    Face enumeration solves one algebraic problem, not model candidate selection.
    Scale-relative FP64 residual checks are numerical diagnostics, not score gates.
    """
    import numpy as np

    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (5, 5) or not np.isfinite(gram).all():
        raise ValueError("finite 5x5 Gram required")
    scale = float(np.max(np.abs(gram)))
    if scale == 0:
        return dict(status="EXACT_ZERO", weights=[.2]*5, squared_norm=0., kkt_residual=0.,
                    numerical_tolerance=0.)
    a = (gram + gram.T) / (2*scale)
    tol = 4096*np.finfo(np.float64).eps
    best = None
    for size in range(1, 6):
        for face in combinations(range(5), size):
            ids = list(face)
            block = np.zeros((size+1, size+1))
            block[:size, :size] = a[np.ix_(ids, ids)]
            block[:size, size] = block[size, :size] = 1
            rhs = np.zeros(size+1); rhs[-1] = 1
            sol = np.linalg.lstsq(block, rhs, rcond=None)[0]
            if np.max(np.abs(block@sol-rhs)) > tol or min(sol[:size]) < -tol:
                continue
            weights = np.zeros(5); weights[ids] = np.maximum(sol[:size], 0)
            weights /= weights.sum()
            norm2 = float(weights@a@weights)
            residual = max(0., float(norm2 - min(a@weights)))
            if residual <= tol and (best is None or norm2 < best[0]):
                best = norm2, weights, residual
    if best is None:
        return dict(status="NUMERICALLY_UNRESOLVED", weights=None, squared_norm=None,
                    kkt_residual=None, numerical_tolerance=tol*scale)
    norm2, weights, residual = best
    return dict(status="NUMERICALLY_ZERO" if norm2 <= tol else "NONZERO",
                weights=weights.tolist(), squared_norm=max(0., norm2*scale),
                kkt_residual=residual*scale, numerical_tolerance=tol*scale)


def directions(gradients):
    """Masking precedes entry. Never normalize individual objective gradients."""
    import torch

    if len(gradients) != 5 or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("five finite masked gradients required")
    # Keep FP64 accumulation on CPU, independently of decoder FP32 arithmetic.
    g = torch.stack([x.detach().cpu().double() for x in gradients])
    flat = g.reshape(5, -1)
    gram = flat @ flat.T
    mean = g.mean(0)
    mean_norm = float(mean.norm())
    solution = simplex_minimum(gram.numpy())
    answer = dict(MEAN=None, COMMON=None)
    if mean_norm > 0:
        answer["MEAN"] = (mean/mean_norm).float()
    common_norm = None
    if solution["weights"] is not None:
        weights = torch.tensor(solution["weights"], dtype=g.dtype)
        common = (weights.reshape(5, *([1]*(g.ndim-1)))*g).sum(0)
        common_norm = float(common.norm())
        if common_norm == 0:
            solution["status"] = "EXACT_ZERO"
        if solution["status"] == "NONZERO" and common_norm > 0:
            answer["COMMON"] = (common/common_norm).float()
    report = dict(gram=gram.tolist(), objective_norms=flat.norm(dim=1).tolist(),
                  mean_norm=mean_norm, common_norm=common_norm, simplex=solution,
                  mean_status="DEFINED" if answer["MEAN"] is not None else "ZERO_MEAN",
                  zero_common_ceiling=("Exact zero only excludes all five strictly positive first-order changes; weak or finite-step improvement remains possible."
                    if solution["status"] == "EXACT_ZERO" else
                    "Numerically zero/unresolved construction does not exclude a strict common direction; no exclusion certificate."
                    if solution["status"] in ("NUMERICALLY_ZERO", "NUMERICALLY_UNRESOLVED") else
                    "Nonzero common construction; finite-step improvement is measured separately."),
                  predicted={name: None if d is None else (flat@d.double().flatten()).tolist()
                             for name, d in answer.items()})
    return answer, report
