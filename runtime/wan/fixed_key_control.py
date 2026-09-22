"""Fixed-marker clean-leaf objective with unchanged native-response controls."""
import torch
from main.tube_state import fixed_key
from . import trajectory
from .payload_control import R_STAR, prepare_direction, controlled_step, cumulative
from . import window_state_mse

DEFAULT_OBJECTIVE = "fixed-marker-11-window-tanh-mean"
WINDOW_STATE_MSE_OBJECTIVE = window_state_mse.OBJECTIVE_ID
OBJECTIVES = (DEFAULT_OBJECTIVE, WINDOW_STATE_MSE_OBJECTIVE)


def objective_identity(objective=DEFAULT_OBJECTIVE):
    if objective == DEFAULT_OBJECTIVE:
        return {
            "id": DEFAULT_OBJECTIVE,
            "formula": "negative mean fixed-code tanh projection",
            "temperature": 1,
            "support_count": 1760,
            "receiver_clipping_used": False,
        }
    if objective == WINDOW_STATE_MSE_OBJECTIVE:
        return window_state_mse.identity()
    raise ValueError(f"unknown fixed-key writer objective: {objective}")


def _loss(clean_leaf, book, objective):
    if objective == DEFAULT_OBJECTIVE:
        return fixed_key.loss(clean_leaf, book), {}
    if objective == WINDOW_STATE_MSE_OBJECTIVE:
        value, states = window_state_mse.loss(clean_leaf, book)
        return value, {"state_means": states.detach().cpu().tolist()}
    objective_identity(objective)
    raise AssertionError("unreachable")


def clean_direction(clean, book, count=None, *, objective=DEFAULT_OBJECTIVE):
    """Return a detached CPU clean-leaf direction for one explicit writer objective."""
    with torch.enable_grad():
        leaf = clean.detach().cpu().double().clone().requires_grad_(True)
        value, diagnostics = _loss(leaf, book, objective)
        if count: count('clean_leaf_backward', False)
        gradient, = torch.autograd.grad(value, leaf)
        if count: count('clean_leaf_backward', True)
    norm = trajectory.measures(gradient)['support_rms']
    if not torch.isfinite(gradient).all() or not 0 < norm < float('inf'):
        raise ValueError('zero/nonfinite fixed-marker gradient')
    return -gradient.detach().float(), dict(objective=objective, objective_identity=objective_identity(objective), temperature=1,
        loss=float(value.detach()), gradient_support_rms=norm, **diagnostics,
        nominal_clean=fixed_key.nominal_record(clean, book),
        meaning='detached CPU current clean leaf; no terminal/model/VAE gradient transport')
