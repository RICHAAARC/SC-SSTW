"""Fixed P0 specification and numeric reporting; no writer or receiver scoring."""
from __future__ import annotations

BRANCHES = ("conditional", "unconditional")
LATENT_SHAPE = (1, 16, 46, 40, 64)
ROIS = ((8, 16, 12, 20), (8, 16, 44, 52), (28, 36, 12, 20), (28, 36, 44, 52))
EXPECTED = dict(transformer_prepare=1, dit_forward=2, dit_input_vjp=2,
                block_forward=60, block_recompute_upper=60, native=0, vae_load=0,
                decode=0, decoder_vjp=0, writer=0, new_video=0, encode=0, codec=0,
                new_source=0, new_baseline=0, retry=0, branches=2)
CEILING = ("P0 fixed saved-input resource/wiring observation only. BF16 autograd returns an approximate "
           "path cotangent, not the derivative of the actual discrete finite map or writer success. "
           "Old prediction comparison spans preparations: observed differences cannot be attributed "
           "to gradmode alone. No writer, native/VAE/video call or scientific PASS.")


def finite_tensor(value, shape=None, name="tensor", dtype=None, *, allow_nonfinite=False):
    import torch
    if not torch.is_tensor(value) or (shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError(name + " has invalid tensor shape")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(name + " has invalid arithmetic dtype: " + str(value.dtype))
    if not allow_nonfinite and not bool(value.isfinite().all()):
        raise FloatingPointError(name + " contains nonfinite values")
    return value


def direction_contract(direction, shape=LATENT_SHAPE, rois=ROIS, time_range=(1, 45)):
    import torch
    finite_tensor(direction, shape, "saved exact direction")
    if not direction.is_floating_point():
        raise ValueError("saved exact direction must be a real floating-point tensor")
    # The original paired-transfer producer normalizes in FP64. Validate the
    # saved values without converting, renormalizing or replacing the cotangent.
    mask = torch.zeros_like(direction, dtype=torch.bool)
    for y0, y1, x0, x1 in rois:
        mask[:, :, time_range[0]:time_range[1], y0:y1, x0:x1] = True
    if bool((direction[~mask] != 0).any()) or abs(float(direction.double().norm()) - 1) > 1e-6:
        raise ValueError("saved d must retain original support and unit norm; no substitute direction")


def statistics(value):
    import torch
    x = value.detach().cpu()
    finite = torch.isfinite(x)
    row = dict(shape=list(x.shape), dtype=str(x.dtype), numel=x.numel(),
               finite_count=int(finite.sum()), nonfinite_count=int((~finite).sum()),
               finite=bool(finite.all()), zero=bool((x == 0).all()))
    if row["finite"]:
        y = x.double()
        row.update(l2=float(y.norm()), max_abs=float(y.abs().max()), mean=float(y.mean()))
    return row


def difference(actual, reference):
    import torch
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    if a.shape != b.shape:
        raise ValueError("prediction comparison shape differs")
    delta = a - b
    row = statistics(delta)
    row.update(exact_equal=bool(torch.equal(a, b)), comparison="cross_preparation_observed_difference",
               causal_attribution=None, threshold=None)
    if row["finite"]:
        row["relative_l2"] = float(delta.norm() / b.norm()) if float(b.norm()) else None
    return row
