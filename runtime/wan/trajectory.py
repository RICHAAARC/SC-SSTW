"""Native scheduler stepping and byte fingerprints used by the GROW reference."""
from __future__ import annotations
import hashlib
import json
import torch

def _fingerprint_value(value):
    if torch.is_tensor(value):
        # NumPy cannot expose bfloat16 values. Reinterpret contiguous tensor
        # storage as bytes; reshape handles scalar tensors as well. For NumPy
        # supported dtypes these are the same bytes as .numpy().tobytes().
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _fingerprint_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def fingerprint(value) -> str:
    encoded = json.dumps(_fingerprint_value(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def native_step(scheduler, z, v, index, count, kind="scheduler_step"):
    cursor = scheduler.step_index
    if not (cursor == index or (index == 0 and cursor is None)):
        raise ValueError("native scheduler cursor mismatch")
    count(kind, False)
    result = scheduler.step(v, scheduler.timesteps[index], z.clone(), return_dict=False)[0]
    count(kind, True)
    if scheduler.step_index != index + 1 or not torch.isfinite(result).all():
        raise RuntimeError("nonfinite native state or scheduler cursor divergence")
    return result.detach()


def validate_scheduler(scheduler) -> None:
    config = scheduler.config
    if config.prediction_type != "flow_prediction" or config.thresholding or not scheduler.predict_x0:
        raise ValueError("native nonthresholded flow-prediction scheduler required")
    if len(scheduler.timesteps) != 50 or not config.lower_order_final or float(scheduler.sigmas[-1]) != 0.0:
        raise ValueError("fixed native 50-step zero-sigma schedule required")
