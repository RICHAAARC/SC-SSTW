"""Received-coordinate M3 view plan/extraction. No model or codec execution.

The common runtime owns actual E41/reset/posterior-mode, Q, DCT2 and journaling.
These helpers only schedule the12 physical views or extract their uint8 inputs.
"""
from main.tube_state.b_m3_time_path_v1 import WINDOWS, RECEIVED_FRAMES


def view_plan(input_id):
    if not isinstance(input_id, str) or not input_id:
        raise ValueError("nonempty opaque input_id required")
    rows = []
    for window, w in WINDOWS.items():
        query_id = f"{input_id}/{window}"
        for p in range(4):
            rows.append(dict(input_id=input_id, query_id=query_id,
                view_id=f"{query_id}/p{p}", window=window, phase=p, received_query_start=w,
                encoder_input_support=dict(received_interval_inclusive=[w+p, w+p+40],
                    received_frames=list(range(w+p, w+p+41)),
                    spatial_shape=[320, 512, 3], complete_rgb_input=True),
                encoder_context=dict(reset_required=True, posterior="mode",
                    preprocessing="uint8_to_float_div255_once_in_encoder_adapter",
                    expected_normalized_latent_shape=[1, 16, 11, 40, 64]),
                status="PLANNED", cause_set=[]))
    return rows


def extract_e41(received, received_query_start, phase):
    """Array/tensor protocol; returns a separate uint8 slice, no normalization."""
    if tuple(getattr(received, "shape", ())) != (RECEIVED_FRAMES, 320, 512, 3):
        raise ValueError("actual received uint8[151,320,512,3] array required")
    if str(getattr(received, "dtype", "")) not in ("uint8", "torch.uint8"):
        raise ValueError("actual received uint8 data required")
    if type(received_query_start) is not int or received_query_start not in WINDOWS.values():
        raise ValueError("received query start must be 0, 53 or 106")
    if type(phase) is not int or phase not in range(4):
        raise ValueError("phase must be 0..3")
    start = received_query_start + phase
    result = received[start:start+41]
    if tuple(getattr(result, "shape", ())) != (41, 320, 512, 3):
        raise ValueError("incomplete41-frame input")
    if callable(getattr(result, "clone", None)):
        return result.clone()
    if callable(getattr(result, "copy", None)):
        return result.copy()
    raise TypeError("received array must support clone() or copy()")
