"""Runtime bridges for the blind temporal-edit receiver."""
from __future__ import annotations

import hashlib


def encode_and_score_framewise(model, received_rgb8, key):
    from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as method

    latent = model.encode(received_rgb8)
    evidence = method.score_framewise(latent, key, method.PUBLIC)
    estimate = method.solve_monotone(evidence["signed_projection"], evidence["rho"], method.PUBLIC)
    operation = method.decode_visible_span(estimate["path"], method.PUBLIC) if estimate["status"] == "ESTIMATED" else None
    return evidence, {
        "status": operation["status"] if operation is not None else "UNRESOLVED",
        "estimate": estimate, "operation": operation, "truth_inputs": False,
    }


def read_payload_general(normalized, key, output_frames, source_coordinate_map):
    """Keep the original FFT coordinates/Counter while exposing effective time rows."""

    import torch
    from main.tube_state import grow_video_reference as layout
    from main.tube_state import payload_reader

    if type(output_frames) is not int or output_frames < 1 or (output_frames - 1) % 4:
        raise ValueError("Wan payload input length must be 1+4*k")
    times = (output_frames - 1) // 4 + 1
    if normalized.ndim != 5 or tuple(normalized.shape[:2]) != (1, 16) or int(normalized.shape[2]) != times:
        raise ValueError("Wan latent temporal shape mismatches corrected RGB length")
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("Wan normalized latent contains nonfinite values")
    if not isinstance(source_coordinate_map, list) or len(source_coordinate_map) != output_frames:
        raise ValueError("source coordinate map must cover the Wan RGB input")
    support = min(44, times - 1)
    if support < 1:
        return {
            "status": "UNSUPPORTED", "reason": "INSUFFICIENT_VISIBLE_WAN_SUPPORT",
            "decoded_bits": None, "missing_bits": 32, "R": support,
            "truth_inputs": False,
        }
    original = payload_reader.payload_read(normalized, key, support)
    spectrum = torch.fft.fft2(normalized.float(), dim=(-2, -1), norm="ortho").real
    coordinates = layout.coordinates(key)
    h = [item[0] for item in coordinates]
    w = [item[1] for item in coordinates]
    selected = torch.stack([spectrum[0, channel, 1:support + 1, h, w] for channel in range(4)])
    signed_votes = torch.sign(selected).to(torch.int8).cpu().numpy()
    zero_mask = (selected == 0).cpu().numpy()
    bit_rows = []
    time_rows = []
    for channel in range(4):
        for slot in range(8):
            values = selected[channel, :, slot::8]
            ones = int((values > 0).sum().item())
            count = int(values.numel())
            bit_rows.append({
                "bit_index": channel * 8 + slot, "payload_channel": channel,
                "ones": ones, "zeros": count - ones, "count": count,
                "decoded": int(original["decoded_bits"][channel * 8 + slot]),
            })
            for latent_index in range(1, support + 1):
                row = values[latent_index - 1]
                row_ones = int((row > 0).sum().item())
                output_coordinate = 4 * latent_index
                time_rows.append({
                    "receiver_latent_index": latent_index,
                    "output_stride_coordinate": output_coordinate,
                    "estimated_source_coordinate": source_coordinate_map[output_coordinate],
                    "bit_index": channel * 8 + slot, "payload_channel": channel,
                    "ones": row_ones, "zeros": int(row.numel()) - row_ones,
                    "count": int(row.numel()),
                    "time_semantics": "receiver-local Wan stride coordinate mapped through blind decoded span; not an independent source receptive field",
                })
    selected_vote_counts = [
        {"ones": row["ones"], "zeros": row["zeros"], "count": row["count"]}
        for row in bit_rows
    ]
    original_vote_counts = [
        {"ones": int(row["ones"]), "zeros": int(row["zeros"]), "count": int(row["ones"] + row["zeros"])}
        for row in original["votes"]
    ]
    return {
        "status": "READ", "R": support, "output_frames": output_frames,
        "decoded_bits": original["decoded_bits"], "votes": original["votes"],
        "signed_votes": signed_votes, "zero_mask": zero_mask,
        "bit_rows": bit_rows, "time_bit_rows": time_rows,
        "original_reader_match": original_vote_counts == selected_vote_counts,
        "original_reader_match_basis": "independent selected sign counts versus payload_reader votes",
        "key_id": hashlib.sha256(key.encode()).hexdigest(),
        "support_rule": "R=min(44,T_latent-1); first Wan latent excluded",
        "truth_inputs": False,
    }


def operate_map(received_rgb8, received_index_map):
    import torch

    if received_rgb8.ndim != 4 or int(received_rgb8.shape[-1]) != 3 or received_rgb8.dtype != torch.uint8:
        raise ValueError("received media must be uint8 [N,H,W,3]")
    if (
        not isinstance(received_index_map, list) or not received_index_map
        or (len(received_index_map) - 1) % 4
        or any(type(index) is not int or index not in range(len(received_rgb8)) for index in received_index_map)
    ):
        raise ValueError("blind correction map must be nonempty 1+4*k received indices")
    indices = torch.tensor(received_index_map, device=received_rgb8.device)
    return received_rgb8.index_select(0, indices).contiguous().clone()
