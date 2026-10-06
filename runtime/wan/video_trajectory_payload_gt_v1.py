"""Stable public-length payload adapter; historical filename, no GT/T method."""
from __future__ import annotations
from main.tube_state import payload_reader as payload_method


def read_payload(normalized, key, received_frames):
    """Public length fixes support; original FFT coordinates and Counter ties stay intact."""
    expected_times = (int(received_frames)-1)//4+1
    if received_frames not in (181,177,89) or int(normalized.shape[2]) != expected_times:
        raise ValueError("received frame/latent length mismatch")
    support = min(44,expected_times-1)
    row = payload_method.payload_read(normalized,key,support)
    row["received_frames"] = int(received_frames)
    row["support_rule"] = "R_eff=min(44,T_latent-1); first latent excluded"
    row["bit_rows"] = [
        dict(bit_index=index,decoded=int(decoded),ones=int(vote["ones"]),zeros=int(vote["zeros"]),
             count=int(vote["count"]),margin=abs(int(vote["ones"])-int(vote["zeros"])),
             normalized_margin=abs(int(vote["ones"])-int(vote["zeros"]))/int(vote["count"]))
        for index,(decoded,vote) in enumerate(zip(row["decoded_bits"],row["votes"]))
    ]
    return row

