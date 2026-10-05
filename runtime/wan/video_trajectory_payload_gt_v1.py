"""Length-based payload support and residual metrics; no experiment imports."""
from __future__ import annotations
import math
from main.tube_state import video_local_fourier_rm_control as payload_method


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


def residual_temporal_metrics(reference_rgb8, candidate_rgb8, *, source_start):
    """Condition-minus-P1 residual structure, separate from video self-difference ratio."""
    import torch
    if reference_rgb8.shape!=candidate_rgb8.shape or reference_rgb8.ndim!=4:
        raise ValueError("equal [T,H,W,3] clips required")
    frames=int(reference_rgb8.shape[0]); energy=roughness=boundary_energy=internal_energy=0.0
    previous=None; lag=prev_energy=next_energy=0.0; boundary_edges=0
    for index in range(frames):
        value=(candidate_rgb8[index].to(torch.float64)-reference_rgb8[index].to(torch.float64))/255.0
        energy+=float(value.square().sum())
        if previous is not None:
            edge=float((value-previous).square().sum());roughness+=edge
            if (source_start+index)%4==0:
                boundary_edges+=1;boundary_energy+=edge
            else:internal_energy+=edge
            lag+=float((previous*value).sum())
            prev_energy+=float(previous.square().sum());next_energy+=float(value.square().sum())
        previous=value
    return dict(definition="delta=condition-P1 in RGB[0,1]; unscaled noncyclic adjacent difference",
        source_start=int(source_start),frames=frames,edges=frames-1,boundary_edges=boundary_edges,
        internal_edges=frames-1-boundary_edges,residual_l2_squared=energy,
        residual_dt_squared=roughness,residual_dt_boundary_squared=boundary_energy,
        residual_dt_internal_squared=internal_energy,
        residual_normalized_roughness=roughness/energy if energy else None,
        residual_lag1_uncentered_cosine=lag/math.sqrt(prev_energy*next_energy) if prev_energy and next_energy else None,
        no_perceptual_flicker_interpretation=True)
