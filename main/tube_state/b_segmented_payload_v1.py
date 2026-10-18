"""B video-time segmented GROW target/read adapter, net32.
Locally reuses A pure target/group/reducer arithmetic; no A clock or selector.
"""
from __future__ import annotations
from collections import Counter
from typing import Any
from main.tube_state import grow_video_reference as original



def source_segment(t):
    if type(t) is not int or not 0<=t<=45:raise ValueError('source Wan time outside 0..45')
    return 0 if t==0 else 3 if t==45 else (t-1)//11


def build_target(latent,key,bits):
    """All 46 layers keep the original mask/alpha; effective layers 1..44."""
    import torch
    original._validate(latent,original.PUBLIC)
    if len(bits)!=32 or any(type(b) is not int or b not in (0,1) for b in bits):
        raise ValueError('exactly 32 binary writer bits required')
    target=torch.zeros_like(latent);mask=torch.zeros_like(latent,dtype=torch.bool)
    coords=original.coordinates(key);h=[p[0] for p in coords];w=[p[1] for p in coords]
    for t in range(46):
        g=source_segment(t)
        values=latent.new_tensor([(2*bits[8*g+j%8]-1)*original.ALPHA for j in range(len(coords))])
        for ch in original.PUBLIC.channels:
            target[0,ch,t,h,w]=values;mask[0,ch,t,h,w]=True
    return target,mask


def segment_plan(estimated_path):
    """Public grouping from an estimated received-frame path, never truth."""
    operation=decode_visible_span(estimated_path)
    support=set(estimated_path)
    segments=[]
    for g in range(4):
        cells=[]
        for r in range(1,operation['wan_support']+1):
            source_t=operation['anchor']//4+r
            if 1<=source_t<=44 and source_segment(source_t)==g:
                sources=list(range(4*source_t-3,4*source_t+1))
                if all(s in support for s in sources):
                    cells.append(dict(receiver_wan_time=r,source_wan_time=source_t,source_frames=sources))
        segments.append(dict(segment=g,cells=cells,status='SUPPORTED' if cells else 'MISSING',
                             observed_cells=len(cells),planned_bits=8))
    return dict(operation=operation,segments=segments,estimated_path=list(estimated_path),
                truth_inputs=False,message_inputs=False,
                support_rule='Only four source positions actually represented by received indices in estimated path; synthetic fill is context, not votes.')


def read_segments(normalized,key,estimated_path):
    """Original FFT/sign/tie semantics, grouped by public estimated source time."""
    import torch
    import numpy as np
    plan=segment_plan(estimated_path);op=plan['operation']
    expected=(1,16,(op['output_frames']-1)//4+1,40,64)
    if tuple(normalized.shape)!=expected or not bool(torch.isfinite(normalized).all()):
        raise ValueError('finite normalized Wan latent with selected output geometry required')
    if op['status']!='SUPPORTED':raise ValueError('selected span has no Wan support')
    spectrum=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm='ortho').real
    coords=original.coordinates(key);h=[p[0] for p in coords];w=[p[1] for p in coords]
    values=torch.stack([spectrum[0,ch,1:op['wan_support']+1,h,w] for ch in range(4)])
    votes=(values>0).to(torch.int8).cpu().numpy()
    zeros=(values==0).to(torch.int8).cpu().numpy()
    return decode_signed_votes(plan,(2*votes-1).astype(np.int8),zeros)


def decode_signed_votes(plan,signed_votes,zeros):
    """One canonical vote reducer, also usable to read saved evidence without a VAE."""
    import numpy as np
    shape=(4,plan['operation']['wan_support'],240)
    if signed_votes.shape!=shape or signed_votes.dtype!=np.int8 or not np.isin(signed_votes,[-1,1]).all():
        raise ValueError('signed votes must be int8 +/-1 with public support geometry')
    if zeros.shape!=shape or zeros.dtype!=np.int8 or not np.isin(zeros,[0,1]).all():
        raise ValueError('zero coefficients must be int8 0/1 with public support geometry')
    if np.any((zeros==1)&(signed_votes!= -1)):raise ValueError('zero FFT coefficients must receive strict >0 vote zero')
    votes=(signed_votes>0).astype(np.int8)
    segments=[];bits=[None]*32
    for item in plan['segments']:
        row={**item,'bits':[]};times=[c['receiver_wan_time']-1 for c in item['cells']]
        for slot in range(8):
            sequence=votes[:,times,slot::8].reshape(-1).tolist()
            if sequence:
                decoded=int(Counter(sequence).most_common(1)[0][0]);ones=sum(sequence);count=len(sequence)
                bit=dict(bit_index=8*item['segment']+slot,decoded=decoded,ones=ones,zeros=count-ones,
                         count=count,tie=ones*2==count,zero_coefficients=int(zeros[:,times,slot::8].sum()))
                bits[bit['bit_index']]=decoded
            else:
                bit=dict(bit_index=8*item['segment']+slot,decoded=None,count=0,status='MISSING')
            row['bits'].append(bit)
        segments.append(row)
    return dict(status='READ',decoded_bits=bits,segments=segments,plan=plan,
                net_payload_bits=32,decoded_bit_count=sum(b is not None for b in bits),truth_inputs=False,
                vote_order='channel, eligible receiver time, coordinate slot::8; Counter encounter ties; strict >0',
                signed_votes=signed_votes,zero_coefficient_mask=zeros)


def decode_visible_span(path) -> dict[str, Any]:
    """Map the estimated visible source span back to received-frame indices."""

    source_path = list(path)
    if not source_path or any(type(item) is not int for item in source_path):
        raise ValueError("estimated path must contain integer source positions")
    if any(item < 0 or item >= 181 for item in source_path):
        raise ValueError("estimated path leaves the public source clock")
    if any(right < left for left, right in zip(source_path, source_path[1:])):
        raise ValueError("estimated path must be nondecreasing")
    source_start, source_end = source_path[0], source_path[-1]
    anchor = (source_start // 4) * 4
    available = source_end - anchor + 1
    output_frames = ((available - 1) // 4) * 4 + 1
    if output_frames < 1:
        raise ValueError("estimated visible span is empty")
    output_sources = list(range(anchor, anchor + output_frames))
    received_map = []
    synthetic = []
    exact_source_hits = set(source_path)
    for output_index, source_index in enumerate(output_sources):
        nearest = min(
            range(len(source_path)),
            key=lambda received_index: (abs(source_path[received_index] - source_index), received_index),
        )
        received_map.append(nearest)
        if source_index not in exact_source_hits:
            synthetic.append(output_index)
    repeated = [
        index for index in range(1, len(received_map))
        if received_map[index] == received_map[index - 1]
    ]
    selected = set(received_map)
    dropped = [index for index in range(len(source_path)) if index not in selected]
    support = min(44, (output_frames - 1) // 4)
    status = "SUPPORTED" if support >= 1 else "UNSUPPORTED"
    return {
        "status": status,
        "reason": None if status == "SUPPORTED" else "INSUFFICIENT_VISIBLE_WAN_SUPPORT",
        "source_start": source_start, "source_end": source_end,
        "visible_span_frames": available,
        "anchor": anchor, "front_boundary_copies": source_start - anchor,
        "tail_discarded_source_positions": available - output_frames,
        "output_frames": output_frames, "source_coordinate_map": output_sources,
        "received_index_map": received_map,
        "synthetic_output_indices": synthetic,
        "repeated_output_indices": repeated,
        "dropped_received_indices": dropped,
        "wan_support": support,
        "wan_input_rule": "output_frames == 1 + 4*k; R=min(44,k); first Wan latent excluded",
        "mapping_rule": "nearest received source estimate; equal distance chooses earlier received index",
        "truth_inputs": False,
    }
