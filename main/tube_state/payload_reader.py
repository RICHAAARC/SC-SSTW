"""Stable repeated payload reader; no state-path or runtime dependency."""
from main.tube_state import grow_video_reference as payload

def payload_read(normalized,key,R):
    """Repeated payload diagnostic only; never used to rank state paths."""
    import torch
    from collections import Counter
    if normalized.ndim!=5 or tuple(normalized.shape[:2])!=(1,16) or tuple(normalized.shape[-2:])!=(40,64) or normalized.shape[2]<R+1:raise ValueError('received geometry mismatch')
    values=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm='ortho').real
    coords=payload.coordinates(key);h=[a for a,b in coords];w=[b for a,b in coords];bits=[];votes=[]
    for ch in range(4):
        raw=(values[0,ch,1:R+1,h,w]>0).to(torch.int64).cpu().numpy()
        for bit in range(8):
            row=raw[:,bit::8].reshape(-1).tolist();bits.append(Counter(row).most_common(1)[0][0])
            votes.append(dict(ones=sum(row),zeros=len(row)-sum(row),count=len(row)))
    return dict(status='READ',decoded_bits=bits,votes=votes,truth_used=False,R=R,role='repeated payload diagnostic; not a path selector')


