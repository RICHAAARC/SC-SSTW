"""Fixed saved-fork zero/control comparison; descriptive observations, no new gate."""
from __future__ import annotations
from main.tube_state.local_joint_paired_transfer_v1 import measure,displacement
from main.tube_state.local_joint_same_point_dual_v1 import summarize

VIEWS=("ZERO_TAIL","CONTROL_TAIL")
PAIR_SPECS=(("ZERO_TAIL_vs_SAVED_ZERO","SAVED_ZERO","ZERO_TAIL"),
            ("CONTROL_TAIL_vs_ZERO_TAIL","ZERO_TAIL","CONTROL_TAIL"))
PAIRS=tuple(x[0] for x in PAIR_SPECS)
CEILING=("User-adopted fixed paired short tail (2026-10-16). One saved source/fork and exact saved terminal direction, "
         "two independent full-history restores, zero or fixed .02 conditional-clean at CFG5 (.10 nominal CFG-clean). "
         "Four ordinary DiT calls, four native updates, two ordinary VAE decodes; no new gradient/baseline/source, "
         "scan, amplitude change, selection, codec, blind recovery or scientific PASS. Numerical runtime drift and "
         "finite-precision/nonlinear transfer are not uniquely attributed to a Jacobian by this comparison.")

def validate_direction(direction,shape,support):
    import torch
    if not torch.is_tensor(direction) or tuple(direction.shape)!=tuple(shape) or not bool(direction.isfinite().all()):
        raise ValueError("saved direction must be a finite matching latent tensor")
    support.validate_for(tuple(shape))
    mask=torch.zeros_like(direction,dtype=torch.bool)
    for y0,y1,x0,x1 in support.rois:mask[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1]=True
    if bool((direction[~mask]!=0).any()) or abs(float(direction.double().norm())-1.)>1e-6:
        raise ValueError("saved direction must have original support and unit L2 norm; no substitute direction")

def comparison(values,reference=None):
    summaries={name:None if values.get(name) is None else summarize(values[name]) for name in VIEWS}
    summaries["SAVED_ZERO"]=None if reference is None else summarize(reference)
    result=dict(views={},saved_reference=summaries["SAVED_ZERO"],changes={},scientific_pass=False,
        ceiling=CEILING,bounded_outcome="DESCRIPTIVE_FIXED_PAIRED_TAIL_NO_SCIENTIFIC_PASS")
    for name in VIEWS:
        s=summaries[name]
        result["views"][name]=dict(summary=s,
            payload_rows=[dict(index=i,status="MISSING" if s is None else "SCORED",margin=None if s is None else s["margins"][i]) for i in range(32)],
            state_rows=[dict(offset=i,status="MISSING" if s is None else "SCORED",correlation=None if s is None else s["correlations"][i]) for i in range(22)])
    for pair,left,right in PAIR_SPECS:
        a,b=summaries[left],summaries[right]
        result["changes"][pair]=None if a is None or b is None else dict(
            loss_gain=a["allbit_loss"]["loss"]-b["allbit_loss"]["loss"],trueG_change=b["state_gap"]-a["state_gap"],
            minimum_margin_change=min(b["margins"])-min(a["margins"]),
            margins=[y-x for x,y in zip(a["margins"],b["margins"])],
            correlations=[y-x for x,y in zip(a["correlations"],b["correlations"])],
            gained_positive=[i for i in range(32) if a["margins"][i]<=0<b["margins"][i]],
            lost_positive=[i for i in range(32) if b["margins"][i]<=0<a["margins"][i]])
    return result
