"""One-point terminal/native transfer; user-adopted objective/amplitude, no scans."""
from __future__ import annotations
import math
from main.tube_state import local_joint_all_bit_v1 as loss
from main.tube_state.local_joint_same_point_dual_v1 import summarize
from main.tube_state.local_joint_readout_m0_v1 import tensor_readout, UndefinedActiveTie

VIEWS=("ZERO","TERMINAL","NATIVE")
OBJECTIVE="historical_allbit_v1"
AMPLITUDE=.10
CONTROL_INDEX=48
CEILING=("One existing source, fresh zero-control native baseline, one terminal VAE VJP, "
         "one direct perturbation and one same-history native short tail. Historical ALLBIT and "
         "a=.10 terminal/CFG-clean (.02 conditional-clean at CFG5) were adopted by the user on 2026-10-15; real execution remains user-run. No scan, "
         "posterior lift, extra terminal optimization, blind recovery, codec, independent source, "
         "generalization or multistep joint-method PASS.")

def scalar(read):
    return loss.scalar_loss(read)

def direction(gradient,support):
    import torch
    g=gradient.detach().cpu().double()
    if not bool(g.isfinite().all()): raise FloatingPointError("nonfinite terminal gradient")
    support.validate_for(tuple(g.shape))
    masked=torch.zeros_like(g)
    for y0,y1,x0,x1 in support.rois:
        masked[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1]=g[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1]
    norm=float(masked.norm())
    return (None if norm==0 else -masked/norm),dict(status="ZERO_GRADIENT" if norm==0 else "DEFINED",
        raw_gradient_norm=float(g.norm()),masked_gradient_norm=norm,sign=-1,normalization="single masked combined loss gradient")

def operands(zero,unit_direction,*,amplitude=AMPLITUDE,guidance=5.):
    import torch
    if amplitude!=AMPLITUDE or guidance!=5.:
        raise ValueError("this fixed adopted comparison uses a=.10 and CFG5")
    if zero.dtype!=torch.float32 or not bool(zero.isfinite().all()) or zero.shape!=unit_direction.shape:
        raise ValueError("matching finite FP32 baseline and direction required")
    direct=(zero.cpu().double()+amplitude*unit_direction).float()
    conditional=(amplitude/guidance*unit_direction).float()
    return direct,conditional

def measure(value):
    import torch
    x=value.detach().cpu().double()
    if not bool(x.isfinite().all()):raise FloatingPointError("nonfinite realized displacement")
    return dict(l2=float(x.norm()),rms=float(x.square().mean().sqrt()),max_abs=float(x.abs().max()),
                nonzero=int(torch.count_nonzero(x)),shape=list(x.shape))

def displacement(value,direction,support):
    import torch
    x=value.detach().cpu().double()
    masked=torch.zeros_like(x)
    for y0,y1,x0,x1 in support.rois:masked[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1]=x[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1]
    projection=float((x*direction).sum());orthogonal=x-projection*direction
    return dict(**measure(x),projection_onto_terminal_direction=projection,
                orthogonal_l2=float(orthogonal.norm()),outside_original_support_l2=float((x-masked).norm()),
                nominal_terminal_amplitude_ratio=float(x.norm())/AMPLITUDE)

def comparison(values):
    rows={n:None if values.get(n) is None else summarize(values[n]) for n in VIEWS}
    result=dict(views={},changes={},scientific_pass=False,ceiling=CEILING)
    for n in VIEWS:
        s=rows[n]
        result["views"][n]=dict(summary=s,payload_rows=[dict(index=i,status="MISSING" if s is None else "SCORED",
            margin=None if s is None else s["margins"][i]) for i in range(32)],
            state_rows=[dict(offset=i,status="MISSING" if s is None else "SCORED",
            correlation=None if s is None else s["correlations"][i]) for i in range(22)])
    for n in VIEWS[1:]:
        a,b=rows["ZERO"],rows[n]
        result["changes"][n]=None if a is None or b is None else dict(
            score_gain=a["allbit_loss"]["loss"]-b["allbit_loss"]["loss"],
            trueG_change=b["state_gap"]-a["state_gap"],
            margins=[y-x for x,y in zip(a["margins"],b["margins"])],
            correlations=[y-x for x,y in zip(a["correlations"],b["correlations"])],
            gained_positive=[i for i in range(32) if a["margins"][i]<=0<b["margins"][i]],
            lost_positive=[i for i in range(32) if b["margins"][i]<=0<a["margins"][i]])
    t,n=result["changes"]["TERMINAL"],result["changes"]["NATIVE"]
    result["bounded_outcome"]=("UNRESOLVED_MISSING_OR_UNDEFINED" if t is None or n is None else
        "TERMINAL_FINITE_STEP_NOT_IMPROVED_CANNOT_BLAME_TAIL" if t["score_gain"]<=0 else
        "BOTH_IMPROVE_LIMITED_TRANSFER_SUPPORTED" if n["score_gain"]>0 else
        "TERMINAL_IMPROVES_NATIVE_RESPONSE_LOST_THIS_CONSTRUCTION")
    result["decision_rule"]="descriptive signs of one fixed loss change; no full32 or quality PASS gate"
    return result
