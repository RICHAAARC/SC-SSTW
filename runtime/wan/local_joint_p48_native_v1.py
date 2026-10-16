"""Actual full-history Native48 coefficient; no extra scheduler/model execution."""
from __future__ import annotations
import math
from main.tube_state.local_joint_p0_input_vjp_v1 import finite_tensor
from runtime.wan.trajectory import validate_scheduler


def coefficients(scheduler, shape):
    import torch
    from diffusers import UniPCMultistepScheduler
    validate_scheduler(scheduler)
    c = scheduler.config
    if not c.use_flow_sigmas:
        raise ValueError("Native48 coefficient requires actual flow-sigma alpha=1-sigma coordinates")
    if not isinstance(scheduler, UniPCMultistepScheduler) or scheduler.solver_p is not None:
        raise ValueError("native UniPC without external solver required")
    if c.solver_order != 2 or c.solver_type != "bh2" or scheduler.step_index != 48 or scheduler.this_order != 2 or scheduler.lower_order_nums < 2:
        raise ValueError("saved full-order Native48 history required")
    if 47 in scheduler.disable_corrector or scheduler.num_inference_steps != 50:
        raise ValueError("corrector48 and 50-step history required")
    finite_tensor(scheduler.last_sample, shape, "native last_sample", torch.float32)
    if len(scheduler.model_outputs) != 2 or len(scheduler.timestep_list) != 2:
        raise ValueError("two native history entries required")
    for i, v in enumerate(scheduler.model_outputs):
        finite_tensor(v, shape, "native history", torch.float32)
        if float(scheduler.timestep_list[i]) != float(scheduler.timesteps[46+i]):
            raise ValueError("native history timestep does not identify step46/47")
    sigmas = scheduler.sigmas.detach().cpu().double()
    finite_tensor(sigmas, (51,), "native sigmas")
    s46,s47,s48,s49,s50 = (float(s) for s in sigmas[46:51])
    if not (0 < s49 < s48 < s47 < s46 < 1 and s50 == 0):
        raise ValueError("invalid terminal native flow sigma coordinates")
    lam = lambda s: math.log(1-s)-math.log(s)
    hc = lam(s48)-lam(s47)
    bh = math.expm1(-hc)
    rc = (lam(s46)-lam(s47))/hc
    phi1 = bh/(-hc)-1
    phi2 = phi1/(-hc)-.5
    rhs = torch.tensor([phi1/bh,2*phi2/bh],dtype=torch.float64)
    rho = torch.linalg.solve(torch.tensor([[1.,1.],[rc,1.]],dtype=torch.float64),rhs)
    corrector = -(1-s48)*bh*float(rho[-1])
    hp = lam(s49)-lam(s48)
    bp = math.expm1(-hp)
    rp = (lam(s47)-lam(s48))/hp
    predictor = -(1-s49)*bp*(1-1/(2*rp))
    k48 = 5*((s49/s48)*corrector+predictor)
    if not math.isfinite(k48): raise FloatingPointError("nonfinite native control coefficient")
    return dict(sigmas46_to50=[s46,s47,s48,s49,s50],corrector48_coefficient=corrector,
        predictor48_coefficient=predictor,k48=k48,sigma48=s48,sigma49=s49,
        corrector49_enabled=48 not in scheduler.disable_corrector,
        final49="pre_corrector_x49 - sigma49 * CFG_velocity49",
        pullback="k48 * (g_z + Jc^T(-5*sigma49*g_z) + Ju^T(4*sigma49*g_z)); then mask S",
        coefficient_arithmetic="CPU FP64 analytic scalar from actual stored sigmas; native FP32 rounding reported by CPU fixture")
