"""Single frozen feedback point; diagnostics never select an update."""
import hashlib
import json
from main.sc_sstw.terminal_feedback import clone_graph_state


def state_digest(value, torch):
    def describe(x):
        if torch.is_tensor(x):
            raw=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(x,dict):return {str(k):describe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [describe(v) for v in x]
        if hasattr(x,'__dict__'):return describe(vars(x))
        return repr(x)
    return hashlib.sha256(json.dumps(describe(value),sort_keys=True).encode()).hexdigest()


def diagnose(adapter, z, state, reference, *, learning_rate, step_rms, amplitude, report, persist, save):
    torch=adapter.torch;guard=adapter.resource_guard
    target=adapter.readout(reference).detach().clone();target[:,1]-=amplitude
    origin=z.detach().clone();before=state_digest(state,torch)
    report.update(state_before=before,next_normal_index=state.next_index,evaluations={})
    def evaluate(value,label,grad_enabled=False):
        report['active']=label;persist()
        with torch.set_grad_enabled(grad_enabled):
            rgb=adapter.rollout(value,clone_graph_state(state,torch));q=adapter.readout(rgb)
            loss=(q-target).square().mean()
            if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_LOSS')
            report['evaluations'][label]=dict(loss=float(loss.detach()),loss_float64_reduction=float((q.detach().double()-target.double()).square().mean()),q=q.detach().cpu().tolist(),state_unchanged=state_digest(state,torch)==before)
            persist()
            return loss
    x=origin.clone().requires_grad_(True)
    loss=evaluate(x,'gradient_baseline',True)
    guard.consume('backward_calls');g=torch.autograd.grad(loss,x)[0];guard.complete('backward_calls')
    if not bool(torch.isfinite(g).all()):raise RuntimeError('NONFINITE_GRADIENT')
    gnorm=float(g.square().mean().sqrt())
    if gnorm==0:raise RuntimeError('ZERO_GRADIENT')
    scale=min(learning_rate,step_rms/gnorm)
    proposed=(origin-scale*g.detach()).detach();delta=proposed-origin
    opposite=(origin-delta).detach();minus_delta=opposite-origin
    ideal=-scale*g.detach()
    def rms(t):return float(t.double().square().mean().sqrt())
    cast=adapter.input_dtype();cast_delta=proposed.to(cast).float()-origin.to(cast).float()
    report['update']=dict(scale=scale,gradient_rms=gnorm,step_rms_cap=step_rms,ideal_update_rms=rms(ideal),actual_update_rms=rms(delta),rounding_error_rms=rms(delta-ideal),changed_fraction=float((delta!=0).float().mean()),g_dot_delta=float((g.double()*delta.double()).sum()),g_dot_opposite_delta=float((g.double()*minus_delta.double()).sum()),cosine_to_negative_gradient=float(((-g.double()*delta.double()).sum())/(g.double().norm()*delta.double().norm())) if bool((delta!=0).any()) else None,input_cast_dtype=str(cast),input_cast_changed_fraction=float((cast_delta!=0).float().mean()),input_cast_delta_rms=rms(cast_delta),opposite_symmetry_error_rms=rms(minus_delta+delta))
    save(dict(latent=origin.cpu(),gradient=g.detach().cpu(),delta=delta.cpu(),opposite_delta=minus_delta.cpu(),target=target.cpu(),scheduler_state=clone_graph_state(state,torch)))
    del loss,x,g,ideal,cast_delta
    persist()
    for label,value in (('baseline_repeat_1',origin),('baseline_repeat_2',origin),('fixed_update',proposed),('opposite_update',opposite)):
        result=evaluate(value,label)
        del result
    e=report['evaluations'];b1=e['baseline_repeat_1']['loss'];b2=e['baseline_repeat_2']['loss'];plus=e['fixed_update']['loss'];minus=e['opposite_update']['loss'];b=(b1+b2)/2
    report['comparison']=dict(repeat_baseline_range=abs(b1-b2),gradient_vs_no_grad_baseline=e['gradient_baseline']['loss']-b,actual_loss_change=plus-b,opposite_loss_change=minus-b,central_directional_difference=(plus-minus)/2,first_order_prediction=report['update']['g_dot_delta'],symmetric_second_difference=plus+minus-2*b,state_unchanged=state_digest(state,torch)==before,interpretation='Diagnostic only: no candidate selection; finite-step curvature and mixed precision remain possible.')
    report['active']=None;report['status']='DIAGNOSTIC_EXECUTED_REQUIRES_REVIEW';persist()
