"""One fixed terminal feedback proposal. No Wan adapter, optimizer or line search."""
import copy,math

def clone_graph_state(value,torch,memo=None):
    """Clone complete mutable solver state, preserving nonleaf tensor autograd links.
    Does not clone model modules. No reset/drop of history or solver counters.
    """
    if memo is None:memo={}
    if id(value) in memo:return memo[id(value)]
    if torch.is_tensor(value):
        result=value.clone();memo[id(value)]=result;return result
    if isinstance(value,torch.nn.Module):raise TypeError('solver state must not contain model modules')
    if isinstance(value,dict):
        if type(value) is dict:
            result={};memo[id(value)]=result
            result.update((k,clone_graph_state(v,torch,memo)) for k,v in value.items());return result
        result=type(value)((k,clone_graph_state(v,torch,memo)) for k,v in value.items());memo[id(value)]=result;return result
    if isinstance(value,list):
        result=[];memo[id(value)]=result;result.extend(clone_graph_state(v,torch,memo) for v in value);return result
    if isinstance(value,tuple):
        values=[clone_graph_state(v,torch,memo) for v in value];result=type(value)(*values) if hasattr(value,'_fields') else tuple(values);memo[id(value)]=result;return result
    if hasattr(value,'__dict__') and not isinstance(value,type):
        result=copy.copy(value);memo[id(value)]=result
        for k,v in vars(value).items():setattr(result,k,clone_graph_state(v,torch,memo))
        return result
    return copy.deepcopy(value,memo)

def terminal_feedback(latent,solver_state,rollout,readout,target,reference_rgb,*,learning_rate,step_rms,quality_rms_budget,torch,strict=False,before_backward=None,after_backward=None,stage_callback=None):
    """rollout(z, deepcopied_full_state) must normally finish solver and decode RGB.

    State is never reset/warped. Targets/quality references are external controls,
    never inputs to the public reader. Real differentiable Wan remains unverified.
    """
    if not all(math.isfinite(x) and x>0 for x in (learning_rate,step_rms,quality_rms_budget)):raise ValueError('positive fixed budgets')
    original=latent.detach().clone();record=dict(accepted=False,attempts=1,step_rms=step_rms,quality_rms_budget=quality_rms_budget)
    try:
        if stage_callback:stage_callback('FORECAST','START')
        z=original.clone().requires_grad_(True)
        rgb=rollout(z,clone_graph_state(solver_state,torch));q=readout(rgb)
        if q.shape!=target.shape or rgb.shape!=reference_rgb.shape:raise ValueError('terminal target/reference shape')
        if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(rgb).all()):raise ValueError('nonfinite terminal')
        if stage_callback:stage_callback('FORECAST','COMPLETE')
        loss=(q-target).square().mean()
        if before_backward is not None:before_backward()
        grad=torch.autograd.grad(loss,z)[0]
        if after_backward is not None:after_backward()
        gnorm=grad.square().mean().sqrt();record['baseline_loss']=float(loss.detach());record['gradient_rms']=float(gnorm.detach())
        if not bool(torch.isfinite(grad).all()) or not bool(torch.isfinite(gnorm)) or not bool(torch.isfinite(loss)):
            if strict:raise RuntimeError('NONFINITE_GRADIENT')
            record['reason']='NONFINITE_GRADIENT';return original,record
        if float(gnorm)==0:
            if strict:raise RuntimeError('ZERO_GRADIENT')
            record['reason']='ZERO_GRADIENT';return original,record
        del rgb,q,loss,z
        scale=min(learning_rate,step_rms/float(gnorm.detach()))
        candidate=(original-scale*grad.detach()).detach()
        # The exact same terminal callback with another full history clone.
        if stage_callback:stage_callback('CANDIDATE','START')
        with torch.no_grad():
            proposed_rgb=rollout(candidate,clone_graph_state(solver_state,torch));proposed_q=readout(proposed_rgb)
            if proposed_rgb.shape!=reference_rgb.shape or proposed_q.shape!=target.shape:raise ValueError('proposal shape')
            proposed_loss=(proposed_q-target).square().mean();quality=(proposed_rgb-reference_rgb).square().mean().sqrt()
        if stage_callback:stage_callback('CANDIDATE','COMPLETE')
        if not bool(torch.isfinite(proposed_loss)) or not bool(torch.isfinite(quality)):raise ValueError('nonfinite proposal')
        record.update(proposed_loss=float(proposed_loss),terminal_rgb_rmse=float(quality),actual_update_rms=float((candidate-original).square().mean().sqrt()))
        record['accepted']=record['proposed_loss']<record['baseline_loss'] and record['terminal_rgb_rmse']<=quality_rms_budget
        record['reason']='ACCEPTED_FIXED_PROPOSAL' if record['accepted'] else 'REJECTED_TERMINAL_LOSS_OR_QUALITY'
        return candidate if record['accepted'] else original,record
    except Exception as exc:
        if strict:raise
        record.update(reason='CALLBACK_OR_DIFFERENTIABILITY_FAILURE',error=repr(exc));return original,record


def receding_feedback(latent,state,*,feedback_indices,advance,rollout,readout,target,reference_rgb,learning_rate,step_rms,cumulative_rms_budget,quality_rms_budget,torch):
    """CPU callback orchestration; no claim of a differentiable Wan implementation.
    Sum of accepted update RMS is bounded. Normal advance occurs after rejection too.
    advance(z, state) must advance exactly one real solver step and update full state.
    """
    if not math.isfinite(cumulative_rms_budget) or cumulative_rms_budget<=0:raise ValueError('cumulative budget')
    if tuple(sorted(set(feedback_indices)))!=tuple(feedback_indices):raise ValueError('ordered unique indices')
    target=target.detach().clone();reference_rgb=reference_rgb.detach().clone()
    state=clone_graph_state(state,torch);z=latent.detach().clone();used=0.;records=[]
    def advance_one(z,state):
        before=state['index'];z,state=advance(z,state)
        if state['index']!=before+1:raise ValueError('advance must increment exactly one')
        return z,state
    for index in feedback_indices:
        while state['index']<index:z,state=advance_one(z,state)
        if state['index']!=index:raise ValueError('advance index mismatch')
        remaining=cumulative_rms_budget-used
        if remaining<=0:record=dict(accepted=False,reason='CUMULATIVE_BUDGET_EXHAUSTED',attempts=0)
        else:
            z,record=terminal_feedback(z,state,rollout,readout,target,reference_rgb,learning_rate=learning_rate,step_rms=min(step_rms,remaining),quality_rms_budget=quality_rms_budget,torch=torch)
            if record['accepted']:used+=record['actual_update_rms']
        record.update(index=index,cumulative_accepted_rms=used);records.append(record)
        z,state=advance_one(z,state)
    return z,state,records
