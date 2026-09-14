"""Fixed saved-state precision comparison; no update selection or prefix rebuild."""
from contextlib import nullcontext
from main.sc_sstw.terminal_feedback import clone_graph_state
from runtime.public_statistic.after47_diagnostic import state_digest


def gradient_comparison(left,right):
    # Called on CPU tensors so diagnostics do not extend GPU graph lifetime.
    import torch
    a=left.double().reshape(-1);b=right.double().reshape(-1);d=a-b
    denom=float(b.norm());norm=float(a.norm())
    return dict(exact=torch.equal(left,right),different_elements=int((left!=right).sum()),max_absolute=float(d.abs().max()),rms_difference=float(d.square().mean().sqrt()),relative_l2=float(d.norm())/denom if denom else None,cosine=float(a.dot(b))/(norm*denom) if norm and denom else None)


def diagnose_precision(adapter,snapshot,report,persist,save):
    torch=adapter.torch;guard=adapter.resource_guard
    origin=snapshot['latent'];state=snapshot['scheduler_state'];target=snapshot['target']
    delta=snapshot['delta'];opposite=snapshot['opposite_delta']
    before=state_digest(state,torch);fixed=state_digest((origin,target,delta,opposite),torch)
    gradients={};report.update(evaluations={},gradient_metrics={},state_before=before,fixed_inputs_before=fixed)
    def evaluate(label,perturbation=None,gradient=False,full_precision=False):
        report['active']=label;persist()
        x=origin.detach().clone() if perturbation is None else (origin+perturbation).detach()
        x.requires_grad_(gradient)
        with (torch.autocast(device_type=x.device.type,enabled=False) if full_precision else nullcontext()),torch.set_grad_enabled(gradient):
            rgb=adapter.rollout(x,clone_graph_state(state,torch));q=adapter.readout(rgb);loss=(q-target).square().mean()
            if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_LOSS')
            report['evaluations'][label]=dict(loss=float(loss.detach()),loss_float64_reduction=float((q.detach().double()-target.double()).square().mean()),q=q.detach().cpu().tolist(),state_unchanged=state_digest(state,torch)==before)
            if gradient:
                guard.consume('backward_calls');g=torch.autograd.grad(loss,x)[0];guard.complete('backward_calls')
                if not bool(torch.isfinite(g).all()):raise RuntimeError('NONFINITE_GRADIENT')
                cpu=g.detach().cpu();gradients[label]=cpu
                report['gradient_metrics'][label]=dict(rms=float(cpu.double().square().mean().sqrt()),g_dot_saved_delta=float((cpu.double()*delta.detach().cpu().double()).sum()))
                save(label,cpu)
        persist()
    # Keep exactly the existing arithmetic settings for the mixed tier.
    report['arithmetic_before']=dict(transformer_input_dtype=str(adapter.input_dtype()),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    for label in ('mixed_gradient_1','mixed_gradient_2'):evaluate(label,gradient=True)
    for suffix,perturbation in (('baseline',None),('plus',delta),('minus',opposite)):evaluate('mixed_'+suffix,perturbation)
    report['gradient_comparisons']=dict(mixed_repeat=gradient_comparison(gradients['mixed_gradient_2'],gradients['mixed_gradient_1']),mixed_vs_saved=gradient_comparison(gradients['mixed_gradient_1'],snapshot['gradient'].detach().cpu()))
    persist()
    # Promote the same rounded parameter values, not a new checkpoint's weights.
    report['active']='PROMOTE_SAME_WEIGHTS_TO_FP32';persist()
    adapter.transformer.float();adapter.prompt=adapter.prompt.float()
    if adapter.negative is not None:adapter.negative=adapter.negative.float()
    old_matmul=torch.backends.cuda.matmul.allow_tf32;old_cudnn=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        report['fp32_scope']=dict(transformer_input_dtype=str(adapter.input_dtype()),same_rounded_weight_values=True,prompt_embeddings='same values promoted to FP32',vae='unchanged FP32',solver='unchanged FP32 with saved mixed-precision history',autocast=False,matmul_allow_tf32=False,cudnn_allow_tf32=False)
        evaluate('fp32_gradient',gradient=True,full_precision=True)
        for suffix,perturbation in (('baseline',None),('plus',delta),('minus',opposite)):evaluate('fp32_'+suffix,perturbation,full_precision=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_matmul;torch.backends.cudnn.allow_tf32=old_cudnn
    report['gradient_comparisons']['fp32_vs_mixed']=gradient_comparison(gradients['fp32_gradient'],gradients['mixed_gradient_1'])
    report['directional_comparisons']={}
    for tier,gradient_label in (('mixed','mixed_gradient_1'),('fp32','fp32_gradient')):
        e=report['evaluations'];b=e[tier+'_baseline']['loss'];p=e[tier+'_plus']['loss'];m=e[tier+'_minus']['loss']
        report['directional_comparisons'][tier]=dict(gradient_vs_no_grad_baseline=e[gradient_label]['loss']-b,plus_loss_change=p-b,minus_loss_change=m-b,central_difference=(p-m)/2,common_change=(p+m)/2-b,first_order_prediction=report['gradient_metrics'][gradient_label]['g_dot_saved_delta'])
    report.update(status='PRECISION_DIAGNOSTIC_EXECUTED_REQUIRES_REVIEW',active=None,state_unchanged=state_digest(state,torch)==before,fixed_inputs_unchanged=state_digest((origin,target,delta,opposite),torch)==fixed,claim='Local consistency diagnosis, not control-method acceptance; no scan or selected direction.')
    persist()
