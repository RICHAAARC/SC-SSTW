"""One bounded fixed-cotangent split audit; no parameter or precision search."""
from main.sc_sstw.terminal_feedback import clone_graph_state
from runtime.public_statistic.after47_precision import gradient_comparison
from runtime.public_statistic.after47_diagnostic import state_digest


def diagnose_split(adapter,snapshot,report,persist,save):
    torch=adapter.torch;guard=adapter.resource_guard
    origin=snapshot['latent'];state=snapshot['scheduler_state'];target=snapshot['target'];delta=snapshot['delta']
    before=state_digest(state,torch);report.update(stages={},active='TERMINAL_INPUT')
    persist()
    def terminal(x):
        local=clone_graph_state(state,torch)
        while local.next_index<len(local.scheduler.timesteps):x,local=adapter.advance(x,local)
        return x
    def release_idle_model(model,label):
        # Only at graph-free stage boundaries: never migrate a pending backward.
        model.to('cpu')
        import gc
        gc.collect()
        if origin.device.type=='cuda':torch.cuda.empty_cache()
        report.setdefault('residency_events',[]).append(dict(stage=label,idle_model_device='cpu',allocated_bytes=torch.cuda.memory_allocated() if origin.device.type=='cuda' else None))
        persist()
    with torch.no_grad():
        u=terminal(origin.detach().clone())
    # The solver forward is finished and has no autograd graph here.
    release_idle_model(adapter.transformer,'VAE_ONLY')
    with torch.no_grad():rgb=adapter.decode_float(u)
    # Freeze the upstream RGB cotangent once, then use it for BOTH VAE VJPs.
    probe=rgb.detach().requires_grad_(True);loss=(adapter.readout(probe)-target).square().mean()
    guard.consume('backward_calls');upstream_rgb=torch.autograd.grad(loss,probe)[0].detach();guard.complete('backward_calls')
    report['baseline_loss']=float(loss.detach());del probe,loss,rgb
    vae_gradients=[]
    for index in (1,2):
        label='vae_vjp_'+str(index);report['active']=label;persist()
        x=u.detach().clone().requires_grad_(True);decoded=adapter.decode_float(x)
        guard.consume('backward_calls');g=torch.autograd.grad(decoded,x,grad_outputs=upstream_rgb)[0];guard.complete('backward_calls')
        if not bool(torch.isfinite(g).all()):raise RuntimeError('NONFINITE_VAE_VJP')
        vae_gradients.append(g.detach().cpu());save(label,vae_gradients[-1]);report['stages'][label]=dict(complete=True,gradient_rms=float(g.detach().double().square().mean().sqrt()))
        del x,decoded,g;persist()
    del upstream_rgb
    report['vae_repeat']=gradient_comparison(vae_gradients[1],vae_gradients[0]);persist()
    # Both VAE graphs were consumed and their locals deleted above.
    release_idle_model(adapter.vae,'SOLVER_ONLY')
    adapter.transformer.to(origin.device)
    # Freeze the FIRST VAE VJP for BOTH solver/Transformer VJPs.
    upstream_u=vae_gradients[0].to(origin.device);solver_gradients=[]
    for index in (1,2):
        label='solver_vjp_'+str(index);report['active']=label;persist()
        x=origin.detach().clone().requires_grad_(True);end=terminal(x)
        output_matches=bool(torch.equal(end.detach(),u))
        guard.consume('backward_calls');g=torch.autograd.grad(end,x,grad_outputs=upstream_u)[0];guard.complete('backward_calls')
        if not bool(torch.isfinite(g).all()):raise RuntimeError('NONFINITE_SOLVER_VJP')
        solver_gradients.append(g.detach().cpu());save(label,solver_gradients[-1]);report['stages'][label]=dict(complete=True,terminal_matches_baseline=output_matches,g_dot_fixed_delta=float((g.detach().double()*delta.double()).sum()))
        del x,end,g;persist()
    report['solver_repeat']=gradient_comparison(solver_gradients[1],solver_gradients[0])
    report.update(status='SPLIT_DIAGNOSTIC_EXECUTED_REQUIRES_REVIEW',active=None,state_unchanged=state_digest(state,torch)==before,claim='One split repeatability audit only. No candidate accepted; no new delta; no automatic follow-up.')
    persist()
