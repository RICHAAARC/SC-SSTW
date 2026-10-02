"""Split encoder/decoder VJPs for one saved-terminal channel-control proposal."""
from __future__ import annotations
from contextlib import contextmanager
import os,shutil,tempfile,time
from pathlib import Path
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import vae as shared
from runtime.wan.gradient_checkpointing import BoundarySpool,BoundaryStorage

class OnDemandSpool(BoundarySpool):
    """Reuse bounded chunked disk storage, without reserving 80 GiB up front.

    The existing 80 GiB live-disk limit and 2 GiB free margin apply to actual
    writes. Encoder and decoder spools never coexist. No hardware-name gate.
    """
    def __init__(self):
        self.kind='VAE'
        root=Path(os.environ.get('RGB_DCT_BOUNDARY_SPOOL_ROOT') or ('/content' if Path('/content').is_dir() else '/tmp'))
        self.directory=tempfile.TemporaryDirectory(prefix='wan-margin-boundary-',dir=root)
        self.path=Path(self.directory.name);self.disk_root=str(root)
        self.disk_free_at_start_bytes=shutil.disk_usage(root).free
        self.live_disk_bytes=self.peak_disk_bytes=0
        self.cpu_live_packed_bytes=self.cpu_peak_packed_bytes=0
        self.d2h_bytes=self.h2d_bytes=0
        self.disk_written_bytes=self.disk_read_bytes=self.files=0
        self.closed=False
    def summary(self):
        result=super().summary();result['preflight_boundary_bytes']=0
        return result

@contextmanager
def checkpoint_native(vae,component,count,receipt,spool_factory=OnDemandSpool):
    """Functionalize native causal cache for exact chunk recomputation.

    Native encode/decode, spatial support, cache history and normalization are
    unchanged. This temporary instance-local wrapper handles both components.
    """
    import torch
    from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
    module=getattr(vae,component);original=module.forward
    spool=spool_factory();chunks=0;started=time.perf_counter()
    device=next(vae.parameters()).device
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    shared._clear_cache(vae)
    def forward(x,feat_cache=None,feat_idx=None,**kwargs):
        nonlocal chunks
        if feat_cache is None or feat_idx is None or feat_idx[0]!=0:
            raise ValueError('native causal cache/cursor unavailable')
        layout=[];inputs=[]
        for v in feat_cache:
            if torch.is_tensor(v):layout.append(('tensor',len(inputs)));inputs.append(v)
            elif v is None:layout.append(('none',None))
            elif isinstance(v,str) and v=='Rep':layout.append(('rep',None))
            else:raise ValueError('unsupported causal cache element')
        expected=None;consumed=None;calls=0
        def functional(value,*values):
            nonlocal expected,consumed,calls
            phase='forward' if calls==0 else 'recompute';calls+=1
            kind=component+'_chunk_'+phase;count(kind,False)
            local=[values[i] if k=='tensor' else ('Rep' if k=='rep' else None) for k,i in layout]
            cursor=[0];output=original(value,feat_cache=local,feat_idx=cursor,**kwargs)
            kinds=[];tensors=[]
            for v in local:
                if torch.is_tensor(v):kinds.append('tensor');tensors.append(v)
                elif v is None:kinds.append('none')
                elif isinstance(v,str) and v=='Rep':kinds.append('rep')
                else:raise ValueError('unsupported output cache element')
            if expected is None:expected=tuple(kinds);consumed=cursor[0]
            elif expected!=tuple(kinds) or consumed!=cursor[0]:raise ValueError('causal replay changed cache layout')
            count(kind,True);return (output,*tensors)
        storage=BoundaryStorage(spool)
        try:
            with torch.autograd.graph.saved_tensors_hooks(storage.pack,storage.unpack),set_checkpoint_early_stop(False):
                result=checkpoint(functional,x,*inputs,use_reentrant=False,preserve_rng_state=True)
        finally:storage.copies.clear()
        values=iter(result[1:])
        feat_cache[:]=[next(values) if k=='tensor' else ('Rep' if k=='rep' else None) for k in expected]
        feat_idx[0]=consumed;chunks+=1
        return result[0]
    module.forward=forward
    try:yield
    finally:
        module.forward=original;shared._clear_cache(vae)
        receipt.update(component=component,chunks=chunks,elapsed_seconds=time.perf_counter()-started,storage=spool.summary())
        receipt['cuda_peak_allocated']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        receipt['cuda_peak_reserved']=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
        spool.close();receipt['storage_cleanup_complete']=True


def encoder_cotangent(vae,rgb,key,count,receipt,*,spool_factory=OnDemandSpool,objective="composite"):
    import torch
    device=next(vae.parameters()).device
    leaf=rgb.to(device=device,dtype=torch.float32).detach().requires_grad_(True)
    with torch.enable_grad(),checkpoint_native(vae,'encoder',count,receipt,spool_factory):
        count('vae_encode_gradient',False)
        raw=vae.encode(leaf.permute(3,0,1,2).unsqueeze(0)*2-1).latent_dist.mode()
        mean,std=shared._scale_tensors(vae,raw);normalized=(raw.float()-mean)/std
        count('vae_encode_gradient',True)
        loss,metrics=method.margin_loss(method.project(normalized,key),key,objective=objective)
        if not loss.requires_grad:raise RuntimeError('encoder gradient detached')
        count('encoder_vjp',False);cotangent,=torch.autograd.grad(loss,leaf);count('encoder_vjp',True)
    if not bool(cotangent.isfinite().all()):raise FloatingPointError('nonfinite encoder cotangent')
    return normalized.detach().cpu(),cotangent.detach().cpu(),metrics


def decoder_vjp(vae,terminal,cotangent,baseline_rgb,count,receipt,*,spool_factory=OnDemandSpool):
    import torch
    device=next(vae.parameters()).device
    leaf=terminal.to(device=device,dtype=torch.float32).detach().requires_grad_(True)
    with torch.enable_grad(),checkpoint_native(vae,'decoder',count,receipt,spool_factory):
        mean,std=shared._scale_tensors(vae,leaf);count('vae_decode_gradient',False)
        decoded=vae.decode(leaf*std+mean,return_dict=False)[0]
        rgb=(decoded[0].permute(1,2,3,0)/2+.5).clamp(0,1)
        count('vae_decode_gradient',True)
        # Full output equality check without allocating another complete GPU movie.
        maximum=max(float((rgb[i].detach().cpu()-baseline_rgb[i]).abs().max()) for i in range(len(rgb)))
        receipt['forward_rgb_max_error']=maximum
        if maximum>1e-6:raise ValueError('checkpoint decode differs from native baseline')
        count('decoder_vjp',False)
        gradient,=torch.autograd.grad(rgb,leaf,grad_outputs=cotangent.to(device))
        count('decoder_vjp',True)
    if not bool(gradient.isfinite().all()):raise FloatingPointError('nonfinite decoder gradient')
    return gradient.detach().cpu()
