"""Two user-run GPU stages, sharing one actual VAE within each stage.

CPU coefficients are consumed unchanged; no solver, DiT, training or codec here.
"""
from pathlib import Path
import time
import math
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p
from main.tube_state.b_framewise_qim_v1_solver import budget_metrics
from .b_terminal_three_group_backend_v1 import WanBackend
from .b_framewise_qim_v1_operator import validate_head, replay_head
from .b_framewise_qim_v1_cache import check_disk_cache_capacity
from .vae import decode_normalized_latent, quantize_rgb8_no_codec

CHUNKS = (0,)+tuple(range(1,181,4))


def array(path, shape, dtype):
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    if value.shape!=shape or value.dtype!=dtype: raise ValueError('invalid array geometry/dtype: '+str(path))
    return value


def stats(delta):
    if not delta.numel(): return dict(count=0,rms=None,linf=None,psnr=None)
    rms=float(delta.square().mean().sqrt())
    return dict(count=delta.numel(),rms=rms,linf=float(delta.abs().max()),psnr=None if rms==0 else -20*math.log10(rms))


class Backend:
    def __init__(self,model_id):
        self.wan=WanBackend(model_id);self.loaded=False;self.encode_seconds={'n01':0.,'e02':0.}

    def load(self):
        started=time.monotonic()
        if self.wan.device=='cuda':torch.cuda.reset_peak_memory_stats()
        result=self.wan.load_vae();self.loaded=True
        validate_head(self.wan.vae.decoder.conv_out)
        if time.monotonic()-started>900:
            error=TimeoutError('VAE load exceeded 900 seconds after return');error.physical_returned=True;raise error
        return dict(result,wall_seconds=time.monotonic()-started,forward=0)

    def capture(self,terminal,output,event):
        """The sole e02 D181, including all 46 actual baseline-head calls."""
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        z=torch.load(terminal,map_location='cpu',weights_only=True)
        if (not isinstance(z,torch.Tensor) or z.dtype!=torch.float32 or tuple(z.shape)!=(1,16,46,40,64)
                or not bool(torch.isfinite(z).all())):
            raise ValueError('original finite FP32 normalized P0 [1,16,46,40,64] required')
        vae=self.wan.vae;channels=vae.decoder.conv_out.in_channels
        check_disk_cache_capacity(output,channels)
        folder=output/'features';folder.mkdir()
        pre=np.lib.format.open_memmap(output/'BASE_preclip.npy',mode='w+',dtype=np.float32,shape=(181,320,512,3))
        cursor=0
        def before(module,args): event('baseline_head',False)
        def captured(module,args,value):
            nonlocal cursor
            h=args[0];length=int(h.shape[2])
            if (h.dtype!=torch.float32 or tuple(h.shape[:2])!=(1,channels) or tuple(h.shape[-2:])!=(320,512)
                    or length!=(1 if cursor==0 else 4) or cursor+length>181
                    or not bool(torch.isfinite(h).all() and torch.isfinite(value).all())):
                raise ValueError('invalid actual native head capture')
            torch.save(h.detach().cpu(),folder/f'{cursor:03}.pt')
            pre[cursor:cursor+length]=(value.detach()[0].permute(1,2,3,0).float()/2+.5).cpu().numpy()
            cursor+=length;event('baseline_head',True)
        hooks=[vae.decoder.conv_out.register_forward_pre_hook(before),vae.decoder.conv_out.register_forward_hook(captured)]
        try:
            event('D181',False)
            with torch.inference_mode(): rgb=decode_normalized_latent(vae,z.to(self.wan.device))
            event('D181',True)
            if cursor!=181: raise ValueError('incomplete D181 capture')
            for t in range(0,181,4):
                expected=torch.from_numpy(np.array(pre[t:t+4],copy=True)).clamp(0,1)
                if not torch.equal(expected,rgb[t:t+4].detach().cpu()): raise ValueError('head capture differs from native output')
            pre.flush()
        finally:
            for hook in hooks: hook.remove()
        return dict(base=str(output/'BASE_preclip.npy'),features=str(folder),frames=181)

    def replay(self,cpu,baseline,features,output,event,dump):
        cpu,output=Path(cpu),Path(output);output.mkdir(parents=True,exist_ok=True)
        base=array(baseline,(181,320,512,3),np.float32)
        q0=array(cpu/'q0.npy',(181,10,63),np.float64)
        target=array(cpu/'fixed_targets.npy',(181,10,63),np.float64)
        with np.load(cpu/'schedule.npz',allow_pickle=False) as schedule: active=torch.from_numpy(schedule['active'].copy())
        if active.shape!=(181,10) or active.dtype!=torch.bool: raise ValueError('CPU active layout')
        expected=(torch.arange(181)[:,None]>=torch.arange(5)[None,:]).repeat(1,2)
        if not torch.equal(active,expected):raise ValueError('fixed startup unwritten-lane semantics required')
        a=torch.load(cpu/'coefficients.pt',map_location='cpu',weights_only=True)
        predicted=torch.load(cpu/'prediction.pt',map_location='cpu',weights_only=True)
        channels=self.wan.vae.decoder.conv_out.in_channels
        if (a.dtype!=torch.float64 or tuple(a.shape)!=(181,10,channels,7,9) or not bool(torch.isfinite(a).all())
                or predicted.dtype!=torch.float64 or tuple(predicted.shape)!=(181,10,3,58,74)
                or not bool(torch.isfinite(predicted).all())): raise ValueError('invalid finite CPU coefficients/prediction')
        paths=[Path(features)/f'{t:03}.pt' for t in CHUNKS]
        # Cache absence is a source failure, never permission for an extra capture.
        if any(not path.is_file() for path in paths): raise FileNotFoundError('incomplete original 46-feature cache')
        def chunks():
            for path in paths:
                h=torch.load(path,map_location='cpu',weights_only=True)
                if not isinstance(h,torch.Tensor) or h.shape[:2]!=(1,channels) or h.shape[-2:]!=(320,512):
                    raise ValueError('feature geometry: '+str(path))
                yield h.to(self.wan.device)
        out=np.lib.format.open_memmap(output/'NEW_preclip.npy',mode='w+',dtype=np.float32,shape=(181,320,512,3))
        mask=p.support(active);frames=[];qdelta=[];max_target=0.;max_prediction=0.;max_host=0.;max_excess=0.
        for start,value in replay_head(self.wan.vae.decoder.conv_out,chunks(),a,active,event):
            value=value.detach().cpu();length=len(value)
            original=torch.from_numpy(np.array(base[start:start+length],copy=True))
            if not bool(torch.isfinite(value).all() and torch.isfinite(original).all()): raise ValueError('nonfinite native pixels')
            delta=value.double()-original.double();packed=p.pack(delta)
            measured=budget_metrics(packed,mask[start:start+length])
            full_mask=torch.zeros((length,320,512,3),dtype=torch.bool)
            for tile,(y,x) in enumerate(p.ROIS):
                h=min(58,321-y)
                full_mask[:,y-1:y-1+h,x-1:x+73]=mask[start:start+length,tile,:,:h].permute(0,2,3,1)
            measured['outside_full_canvas']=float(torch.where(full_mask,0,delta).abs().max())
            measured['excess']=max(measured['excess'],measured['outside_full_canvas'])
            max_excess=max(max_excess,measured['excess'])
            max_prediction=max(max_prediction,float((packed-predicted[start:start+length]).abs().max()))
            q=255*p.project(p.pack(value.double()))
            fixed_target=torch.from_numpy(np.array(target[start:start+length],copy=True))
            saved_q0=torch.from_numpy(np.array(q0[start:start+length],copy=True))
            if not bool(torch.isfinite(fixed_target).all() and torch.isfinite(saved_q0).all()):raise ValueError('nonfinite CPU q0/targets')
            max_target=max(max_target,float(torch.where(active[start:start+length,:,None],q-fixed_target,0).abs().max()))
            host_q=255*p.project(p.pack(original.double()))
            diff=host_q-saved_q0;qdelta.append(diff.numpy())
            max_host=max(max_host,float(diff.abs().max()))
            rgb_delta=(quantize_rgb8_no_codec(value).double()-quantize_rgb8_no_codec(original).double())/255
            for k in range(length):
                frames.append(dict(frame=start+k,preclip=dict(whole=stats(delta[k]),roi=stats(delta[k][full_mask[k]]),outside=stats(delta[k][~full_mask[k]])),
                    rgb8=dict(whole=stats(rgb_delta[k]),roi=stats(rgb_delta[k][full_mask[k]]),outside=stats(rgb_delta[k][~full_mask[k]])),
                    base_clip_rate=float(((original[k]<0)|(original[k]>1)).double().mean()),
                    new_clip_rate=float(((value[k]<0)|(value[k]>1)).double().mean())))
            dump(output/f'native_chunk_{start:03}.json',dict(start=start,**measured))
            out[start:start+length]=value.numpy();out.flush()
        np.save(output/'host_q0_difference.npy',np.concatenate(qdelta),allow_pickle=False)
        result=dict(status='NATIVE_MEASURED',preclip='NEW_preclip.npy',frames=181,target_max=max_target,
                    linear_prediction_max=max_prediction,host_q0_difference_max=max_host,
                    native_residual_status='NATIVE_RESIDUAL_WITHIN_TOLERANCE' if max_excess<=1e-7 and max_target<=1e-3 else 'NUMERICALLY_UNRESOLVED',
                    original_tolerances=dict(excess=1e-7,target_max=1e-3),maximum_budget_excess=max_excess,
                    host_comparison='HOST_COMPARISON_LIMITED' if max_host>0 else 'SAME_SAVED_Q0',
                    targets_replaced=False,new_K=0,formal_output=None,quality_pass=None)
        dump(output/'native.json',result);dump(output/'quality.json',dict(frames=frames,quality_pass=None,reference='same native BASE'))
        return result

    def check_encode_budget(self,source):
        if self.encode_seconds[source]>=7200: raise TimeoutError('per-source cumulative E41 7200 seconds consumed')

    def encode(self,pixels,start,source):
        self.check_encode_budget(source)
        if pixels.dtype!=np.uint8 or pixels.ndim!=4 or pixels.shape[1:]!=(320,512,3) or not 0<=start<=len(pixels)-41:
            raise ValueError('received uint8 RGB and actual E41 bounds required')
        began=time.monotonic()
        try:
            with torch.inference_mode(): result=self.wan.encode_view(torch.from_numpy(np.array(pixels[start:start+41],copy=True)))
            if time.monotonic()-began>600 or self.encode_seconds[source]+time.monotonic()-began>7200:
                error=TimeoutError('E41 individual/cumulative limit exceeded after return');error.physical_returned=True;raise error
            return result
        finally:self.encode_seconds[source]+=time.monotonic()-began

    def project(self,latent): return self.wan.payload_project(latent,list(range(1,11)),['R0','R1'])
    def bits(self,projection,view): return self.wan.payload_decode(projection,'watermark',view,'key00')

    def close(self):
        peak=int(torch.cuda.max_memory_allocated()) if self.wan.device=='cuda' else None
        if self.loaded:self.wan.release_vae()
        return dict(E41_seconds=self.encode_seconds,device=self.wan.device,peak_cuda_allocated_bytes=peak,
                    model_E41_limit_enforcement='elapsed checked after return; no hard model interruption guarantee')
