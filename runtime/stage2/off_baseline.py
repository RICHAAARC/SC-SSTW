"""Four fixed OFF outputs, no control intervention and no automatic quality gate."""
import copy
import gc
import importlib.metadata
import json
import sys
import time
import types
from pathlib import Path
from .wan_translation import write,velocity,continue_two,decode_rgb,encode,read_public

OUTPUTS=('M8','P8','V8','P50')


def fresh_scheduler(scheduler_class,config):
    return scheduler_class.from_config(copy.deepcopy(config))


def tensor_comparison(left,right):
    import torch
    if left.shape!=right.shape:return dict(shape_equal=False)
    d=left.float()-right.float()
    return dict(shape_equal=True,exact=bool(torch.equal(left,right)),max_absolute=float(d.abs().max()),rms=float(d.square().mean().sqrt()))


def describe_rgb(array,np):
    x=array.astype(np.float64)
    temporal=np.abs(x[1:]-x[:-1])
    return dict(frames=len(x),minimum=int(x.min()),maximum=int(x.max()),mean=float(x.mean()),std=float(x.std()),
         per_frame_std=x.std(axis=(1,2,3)).tolist(),per_frame_mean=x.mean(axis=(1,2,3)).tolist(),
         spatial_neighbor_absolute_mean=float((np.abs(x[:,:,1:]-x[:,:,:-1]).mean()+np.abs(x[:,1:]-x[:,:-1]).mean())/2),
         per_pair_temporal_max=temporal.max(axis=(1,2,3)).tolist(),per_pair_temporal_mean=temporal.mean(axis=(1,2,3)).tolist(),
         visual_clarity_and_motion='NOT_AUTOMATICALLY_ADJUDICATED')


def save_preview(array,path):
    from PIL import Image,ImageDraw
    h,w=array.shape[1:3];im=Image.new('RGB',(3*w,h+24),'white')
    for j,i in enumerate((0,24,48)):
        im.paste(Image.fromarray(array[i]),(j*w,24));ImageDraw.Draw(im).text((j*w+4,4),f'frame {i}, t={i/8}s',fill='black')
    im.save(path)



def load_fp32_vae(vae_class,model,torch):
    """Reload source weights; do not upcast an existing BF16 model instance."""
    return vae_class.from_pretrained(model['id'],revision=model['revision'],subfolder='vae',torch_dtype=torch.float32).eval()


def stock_latent(pipe,initial,g,steps,arm_path,torch):
    # Observe the actual public pipeline preparation, without changing its value.
    original=pipe.prepare_latents;seen=[]
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);seen.append(value.detach().cpu().clone())
        return value
    pipe.prepare_latents=capture
    try:
        output=pipe(prompt=g['prompt'],negative_prompt=g['negative_prompt'],height=g['height'],width=g['width'],
            num_frames=g['frames'],num_inference_steps=steps,guidance_scale=g['guidance'],num_videos_per_prompt=1,
            generator=torch.Generator(device=initial.device).manual_seed(g['seed']),latents=initial.clone(),
            max_sequence_length=g['max_sequence_length'],output_type='latent',return_dict=True)
    finally:pipe.prepare_latents=original
    if len(seen)!=1:raise RuntimeError('official prepare_latents did not expose exactly one initial state')
    torch.save(seen[0],arm_path/'actual_initial_latent.pt')
    comparison=tensor_comparison(seen[0],initial.cpu());write(arm_path/'initial_comparison.json',comparison)
    if not comparison.get('exact'):raise RuntimeError('official initial latent differs from frozen common initial')
    return output.frames.detach()


def manual_latent(pipe,initial,g,path,torch):
    device=initial.device
    cond,uncond=pipe.encode_prompt(prompt=g['prompt'],negative_prompt=g['negative_prompt'],do_classifier_free_guidance=True,
       num_videos_per_prompt=1,max_sequence_length=g['max_sequence_length'],device=device)
    cond=cond.to(pipe.transformer.dtype);uncond=uncond.to(pipe.transformer.dtype)
    pipe.scheduler.set_timesteps(8,device=device)
    if hasattr(pipe.scheduler,'set_begin_index'):pipe.scheduler.set_begin_index(0)
    ts=pipe.scheduler.timesteps;z=initial.clone()
    torch.save(z.cpu(),path/'actual_initial_latent.pt')
    for i in range(6):
        cv=velocity(pipe,z,ts[i],cond,'cond');uv=velocity(pipe,z,ts[i],uncond,'uncond')
        z=pipe.scheduler.step(uv+g['guidance']*(cv-uv),ts[i],z,return_dict=False)[0].detach()
    snapshot=copy.deepcopy(pipe.scheduler)
    torch.save(z.cpu(),path/'prefix_step5_latent.pt')
    write(path/'manual_prefix_state.json',dict(step_index=snapshot._step_index,timesteps=ts.cpu().tolist(),unshifted_history=True))
    return continue_two(pipe,z.clone(),snapshot,ts,cond,uncond,g['guidance'])



def attach_forward_counter(transformer,counts,trace,active_output):
    """Module hook survives Accelerate replacing/restoring .forward."""
    def count(module,args,kwargs):
        counts['transformer_calls']+=1
        z=kwargs.get('hidden_states');t=kwargs.get('timestep')
        if z is None or t is None:raise RuntimeError('transformer instrumentation arguments missing')
        trace.append(dict(output=active_output(),call=counts['transformer_calls'],shape=list(z.shape),dtype=str(z.dtype),timestep=t.detach().cpu().tolist()))
    return transformer.register_forward_pre_hook(count,with_kwargs=True)


def run(config_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path(config_path).read_text());g=c['generation'];write(out/'config.json',c)
    states={a:dict(status='NOT_EXECUTED') for a in OUTPUTS};write(out/'output_status.json',states)
    counts={'transformer_calls':0,'vae_decodes':0,'mp4':0};trace=[];encode_log=[];start=time.monotonic();active=None;counter_handle=None
    def progress(stage):print(json.dumps(dict(stage=stage,counts=counts,elapsed=time.monotonic()-start)),flush=True)
    try:
        import torch,numpy as np,diffusers
        from diffusers import WanPipeline,AutoencoderKLWan,UniPCMultistepScheduler
        if diffusers.__version__!=c['model']['diffusers']:raise RuntimeError('requires diffusers0.35.2')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 GPU required')
        write(out/'runtime.json',dict(torch=torch.__version__,diffusers=diffusers.__version__,python=sys.version,cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(),packages={x:importlib.metadata.version(x) for x in ('transformers','accelerate','ftfy','sentencepiece','safetensors','huggingface_hub','numpy','Pillow')}))
        torch.cuda.reset_peak_memory_stats();progress('load_bf16_pipeline')
        pipe=WanPipeline.from_pretrained(c['model']['id'],revision=c['model']['revision'],torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload();device=pipe._execution_device
        if type(pipe.scheduler).__name__!='UniPCMultistepScheduler':raise RuntimeError('scheduler class mismatch')
        scheduler_config=dict(pipe.scheduler.config);write(out/'scheduler_config.json',scheduler_config)
        counter_handle=attach_forward_counter(pipe.transformer,counts,trace,lambda:active)
        fp32_vae=None;finals={};arrays={};observations={};metrics={};comparisons={}
        with torch.inference_mode():
            initial=pipe.prepare_latents(1,int(pipe.transformer.config.in_channels),g['height'],g['width'],g['frames'],torch.float32,
                  device,torch.Generator(device=device).manual_seed(g['seed']),None).detach()
            if tuple(initial.shape)!=(1,16,13,40,64):raise RuntimeError('initial latent geometry')
            torch.save(initial.cpu(),out/'common_initial_latent.pt')
            for arm in OUTPUTS:
                active=arm;path=out/arm;path.mkdir();before=counts['transformer_calls'];progress(arm+'_begin')
                try:
                    spec=c['outputs'][arm]
                    if arm=='V8':
                        if 'P8' not in finals:raise RuntimeError('P8 final missing; V8 dependency blocked')
                        final=finals['P8'].to(device).clone()
                        write(path/'reuse.json',dict(source='../P8/final_latent.pt',no_denoising=True,exact=tensor_comparison(final.cpu(),finals['P8'])))
                    else:
                        pipe.scheduler=fresh_scheduler(UniPCMultistepScheduler,scheduler_config)
                        final=manual_latent(pipe,initial,g,path,torch) if arm=='M8' else stock_latent(pipe,initial,g,spec['steps'],path,torch)
                        write(path/'timesteps.json',dict(steps=spec['steps'],values=pipe.scheduler.timesteps.cpu().tolist(),scheduler_class=type(pipe.scheduler).__name__))
                    if tuple(final.shape)!=tuple(initial.shape) or not bool(torch.isfinite(final).all()):raise RuntimeError('invalid final latent')
                    finals[arm]=final.detach().cpu().clone();torch.save(finals[arm],path/'final_latent.pt')
                    actual_calls=counts['transformer_calls']-before
                    if actual_calls!=spec['transformer_calls']:raise RuntimeError('actual transformer count mismatch')
                    if spec['vae']=='bf16':
                        counts['vae_decodes']+=1;array=decode_rgb(pipe,final,torch)
                    else:
                        # Reload original checkpoint at FP32; never .float() quantized BF16 weights.
                        pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
                        if fp32_vae is None:
                            progress('load_original_fp32_vae')
                            fp32_vae=load_fp32_vae(AutoencoderKLWan,c['model'],torch)
                            write(out/'fp32_vae_provenance.json',dict(model=c['model'],subfolder='vae',loaded_dtype=str(fp32_vae.dtype),reloaded_original_weights=True,not_bf16_upcast=True))
                        fp32_vae.to(device);counts['vae_decodes']+=1
                        array=decode_rgb(types.SimpleNamespace(vae=fp32_vae),final,torch)
                        fp32_vae.to('cpu')
                    np.save(path/'preencode_rgb.npy',array,allow_pickle=False);arrays[arm]=array
                    write(path/'preencode_rgb_metrics.json',describe_rgb(array,np));save_preview(array,path/'first_middle_last.png')
                    encode(array,path/'preencode_lossless.mkv',True,encode_log);encode(array,path/'saved.mp4',False,encode_log);counts['mp4']+=1
                    del final;pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
                    streams={}
                    for label,filename in (('preencode','preencode_lossless.mkv'),('mp4','saved.mp4')):
                        dest=path/label;dest.mkdir();streams[label]=read_public(path/filename,dest,c)
                        meta=json.loads((dest/'decode_metadata.json').read_text())
                        gray=np.frombuffer((dest/'decoded_gray.raw').read_bytes(),dtype=np.uint8).reshape(meta['frames'],meta['height'],meta['width'])
                        d=np.abs(gray[1:].astype(np.int16)-gray[:-1].astype(np.int16))
                        write(dest/'gray_temporal_metrics.json',dict(frames=len(gray),per_pair_max=d.max(axis=(1,2)).tolist(),per_pair_mean=d.mean(axis=(1,2)).tolist(),per_pair_support_gt12=(d>12).sum(axis=(1,2)).tolist(),valid_count=sum(v['valid'] for v in streams[label])))
                    observations[arm]=streams;states[arm]=dict(status='COMPLETE',transformer_calls=actual_calls,vae_decodes=1,mp4=1,visual_review='PENDING_HUMAN_CLARITY_AND_MOTION')
                except Exception as exc:
                    states[arm]=dict(status='OPERATIONAL_BLOCKED',error=repr(exc),transformer_calls=counts['transformer_calls']-before)
                    write(path/'failure.json',states[arm])
                    if isinstance(exc,(torch.cuda.OutOfMemoryError,TimeoutError)):raise
                finally:
                    write(out/'output_status.json',states);write(out/'actual_transformer_calls.json',trace);write(out/'encode_processes.json',encode_log)
            for a,b in (('M8','P8'),('P8','V8'),('V8','P50')):
                if a in finals and b in finals:
                    comp=dict(final_latent=tensor_comparison(finals[a],finals[b]))
                    if a in arrays and b in arrays:
                        d=arrays[a].astype(np.float64)-arrays[b].astype(np.float64)
                        comp['preencode_rgb']=dict(exact=bool(np.array_equal(arrays[a],arrays[b])),max_absolute=float(np.abs(d).max()),rmse_0_1=float(np.sqrt(np.mean(d*d))/255))
                    comparisons[a+'_vs_'+b]=comp
            complete=all(s['status']=='COMPLETE' for s in states.values()) and counts==dict(transformer_calls=132,vae_decodes=4,mp4=4)
            result=dict(status='OFF_DIAGNOSTIC_EXECUTED_REQUIRES_VISUAL_REVIEW' if complete else 'OPERATIONAL_BLOCKED',outputs=states,counts=counts,
               comparisons=comparisons,elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               science_denominator=0,no_automatic_clarity_pass=True,no_global_causal_attribution=True,gpu_executed=True)
            write(out/'result.json',result);progress('complete');return result
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),counts=counts,outputs=states,elapsed_seconds=time.monotonic()-start));raise

    finally:
        if counter_handle is not None:counter_handle.remove()
