"""Single fixed Wan latent-translation diagnostic. GPU execution is separate."""
import copy
import importlib.metadata
import sys
import dataclasses
import gc
import json
import subprocess
import time
from pathlib import Path
from main.sc_sstw.latent_translation import ARMS,translate_latent,evaluate_response
from runtime.stage1.observation import decode_video,observe_frames,ObserverConfig


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fork_scheduler(snapshot):
    if type(snapshot).__name__!='UniPCMultistepScheduler' or snapshot._step_index!=6:
        raise ValueError('expected full UniPC state after zero-based step5')
    return copy.deepcopy(snapshot)


def velocity(pipe,z,t,embedding,branch):
    with pipe.transformer.cache_context(branch):
        return pipe.transformer(hidden_states=z.to(pipe.transformer.dtype),timestep=t.expand(z.shape[0]),
            encoder_hidden_states=embedding,attention_kwargs=None,return_dict=False)[0].detach()


def continue_two(pipe,z,snapshot,timesteps,cond,uncond,guidance,on_call=None):
    scheduler=fork_scheduler(snapshot)
    for i in (6,7):
        if scheduler._step_index!=i: raise RuntimeError('scheduler index drift')
        cv=velocity(pipe,z,timesteps[i],cond,'cond')
        if on_call:on_call()
        uv=velocity(pipe,z,timesteps[i],uncond,'uncond')
        if on_call:on_call()
        z=scheduler.step(uv+guidance*(cv-uv),timesteps[i],z,return_dict=False)[0].detach()
    if scheduler._step_index!=8 or snapshot._step_index!=6: raise RuntimeError('fork changed prefix')
    return z


def decode_rgb(pipe,z,torch):
    vae=pipe.vae
    v=z.to(vae.dtype)
    shape=(1,int(vae.config.z_dim),1,1,1)
    mean=torch.tensor(vae.config.latents_mean,device=v.device,dtype=v.dtype).reshape(shape)
    inverse=1/torch.tensor(vae.config.latents_std,device=v.device,dtype=v.dtype).reshape(shape)
    decoded=vae.decode(v/inverse+mean,return_dict=False)[0]
    if tuple(decoded.shape)!=(1,3,49,320,512) or not bool(torch.isfinite(decoded).all()):
        raise RuntimeError('VAE shape/nonfinite')
    return ((decoded[0].permute(1,2,3,0).float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).cpu().numpy()


def encode(array,path,lossless,log):
    # Real VAE uint8 RGB is encoded unchanged; no rendered/shifted final pixels.
    h,w=array.shape[1:3]
    cmd=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
         '-framerate','8','-i','pipe:0','-an']
    cmd+=['-c:v','ffv1','-pix_fmt','bgr0'] if lossless else ['-c:v','libx264','-crf','18','-pix_fmt','yuv420p']
    cmd+=['-threads','1','-n',str(path)]
    begin=time.monotonic();r=subprocess.run(cmd,input=array.tobytes(),capture_output=True,timeout=120)
    log.append(dict(command=cmd,exit_code=r.returncode,seconds=time.monotonic()-begin,stderr=r.stderr.decode(errors='replace')))
    if r.returncode: raise RuntimeError('encode failed: '+r.stderr.decode(errors='replace'))


def read_public(path,out,config):
    c=config['observer'];log=[];original_run=subprocess.run
    def traced(*args,**kwargs):
        begin=time.monotonic()
        try:
            r=original_run(*args,**kwargs)
            log.append(dict(command=args[0] if args else kwargs.get('args'),exit_code=r.returncode,seconds=time.monotonic()-begin))
            return r
        except Exception as exc:
            log.append(dict(command=args[0] if args else kwargs.get('args'),exit_code=getattr(exc,'returncode',None),seconds=time.monotonic()-begin,error=repr(exc)))
            raise
    subprocess.run=traced
    try:
        decoded=decode_video(str(path),sample_hz=c['sample_hz'],max_frames=c['max_frames'],max_pixels=c['max_pixels'],timeout_seconds=c['timeout_seconds'])
    finally:
        subprocess.run=original_run
        write(out/'decode_processes.json',log)
    width,height,frames=decoded
    cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time','-of','json',str(path)]
    probe=original_run(cmd,capture_output=True,check=True,timeout=60)
    metadata=json.loads(probe.stdout);write(out/'source_probe.json',metadata)
    pts=[float(r['best_effort_timestamp_time']) for r in metadata['frames']]
    if len(pts)!=49 or any(abs(v-i/8)>1e-6 for i,v in enumerate(pts)) or (width,height)!=(512,320):
        raise RuntimeError('source frame/PTS/geometry mismatch')
    log.append(dict(command=cmd,exit_code=probe.returncode))
    write(out/'decode_processes.json',log)
    rows=observe_frames(frames,width=width,height=height,sample_hz=5,
         config=ObserverConfig(c['pixel_delta'],c['min_support_fraction'],c['max_support_fraction']))
    # The byte stream persisted here is exactly the tuple passed to observe.
    with (out/'decoded_gray.raw').open('xb') as f:
        for frame in frames:f.write(frame)
    write(out/'decode_metadata.json',dict(width=width,height=height,frames=len(frames),metadata=metadata,
         output_axis=[i/5 for i in range(len(frames))],source_index_not_assumed=True))
    records=[dataclasses.asdict(r) for r in rows];write(out/'observations.json',records)
    return records


def run(config_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(config_path).read_text());write(output/'config.json',config)
    write(output/'arm_status.json',{a:dict(status='NOT_EXECUTED') for a in ARMS})
    counts=dict(transformer_calls=0,vae_decodes=0,mp4=0);started=time.monotonic();statuses={a:dict(status='NOT_EXECUTED') for a in ARMS};log=[]
    def progress(stage):
        if time.monotonic()-started>config['engineering_budgets']['gpu_wall_seconds']:raise TimeoutError('fixed GPU wall budget')
        print(json.dumps(dict(stage=stage,elapsed=time.monotonic()-started,counts=counts)),flush=True)
    def call():counts['transformer_calls']+=1
    try:
        import torch,numpy as np,diffusers
        from diffusers import WanPipeline
        if diffusers.__version__!=config['model']['diffusers']:raise RuntimeError('requires diffusers0.35.2')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required; CPU run must not load model')
        runtime=dict(torch=torch.__version__,diffusers=diffusers.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,python=sys.version,dependencies={name:importlib.metadata.version(name) for name in ('transformers','accelerate','ftfy','sentencepiece','safetensors','huggingface_hub','numpy','Pillow')})
        write(output/'runtime.json',runtime)
        torch.cuda.reset_peak_memory_stats();progress('model_loading')
        pipe=WanPipeline.from_pretrained(config['model']['id'],revision=config['model']['revision'],torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload();device=pipe._execution_device;g=config['generation'];inter=config['intervention']
        if type(pipe.scheduler).__name__!='UniPCMultistepScheduler':raise RuntimeError('unsupported scheduler')
        with torch.inference_mode():
            cond,uncond=pipe.encode_prompt(prompt=g['prompt'],negative_prompt=g['negative_prompt'],do_classifier_free_guidance=True,num_videos_per_prompt=1,max_sequence_length=g['max_sequence_length'],device=device)
            cond=cond.to(pipe.transformer.dtype);uncond=uncond.to(pipe.transformer.dtype)
            pipe.scheduler.set_timesteps(g['steps'],device=device)
            if hasattr(pipe.scheduler,'set_begin_index'):pipe.scheduler.set_begin_index(0)
            ts=pipe.scheduler.timesteps
            if len(ts)!=8:raise RuntimeError('step count')
            z=pipe.prepare_latents(1,int(pipe.transformer.config.in_channels),g['height'],g['width'],g['frames'],torch.float32,device,torch.Generator(device=device).manual_seed(g['seed']),None)
            if tuple(z.shape)!=(1,16,13,40,64):raise RuntimeError('latent geometry')
            for i in range(6):
                progress('prefix_'+str(i));cv=velocity(pipe,z,ts[i],cond,'cond');call();uv=velocity(pipe,z,ts[i],uncond,'uncond');call()
                z=pipe.scheduler.step(uv+g['guidance']*(cv-uv),ts[i],z,return_dict=False)[0].detach()
            prefix=z.clone();snapshot=copy.deepcopy(pipe.scheduler)
            torch.save(prefix.cpu(),output/'prefix_latent.pt')
            write(output/'prefix_state.json',dict(step_index=snapshot._step_index,timesteps=ts.cpu().tolist(),scheduler_config=dict(snapshot.config),history_policy='full independent copy; history not shifted or reset'))
            off_finals={};off_arrays={};off_mp4={};all_rows={};quality={};saved_quality={};diagnostics={}
            for arm in ARMS:
                path=output/arm;path.mkdir();progress(arm+'_begin')
                try:
                    dx,dy=inter['arms'][arm];injected=translate_latent(prefix,dx,dy,torch)
                    torch.save(injected.cpu(),path/'injected_latent.pt')
                    rms0=float(prefix.float().square().mean().sqrt())
                    changed=float((injected-prefix).float().square().mean().sqrt())
                    diag=dict(command_latent_pixel=[dx,dy],input_rms=rms0,effective_change_rms=changed,relative_change_rms=changed/rms0 if rms0 else None)
                    final=continue_two(pipe,injected,snapshot,ts,cond,uncond,g['guidance'],call)
                    torch.save(final.cpu(),path/'final_latent.pt')
                    # External continuation diagnostic; never used to modify any arm.
                    diag['continuation_prediction']={}
                    for off,base in off_finals.items():
                        prediction=translate_latent(base.to(device),dx,dy,torch)
                        residual=(final-prediction).float();pred_delta=(prediction-base.to(device)).float();den=float(pred_delta.square().sum())
                        diag['continuation_prediction'][off]=dict(rmse=float(residual.square().mean().sqrt()),projection_gain=float(((final-base.to(device)).float()*pred_delta).sum())/den if den else None)
                    if arm.startswith('OFF'):off_finals[arm]=final.cpu().clone()
                    array=decode_rgb(pipe,final,torch);counts['vae_decodes']+=1
                    np.save(path/'preencode_rgb.npy',array,allow_pickle=False)
                    quality[arm]={off:float(np.sqrt(np.mean(((array.astype(np.float64)-base)/255)**2))) for off,base in off_arrays.items()}
                    if arm.startswith('OFF'):off_arrays[arm]=array.copy()
                    encode(array,path/'preencode_lossless.mkv',True,log)
                    encode(array,path/'saved.mp4',False,log);counts['mp4']+=1
                    cmd=['ffmpeg','-v','error','-threads','1','-i',str(path/'saved.mp4'),'-map','0:v:0','-pix_fmt','rgb24','-f','rawvideo','-']
                    proc=subprocess.run(cmd,capture_output=True,timeout=120)
                    log.append(dict(command=cmd,exit_code=proc.returncode))
                    if proc.returncode or len(proc.stdout)!=array.size: raise RuntimeError('saved RGB decoding mismatch')
                    saved=np.frombuffer(proc.stdout,dtype=np.uint8).reshape(array.shape)
                    saved_quality[arm]={off:float(np.sqrt(np.mean(((saved.astype(np.float64)-base)/255)**2))) for off,base in off_mp4.items()}
                    if arm.startswith('OFF'):off_mp4[arm]=saved.copy()
                    del saved,proc
                    del array,final,injected
                    pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
                    streams={}
                    for label,video in [('preencode','preencode_lossless.mkv'),('mp4','saved.mp4')]:
                        dest=path/label;dest.mkdir();streams[label]=read_public(path/video,dest,config)
                    all_rows[arm]=streams;diagnostics[arm]=diag;write(path/'latent_diagnostic.json',diag)
                    statuses[arm]=dict(status='COMPLETE')
                except Exception as exc:
                    statuses[arm]=dict(status='OPERATIONAL_BLOCKED',error=repr(exc));write(path/'failure.json',statuses[arm])
                    write(output/'arm_status.json',statuses)
                    if isinstance(exc,(torch.cuda.OutOfMemoryError,TimeoutError)):raise
                    # Retain missing arm; continue remaining fixed meaningful arms.
                    pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
                write(output/'arm_status.json',statuses);write(output/'encode_processes.json',log)
            b=config['engineering_budgets'];responses={}
            for stream in ('preencode','mp4'):
                responses[stream]=evaluate_response({a:all_rows[a][stream] for a in ARMS},inter['delta_latent_pixel'],b) if len(all_rows)==6 else dict(status='OPERATIONAL_BLOCKED',reason='ARM_MISSING')
            quality_ok=all(a in table and all(table[a].get(o,float('inf'))<=b['maximum_rgb_absolute_rmse'] for o in ('OFF1','OFF2')) for table in (quality,saved_quality) for a in ARMS[2:])
            primary=responses['mp4']['status']
            complete_counts=counts==dict(transformer_calls=36,vae_decodes=6,mp4=6)
            if not complete_counts:primary='OPERATIONAL_BLOCKED'
            status='OPERATIONAL_BLOCKED' if primary=='OPERATIONAL_BLOCKED' else ('ENGINEERING_PRIMITIVE_READY' if primary=='ENGINEERING_RESPONSE_READY' and quality_ok else 'ENGINEERING_BUDGET_NOT_MET')
            result=dict(status=status,responses=responses,preencode_rgb_absolute_rmse=quality,saved_rgb_absolute_rmse=saved_quality,quality_budget_met=quality_ok,arms=statuses,counts=counts,complete_counts=complete_counts,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,
                        science_denominator=0,claim_ceiling='common relocation only; not temporal AISB writing or watermark',gpu_executed=True)
            write(output/'result.json',result);print(json.dumps(result),flush=True)
            return result
    except Exception as exc:
        write(output/'failure.json',dict(error=repr(exc),counts=counts,elapsed_seconds=time.monotonic()-started,arms=statuses))
        raise
