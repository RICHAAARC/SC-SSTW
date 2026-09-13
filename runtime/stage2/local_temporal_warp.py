"""Six fixed local-warp branches. GPU use is a separately approved run."""
import copy,gc,json,time,types,sys,importlib.metadata
from pathlib import Path
from main.sc_sstw.local_temporal_warp import ARMS,warp_latent
from .wan_translation import write,velocity,decode_rgb,encode,read_public
from .off_baseline import attach_forward_counter,load_fp32_vae,tensor_comparison,describe_rgb,save_preview

def continue_five(pipe,z,snapshot,ts,cond,uncond,guidance):
    if snapshot._step_index!=45 or len(ts)!=50:raise ValueError('expected after zero-based44')
    scheduler=copy.deepcopy(snapshot)
    for i in range(45,50):
        if scheduler._step_index!=i:raise RuntimeError('scheduler drift')
        cv=velocity(pipe,z,ts[i],cond,'cond');uv=velocity(pipe,z,ts[i],uncond,'uncond')
        z=scheduler.step(uv+guidance*(cv-uv),ts[i],z,return_dict=False)[0].detach()
    if snapshot._step_index!=45 or scheduler._step_index!=50:raise RuntimeError('history changed')
    return z

def reference_package(dest):
    """Only actual public-reader gray bytes. No command/grid/q used to locate p."""
    import numpy as np
    from PIL import Image,ImageDraw
    meta=json.loads((dest/'decode_metadata.json').read_text());h=meta['height'];w=meta['width']
    frames=np.frombuffer((dest/'decoded_gray.raw').read_bytes(),dtype=np.uint8).reshape(-1,h,w)
    rows=[]
    for i,frame in enumerate(frames):
        image=Image.fromarray(frame).convert('RGB');draw=ImageDraw.Draw(image);candidates={}
        for name,band,lines,horizontal in [('left',(50,150),range(130,251,20),True),('right',(320,440),range(130,251,20),True),('top',(50,140),range(160,301,20),False),('bottom',(235,310),range(160,301,20),False)]:
            records=[]
            for line in lines:
                signal=(frame[line] if horizontal else frame[:,line]).astype(float)
                grad=np.abs(np.diff(signal));a,b=band
                peaks=[x for x in range(a+1,b-1) if grad[x]>=grad[x-1] and grad[x]>grad[x+1] and grad[x]>0]
                records.append(dict(scanline=line,candidates=[dict(position=x+.5,strength=float(grad[x])) for x in peaks]))
                if horizontal:draw.line((a,line,b,line),fill=(0,180,0))
                else:draw.line((line,a,line,b),fill=(0,180,0))
            candidates[name]=records
        image.save(dest/f'reference_overlay_{i:03d}.png')
        rows.append(dict(sample_index=i,time_seconds=i/5,p_pixel=None,subjective_radius_px=None,status='PENDING_INDEPENDENT_HUMAN_OUTLINE_REVIEW',alignment_confirmed=True,candidates=candidates))
    write(dest/'reference_candidates.json',rows)
    write(dest/'reference_annotation_template.json',[{k:v for k,v in r.items() if k!='candidates'} for r in rows])

def run(config_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);c=json.loads(Path(config_path).read_text());g=c['generation'];write(out/'config.json',c)
    states={a:dict(status='NOT_EXECUTED') for a in ARMS};counts=dict(transformer_calls=0,vae_decodes=0,mp4=0)
    trace=[];enc=[];handle=None;active='PREFIX';start=time.monotonic()
    def persist():
        write(out/'output_status.json',states);write(out/'actual_transformer_calls.json',trace);write(out/'counts.json',counts);write(out/'encode_processes.json',enc)
    persist()
    fixed=c['intervention']
    if fixed!={'after_zero_based_step':44,'amplitude_latent_pixels':1.,'curve':'(k-6)/6; k=0..12','mask_x':[5,8,51,54],'mask_y':[6,9,36,39],'interpolation':'bilinear','padding':'border','align_corners':True,'scheduler_history':'deepcopy unchanged, not warped or reset'}:raise ValueError('fixed intervention mismatch')
    try:
        import torch,numpy as np,diffusers
        from diffusers import WanPipeline,AutoencoderKLWan
        if diffusers.__version__!='0.35.2' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('fixed diffusers/CUDA BF16 required')
        if g['steps']!=50 or c['intervention']['after_zero_based_step']!=44 or c['intervention']['amplitude_latent_pixels']!=1:raise ValueError('frozen method mismatch')
        write(out/'runtime.json',dict(torch=torch.__version__,diffusers=diffusers.__version__,python=sys.version,gpu=torch.cuda.get_device_name(),packages={x:importlib.metadata.version(x) for x in ('transformers','accelerate','numpy','Pillow')}))
        torch.cuda.reset_peak_memory_stats()
        pipe=WanPipeline.from_pretrained(c['model']['id'],revision=c['model']['revision'],torch_dtype=torch.bfloat16);pipe.enable_model_cpu_offload();device=pipe._execution_device
        if type(pipe.scheduler).__name__!='UniPCMultistepScheduler':raise RuntimeError('UniPC required')
        handle=attach_forward_counter(pipe.transformer,counts,trace,lambda:active)
        with torch.inference_mode():
            initial=pipe.prepare_latents(1,int(pipe.transformer.config.in_channels),g['height'],g['width'],g['frames'],torch.float32,device,torch.Generator(device=device).manual_seed(g['seed']),None).detach()
            torch.save(initial.cpu(),out/'common_initial_latent.pt')
            cond,uncond=pipe.encode_prompt(prompt=g['prompt'],negative_prompt=g['negative_prompt'],do_classifier_free_guidance=True,num_videos_per_prompt=1,max_sequence_length=g['max_sequence_length'],device=device)
            cond=cond.to(pipe.transformer.dtype);uncond=uncond.to(pipe.transformer.dtype)
            pipe.scheduler.set_timesteps(50,device=device)
            if hasattr(pipe.scheduler,'set_begin_index'):pipe.scheduler.set_begin_index(0)
            ts=pipe.scheduler.timesteps;z=initial.clone()
            print('PREFIX_begin',flush=True);persist()
            for i in range(45):
                cv=velocity(pipe,z,ts[i],cond,'cond');uv=velocity(pipe,z,ts[i],uncond,'uncond')
                z=pipe.scheduler.step(uv+g['guidance']*(cv-uv),ts[i],z,return_dict=False)[0].detach()
                persist()
            if counts['transformer_calls']!=90:raise RuntimeError('prefix count')
            snapshot=copy.deepcopy(pipe.scheduler);torch.save(z.cpu(),out/'prefix_latent.pt')
            write(out/'prefix_state.json',dict(step_index=snapshot._step_index,timesteps=ts.cpu().tolist(),config=dict(snapshot.config),history_unmodified=True))
            vae=None;off_arrays={};finals={}
            for arm in ARMS:
                active=arm;path=out/arm;path.mkdir();before=counts['transformer_calls'];states[arm]=dict(status='RUNNING');persist();print(arm+'_begin',flush=True)
                try:
                    injected,grid,mask=warp_latent(z,arm,torch)
                    torch.save(dict(grid=grid.cpu(),mask=mask.cpu()),path/'actual_sampling_map.pt');torch.save(injected.cpu(),path/'injected_latent.pt')
                    write(path/'injection_difference.json',tensor_comparison(injected,z))
                    final=continue_five(pipe,injected,snapshot,ts,cond,uncond,g['guidance'])
                    torch.save(final.cpu(),path/'final_latent.pt');finals[arm]=final.cpu()
                    if counts['transformer_calls']-before!=10:raise RuntimeError('branch count')
                    pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
                    if vae is None:
                        vae=load_fp32_vae(AutoencoderKLWan,c['model'],torch)
                        write(out/'vae_provenance.json',dict(original_checkpoint=c['model'],dtype=str(vae.dtype),not_bf16_upcast=True))
                    vae.to(device);counts['vae_decodes']+=1;array=decode_rgb(types.SimpleNamespace(vae=vae),final,torch);vae.to('cpu')
                    np.save(path/'preencode_rgb.npy',array,allow_pickle=False);write(path/'rgb_metrics.json',describe_rgb(array,np));save_preview(array,path/'first_middle_last.png')
                    if arm.startswith('OFF'):off_arrays[arm]=array
                    quality={k:float(np.sqrt(np.mean(((array.astype(float)-v)/255)**2))) for k,v in off_arrays.items()}
                    write(path/'unregistered_rgb_rmse_vs_off.json',dict(values=quality,descriptive_only=True,not_perceptual_quality=True))
                    encode(array,path/'preencode_lossless.mkv',True,enc);encode(array,path/'saved.mp4',False,enc);counts['mp4']+=1
                    for label,filename in [('preencode','preencode_lossless.mkv'),('mp4','saved.mp4')]:
                        dest=path/label;dest.mkdir();read_public(path/filename,dest,c);reference_package(dest)
                    states[arm]=dict(status='COMPLETE_PENDING_EXTERNAL_REVIEW',transformer_calls=10,reference='p null until independent review')
                    del final,injected;gc.collect();torch.cuda.empty_cache()
                except Exception as exc:
                    states[arm]=dict(status='OPERATIONAL_BLOCKED',error=repr(exc),transformer_calls=counts['transformer_calls']-before);write(path/'failure.json',states[arm])
                    if isinstance(exc,(torch.cuda.OutOfMemoryError,TimeoutError)):raise
                finally:
                    persist()
                    if vae is not None:vae.to('cpu')
                    pipe.maybe_free_model_hooks();gc.collect();torch.cuda.empty_cache()
            if 'OFF1' in finals and 'OFF2' in finals:write(out/'off_repeat_latent.json',tensor_comparison(finals['OFF1'],finals['OFF2']))
            complete=all(s['status']=='COMPLETE_PENDING_EXTERNAL_REVIEW' for s in states.values()) and counts==dict(transformer_calls=150,vae_decodes=6,mp4=6)
            result=dict(status='EXECUTED_PENDING_EXTERNAL_REVIEW' if complete else 'OPERATIONAL_BLOCKED',counts=counts,outputs=states,elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),science_denominator=0,no_automatic_subject_motion_pass=True)
            write(out/'result.json',result);return result
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),counts=counts,outputs=states));raise
    finally:
        persist()
        if handle is not None:handle.remove()
