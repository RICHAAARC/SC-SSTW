"""One real-channel update around a saved terminal, executed only by the user."""
from __future__ import annotations
import argparse,gzip,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from main.tube_state import video_overlap_zero_mean_state as state,video_overlap_zero_mean_control as payload
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import zero_mean_channel_margin as gradient,zero_mean_c1_layered_transport as media
from runtime.wan import zero_mean_c1_yuv444_no_h264 as color
from runtime.wan import vae as shared
from experiments.wan_state_clock.zero_mean_c1_yuv444_no_h264_run import dump,dump_gzip,save_tensor,environment
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_margin_v1.json'
OBS=('BEFORE','AFTER')

def posthoc(inf):
    costs=np.array(inf['path_costs']);ids=inf['valid_catalog_indices'];true=costs[ids.index(0)]
    wrong=np.delete(costs,ids.index(0));top=inf['summary']['top_catalog_indices']
    return dict(status='EVALUATED',true_rank=1+int(np.sum(costs<true-state.PUBLIC.tie_atol)),delta=float(wrong.min()-true),
                true_cost=float(true),top=top,truth_unique_top=top==[0],
                canonical_catalog_index=inf['summary']['canonical_catalog_index'],fixed_phase=0,R=44)

class Run:
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        self.data=dict(status='RUNNING',stage='INITIAL',config=cfg,environment=environment(),
          source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          calls={k:dict(attempted=0,completed=0) for k in cfg['planned_calls']},fixed_denominator=cfg['fixed_denominator'],failures=[],references={},
          observations={o:dict(status='PENDING') for o in OBS},
          reads={o+'/'+k:dict(status='PENDING',scored=0) for o in OBS for k in cfg['keys']},
          posthoc={o+'/'+k:dict(status='PENDING') for o in OBS for k in cfg['keys']},
          message_evaluations={o+'/'+k+'/'+t:dict(status='PENDING') for o in OBS for k in cfg['keys'] for t in cfg['messages']},
          resources={},update=dict(status='PENDING'),
          score_status='UNCALIBRATED_DIAGNOSTIC',evidence_ceiling=cfg['evidence_ceiling'])
        files=[str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),
          'main/tube_state/zero_mean_channel_margin.py','runtime/wan/zero_mean_channel_margin.py',
          'main/tube_state/video_overlap_zero_mean_state.py','main/tube_state/video_overlap_zero_mean_control.py',
          'runtime/wan/zero_mean_c1_yuv444_no_h264.py','runtime/wan/gradient_checkpointing.py']
        self.data['source_files']={p:media.file_sha256(ROOT/p) for p in files};self.save()
    def save(self):dump(self.output/'result.json',self.data)
    def count(self,kind,completed):
        row=self.data['calls'].setdefault(kind,dict(attempted=0,completed=0));row['completed' if completed else 'attempted']+=1
        self.save()
    def stage(self,name):self.data['stage']=name;self.save();print(name,flush=True)
    def call(self,kind,fn):
        self.count(kind,False);v=fn();self.count(kind,True);return v
    def saved(self,name,value):return save_tensor(self.output/name,value)
    def read(self,obs,z):
        # Persist raw blind outputs before any writer-truth join.
        received=z[0,:,1:45].float().numpy().transpose(1,0,2,3)
        for kid,key in self.cfg['keys'].items():
            sid=obs+'/'+kid
            inf=self.call('path_read',lambda:state.infer(received,key,np.ones((44,4),bool)))
            pay=self.call('payload_read',lambda:payload.payload_read(z,key,44))
            raw=dict(inference=inf,payload=pay,truth_used=False,writer_inputs=False,R=44,fixed_phase=0)
            path=self.output/obs/(kid+'.raw.json.gz');dump_gzip(path,raw)
            self.data['reads'][sid]=dict(status='SAVED',path=str(path),sha256=media.file_sha256(path),scored=inf['counts']['scored'])
            self.save()
            self.data['posthoc'][sid]=posthoc(inf)
            for label,message in self.cfg['messages'].items():
                bits=payload.message_bits(message);errors=sum(int(a)!=int(b) for a,b in zip(bits,pay['decoded_bits']))
                self.data['message_evaluations'][sid+'/'+label]=dict(status='EVALUATED',errors=errors,BER=errors/32,exact=errors==0)
            self.save()
    def roundtrip(self,obs,q8):
        self.saved(obs+'/rgb8.pt',q8)
        def event(stage,row):
            self.data['observations'][obs].setdefault('conversion',{})[stage]=row;self.save()
        return color.roundtrip(q8,self.output/obs/'roundtrip.yuv',self.output/obs/'roundtrip.rgb',count=self.count,event=event).float()/255.


def inputs(run,input_root,reference_root):
    import torch
    cfg=run.cfg
    if media.file_sha256(input_root/'result.json')!=cfg['input']['result_sha256']:raise ValueError('writer source result mismatch')
    terminal=input_root/cfg['input']['terminal_relative_path']
    if media.file_sha256(terminal)!=cfg['input']['terminal_sha256']:raise ValueError('full writer terminal mismatch')
    z=torch.load(terminal,map_location='cpu',weights_only=True)
    if tuple(z.shape)!=(1,16,46,40,64) or z.dtype!=torch.float32 or not bool(z.isfinite().all()):raise ValueError('full terminal invalid')
    run.data['writer_input']=dict(path=str(terminal),sha256=media.file_sha256(terminal),shape=list(z.shape),dtype=str(z.dtype))
    if media.file_sha256(reference_root/'result.json')!=cfg['reference']['result_sha256']:raise ValueError('reference result mismatch')
    ref=json.loads((reference_root/'result.json').read_text())
    for arm in ('OFF','PAYLOAD_MULTI','OVERLAP_MULTI'):
        n=ref['normalized'][arm];path=reference_root/arm/'normalized.pt'
        if media.file_sha256(path)!=n['sha256']:raise ValueError('reference normalized mismatch')
        run.data['references'][arm]=dict(normalized=dict(path=str(path),sha256=n['sha256']),reads={})
        for kid in cfg['keys']:
            sid=arm+'/'+kid;raw=reference_root/arm/(kid+'.raw.json.gz')
            if media.file_sha256(raw)!=ref['reads'][sid]['sha256']:raise ValueError('reference raw mismatch')
            run.data['references'][arm]['reads'][kid]=dict(path=str(raw),sha256=ref['reads'][sid]['sha256'],posthoc=ref['path_posthoc'][sid],
                messages={t:ref['message_evaluations'][sid+'/'+t] for t in cfg['messages']})
    run.save();return z,ref


def execute(run,input_root,reference_root):
    import torch,gc
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');terminal,ref=inputs(run,input_root,reference_root)
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('reference full-frame VAE requires native untiled path')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),
       latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    run.stage('BEFORE_REAL_CHANNEL')
    rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,terminal))
    q8=shared.quantize_rgb8_no_codec(rgb);before=run.roundtrip('BEFORE',q8);del q8
    run.data['observations']['BEFORE']['clamp']=clamp
    run.stage('ENCODER_VJP');enc_resource={}
    try:z,cotangent,loss=gradient.encoder_cotangent(vae,before,run.cfg['key'],run.count,enc_resource)
    finally:run.data['resources']['encoder']=enc_resource;run.save()
    run.data['observations']['BEFORE'].update(status='SAVED',normalized=run.saved('BEFORE/normalized.pt',z),writer_loss=loss)
    previous=torch.load(reference_root/'OVERLAP_MULTI/normalized.pt',map_location='cpu',weights_only=True)
    run.data['observations']['BEFORE']['historical_normalized_max_error']=float((z-previous).abs().max());del previous
    run.read('BEFORE',z);del z
    run.stage('DECODER_VJP');dec_resource={}
    try:g=gradient.decoder_vjp(vae,terminal,cotangent,rgb,run.count,dec_resource)
    finally:run.data['resources']['decoder']=dec_resource;run.save()
    del cotangent,rgb;gc.collect()
    run.data['update']['gradient']=run.saved('terminal_gradient.pt',g)
    step,receipt=method.constrained_step(g,run.cfg['key'],run.cfg['update']['eta'],run.cfg['update']['cap_l2']);del g
    changed=terminal+step;actual=changed-terminal
    run.data['update'].update(status='SAVED',step=run.saved('step.pt',step),terminal=run.saved('AFTER/terminal.pt',changed),
       receipt=receipt,actual_terminal_change_l2=float(actual.double().norm()),
       step_rounding_max_error=float((actual-step).abs().max()),
       semantics='terminal-only local controllability; not a generated trajectory watermark')
    run.save();del terminal,step,actual
    run.stage('AFTER_REAL_CHANNEL')
    after_rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,changed));del changed
    after=run.roundtrip('AFTER',shared.quantize_rgb8_no_codec(after_rgb));del after_rgb
    z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,after))
    _,loss=method.margin_loss(method.project(z,run.cfg['key']),run.cfg['key'])
    diff=(after-before).double();rmse=float(diff.square().mean().sqrt())
    run.data['observations']['AFTER'].update(status='SAVED',normalized=run.saved('AFTER/normalized.pt',z),writer_loss=loss,clamp=clamp,
       quality=dict(rgb_rmse_vs_before=rmse,psnr_db=None if rmse==0 else float(-20*np.log10(rmse)),diagnostic_only=True,threshold=None))
    run.read('AFTER',z)
    pre=run.data['posthoc']['BEFORE/CORRECT'];post=run.data['posthoc']['AFTER/CORRECT']
    run.data['comparison']=dict(delta_before=pre['delta'],delta_after=post['delta'],delta_change=post['delta']-pre['delta'],
       rank_before=pre['true_rank'],rank_after=post['true_rank'],real_delta_increased=post['delta']>pre['delta'],
       all_competitors_defeated=post['truth_unique_top'],scientific_pass=False)
    run.data['status']='EXECUTION_COMPLETE';run.stage('COMPLETE')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-root',type=Path);parser.add_argument('--reference-root',type=Path)
    args=parser.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(args.output,cfg)
    try:execute(run,args.input_root or Path(cfg['input']['root']),args.reference_root or Path(cfg['reference']['root']))
    except Exception as exc:
        run.data['status']='EXECUTION_FAILED';run.data['failures'].append(dict(stage=run.data['stage'],type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        for row in run.data['observations'].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
        for row in run.data['reads'].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
        run.save();raise
    finally:
        run.data['counts']=dict(observations=sum(r['status']=='SAVED' for r in run.data['observations'].values()),
          path_reads=sum(r['status']=='SAVED' for r in run.data['reads'].values()),valid_costs=sum(r['scored'] for r in run.data['reads'].values()))
        run.save()
    print(json.dumps(run.data.get('comparison'),indent=2));print('Result:',run.output/'result.json')
if __name__=='__main__':main()
