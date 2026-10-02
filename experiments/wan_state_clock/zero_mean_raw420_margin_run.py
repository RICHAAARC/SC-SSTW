"""One actual-raw420-objective step from the saved step7 terminal."""
from __future__ import annotations
import argparse,json,traceback,gc
from pathlib import Path
import numpy as np
from experiments.wan_state_clock import zero_mean_channel_margin_run as base
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import zero_mean_channel_margin as gradient,zero_mean_c1_layered_transport as media
from runtime.wan import zero_mean_c1_yuv420_no_h264 as raw420,zero_mean_c1_yuv444_no_h264 as raw444
from runtime.wan import vae as shared
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_raw420_margin_v1.json'

class Run(base.Run):
    def __init__(self,output,cfg):
        super().__init__(output,cfg)
        self.data['observations']={o:dict(status='PENDING') for o in cfg['observations']}
        self.data['reads']={o+'/'+k:dict(status='PENDING',scored=0) for o in cfg['observations'] for k in cfg['keys']}
        self.data['posthoc']={s:dict(status='PENDING') for s in self.data['reads']}
        self.data['message_evaluations']={s+'/'+t:dict(status='PENDING') for s in self.data['reads'] for t in cfg['messages']}
        files=['experiments/wan_state_clock/zero_mean_raw420_margin_run.py',str(CONFIG.relative_to(ROOT)),
               'runtime/wan/zero_mean_c1_yuv420_no_h264.py','runtime/wan/zero_mean_c1_layered_transport.py','runtime/wan/vae.py']
        self.data['source_files'].update({p:media.file_sha256(ROOT/p) for p in files})
        self.data['comparisons']={};self.save()
    def transport(self,obs,q8):
        adapter=raw420 if obs.endswith('420') else raw444
        self.saved(obs+'/rgb8.pt',q8)
        def event(stage,row):
            self.data['observations'][obs].setdefault('conversion',{})[stage]=row;self.save()
        return adapter.roundtrip(q8,self.output/obs/'roundtrip.yuv',self.output/obs/'roundtrip.rgb',count=self.count,event=event).float()/255.
    def record(self,obs,z,clamp):
        _,loss=method.margin_loss(method.project(z,self.cfg['key']),self.cfg['key'])
        self.data['observations'][obs].update(status='SAVED',normalized=self.saved(obs+'/normalized.pt',z),writer_loss=loss,clamp=clamp)
        self.save();self.read(obs,z)

def quality(a,b):
    rmse=float((a.double()-b.double()).square().mean().sqrt())
    return dict(rgb_rmse=rmse,psnr_db=None if rmse==0 else float(-20*np.log10(rmse)),diagnostic_only=True,threshold=None)

def execute(run,input_root,reference_root):
    import torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');terminal,_=base.inputs(run,input_root,reference_root)
    run.data['baseline_semantics']='Fresh same-run terminal decode for both channels; not the old byte-frozen raster. Historical controls are references only.'
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('reference full-frame VAE requires native untiled path')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),
        latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    run.stage('BEFORE_RAW420')
    rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,terminal))
    q8=shared.quantize_rgb8_no_codec(rgb)
    before420=run.transport('BEFORE_RAW420',q8)
    run.stage('ENCODER_VJP_RAW420');resource={}
    try:z,cotangent,loss=gradient.encoder_cotangent(vae,before420,run.cfg['key'],run.count,resource)
    finally:run.data['resources']['encoder']=resource;run.save()
    run.record('BEFORE_RAW420',z,clamp);del z
    run.stage('BEFORE_RAW444')
    before444=run.transport('BEFORE_RAW444',q8);del q8
    z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,before444))
    run.record('BEFORE_RAW444',z,clamp);del z
    run.stage('DECODER_VJP_IDENTITY_STE');resource={}
    try:g=gradient.decoder_vjp(vae,terminal,cotangent,rgb,run.count,resource)
    finally:run.data['resources']['decoder']=resource;run.save()
    del cotangent,rgb;gc.collect()
    run.data['update']['gradient']=run.saved('terminal_gradient.pt',g)
    step,receipt=method.constrained_step(g,run.cfg['key'],run.cfg['update']['eta'],run.cfg['update']['cap_l2']);del g
    changed=terminal+step;actual=changed-terminal
    run.data['update'].update(status='SAVED',step=run.saved('step.pt',step),terminal=run.saved('AFTER/terminal.pt',changed),receipt=receipt,
        actual_terminal_change_l2=float(actual.double().norm()),step_rounding_max_error=float((actual-step).abs().max()),
        objective_channel='actual materialized raw420',channel_backward=run.cfg['update']['channel_backward'],
        semantics='one terminal-only local controllability step from step7; no generation trajectory claim')
    run.save();del terminal,step,actual
    run.stage('AFTER_DECODE')
    after_rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,changed));del changed
    q8=shared.quantize_rgb8_no_codec(after_rgb);del after_rgb
    # Both post-update observations remain in the denominator even if one fails.
    for channel,before in [('RAW420',before420),('RAW444',before444)]:
        obs='AFTER_'+channel
        try:
            run.stage(obs);after=run.transport(obs,q8)
            z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,after))
            run.data['observations'][obs]['quality_vs_same_channel_before']=quality(after,before)
            run.record(obs,z,clamp);del z,after
            pre=run.data['posthoc']['BEFORE_'+channel+'/CORRECT'];post=run.data['posthoc'][obs+'/CORRECT']
            run.data['comparisons'][channel]=dict(delta_before=pre['delta'],delta_after=post['delta'],delta_change=post['delta']-pre['delta'],
                rank_before=pre['true_rank'],rank_after=post['true_rank'],real_delta_increased=post['delta']>pre['delta'],
                all_competitors_defeated=post['truth_unique_top'],scientific_pass=False)
            run.save()
        except Exception as exc:
            run.data['failures'].append(dict(stage=obs,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            if run.data['observations'][obs]['status']=='PENDING':run.data['observations'][obs]['status']='FAILED'
            run.save()
    run.data['status']='EXECUTION_PARTIAL' if run.data['failures'] else 'EXECUTION_COMPLETE';run.stage('COMPLETE')

def finalize(run):
    for group in ('observations','reads','posthoc','message_evaluations'):
        for row in run.data[group].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
    if run.data['update']['status']=='PENDING':run.data['update']['status']='NOT_COMPLETED'
    run.data['counts']=dict(observations=sum(r['status']=='SAVED' for r in run.data['observations'].values()),
        updates=int(run.data['update']['status']=='SAVED'),path_reads=sum(r['status']=='SAVED' for r in run.data['reads'].values()),
        payload_reads=run.data['calls']['payload_read']['completed'],message_evaluations=sum(r['status']=='EVALUATED' for r in run.data['message_evaluations'].values()),
        valid_costs=sum(r['scored'] for r in run.data['reads'].values()))
    run.save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--input-root',type=Path);p.add_argument('--reference-root',type=Path)
    args=p.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(args.output,cfg)
    try:execute(run,args.input_root or Path(cfg['input']['root']),args.reference_root or Path(cfg['reference']['root']))
    except Exception as exc:
        run.data['status']='EXECUTION_FAILED';run.data['failures'].append(dict(stage=run.data['stage'],type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));raise
    finally:finalize(run)
    print(json.dumps(run.data['comparisons'],indent=2));print('Result:',run.output/'result.json')
if __name__=='__main__':main()
