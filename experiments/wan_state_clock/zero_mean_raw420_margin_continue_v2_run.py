"""Fixed three-step raw420 continuation; every point is retained."""
from __future__ import annotations
import argparse,json,traceback,gc,gzip
from pathlib import Path
from experiments.wan_state_clock import zero_mean_raw420_margin_run as single
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import zero_mean_channel_margin as gradient,zero_mean_c1_layered_transport as media
from runtime.wan import vae as shared
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_raw420_margin_continue_v2.json'

class Run(single.Run):
    def __init__(self,output,cfg):
        super().__init__(output,cfg)
        del self.data['update']
        self.data['updates']={f'POINT{i}_TO_POINT{i+1}':dict(status='PENDING') for i in range(3)}
        self.data['fixed_endpoint']=dict(point='POINT3',status='PENDING',selected_by_score=False)
        self.data['baseline_replay']={};self.data['parent_references']={};self.data['boundary_diagnostics']={}
        self.data['prior_raw420_updates']=cfg['prior_raw420_updates']
        for f in (str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))):self.data['source_files'][f]=media.file_sha256(ROOT/f)
        self.save()
    def failure(self,stage,exc):
        self.data['failures'].append(dict(stage=stage,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));self.save()


def inputs(run,input_root,reference_root):
    import torch
    terminal,_=single.base.inputs(run,input_root,reference_root)
    parent=json.loads((input_root/'result.json').read_text())
    for field in ('eta','cap_l2','support','loss','channel_backward'):
        if parent['config']['update'][field]!=run.cfg['update'][field]:raise ValueError('parent update semantics mismatch: '+field)
    for channel in ('RAW420','RAW444'):
        obs='POINT3_'+channel;n=parent['observations'][obs]['normalized'];path=input_root/obs/'normalized.pt'
        if media.file_sha256(path)!=n['sha256']:raise ValueError('parent normalized mismatch')
        ref=dict(normalized=dict(path=str(path),sha256=n['sha256']),reads={})
        for key in run.cfg['keys']:
            row=parent['reads'][obs+'/'+key];raw=input_root/obs/(key+'.raw.json.gz')
            if media.file_sha256(raw)!=row['sha256']:raise ValueError('parent raw mismatch')
            ref['reads'][key]=dict(path=str(raw),sha256=row['sha256'],posthoc=parent['posthoc'][obs+'/'+key])
        run.data['parent_references'][channel]=ref
    run.data['prior_raw420_step_l2']=parent['updates']['POINT2_TO_POINT3']['sum_raw420_actual_step_l2']
    run.save();return terminal


def compare(run,point,channel):
    obs=point+'_'+channel;row=run.data['posthoc'][obs+'/CORRECT']
    if row['status']!='EVALUATED':return
    i=run.cfg['points'].index(point)
    for label,origin in [('baseline','POINT0')]+([('previous',run.cfg['points'][i-1])] if i else []):
        before=run.data['posthoc'][origin+'_'+channel+'/CORRECT']
        if before['status']!='EVALUATED':continue
        run.data['comparisons'][obs+'/'+label]=dict(from_point=origin,to_point=point,rank_before=before['true_rank'],rank_after=row['true_rank'],
            delta_before=before['delta'],delta_after=row['delta'],delta_change=row['delta']-before['delta'],
            all_competitors_defeated=row['truth_unique_top'],scientific_pass=False)
    run.save()



def report_boundaries(run,obs):
    # Reporting-only join after the full blind readout has already been saved.
    row=run.data['reads'][obs+'/CORRECT']
    inf=json.loads(gzip.decompress(Path(row['path']).read_bytes()))['inference']
    ids=inf['valid_catalog_indices'];costs=inf['path_costs'];truth=costs[ids.index(0)]
    run.data['boundary_diagnostics'][obs]={str(cid):dict(delta=costs[ids.index(cid)]-truth,
        interpretation=label) for cid,label in ((86,'tail SKIP at44'),(88,'head REPEAT at2 starting2'))}
    run.save()


def execute(run,input_root,reference_root):
    import torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');terminal=inputs(run,input_root,reference_root);start=terminal.clone()
    run.data['baseline_semantics']='Fresh same-runtime decode of the saved fourth raw420 update (previous fixed POINT3); parent scores are references, not a substituted baseline.'
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('reference full-frame VAE requires native untiled path')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),
        latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    baseline={};previous={};previous_point={};sum_steps=0.
    for i,point in enumerate(run.cfg['points']):
        run.stage(point+'_DECODE')
        rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,terminal))
        q8=shared.quantize_rgb8_no_codec(rgb);cotangent=None
        # The raw420 forward remains the writer's only channel objective.
        for channel in ('RAW420','RAW444'):
            obs=point+'_'+channel;run.stage(obs)
            try:
                received=run.transport(obs,q8)
                if channel=='RAW420' and i<3:
                    resource={}
                    try:z,cotangent,_=gradient.encoder_cotangent(vae,received,run.cfg['key'],run.count,resource)
                    finally:run.data['resources'][point+'_encoder']=resource;run.save()
                else:z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,received))
                if i==0:
                    ref=run.data['parent_references'][channel]
                    old=torch.load(ref['normalized']['path'],map_location='cpu',weights_only=True)
                    d=(z-old).double()
                    run.data['baseline_replay'][channel]=dict(normalized_max_error=float(d.abs().max()),normalized_rmse=float(d.square().mean().sqrt()),
                        parent_posthoc=ref['reads']['CORRECT']['posthoc'])
                    baseline[channel]=received.clone();del old,d
                else:
                    if channel in baseline:run.data['observations'][obs]['quality_vs_baseline']=single.quality(received,baseline[channel])
                    if channel in previous:
                        run.data['observations'][obs]['quality_vs_previous_available']=single.quality(received,previous[channel])
                        run.data['observations'][obs]['quality_previous_point']=previous_point[channel]
                previous[channel]=received;previous_point[channel]=point
                run.record(obs,z,clamp);compare(run,point,channel);report_boundaries(run,obs);del z,received
            except Exception as exc:
                run.failure(obs,exc)
                if run.data['observations'][obs]['status']=='PENDING':run.data['observations'][obs]['status']='FAILED'
                run.save()
                if channel=='RAW420' and i<3:raise
        del q8
        if i==3:
            run.data['fixed_endpoint'].update(status=run.data['observations'][point+'_RAW420']['status'],
                posthoc={ch:run.data['posthoc'][point+'_'+ch+'/CORRECT'] for ch in ('RAW420','RAW444')},
                terminal=run.data['updates']['POINT2_TO_POINT3']['terminal'],start_terminal_displacement_l2=float((terminal-start).double().norm()))
            run.save();del rgb;continue
        name=point+'_TO_'+run.cfg['points'][i+1];run.stage(name+'_DECODER_VJP');resource={}
        try:g=gradient.decoder_vjp(vae,terminal,cotangent,rgb,run.count,resource)
        finally:run.data['resources'][point+'_decoder']=resource;run.save()
        del cotangent,rgb;gc.collect()
        row=run.data['updates'][name];row['gradient']=run.saved(name+'/terminal_gradient.pt',g)
        step,receipt=method.constrained_step(g,run.cfg['key'],run.cfg['update']['eta'],run.cfg['update']['cap_l2']);del g
        changed=terminal+step;actual=changed-terminal;actual_l2=float(actual.double().norm());sum_steps+=actual_l2
        row.update(status='SAVED',step=run.saved(name+'/step.pt',step),terminal=run.saved(run.cfg['points'][i+1]+'/terminal.pt',changed),receipt=receipt,
            actual_terminal_change_l2=actual_l2,step_rounding_max_error=float((actual-step).abs().max()),
            raw420_update_index=run.cfg['prior_raw420_updates']+i+1,
            sum_new_actual_step_l2=sum_steps,sum_raw420_actual_step_l2=run.data['prior_raw420_step_l2']+sum_steps,
            start_terminal_displacement_l2=float((changed-start).double().norm()))
        run.save();terminal=changed;del step,actual
    run.data['status']='EXECUTION_PARTIAL' if run.data['failures'] else 'EXECUTION_COMPLETE';run.stage('COMPLETE')


def finalize(run):
    for group in ('observations','reads','posthoc','message_evaluations','updates'):
        for row in run.data[group].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
    if run.data['fixed_endpoint']['status']=='PENDING':run.data['fixed_endpoint']['status']='NOT_COMPLETED'
    run.data['counts']=dict(observations=sum(r['status']=='SAVED' for r in run.data['observations'].values()),
        updates=sum(r['status']=='SAVED' for r in run.data['updates'].values()),path_reads=sum(r['status']=='SAVED' for r in run.data['reads'].values()),
        payload_reads=run.data['calls']['payload_read']['completed'],message_evaluations=sum(r['status']=='EVALUATED' for r in run.data['message_evaluations'].values()),
        valid_costs=sum(r['scored'] for r in run.data['reads'].values()))
    run.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--input-root',type=Path);p.add_argument('--reference-root',type=Path)
    a=p.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(a.output,cfg)
    try:execute(run,a.input_root or Path(cfg['input']['root']),a.reference_root or Path(cfg['reference']['root']))
    except Exception as exc:run.data['status']='EXECUTION_FAILED';run.failure(run.data['stage'],exc);raise
    finally:finalize(run)
    print('Result:',run.output/'result.json')
if __name__=='__main__':main()
