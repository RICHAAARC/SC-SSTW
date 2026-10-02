"""Matched one-step composite vs worst-only writer objectives, user-run only."""
from __future__ import annotations
import argparse,json,traceback,gc
from pathlib import Path
from experiments.wan_state_clock import zero_mean_raw420_margin_continue_v2_run as previous
from experiments.wan_state_clock import zero_mean_raw420_margin_run as single
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import zero_mean_channel_margin as gradient,zero_mean_c1_layered_transport as media
from runtime.wan import vae as shared
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_raw420_objective_ablation_v1.json'

class Run(single.Run):
    def __init__(self,output,cfg):
        super().__init__(output,cfg);del self.data['update']
        self.data['updates']={arm:dict(status='PENDING',objective=obj) for arm,obj in cfg['arms'].items()}
        self.data['parent_references']={};self.data['baseline_replay']={};self.data['boundary_diagnostics']={}
        self.data['fixed_endpoints']={arm:dict(status='PENDING',selected_by_score=False) for arm in cfg['arms']}
        self.data['prior_raw420_updates']=cfg['prior_raw420_updates']
        for f in (str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),
                  'experiments/wan_state_clock/zero_mean_raw420_margin_continue_v2_run.py'):
            self.data['source_files'][f]=media.file_sha256(ROOT/f)
        self.save()
    def failure(self,stage,exc):
        self.data['failures'].append(dict(stage=stage,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));self.save()


def inputs(run,input_root,reference_root):return previous.inputs(run,input_root,reference_root)


def observe(run,obs,z,clamp):
    run.record(obs,z,clamp);previous.report_boundaries(run,obs)


def execute(run,input_root,reference_root):
    import torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');terminal=inputs(run,input_root,reference_root)
    run.data['baseline_semantics']='Both arms start from the identical saved POINT3 terminal after seven raw420 updates and the same fresh decoded RGB8/raw420 baseline. No sequential arm chaining.'
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('reference full-frame VAE requires native untiled path')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),
        latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    run.stage('BASELINE_DECODE')
    rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,terminal))
    q8=shared.quantize_rgb8_no_codec(rgb);baseline={};baseline_z=None
    for channel in ('RAW420','RAW444'):
        obs='BASELINE_'+channel;run.stage(obs)
        baseline[channel]=run.transport(obs,q8)
        z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,baseline[channel]))
        old=torch.load(run.data['parent_references'][channel]['normalized']['path'],map_location='cpu',weights_only=True)
        d=(z-old).double();run.data['baseline_replay'][channel]=dict(normalized_max_error=float(d.abs().max()),normalized_rmse=float(d.square().mean().sqrt()),
            parent_posthoc=run.data['parent_references'][channel]['reads']['CORRECT']['posthoc'])
        observe(run,obs,z,clamp)
        if channel=='RAW420':baseline_z=z
        del old,d,z
    del q8
    for arm,objective in run.cfg['arms'].items():
        row=run.data['updates'][arm]
        try:
            run.stage(arm+'_ENCODER_VJP');resource={}
            try:z,cotangent,metrics=gradient.encoder_cotangent(vae,baseline['RAW420'],run.cfg['key'],run.count,resource,objective=objective)
            finally:run.data['resources'][arm+'_encoder']=resource;run.save()
            d=(z-baseline_z).double()
            row.update(writer_objective_before=metrics,encoder_replay_normalized_max_error=float(d.abs().max()),encoder_replay_normalized_rmse=float(d.square().mean().sqrt()),
                encoder_replay_normalized=run.saved(arm+'/encoder_replay_normalized.pt',z))
            del d,z
            run.stage(arm+'_DECODER_VJP');resource={}
            try:g=gradient.decoder_vjp(vae,terminal,cotangent,rgb,run.count,resource)
            finally:run.data['resources'][arm+'_decoder']=resource;run.save()
            del cotangent;gc.collect()
            row['gradient']=run.saved(arm+'/terminal_gradient.pt',g)
            step,receipt=method.constrained_step(g,run.cfg['key'],run.cfg['update']['eta'],run.cfg['update']['cap_l2']);del g
            changed=terminal+step;actual=changed-terminal
            row.update(status='SAVED',step=run.saved(arm+'/step.pt',step),terminal=run.saved(arm+'/terminal.pt',changed),receipt=receipt,
                shared_input_sha256=run.data['writer_input']['sha256'],actual_terminal_change_l2=float(actual.double().norm()),
                step_rounding_max_error=float((actual-step).abs().max()),
                sum_raw420_actual_step_l2_on_this_branch=run.data['prior_raw420_step_l2']+float(actual.double().norm()))
            run.save();del actual,step
            run.stage(arm+'_DECODE')
            after_rgb,after_clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,changed));del changed
            q8=shared.quantize_rgb8_no_codec(after_rgb);del after_rgb
            for channel in ('RAW420','RAW444'):
                obs=arm+'_'+channel
                try:
                    run.stage(obs);received=run.transport(obs,q8)
                    z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,received))
                    run.data['observations'][obs]['quality_vs_shared_baseline']=single.quality(received,baseline[channel])
                    _,metrics=method.margin_loss(method.project(z,run.cfg['key']),run.cfg['key'],objective=objective)
                    run.data['observations'][obs]['selected_writer_objective_loss']=dict(objective=objective,metrics=metrics)
                    observe(run,obs,z,after_clamp);del received,z
                    a=run.data['posthoc']['BASELINE_'+channel+'/CORRECT'];b=run.data['posthoc'][obs+'/CORRECT']
                    run.data['comparisons'][obs]=dict(delta_before=a['delta'],delta_after=b['delta'],delta_change=b['delta']-a['delta'],
                        rank_before=a['true_rank'],rank_after=b['true_rank'],all_competitors_defeated=b['truth_unique_top'],scientific_pass=False)
                    run.save()
                except Exception as exc:
                    run.failure(obs,exc)
                    if run.data['observations'][obs]['status']=='PENDING':run.data['observations'][obs]['status']='FAILED'
                    run.save()
            del q8
            run.data['fixed_endpoints'][arm].update(status=run.data['observations'][arm+'_RAW420']['status'],terminal=row['terminal'],
                posthoc={ch:run.data['posthoc'][arm+'_'+ch+'/CORRECT'] for ch in ('RAW420','RAW444')})
            run.save()
        except Exception as exc:
            run.failure(run.data['stage'],exc)
            if row['status']=='PENDING':row['status']='FAILED'
            run.data['fixed_endpoints'][arm]['status']='NOT_COMPLETED';run.save()
    run.data['status']='EXECUTION_PARTIAL' if run.data['failures'] else 'EXECUTION_COMPLETE';run.stage('COMPLETE')


def finalize(run):
    for group in ('observations','reads','posthoc','message_evaluations','updates','fixed_endpoints'):
        for row in run.data[group].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
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
