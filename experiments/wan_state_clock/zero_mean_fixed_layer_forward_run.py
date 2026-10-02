"""Fixed saved endpoints: float RGB and RGB8 forward localization, user-run only."""
from __future__ import annotations
import argparse,gzip,hashlib,json,traceback
from pathlib import Path
from experiments.wan_state_clock import zero_mean_raw420_margin_run as base
from runtime.wan import zero_mean_c1_layered_transport as media,vae as shared
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_fixed_layer_forward_v1.json'

class Run(base.Run):
    def __init__(self,output,cfg):
        super().__init__(output,cfg);del self.data['update']
        self.data.update(inputs={},reference_observations={},replay={},surrogate_predictions={},layer_increments={})
        for f in (str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),'runtime/wan/generation.py'):
            self.data['source_files'][f]=media.file_sha256(ROOT/f)
        self.data['comparisons']={a+'_'+l:dict(status='PENDING') for a in ('COMPOSITE','WORST_ONLY') for l in ('FLOAT','RGB8','RAW420','RAW444')}
        self.save()
    def failure(self,stage,exc):
        self.data['failures'].append(dict(stage=stage,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));self.save()

def checked(path,sha):
    if media.file_sha256(path)!=sha:raise ValueError('saved input identity mismatch: '+str(path))
    return dict(path=str(path),sha256=sha)

def inputs(run,input_root,reference_root):
    import torch
    cfg=run.cfg
    checked(input_root/'result.json',cfg['input']['result_sha256'])
    old=json.loads((input_root/'result.json').read_text())
    checked(reference_root/'result.json',cfg['reference']['result_sha256'])
    terminals={}
    for point in cfg['points']:
        path=reference_root/cfg['reference']['terminal_relative_path'] if point=='BASELINE' else input_root/point/'terminal.pt'
        rec=checked(path,cfg['terminals'][point]);z=torch.load(path,map_location='cpu',weights_only=True)
        if tuple(z.shape)!=(1,16,46,40,64) or z.dtype!=torch.float32 or not bool(z.isfinite().all()):raise ValueError('full terminal invalid')
        terminals[point]=z;run.data['inputs'][point]=rec
        if point!='BASELINE':
            row=old['updates'][point]
            if row['shared_input_sha256']!=cfg['terminals']['BASELINE']:raise ValueError('ablation baseline mismatch')
            run.data['surrogate_predictions'][point]=dict(objective=row['objective'],gradient_dot_step=row['receipt']['gradient_dot_step'],actual_terminal_change_l2=row['actual_terminal_change_l2'],domain='stored raw420 identity-STE objective at the original common baseline; not a new layer gradient')
        for layer in ('RAW420','RAW444'):
            obs=point+'_'+layer;row=old['observations'][obs]
            ref=dict(normalized=checked(input_root/obs/'normalized.pt',row['normalized']['sha256']),writer_loss=row['writer_loss'],reads={},posthoc={},messages={})
            for kid in cfg['keys']:
                sid=obs+'/'+kid;raw=input_root/obs/(kid+'.raw.json.gz')
                ref['reads'][kid]=checked(raw,old['reads'][sid]['sha256'])
                data=json.loads(gzip.decompress(raw.read_bytes()))
                if data['truth_used'] or data['writer_inputs'] or len(data['inference']['path_costs'])!=174:raise ValueError('reference raw read invalid')
                ref['posthoc'][kid]=old['posthoc'][sid]
                ref['messages'][kid]={m:old['message_evaluations'][sid+'/'+m] for m in cfg['messages']}
            stage='yuv420' if layer=='RAW420' else 'yuv444'
            ref['input_raster_sha256']=row['conversion'][stage]['input_raster_sha256']
            run.data['reference_observations'][obs]=ref
    run.save();return terminals

def compare(run):
    def view(point,layer):
        obs=point+'_'+layer
        if layer in ('RAW420','RAW444'):
            v=run.data['reference_observations'].get(obs,{})
            return v.get('posthoc',{}).get('CORRECT',{}),v.get('writer_loss',{})
        return run.data['posthoc'][obs+'/CORRECT'],run.data['observations'][obs].get('writer_loss',{})
    for arm in ('COMPOSITE','WORST_ONLY'):
        for layer in ('FLOAT','RGB8','RAW420','RAW444'):
            a,la=view('BASELINE',layer);b,lb=view(arm,layer)
            if a.get('status')!='EVALUATED' or b.get('status')!='EVALUATED':continue
            run.data['comparisons'][arm+'_'+layer]=dict(status='EVALUATED',delta_before=a['delta'],delta_after=b['delta'],delta_change=b['delta']-a['delta'],rank_before=a['true_rank'],rank_after=b['true_rank'],truth_unique_top=b['truth_unique_top'],composite_loss_change=lb['loss']-la['loss'],global_loss_change=lb['global_term']-la['global_term'],local_mean_change=lb['local_mean']-la['local_mean'],scientific_pass=False)
        for before,after in (('FLOAT','RGB8'),('RGB8','RAW420'),('RGB8','RAW444')):
            a=run.data['comparisons'][arm+'_'+before];b=run.data['comparisons'][arm+'_'+after]
            if a['status']=='EVALUATED' and b['status']=='EVALUATED':
                run.data['layer_increments'][arm+'/'+before+'->'+after]=dict(delta_response_difference=b['delta_change']-a['delta_change'],global_loss_response_difference=b['global_loss_change']-a['global_loss_change'],composite_loss_response_difference=b['composite_loss_change']-a['composite_loss_change'],interpretation='Difference of finite responses across layers, not an isolated causal effect')
    run.save()

def execute(run,input_root,reference_root):
    import torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');terminals=inputs(run,input_root,reference_root)
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('native untiled VAE required')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    for point in run.cfg['points']:
        try:
            run.stage(point+'_DECODE');rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,terminals[point]))
            q8=shared.quantize_rgb8_no_codec(rgb)
            sha=hashlib.sha256(q8.contiguous().numpy().tobytes()).hexdigest()
            expected={l:run.data['reference_observations'][point+'_'+l]['input_raster_sha256'] for l in ('RAW420','RAW444')}
            run.data['replay'][point]=dict(rgb8_sha256=sha,expected=expected,exact_for_both_references=all(sha==v for v in expected.values()),rgb8=run.saved(point+'/rgb8.pt',q8),float_rgb=run.saved(point+'/float_rgb.pt',rgb))
            run.save()
            for layer in run.cfg['layers']:
                obs=point+'_'+layer
                try:
                    run.stage(obs);received=rgb if layer=='FLOAT' else q8.float()/255.
                    z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,received))
                    run.record(obs,z,clamp);del received,z
                except Exception as exc:
                    run.failure(obs,exc)
                    if run.data['observations'][obs]['status']=='PENDING':run.data['observations'][obs]['status']='FAILED'
                    run.save()
            del rgb,q8
        except Exception as exc:run.failure(point,exc)
    compare(run)
    run.data['reference_comparison_alignment']='EXACT_RGB8_REPLAY' if len(run.data['replay'])==3 and all(v['exact_for_both_references'] for v in run.data['replay'].values()) else 'REPLAY_DIFFERS_OR_INCOMPLETE'
    run.data['status']='EXECUTION_PARTIAL' if run.data['failures'] else 'EXECUTION_COMPLETE';run.stage('COMPLETE')

def finalize(run):
    for group in ('observations','reads','posthoc','message_evaluations','comparisons'):
        for row in run.data[group].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
    run.data['counts']=dict(observations=sum(v['status']=='SAVED' for v in run.data['observations'].values()),updates=0,path_reads=sum(v['status']=='SAVED' for v in run.data['reads'].values()),payload_reads=run.data['calls']['payload_read']['completed'],message_evaluations=sum(v['status']=='EVALUATED' for v in run.data['message_evaluations'].values()),valid_costs=sum(v['scored'] for v in run.data['reads'].values()),reference_observations=len(run.data['reference_observations']),reference_raw=sum(len(v['reads']) for v in run.data['reference_observations'].values()))
    run.save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--input-root',type=Path);p.add_argument('--reference-root',type=Path)
    a=p.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(a.output,cfg)
    try:execute(run,a.input_root or Path(cfg['input']['root']),a.reference_root or Path(cfg['reference']['root']))
    except Exception as exc:run.data['status']='EXECUTION_FAILED';run.failure(run.data['stage'],exc);raise
    finally:finalize(run)
    print('Result:',run.output/'result.json')
if __name__=='__main__':main()
