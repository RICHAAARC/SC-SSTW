"""Fixed saved-direction attribution using two raw420 baseline encoder VJPs."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,traceback
from pathlib import Path
import numpy as np
from experiments.wan_state_clock import zero_mean_raw420_margin_run as base
from experiments.wan_state_clock.zero_mean_fixed_layer_forward_run import checked
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import zero_mean_channel_margin as gradient,zero_mean_c1_layered_transport as media,vae as shared
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_direction_v1.json'
RGB_SHAPE=(181,320,512,3)

def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)

def rgb(path,layer):
    import torch
    if layer=='RAW420':
        x=torch.from_numpy(np.fromfile(path,dtype=np.uint8).reshape(RGB_SHAPE).copy()).float()/255.
    else:
        x=load(path)
        if layer=='RGB8':
            if x.dtype!=torch.uint8:raise ValueError('RGB8 dtype mismatch')
            x=x.float()/255.
    if tuple(x.shape)!=RGB_SHAPE or x.dtype!=torch.float32 or not bool(x.isfinite().all()):raise ValueError('RGB shape/dtype/finiteness mismatch')
    return x

def direction_dot(cotangent,before,after):
    if cotangent.shape!=before.shape or before.shape!=after.shape:raise ValueError('dot geometry mismatch')
    frames=[float((cotangent[i].double()*(after[i].double()-before[i].double())).sum()) for i in range(len(before))]
    return dict(total=math.fsum(frames),per_frame=frames)

class Run(base.Run):
    def __init__(self,output,cfg):
        super().__init__(output,cfg);del self.data['update']
        self.data.update(input_files={},input_replay={},reference_observations={},encoder_replays={a:dict(status='PENDING') for a in cfg['arms']},attributions={a:dict(status='PENDING') for a in cfg['arms']})
        for f in (str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),'experiments/wan_state_clock/zero_mean_fixed_layer_forward_run.py','runtime/wan/generation.py'):
            self.data['source_files'][f]=media.file_sha256(ROOT/f)
        self.save()
    def failure(self,stage,exc):
        self.data['failures'].append(dict(stage=stage,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));self.save()

def inputs(run,input_root,reference_root,terminal_root):
    cfg=run.cfg
    for name,path in [('input',input_root),('reference',reference_root),('terminal',terminal_root)]:checked(path/'result.json',cfg[name]['result_sha256'])
    fl=json.loads((input_root/'result.json').read_text());old=json.loads((reference_root/'result.json').read_text())
    points=['BASELINE',*cfg['arms']];files={}
    for point in points:
        f={};files[point]=f;proof={};run.data['input_files'][point]=proof
        for layer,key in [('FLOAT','float_rgb'),('RGB8','rgb8')]:
            path=input_root/point/(key+'.pt');proof[layer]=checked(path,fl['replay'][point][key]['sha256']);f[layer]=path
        path=reference_root/(point+'_RAW420')/'roundtrip.rgb';row=old['observations'][point+'_RAW420']
        proof['RAW420']=checked(path,row['conversion']['rgb24']['sha256']);f['RAW420']=path
        norm=reference_root/(point+'_RAW420')/'normalized.pt';proof['normalized']=checked(norm,row['normalized']['sha256']);f['normalized']=norm
        terminal=terminal_root/cfg['terminal']['terminal_relative_path'] if point=='BASELINE' else reference_root/point/'terminal.pt'
        proof['terminal']=checked(terminal,fl['config']['terminals'][point]);f['terminal']=terminal
        if point!='BASELINE':
            update=old['updates'][point]
            if update['shared_input_sha256']!=cfg['terminal']['terminal_sha256']:raise ValueError('terminal baseline mismatch')
            for key in ('gradient','step'):
                path=reference_root/point/('terminal_gradient.pt' if key=='gradient' else 'step.pt')
                proof[key]=checked(path,update[key]['sha256']);f[key]=path
            run.data['attributions'][point]['stored_gradient_dot_step']=update['receipt']['gradient_dot_step']
        ref=dict(writer_loss=row['writer_loss'],reads={},posthoc={},messages={})
        for kid in cfg['keys']:
            sid=point+'_RAW420/'+kid;path=reference_root/(point+'_RAW420')/(kid+'.raw.json.gz')
            ref['reads'][kid]=checked(path,old['reads'][sid]['sha256'])
            raw=json.loads(gzip.decompress(path.read_bytes()))
            if raw['truth_used'] or raw['writer_inputs'] or len(raw['inference']['path_costs'])!=174:raise ValueError('reference read invalid')
            ref['posthoc'][kid]=old['posthoc'][sid];ref['messages'][kid]={m:old['message_evaluations'][sid+'/'+m] for m in cfg['messages']}
        run.data['reference_observations'][point]=ref
        # Verify the previously oversized float files on Colab directly, frame by frame.
        x=rgb(f['FLOAT'],'FLOAT');q=load(f['RGB8'])
        exact=all(np.array_equal(shared.quantize_rgb8_no_codec(x[i:i+1]).numpy(),q[i:i+1].numpy()) for i in range(len(x)))
        sha=hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest()
        run.data['input_replay'][point]=dict(float_quantizes_to_saved_rgb8=exact,rgb8_sha256=sha,matches_old_raw420_input=sha==row['conversion']['yuv420']['input_raster_sha256'])
        del x,q
        z=load(norm);_,metrics=method.margin_loss(method.project(z,cfg['key']),cfg['key']);del z
        # Full-family objective recomputation; all receiver evidence stays reference-only.
        ref['recomputed_writer_loss']=metrics
        run.save()
    return files

def save_cotangent(run,arm,cot):
    parts=[];size=run.cfg['cotangent_chunk_frames']
    for start in range(0,len(cot),size):
        end=min(start+size,len(cot))
        rec=run.saved(arm+f'/rgb_cotangent_{start:03d}_{end:03d}.pt',cot[start:end].clone())
        rec.update(frame_start=start,frame_end_exclusive=end);parts.append(rec)
    return parts

def execute(run,input_root,reference_root,terminal_root):
    import torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');files=inputs(run,input_root,reference_root,terminal_root)
    baseline=rgb(files['BASELINE']['RAW420'],'RAW420');baseline_z=load(files['BASELINE']['normalized'])
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('native untiled VAE required')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    for arm,objective in run.cfg['arms'].items():
        try:
            run.stage(arm+'_ENCODER_VJP');resource={}
            try:z,cot,metrics=gradient.encoder_cotangent(vae,baseline,run.cfg['key'],run.count,resource,objective=objective)
            finally:run.data['resources'][arm+'_encoder']=resource;run.save()
            d=(z-baseline_z).double()
            replay=run.data['encoder_replays'][arm]
            replay.update(status='SAVED',normalized=run.saved(arm+'/encoder_replay_normalized.pt',z),normalized_max_error=float(d.abs().max()),normalized_rmse=float(d.square().mean().sqrt()),objective=objective,metrics=metrics,cotangent_parts=save_cotangent(run,arm,cot))
            run.save();del z,d
            row=run.data['attributions'][arm]
            for layer in ('FLOAT','RGB8','RAW420'):
                run.stage(arm+'_'+layer+'_DOT')
                before=baseline if layer=='RAW420' else rgb(files['BASELINE'][layer],layer)
                after=rgb(files[arm][layer],layer)
                row[layer]=direction_dot(cot,before,after);del before,after
                run.save()
            g=load(files[arm]['gradient']);step=load(files[arm]['step']);start=load(files['BASELINE']['terminal']);end=load(files[arm]['terminal'])
            if not torch.equal(start+step,end):raise ValueError('saved terminal/step mismatch')
            actual=end-start
            row['terminal_prediction']=float((g.double()*actual.double()).sum())
            row['stored_step_prediction_recomputed']=float((g.double()*step.double()).sum())
            row['actual_terminal_change_l2']=float(actual.double().norm());del g,step,start,end,actual
            key='loss' if objective=='composite' else 'global_term'
            a=run.data['reference_observations']['BASELINE']['recomputed_writer_loss'][key]
            b=run.data['reference_observations'][arm]['recomputed_writer_loss'][key]
            row['actual_objective_change']=b-a
            pz=row['terminal_prediction'];px=row['FLOAT']['total'];pq=row['RGB8']['total'];py=row['RAW420']['total'];actual=b-a
            row['differences']=dict(float_minus_terminal=px-pz,rgb8_minus_float=pq-px,raw420_minus_rgb8=py-pq,actual_minus_raw420=actual-py)
            row['telescoping_residual']=math.fsum(row['differences'].values())-(actual-pz)
            row.update(status='EVALUATED',objective=objective,scientific_pass=False)
            run.save();del cot
        except Exception as exc:
            run.failure(run.data['stage'],exc);run.data['attributions'][arm]['status']='FAILED';run.save()
    exact=all(v['float_quantizes_to_saved_rgb8'] and v['matches_old_raw420_input'] for v in run.data['input_replay'].values())
    run.data['raster_alignment']='EXACT' if len(run.data['input_replay'])==3 and exact else 'DIFFERS_OR_INCOMPLETE'
    run.data['status']='EXECUTION_PARTIAL' if run.data['failures'] else 'EXECUTION_COMPLETE';run.stage('COMPLETE')

def finalize(run):
    for group in ('encoder_replays','attributions'):
        for row in run.data[group].values():
            if row['status']=='PENDING':row['status']='NOT_COMPLETED'
    run.data['counts']=dict(encoder_vjps=run.data['calls']['encoder_vjp']['completed'],replay_normalized=sum(v['status']=='SAVED' for v in run.data['encoder_replays'].values()),rgb_cotangents=sum(bool(v.get('cotangent_parts')) for v in run.data['encoder_replays'].values()),attributions=sum(v['status']=='EVALUATED' for v in run.data['attributions'].values()),updates=0,new_path_reads=0,new_payload_reads=0,reference_observations=len(run.data['reference_observations']),reference_raw=sum(len(v['reads']) for v in run.data['reference_observations'].values()))
    run.save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    for arg in ('input-root','reference-root','terminal-root'):p.add_argument('--'+arg,type=Path)
    a=p.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(a.output,cfg)
    try:execute(run,a.input_root or Path(cfg['input']['root']),a.reference_root or Path(cfg['reference']['root']),a.terminal_root or Path(cfg['terminal']['root']))
    except Exception as exc:run.data['status']='EXECUTION_FAILED';run.failure(run.data['stage'],exc);raise
    finally:finalize(run)
    print('Result:',run.output/'result.json')
if __name__=='__main__':main()
