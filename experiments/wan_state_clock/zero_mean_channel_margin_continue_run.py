"""Fixed three-update continuation of the saved real-channel terminal diagnostic."""
from __future__ import annotations
import argparse,gc,json,subprocess,traceback
from pathlib import Path
import numpy as np
from experiments.wan_state_clock import zero_mean_channel_margin_run as single
from experiments.wan_state_clock.zero_mean_c1_yuv444_no_h264_run import dump,environment
from main.tube_state import zero_mean_channel_margin as method
from runtime.wan import zero_mean_channel_margin as gradient,zero_mean_c1_layered_transport as media
from runtime.wan import zero_mean_c1_yuv444_no_h264 as color,vae as shared
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_margin_continue_v1.json'
POINTS=('POINT1','POINT2','POINT3','POINT4')
TRANSITIONS=('POINT1_TO_POINT2','POINT2_TO_POINT3','POINT3_TO_POINT4')

class Run(single.Run):
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        self.data=dict(status='RUNNING',stage='INITIAL',config=cfg,environment=environment(),
          source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          calls={k:dict(attempted=0,completed=0) for k in cfg['planned_calls']},fixed_denominator=cfg['fixed_denominator'],
          failures=[],references={},parent_reference={},
          observations={p:dict(status='PENDING') for p in POINTS},
          reads={p+'/'+k:dict(status='PENDING',scored=0) for p in POINTS for k in cfg['keys']},
          posthoc={p+'/'+k:dict(status='PENDING') for p in POINTS for k in cfg['keys']},
          message_evaluations={p+'/'+k+'/'+t:dict(status='PENDING') for p in POINTS for k in cfg['keys'] for t in cfg['messages']},
          resources={},updates={s:dict(status='PENDING') for s in TRANSITIONS},comparisons={},
          total_update_indices=cfg['total_update_indices'],primary_endpoint='POINT4',selection='fixed final endpoint; no best-iterate selection',
          score_status='UNCALIBRATED_DIAGNOSTIC',evidence_ceiling=cfg['evidence_ceiling'])
        files=[str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),
          'experiments/wan_state_clock/zero_mean_channel_margin_run.py',
          'main/tube_state/zero_mean_channel_margin.py','runtime/wan/zero_mean_channel_margin.py',
          'main/tube_state/video_overlap_zero_mean_state.py','main/tube_state/video_overlap_zero_mean_control.py',
          'runtime/wan/zero_mean_c1_yuv444_no_h264.py','runtime/wan/gradient_checkpointing.py']
        self.data['source_files']={p:media.file_sha256(ROOT/p) for p in files};self.save()
    def fail(self,exc):
        self.data['status']='EXECUTION_FAILED'
        self.data['failures'].append(dict(stage=self.data['stage'],type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        for table in ('observations','reads','posthoc','message_evaluations','updates'):
            for row in self.data[table].values():
                if row['status']=='RUNNING':row['status']='FAILED'
                elif row['status']=='PENDING':row['status']='NOT_COMPLETED'
        self.save()
    def finalize(self):
        self.data['counts']=dict(observations=sum(v['status']=='SAVED' for v in self.data['observations'].values()),
          updates=sum(v['status']=='SAVED' for v in self.data['updates'].values()),
          path_reads=sum(v['status']=='SAVED' for v in self.data['reads'].values()),
          message_evaluations=sum(v['status']=='EVALUATED' for v in self.data['message_evaluations'].values()),
          valid_costs=sum(v['scored'] for v in self.data['reads'].values()))
        self.save()


def quality(rgb,reference):
    """Frame-wise FP64 reduction, bounded CPU memory; incremental diagnostics only."""
    if rgb.shape!=reference.shape:raise ValueError('same full-frame geometry required')
    total=0.
    for a,b in zip(rgb,reference):total+=float((a-b).double().square().sum())
    rmse=float(np.sqrt(total/rgb.numel()))
    return dict(rgb_rmse=rmse,psnr_db=None if rmse==0 else float(-20*np.log10(rmse)),
                identical_pixels=rmse==0,diagnostic_only=True,threshold=None)


def apply_update(g,key,eta,cap):
    """Zero supported gradient remains a recorded no-op, not early convergence."""
    import torch
    try:return method.constrained_step(g,key,eta,cap)
    except ValueError as exc:
        if str(exc)!='zero supported gradient; no alternate direction or retry':raise
        return torch.zeros_like(g),dict(eta=eta,cap=cap,raw_l2=0.,scale=1.,actual_l2=0.,gradient_dot_step=0.,
          reason='ZERO_SUPPORTED_GRADIENT_NOOP',no_receiver_change=True,no_budget_scan=True)


def load_inputs(run,parent_root,writer_root,reference_root,original_rgb_root=None):
    import torch
    original,_=single.inputs(run,writer_root,reference_root)
    cfg=run.cfg;path=parent_root/'result.json'
    if media.file_sha256(path)!=cfg['parent_run']['result_sha256']:raise ValueError('fixed parent run mismatch')
    parent=json.loads(path.read_text());obs=cfg['parent_run']['observation']
    if parent['source_sha']!=cfg['parent_run']['source_sha'] or parent['status']!='EXECUTION_COMPLETE':raise ValueError('parent source/status mismatch')
    row=parent['observations'][obs]
    terminal_path=parent_root/cfg['parent_run']['terminal_relative_path']
    if media.file_sha256(terminal_path)!=cfg['parent_run']['terminal_sha256'] or cfg['parent_run']['terminal_sha256']!=row['terminal']['sha256']:
        raise ValueError('continuation terminal mismatch')
    terminal=torch.load(terminal_path,map_location='cpu',weights_only=True)
    npth=parent_root/obs/'normalized.pt';nr=row['normalized']
    if media.file_sha256(npth)!=nr['sha256']:raise ValueError('parent normalized mismatch')
    normalized=torch.load(npth,map_location='cpu',weights_only=True)
    for value in (terminal,normalized):
        if value.shape!=original.shape or value.dtype!=torch.float32 or not bool(value.isfinite().all()):raise ValueError('full parent tensor invalid')
    receipts={}
    for kid in cfg['keys']:
        record=parent['reads'][obs+'/'+kid];p=parent_root/obs/(kid+'.raw.json.gz')
        if media.file_sha256(p)!=record['sha256']:raise ValueError('parent raw mismatch')
        receipts[kid]=dict(path=str(p),sha256=record['sha256'],posthoc=parent['posthoc'][obs+'/'+kid],
          messages={t:parent['message_evaluations'][obs+'/'+kid+'/'+t] for t in cfg['messages']})
    anchor_root=Path(original_rgb_root or cfg['original_rgb_reference']['root'])
    if media.file_sha256(anchor_root/'result.json')!=cfg['original_rgb_reference']['result_sha256']:raise ValueError('original RGB reference result mismatch')
    anchor=json.loads((anchor_root/'result.json').read_text())
    original_record=anchor['observations']['BEFORE']['conversion']['rgb24'];start_record=row['conversion']['rgb24']
    rgbs=dict(BEFORE=color.reopen_rgb(anchor_root/'BEFORE/roundtrip.rgb',original_record['sha256']).float()/255.,
              AFTER=color.reopen_rgb(parent_root/obs/'roundtrip.rgb',start_record['sha256']).float()/255.)
    cumulative=parent['updates']['POINT3_TO_POINT4']['sum_all_step_l2']
    run.data['parent_reference']=dict(result_sha256=cfg['parent_run']['result_sha256'],observation=obs,total_updates=4,
      terminal=dict(path=str(terminal_path),sha256=cfg['parent_run']['terminal_sha256']),
      normalized=dict(path=str(npth),sha256=nr['sha256']),reads=receipts,
      baseline_cumulative_step_l2=cumulative,
      original_terminal_displacement_l2=float((terminal-original).double().norm()),
      before_rgb_sha256=original_record['sha256'],start_rgb_sha256=start_record['sha256'],
      original_rgb_reference_result_sha256=cfg['original_rgb_reference']['result_sha256'])
    run.save();return original,terminal,normalized,rgbs,parent


def observe(run,name,z,received,rgb,clamp,loss,original,start,original_rgb,start_rgb,previous_rgb):
    point=run.data['observations'][name]
    index=POINTS.index(name)
    terminal_record=(run.data['updates'][TRANSITIONS[index-1]]['terminal'] if index else run.saved(name+'/terminal.pt',z))
    if media.file_sha256(terminal_record['path'])!=terminal_record['sha256']:raise ValueError('saved iterate terminal changed')
    point.update(status='SAVED',normalized=run.saved(name+'/normalized.pt',received),writer_loss=loss,clamp=clamp,
      terminal=terminal_record,
      original_terminal_displacement_l2=float((z-original).double().norm()),
      start_terminal_displacement_l2=float((z-start).double().norm()),
      quality_vs_original_marked=quality(rgb,original_rgb),quality_vs_start=quality(rgb,start_rgb),
      quality_vs_previous_point=quality(rgb,previous_rgb))
    run.save();run.read(name,received)
    current=run.data['posthoc'][name+'/CORRECT'];start=run.data['posthoc']['POINT1/CORRECT']
    run.data['comparisons'][name]=dict(total_updates=run.cfg['total_update_indices'][name],delta=current['delta'],rank=current['true_rank'],
       delta_change_from_point1=current['delta']-start['delta'],true_unique_top=current['truth_unique_top'],
       original_terminal_displacement_l2=point['original_terminal_displacement_l2'],scientific_pass=False,top_catalog_indices=current['top'],
       global_margin=loss['global_margin'],global_shortfall=loss['global_term'],local_active=loss['local_active'])
    run.save()


def execute(run,parent_root,writer_root,reference_root,original_rgb_root=None):
    import torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS')
    original,terminal,parent_norm,rgbs,parent=load_inputs(run,parent_root,writer_root,reference_root,original_rgb_root)
    start=terminal.clone();previous_rgb=rgbs['AFTER'];sum_l2=sum_sq=0.
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('same native untiled full-frame VAE required')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),
      latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    for i,name in enumerate(POINTS):
        run.data['observations'][name]['status']='RUNNING';run.stage(name+'_REAL_CHANNEL')
        rgb,clamp=run.call('vae_decode_native',lambda:media.decode_with_clamp_receipt(vae,terminal))
        received_rgb=run.roundtrip(name,shared.quantize_rgb8_no_codec(rgb))
        cotangent=None
        if i<3:
            resource={};run.stage(name+'_ENCODER_VJP')
            try:received,cotangent,loss=gradient.encoder_cotangent(vae,received_rgb,run.cfg['key'],run.count,resource)
            finally:run.data['resources'][name+'/encoder']=resource;run.save()
        else:
            received=run.call('vae_encode_native',lambda:media.encode_normalized(vae,received_rgb))
            _,loss=method.margin_loss(method.project(received,run.cfg['key']),run.cfg['key'])
        if i==0:
            run.data['baseline_replay']=dict(normalized_max_error=float((received-parent_norm).abs().max()),
              rgb_sha_matches_parent=run.data['observations'][name]['conversion']['rgb24']['sha256']==run.data['parent_reference']['start_rgb_sha256'])
            del parent_norm
        run.stage(name+'_READOUT')
        observe(run,name,terminal,received,received_rgb,clamp,loss,original,start,rgbs['BEFORE'],rgbs['AFTER'],previous_rgb)
        del received;previous_rgb=received_rgb
        if i==3:
            del rgb;break
        transition=TRANSITIONS[i];update=run.data['updates'][transition];update['status']='RUNNING'
        resource={};run.stage(name+'_DECODER_VJP')
        try:g=gradient.decoder_vjp(vae,terminal,cotangent,rgb,run.count,resource)
        finally:run.data['resources'][name+'/decoder']=resource;run.save()
        del rgb,cotangent;gc.collect()
        run.stage(name+'_APPLY_UPDATE')
        update['gradient']=run.saved(transition+'/terminal_gradient.pt',g)
        step,receipt=apply_update(g,run.cfg['key'],run.cfg['update']['eta'],run.cfg['update']['cap_l2']);del g
        changed=terminal+step;actual=changed-terminal
        sum_l2+=receipt['actual_l2'];sum_sq+=receipt['actual_l2']**2
        update.update(status='SAVED',from_point=name,to_point=POINTS[i+1],
          step=run.saved(transition+'/step.pt',step),terminal=run.saved(POINTS[i+1]+'/terminal.pt',changed),receipt=receipt,
          actual_change_l2=float(actual.double().norm()),step_rounding_max_error=float((actual-step).abs().max()),
          sum_new_step_l2=sum_l2,sum_new_step_l2_squared=sum_sq,
          sum_all_step_l2=run.data['parent_reference']['baseline_cumulative_step_l2']+sum_l2,
          original_terminal_displacement_l2=float((changed-original).double().norm()),
          start_terminal_displacement_l2=float((changed-start).double().norm()))
        run.save();terminal=changed;del step,actual
    for k,n in run.cfg['planned_calls'].items():
        if run.data['calls'][k]!=dict(attempted=n,completed=n):raise RuntimeError('fixed call count mismatch: '+k)
    run.data['fixed_endpoint']=dict(point='POINT4',**run.data['comparisons']['POINT4'],
       payload=run.data['message_evaluations']['POINT4/CORRECT/REGISTERED'],
       quality=run.data['observations']['POINT4']['quality_vs_original_marked'],best_iterate_selected=False)
    run.data['status']='EXECUTION_COMPLETE';run.stage('COMPLETE')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-root',type=Path);parser.add_argument('--writer-root',type=Path);parser.add_argument('--reference-root',type=Path);parser.add_argument('--original-rgb-root',type=Path)
    args=parser.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(args.output,cfg)
    try:execute(run,args.input_root or Path(cfg['parent_run']['root']),args.writer_root or Path(cfg['input']['root']),args.reference_root or Path(cfg['reference']['root']),args.original_rgb_root)
    except Exception as exc:run.fail(exc);raise
    finally:run.finalize()
    print(json.dumps(run.data.get('fixed_endpoint'),indent=2));print('Result:',run.output/'result.json')
if __name__=='__main__':main()
