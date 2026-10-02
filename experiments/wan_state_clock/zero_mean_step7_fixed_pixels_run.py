"""Original saved step7 pixels: encoder anchor, raw420 and standardMP4."""
from __future__ import annotations
import argparse,gzip,json,subprocess,traceback
from pathlib import Path
from experiments.wan_state_clock import zero_mean_channel_margin_run as single
from experiments.wan_state_clock.zero_mean_channel_margin_loop_run import quality
from experiments.wan_state_clock.zero_mean_c1_yuv444_no_h264_run import environment,validate_latent
from runtime.wan import zero_mean_c1_layered_transport as media,zero_mean_c1_yuv444_no_h264 as color,vae as shared
from runtime.wan import zero_mean_c1_yuv420_no_h264 as raw420
from main.tube_state import zero_mean_channel_margin as method
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_step7_fixed_pixels_v1.json'
OBS=('SAVED_RAW444','RAW420','MP4')

class Run(single.Run):
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        self.data=dict(status='RUNNING',stage='INITIAL',config=cfg,environment=environment(),
          source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          calls={k:dict(attempted=0,completed=0) for k in cfg['planned_calls']},fixed_denominator=cfg['fixed_denominator'],
          failures=[],references={},parent_reference={},observations={p:dict(status='PENDING') for p in OBS},
          reads={p+'/'+k:dict(status='PENDING',scored=0) for p in OBS for k in cfg['keys']},
          posthoc={p+'/'+k:dict(status='PENDING') for p in OBS for k in cfg['keys']},
          message_evaluations={p+'/'+k+'/'+t:dict(status='PENDING') for p in OBS for k in cfg['keys'] for t in cfg['messages']},
          mp4=dict(save=dict(status='PENDING'),read=dict(status='PENDING')),comparisons={},primary_endpoint='MP4',
          score_status='UNCALIBRATED_DIAGNOSTIC',evidence_ceiling=cfg['evidence_ceiling'])
        files=[str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),
          'experiments/wan_state_clock/zero_mean_channel_margin_run.py','experiments/wan_state_clock/zero_mean_channel_margin_loop_run.py',
          'experiments/wan_state_clock/zero_mean_c1_yuv444_no_h264_run.py','main/tube_state/zero_mean_channel_margin.py',
          'main/tube_state/video_overlap_zero_mean_state.py','main/tube_state/video_overlap_zero_mean_control.py',
          'runtime/wan/zero_mean_c1_layered_transport.py','runtime/wan/zero_mean_c1_yuv444_no_h264.py',
          'runtime/wan/generation.py','runtime/wan/vae.py','runtime/wan/io.py','runtime/wan/zero_mean_c1_yuv420_no_h264.py']
        self.data['source_files']={p:media.file_sha256(ROOT/p) for p in files};self.save()
    def roundtrip420(self,q8):
        def event(stage,row):
            self.data['observations']['RAW420'].setdefault('conversion',{})[stage]=row;self.save()
        return raw420.roundtrip(q8,self.output/'RAW420/roundtrip.yuv',self.output/'RAW420/roundtrip.rgb',count=self.count,event=event).float()/255.
    def failure(self,exc,obs=None):
        rec=dict(stage=self.data['stage'],observation=obs,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        stderr=getattr(exc,'stderr',None)
        if stderr:
            path=self.output/((obs or 'shared')+'.failure.stderr.txt');path.write_bytes(stderr.encode() if isinstance(stderr,str) else stderr)
            rec.update(stderr_path=str(path),stderr_sha256=media.file_sha256(path))
        self.data['failures'].append(rec)
        for p in (OBS if obs is None else [obs]):
            row=self.data['observations'][p]
            if row['status']=='RUNNING':row['status']='FAILED'
            elif row['status']=='PENDING':row['status']='NOT_COMPLETED'
            for table in ('reads','posthoc','message_evaluations'):
                for key,row in self.data[table].items():
                    if key.startswith(p+'/') and row['status'] in ('PENDING','RUNNING'):row['status']='NOT_COMPLETED'
        if obs in (None,'MP4'):
            for row in self.data['mp4'].values():
                if row['status']=='RUNNING':row['status']='FAILED'
                elif row['status']=='PENDING':row['status']='NOT_COMPLETED'
        self.save()
    def finalize(self):
        conv=self.data['observations']['RAW420'].get('conversion',{})
        counts=dict(raw420_files=int(conv.get('yuv420',{}).get('status')=='SAVED'),raw420_rgb_readbacks=int(conv.get('rgb24',{}).get('status')=='SAVED'),observations=sum(v['status']=='SAVED' for v in self.data['observations'].values()),
          mp4_saved=int(self.data['mp4']['save']['status']=='SAVED'),mp4_read=int(self.data['mp4']['read']['status']=='SAVED'),
          path_reads=sum(v['status']=='SAVED' for v in self.data['reads'].values()),
          message_evaluations=sum(v['status']=='EVALUATED' for v in self.data['message_evaluations'].values()),
          valid_costs=sum(v['scored'] for v in self.data['reads'].values()))
        self.data['counts']=counts
        complete=all(counts[k]==self.cfg['fixed_denominator'][k] for k in counts)
        calls=all(self.data['calls'][k]==dict(attempted=n,completed=n) for k,n in self.cfg['planned_calls'].items())
        self.data['status']='EXECUTION_COMPLETE' if complete and calls and not self.data['failures'] else 'INCOMPLETE'
        self.data['stage']='COMPLETE' if self.data['status']=='EXECUTION_COMPLETE' else 'FINISHED_WITH_FAILURES';self.save()


def load_inputs(run,parent_root,reference_root):
    import hashlib,torch
    cfg=run.cfg;p=parent_root/'result.json'
    if media.file_sha256(p)!=cfg['parent_run']['result_sha256']:raise ValueError('fixed step7 result mismatch')
    parent=json.loads(p.read_text());obs=cfg['parent_run']['observation']
    if parent['status']!='EXECUTION_COMPLETE' or parent['source_sha']!=cfg['parent_run']['source_sha']:raise ValueError('parent source/status mismatch')
    row=parent['observations'][obs];qpath=parent_root/cfg['parent_run']['q8_relative_path']
    if media.file_sha256(qpath)!=cfg['parent_run']['q8_file_sha256']:raise ValueError('saved original step7 RGB8 file mismatch')
    q8=torch.load(qpath,map_location='cpu',weights_only=True);media.validate_rgb(q8,torch.uint8)
    raster_sha=hashlib.sha256(q8.numpy().tobytes()).hexdigest()
    if raster_sha!=cfg['parent_run']['q8_raster_sha256'] or raster_sha!=row['conversion']['yuv444']['input_raster_sha256']:raise ValueError('saved original step7 RGB8 raster mismatch')
    npth=parent_root/obs/'normalized.pt'
    if media.file_sha256(npth)!=row['normalized']['sha256']:raise ValueError('parent raw444 latent mismatch')
    normalized=validate_latent(torch.load(npth,map_location='cpu',weights_only=True))
    reads={}
    for kid in cfg['keys']:
        sid=obs+'/'+kid;raw=parent_root/obs/(kid+'.raw.json.gz')
        if media.file_sha256(raw)!=parent['reads'][sid]['sha256']:raise ValueError('parent raw reader mismatch')
        reads[kid]=dict(path=str(raw),sha256=parent['reads'][sid]['sha256'],posthoc=parent['posthoc'][sid],
          messages={t:parent['message_evaluations'][sid+'/'+t] for t in cfg['messages']})
    rgb_record=row['conversion']['rgb24'];rgb_path=parent_root/obs/'roundtrip.rgb'
    old_rgb=color.reopen_rgb(rgb_path,rgb_record['sha256']).float()/255.
    if media.file_sha256(reference_root/'result.json')!=cfg['reference']['result_sha256']:raise ValueError('reference result mismatch')
    ref=json.loads((reference_root/'result.json').read_text())
    for arm in ('OFF','PAYLOAD_MULTI','OVERLAP_MULTI'):
        n=ref['normalized'][arm];path=reference_root/arm/'normalized.pt'
        if media.file_sha256(path)!=n['sha256']:raise ValueError('reference normalized mismatch')
        run.data['references'][arm]=dict(normalized=dict(path=str(path),sha256=n['sha256']),reads={})
        for kid in cfg['keys']:
            sid=arm+'/'+kid;path=reference_root/arm/(kid+'.raw.json.gz')
            if media.file_sha256(path)!=ref['reads'][sid]['sha256']:raise ValueError('reference raw mismatch')
            run.data['references'][arm]['reads'][kid]=dict(path=str(path),sha256=ref['reads'][sid]['sha256'],posthoc=ref['path_posthoc'][sid],messages={t:ref['message_evaluations'][sid+'/'+t] for t in cfg['messages']})
    run.data['parent_reference']=dict(result_sha256=cfg['parent_run']['result_sha256'],observation=obs,total_updates=7,
      q8=dict(path=str(qpath),sha256=cfg['parent_run']['q8_file_sha256'],raster_sha256=raster_sha),
      normalized=dict(path=str(npth),sha256=row['normalized']['sha256']),reads=reads,
      rgb=dict(path=str(rgb_path),sha256=rgb_record['sha256']),environment=parent['environment'],vae_runtime=parent['vae_runtime'])
    run.data['environment_package_changes']={k:dict(parent=parent['environment']['packages'].get(k),current=v) for k,v in run.data['environment']['packages'].items() if v!=parent['environment']['packages'].get(k)}
    run.save();return q8,normalized,old_rgb


def save_readback(run,name,rgb):
    import torch
    q8=shared.quantize_rgb8_no_codec(rgb)
    if not torch.equal(q8.float()/255.,rgb):raise ValueError('readback not exact RGB8/255')
    path=run.output/name/'roundtrip.rgb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(q8.numpy().tobytes());sha=media.file_sha256(path)
    record=dict(status='SAVED',path=str(path),sha256=sha,bytes=path.stat().st_size,frames=int(q8.shape[0]))
    run.data['mp4']['read'].update(record);run.save()
    return color.reopen_rgb(path,sha).float()/255.


def mp4_channel(run,q8):
    target=run.output/'MP4/source.mp4';target.parent.mkdir(parents=True,exist_ok=True)
    save=run.data['mp4']['save'];save.update(status='RUNNING',path=str(target),codec=run.cfg['codec'],input_raster_sha256=run.data['q8_raster_sha256']);run.save()
    run.call('mp4_save',lambda:media.encode_raster(q8,target))
    save.update(status='SAVED',sha256=media.file_sha256(target),bytes=target.stat().st_size);run.save()
    read=run.data['mp4']['read'];read.update(status='RUNNING',source_sha256=save['sha256']);run.save()
    # Existing read_full_mp4 validates all181 frames, no padding or dropping.
    rgb=run.call('mp4_read',lambda:media.read_full_mp4(target))
    if media.file_sha256(target)!=save['sha256']:raise ValueError('MP4 changed during readback')
    rgb=save_readback(run,'MP4',rgb)
    command=['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(target)]
    probe=subprocess.run(command,capture_output=True,text=True,check=False)
    run.data['mp4']['probe']=dict(command=command,returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr)
    run.save();return rgb


def execute(run,parent_root,reference_root):
    import hashlib,torch
    from runtime.wan.generation import load_frozen_vae
    run.stage('SAVED_INPUTS');q8,parent_norm,parent_rgb=load_inputs(run,parent_root,reference_root)
    run.stage('VAE_LOAD');vae=load_frozen_vae(run.cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(vae,'use_tiling',False):raise ValueError('same native untiled VAE required')
    run.data['vae_runtime']=dict(device=str(next(vae.parameters()).device),dtype=str(next(vae.parameters()).dtype),
      latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std),posterior='mode')
    run.data['shared_q8']=dict(run.data['parent_reference']['q8'],status='VERIFIED_REFERENCE')
    run.data['q8_raster_sha256']=hashlib.sha256(q8.numpy().tobytes()).hexdigest()
    run.data['saved_raster_matches_parent']=run.data['q8_raster_sha256']==run.data['parent_reference']['q8']['raster_sha256'];run.save()
    raw_rgb=None
    for name in OBS:
        row=run.data['observations'][name];row['status']='RUNNING';run.stage(name+'_CHANNEL')
        try:
            if name=='SAVED_RAW444':
                received=parent_rgb;row['received_rgb']=dict(run.data['parent_reference']['rgb'],status='VERIFIED_REFERENCE')
            elif name=='RAW420':received=run.roundtrip420(q8)
            else:received=mp4_channel(run,q8)
            row['quality_vs_shared_q8']=quality(received,q8.float()/255.)
            row['quality_vs_parent_raw444']=quality(received,parent_rgb)
            if name=='RAW420':raw_rgb=received
            elif raw_rgb is not None:row['quality_vs_current_raw420']=quality(received,raw_rgb)
            else:row['quality_vs_current_raw420']=dict(status='REFERENCE_UNAVAILABLE')
            run.stage(name+'_ENCODE');z=run.call('vae_encode_native',lambda:media.encode_normalized(vae,received))
            row.update(status='SAVED',normalized=run.saved(name+'/normalized.pt',z))
            if name=='SAVED_RAW444':
                d=(z-parent_norm).double()
                run.data['baseline_replay']=dict(normalized_max_error=float(d.abs().max()),normalized_rmse=float(d.square().mean().sqrt()),rgb_sha_matches_parent=True,decoder_calls=0)
                del d
            run.save();run.stage(name+'_READOUT');run.read(name,z)
            # Reporting only, after the unchanged blind receiver persists both keys.
            _,loss=method.margin_loss(method.project(z,run.cfg['key']),run.cfg['key']);row['writer_loss_diagnostic']=loss
            p=run.data['posthoc'][name+'/CORRECT']
            raw=json.loads(gzip.decompress(Path(run.data['reads'][name+'/CORRECT']['path']).read_bytes()))['inference']
            if name=='SAVED_RAW444':
                old=json.loads(gzip.decompress(Path(run.data['parent_reference']['reads']['CORRECT']['path']).read_bytes()))['inference']
                if raw['valid_catalog_indices']!=old['valid_catalog_indices']:raise ValueError('fixed family changed')
                run.data['baseline_replay']['max_correct_key_path_cost_error']=max(abs(a-b) for a,b in zip(raw['path_costs'],old['path_costs']))
            wrong=[(i,c) for i,c in zip(raw['valid_catalog_indices'],raw['path_costs']) if i!=0]
            minimum=min(c for _,c in wrong)
            from main.tube_state.video_overlap_zero_mean_state import PUBLIC
            nearest=[i for i,c in wrong if c-minimum<=PUBLIC.tie_atol]
            run.data['comparisons'][name]=dict(delta=p['delta'],rank=p['true_rank'],top_catalog_indices=p['top'],
                true_unique_top=p['truth_unique_top'],global_margin=loss['global_margin'],global_shortfall=loss['global_term'],
                local_active=loss['local_active'],nearest_wrong_catalog_indices=nearest,payload=run.data['message_evaluations'][name+'/CORRECT/REGISTERED'],scientific_pass=False)
            run.save();del z
        except Exception as exc:run.failure(exc,name)
    if 'MP4' in run.data['comparisons']:
        run.data['fixed_endpoint']=dict(observation='MP4',frozen_total_updates=7,**run.data['comparisons']['MP4'],selection='fixed MP4 endpoint')
        for refname in ('SAVED_RAW444','RAW420'):
            if refname in run.data['comparisons']:run.data['fixed_endpoint']['delta_change_vs_'+refname.lower()]=run.data['comparisons']['MP4']['delta']-run.data['comparisons'][refname]['delta']
    run.save()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    for name in ('input-root','reference-root'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();cfg=json.loads(CONFIG.read_text());run=Run(args.output,cfg)
    try:execute(run,args.input_root or Path(cfg['parent_run']['root']),args.reference_root or Path(cfg['reference']['root']))
    except Exception as exc:run.failure(exc)
    finally:run.finalize()
    print(json.dumps(run.data.get('fixed_endpoint'),indent=2));print('Result:',run.output/'result.json')
    if run.data['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
