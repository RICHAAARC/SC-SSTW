"""Fixed real-RGB crop diagnostic from six fixed holdout MP4s. No Transformer or generation."""
import argparse,gc,hashlib,json,platform,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import state_clock
from runtime.wan.generation import load_frozen_vae
from runtime.wan.io import dump,read_mp4
from runtime.wan.vae import reencode_rgb24_readback,_clear_cache
from . import flow_tube_crop_analysis as analysis
MANIFEST=Path(__file__).parent/'configs/flow_tube_holdout_crop.json'
SOURCE_RUN='flow_tube_state_holdout_20260920T033109891109Z'
SOURCE_COMMIT='0797064cc5f1166170ed5c229e18defd681d8185'
SOURCE_ROOT=Path('/content/drive/MyDrive/Video-WM/FlowTubeStateHoldout')/SOURCE_RUN
CASES=('holdout_p0_s0','holdout_p1_s0')
ARMS=('OFF','LAST_A','LAST_B');STARTS=(0,4,5);FRAMES=129
SOURCE_SHAPE=(181,320,512,3);CROP_SHAPE=(129,320,512,3)
MODES=('global_matched','global_state','local_matched','local_without_update','local_state',*analysis.NO_SEARCH_MODES)
PLAN={'transformer':0,'vae_decode':0,'mp4_save':0,'source_mp4_read':3,'crop_save':9,'crop_read':9,'vae_encode':36,'blind_read':9}

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def empty(status):return dict(status=status,fragments={a+'_'+str(s):dict(status='NOT_RUN',arm=a,start_reporting_only=s,observations={str(g):{'status':'NOT_RUN'} for g in range(4)},rankings={},reporting_only={}) for a in ARMS for s in STARTS},failures=[])
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def validate_manifest(m):
    expected=dict(protocol='flow_tube_holdout_real_crop_v1',source_run=SOURCE_RUN,source_commit=SOURCE_COMMIT,cases=list(CASES),arms=list(ARMS),starts=list(STARTS),frames=FRAMES,origins=[0,1,2,3],storage='uint8_npy_no_second_codec',source_shape=list(SOURCE_SHAPE),crop_shape=list(CROP_SHAPE),no_search=analysis.NO_SEARCH)
    if m!=expected:raise ValueError('fixed crop protocol mismatch')

def validate_output(output,source_root):
    output=Path(output).resolve();source_root=Path(source_root).resolve()
    if output==source_root or output.is_relative_to(source_root):raise ValueError('output must be independent of source')
    return output,source_root


def load_received_crop(path,expected_hash):
    if sha(path)!=expected_hash:raise ValueError('persisted crop hash mismatch')
    pixels=np.load(path,allow_pickle=False)
    if pixels.dtype!=np.uint8 or tuple(pixels.shape)!=CROP_SHAPE:raise ValueError('invalid persisted uint8 RGB crop')
    return torch.from_numpy(pixels).float()/255.


def encode_origins(vae,rgb,count,store,record_failure):
    """No start/message/source-latent arguments. Encode actual public RGB phases."""
    if tuple(rgb.shape)!=CROP_SHAPE or not torch.isfinite(rgb).all():raise ValueError('invalid received RGB')
    obs={};rows={str(g):{'status':'NOT_RUN'} for g in range(4)}
    for g in range(4):
        encoded=None
        try:
            groups,tail=divmod(len(rgb)-g-1,4);view=rgb[g:g+1+4*groups]
            count('vae_encode',False);encoded=reencode_rgb24_readback(vae,view).cpu().float();count('vae_encode',True)
            if tuple(encoded.shape)!=(1,16,1+groups,40,64) or not torch.isfinite(encoded).all():raise ValueError('nonfinite or wrong receiver latent geometry')
            store(g,encoded);obs[g]=encoded.numpy()
            rows[str(g)]=dict(status='COMPLETE',frames_used=len(view),local_rgb_start=g,tail_discarded=tail,latent_groups=groups)
        except Exception as exc:rows[str(g)]={'status':'FAILED','error':repr(exc)};record_failure('phase'+str(g),exc)
        finally:encoded=None;_clear_cache(vae);release()
    return obs,rows


def run_case(case_id,output,source_root=SOURCE_ROOT):
    manifest=load(MANIFEST);validate_manifest(manifest)
    output,source_root=validate_output(output,source_root);output.mkdir(parents=True,exist_ok=False)
    result=empty('RUNNING');result.update(case=case_id,source_root=str(source_root),fixed_calls=PLAN,actual_calls={k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},file_sha256={},scientific_pass=None)
    def save():dump(output/'result.json',result)
    def count(k,done):result['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,exc):result['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    save();dump(output/'manifest.json',manifest);vae=None
    try:
        if source_root.name!=SOURCE_RUN:raise ValueError('fixed source run basename required')
        source_case=load(source_root/'result.json')['cases'][case_id];source_dir=source_root/case_id
        if source_case.get('source_commit')!=SOURCE_COMMIT:raise ValueError('unexpected source commit')
        for name in ('config.json','codebook.npz'):
            if sha(source_dir/name)!=source_case['file_sha256'].get(name):raise ValueError('source metadata hash mismatch: '+name)
        config=load(source_dir/'config.json');book=state_clock.codebook(config['key_utf8'].encode())
        with np.load(source_dir/'codebook.npz',allow_pickle=False) as stored:
            if set(stored.files)!=set(book) or any(not np.array_equal(stored[k],v) for k,v in book.items()):raise ValueError('source book differs from public-key reconstruction')
        dump(output/'source_config.json',config);np.savez(output/'public_codebook.npz',**book)
        result['source_metadata']=dict(source_commit=SOURCE_COMMIT,result_sha256=sha(source_root/'result.json'),config_sha256=sha(source_dir/'config.json'),codebook_sha256=sha(source_dir/'codebook.npz'))
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['implementation_sha256']={str(p):sha(p) for p in [Path(__file__),MANIFEST,Path(analysis.__file__),Path(state_clock.__file__),Path(analysis.carrier.__file__)]}
        import diffusers
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        vae=load_frozen_vae(config);result['resolved_vae_revision']=getattr(vae.config,'_commit_hash',None)
        for arm in ARMS:
            raw=None
            try:
                src=source_dir/'received_videos'/(arm+'.mp4');actual_hash=sha(src)
                if actual_hash!=source_case['file_sha256'].get('received_videos/'+arm+'.mp4'):raise ValueError('source MP4 hash mismatch')
                count('source_mp4_read',False);rgb=read_mp4(src);count('source_mp4_read',True)
                if tuple(rgb.shape)!=SOURCE_SHAPE or not torch.isfinite(rgb).all() or bool(((rgb<0)|(rgb>1)).any()):raise ValueError('invalid source MP4 RGB readback')
                raw=(rgb*255).round().to(torch.uint8).numpy();del rgb
                for start in STARTS:
                    key=arm+'_'+str(start);item=result['fragments'][key];item['status']='RUNNING';dest=output/key;dest.mkdir()
                    item['attack_metadata_reporting_only']=dict(source_mp4=str(src),source_sha256=actual_hash,source_rgb_range_zero_based=[start,start+FRAMES],range_end_exclusive=True,frames=FRAMES,second_codec=False)
                    obs=received=detection=None
                    try:
                        cropped=np.array(raw[start:start+FRAMES],copy=True);path=dest/'crop_rgb8.npy'
                        count('crop_save',False);np.save(path,cropped,allow_pickle=False);count('crop_save',True)
                        result['file_sha256'][str(path.relative_to(output))]=sha(path);del cropped
                        count('crop_read',False);received=load_received_crop(path,result['file_sha256'][str(path.relative_to(output))]);count('crop_read',True)
                        def store(g,z):
                            p=dest/('g'+str(g)+'.pt');torch.save(z,p);result['file_sha256'][str(p.relative_to(output))]=sha(p)
                        obs,rows=encode_origins(vae,received,count,store,lambda stage,exc:fail(key+'/'+stage,exc));item['observations']=rows
                        count('blind_read',False);detection=state_clock.read(obs,book);count('blind_read',True)
                        # Fixed no-search scores are extracted without crop truth or message.
                        rankings=dict(detection['rankings'])|analysis.no_search_rankings(detection)
                        dump(dest/'detection.json',detection);item['rankings']=rankings
                        truth=None if arm=='OFF' else (0 if arm=='LAST_A' else 1)
                        item['reporting_only']=analysis.ranked_report(detection,rankings,truth,start)
                        item['status']='COMPLETE' if len(obs)==4 else 'PARTIAL_OR_FAILED'
                    except Exception as exc:item['status']='FAILED';fail(key,exc)
                    finally:obs=received=detection=None;release();save()
            except Exception as exc:
                for start in STARTS:
                    item=result['fragments'][arm+'_'+str(start)]
                    if item['status'] in ('NOT_RUN','RUNNING'):item['status']='FAILED_SOURCE'
                fail(arm,exc)
            finally:raw=None;release()
    except Exception as exc:fail('setup',exc)
    finally:vae=None;release()
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['fragments'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    save();return result


def summarize(cases):
    result={}
    for start in STARTS:
        result[str(start)]={}
        for mode in MODES:
            done=correct=0;gaps=[];coverage=[];mapping=[];off=[]
            for case in CASES:
                for arm in ARMS:
                    v=cases.get(case,{}).get('fragments',{}).get(arm+'_'+str(start),{});row=v.get('reporting_only',{}).get(mode,{})
                    if arm=='OFF':off.append(dict(case=case,status=v.get('status','MISSING'),best=row.get('best'),reference_score_0_minus_1=row.get('reference_score_0_minus_1')));continue
                    if v.get('status')!='COMPLETE' or not row:continue
                    done+=1;correct+=int(row.get('unique_correct_reporting_only') is True)
                    gaps.append(dict(case=case,arm=arm,value=row.get('best_correct_minus_other_reporting_only')))
                    coverage.append(dict(case=case,arm=arm,supports=row.get('best_matched_supports')))
                    mapping.append(dict(case=case,arm=arm,**row.get('crop_alignment_reporting_only',{})))
            result[str(start)][mode]=dict(marked_denominator=4,complete=done,missing_or_failed=4-done,unique_correct=correct,message_margins=gaps,coverage=coverage,window_mapping=mapping,OFF=off,claim='two-candidate attribution and window mapping, not FPR or bit payload recovery')
    return result


def compact_summary(summary):
    rows=[]
    for start,modes in summary.items():
        for mode,row in modes.items():
            margins=[v['value'] for v in row['message_margins'] if v['value'] is not None]
            mappings=row['window_mapping'];fractions=[v['matched_window_fraction'] for v in mappings if v.get('matched_window_fraction') is not None]
            rows.append(dict(start=int(start),mode=mode,denominator=row['marked_denominator'],complete=row['complete'],missing_or_failed=row['missing_or_failed'],unique_correct=row['unique_correct'],
                margin_available=len(margins),margin_min=min(margins) if margins else None,margin_max=max(margins) if margins else None,
                structural_reference_windows=len(analysis.reference_geometry(int(start))['structural_complete_windows']),
                mapping_evaluable=len(fractions),mapping_fraction_min=min(fractions) if fractions else None,mapping_fraction_max=max(fractions) if fractions else None,
                full_reference_window_matches=sum(v==1. for v in fractions),reference_class_in_ties=sum(v.get('top_ties_contain_reference_class') is True for v in mappings),
                OFF_denominator=2,OFF_complete=sum(v['status']=='COMPLETE' for v in row['OFF'])))
    return rows


def run_all(output,source_root=SOURCE_ROOT):
    output,source_root=validate_output(output,source_root);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',source_root=str(source_root),source_video_denominator=6,fragment_denominator=18,marked_denominator=12,receiver_encode_denominator=72,cases={c:empty('NOT_RUN') for c in CASES},fixed_calls={k:2*v for k,v in PLAN.items()},scientific_pass=None)
    dump(output/'manifest.json',load(MANIFEST));dump(output/'result.json',result)
    for case in CASES:
        log=output/(case+'.log');print(case,'crop started; log:',log,flush=True)
        try:
            with log.open('w') as f:child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_holdout_crop_run','--output',str(output/case),'--case-id',case,'--source-root',str(source_root)],stdout=f,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=load(output/case/'result.json')|{'exit_code':child.returncode}
        except Exception as exc:result['cases'][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
        dump(output/'result.json',result)
    result['summary']=summarize(result['cases']);result['compact_summary']=compact_summary(result['summary'])
    result['actual_calls_observed']={k+'_'+s:sum(c.get('actual_calls',{}).get(k+'_'+s,0) for c in result['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(c['status']=='EXECUTION_COMPLETE' and c.get('exit_code')==0 for c in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case-id',choices=CASES);p.add_argument('--source-root',default=str(SOURCE_ROOT));a=p.parse_args()
    r=run_case(a.case_id,a.output,a.source_root) if a.case_id else run_all(a.output,a.source_root)
    if r['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
