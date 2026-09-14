"""Fixed seven new minimum-energy arms plus five explicit reused arms."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from main.sc_sstw.public_luma_statistic import basis,read_rgb
from runtime.public_statistic.phase1 import aggregate_evaluation,validate_config
A=2/255
ARMS=['OFF1','OFF2','X_PLUS','X_MINUS','Y_PLUS','Y_MINUS']
IDS=['P50','JumpingJack']

def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def roster(old,single):
    rows=[]
    for sid in IDS:
        for arm in ARMS:
            reused=arm.startswith('OFF') or (sid=='JumpingJack' and arm=='Y_MINUS')
            source=str(Path(single) if sid=='JumpingJack' and arm=='Y_MINUS' else Path(old)/sid/arm) if reused else None
            rows.append(dict(input_id=sid,arm=arm,mode='REUSED' if reused else 'NEW',source=source,frames=16,previously_seen_development=bool(reused)))
    return rows

def perturb(source,arm,np):
    if arm not in ARMS[2:]:raise ValueError('fixed dynamic arm')
    w=np.array([.2126,.7152,.0722]);h,wid=source.shape[1:3]
    field=basis(wid,np)[None,None,:,None] if arm.startswith('X') else basis(h,np)[None,:,None,None]
    sign=1 if arm.endswith('PLUS') else -1;unbounded=source+sign*A*field*w/(w@w);bounded=np.clip(unbounded,0,1);pixels=np.rint(255*bounded).astype(np.uint8)
    clip=[dict(index=i,clipped_channels=int(((x<0)|(x>1)).sum()),clipped_channel_fraction=float(((x<0)|(x>1)).mean()),clipping_rms=float(np.sqrt(np.mean((x-y)**2))),clipping_maximum=float(abs(x-y).max()),quantization_rms=float(np.sqrt(np.mean((z/255-y)**2)))) for i,(x,y,z) in enumerate(zip(unbounded,bounded,pixels))]
    return pixels,clip

def statistics(source,pre,decoded,np):
    record={}
    for label,x in [('preencode',pre),('mp4',decoded)]:
        q=read_rgb(x,np);err=np.sqrt(np.mean((x-source)**2,axis=(1,2,3)))
        record[label]=dict(q=q.tolist(),per_frame_rmse_source=err.tolist(),total_rmse_source=float(np.sqrt(np.mean((x-source)**2))),worst_frame_rmse_source=float(err.max()),rows=[dict(index=i,time=i/8,q=q[i].tolist()) for i in range(16)])
    W=pre-source;K=decoded-pre
    record['decomposition']=dict(writer_mse=float(np.mean(W**2)),codec_mse=float(np.mean(K**2)),twice_cross=float(2*np.mean(W*K)),total_mse=float(np.mean((decoded-source)**2)),rows=[dict(index=i,writer_mse=float(np.mean(W[i]**2)),codec_mse=float(np.mean(K[i]**2)),twice_cross=float(2*np.mean(W[i]*K[i])),total_mse=float(np.mean((decoded[i]-source[i])**2))) for i in range(16)])
    return record

def old_C_comparison(records,oldC,np,original_records=None):
    output={}
    for sid in IDS:
        output[sid]={};arms=records.get(sid,{}).get('arms',{})
        for arm in ARMS[2:]:
            if arm not in arms or any(off not in arms for off in ARMS[:2]):output[sid][arm]=dict(status='INCOMPLETE',rows=[dict(index=i,time=i/8,error=None) for i in range(16)]);continue
            axis=0 if arm.startswith('X') else 1;sign=1 if arm.endswith('PLUS') else -1;q=np.asarray(arms[arm]['mp4']['q']);rows=[]
            for i in range(16):
                values={}
                for off in ARMS[:2]:
                    response=q[i]-np.asarray(arms[off]['mp4']['q'][i]);error=response-sign*oldC[:,axis]
                    values[off]=dict(actual_response=response.tolist(),error_old_C=error.tolist(),norm=float(np.linalg.norm(error)),within_original_tolerance=bool(np.linalg.norm(error)<=A/4))
                    if original_records is not None:
                        try:
                            oldarms=original_records[sid]['arms'];oldresponse=np.asarray(oldarms[arm]['mp4']['q'][i])-np.asarray(oldarms[off]['mp4']['q'][i]);values[off].update(old_equal_rgb_response=oldresponse.tolist(),new_minus_old_response=(response-oldresponse).tolist())
                        except (KeyError,IndexError,ValueError):values[off]['old_response_status']='UNAVAILABLE'
                rows.append(dict(index=i,time=i/8,by_off=values))
            output[sid][arm]=dict(status='COMPLETE',rows=rows,old_C_common_compatibility_mask=[all(x['within_original_tolerance'] for x in row['by_off'].values()) for row in rows],mean_response_by_off={off:np.mean([row['by_off'][off]['actual_response'] for row in rows],axis=0).tolist() for off in ARMS[:2]},old_comparison_diagnostic_only=True,mean_old_equal_rgb_response_by_off={off:(np.mean([row['by_off'][off]['old_equal_rgb_response'] for row in rows],axis=0).tolist() if all('old_equal_rgb_response' in row['by_off'][off] for row in rows) else None) for off in ARMS[:2]})
    return output

def worker(old,single,config,out):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'entry_source_snapshot.py').write_bytes(Path(__file__).read_bytes());old=Path(old);single=Path(single);c=json.loads(Path(config).read_text());validate_config(c)
    manifest=roster(old,single);write(out/'manifest.json',dict(rows=manifest,new_arms=7,reused_arms=5,new_rows=112,total_rows=192,formula='sign*(2/255)*h_axis*w/(w dot w)',w=[.2126,.7152,.0722],new_C='all32 D using frozen v2',old_C='original phase1_cpu_run01 read only',wall_seconds=600,command_seconds=120,attempts=1,codec=c['codec'],config=c))
    status={s:{a:dict(status='NOT_EXECUTED') for a in ARMS} for s in IDS};records={s:dict(arms={}) for s in IDS};log=[];start=time.monotonic();encoder_attempts=0
    def persist():write(out/'status.json',status);write(out/'processes.json',log)
    persist()
    write(out/'fixed_slots.json',[dict(input_id=s,arm=a,index=i,time=i/8) for s in IDS for a in ARMS for i in range(16)])
    def call(cmd,dest,stdin=None):
        n=len(log);entry=dict(command=cmd,status='RUNNING');log.append(entry);persist();t=time.monotonic()
        try:
            p=subprocess.run(cmd,input=stdin,capture_output=True,timeout=120)
            (dest/f'command_{n}.stdout').write_bytes(p.stdout);(dest/f'command_{n}.stderr').write_bytes(p.stderr)
            entry.update(status='COMPLETE',exit_code=p.returncode,seconds=time.monotonic()-t)
            if p.returncode:raise RuntimeError('subprocess failure; no retry')
            return p.stdout
        except Exception as e:entry.update(error=repr(e),seconds=time.monotonic()-t);raise
        finally:persist()
    def load(path,shape,source=False):
        x=np.load(path,allow_pickle=False)
        if x.shape!=shape or not np.isfinite(x).all():raise ValueError('array shape/finite')
        if source:
            if not np.issubdtype(x.dtype,np.floating) or x.min()<0 or x.max()>1:raise ValueError('normalized source')
            return x
        if x.dtype!=np.uint8:raise ValueError('uint8 artifact required')
        return x/255
    oldC=None;original_records=None
    try:
        old_result=json.loads((old/'result.json').read_text());oldC=np.asarray(old_result['aggregate']['mp4']['common_C'],float);original_records=old_result.get('results')
        if oldC.shape!=(2,2) or not np.isfinite(oldC).all():raise ValueError('original C unavailable')
        write(out/'old_C.json',dict(C=oldC.tolist(),source=str(old/'result.json'),reestimated=False))
    except Exception as e:write(out/'old_C_failure.json',dict(error=repr(e)))
    for spec in c['inputs']:
        sid=spec['id'];h=spec['height'];w=spec['width'];shape=(16,h,w,3);base=old/sid;folder=out/sid;folder.mkdir();arrays={}
        try:source=load(base/'source_sampled_rgb.npy',shape,True)
        except Exception as e:
            for arm in ARMS:status[sid][arm]=dict(status='SOURCE_FAILURE',error=repr(e))
            persist();continue
        for row in [x for x in manifest if x['input_id']==sid]:
            arm=row['arm'];dest=folder/arm;dest.mkdir();status[sid][arm]=dict(status='RUNNING',mode=row['mode']);persist()
            try:
                if row['mode']=='REUSED':
                    origin=Path(row['source']);pre=load(origin/'preencode_rgb.npy',shape);decoded=load(origin/'decoded_rgb.npy',shape)
                    expected=np.rint(source*255).astype(np.uint8) if arm.startswith('OFF') else perturb(source,arm,np)[0]
                    if not np.array_equal(np.rint(pre*255).astype(np.uint8),expected):raise ValueError('reused preencode/source/formula mismatch')
                    write(dest/'reused.json',dict(source=str(origin),mode='REUSED_NO_ENCODING_NO_MEDIA_COPY',previously_seen_development=True))
                else:
                    pixels,clip=perturb(source,arm,np);pre=pixels/255;np.save(dest/'preencode_rgb.npy',pixels,allow_pickle=False);write(dest/'clipping.json',clip)
                    cmd=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','1','-n',str(dest/'saved.mp4')]
                    encoder_attempts+=1;write(out/'encoder_attempts.json',dict(count=encoder_attempts,maximum=7));call(cmd,dest,pixels.tobytes())
                    probe=json.loads(call(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time:stream=width,height','-of','json',str(dest/'saved.mp4')],dest));write(dest/'probe.json',probe)
                    times=[float(x['best_effort_timestamp_time']) for x in probe['frames']]
                    if len(times)!=16 or any(abs(t-i/8)>1e-6 for i,t in enumerate(times)):raise ValueError('PTS mismatch')
                    raw=call(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(dest/'saved.mp4'),'-map','0:v:0','-vf','fps=fps=8:start_time=0','-frames:v','16','-pix_fmt','rgb24','-f','rawvideo','-'],dest)
                    pixels=np.frombuffer(raw,np.uint8).reshape(shape).copy();np.save(dest/'decoded_rgb.npy',pixels,allow_pickle=False);decoded=pixels/255
                record=statistics(source,pre,decoded,np);record['identity']=row;write(dest/'statistics.json',record);records[sid]['arms'][arm]=record;arrays[arm]=(pre,decoded)
                status[sid][arm]=dict(status='COMPLETE',mode=row['mode'],rows=16)
            except Exception as e:status[sid][arm]=dict(status='FAILURE',mode=row['mode'],error=repr(e));write(dest/'failure.json',status[sid][arm])
            finally:persist()
        for arm,record in records[sid]['arms'].items():
            for j,label in enumerate(('preencode','mp4')):
                record[label]['rmse_vs_each_off']={}
                for off in ARMS[:2]:
                    if off in arrays:
                        diff=arrays[arm][j]-arrays[off][j];record[label]['rmse_vs_each_off'][off]=dict(total=float(np.sqrt(np.mean(diff**2))),per_frame=np.sqrt(np.mean(diff**2,axis=(1,2,3))).tolist())
        write(folder/'evaluation_inputs.json',records[sid])
    aggregate=aggregate_evaluation(records,c,np)
    result=dict(status='COMPLETE_REQUIRES_INTERPRETATION' if all(x['status']=='COMPLETE' for aa in status.values() for x in aa.values()) else 'INCOMPLETE',fixed_arms=12,new_arms=7,reused_arms=5,fixed_rows=192,new_rows=112,encoder_attempts=encoder_attempts,states=status,results=records,aggregate=aggregate,old_C_comparison=old_C_comparison(records,oldC,np,original_records) if oldC is not None else dict(status='INCOMPLETE_OLD_C'),elapsed_seconds=time.monotonic()-start,science_denominator=0)
    result['all_fixed_rows']=[dict(input_id=s,arm=a,index=i,time=i/8,status=status[s][a]['status'],q=records[s]['arms'].get(a,{}).get('mp4',{}).get('q',[None]*16)[i]) for s in IDS for a in ARMS for i in range(16)]
    write(out/'result.json',result);persist();return 0 if result['status'].startswith('COMPLETE') else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--single',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:raise SystemExit(worker(a.old,a.single,a.config,a.output))
    out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);log=out.parent/(out.name+'.log');ex=out.parent/(out.name+'.exit.json')
    if any(x.exists() for x in (out,log,ex)):raise FileExistsError('prior attempt preserved')
    cmd=[sys.executable,'-m','experiments.public_statistic.minimum_rgb_complete','--worker','--old',str(Path(a.old).resolve()),'--single',str(Path(a.single).resolve()),'--config',str(Path(a.config).resolve()),'--output',str(out)];start=time.monotonic();timeout=False
    with log.open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=600)
        except subprocess.TimeoutExpired:timeout=True;os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    write(ex,dict(command=cmd,exit_code=code,timed_out=timeout,elapsed_seconds=time.monotonic()-start,new_arms=7,reused_arms=5,fixed_rows=192,new_rows=112,automatic_retries=0))
    if timeout and out.exists():write(out/'timeout_status.json',dict(status='INCOMPLETE_TIMEOUT',fixed_rows=192,new_rows=112,slots=[dict(input_id=s,arm=arm,index=i,time=i/8,q=None) for s in IDS for arm in ARMS for i in range(16)]))
    if code:raise SystemExit(124 if timeout else 1)
if __name__=='__main__':main()
