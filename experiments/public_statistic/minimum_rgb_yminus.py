"""One minimum-RGB-energy Y_MINUS writer experiment. No original protocol edits."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from main.sc_sstw.public_luma_statistic import basis,read_rgb

A=2/255;BUDGET=3/255

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def candidate(source,np):
    w=np.array([.2126,.7152,.0722]);delta=-A*basis(source.shape[1],np)[None,:,None,None]*w[None,None,None,:]/(w@w)
    unbounded=source+delta;bounded=np.clip(unbounded,0,1);pixels=np.rint(bounded*255).astype(np.uint8)
    clipping=[dict(index=i,clipped_channels=int(((v<0)|(v>1)).sum()),clipped_channel_fraction=float(((v<0)|(v>1)).mean()),clipping_rms=float(np.sqrt(np.mean((v-b)**2))),clipping_max=float(abs(v-b).max()),quantization_rms=float(np.sqrt(np.mean((p/255-b)**2)))) for i,(v,b,p) in enumerate(zip(unbounded,bounded,pixels))]
    return pixels,clipping

def evaluate(source,pre,decoded,offs,C,np):
    qs=read_rgb(source,np);qp=read_rgb(pre,np);qd=read_rgb(decoded,np)
    W=pre-source;K=decoded-pre
    source_errors=np.sqrt(np.mean((decoded-source)**2,axis=(1,2,3)));total=float(np.sqrt(np.mean((decoded-source)**2)))
    rows=[]
    for i in range(len(source)):
        by_off={}
        for name,off in offs.items():
            error=qd[i]-read_rgb(off[i:i+1],np)[0]+C[:,1]
            by_off[name]=dict(vector_error=error.tolist(),vector_norm=float(np.linalg.norm(error)),passes=bool(np.linalg.norm(error)<=A/4),rgb_rmse=float(np.sqrt(np.mean((decoded[i]-off[i])**2))))
        passed=source_errors[i]<=BUDGET and all(v['passes'] for v in by_off.values())
        rows.append(dict(index=i,time=i/8,q_source=qs[i].tolist(),q_preencode=qp[i].tolist(),q_decoded=qd[i].tolist(),source_rmse=float(source_errors[i]),source_quality_pass=bool(source_errors[i]<=BUDGET),by_off=by_off,common_pass=bool(passed),writer_mse=float(np.mean(W[i]**2)),codec_mse=float(np.mean(K[i]**2)),twice_cross=float(2*np.mean(W[i]*K[i])),total_mse=float(np.mean((W[i]+K[i])**2))))
    return dict(rows=rows,common_pass_mask=[x['common_pass'] for x in rows],common_passing_frames=sum(x['common_pass'] for x in rows),fixed_rows=16,total_source_rmse=total,total_quality_pass=bool(total<=BUDGET),single_arm_gate=bool(len(rows)==16 and sum(x['common_pass'] for x in rows)>=15 and total<=BUDGET),q_preencode=qp.tolist(),q_decoded=qd.tolist(),off_total_rmse={k:float(np.sqrt(np.mean((decoded-v)**2))) for k,v in offs.items()},decomposition=dict(writer_mse=float(np.mean(W**2)),codec_mse=float(np.mean(K**2)),twice_cross=float(2*np.mean(W*K)),total_mse=float(np.mean((W+K)**2))))

def worker(old,out):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False);old=Path(old);started=time.monotonic();log=[]
    result=dict(status='INCOMPLETE',fixed_rows=16,rows=[dict(index=i,time=i/8,q=None,status='NOT_EXECUTED') for i in range(16)],encoder_attempts=0,science_denominator=0)
    write(out/'result.json',result)
    write(out/'frozen.json',dict(formula='deltaRGB=-(2/255)*hy*w/(w dot w)',w=[.2126,.7152,.0722],amplitude=A,quality=BUDGET,vector_error=A/4,minimum_common_frames=15,C='reuse original result.aggregate.mp4.common_C without re-estimation',old=str(old),frames=16,width=320,height=240,fps=8,codec='libx264 crf18 yuv420p threads1',wall_seconds=120,command_seconds=60,attempts=1))
    def call(cmd,stdin=None):
        index=len(log);t=time.monotonic();entry=dict(command=cmd,status='RUNNING');log.append(entry);write(out/'processes.json',log)
        try:
            proc=subprocess.run(cmd,input=stdin,capture_output=True,timeout=60)
            (out/f'command_{index}.stdout').write_bytes(proc.stdout);(out/f'command_{index}.stderr').write_bytes(proc.stderr)
            entry.update(status='COMPLETE',exit_code=proc.returncode,seconds=time.monotonic()-t)
            if proc.returncode:raise RuntimeError('subprocess failed')
            return proc.stdout
        except Exception as exc:entry.update(error=repr(exc),seconds=time.monotonic()-t);raise
        finally:write(out/'processes.json',log)
    def load(path,normalized=False):
        x=np.load(path,allow_pickle=False)
        if x.shape!=(16,240,320,3) or not np.isfinite(x).all():raise ValueError('fixed array shape/finite')
        if normalized:
            if not np.issubdtype(x.dtype,np.floating) or x.min()<0 or x.max()>1:raise ValueError('source normalized RGB')
            return x
        if x.dtype!=np.uint8:raise ValueError('expected original uint8 RGB')
        return x/255
    try:
        base=old/'JumpingJack';source=load(base/'source_sampled_rgb.npy',True);previous_pre=load(base/'Y_MINUS/preencode_rgb.npy');previous_decoded=load(base/'Y_MINUS/decoded_rgb.npy')
        offs={k:load(base/k/'decoded_rgb.npy') for k in ('OFF1','OFF2')};record=json.loads((old/'result.json').read_text());C=np.asarray(record['aggregate']['mp4']['common_C'],float)
        if C.shape!=(2,2) or not np.isfinite(C).all():raise ValueError('original common C unavailable')
        write(out/'reused_C.json',dict(common_C=C.tolist(),source='original 32-point common C',reestimated=False))
        pre,clip=candidate(source,np);np.save(out/'preencode_rgb.npy',pre,allow_pickle=False);write(out/'clipping.json',clip)
        cmd=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','320x240','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','1','-n',str(out/'saved.mp4')]
        result.update(status='RUNNING',encoder_attempts=1);write(out/'result.json',result);call(cmd,pre.tobytes())
        probe=json.loads(call(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time:stream=width,height','-of','json',str(out/'saved.mp4')]));write(out/'probe.json',probe)
        times=[float(x['best_effort_timestamp_time']) for x in probe['frames']]
        if len(times)!=16 or any(abs(t-i/8)>1e-6 for i,t in enumerate(times)):raise ValueError('output PTS mismatch')
        raw=call(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(out/'saved.mp4'),'-map','0:v:0','-vf','fps=fps=8:start_time=0','-frames:v','16','-pix_fmt','rgb24','-f','rawvideo','-'])
        decoded=np.frombuffer(raw,np.uint8).reshape(16,240,320,3).copy();np.save(out/'decoded_rgb.npy',decoded,allow_pickle=False)
        new=evaluate(source,pre/255,decoded/255,offs,C,np);previous=evaluate(source,previous_pre,previous_decoded,offs,C,np)
        write(out/'old_yminus_reused_evaluation.json',previous)
        result.update(new);result.update(status='COMPLETE_REQUIRES_INTERPRETATION',old_new_rows=[dict(index=i,old=previous['rows'][i],new=new['rows'][i]) for i in range(16)],elapsed_seconds=time.monotonic()-started)
    except Exception as exc:result.update(status='INCOMPLETE',error=repr(exc),elapsed_seconds=time.monotonic()-started)
    finally:write(out/'result.json',result);write(out/'processes.json',log)
    return 0 if result['status'].startswith('COMPLETE') else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:raise SystemExit(worker(args.old,args.output))
    out=Path(args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);log=out.parent/(out.name+'.log');exitfile=out.parent/(out.name+'.exit.json')
    if any(x.exists() for x in (out,log,exitfile)):raise FileExistsError('preserve old attempt')
    cmd=[sys.executable,'-m','experiments.public_statistic.minimum_rgb_yminus','--worker','--old',str(Path(args.old).resolve()),'--output',str(out)];start=time.monotonic();timeout=False
    with log.open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=120)
        except subprocess.TimeoutExpired:timeout=True;os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    write(exitfile,dict(command=cmd,exit_code=code,timed_out=timeout,seconds=time.monotonic()-start,attempts=1,fixed_rows=16))
    if timeout:
        record=dict(status='INCOMPLETE_TIMEOUT',fixed_rows=16,rows=[dict(index=i,time=i/8,q=None) for i in range(16)])
        if out.exists():write(out/'timeout_status.json',record)
    if code:raise SystemExit(124 if timeout else 1)
if __name__=='__main__':main()
