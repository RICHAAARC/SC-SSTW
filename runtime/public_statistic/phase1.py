"""Phase1 fixed saved-video pixel perturbation. Run only after explicit approval."""
import json,subprocess,time
from pathlib import Path
from main.sc_sstw.public_luma_statistic import basis,read_rgb

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def decode_command(path,fps,frames):
    return ['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(path),'-map','0:v:0','-vf',f'fps=fps={fps}:start_time=0','-frames:v',str(frames),'-pix_fmt','rgb24','-f','rawvideo','-']
def encode_command(path,w,h,c):
    return ['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','8','-i','pipe:0','-an','-c:v',c['codec']['name'],'-crf',str(c['codec']['crf']),'-pix_fmt',c['codec']['pixel_format'],'-threads','1','-n',str(path)]
def call(command,log,input=None):
    t=time.monotonic()
    try:
        r=subprocess.run(command,input=input,capture_output=True,timeout=120);log.append(dict(command=command,exit_code=r.returncode,seconds=time.monotonic()-t,stderr=r.stderr.decode(errors='replace')))
        if r.returncode:raise RuntimeError('codec command failed')
        return r.stdout
    except Exception as exc:
        log.append(dict(command=command,error=repr(exc),seconds=time.monotonic()-t));raise

def validate_config(c):
    fixed=dict(protocol_id='public_luma_phase1_v2',sample_hz=8,frames=16,segment_seconds=[0,2],arms=['OFF1','OFF2','X_PLUS','X_MINUS','Y_PLUS','Y_MINUS'],amplitude=2/255,quality_rms_budget=3/255,signed_minimum=1/255,crosstalk_maximum=.5/255,minimum_passing_frames=15,maximum_condition=3,minimum_gain=.5,vector_error_fraction=.25,codec=dict(name='libx264',crf=18,pixel_format='yuv420p',threads=1),wall_seconds=600,automatic_retries=0,science_denominator=0)
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('frozen configuration mismatch')
    if [(x.get('id'),x.get('width'),x.get('height'),x.get('source_fps')) for x in c.get('inputs',[])]!=[('P50',512,320,'8/1'),('JumpingJack',320,240,'30000/1001')]:raise ValueError('fixed input roster mismatch')
    if any(not isinstance(x.get('path'),str) or not x['path'] for x in c['inputs']):raise ValueError('source path required')

def perturb(source,arm,amplitude,np):
    if arm not in ('OFF1','OFF2','X_PLUS','X_MINUS','Y_PLUS','Y_MINUS'):raise ValueError('unknown arm')
    h,w=source.shape[1:3];delta=np.zeros((h,w,1),float)
    if arm.startswith('X'):delta=np.broadcast_to(basis(w,np)[None,:,None],(h,w,1))*amplitude
    if arm.startswith('Y'):delta=np.broadcast_to(basis(h,np)[:,None,None],(h,w,1))*amplitude
    if arm.endswith('MINUS'):delta=-delta
    unbounded=source+delta;clipped=np.clip(unbounded,0,1)
    rows=[dict(clipped_channel_fraction=float(((v<0)|(v>1)).mean()),clip_rmse=float(np.sqrt(np.mean((v-z)**2))),clip_maximum=float(np.abs(v-z).max())) for v,z in zip(unbounded,clipped)]
    return clipped,rows

def causal_evaluation(arms,c,np):
    result={};a=c['amplitude']
    for stream in ('preencode','mp4'):
        details={};columns=[]
        for axis in ('X','Y'):
            plus=arms.get(axis+'_PLUS');minus=arms.get(axis+'_MINUS');dim=0 if axis=='X' else 1
            if plus and minus:columns.append((np.asarray(plus[stream]['q'])-np.asarray(minus[stream]['q'])).mean(0)/2)
            for sign in ('PLUS','MINUS'):
                key=axis+'_'+sign;arm=arms.get(key)
                if not arm:details[key]=dict(status='MISSING_FIXED_ARM',passing_frames=0,denominator=16);continue
                q=np.asarray(arm[stream]['q']);by_off={}
                for off in ('OFF1','OFF2'):
                    if off not in arms:by_off[off]=dict(status='MISSING_OFF',passing_frames=0,denominator=16);continue
                    response=q-np.asarray(arms[off][stream]['q']);signed=response[:,dim]*(1 if sign=='PLUS' else -1);cross=np.abs(response[:,1-dim])
                    quality=np.asarray(arm[stream]['per_frame_rmse_source'])
                    passed=(signed>=c['signed_minimum'])&(cross<=c['crosstalk_maximum'])&(quality<=c['quality_rms_budget'])
                    by_off[off]=dict(response_rows=response.tolist(),signed_rows=signed.tolist(),cross_absolute_rows=cross.tolist(),per_frame_pass=passed.tolist(),passing_frames=int(passed.sum()),denominator=16,total_rmse_source=float(arm[stream]['total_rmse_source']),total_quality_within_budget=bool(arm[stream]['total_rmse_source']<=c['quality_rms_budget']),raw_diagonal_diagnostic_pass=bool(passed.sum()>=15 and arm[stream]['total_rmse_source']<=c['quality_rms_budget']))
                details[key]=dict(by_off=by_off)
        if len(columns)==2:
            matrix=np.stack(columns,axis=1);sv=np.linalg.svd(matrix,compute_uv=False);condition=float(sv[0]/sv[1]) if sv[1]>0 else None
            matrix_result=dict(columns=matrix.T.tolist(),singular_values=sv.tolist(),rank=int(np.linalg.matrix_rank(matrix)),condition=condition,raw_diagonal_diagnostic_pass=bool(sv[1]>0 and condition<=3))
        else:matrix_result=dict(status='INCOMPLETE')
        result[stream]=dict(raw_diagonal_diagnostic_only=True,arms=details,response_matrix=matrix_result,visual_quality='PENDING_HUMAN_REVIEW',not_AISB=True)
    return result

def aggregate_evaluation(results,c,np):
    """External descriptive common response; never feeds the public RGB reader.
    No fit/offset/inverse, no good-frame selection when forming C.
    """
    output={};ids=['P50','JumpingJack'];dynamic=['X_PLUS','X_MINUS','Y_PLUS','Y_MINUS'];a=c['amplitude'];budget=c['quality_rms_budget']
    def spectrum(matrix):
        sv=np.linalg.svd(matrix/a,compute_uv=False);k=float(sv[0]/sv[1]) if sv[1]>0 else None
        return dict(singular_values=sv.tolist(),condition=k,passes=bool(sv[1]>=c['minimum_gain'] and k is not None and k<=c['maximum_condition']))
    for stream in ('preencode','mp4'):
        data={};issues=[]
        for sid in ids:
            data[sid]={}
            for arm in c['arms']:
                try:
                    row=results[sid]['arms'][arm][stream];q=np.asarray(row['q'],dtype=float);quality=np.asarray(row['per_frame_rmse_source'],dtype=float);total=float(row['total_rmse_source'])
                    if q.shape!=(16,2) or quality.shape!=(16,) or not np.isfinite(q).all() or not np.isfinite(quality).all() or not np.isfinite(total) or (quality<0).any() or total<0:raise ValueError('shape/nonfinite/negative quality')
                    data[sid][arm]=dict(q=q,quality=quality,total=total)
                except (KeyError,ValueError,TypeError) as exc:issues.append(dict(input_id=sid,arm=arm,reason=str(exc)))
        if issues:
            output[stream]=dict(status='INCOMPLETE',issues=issues,fixed_contents=2,fixed_times_per_content=16,common_C=None,meets_method_gate=False,diagnostic_only=stream=='preencode');continue
        matrices={sid:np.stack([(data[sid]['X_PLUS']['q']-data[sid]['X_MINUS']['q'])/2,(data[sid]['Y_PLUS']['q']-data[sid]['Y_MINUS']['q'])/2],axis=-1) for sid in ids}
        C=np.concatenate([matrices[sid] for sid in ids],axis=0).mean(axis=0)
        global_spectrum=spectrum(C);contents={};all_totals=[]
        for sid in ids:
            rows=[];totals={arm:dict(rmse=data[sid][arm]['total'],passes=bool(data[sid][arm]['total']<=budget)) for arm in c['arms']};all_totals.extend(x['passes'] for x in totals.values())
            for t in range(16):
                local=spectrum(matrices[sid][t]);errors={};quality={};raw={}
                for arm in dynamic:
                    axis=0 if arm.startswith('X') else 1;sign=1 if arm.endswith('PLUS') else -1
                    quality[arm]=dict(rmse=float(data[sid][arm]['quality'][t]),passes=bool(data[sid][arm]['quality'][t]<=budget))
                    for off in ('OFF1','OFF2'):
                        response=data[sid][arm]['q'][t]-data[sid][off]['q'][t];err=response-sign*C[:,axis];norm=float(np.linalg.norm(err));key=arm+'__'+off
                        errors[key]=dict(vector=err.tolist(),norm=norm,passes=bool(norm<=a*c['vector_error_fraction']))
                        raw[key]=dict(response=response.tolist(),own_signed=float(sign*response[axis]),cross_absolute=float(abs(response[1-axis])),near_diagonal_diagnostic_pass=bool(sign*response[axis]>=c['signed_minimum'] and abs(response[1-axis])<=c['crosstalk_maximum']))
                passed=local['passes'] and all(x['passes'] for x in errors.values()) and all(x['passes'] for x in quality.values())
                rows.append(dict(index=t,time=t/8,D=matrices[sid][t].tolist(),spectrum=local,vector_errors=errors,source_quality=quality,raw_diagnostic_only=raw,common_pass=bool(passed)))
            count=sum(x['common_pass'] for x in rows)
            contents[sid]=dict(rows=rows,common_pass_mask=[x['common_pass'] for x in rows],common_passing_frames=count,denominator=16,coverage_pass=bool(count>=15),total_quality=totals,off_repeat_rows=(data[sid]['OFF1']['q']-data[sid]['OFF2']['q']).tolist())
        output[stream]=dict(status='COMPLETE',common_C=C.tolist(),C_denominator=32,common_spectrum=global_spectrum,contents=contents,all_12_total_quality_pass=all(all_totals),meets_method_gate=bool(global_spectrum['passes'] and all(all_totals) and all(x['coverage_pass'] for x in contents.values())),diagnostic_only=stream=='preencode',claim='common additive linear response compatibility relative to each frame OFF; no fitted calibration or noise bound')
    return output

def run(config_path,output):
    import numpy as np
    c=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False);write(out/'config.json',c)
    validate_config(c)
    states={s['id']:{a:dict(status='NOT_EXECUTED') for a in c['arms']} for s in c['inputs']};write(out/'status.json',states)
    log=[];all_results={};start=time.monotonic()
    try:
        for spec in c['inputs']:
            sid=spec['id'];folder=out/sid;folder.mkdir();arms={}
            try:
                probe=json.loads(call(['ffprobe','-v','error','-show_streams','-show_frames','-show_format','-of','json',spec['path']],log));write(folder/'source_probe.json',probe)
                stream=next(s for s in probe['streams'] if s['codec_type']=='video')
                if (stream['width'],stream['height'])!=(spec['width'],spec['height']):raise ValueError('source dimensions changed')
                if any(float(d.get('rotation',0))!=0 for d in stream.get('side_data_list',[])) or float(stream.get('tags',{}).get('rotate',0))!=0:raise ValueError('rotation unsupported')
                raw=call(decode_command(spec['path'],8,16),log);source=np.frombuffer(raw,np.uint8).reshape(16,spec['height'],spec['width'],3).copy()/255
                np.save(folder/'source_sampled_rgb.npy',source,allow_pickle=False)
                write(folder/'source_axis.json',dict(output_times=[i/8 for i in range(16)],segment=[0,2],source_frame_index_not_assumed=True,source_pts_in_probe=True))
            except Exception as exc:
                for arm in c['arms']:states[sid][arm]=dict(status='SOURCE_FAILURE',error=repr(exc),fixed_rows=[dict(index=i,time=i/8,q=None) for i in range(16)])
                write(out/'status.json',states);continue
            for arm in c['arms']:
                dest=folder/arm;dest.mkdir();states[sid][arm]=dict(status='RUNNING');write(out/'status.json',states)
                try:
                    candidate,clipping=perturb(source,arm,c['amplitude'],np);uint=np.rint(candidate*255).astype(np.uint8);actual=uint/255
                    write(dest/'clipping.json',clipping);np.save(dest/'preencode_rgb.npy',uint,allow_pickle=False)
                    pre_q=read_rgb(actual,np);call(encode_command(dest/'saved.mp4',spec['width'],spec['height'],c),log,uint.tobytes())
                    probe=json.loads(call(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time:stream=width,height','-of','json',str(dest/'saved.mp4')],log));write(dest/'saved_probe.json',probe)
                    pts=[float(f['best_effort_timestamp_time']) for f in probe['frames']]
                    if len(pts)!=16 or any(abs(t-i/8)>1e-6 for i,t in enumerate(pts)):raise ValueError('saved PTS mismatch')
                    raw=call(decode_command(dest/'saved.mp4',8,16),log);decoded=np.frombuffer(raw,np.uint8).reshape(uint.shape).copy();np.save(dest/'decoded_rgb.npy',decoded,allow_pickle=False)
                    # Public reader sees only decoded RGB; OFF/control labels enter evaluation later.
                    q=read_rgb(decoded/255,np);record={}
                    for label,x,values in [('preencode',actual,pre_q),('mp4',decoded/255,q)]:
                        record[label]=dict(q=values.tolist(),per_frame_rmse_source=np.sqrt(np.mean((x-source)**2,axis=(1,2,3))).tolist(),total_rmse_source=float(np.sqrt(np.mean((x-source)**2))),worst_frame_rmse_source=float(np.sqrt(np.mean((x-source)**2,axis=(1,2,3))).max()),rows=[dict(index=i,time=i/8,q=values[i].tolist()) for i in range(16)])
                    write(dest/'blind_statistics.json',record);arms[arm]=record
                    states[sid][arm]=dict(status='COMPLETE',rows=16)
                except Exception as exc:states[sid][arm]=dict(status='FAILURE',error=repr(exc),fixed_rows=[dict(index=i,time=i/8,q=None) for i in range(16)]);write(dest/'failure.json',states[sid][arm])
                finally:write(out/'status.json',states);write(out/'processes.json',log)
            # Public statistics of all fixed arms are already persisted.
            for arm,record in arms.items():
                for label,filename in [('preencode','preencode_rgb.npy'),('mp4','decoded_rgb.npy')]:
                    x=np.load(folder/arm/filename)/255;record[label]['rmse_vs_each_off']={}
                    for off in ('OFF1','OFF2'):
                        if off in arms:
                            y=np.load(folder/off/filename)/255;record[label]['rmse_vs_each_off'][off]=dict(total=float(np.sqrt(np.mean((x-y)**2))),per_frame=np.sqrt(np.mean((x-y)**2,axis=(1,2,3))).tolist())
            all_results[sid]=dict(arms=arms,development_causal_evaluation=causal_evaluation(arms,c,np));write(folder/'evaluation.json',all_results[sid])
        aggregate=aggregate_evaluation(all_results,c,np)
        result=dict(aggregate=aggregate,protocol=c['protocol_id'],status='EXECUTED_REQUIRES_VISUAL_REVIEW',fixed_inputs=2,fixed_arms=12,fixed_rows=192,states=states,results=all_results,elapsed_seconds=time.monotonic()-start,science_denominator=0)
        if any(s['status']!='COMPLETE' for arms in states.values() for s in arms.values()):result['status']='INCOMPLETE_FIXED_DENOMINATOR_RETAINED'
        result['all_fixed_rows']=[dict(input_id=sid,arm=arm,index=i,time=i/8,status=state['status'],q=(all_results.get(sid,{}).get('arms',{}).get(arm,{}).get('mp4',{}).get('q',[None]*16)[i])) for sid,aa in states.items() for arm,state in aa.items() for i in range(16)]
        write(out/'result.json',result);return result
    finally:write(out/'processes.json',log);write(out/'status.json',states)
