"""Build the unpublished user-run M05 four-arm trajectory-sync Colab notebook."""
from pathlib import Path
import argparse, ast, json, re

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"notebooks/video_trajectory_payload_framewise_sync_m05_colab.ipynb"
MARKDOWN="""# Wan 32-bit payload + framewise VAE trajectory sync M05

User-run handoff for one fixed saved PAYLOAD_MULTI RGB8 source and one adopted
four-condition comparison. It does not generate Wan video and never replaces
the native Wan decoder. P0 is original RGB and bypasses the 2D decoder. P1 is
deterministic per-frame reconstruction without a sync write. M1 and M05 copy
the same unchanged source encoding independently, target signed projection
margins 1.0 and 0.5, and decode separately. The 2D VAE is pinned to
stabilityai/sd-vae-ft-mse at 31f26fdeee1355a5c34592e401dd41e45d25a493.
Method coordinates are posterior mode times 0.18215 in float32.

Each condition has one full 181-frame MP4 save/read. Crop [1:178] is sliced
after readback with no second codec pass. FULL offset {0} is public geometry,
not synchronization evidence. CROP scores all offsets {0,1,2,3,4}. The blind
receiver uses only received video, key, and public protocol; it does not
receive the condition or target. All candidates, local supports, K0/K1 scores,
ties, erasures, nonfinite values, missing reads, and failures remain. Blind
readouts are saved before truth join. Original payload reading stays R44.

Display order is P1-P0 compatibility first, then same-run M1-P1 versus M05-P1
and direct M05-M1. Synchronization differences are computed only after sealed
blind readouts are joined to truth. No threshold or automatic scientific PASS
exists.

SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time,traceback
if SOURCE_SHA is None:
    raise RuntimeError('UNPUBLISHED_DRAFT: publish reviewed source and rebuild with its immutable SHA before Run all')
if not isinstance(SOURCE_SHA,str) or len(SOURCE_SHA)!=40:
    raise RuntimeError('immutable 40-character source SHA required')
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Trajectory-Payload-Framewise-Sync-M05')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
RUN_OUTPUT=OUTPUT/'fixed_reference'
REPO=Path('/content/SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-'+STAMP)
PYTHON=sys.executable
def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\\n');os.replace(tmp,path)
def failed(stage,exc):
    captured_traceback=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))
    write_json(OUTPUT/'setup_failure.json',dict(stage=stage,error=f'{type(exc).__name__}: {exc}',traceback=captured_traceback,fixed_denominator=FIXED))
def logged(command,stage,cwd=None,check=True):
    child=None;code=None;primary=None;primary_tb=None;cleanup_errors=[]
    def retain_cleanup_error(label,exc):
        nonlocal primary,primary_tb
        if primary is None and not isinstance(exc,Exception):
            primary=exc;primary_tb=exc.__traceback__
        else:
            cleanup_errors.append(label+': '+repr(exc))
    try:
        with (OUTPUT/'execution.log').open('a') as log:
            log.write('COMMAND '+repr(command)+'\\n');log.flush()
            popen_group={'start_new_session':True} if os.name=='posix' else {}
            child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,**popen_group)
            try:
                for line in child.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=child.wait()
            except BaseException as exc:
                primary=exc;primary_tb=exc.__traceback__
            finally:
                if child is not None:
                    if os.name=='posix':
                        pgid=child.pid
                        try:os.killpg(pgid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        except BaseException as exc:retain_cleanup_error('group SIGTERM',exc)
                        try:
                            deadline=time.monotonic()+5.0
                            while time.monotonic()<deadline:
                                try:os.killpg(pgid,0)
                                except ProcessLookupError:break
                                except BaseException as exc:
                                    retain_cleanup_error('group probe',exc);break
                                time.sleep(0.05)
                        except BaseException as exc:
                            retain_cleanup_error('group TERM grace',exc)
                        finally:
                            try:os.killpg(pgid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            except BaseException as exc:retain_cleanup_error('group SIGKILL',exc)
                    elif child.poll() is None:
                        try:child.terminate()
                        except ProcessLookupError:pass
                        except BaseException as exc:retain_cleanup_error('terminate',exc)
                    try:
                        try:code=child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:child.kill()
                            except ProcessLookupError:pass
                            except BaseException as exc:retain_cleanup_error('kill',exc)
                            try:code=child.wait(timeout=10)
                            except BaseException as exc:retain_cleanup_error('wait after kill',exc)
                        except BaseException as exc:retain_cleanup_error('wait',exc)
                    finally:
                        if child.stdout is not None:
                            try:child.stdout.close()
                            except BaseException as exc:retain_cleanup_error('stdout close',exc)
            if primary is None and not cleanup_errors:
                log.write('EXIT '+str(code)+'\\n');log.flush()
    except BaseException as wrapper_error:
        if primary is None:
            primary=wrapper_error;primary_tb=wrapper_error.__traceback__
        elif wrapper_error is not primary:
            cleanup_errors.append('log wrapper/close: '+repr(wrapper_error))
    if primary is None and cleanup_errors:
        primary=RuntimeError('process cleanup failed: '+'; '.join(cleanup_errors))
        primary_tb=primary.__traceback__
    if primary is None and check and code:
        primary=subprocess.CalledProcessError(code,command)
        primary_tb=primary.__traceback__
    if primary is not None:
        if cleanup_errors and hasattr(primary,'add_note'):
            primary.add_note('secondary cleanup errors: '+'; '.join(cleanup_errors))
        try:failed(stage,primary)
        except BaseException as record_error:
            if hasattr(primary,'add_note'):primary.add_note('failure record error: '+repr(record_error))
        raise primary.with_traceback(primary_tb)
    return code
write_json(OUTPUT/'setup_receipt.json',dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED))
"""
ENVIRONMENT="""try:
    logged(['git','clone','--filter=blob:none','https://github.com/RICHAAARC/SC-SSTW.git',str(REPO)],'SOURCE_CLONE')
    logged(['git','-C',str(REPO),'fetch','origin',SOURCE_SHA],'SOURCE_FETCH')
    logged(['git','-C',str(REPO),'checkout','--detach',SOURCE_SHA],'SOURCE_CHECKOUT')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
    if actual!=SOURCE_SHA or dirty:raise RuntimeError('source SHA/clean checkout mismatch')
    import shutil
    if any(shutil.which(name) is None for name in ('ffmpeg','ffprobe')):
        logged(['apt-get','update'],'MEDIA_TOOL_UPDATE',check=False)
        logged(['apt-get','install','-y','ffmpeg'],'MEDIA_TOOL_REPAIR',check=False)
    if logged([PYTHON,'-c','import torch; print(torch.__version__)'],'TORCH_PROBE',check=False):
        logged([PYTHON,'-m','pip','install','--upgrade','torch'],'TORCH_REPAIR')
    probe='from diffusers import AutoencoderKL,AutoencoderKLWan; print(AutoencoderKL.__name__,AutoencoderKLWan.__name__)'
    if logged([PYTHON,'-c',probe],'VAE_IMPORT_PROBE',check=False):
        logged([PYTHON,'-m','pip','install','diffusers>=0.35.1,<0.40','transformers>=4.45,<5','accelerate>=0.34','safetensors>=0.4.5'],'VAE_DEPENDENCY_REPAIR')
    dependency_code=logged([PYTHON,'-m','pip','check'],'DEPENDENCY_REPORT',check=False)
    (OUTPUT/'environment_freeze.txt').write_text(subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True))
    write_json(OUTPUT/'source_receipt.json',dict(source_sha=actual,clean=True))
    write_json(OUTPUT/'setup_receipt.json',dict(status='SETUP_COMPLETE',source_sha=actual,fixed_denominator=FIXED,dependency_check_returncode=dependency_code))
except Exception as exc:
    try:failed('SOURCE_OR_ENVIRONMENT',exc)
    except BaseException as record_error:
        if hasattr(exc,'add_note'):exc.add_note('failure record error: '+repr(record_error))
    raise
"""
RUN="""try:
    from google.colab import userdata
    token=userdata.get('HF_TOKEN')
    if token:os.environ['HF_TOKEN']=token
except Exception:pass
finally:token=None
command=[PYTHON,'-u','-m','experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run','--output',str(RUN_OUTPUT)]
try:
    returncode=logged(command,'FIXED_RUN',cwd=REPO,check=False)
    RESULT_PATH=RUN_OUTPUT/'result.json'
    write_json(OUTPUT/'execution_receipt.json',dict(command=command,returncode=returncode,result_path=str(RESULT_PATH),result_exists=RESULT_PATH.is_file()))
    if not RESULT_PATH.is_file():raise RuntimeError('No runner result; fixed denominator retained')
except Exception as exc:
    try:failed('RUNNER',exc)
    except BaseException as record_error:
        if hasattr(exc,'add_note'):exc.add_note('failure record error: '+repr(record_error))
    raise
"""
DISPLAY="""import statistics
result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'],'Calls:',result['calls'])
print('Writers:')
for condition,row in result['writers'].items():
    print(condition,'target=',row.get('target_margin'),'raw/applied=',row.get('raw_target_delta_l2'),row.get('applied_float32_delta_l2'),'path=',row.get('path'))
def truth_joined_rows(group):
    joined=[]
    for name,row in sorted(result[group].items()):
        oid,key_id=name.rsplit('/',1)
        observation=result['observations'][oid]
        value=dict(row)
        value.setdefault('condition',observation['condition_truth_join_only'])
        value.setdefault('view',observation['view_truth_join_only'])
        value.setdefault('key_role','REGISTERED' if key_id=='K0' else 'WRONG_KEY')
        joined.append((name,value))
    return joined
payload_rows=truth_joined_rows('payload_posthoc')
sync_rows=truth_joined_rows('sync_posthoc')
def payload_vote_margins(name):
    votes=result.get('payload_reads',{}).get(name,{}).get('votes')
    if not isinstance(votes,list) or not votes:return None,None
    margins=[]
    for vote in votes:
        if not isinstance(vote,dict) or 'ones' not in vote or 'zeros' not in vote:return None,None
        margins.append(abs(int(vote['ones'])-int(vote['zeros'])))
    return min(margins),statistics.median(margins)
print('=== P1-P0 compatibility first ===')
for name,row in payload_rows:
    if row['condition'] in ('P0_ORIGINAL_RGB','P1_FRAMEWISE_RECON'):
        vote_min,vote_median=payload_vote_margins(name)
        print('Payload compatibility:',name,row['status'],'key=',row['key_role'],'errors=',row.get('bit_errors'),'exact=',row.get('exact_payload'),'vote_min=',vote_min,'vote_median=',vote_median,'error=',row.get('error'))
for space in ('PRECODEC','FULL_MP4','CROP_MP4'):
    qid=space+'/P1_FRAMEWISE_RECON_vs_P0_ORIGINAL_RGB'
    print('Quality compatibility:',qid,result['quality'][qid])
print('=== Same-run incremental quality ===')
for space in ('PRECODEC','FULL_MP4','CROP_MP4'):
    for pair in (
        'M1_FRAMEWISE_SYNC_vs_P1_FRAMEWISE_RECON',
        'M05_FRAMEWISE_SYNC_vs_P1_FRAMEWISE_RECON',
        'M05_FRAMEWISE_SYNC_vs_M1_FRAMEWISE_SYNC',
    ):
        qid=space+'/'+pair;print('Quality:',qid,result['quality'][qid])
print('=== Truth-joined CROP sync scores and differences ===')
sync_index={(row['condition'],row['view'],row['key_role']):row for _,row in sync_rows}
for key_role in ('REGISTERED','WRONG_KEY'):
    p1=sync_index.get(('P1_FRAMEWISE_RECON','CROP',key_role),{})
    m1=sync_index.get(('M1_FRAMEWISE_SYNC','CROP',key_role),{})
    m05=sync_index.get(('M05_FRAMEWISE_SYNC','CROP',key_role),{})
    def score(row):return row.get('true_score')
    def difference(left,right):
        a=score(left);b=score(right)
        return None if a is None or b is None else a-b
    print(key_role,'P1=',score(p1),'M1=',score(m1),'M05=',score(m05),
          'M1-P1=',difference(m1,p1),'M05-P1=',difference(m05,p1),
          'M05-M1=',difference(m05,m1))
for name,row in sync_rows:
    if row['view']=='CROP':
        print('CROP sync:',name,row['status'],'condition=',row['condition'],'key=',row['key_role'],'score=',row.get('true_score'),'rank=',row.get('true_rank'),'top=',row.get('truth_in_top'),'unique=',row.get('unique_truth'),'error=',row.get('error'))
for name,row in sync_rows:
    if row['view']=='FULL':
        print('FULL geometry only:',name,row['status'],'condition=',row['condition'],'key=',row['key_role'],'score=',row.get('true_score'),'error=',row.get('error'))
print('=== All payload rows, including K1 ===')
for name,row in payload_rows:
    vote_min,vote_median=payload_vote_margins(name)
    print('Payload:',name,row['status'],'condition=',row['condition'],'view=',row['view'],'key=',row['key_role'],'errors=',row.get('bit_errors'),'exact=',row.get('exact_payload'),'vote_min=',vote_min,'vote_median=',vote_median,'error=',row.get('error'))
from IPython.display import Video,display
for condition,row in result['transport'].items():
    video=row.get('events',{}).get('mp4',{});path=video.get('path')
    if video.get('status')=='SAVED' and isinstance(path,str) and Path(path).is_file():
        try:print('MP4:',condition);display(Video(str(path),embed=True))
        except Exception as preview_error:print('Preview unavailable:',preview_error)
print('Failures:',result['failures']);print('Evidence ceiling:',result['evidence_ceiling']);print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):raise ValueError("published immutable source SHA required")
    fixed=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_v1.json").read_text())["fixed_denominator"]
    note=("Published source: "+source_sha+"." if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None. Publish reviewed source and rebuild before Run all.")
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),("markdown",MARKDOWN.replace("SOURCE_NOTE",note)),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+SETUP),("code",ENVIRONMENT),("code",RUN),("code",DISPLAY)]
    cells=[]
    for index,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"trajectory-payload-framewise-sync-m05-{index}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),candidate_binding=dict(candidate="trajectory-payload-framewise-sync-m05",source_sha=source_sha,status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",execution="user-run only after immutable SHA binding; no agent real model execution")))
    output.write_text(json.dumps(value,indent=1)+"\n",encoding="utf-8");return output

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT);args=parser.parse_args();print(build(args.source_sha,args.output))
