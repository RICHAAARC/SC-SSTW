"""Build fixed video-reference Colab; published source binds later."""
from pathlib import Path
import argparse
import ast
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"notebooks/video_temporal_sync_bridge_v1_colab.ipynb"


def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}",source_sha):
        raise ValueError("immutable published source SHA required")
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,metadata={},id=f"temporal-sync-{len(cells)}",source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add("code","from google.colab import drive\ndrive.mount('/content/drive')\n")
    add("markdown","""# Video-Temporal-Sync-Bridge-V1

One copper-kettle source, three independent complete trajectories:
OFF / PAYLOAD_LAST / PILOT_LAST. LAST49 uses the original payload on channels
0..3; PILOT_LAST additionally writes the finite temporal pilot on channel4.
N=44160/47104, eta=5520/5888; conditional per-coefficient strength is preserved,
not total energy. No denoiser/tail gradient or history reset.

Three source MP4s each yield matched second-codec FULL_RESAVED181 and CROP5_129.
The blind receiver uses only observed length, public key/protocol and frozen
VAE: one full phase or four actual RGB crop phases. Soft pilot cosine chooses
among1/53 fixed candidates, threshold0.5 and absolute tie tolerance1e-12.
Normal NO_PILOT/AMBIGUOUS rejection is a completed negative result. Payload
never chooses phase or resolves ties. Full localization is trivial.

The fixed roster is3 sources,6 derived MP4s,15 normalized phase tensors,
30 phase payload reads,324 pilot candidates,12 searches and24 posthoc rows.
All raw observations precede truth evaluation; missing rows remain. Two
positive and10 negative pilot controls are dependent, not independent samples.
Oracle and fixed-g0 payload are diagnostics only; quality has no threshold.

Run all uses current Python and serial fresh children, with no venv, exact
Python-version or GPU-model gate. Model generation ends before FP32 VAE load.
No implementation-agent GPU/Colab execution. All outputs use one new Drive
timestamp directory. SOURCE_SHA is intentionally unset until publication.

"""+("Published source bound.\n" if source_sha else "UNPUBLISHED DRAFT.\n"))
    add("code",f"SOURCE_SHA = {source_sha!r}\n"+r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Temporal-Sync-Bridge-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
REPO=Path('/content/SC-SSTW-TEMPORAL-SYNC-'+STAMP)
RUN_OUTPUT=OUTPUT/'fixed_reference'
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_LAST','PILOT_LAST');VIEWS=('FULL_RESAVED181','CROP5_129');KEYS=('CORRECT','WRONG')
FIXED=dict(source_cases=1,arms=3,source_mp4=3,derived_mp4=6,normalized_phase_features=15,
    phase_payload_reads=30,primary_searches=12,pilot_candidates=324,posthoc_evaluations=24,pilot_positives=2,pilot_negatives=10)
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
    terminal_diagnostics={a+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for k in KEYS},
    generation={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},sources={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
    views={a+'/'+v:{'status':'NOT_RUN_SETUP'} for a in ARMS for v in VIEWS},
    normalized={a+'/'+v+'/'+str(g):{'status':'NOT_RUN_SETUP'} for a in ARMS for v in VIEWS for g in ([0] if v==VIEWS[0] else range(4))},
    phase_reads={a+'/'+v+'/'+str(g)+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for v in VIEWS for g in ([0] if v==VIEWS[0] else range(4)) for k in KEYS},
    candidates={a+'/'+v+'/'+k+'/'+str(b):{'status':'NOT_RUN_SETUP'} for a in ARMS for v in VIEWS for k in KEYS for b in ([0] if v==VIEWS[0] else range(53))},
    searches={a+'/'+v+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for v in VIEWS for k in KEYS},
    evaluations={a+'/'+v+'/'+k+'/'+t:{'status':'NOT_RUN_SETUP'} for a in ARMS for v in VIEWS for k in KEYS for t in ('REGISTERED','WRONG_MESSAGE')})
def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def failed(stage,exc):
    setup.update(status='SETUP_FAILED',stage=stage,error=f'{type(exc).__name__}: {exc}')
    write_json(OUTPUT/'setup_receipt.json',setup)
    write_json(OUTPUT/'setup_failure.json',dict(stage=stage,error=str(exc),traceback=traceback.format_exc()))
def logged(command,stage,cwd=None,check=True):
    try:
        with (OUTPUT/'execution.log').open('a') as log:
            log.write('COMMAND '+repr(command)+'\n');log.flush()
            child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait();log.write('EXIT '+str(code)+'\n')
        if check and code:raise subprocess.CalledProcessError(code,command)
        return code
    except Exception as exc:failed(stage,exc);raise
write_json(OUTPUT/'setup_receipt.json',setup)
print('Fixed output:',OUTPUT)
if SOURCE_SHA is None:
    error=RuntimeError('Unpublished draft: bind verified published source SHA before Run all')
    failed('SOURCE_BINDING',error);raise error
''')
    add("code",r'''try:
    logged(['git','clone','--filter=blob:none','https://github.com/RICHAAARC/SC-SSTW.git',str(REPO)],'SOURCE_CLONE')
    logged(['git','-C',str(REPO),'fetch','origin',SOURCE_SHA],'SOURCE_FETCH')
    logged(['git','-C',str(REPO),'checkout','--detach',SOURCE_SHA],'SOURCE_CHECKOUT')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
    if actual!=SOURCE_SHA or dirty:raise RuntimeError('source SHA/clean checkout mismatch')
    write_json(OUTPUT/'source_receipt.json',dict(source_sha=actual,clean=True))
    torch_ok=logged([PYTHON,'-c','import torch,torchvision; print(torch.__version__,torchvision.__version__)'],'TORCH_PROBE',check=False)==0
    if not torch_ok:
        logged([PYTHON,'-m','pip','install','--upgrade','torch','torchvision'],'TORCH_REPAIR')
    logged([PYTHON,'-m','pip','install','-r',str(REPO/'experiments/wan_state_clock/requirements-grow-video-reference.txt')],'DEPENDENCIES')
    dependency_code=logged([PYTHON,'-m','pip','check'],'DEPENDENCY_REPORT',check=False)
    setup['dependency_check_returncode']=dependency_code
    if dependency_code:print('Metadata conflicts retained; actual imports determine availability.')
    (OUTPUT/'environment_freeze.txt').write_text(subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True))
    import_code='import json,torch; from diffusers import WanPipeline,AutoencoderKLWan,UniPCMultistepScheduler; from experiments.wan_state_clock.video_temporal_sync_bridge_run import environment; r=environment(); r.update(cuda_available=torch.cuda.is_available()); print(json.dumps(r))'
    env=json.loads(subprocess.check_output([PYTHON,'-c',import_code],cwd=REPO,text=True))
    write_json(OUTPUT/'environment_receipt.json',env)
    if not env['cuda_available']:print('CUDA unavailable: CPU path selected; much slower.')
    logged(['ffmpeg','-version'],'FFMPEG_PROBE');logged(['ffprobe','-version'],'FFPROBE_PROBE')
    setup.update(status='SETUP_COMPLETE',python_executable=PYTHON,environment_strategy='current_python_fresh_children')
    write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:failed('SOURCE_OR_ENVIRONMENT',exc);raise
''')
    add("code",r'''try:
    from google.colab import userdata
    token=userdata.get('HF_TOKEN')
    if token:os.environ['HF_TOKEN']=token
except Exception:
    pass
finally:
    token=None
command=[PYTHON,'-u','-m','experiments.wan_state_clock.video_temporal_sync_bridge_run','--output',str(RUN_OUTPUT)]
try:
    returncode=logged(command,'FIXED_RUN',cwd=REPO,check=False)
    RESULT_PATH=RUN_OUTPUT/'result.json'
    write_json(OUTPUT/'execution_receipt.json',dict(command=command,returncode=returncode,result_path=str(RESULT_PATH),result_exists=RESULT_PATH.is_file()))
    if not RESULT_PATH.is_file():raise RuntimeError('No runner result; full setup denominator is retained')
    setup.update(status='SUPERSEDED_BY_RESULT',result_path=str(RESULT_PATH));write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:failed('RUNNER',exc);raise
''')
    add("code",r'''result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity/denominator mismatch')
print(result['status'],result['counts'])
print('Blind pilot search and accepted payload:')
for name,row in result['searches'].items():
    print(name,row['status'],'b=',row.get('selected_b'),'phase=',row.get('selected_g'),'score=',row.get('max_score'),'ties=',row.get('ties'))
for name,row in result['evaluations'].items():
    print(name,row['status'],'accepted=',row.get('accepted_payload'),'errors=',row.get('bit_errors'),'localized=',row.get('localized_correctly'))
print('Fixed324 candidates,30 raw phase reads and24 posthoc rows retained at',RUN_OUTPUT)
print('Quality diagnostics:',result.get('quality'))
print('Evidence ceiling:',result['evidence_ceiling'])
from IPython.display import Video,display
for name,row in list(result['sources'].items())+list(result['views'].items()):
    print(name,row['status'])
    if row['status']=='SAVED' and Path(row['path']).is_file():
        try:
            display(Video(str(row['path']),embed=True))
        except Exception as exc:
            print('Preview unavailable; saved MP4 remains at',row['path'],type(exc).__name__,str(exc))
print('Inspect visible quality separately; recovery results do not imply quality PASS.')
''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python3",language="python",name="python3"),
        language_info=dict(name="python"),candidate_binding=dict(source_sha=source_sha,
            status="PUBLISHED_SHA_BOUND" if source_sha else "UNPUBLISHED_DRAFT",
            execution="User-run only; no agent GPU/Colab execution")))
    Path(output).write_text(json.dumps(notebook,indent=1)+'\n');return Path(output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-sha");p.add_argument("--output",type=Path,default=OUTPUT)
    a=p.parse_args();print(build(a.source_sha,a.output))
