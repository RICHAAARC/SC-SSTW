"""Build fixed video-reference Colab; published source binds later."""
from pathlib import Path
import argparse
import ast
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"notebooks/grow_video_reference_v1_colab.ipynb"


def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}",source_sha):
        raise ValueError("immutable published source SHA required")
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,metadata={},id=f"grow-video-{len(cells)}",source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add("code","from google.colab import drive\ndrive.mount('/content/drive')\n")
    add("markdown","""# Fixed GROW image-to-Wan video reference

One preselected copper-kettle prompt/seed; OFF, MULTI25..49 and LAST49.
Three saved MP4s; terminal/float-RGB/RGB8/MP4 layers, two blind keys per layer,
24 raw readouts then48 post-read evaluations. MP4 is primary. These dependent
layers are diagnostics, not independent samples. MULTI includes LAST's step;
this comparison cannot isolate the non-last contribution.

The published GROW FFT-real/mean-MSE construction is adapted to native Wan
Flow x0=z-sigma*v, conditional update then FP32 CFG5. Four channels/32bits,
46 temporal repeats, actual-mask44160 ->eta5520. No total-budget matching,
three-direction solver, positive-group acceptance guard or timing search.

Run all uses the current Python and fresh serial children. Generation ends
before a separate FP32-VAE/media worker. No virtual environment,ensurepip,
Python-version or GPU-model gate. GPU is recommended; CPU remains allowed.
The notebook has not been executed by the implementation agent. Results use
one new Drive timestamp directory and preserve failures and all fixed rows.

"""+("Published source bound.\n" if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA is intentionally unset.\n"))
    add("code",f"SOURCE_SHA = {source_sha!r}\n"+r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/GROW-Video-Reference-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
REPO=Path('/content/SC-SSTW-GROW-VIDEO-'+STAMP)
RUN_OUTPUT=OUTPUT/'fixed_reference'
PYTHON=sys.executable
ARMS=('OFF','MULTI','LAST');LAYERS=('terminal','float_rgb','rgb8','mp4');KEYS=('CORRECT','WRONG')
FIXED=dict(sources=1,arms=3,mp4=3,layers_per_arm=4,keys=2,raw_reads=24,evaluations=48)
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
    generation={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
    videos={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
    reads={a+'/'+l+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for l in LAYERS for k in KEYS},
    evaluations={a+'/'+l+'/'+k+'/'+t:{'status':'NOT_RUN_SETUP'} for a in ARMS for l in LAYERS for k in KEYS for t in ('REGISTERED','WRONG_MESSAGE')})
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
    import_code='import json,torch; from diffusers import WanPipeline,AutoencoderKLWan,UniPCMultistepScheduler; from experiments.wan_state_clock.grow_video_reference_run import environment; r=environment(); r.update(cuda_available=torch.cuda.is_available()); print(json.dumps(r))'
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
command=[PYTHON,'-u','-m','experiments.wan_state_clock.grow_video_reference_run','--output',str(RUN_OUTPUT)]
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
print('Primary MP4 payload results:')
for name,row in result['evaluations'].items():
    if row['layer']=='mp4':print(name,row['status'],'errors=',row.get('bit_errors'),'BER=',row.get('ber'),'exact=',row.get('exact_bits'))
print('All48 evaluation rows and24 raw readouts retained at',RUN_OUTPUT)
print('MP4 relative-to-OFF quality diagnostics:',result.get('quality'))
print('Evidence ceiling:',result['evidence_ceiling'])
from IPython.display import Video,display
for arm,row in result['videos'].items():
    print(arm,row['status'])
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
