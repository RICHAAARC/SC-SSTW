"""Build the fixed image-reference Colab draft; publish before SHA binding."""
from pathlib import Path
import argparse
import ast
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"notebooks/image_trajectory_reference_v1_colab.ipynb"


def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}",source_sha):
        raise ValueError("published immutable source SHA required")
    template=json.loads((ROOT/"notebooks/rgb_dct_structured_terminal_feedback_v1_colab.ipynb").read_text())
    mount="".join(template["cells"][0]["source"])
    assert mount=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True),id=f"image-reference-{len(cells)}")
        if kind=="code":
            ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add("code",mount)
    add("markdown","""# Fixed GROW published-code image reference

One official owl prompt, seed 42: OFF + GROW, two saved PNGs, four independent
32-bit reads (correct/wrong layout), eight post-read evaluations (OKOK/NOPE).
This follows the pinned author's released FFT-real/mean-MSE code, not a claim
of reproducing the paper's DCT/settings. No parameter scan or video guard.

Status: """+("published source bound.\n" if source_sha else "UNPUBLISHED DRAFT; source SHA intentionally unset.\n")+"""
This run uses the public community mirror sd2-community/stable-diffusion-2-1-base
at revision 4e63672c03103b6c636b8fb4119ba982469b2955. It needs no mandatory HF_TOKEN
or live model-info lookup. The original author methods and fixed 2/4/8 rows
are retained. This is a community-source reference, not a claim that every
weight was compared against the inaccessible official repository.

Use Run all in the current Colab Python runtime. Setup reuses its working
Torch/Torchvision pair, installs compatible missing dependencies, and records
actual versions. GPU is recommended; CPU remains allowed. There is no exact
Python version or GPU model requirement. No real-model run has been performed
by the agent. Results go to a new Drive timestamp directory.
""")
    add("code",f"SOURCE_SHA = {source_sha!r}\n"+r'''from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, traceback
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/Image-Trajectory-Reference-V1') / STAMP
OUTPUT.mkdir(parents=True, exist_ok=False)
RUN_OUTPUT = OUTPUT / 'fixed_reference'
REPO = Path('/content/SC-SSTW-IMAGE-REFERENCE-' + STAMP)
VENV = Path('/content/image-reference-venv-' + STAMP)
PYTHON = VENV / 'bin/python'
FIXED = {'sources': 1, 'png_images': 2, 'blind_readouts': 4, 'evaluations': 8}
setup = {'status': 'SETUP_STARTED', 'source_sha': SOURCE_SHA, 'fixed_denominator': FIXED,
         'images': {a: {'status': 'NOT_RUN_SETUP'} for a in ('OFF','GROW')},
         'reads': {a+'/'+k: {'status': 'NOT_RUN_SETUP'} for a in ('OFF','GROW') for k in ('CORRECT','WRONG')},
         'evaluations': {a+'/'+k+'/'+t: {'status': 'NOT_RUN_SETUP'} for a in ('OFF','GROW') for k in ('CORRECT','WRONG') for t in ('REGISTERED','WRONG_MESSAGE')}}
def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(temp,path)
def failed(stage,exc):
    setup.update(status='SETUP_FAILED',stage=stage,error=f'{type(exc).__name__}: {exc}')
    write_json(OUTPUT/'setup_receipt.json',setup)
    write_json(OUTPUT/'setup_failure.json',{'stage':stage,'error':str(exc),'traceback':traceback.format_exc()})
def logged(command,stage,cwd=None,check=True):
    try:
        with (OUTPUT/'execution.log').open('a') as log:
            log.write('COMMAND '+repr(command)+'\n');log.flush()
            child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait();log.write('EXIT '+str(code)+'\n')
        if check and code: raise subprocess.CalledProcessError(code,command)
        return code
    except Exception as exc:
        failed(stage,exc);raise
write_json(OUTPUT/'setup_receipt.json',setup)
print('Fixed output:',OUTPUT)
if SOURCE_SHA is None:
    error=RuntimeError('Unpublished draft: bind the verified published candidate SHA before Run all.')
    failed('SOURCE_BINDING',error)
    raise error
''')
    add("code",r'''try:
    logged(['git','clone','--filter=blob:none','https://github.com/RICHAAARC/SC-SSTW.git',str(REPO)],'SOURCE_CLONE')
    logged(['git','-C',str(REPO),'fetch','origin',SOURCE_SHA],'SOURCE_FETCH')
    logged(['git','-C',str(REPO),'checkout','--detach',SOURCE_SHA],'SOURCE_CHECKOUT')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
    if actual!=SOURCE_SHA or dirty: raise RuntimeError('source identity/clean checkout mismatch')
    source_files={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (REPO/'experiments/image_trajectory_reference').glob('*') if p.is_file()}
    write_json(OUTPUT/'source_receipt.json',{'source_sha':actual,'clean':True,'files':source_files})
except Exception as exc:
    failed('SOURCE',exc);raise
''')
    add("code",r'''try:
    logged([sys.executable,'-m','venv','--system-site-packages',str(VENV)],'RUNTIME_ENV')
    # Probe in a child process so repairs do not require a notebook restart.
    torch_ok=logged([str(PYTHON),'-c','import torch, torchvision; print(torch.__version__, torchvision.__version__)'],'TORCH_PROBE',check=False)==0
    if not torch_ok:
        logged([str(PYTHON),'-m','pip','install','--upgrade','torch','torchvision'],'TORCH_REPAIR')
    logged([str(PYTHON),'-m','pip','install','-r',str(REPO/'experiments/image_trajectory_reference/requirements-colab.txt')],'COMPATIBLE_DEPENDENCIES')
    dependency_code=logged([str(PYTHON),'-m','pip','check'],'DEPENDENCY_REPORT',check=False)
    setup['dependency_check_returncode']=dependency_code
    if dependency_code:
        print('Package metadata conflicts recorded; continuing to actual import and run checks.')
    frozen=subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True)
    (OUTPUT/'environment_freeze.txt').write_text(frozen)
except Exception as exc:
    failed('ENVIRONMENT',exc);raise
''')
    add("code",r'''try:
    import urllib.request, zipfile
    sha='6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870'
    archive=Path('/content/grow-reference-'+STAMP+'.zip')
    url='https://api.github.com/repos/luopengchen/GROW/zipball/'+sha
    urllib.request.urlretrieve(url,archive)
    extracted=Path('/content/grow-reference-'+STAMP)
    extracted.mkdir()
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            if not (extracted/item.filename).resolve().is_relative_to(extracted.resolve()):
                raise ValueError('invalid archive path')
        package.extractall(extracted)
    roots=[p for p in extracted.iterdir() if p.is_dir()]
    if len(roots)!=1: raise RuntimeError('unexpected official archive root')
    UPSTREAM=roots[0]
    verify_code='from experiments.image_trajectory_reference.official_bridge import verify_source; import json,sys; print(json.dumps(verify_source(sys.argv[1])))'
    checked=subprocess.check_output([str(PYTHON),'-c',verify_code,str(UPSTREAM)],cwd=REPO,text=True)
    write_json(OUTPUT/'upstream_receipt.json',{'url':url,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verification':json.loads(checked)})
    environment_code='import json,torch; from experiments.image_trajectory_reference.run import environment_receipt; from diffusers import DDIMScheduler,StableDiffusionPipeline; r=environment_receipt(); r.update(cuda_available=torch.cuda.is_available()); print(json.dumps(r))'
    environment=json.loads(subprocess.check_output([str(PYTHON),'-c',environment_code],cwd=REPO,text=True))
    write_json(OUTPUT/'environment_receipt.json',environment)
    if not environment['cuda_available']: print('GPU unavailable; continuing on CPU (slower).')
    setup.update(status='SETUP_COMPLETE',upstream=str(UPSTREAM));write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:
    failed('UPSTREAM_OR_ENVIRONMENT',exc);raise
''')
    add("code",r'''# Optional access token for Hub download rate limits; the public mirror needs none.
try:
    from google.colab import userdata
    token=userdata.get('HF_TOKEN')
    if token: os.environ['HF_TOKEN']=token
except Exception:
    pass  # Public mirror: anonymous access is supported.
finally:
    token=None
command=[str(PYTHON),'-u','-m','experiments.image_trajectory_reference.run','--upstream-root',str(UPSTREAM),'--output',str(RUN_OUTPUT)]
try:
    returncode=logged(command,'FIXED_RUN',cwd=REPO,check=False)
    RESULT_PATH=RUN_OUTPUT/'result.json'
    write_json(OUTPUT/'execution_receipt.json',{'command':command,'returncode':returncode,'result_path':str(RESULT_PATH),'result_exists':RESULT_PATH.is_file()})
    if not RESULT_PATH.is_file(): raise RuntimeError('runner failed before retained result; fixed setup rows remain in setup_receipt.json')
    setup.update(status='SUPERSEDED_BY_RESULT',result_path=str(RESULT_PATH));write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:
    failed('RUNNER',exc);raise
''')
    add("code",r'''result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:
    raise RuntimeError('source/denominator mismatch')
print('Status:',result['status'])
print('Model assets:',result.get('assets'))
print('Fixed counts:',result['counts'])
for name,row in result['evaluations'].items():
    print(name,row['status'],'bit errors=',row.get('bit_errors'),'BER=',row.get('ber'),'exact=',row.get('exact_bits'))
print('Saved-PNG quality diagnostic:',result.get('quality'))
print('Evidence ceiling:',result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
from IPython.display import display
from PIL import Image
for arm,row in result['images'].items():
    print(arm,row['status'])
    path=Path(row['path'])
    if row['status']=='SAVED' and path.is_file():
        with Image.open(path) as saved_png:
            display(saved_png.copy())
''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),
        language_info=dict(name="python"),candidate_binding=dict(
            source_sha=source_sha,status="PUBLISHED_SHA_BOUND" if source_sha else "UNPUBLISHED_DRAFT",
            execution="User-run Colab only; local AST/CPU validation does not execute this notebook")))
    Path(output).write_text(json.dumps(notebook,indent=1)+'\n')
    return Path(output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-sha");p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args();print(build(args.source_sha,args.output))
