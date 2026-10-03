"""Build the fixed saved-terminal, VAE-only user-run diagnostic notebook."""
from pathlib import Path
import argparse,ast,json,re
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_channel_v1_colab.ipynb'
MARKDOWN="""# 局部 Fourier RM 通道定位 V1

直接 Run all：复用 `20261003T050759442813Z` 已保存三臂终态，不重新生成，不改变 MULTI25–49、局部支持、码本、alpha、预算或接收器。每臂只做一次 VAE 解码，再把同一 RGB 分别浮点重编码、按原 np.rint 规则转 RGB8 后 /255 重编码。共 3 decode、6 encode，无 FFmpeg、MP4 保存、视频读取或全相位搜索。

FLOAT 包含原解码函数的 [0,1] clamp，不能称无裁切纯 VAE。RGB8 只增加一次 uint8 舍入。当前两分支共用同次解码，因而 FLOAT→RGB8 可定位量化增量；旧 MP4 来自上一进程，未保存旧解码 RGB，因此 RGB8→旧 MP4 只作跨运行一致性参考，不能单独确认 codec 因果。

6 个新观测与 6 个既存终态44／MP4-g0参考均以相同 g0／R44 回放；24 个密钥读取保留 47520 局部状态成本、4176 路径成本和重复载荷，48 条后评保留成功、失败和缺失。终态与旧 MP4 参考分表，真值只在保存读取后加入报告。所有 acceptance 均为 false；本轮是通道诊断，未完成独立盲视频同步，也不证明时间相关载荷或时域鲁棒性。

使用当前 Python 和可用 Torch，只探测实际 Wan VAE 依赖；无 venv、GPU 型号或包版本硬门槛。SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Local-Fourier-RM-Channel-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
RUN_OUTPUT=OUTPUT/'fixed_reference'
REPO=Path('/content/SC-SSTW-LOCAL-FOURIER-RM-CHANNEL-'+STAMP)
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_MULTI','STATE_MULTI');KEYS=('CORRECT','WRONG')
STAGES=('TERMINAL44_REFERENCE','FLOAT_VAE_ROUNDTRIP','RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC','MP4_G0_REFERENCE')
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
 observation_tensors={a+'/'+s:{'status':'NOT_RUN_SETUP'} for a in ARMS for s in STAGES},
 reads={a+'/'+s+'/'+k:{'status':'NOT_RUN_SETUP','accepted_payload':False,'state_path_accepted':False} for a in ARMS for s in STAGES for k in KEYS},
 evaluations={a+'/'+s+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP'} for a in ARMS for s in STAGES for k in KEYS for m in ('REGISTERED','WRONG_MESSAGE')})
def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\\n');os.replace(tmp,path)
def failed(stage,exc):
    setup.update(status='SETUP_FAILED',stage=stage,error=f'{type(exc).__name__}: {exc}')
    write_json(OUTPUT/'setup_receipt.json',setup)
    write_json(OUTPUT/'setup_failure.json',dict(stage=stage,error=str(exc),traceback=traceback.format_exc()))
def logged(command,stage,cwd=None,check=True):
    try:
        with (OUTPUT/'execution.log').open('a') as log:
            log.write('COMMAND '+repr(command)+'\\n');log.flush()
            child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait();log.write('EXIT '+str(code)+'\\n')
        if check and code:raise subprocess.CalledProcessError(code,command)
        return code
    except Exception as exc:failed(stage,exc);raise
write_json(OUTPUT/'setup_receipt.json',setup)
print('Fixed output:',OUTPUT)
if SOURCE_SHA is None:
    error=RuntimeError('Unpublished draft: bind published source SHA before Run all');failed('SOURCE_BINDING',error);raise error
"""
ENVIRONMENT="""try:
    logged(['git','clone','--filter=blob:none','https://github.com/RICHAAARC/SC-SSTW.git',str(REPO)],'SOURCE_CLONE')
    logged(['git','-C',str(REPO),'fetch','origin',SOURCE_SHA],'SOURCE_FETCH')
    logged(['git','-C',str(REPO),'checkout','--detach',SOURCE_SHA],'SOURCE_CHECKOUT')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
    if actual!=SOURCE_SHA or dirty:raise RuntimeError('source SHA/clean checkout mismatch')
    write_json(OUTPUT/'source_receipt.json',dict(source_sha=actual,clean=True))
    torch_ok=logged([PYTHON,'-c','import torch; print(torch.__version__)'],'TORCH_PROBE',check=False)==0
    if not torch_ok:logged([PYTHON,'-m','pip','install','--upgrade','torch'],'TORCH_REPAIR',check=False)
    vae_probe='from diffusers import AutoencoderKLWan; print(AutoencoderKLWan.__name__)'
    vae_ok=logged([PYTHON,'-c',vae_probe],'VAE_IMPORT_PROBE',check=False)==0
    if not vae_ok:
        logged([PYTHON,'-m','pip','install','diffusers>=0.35.1,<0.40','transformers>=4.45,<5','accelerate>=0.34','safetensors>=0.4.5'],'VAE_DEPENDENCY_REPAIR',check=False)
    dependency_code=logged([PYTHON,'-m','pip','check'],'DEPENDENCY_REPORT',check=False)
    setup['dependency_check_returncode']=dependency_code
    if dependency_code:print('Metadata conflicts retained; actual VAE import/load determines availability.')
    (OUTPUT/'environment_freeze.txt').write_text(subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True))
    environment_code='import json,torch; from experiments.wan_state_clock.video_local_fourier_rm_channel_run import environment; r=environment(); r.update(cuda_available=torch.cuda.is_available()); print(json.dumps(r))'
    env=json.loads(subprocess.check_output([PYTHON,'-c',environment_code],cwd=REPO,text=True))
    write_json(OUTPUT/'environment_receipt.json',env)
    if not env['cuda_available']:print('CUDA unavailable: CPU path selected; much slower.')
    setup.update(status='SETUP_COMPLETE',python_executable=PYTHON,environment_strategy='current_python_fresh_child_VAE_only')
    write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:failed('SOURCE_OR_ENVIRONMENT',exc);raise
"""
RUN="""try:
    from google.colab import userdata
    token=userdata.get('HF_TOKEN')
    if token:os.environ['HF_TOKEN']=token
except Exception:
    pass
finally:
    token=None
command=[PYTHON,'-u','-m','experiments.wan_state_clock.video_local_fourier_rm_channel_run','--output',str(RUN_OUTPUT)]
try:
    returncode=logged(command,'FIXED_RUN',cwd=REPO,check=False)
    RESULT_PATH=RUN_OUTPUT/'result.json'
    write_json(OUTPUT/'execution_receipt.json',dict(command=command,returncode=returncode,result_path=str(RESULT_PATH),result_exists=RESULT_PATH.is_file()))
    if not RESULT_PATH.is_file():raise RuntimeError('No runner result; setup denominator retained')
    setup.update(status='SUPERSEDED_BY_RESULT',result_path=str(RESULT_PATH));write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:failed('RUNNER',exc);raise
"""
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Model calls:',result['calls'])
for name,row in result['local_posthoc'].items():
    print(name,row['status'],'unique local=',row.get('unique_truth_rows'),'/44','path rank=',row.get('true_path_rank'),'path Delta=',row.get('true_path_delta'))
for name,row in result['evaluations'].items():
    print(name,row['status'],'bit errors=',row.get('bit_errors'),'exact=',row.get('exact_bits'),'path matches=',row.get('canonical_path_matches'))
print('Baseline:',result['baseline'])
print('Channel input diagnostics:',result['channel_inputs'])
print('Reference limit:',result['reference_environment_limit'])
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('published immutable source SHA required')
    fixed=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_channel_v1.json').read_text())['fixed_denominator']
    note='Published source: `'+source_sha+'`.' if source_sha else 'UNPUBLISHED DRAFT: bind published immutable source SHA before running.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),
      ('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)]
    cells=[]
    for i,(kind,text) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'local-fourier-rm-channel-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
      language_info=dict(name='python'),candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run only; no agent model execution')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);args=p.parse_args();print(build(args.source_sha,args.output))
