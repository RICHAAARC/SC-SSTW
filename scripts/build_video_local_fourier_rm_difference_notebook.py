"""Build the fixed saved-projection, NumPy-only user-run notebook."""
from pathlib import Path
import argparse,ast,json,re
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_difference_v1_colab.ipynb'
MARKDOWN="""# 局部 Fourier RM 相邻窗口差分接收诊断 V1

直接 Run all：固定复用 `Video-Local-Fourier-RM-Channel-V1/20261003T064701431228Z/fixed_reference` 的24个已保存密钥投影。仅计算相邻窗口差分；不重读latent张量，不运行生成、VAE、MP4、全相位搜索，不改MULTI25–49、载体、RM码本、alpha或预算。只需当前Python与NumPy，不需要GPU。

44个原窗口形成43个相关差分；每条原174条有效路径按真实候选taus计算差分模板。支持为相邻公开mask交集，联合SSE除以有效edge维数；完整支持为5504维。132种局部stay/+1/+2转移的43×132成本完整保留。45种stay的局部模板都为零，不能把任意canonical stay解释成源位置。保留差分精确等价类、44窗feasible sets及数值ties，不加编辑惩罚或自由拟合。

同一投影的绝对对照与原saved infer完整核对；24 absolute＋24 difference，8352路径成本、136224新edge局部成本。缓存24个payload仅标 REUSED，48条消息后评不能算新恢复，也不参与选path。终态与旧MP4参考分表，truth只在保存48mode接收证据之后加入。全部acceptance为false；这是接收器诊断，未完成独立盲视频同步或时域攻击验证。旧MP4来自上一次媒体运行，仅作跨运行参考。

失败、缺失和所有候选保留，EXECUTION_COMPLETE仅表示固定诊断执行完整。SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Local-Fourier-RM-Difference-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
RUN_OUTPUT=OUTPUT/'fixed_reference'
REPO=Path('/content/SC-SSTW-LOCAL-FOURIER-RM-DIFFERENCE-'+STAMP)
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_MULTI','STATE_MULTI');KEYS=('CORRECT','WRONG')
STAGES=('TERMINAL44_REFERENCE','FLOAT_VAE_ROUNDTRIP','RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC','MP4_G0_REFERENCE')
MODES=('ABSOLUTE_CONTROL','ADJACENT_DIFFERENCE')
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
 inputs={a+'/'+s+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for s in STAGES for k in KEYS},
 mode_reads={a+'/'+s+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP','state_path_accepted':False,'accepted_payload':False} for a in ARMS for s in STAGES for k in KEYS for m in MODES},
 payload_evaluations={a+'/'+s+'/'+k+'/'+t:{'status':'NOT_RUN_SETUP','payload_source':'REUSED'} for a in ARMS for s in STAGES for k in KEYS for t in ('REGISTERED','WRONG_MESSAGE')})
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
    numpy_ok=logged([PYTHON,'-c','import numpy; print(numpy.__version__)'],'NUMPY_PROBE',check=False)==0
    if not numpy_ok:logged([PYTHON,'-m','pip','install','numpy'],'NUMPY_REPAIR',check=False)
    setup.update(status='SETUP_COMPLETE',python_executable=PYTHON,environment_strategy='current_python_numpy_only')
    write_json(OUTPUT/'setup_receipt.json',setup)
except Exception as exc:failed('SOURCE_OR_ENVIRONMENT',exc);raise
"""
RUN="""command=[PYTHON,'-u','-m','experiments.wan_state_clock.video_local_fourier_rm_difference_run','--output',str(RUN_OUTPUT)]
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
for name,row in result['path_posthoc'].items():
    print(name,row['status'],'path rank=',row.get('true_path_rank'),'Delta=',row.get('true_path_delta'),'matches=',row.get('canonical_path_matches'))
print('Input failures:',{k:v for k,v in result['inputs'].items() if v['status']!='READ'})
print('Calls (zero model/media):',result['calls'])
print('Payload:',{k:v['status'] for k,v in result['cached_payload'].items()},'REUSED; no new payload recovery')
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('published immutable source SHA required')
    fixed=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_difference_v1.json').read_text())['fixed_denominator']
    note='Published source: `'+source_sha+'`.' if source_sha else 'UNPUBLISHED DRAFT: bind published immutable source SHA before running.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),
        ('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)]
    cells=[]
    for i,(kind,text) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'local-fourier-rm-difference-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),
        candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run cached projections only; no agent cloud execution')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
