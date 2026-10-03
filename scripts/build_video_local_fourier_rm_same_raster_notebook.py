"""Build the fixed same-raster media user-run notebook; no agent model execution."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_video_local_fourier_rm_channel_notebook as template
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_same_raster_v1_colab.ipynb'
MARKDOWN="""# Local Fourier RM 同一 RGB8 媒体链诊断 V1

直接 Run all：固定复用原20261003T050759442813Z的OFF／PAYLOAD_MULTI／STATE_MULTI终态，不重新生成，不改MULTI25–49、carrier、codebook、alpha或budget。每臂仅一次现有VAE解码/clamp＋原np.rint RGB8量化，保存完整RGB8 bytes及hash。DIRECT_RGB8、RAW420、MP4三支从同一文件重开，再分别重编码；3 decode、9 encode。

DIRECT为同bytes FP32 /255；RAW420为原materialized RGB24→YUV420→RGB24；MP4为原libx264/CRF18/yuv420p/8fps保存与独立RGB24读回。保存raster、YUV420、两种RGB24读回bytes、实际commands/stdout/stderr/ffprobe与VAE环境。RAW420与MP4可能有不同内部color转换/标签，差异不单独证明H264因果；新DIRECT若已失败完整保留，不选择旧正样例raster。

9 normalized／18 q各按冻结ABSOLUTE_CONTROL和ADJACENT_DIFFERENCE评分：36mode、6264path、102168difference edge及35640absolute local costs。174有效路径、等价类、tie、拒绝完整保留。18次新重复payload read、36消息后评不参与path选择，也不证明时间相关载荷。所有raw/snapshots先封存再truth；全部acceptance=false。EXECUTION_COMPLETE只表示工程完整，未完成独立盲视频同步、时域编辑或整个方法闭合。

优先当前Python与已有Torch，实际VAE依赖缺失才修复，无venv、GPU型号或包版本硬门槛。用户选择可用计算环境后Run all，所有失败、缺失保留固定分母。SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Local-Fourier-RM-Same-Raster-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
RUN_OUTPUT=OUTPUT/'fixed_reference'
REPO=Path('/content/SC-SSTW-LOCAL-FOURIER-RM-SAME-RASTER-'+STAMP)
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_MULTI','STATE_MULTI');KEYS=('CORRECT','WRONG')
CHANNELS=('DIRECT_RGB8','RAW420','MP4');MODES=('ABSOLUTE_CONTROL','ADJACENT_DIFFERENCE')
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
 terminal_inputs={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},rasters={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 transport={a+'/'+c:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS},
 normalized={a+'/'+c:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS},
 projections={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 mode_reads={a+'/'+c+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP','accepted_payload':False,'state_path_accepted':False} for a in ARMS for c in CHANNELS for k in KEYS for m in MODES},
 path_posthoc={a+'/'+c+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS for m in MODES},
 edge_posthoc={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 payload_reads={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 payload_evaluations={a+'/'+c+'/'+k+'/'+t:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS for t in ('REGISTERED','WRONG_MESSAGE')})
"""+template.SETUP[template.SETUP.index('def write_json'):]
ENVIRONMENT=template.ENVIRONMENT.replace('video_local_fourier_rm_channel_run','video_local_fourier_rm_same_raster_run').replace('current_python_fresh_child_VAE_only','current_python_fresh_child_VAE_same_raster')
MEDIA_REPAIR="""    import shutil
    tools={name:shutil.which(name) for name in ('ffmpeg','ffprobe')}
    tool_codes={name:logged([path,'-version'],name.upper()+'_PROBE',check=False) if path else None for name,path in tools.items()}
    if any(code!=0 for code in tool_codes.values()):
        try:
            logged(['apt-get','update'],'MEDIA_TOOL_UPDATE',check=False)
            logged(['apt-get','install','-y','ffmpeg'],'MEDIA_TOOL_REPAIR',check=False)
        except Exception as repair_error:print('Media-tool repair error retained:',repair_error)
    setup['media_tool_probe_returncodes']=tool_codes
"""
ENVIRONMENT=ENVIRONMENT.replace('    torch_ok=',MEDIA_REPAIR+'    torch_ok=')
RUN=template.RUN.replace('video_local_fourier_rm_channel_run','video_local_fourier_rm_same_raster_run')
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Calls:',result['calls'],'Generation:',result['actual_generation_calls'],'Writer updates:',result['actual_writer_update_calls'])
for name,row in result['path_posthoc'].items():
    print(name,row['status'],'path rank=',row.get('true_path_rank'),'Delta=',row.get('true_path_delta'),'matches=',row.get('canonical_path_matches'))
for name,row in result['payload_evaluations'].items():
    print(name,row['status'],'bit errors=',row.get('bit_errors'),'exact=',row.get('exact_bits'))
print('Saved same rasters:',result['rasters'])
print('Transport failures:',{k:v for k,v in result['transport'].items() if v['status']!='COMPLETE'})
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('published immutable source SHA required')
    fixed=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_same_raster_v1.json').read_text())['fixed_denominator']
    note='Published source: `'+source_sha+'`.' if source_sha else 'UNPUBLISHED DRAFT: bind published immutable source SHA before running.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),
        ('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)]
    cells=[]
    for i,(kind,text) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'local-fourier-rm-same-raster-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),
        candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run only; no agent VAE/generation')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
