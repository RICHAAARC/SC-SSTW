"""Build the adopted one-candidate four-arm lowband user-run Colab notebook."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_video_local_fourier_rm_same_raster_notebook as template
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_lowband_v1_colab.ipynb'
MARKDOWN="""# Local Fourier RM 唯一16×16低带宽局部基 V1

直接 Run all：同一 source/seed、同模型revision、同shared initial/pristine native scheduler，四臂 OFF／PAYLOAD_MULTI／STATE_OLD_MULTI／STATE_LOWBAND_MULTI 各真实50采样步，MULTI25–49。原payload、RM32 chips／45states／4ages／alpha=1/sqrt5568／pilot eta696/cap1不变；cap只缩小，不强制放大。

唯一新载体：四16×16 supports (4:20,8:24),(4:20,40:56),(24:40,8:24),(24:40,40:56)，16网格非DC共轭representative按signed频率平方、原(h,w)lex取前32，沿用原密钥置换/符号域。较大支持与较低频率是一个固定构造，不证明单独原因，底部支持触h40边界。不扫描、不改时域family/phase/惩罚/阈值。

generation与VAE/media分子进程释放内存，保存100个真实writer sidecars及实际norm，但sidecars不入receiver。每臂一次现有VAEdecode/clamp＋原np.rint RGB8；同一保存bytes分别DIRECT、原RAW420、原libx264 CRF18/yuv420p/8fps MP4，再12次VAEencode。保存源raster、YUV、两种RGB24、命令/日志/probe及环境。RAW420/MP4差异可含color转换/metadata，不单独归因H264。

两个公开receiver family OLD8／LOWBAND16各自独立读取全部四臂×三通道×两key：48q、96ABS/DIFF、16704path、272448difference edge、95040absolute local costs；同模板/同raster比较不是独立样本。每channel/key仅一次新重复payload，共24read/48message后评。两个family分别报告，不能择优合并。先封存全部raw与snapshots，再truth/message后评；全部acceptance=false。

EXECUTION_COMPLETE仅工程完整，不等于scientific PASS。该轮检验开题中的媒体稳定局部状态载体；时间相关片段payload、独立盲片段/声明时域编辑、片段—序列归因与必要拒绝尚未完成。quality仅已保存bytes的诊断，无质量阈值。当前Python/Torch优先，实际WanPipeline/AutoencoderKLWan和媒体工具可用性缺失才必要修复，无venv/版本/GPU型号门禁。SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Local-Fourier-RM-Lowband-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
RUN_OUTPUT=OUTPUT/'fixed_reference'
REPO=Path('/content/SC-SSTW-LOCAL-FOURIER-RM-LOWBAND-'+STAMP)
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_MULTI','STATE_OLD_MULTI','STATE_LOWBAND_MULTI');KEYS=('CORRECT','WRONG')
CHANNELS=('DIRECT_RGB8','RAW420','MP4');FAMILIES=('OLD8','LOWBAND16');MODES=('ABSOLUTE_CONTROL','ADJACENT_DIFFERENCE')
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
 generation={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},writer_diagnostics={a+'/'+str(i):{'status':'NOT_RUN_SETUP'} for a in ARMS for i in range(25,50)},
 terminal_inputs={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},rasters={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 transport={a+'/'+c:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS},normalized={a+'/'+c:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS},
 projections={a+'/'+c+'/'+f+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for f in FAMILIES for k in KEYS},
 mode_reads={a+'/'+c+'/'+f+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP','accepted_payload':False,'state_path_accepted':False} for a in ARMS for c in CHANNELS for f in FAMILIES for k in KEYS for m in MODES},
 path_posthoc={a+'/'+c+'/'+f+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for f in FAMILIES for k in KEYS for m in MODES},
 edge_posthoc={a+'/'+c+'/'+f+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for f in FAMILIES for k in KEYS},
 payload_reads={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 payload_evaluations={a+'/'+c+'/'+k+'/'+t:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS for t in ('REGISTERED','WRONG_MESSAGE')},
 quality={c+'/'+a+'_vs_'+b:{'status':'NOT_RUN_SETUP'} for c in CHANNELS for i,a in enumerate(ARMS) for b in ARMS[:i]})
"""+template.SETUP[template.SETUP.index('def write_json'):]
ENVIRONMENT=template.ENVIRONMENT.replace('video_local_fourier_rm_same_raster_run','video_local_fourier_rm_lowband_v1_run').replace('current_python_fresh_child_VAE_same_raster','current_python_separate_generation_media_workers')
ENVIRONMENT=ENVIRONMENT.replace("from diffusers import AutoencoderKLWan; print(AutoencoderKLWan.__name__)","from diffusers import AutoencoderKLWan,WanPipeline; print(AutoencoderKLWan.__name__,WanPipeline.__name__)")
RUN=template.RUN.replace('video_local_fourier_rm_same_raster_run','video_local_fourier_rm_lowband_v1_run')
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Calls:',result['calls'],'Workers:',result['workers'])
print('Shared before25:',result.get('before_step25_identity'))
for arm,row in result['generation'].items():print('Writer:',arm,row['status'],(row.get('receipt') or {}).get('control_norm_statistics'),(row.get('receipt') or {}).get('writer_diagnostics'))
for family in FAMILIES:
    print('Independent reader family:',family)
    for name,row in result['path_posthoc'].items():
        if '/'+family+'/' in name:print(name,row['status'],'rank=',row.get('true_path_rank'),'Delta=',row.get('true_path_delta'),'matches=',row.get('canonical_path_matches'))
for name,row in result['payload_evaluations'].items():print('Repeated payload:',name,row['status'],'bit errors=',row.get('bit_errors'),'exact=',row.get('exact_bits'))
print('Saved-RGB quality diagnostics:',result['quality'])
from IPython.display import Video,display
for arm in ARMS:
    video=result['transport'].get(arm+'/MP4',{}).get('events',{}).get('mp4',{})
    path=video.get('path')
    if video.get('status')=='SAVED' and isinstance(path,str) and Path(path).is_file():
        try:print('Saved MP4 preview:',arm);display(Video(str(path),embed=True))
        except Exception as preview_error:print('Preview unavailable; result retained:',arm,preview_error)
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('published immutable source SHA required')
    fixed=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_lowband_v1.json').read_text())['fixed_denominator'];note='Published source: `'+source_sha+'`.' if source_sha else 'UNPUBLISHED DRAFT: bind published immutable source SHA before running.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)];cells=[]
    for i,(kind,text) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'local-fourier-rm-lowband-v1-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run only; no agent real model execution')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
