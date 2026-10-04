"""Build the unpublished OLD8 two-state user-run Colab notebook."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_video_local_fourier_rm_lowband_v1_notebook as template

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_old8_two_state_v1_colab.ipynb'
MARKDOWN="""# OLD8 双状态静态／换序局部载体 V1

发布并绑定不可变 source SHA 后直接 Run all。固定同一开发 source/seed2026092501、同模型 revision、shared initial/pristine scheduler；六臂 OFF、PAYLOAD_MULTI、STATE_A、STATE_B、STATE_AB、STATE_BA 各真实50步，控制固定 MULTI25..49。

A/B 只取原 OLD8 key 码本排序第0/1个32-chip字，不看保存响应。换序边界固定在起点23：1..22与23..45交换；u23..25保留真实四-age混合过渡。OLD8四个8x8支持、真实余弦基、channel4、四age、alpha、payload及 eta696/cap1 shrink-only 更新均不变。每臂记录实际 raw/applied norm；相同 cap 不称等实际预算，不匹配、不放大。

每个收到的保存观测与 key 在不知道 arm、真序列、消息、writer latent/sidecar、同源OFF或攻击真值时，同时评分 AA/BB/AB/BA。ABS 是主四序列整体可辨识诊断；ADJACENT_DIFFERENCE 是预先保留的独立诊断，恒A/恒B在部分支持上的精确等价完整保留，不要求 DIFF 也唯一。逐窗四模板软代价及每个 window×age 的 A/B 32-chip cost margin 全部保存；首端、转换区、tie、缺支持、nonfinite和失败不删除。盲 readout 先保存封存，随后才加载该文件做 truth/message join。配对差仅事后质量诊断。

DIRECT_RGB8、RAW420、MP4 均从各臂同一保存 RGB8 raster进入，generation与VAE/media分子进程释放。固定g0/R44是公开诊断相位，不能称未知起点或任意编辑盲同步。重复payload不是时间相关payload。所有 acceptance=false；EXECUTION_COMPLETE只表示固定工程执行完整，不是科学PASS或已成功盲同步。

SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
if SOURCE_SHA is None:raise RuntimeError('UNPUBLISHED_DRAFT: publish source and rebuild with its immutable SHA before Run all')
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Local-Fourier-RM-OLD8-Two-State-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
RUN_OUTPUT=OUTPUT/'fixed_reference'
REPO=Path('/content/SC-SSTW-LOCAL-FOURIER-RM-OLD8-TWO-STATE-'+STAMP)
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_MULTI','STATE_A_MULTI','STATE_B_MULTI','STATE_AB_MULTI','STATE_BA_MULTI')
CHANNELS=('DIRECT_RGB8','RAW420','MP4');KEYS=('K0','K1');MODES=('ABSOLUTE_CONTROL','ADJACENT_DIFFERENCE')
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
 generation={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 writer_diagnostics={a+'/'+str(i):{'status':'NOT_RUN_SETUP'} for a in ARMS for i in range(25,50)},
 terminal_inputs={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},rasters={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 transport={a+'/'+c:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS},
 normalized={a+'/'+c:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS},
 projections={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 sequence_reads={a+'/'+c+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP','accepted_payload':False,'state_sequence_accepted':False} for a in ARMS for c in CHANNELS for k in KEYS for m in MODES},
 payload_reads={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 sequence_posthoc={a+'/'+c+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS for m in MODES},
 local_posthoc={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS},
 payload_evaluations={a+'/'+c+'/'+k+'/'+t:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS for t in ('REGISTERED','WRONG_MESSAGE')},
 quality={c+'/'+a+'_vs_'+b:{'status':'NOT_RUN_SETUP'} for c in CHANNELS for i,a in enumerate(ARMS) for b in ARMS[:i]})
"""+template.SETUP[template.SETUP.index('def write_json'):]
ENVIRONMENT=template.ENVIRONMENT.replace('video_local_fourier_rm_lowband_v1_run','video_local_fourier_rm_old8_two_state_v1_run').replace('LOWBAND','OLD8-TWO-STATE')
RUN=template.RUN.replace('video_local_fourier_rm_lowband_v1_run','video_local_fourier_rm_old8_two_state_v1_run')
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Calls:',result['calls'],'Workers:',result['workers'],'Blind SHA:',result.get('blind_receiver_sha256'))
print('A/B selection:',result['selection'],'Shared before25:',result.get('before_step25_identity'))
for arm,row in result['generation'].items():print('Writer:',arm,row['status'],(row.get('receipt') or {}).get('control_norm_statistics'))
for name,row in result['sequence_posthoc'].items():print('Sequence:',name,row['status'],'rank=',row.get('true_rank'),'delta=',row.get('true_delta'),'top=',row.get('top'))
for name,row in result['local_posthoc'].items():print('Local A/B:',name,row['status'],'scored=',row.get('scored'),'truth-in-top=',row.get('truth_in_top'),'unique=',row.get('unique_truth'))
for name,row in result['payload_evaluations'].items():print('Repeated payload:',name,row['status'],'bit errors=',row.get('bit_errors'))
print('Saved-RGB quality diagnostics:',result['quality'])
from IPython.display import Video,display
for arm in ARMS:
    video=result['transport'].get(arm+'/MP4',{}).get('events',{}).get('mp4',{});path=video.get('path')
    if video.get('status')=='SAVED' and isinstance(path,str) and Path(path).is_file():
        try:print('Saved MP4 preview:',arm);display(Video(str(path),embed=True))
        except Exception as preview_error:print('Preview unavailable; result retained:',arm,preview_error)
print('Failures:',result['failures']);print('Evidence ceiling:',result['evidence_ceiling']);print('Retained result:',RESULT_PATH)
"""


def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('published immutable source SHA required')
    fixed=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_old8_two_state_v1.json').read_text())['fixed_denominator']
    note='Published source: `'+source_sha+'`.' if source_sha else 'UNPUBLISHED DRAFT: SOURCE_SHA=None. Publish the reviewed source, then rebuild with that immutable SHA before the user runs Colab.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)];cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'old8-two-state-v1-{i}',metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run only after immutable SHA binding; no agent real model execution')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha');parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();print(build(args.source_sha,args.output))
