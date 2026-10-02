"""SHA-bound user-run first local Fourier RM state diagnostic."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_video_overlap_zero_mean_c1_notebook as template
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_v1_colab.ipynb'
MARKDOWN="""# 局部 Fourier 状态码与 MULTI 共存验证 V1

Run all 执行固定三臂 OFF / PAYLOAD_MULTI / STATE_MULTI；两个标记臂在真实 Wan index25..49 控制，并继续完整原生 CFG/scheduler 生成。状态臂使用四个8x8局部块、非DC实 Fourier 正交分量、45个32-chip RM(1,5)状态、四条重叠age通道。状态目标总能量1；实际状态更新仅在L2超过1时缩小，不强制放大。载荷沿用原整帧FFT重复32bit；本轮不把它称为时间相关恢复。

这是新载体假设，不继承整帧FFT或旧空间DCT的存留结论。没有频带或强度扫描。实际局部空间支持和完整码本由公共key生成，不使用sample ID。每个物理时间位置联合承载延迟状态；相关重叠分量不作为独立票。

第一轮只保存3个源MP4（181帧，yuv420p），无派生裁剪或第二次压缩。每个视频以四个合法全局相位独立重编码，共12个VAE观测；所有相位均用相同44个规则潜窗口评分，额外行仅保存。接收器保留47520个局部状态代价、4176个有限零/单事件路径代价、24个相位/密钥文件。候选只在合法相位完整序列之间比较，不逐窗口拼接相位。75个写端诊断独立保存，不进入盲接收表。

末态、真相位局部排名、全相位有限路径排名、载荷和质量分别报告；没有自动方法PASS、校准存在检测或盲同步闭合结论。局部唯一不是最终同步充分条件，也不要求每个窗口独立唯一才研究序列。下一步接口为局部软观测、接收坐标、可用支持与合法相位分组；位置相关载荷、受扰片段及序列聚合尚待同链验证。

选择GPU后直接Run all。沿用当前Python及已可用Torch；实际依赖失败如实保留。SOURCE_NOTE
"""
SETUP="""from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/Video-Local-Fourier-RM-V1')/STAMP
OUTPUT.mkdir(parents=True,exist_ok=False)
REPO=Path('/content/SC-SSTW-LOCAL-FOURIER-RM-'+STAMP)
RUN_OUTPUT=OUTPUT/'fixed_reference'
PYTHON=sys.executable
ARMS=('OFF','PAYLOAD_MULTI','STATE_MULTI');KEYS=('CORRECT','WRONG')
setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator=FIXED,
 generation={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},sources={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 normalized={a+'/'+str(g):{'status':'NOT_RUN_SETUP'} for a in ARMS for g in range(4)},
 phase_reads={a+'/'+str(g)+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for g in range(4) for k in KEYS},
 writer_diagnostics={a+'/'+str(i):{'status':'NOT_RUN_SETUP'} for a in ARMS for i in range(25,50)})
"""+template.SETUP[template.SETUP.index('def write_json'):]
ENVIRONMENT=template.ENVIRONMENT.replace('video_overlap_zero_mean_c1_run','video_local_fourier_rm_run')
RUN=template.RUN.replace('video_overlap_zero_mean_c1_run','video_local_fourier_rm_run')
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Writer diagnostics:',result['writer_diagnostic_counts'])
for name,row in result['local_posthoc'].items():
    print('True-phase local:',name,row['status'],'unique rows=',row.get('unique_truth_rows'),'/44',
          'worst Delta=',row.get('worst_local_delta'),'path rank=',row.get('true_path_rank'),'path Delta=',row.get('true_path_delta'))
for name,row in result['searches'].items():
    print('Blind phase/path:',name,row['status'],'canonical=',row.get('canonical'),'ties=',len(row.get('top',[])))
for name,row in result['evaluations'].items():
    print(name,row['status'],'path matches=',row.get('canonical_path_matches'),'payload errors=',row.get('diagnostic_bit_errors'),
          'true-phase payload errors=',row.get('oracle_bit_errors'))
print('Quality:',result['quality'])
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
from IPython.display import Video,display
for arm,row in result['sources'].items():
    print(arm,row['status'])
    if row['status']=='SAVED' and Path(row['path']).is_file():
        try:display(Video(str(row['path']),embed=True))
        except Exception as exc:print('Preview unavailable:',type(exc).__name__,str(exc))
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('immutable published source SHA required')
    fixed=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_v1.json').read_text())['fixed_denominator']
    note='Published source: `'+source_sha+'`.' if source_sha else 'UNPUBLISHED DRAFT: bind a published immutable source SHA before running.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),
      ('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)]
    cells=[]
    for i,(kind,text) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'local-fourier-rm-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
      language_info=dict(name='python'),candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run only')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
