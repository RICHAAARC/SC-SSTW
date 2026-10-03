"""Build the fixed masked-contrast native MULTI user-run Colab notebook."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_video_local_fourier_rm_lowband_v1_notebook as template
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/video_local_fourier_rm_contrast_v1_colab.ipynb'
MARKDOWN="""# 局部支持内判别方向控制：真实 MULTI V1

直接 Run all。四臂 OFF／PAYLOAD_MULTI／STATE_LOWBAND_MULTI／STATE_CONTRAST_MULTI 使用同一来源、种子、初始状态与各自新复制的原生 scheduler，各完成50步，水印作用于25–49步。原 LOWBAND16 是本次同源对照；新臂只改变 pilot 原始更新的方向。

沿用四个16×16局部支持、RM32／45状态／4年龄、alpha、eta696、cap1和原payload。在R44原active支持内，用全部公开合法候选定义 ABS 与 DIFF 联合判别空间，原始pilot先投影，再与第45窗原始更新一起执行原 shrink-only cap。空间在观测前由公开协议和密钥冻结。第45窗的实际更新会随整体截断比例变化；相同上限不代表相同实际能量。

保留100份真实写端侧车和50份状态控制方向诊断，报告实际更新量、截断及所有候选相对分数变化。方向诊断与原生终态均属写端诊断，独立于主盲接收。原生终态另保留16q／32评分及完整候选，不作为媒体检测输入。

每臂一次真实VAE解码及RGB8保存，同一份字节分别通过 DIRECT_RGB8、RAW420、MP4，再共12次VAE重编码。OLD8／LOWBAND16两个公开接收协议分别读取全部臂×通道×正确/错误密钥：48q、96评分、16704条路径成本。载荷新读24次、消息后评48条；全部接收快照先封存，再进行真值后评。重复载荷不是时间相关片段证据。

结果分支预先限定：若新臂终态仍失败，停止修补这一LOWBAND16构造并回到有终态／DIRECT正证据的OLD8；若终态形成而RAW420／MP4失败，再研究通道存留；若MP4获得可复现的正确路径优势，再推进未知相位及真实编辑。Notebook只运行当前固定实验，不自动启动后续分支。

EXECUTION_COMPLETE只表示执行完整。MSE降低、同一cap或重复载荷零误码均不表示同步成功；质量仅报告保存RGB字节的18组成对诊断。当前Python/Torch及已验证的运行方式优先，缺少实际必需组件时才修复。真实模型运行由用户执行。SOURCE_NOTE
"""
SETUP=template.SETUP.replace("Video-Local-Fourier-RM-Lowband-V1","Video-Local-Fourier-RM-Contrast-V1").replace("SC-SSTW-LOCAL-FOURIER-RM-LOWBAND-","SC-SSTW-LOCAL-FOURIER-RM-CONTRAST-").replace("('OFF','PAYLOAD_MULTI','STATE_OLD_MULTI','STATE_LOWBAND_MULTI')","('OFF','PAYLOAD_MULTI','STATE_LOWBAND_MULTI','STATE_CONTRAST_MULTI')")
_before,_after=SETUP.split('def write_json',1)
SETUP=_before+"""
setup['direction_diagnostics']={a+'/'+str(i):{'status':'NOT_RUN_SETUP'} for a in ('STATE_LOWBAND_MULTI','STATE_CONTRAST_MULTI') for i in range(25,50)}
setup['direction_calls']={}
setup['direction_diagnostic_fixed']=DIRECTION_FIXED
setup['terminal_diagnostic_fixed']=TERMINAL_FIXED
setup['space_recipe']={'status':'NOT_RUN_SETUP','truth_inputs':False}
setup['terminal_diagnostics']=dict(
 inputs={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 projections={a+'/'+f+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for f in FAMILIES for k in KEYS},
 mode_reads={a+'/'+f+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP','accepted_payload':False,'state_path_accepted':False} for a in ARMS for f in FAMILIES for k in KEYS for m in MODES},
 path_posthoc={a+'/'+f+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP'} for a in ARMS for f in FAMILIES for k in KEYS for m in MODES},
 edge_posthoc={a+'/'+f+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for f in FAMILIES for k in KEYS},
 counts={},calls={},role='writer diagnostic only; never primary blind input')
"""+'def write_json'+_after
ENVIRONMENT=template.ENVIRONMENT.replace('video_local_fourier_rm_lowband_v1_run','video_local_fourier_rm_contrast_v1_run')
RUN=template.RUN.replace('video_local_fourier_rm_lowband_v1_run','video_local_fourier_rm_contrast_v1_run')
DISPLAY=template.DISPLAY
DISPLAY=DISPLAY.replace("print('Shared before25:',result.get('before_step25_identity'))","""print('Shared before25:',result.get('before_step25_identity'))
space=result.get('space_recipe',{});receipt=space.get('receipt',{})
print('Frozen space: status=',space.get('status'),'rank=',receipt.get('rank'),'recipe=',space.get('path'),'basis=',space.get('basis_path'))
print('Direction counts/calls:',result.get('direction_counts'),result.get('direction_calls'))
for name,row in result.get('direction_diagnostics',{}).items():
    if name.endswith(('/25','/49')):
        print('Direction diagnostic:',name,'status=',row.get('status'),'source=',row.get('source'),'candidate effect values=',row.get('candidate_effect_values'))
terminal=result.get('terminal_diagnostics',{})
print('Writer terminal diagnostic counts/calls:',terminal.get('counts'),terminal.get('calls'))
for name,row in terminal.get('path_posthoc',{}).items():
    print('Writer terminal only:',name,row['status'],'rank=',row.get('true_path_rank'),'Delta=',row.get('true_path_delta'),'matches=',row.get('canonical_path_matches'))
""")

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('published immutable source SHA required')
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_contrast_v1.json').read_text())
    fixed=cfg['fixed_denominator'];direction=cfg['direction_diagnostic_fixed'];terminal=cfg['terminal_diagnostic_fixed']
    note='Published source: '+source_sha+'.' if source_sha else 'UNPUBLISHED DRAFT: bind published immutable source SHA before running.'
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',MARKDOWN.replace('SOURCE_NOTE',note)),('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\nDIRECTION_FIXED = {direction!r}\nTERMINAL_FIXED = {terminal!r}\n'+SETUP),('code',ENVIRONMENT),('code',RUN),('code',DISPLAY)]
    cells=[]
    for i,(kind,text) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'local-fourier-rm-contrast-v1-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT',execution='user-run only; no agent real model execution')))
    Path(output).write_text(json.dumps(value,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
