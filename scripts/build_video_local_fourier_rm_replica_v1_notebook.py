"""Build the fixed OLD8 spatial-replica native MULTI user-run Colab notebook."""
from pathlib import Path
import argparse, ast, json, re
from scripts import build_video_local_fourier_rm_lowband_v1_notebook as template

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/video_local_fourier_rm_replica_v1_colab.ipynb"
MARKDOWN = """# OLD8 局部时间状态：双空间副本验证

直接 Run all。固定四臂 OFF／PAYLOAD_MULTI／STATE_OLD_MULTI／STATE_REPLICA_MULTI，共享同一来源、种子、初始状态与各自新复制的原生 scheduler，各运行50步，控制作用于25–49步。每个时间窗的第二组四个8×8块复制第一组的逻辑时间状态；不同时间位置仍携带不同状态。频率、实余弦基、RM32状态码及重复载荷算法不变。

组A保留原四块；组B将各块横向移动8个潜空间采样点，八块不重叠，位置在运行前固定。两副本目标各为原alpha/√2，总目标L2仍为1。联合11136个active系数取mean-MSE，eta1392抵消坐标数翻倍。先完成原OLD8，同次保存第25–49步实际物理pilot范数；新臂联合梯度逐步缩放至该范数，允许放大，明确区别于旧shrink-only cap。组间实际能量不必各半；保留每组能量、缩放和匹配误差。配对预算只用于写端，不提供给盲接收器。载荷规则不变但实际载荷增量可能变化，完整merged更新不宣称等能量。

盲接收按公开对应分别提取A/B逐窗口软响应，各乘√2后等权合并，继续用原alpha、ABS/DIFF评分、174条合法路径及固定等价类。合并使用支持交集，不按真值、消息、表现选组或权重，不跨时间平均。OLD8单组和REPLICA8合并两个公开接收协议均读取全部臂、通道和正确/错误密钥，预定比较为原臂的OLD8读出与新臂的REPLICA8读出，交叉结果全部保留；ABS/DIFF分别报告，不能事后择优宣称通过。组原始响应、支持、接收窗口坐标及合并响应全部保留；副本不是独立样本。

每臂真实VAE解码后保存一份RGB8，分别形成DIRECT_RGB8、RAW420、MP4，共12次重编码。主盲读取48份q、96份评分、16704条路径成本；24次重复载荷读取和48条消息后评。100份写端侧车与原生终态16q／32评分另存，终态不作为媒体检测输入。当前只运行完整视频g0/R44诊断；不把公开候选目录中含编辑模型等同于真实裁剪、删除、重复和速度变化已经验证。

首轮检验固定状态能量下的空间冗余与软合并能否保留时间证据，不预设提高信噪比。若终态失败，核对写入与归一化；若DIRECT正确而MP4失败，按逐组响应定位损失并结束当前候选判断，不自动加强度、加副本、换频率或改同步算法。只有MP4正确时间对应或预定等价类可被盲接收区分，才推进后续编辑同步与时间相关载荷；排名略有改善不算完成。

EXECUTION_COMPLETE只表示执行完整，所有acceptance仍未校准。重复载荷零误码不证明同步帮助恢复，画质仅保留18组成对诊断。沿用当前Python和已跑通的依赖探测，不强制GPU型号或新建venv。真实GPU／VAE／媒体实验由用户运行。SOURCE_NOTE
"""
SETUP = template.SETUP.replace(
    "Video-Local-Fourier-RM-Lowband-V1", "Video-Local-Fourier-RM-Replica-V1"
).replace(
    "SC-SSTW-LOCAL-FOURIER-RM-LOWBAND-", "SC-SSTW-LOCAL-FOURIER-RM-REPLICA-"
).replace("STATE_LOWBAND_MULTI", "STATE_REPLICA_MULTI").replace("LOWBAND16", "REPLICA8")
_before, _after = SETUP.split("def write_json", 1)
SETUP = _before + """
setup['group_observations']={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS}
setup['state_budget_schedule']={'status':'NOT_RUN_SETUP','source_arm':'STATE_OLD_MULTI','target_arm':'STATE_REPLICA_MULTI','role':'writer only; never receiver input','steps':{str(i):{'status':'NOT_RUN_SETUP','target_l2':None} for i in range(25,50)}}
setup['terminal_diagnostic_fixed']=TERMINAL_FIXED
setup['terminal_diagnostics']=dict(
 inputs={a:{'status':'NOT_RUN_SETUP'} for a in ARMS},
 group_observations={a+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for k in KEYS},
 projections={a+'/'+f+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for f in FAMILIES for k in KEYS},
 mode_reads={a+'/'+f+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP','accepted_payload':False,'state_path_accepted':False} for a in ARMS for f in FAMILIES for k in KEYS for m in MODES},
 path_posthoc={a+'/'+f+'/'+k+'/'+m:{'status':'NOT_RUN_SETUP'} for a in ARMS for f in FAMILIES for k in KEYS for m in MODES},
 edge_posthoc={a+'/'+f+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for f in FAMILIES for k in KEYS},
 counts={},calls={},role='writer diagnostic only; never primary blind input')
""" + "def write_json" + _after
ENVIRONMENT = template.ENVIRONMENT.replace("video_local_fourier_rm_lowband_v1_run", "video_local_fourier_rm_replica_v1_run")
RUN = template.RUN.replace("video_local_fourier_rm_lowband_v1_run", "video_local_fourier_rm_replica_v1_run")
DISPLAY = template.DISPLAY.replace(
    "print('Shared before25:',result.get('before_step25_identity'))",
    """print('Shared before25:',result.get('before_step25_identity'))
budget=result.get('state_budget_schedule',{})
print('State budget schedule:',budget)
for arm,row in result['generation'].items():
    for step in row.get('steps',[]):
        if arm.startswith('STATE_') and step.get('index') in (25,49):
            print('Applied state budget:',arm,step['index'],step.get('pilot'))
terminal=result.get('terminal_diagnostics',{})
print('Writer terminal diagnostic counts/calls:',terminal.get('counts'),terminal.get('calls'))
for name,row in terminal.get('path_posthoc',{}).items():
    print('Writer terminal only:',name,row['status'],'rank=',row.get('true_path_rank'),'Delta=',row.get('true_path_delta'),'matches=',row.get('canonical_path_matches'))
print('Group responses retained:',len(result.get('group_observations',{})),'primary and',len(terminal.get('group_observations',{})),'terminal records; correlated, not independent samples')
"""
)

def build(source_sha=None, output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}", source_sha):
        raise ValueError("published immutable source SHA required")
    cfg = json.loads((ROOT / "experiments/wan_state_clock/configs/video_local_fourier_rm_replica_v1.json").read_text())
    fixed, terminal = cfg["fixed_denominator"], cfg["terminal_diagnostic_fixed"]
    note = "Published source: " + source_sha + "." if source_sha else "UNPUBLISHED DRAFT: bind published immutable source SHA before running."
    rows = [
        ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"),
        ("markdown", MARKDOWN.replace("SOURCE_NOTE", note)),
        ("code", f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\nTERMINAL_FIXED = {terminal!r}\n" + SETUP),
        ("code", ENVIRONMENT), ("code", RUN), ("code", DISPLAY),
    ]
    cells = []
    for i, (kind, text) in enumerate(rows):
        cell = dict(cell_type=kind, id=f"local-fourier-rm-replica-v1-{i}", metadata={}, source=text.splitlines(keepends=True))
        if kind == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    value = dict(
        nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python3", language="python", name="python3"),
            language_info=dict(name="python"),
            candidate_binding=dict(source_sha=source_sha, status="PUBLISHED_SHA_BOUND" if source_sha else "UNPUBLISHED_DRAFT",
                execution="user-run only; no agent real model execution"),
        ),
    )
    Path(output).write_text(json.dumps(value, indent=1) + "\n")
    return Path(output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-sha")
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    print(build(a.source_sha, a.output))
