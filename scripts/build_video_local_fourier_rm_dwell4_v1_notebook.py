"""Build the fixed OLD8 four-start dwell native MULTI user-run Colab notebook."""
from pathlib import Path
import argparse, ast, json, re
from scripts import build_video_local_fourier_rm_lowband_v1_notebook as template

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/video_local_fourier_rm_dwell4_v1_colab.ipynb"
MARKDOWN = '# OLD8 / DWELL4 局部管状块时间状态对照\n\n直接 Run all。固定沿用铜壶来源 seed2026092501；这是同来源开发对照，不是独立内容确认。四臂 OFF、PAYLOAD_MULTI、STATE_OLD_MULTI、STATE_DWELL4_MULTI 各运行50个真实原生采样步，MULTI控制25–49；空间四块8×8、32个Fourier系数、四age、alpha和载荷规则不变。\n\n唯一时间改变：OLD8 的每个管状块起点s使用S[s-1]；DWELL4使用S[(s-1)//4]。四个连续起点共享码字，物理窗口的age交接仍保留；不是跨窗口平均读出。历史三窗PN驻留及四age重复已存在，这个组合是当前候选假设，不宣称时间冗余首次提出。已有STATE−PAYLOAD差含内容变化，不能据此断定快速切换是失败原因。\n\nOLD8原eta696/shrink-cap1；DWELL4同5568 active系数mean-MSE、eta696，原始梯度逐步缩放到同次OLD8实际物理pilot L2，允许放大。OLD范数零则零更新，新原始方向零但目标正则明确失败；不扫描驻留长度或强度。逐步实际norm、缩放、边界及空间能量、payload和merged范数全部保留。仅pilot匹配，不宣称merged预算相同；预算记录只用于写端。\n\n每臂保存同一RGB8并形成DIRECT_RGB8、RAW420、MP4，共12次真实VAE重编码。每个通道/密钥仅提取一个物理局部q，共24份；以OLD8与DWELL4两套时间模板产生48份协议投影视图、96个ABS/DIFF评分、16704条合法路径成本。两视图共享同一观测，不是独立样本。另保存100份写端侧车、终态诊断，均不作为主盲检测输入。24次重复载荷读取、48条消息后评不参与路径选择。\n\n运行前固定解释：原臂对应OLD8、新臂对应DWELL4；交叉协议、OFF、PAYLOAD和错误密钥全部报告。ABS与DIFF分别报告，不能事后择优。每模式保留174条合法路径、精确等价类、歧义和拒绝。只有预定真实路径/精确等价集合在MP4中被区分才构成该模式的固定样例正证据；真路径仅进入封存后的评价。rank1但并列不等同唯一恢复，驻留导致的近似路径不事后算正确。信号增大、排名改善和重复载荷零误码不能替代同步成立。\n\n当前只验证完整视频g0/R44；不声称实际时域编辑、时间相关片段载荷和序列归因已闭合。CPU构造可分不能替代媒体存活。EXECUTION_COMPLETE仅执行完整，acceptance保持未校准，无自动科学PASS。沿用当前Python及成功依赖探测，不要求新venv或特定GPU。真实GPU执行由用户完成。SOURCE_NOTE\n'

SETUP = template.SETUP.replace(
    "Video-Local-Fourier-RM-Lowband-V1", "Video-Local-Fourier-RM-Dwell4-V1"
).replace(
    "SC-SSTW-LOCAL-FOURIER-RM-LOWBAND-", "SC-SSTW-LOCAL-FOURIER-RM-DWELL4-"
).replace("STATE_LOWBAND_MULTI", "STATE_DWELL4_MULTI").replace("LOWBAND16", "DWELL4")
_before, _after = SETUP.split("def write_json", 1)
SETUP = _before + """
setup['group_observations']={a+'/'+c+'/'+k:{'status':'NOT_RUN_SETUP'} for a in ARMS for c in CHANNELS for k in KEYS}
setup['state_budget_schedule']={'status':'NOT_RUN_SETUP','source_arm':'STATE_OLD_MULTI','target_arm':'STATE_DWELL4_MULTI','role':'writer only; never receiver input','steps':{str(i):{'status':'NOT_RUN_SETUP','target_l2':None} for i in range(25,50)}}
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
ENVIRONMENT = template.ENVIRONMENT.replace("video_local_fourier_rm_lowband_v1_run", "video_local_fourier_rm_dwell4_v1_run")
RUN = template.RUN.replace("video_local_fourier_rm_lowband_v1_run", "video_local_fourier_rm_dwell4_v1_run")
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

DISPLAY=DISPLAY.replace("print('Failures:',result['failures'])", """for name,row in result['path_posthoc'].items():
    print('Exact path/class:',name,'unique truth=',row.get('unique_exact_true_path'),'exact class=',row.get('top_equals_true_exact_class'),'top=',row.get('top_catalog_indices'))
print('Failures:',result['failures'])""")

def build(source_sha=None, output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}", source_sha):
        raise ValueError("published immutable source SHA required")
    cfg = json.loads((ROOT / "experiments/wan_state_clock/configs/video_local_fourier_rm_dwell4_v1.json").read_text())
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
        cell = dict(cell_type=kind, id=f"local-fourier-rm-dwell4-v1-{i}", metadata={}, source=text.splitlines(keepends=True))
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
