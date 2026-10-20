"""Self-contained editable plain-source clean position-margin user notebook."""
from pathlib import Path
import ast,json,zipfile
from scripts.build_b_legacy_migration_v1_notebook import SOURCE_FILES as OLD_FILES
from scripts.build_c73_identity_v1_notebook import SETUP as OLD_SETUP
ROOT=Path(__file__).resolve().parents[1]
STEM="b_legacy_position_margin_v1"
EXTRA=["main/tube_state/b_legacy_position_margin_v1.py","runtime/wan/b_legacy_position_margin_v1.py",
       "runtime/wan/b_legacy_position_margin_evaluation_v1.py","experiments/wan_state_clock/b_legacy_position_margin_v1_run.py",
       "experiments/wan_state_clock/configs/b_legacy_position_margin_v1.json","scripts/build_b_legacy_position_margin_v1_notebook.py",
       "tests/test_b_legacy_position_margin_v1.py","docs/b_legacy_position_margin_v1.md"]
SOURCE_FILES=sorted(set(OLD_FILES+EXTRA))
SETUP=OLD_SETUP.replace("experiments.wan_state_clock.c73_identity_v1_run","experiments.wan_state_clock.b_legacy_position_margin_v1_run")
DISPLAY=r"""
saved={}
for name in ('result.json','checkpoint.json','ledger.json','evaluation.json','numeric_completion.json'):
    path=OUTPUT/name
    if path.is_file():
        try:saved[name]=json.loads(path.read_text())
        except (OSError,ValueError) as exc:print('Unreadable saved record',name,type(exc).__name__,str(exc))
r=saved.get('result.json',saved.get('checkpoint.json',{}))
print('Execution',r.get('status'),'Planned',r.get('fixed_denominator'),'Actual calls',r.get('counts'))
print('Native correction',r.get('construction'),'Native audit',r.get('native_audit'))
e=saved.get('evaluation.json',{})
print('Identity/phase',e.get('identity_status'),e.get('phase_status'),'Truth',e.get('truth_status'))
print('Canonical payload side effect only',e.get('payload_pairs'))
for layer,row in r.get('quality',{}).items():
    print('Quality',layer,row.get('status'),row.get('whole'),row.get('adjacent_residual'))
for name,row in saved.get('ledger.json',{}).items():
    if row.get('status')!='RETURNED':print('Retained failed/unentered slot',name,row)
print('Full signed score tables and votes: blind/. P0/PM contrasts: evaluation.json.')
print('No reliable identity/phase or unknown-offset/deletion/sync-gain claim. Output',OUTPUT)
"""


def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),outputs=[],execution_count=None))
    md("""# B位置签名＋宿主感知最小修正：clean跨链证据候选（卡79）

已采纳方法方向。具体单候选推荐γ=1.0、44位置×4ROI×768维；真实预算仍待主会话／用户采纳后运行。
P0仅载荷缓存终态→PM终态同步；保留B多步4×8净32载荷，候选替换旧22，不把新方向投影到旧22正交补。
旧uniform/L2/round8方向规则，使用实际norm²分母满足半空间最小修正；只float64算后一次FP32，不补投影或重试。
实际媒体候选尚未执行。score-only：保留全部44×44候选方向证据、竞争margin及并列，身份／相位UNCALIBRATED。

拟议最小新物理预算：VAEload1，P0/PM fresh D2、quant2、codec2/readback2、full181 E2、FFT2供两钥四读；无DiT/梯度/RGB8 E/local E/新编辑。
已缓存P0可作跨run背景参考；fresh P0用于本次新writer质量、方向增量和载荷副作用配对，来源/版本相等不是门禁。
clean完整规范分组载荷读只测副作用，不算未知起点、删帧、相位或同步收益。所有盲分数／原始票和失败先存后读真消息。

源码作为普通可编辑文本随本notebook提供；无需手传ZIP或未发布URL。也提供portable_source.zip和完整CLI。
第一次Run all直接固定执行；存在attempt或科学残留时后续只读，失败／中断消耗本次预算。
agent只做CPU／合成／静态检查，未执行本model/media notebook；用户Colab Run all填补这一执行缺口。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    runtime=(root/'runtime/wan/b_legacy_position_margin_v1.py').read_text()
    helper=ast.get_source_segment(runtime,next(n for n in ast.parse(runtime).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts'))
    payload={name:(root/name).read_text() for name in SOURCE_FILES}
    source="from pathlib import Path\nimport json\nOUTPUT=Path('/content/drive/MyDrive/Video-WM/B-Legacy-Position-Margin-V1/fixed-clean-transfer-v1')\n"+helper+"\nEXISTING_ARTIFACTS=execution_artifacts(OUTPUT)\nif not EXISTING_ARTIFACTS:\n    SOURCE_ROOT=Path('/content/b_legacy_position_margin_source')\n    SOURCE_TEXT="+repr(payload)+"\n    for name,text in SOURCE_TEXT.items():\n        path=SOURCE_ROOT/name\n        path.parent.mkdir(parents=True,exist_ok=True)\n        path.write_text(text,encoding='utf-8')\n    OUTPUT.mkdir(parents=True,exist_ok=True)\n    print('Available editable source',SOURCE_ROOT)\nelse:\n    print('Existing attempt: saved results only; no source/model reentry')\n"
    code(source)
    md('固定执行：沿用已成功Wan安装／缓存／VAE／FFmpeg路径，只处理实际缺失依赖。无摘要、Git干净状态、版本或GPU型号门禁。')
    code(SETUP)
    md('保存结果：执行状态与科学状态分开；全拒读、数值并列、失败、缺失均保留。')
    code(DISPLAY)
    for i,c in enumerate(cells):c['id']=f'b-position-margin-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    target=root/'notebooks';target.mkdir(exist_ok=True)
    notebook=target/(STEM+'.ipynb');notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive=target/(STEM+'_portable_source.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in SOURCE_FILES:bundle.write(root/name,name)
    return notebook,archive

if __name__=='__main__':
    for path in build():print(path)
