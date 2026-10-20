"""Card77 self-contained ordinary source text + editable ZIP, no unpublished URL."""
from pathlib import Path
import ast,json,zipfile
from scripts.build_c73_identity_v1_notebook import SOURCE_FILES as C73_FILES, SETUP as C73_SETUP
ROOT=Path(__file__).resolve().parents[1]
STEM='b_legacy_migration_v1'
EXTRA=['main/tube_state/b_legacy_migration_v1.py','runtime/wan/b_legacy_migration_v1.py',
       'runtime/wan/b_legacy_migration_evaluation_v1.py','experiments/wan_state_clock/b_legacy_migration_v1_run.py',
       'experiments/wan_state_clock/legacy_saved_reproduction_v1_run.py','experiments/wan_state_clock/legacy_reference_v1_run.py',
       'experiments/wan_state_clock/configs/b_legacy_migration_v1.json','docs/b_legacy_migration_v1.md',
       'docs/evidence/legacy_score_replay_v1.json','docs/evidence/b_saved_positive_reference_v1.json',
       'scripts/build_b_legacy_migration_v1_notebook.py','tests/test_b_legacy_migration_v1.py']
SOURCE_FILES=sorted(set(C73_FILES+EXTRA+[str(p.relative_to(ROOT)) for p in (ROOT/'main/tube_state/legacy_reference_v1').glob('*.py')]))
SETUP=C73_SETUP.replace('experiments.wan_state_clock.c73_identity_v1_run','experiments.wan_state_clock.b_legacy_migration_v1_run')
# E-only reads stored tensors; no FFmpeg/ffprobe install or tool gate.
_media_start=SETUP.index('    def ensure_media_tools():')
_media_end=SETUP.index('    with (OUTPUT/\"runner.log\")',_media_start)
SETUP=SETUP[:_media_start]+SETUP[_media_end:]
DISPLAY=r"""
saved={}
for name in ('result.json','checkpoint.json','evaluation.json','ledger.json','numeric_completion.json'):
    path=OUTPUT/name
    if path.is_file():
        try:saved[name]=json.loads(path.read_text())
        except (OSError,ValueError) as exc:print('Unreadable saved record',name,type(exc).__name__,str(exc))
r=saved.get('result.json',saved.get('checkpoint.json',{}))
print('Execution',r.get('status'),'Fixed denominators',r.get('fixed_denominator'),'Calls',r.get('counts'))
e=saved.get('evaluation.json',{})
for branch,value in e.get('branches',{}).items():print(branch,value)
print('Matched identity retention only',e.get('comparisons'))
print('Prior/current readout context; no identity gate',r.get('readout_context'))
print('History',r.get('legacy_reference_summary'))
for name,row in saved.get('ledger.json',{}).items():
    if row.get('status')!='RETURNED':print('Retained failed/unentered slot',name,row)
print('No payload/path extraction. FFT0. Prior50 cells/400 bits retained, not new results. Output',OUTPUT)
"""


def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),outputs=[],execution_count=None))
    md("""# B历史复现＋现Wan观察算子对照（卡77）

固定用户任务：新增VAEload1/E12，D/codec/FFT/writer/梯度0；复用C73同次三臂RGB8/readback及12份local-MP4盲观测。
只比较clean A/D的RGB8/MP4×full181/local9，不改变码、0.5强度、rho界、净32、B default或A。
历史21000条分数/指标已保存复现；旧projection_margin/签名/对齐/fusion仅隔离CPU参考，不称等价Wan迁移成功。
48匹配向量/2208身份槽，36新、12复用；全部失败/未定义保留，复用不是新物理成功。
先存盲读后后评；writer trace/真消息/真编辑位置/P-S差分不进盲决策。新payload/path未测。

源码作为普通可编辑文本随本地notebook提供，无B64、手传ZIP依赖或未发布URL；另有portable_source.zip/完整CLI。
已有attempt时Run all只读，跳过源码创建/依赖/模型，失败和中断不补跑；真实E由用户运行。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    runtime=(root/'runtime/wan/b_legacy_migration_v1.py').read_text();tree=ast.parse(runtime)
    helper=ast.get_source_segment(runtime,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts'))
    payload={name:(root/name).read_text() for name in SOURCE_FILES}
    source="from pathlib import Path\nimport json\nOUTPUT=Path('/content/drive/MyDrive/Video-WM/B-Legacy-Migration-V1/fixed-seen-operator-control-v1')\n"+helper+"\nEXISTING_ARTIFACTS=execution_artifacts(OUTPUT)\nif not EXISTING_ARTIFACTS:\n    SOURCE_ROOT=Path('/content/b_legacy_migration_source')\n    SOURCE_TEXT="+repr(payload)+"\n    for name,text in SOURCE_TEXT.items():\n        path=SOURCE_ROOT/name\n        path.parent.mkdir(parents=True,exist_ok=True)\n        path.write_text(text,encoding='utf-8')\n    OUTPUT.mkdir(parents=True,exist_ok=True)\n    print('Available editable source',SOURCE_ROOT)\nelse:\n    print('Existing attempt: saved results only; no source/model reentry')\n"
    code(source)
    md('固定12E执行。沿用已成功的Wan依赖/缓存路径，只修复实际缺失依赖，不设GPU型号/资源/摘要门禁。')
    code(SETUP)
    code(DISPLAY)
    for i,c in enumerate(cells):c['id']=f'b-legacy-migration-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    target=root/'notebooks';target.mkdir(exist_ok=True)
    notebook=target/(STEM+'.ipynb');notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive=target/(STEM+'_portable_source.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in SOURCE_FILES:z.write(root/name,name)
    return notebook,archive

if __name__=='__main__':
    for path in build():print(path)
