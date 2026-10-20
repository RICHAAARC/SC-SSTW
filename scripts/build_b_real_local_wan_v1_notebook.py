"""Fixed local45 user notebook with ordinary embedded editable source."""
from pathlib import Path
import ast,json,zipfile
from scripts.build_b_legacy_position_margin_v1_notebook import SOURCE_FILES as PREVIOUS_FILES
from scripts.build_b_legacy_migration_v1_notebook import SETUP as PREVIOUS_SETUP
ROOT=Path(__file__).resolve().parents[1]
STEM='b_real_local_wan_v1'
EXTRA=['main/tube_state/b_real_local_wan_v1.py','runtime/wan/b_real_local_wan_v1.py',
 'runtime/wan/b_real_local_wan_evaluation_v1.py','experiments/wan_state_clock/b_real_local_wan_v1_run.py',
 'experiments/wan_state_clock/configs/b_real_local_wan_v1.json','scripts/build_b_real_local_wan_v1_notebook.py',
 'tests/test_b_real_local_wan_v1.py','docs/b_real_local_wan_v1.md']
SOURCE_FILES=sorted(set(PREVIOUS_FILES+EXTRA))
SETUP=PREVIOUS_SETUP.replace('experiments.wan_state_clock.b_legacy_migration_v1_run','experiments.wan_state_clock.b_real_local_wan_v1_run')
DISPLAY=r"""
saved={}
for name in ('result.json','checkpoint.json','ledger.json','numeric_completion.json','evaluation.json'):
    path=OUTPUT/name
    if path.is_file():
        try:saved[name]=json.loads(path.read_text())
        except (OSError,ValueError) as exc:print('Unreadable saved record',name,type(exc).__name__,str(exc))
r=saved.get('result.json',saved.get('checkpoint.json',{}));e=saved.get('evaluation.json',{})
print('Execution',r.get('status'),'Fixed denominator',r.get('fixed_denominator'),'Calls',r.get('counts'))
for name,row in e.get('tables',{}).items():
    sequence=row.get('sequence',{})
    print(name,'arm',row.get('arm'),'singleton/mixed rows',row.get('singleton_geometry_rows'),row.get('mixed_identity_rows'),
          'provisional start',sequence.get('provisional_start'),'exact ties',sequence.get('exact_max_ties'),
          'conditional singleton correct/wrong',row.get('conditional_correct_singleton_cells'),row.get('conditional_wrong_singleton_cells'),
          'mixed unsupported claims',row.get('unsupported_mixed_singleton_claims'),
          'present readable/errors',row.get('readable_present_bits'),row.get('present_bit_errors'),
          'missing-group false coverage',row.get('conditional_false_missing_group_bits'),'undefined local targets',row.get('undefined_bit_references'))
print('Identity/phase',e.get('identity_status'),e.get('phase_status'),'trusted full message',e.get('trusted_full_messages'))
for name,row in saved.get('ledger.json',{}).items():
    if row.get('status')!='RETURNED':print('Retained failed/interrupted/unentered slot',name,row)
print('Full observations/votes: blind/. Exact geometry and all bit/source rows: evaluation.json.')
print('No floor truth for mixed cells; no identity/phase certification; no new writer/D/codec. Output',OUTPUT)
"""


def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),outputs=[],execution_count=None))
    md("""# B真实局部Wan读取：固定45帧E4（卡85）

当前已采用的用户任务：P0/PM各两片[60,105)、[61,106)，45帧全部进E，仅local latent0不读。
复用已存实际MP4 readback像素；VAEload1/E4/FFT4、两钥8读/8条件归约；D/codec/媒体decode/新writer/梯度0。
同一E供Q及FFT，两钥共享谱；L11和全部34起点不改。新272分数＋后评旧参考136＝408。
源cell身份、段号、局部raw bit、条件消息分开；32位每输入仍保留16存在/16结构缺失。

非对齐B的11名义组全mixed，精确source-cell准确率未定义，80单段兼容位/8跨段位未定义；不floor、dominant或消息救真值。
单查询不继承Card83跨窗碰撞拒读；错钥也可能有完整错误条件声称。身份/相位/FPR/可信消息未校准。
所有盲Q/分数/ties/原位票/条件消息及失败先存，之后才读取geometry/message/旧full-history参考后评。

直接Run all，无手工上传ZIP/未发布URL依赖。普通可编辑源码内置，同步提供portable_source.zip和独立CLI。
首个代码cell只Drive mount；attempt/科学残留存在则后续Run all仅显示保存结果，不补跑或覆盖源码。
无hash/manifest/版本/Git-clean/GPU型号或资源硬上限。agent仅CPU/static/fake与两固定几何检查，尚未真实执行本VAE/FFT任务。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    runtime=(root/'runtime/wan/b_real_local_wan_v1.py').read_text()
    helper=ast.get_source_segment(runtime,next(n for n in ast.parse(runtime).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts'))
    payload={name:(root/name).read_text() for name in SOURCE_FILES}
    code("from pathlib import Path\nimport json\nOUTPUT=Path('/content/drive/MyDrive/Video-WM/B-Real-Local-Wan-V1/fixed-45-frame-phase01-v1')\n"+helper+"\nEXISTING_ARTIFACTS=execution_artifacts(OUTPUT)\nif not EXISTING_ARTIFACTS:\n    SOURCE_ROOT=Path('/content/b_real_local_wan_source')\n    SOURCE_TEXT="+repr(payload)+"\n    for name,text in SOURCE_TEXT.items():\n        path=SOURCE_ROOT/name\n        path.parent.mkdir(parents=True,exist_ok=True)\n        path.write_text(text,encoding='utf-8')\n    OUTPUT.mkdir(parents=True,exist_ok=True)\n    print('Available editable source',SOURCE_ROOT)\nelse:\n    print('Existing attempt: saved records only; source/model not reentered')\n")
    md('沿已成功Wan依赖／缓存路径执行固定任务，只修复实际缺失导入。只读已存RGB像素，无FFmpeg安装或解码回退。')
    code(SETUP)
    md('保存结果：技术失败、mixed/undefined、结构缺失、错误归属与假覆盖分别保留，不能由最高分认证。')
    code(DISPLAY)
    for i,c in enumerate(cells):c['id']=f'b-real-local-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    folder=root/'notebooks';folder.mkdir(exist_ok=True)
    notebook=folder/(STEM+'.ipynb');notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive=folder/(STEM+'_portable_source.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in SOURCE_FILES:z.write(root/name,name)
    return notebook,archive

if __name__=='__main__':
    for path in build():print(path)
