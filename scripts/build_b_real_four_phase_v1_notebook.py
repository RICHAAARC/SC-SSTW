"""Card89 user package; notebook loads available local/GitHub source, no embedding."""
from pathlib import Path
import ast
import json
import zipfile
from scripts.build_b_real_local_wan_v1_notebook import SOURCE_FILES as PREVIOUS_FILES, SETUP as PREVIOUS_SETUP

ROOT=Path(__file__).resolve().parents[1]
STEM='b_real_four_phase_v1'
SIDECAR='experiments/wan_state_clock/data/b_real_four_phase_v1_saved.zip'
EXTRA=['main/tube_state/b_phase_centered_saved_v1.py','main/tube_state/b_real_four_phase_v1.py',
    'runtime/wan/b_real_four_phase_v1.py','runtime/wan/b_real_four_phase_evaluation_v1.py',
    'experiments/wan_state_clock/b_real_four_phase_v1_run.py',
    'experiments/wan_state_clock/configs/b_real_four_phase_v1.json',SIDECAR,
    'scripts/build_b_real_four_phase_v1_notebook.py','tests/test_b_real_four_phase_v1.py','docs/b_real_four_phase_v1.md']
SOURCE_FILES=sorted(set(PREVIOUS_FILES+EXTRA))
SETUP=PREVIOUS_SETUP.replace('experiments.wan_state_clock.b_real_local_wan_v1_run',
                           'experiments.wan_state_clock.b_real_four_phase_v1_run')
OUTPUT='/content/drive/MyDrive/Video-WM/B-Real-Four-Phase-V1/fixed-seen-prefix-completion-v1'
DISPLAY=r'''
saved={}
for name in ('result.json','checkpoint.json','ledger.json','numeric_completion.json','evaluation.json'):
    path=OUTPUT/name
    if path.is_file():
        try:saved[name]=json.loads(path.read_text())
        except (OSError,ValueError) as error:print('Unreadable saved file',name,type(error).__name__,str(error))
result=saved.get('result.json',saved.get('checkpoint.json',{}))
evaluation=saved.get('evaluation.json',{})
print('Execution',result.get('status'),'Actual new observation',result.get('actual_new_observation'))
print('New CPU',result.get('actual_new_CPU'),'Fixed denominators',result.get('fixed_denominator'))
for name,table in evaluation.get('tables',{}).items():
    print(name,'Family',table.get('family_status'),'Incomplete phases',table.get('incomplete_phases'))
    for variant,peak in table.get('variants',{}).items():
        row=peak.get('summary') or {}
        print(variant,peak.get('status'),'exact ties',peak.get('exact_max_ties'),
              'conditional phase/origin',row.get('phase'),row.get('public_origin_hypothesis'),
              'exact cells',row.get('exact_correct_cells'),'mixed unsupported',row.get('mixed_unsupported_single_cell_claims'),
              'group correct/wrong/undefined',row.get('group_correct_rows'),row.get('group_wrong_rows'),row.get('group_undefined_rows'),
              'present readable/errors',row.get('readable_present_bits'),row.get('present_errors'),
              'absent/partial source claims',row.get('absent_source_claims'),row.get('partial_source_claims'),
              'false missing-group bits',row.get('missing_group_false_coverage'))
for name,row in saved.get('ledger.json',{}).items():
    if row.get('status')!='RETURNED':print('Retained failed/interrupted/unentered',name,row)
print('No threshold, identity/phase certificate, FPR or trusted full message. Reuse is not new physical capacity.')
print('Full phase-specific support/undefined/source/message evidence: evaluation.json. Output',OUTPUT)
'''


def prepare_sidecar(root, saved_root=None):
    root=Path(root)
    saved_root=Path(saved_root) if saved_root is not None else root.parent.parent/'diagnostics/b-line-phase-centered-saved-cpu-20261021/run'
    target=root/SIDECAR
    if not saved_root.is_dir():
        if target.is_file():
            return target
        raise FileNotFoundError('actual saved Card88 blind JSON sidecar is unavailable')
    files=[f'blind/{kind}/c{i:02}/key{k:02}.json' for kind in ('Q','votes','tables') for i in range(4) for k in range(2)]
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(saved_root/name,name)
        bundle.write(saved_root/'evaluation.json','post/card88_evaluation.json')
    return target


def build(root=ROOT,saved_root=None):
    root=Path(root)
    prepare_sidecar(root,saved_root)
    cells=[]
    def md(value):cells.append(dict(cell_type='markdown',metadata={},source=value.splitlines(keepends=True)))
    def code(value):
        ast.parse(value)
        cells.append(dict(cell_type='code',metadata={},source=value.splitlines(keepends=True),outputs=[],execution_count=None))
    md('''# B实际四相位补齐：Card89用户运行包

Run all直接从GitHub现有`dev/local-joint-state-payload-v1`分支加载源码及保存JSON，无需上传ZIP。
若`/content/SC-SSTW-b-real-four-phase-v1`已有可用源码，则直接使用；本地user_bundle.zip仅作可选副本。
Notebook不内嵌源码/数据大字典，实际下载、导入或文件读取失败会直接报告。

固定用户任务：VAE加载1，新增E6/FFT6、encoder名义66，两钥新Q/票各12；D/新writer/codec/媒体decode/梯度0。
复用Card88普通JSON sidecar：旧E45前10行、280个已存候选归约不重算；6个新E41只补起点62/63/64。
16逻辑相位视图/32两钥表不等于16个新E。每查询137状态；408新候选，raw/centered/min均保留。
完整相位族缺项保持未决，不删掉失败项取峰。没有tau/标定、身份认证、总体FPR或可信完整32位。

首cell只mount Drive。已有attempt或科学残留时Run all只读，跳过源码/安装/模型，不补失败调用。
代理仅必要CPU工程/静态检查；真实E6/FFT6和完整Run all仍待用户。无摘要/版本/Git-clean/GPU型号或资源硬上限。
''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    runtime=(root/'runtime/wan/b_real_four_phase_v1.py').read_text()
    node=next(n for n in ast.parse(runtime).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper=ast.get_source_segment(runtime,node)
    bootstrap="from pathlib import Path\nimport json, subprocess\nOUTPUT=Path("+repr(OUTPUT)+")\n"+helper
    bootstrap+='''
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-real-four-phase-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch',
                        'dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',
                        str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_real_four_phase_v1_run.py').is_file():
        raise FileNotFoundError('可用源码目录中缺少Card89运行入口，请使用包含该入口的源码目录：'+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Available source',SOURCE_ROOT)
else:
    print('Existing attempt: saved records only; no source/setup/model reentry')
'''
    code(bootstrap)
    md('沿已成功Wan加载与导入修复路径执行；不会安装或运行媒体编解码工具。数据sidecar仅按成员读取，post成员在numeric marker后才读取。')
    code(SETUP)
    md('保存结果：实际E/FFT与逻辑复用分开，源cell/段组/消息/undefined和假覆盖分开；最高分不是认证。')
    code(DISPLAY)
    for i,cell in enumerate(cells):cell['id']=f'b-real-four-phase-{i:02}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    folder=root/'notebooks';folder.mkdir(exist_ok=True)
    notebook=folder/(STEM+'.ipynb');notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive=folder/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in SOURCE_FILES+[str(notebook.relative_to(root))]:bundle.write(root/name,name)
    return notebook,archive


if __name__=='__main__':
    for path in build():print(path)
