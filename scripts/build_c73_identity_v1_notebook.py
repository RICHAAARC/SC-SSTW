"""Card75 ordinary editable dev source download and matching C73 notebook."""
from pathlib import Path
import ast
import json
import zipfile
from scripts.build_local_segmented_terminal_sync_v1_notebook import SOURCE_FILES as BASE_FILES, SETUP as BASE_SETUP

ROOT = Path(__file__).resolve().parents[1]
STEM = "c73_identity_v1"
SOURCE_FILES = sorted(set(BASE_FILES + [
    "main/tube_state/c73_identity_v1.py", "runtime/wan/c73_identity_v1.py",
    "runtime/wan/c73_identity_evaluation_v1.py", "runtime/wan/c73_identity_quality_v1.py",
    "experiments/wan_state_clock/c73_identity_v1_run.py",
    "experiments/wan_state_clock/configs/c73_identity_v1.json", "docs/c73_identity_v1.md",
    "scripts/build_c73_identity_v1_notebook.py"]))
SOURCE = r"""
EXISTING_ARTIFACTS = execution_artifacts(OUTPUT)
if EXISTING_ARTIFACTS:
    print('Existing attempt: saved records only', EXISTING_ARTIFACTS)
else:
    import stat, urllib.request, zipfile
    SOURCE_LOCATION = 'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/notebooks/c73_identity_v1_portable_source.zip'
    source = Path(SOURCE_LOCATION)
    if source.is_dir():
        SOURCE_ROOT = source
    else:
        archive = source
        if SOURCE_LOCATION.startswith(('https://','http://')):
            archive = Path('/content/c73_downloaded_source.zip')
            urllib.request.urlretrieve(SOURCE_LOCATION, archive)
        SOURCE_ROOT = Path('/content/c73_identity_source')
        SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                target = (SOURCE_ROOT/info.filename).resolve()
                if not target.is_relative_to(SOURCE_ROOT.resolve()) or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('unsafe source archive path')
            bundle.extractall(SOURCE_ROOT)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/c73_identity_v1_run.py').is_file():
        raise FileNotFoundError('actual C73 runner missing from available source')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Available editable source:', SOURCE_ROOT)
"""
SETUP = BASE_SETUP.replace('experiments.wan_state_clock.local_segmented_terminal_sync_v1_run',
                           'experiments.wan_state_clock.c73_identity_v1_run')
DISPLAY = r"""
saved = {}
for name in ('result.json','checkpoint.json','ledger.json','numeric_completion.json','evaluation.json','attempt.json'):
    path = OUTPUT/name
    if path.is_file():
        try:
            saved[name] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print('Unreadable saved record:', name, type(exc).__name__, str(exc))
r = saved.get('result.json',saved.get('checkpoint.json',{}))
print('Execution:', r.get('status'), 'Fixed denominators:', r.get('fixed_denominator'))
print('Calls:', r.get('counts'))
evaluation = saved.get('evaluation.json',{})
print('Positive retained / planned3:', evaluation.get('retained_positive'))
print('All-ID false attributions:', evaluation.get('false_attributions'))
print('Positive losses/failures:', evaluation.get('positive_not_retained'))
print('Actual construction:', r.get('construction'))
print('FP32/native old22 audit:', r.get('native_audit'))
print('Quality:', r.get('quality'), 'Preclip:', r.get('preclip'))
for name, row in saved.get('ledger.json',{}).items():
    if row.get('status') != 'RETURNED':
        print('Retained failed/unentered/started slot:', name, row)
print('Code membership is not phase authentication; no gap path, FPR or full32 payload claim. FFT0.')
print('Output:', OUTPUT)
"""


def build(root=ROOT):
    root = Path(root)
    cells = []
    def md(text):
        cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type='code',metadata={},source=text.splitlines(keepends=True),execution_count=None,outputs=[]))
    md("""# C73：固定三臂身份／相位可行性运行包（卡74）

已采纳新增64维同步形状约束，目标0.5、rho<1/16；净32和A不变。
用户GPU运行：P0/S_base/S_ID全部fresh D，3D/18个九帧reset E，0 FFT/DiT/梯度/新host。
六个独立clip每臂，G只复用同臂D（含失败）。完整target与连续context分别评价。
36物理向量、1656身份槽（3正/1653负），全部失败/未定义保留；全拒读不是成功。
先持久保存全部y/s/rho/status才后评角色/真几何；码成员不等于相位认证，不恢复完整gap。
质量6表1086帧/1080邻接及3preclip。载荷层未采纳，不能评价全32退步或独立FPR。

直接Run all默认从现有dev分支的普通URL自动下载配套源码ZIP，随后执行固定流程，无需手动上传ZIP、改参数或新增代码。SOURCE_LOCATION也支持可用源码目录、本地ZIP或普通URL。
任何attempt/科学残留存在时，后续Run all只读保存结果，不补跑模型/媒体/失败槽。
无hash/manifest/版本/Git干净状态或资源硬上限。agent仅CPU核验，未运行本GPU notebook。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    runtime = (root/'runtime/wan/c73_identity_v1.py').read_text()
    node = next(n for n in ast.parse(runtime).body if isinstance(n,ast.FunctionDef) and n.name == 'execution_artifacts')
    helper = ast.get_source_segment(runtime,node)
    code("from pathlib import Path\nimport json\nOUTPUT=Path('/content/drive/MyDrive/Video-WM/B-C73-Identity-V1/fixed-seen-host-v1')\n"+helper+SOURCE)
    md('依赖及一次固定运行：仅修复实际导入/媒体依赖问题。已有attempt时跳过源码下载、依赖安装及物理调用。')
    code(SETUP)
    md('保存记录：失败/中断/未进入槽均保留；再次Run all仅显示，不恢复执行。')
    code(DISPLAY)
    for i,cell in enumerate(cells):
        cell['id'] = f'c73-identity-{i:02d}'
    nb = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',
              kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    target = root/'notebooks'
    target.mkdir(exist_ok=True)
    notebook = target/f'{STEM}.ipynb'
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive = target/f'{STEM}_portable_source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in SOURCE_FILES:
            bundle.write(root/name,name)
    return notebook,archive


if __name__ == '__main__':
    for path in build():
        print(path)
