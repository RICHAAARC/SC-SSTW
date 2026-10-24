"""Build the branch-source notebook and optional editable source ZIP; never run models."""
import ast
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
STEM='b_state_observation_boundaries_v1'
ENTRY=f'experiments/wan_state_clock/{STEM}_run.py'
SEEDS=[ENTRY,f'scripts/build_{STEM}_notebook.py',f'docs/{STEM}.md']

def source_files(root=ROOT):
    """Actual Python import closure, not a provenance/admission mechanism."""
    root = Path(root)
    pending, found = list(SEEDS), set()
    while pending:
        rel = pending.pop()
        if rel in found:
            continue
        path = root/rel
        if not path.is_file():
            raise FileNotFoundError('actual delivery dependency unavailable: '+str(path))
        found.add(rel)
        for parent in path.relative_to(root).parents:
            init = (parent/'__init__.py').as_posix()
            if (root/init).is_file() and init not in found:
                pending.append(init)
        if path.suffix != '.py':
            continue
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [a.name.split('.') for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = (package[:len(package)-node.level+1] if node.level else []) + (node.module.split('.') if node.module else [])
                candidates = [prefix]+[prefix+a.name.split('.') for a in node.names if a.name != '*']
            for parts in candidates:
                if not parts or parts[0] not in ('main', 'runtime', 'experiments', 'scripts'):
                    continue
                for candidate in (Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py'):
                    if (root/candidate).is_file() and candidate.as_posix() not in found:
                        pending.append(candidate.as_posix())
    return sorted(found)

SETUP='    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub,transformers,sentencepiece,ftfy; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE\')\n'

def recovery_helpers():
    text=(ROOT/'runtime/wan/b_state_observation_boundaries_v1.py').read_text(encoding='utf-8')
    tree=ast.parse(text)
    names={'make_plan','execution_artifacts','recover_output'}
    return '\n\n'.join(ast.get_source_segment(text,n) for n in tree.body
        if (isinstance(n,ast.FunctionDef) and n.name in names) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets)))

BOOTSTRAP=r'''
from pathlib import Path
import json, os, stat, subprocess, sys, tempfile, time, urllib.request, zipfile
INPUT = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-State-Observation-Boundaries-V1/fixed-e01-e02-v1')
SOURCE_DIRECTORY = None  # Optional existing directory containing the new runner.
SOURCE_ZIP = None  # Optional ordinary local ZIP; the default downloads the B branch.
SOURCE_REF = 'dev/local-joint-state-payload-v1'
SOURCE_URL = 'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/' + SOURCE_REF
{HELPERS}
EXISTING = bool(execution_artifacts(OUTPUT))
SOURCE = None
if EXISTING:
    print('Existing fixed attempt: read-only. No source download, setup, model load or encoding.')
else:
    if SOURCE_DIRECTORY:
        SOURCE = Path(SOURCE_DIRECTORY).expanduser().resolve()
    else:
        workspace = Path(tempfile.mkdtemp(prefix='b-state-observation-source-')).resolve()
        if SOURCE_ZIP is None:
            archive_path = workspace/'source.zip'
            print('Download available B source:', SOURCE_URL, flush=True)
            urllib.request.urlretrieve(SOURCE_URL, archive_path)
        else:
            archive_path = Path(SOURCE_ZIP).expanduser().resolve()
        target = workspace/'source'
        target.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                normalized = member.filename.replace('\\','/')
                parts = normalized.split('/')
                resolved = (target/normalized).resolve()
                if normalized.startswith('/') or any(':' in part for part in parts) or not resolved.is_relative_to(target):
                    raise ValueError('Unsafe ZIP path: '+member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('ZIP symbolic links are not supported: '+member.filename)
            archive.extractall(target)
        candidates = [p.parents[2] for p in target.rglob('b_state_observation_boundaries_v1_run.py')
                      if p.parent.name == 'wan_state_clock' and p.parent.parent.name == 'experiments']
        if len(candidates) != 1:
                raise ValueError('The source archive must contain one diagnostic entry point.')
        SOURCE = candidates[0]
    entry = SOURCE/'experiments/wan_state_clock/b_state_observation_boundaries_v1_run.py'
    if not entry.is_file():
        raise FileNotFoundError('New diagnostic runner missing: '+str(entry))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Source directory:',SOURCE)
    print('Original saved input:',INPUT)
    print('Fixed new output:',OUTPUT)
    print('Maximum 12 VAE encodes; first failure stops. Repeating Run all is read-only.')
'''

RUN = "if EXISTING:\n    print('Reading existing attempt; no physical work.')\nelse:\n"+SETUP+r'''
    command([sys.executable,'-c',
        'import runtime.wan.b_state_observation_boundaries_v1 as r; '
        'import runtime.wan.b_state_observation_boundaries_backend_v1 as b; '
        'print(r.__file__); print(b.__file__)'], 'Probe actual source imports')
    args = [sys.executable,'-m','experiments.wan_state_clock.b_state_observation_boundaries_v1_run',
            '--input',str(INPUT),'--output',str(OUTPUT)]
    log = OUTPUT/'run.log'
    with log.open('a') as stream:
        worker = subprocess.Popen(args,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress = OUTPUT/'progress.json'
                    print(progress.read_text() if progress.is_file() else 'Worker starting; log: '+str(log),flush=True)
        except BaseException:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill();worker.wait()
            raise
    print('Worker exit:',worker.returncode,'Log:',log)
    if worker.returncode:
        print('Attempt stopped. The next cell reads retained rows; Run all does not retry.')
'''
RUN=RUN.replace("subprocess.run(args, stdout=stream", "subprocess.run(args, cwd=SOURCE, stdout=stream")

DISPLAY=r'''
report = recover_output(OUTPUT)
print(json.dumps({k:report[k] for k in ('status','denominators','counts','physical_counts','read_errors')},indent=2))
for slot,row in report['layers'].items():
    metric=row.get('metrics') or {}
    print(slot,row['status'],'attempts=',row.get('attempts'),
          'S=',metric.get('S'),'signed alpha=',metric.get('alpha_signed'),
          'residual energy=',metric.get('residual_energy'))
evaluation=report.get('evaluation') or {}
for slot,row in evaluation.get('pairs',{}).items():
    print('PM-P0',slot,row['status'],'raw direction=',row.get('cosine_native'),'mean gain=',row.get('mean_gain'))
print('Reference line only:',evaluation.get('tau_reference_only'))
print('Descriptive first loss:',evaluation.get('first_reference_line_loss_boundary'))
print('Saved outputs:',OUTPUT)
print('Fixed 20 layer/case rows, 800 coordinates. This is post-truth diagnosis, not blind detection or scientific PASS.')
'''


def build(root=ROOT):
    root=Path(root)
    code=["from google.colab import drive\ndrive.mount('/content/drive')\n",
          BOOTSTRAP.replace('{HELPERS}',recovery_helpers()),RUN,DISPLAY]
    for cell in code:ast.parse(cell)
    cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=c.splitlines(keepends=True)) for c in code]
    notes=[
        '# 固定观测边界诊断：e01/e02 × P0/PM\n\n对已完成三组结果做事后真值诊断。选择 Run all；第一个可执行单元只挂载 Drive，随后自动加载 GitHub 的 dev/local-joint-state-payload-v1 分支源码，无需上传 ZIP 或填写参数。也可选用第二代码单元的 SOURCE_DIRECTORY 或 SOURCE_ZIP。',
        '## 1. 加载已保存输入与 B 分支源码\n\n输入：B-Terminal-Three-Group-V1/fixed-four-new-v1。新输出：B-State-Observation-Boundaries-V1/fixed-e01-e02-v1（均在 MyDrive/Video-WM 下）。已有执行记录时跳过源码下载和安装，直接只读恢复。',
        '## 2. 执行固定的一次预算\n\n最多 12 次 VAE 编码：4 次 E181 + 8 次 E41，共 1052 输入帧。N/D 复用旧 Q；A/B/C 新编码。首个失败或中断即停止，后续行保留 NOT_ENTERED；重复 Run all 不补跑。首次使用沿用已跑通的依赖修复与 FP32 VAE。',
        '## 3. 查看保留的全部结果\n\n固定 20 个层/案例、800 个坐标。显示 Q40 分解和 PM-P0 配对方向/传递比；失败和零分母明确保留。tau 仅为参考线。本次使用已知真值定位固定窗口，不是盲恢复成功，也不能单独证明普遍因果。']
    cells=[item for note,cell in zip(notes,cells) for item in (dict(cell_type='markdown',metadata={},source=(note+'\n').splitlines(keepends=True)),cell)]
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name=STEM+'.ipynb')),nbformat=4,nbformat_minor=5)
    notebook_path=root/'notebooks'/(STEM+'.ipynb')
    notebook_path.parent.mkdir(parents=True,exist_ok=True)
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    files=source_files(root)+['notebooks/'+STEM+'.ipynb']
    bundle=root/'notebooks'/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in files:archive.write(root/rel,rel)
    return dict(status='BRANCH_SOURCE_DEFAULT',source_ref='dev/local-joint-state-payload-v1',notebook=str(notebook_path),
                bundle=str(bundle),source_files=len(files),code_cells=4)


if __name__=='__main__':print(json.dumps(build(),indent=2))
