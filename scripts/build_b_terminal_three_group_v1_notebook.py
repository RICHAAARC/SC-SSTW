"""Build the real user notebook and ordinary editable source ZIP; never run it."""
import ast
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_terminal_three_group_v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1'
ENTRY = f'experiments/wan_state_clock/{STEM}_run.py'
SEEDS = [ENTRY, f'scripts/build_{STEM}_notebook.py', f'docs/{STEM}.md',
         f'experiments/wan_state_clock/configs/{STEM}.json',
         f'experiments/wan_state_clock/configs/{STEM}_background.json']


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


BOOTSTRAP = r'''
from pathlib import Path
import json, subprocess, sys, tempfile, urllib.request, zipfile
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
SOURCE_REF = 'dev/local-joint-state-payload-v1'
# Optional editable source folder from the ordinary ZIP. None downloads GitHub.
SOURCE_DIRECTORY = None
'''

DOWNLOAD = r'''
EXISTING_ARTIFACTS = execution_artifacts(OUTPUT)
if EXISTING_ARTIFACTS:
    print('Existing attempt: read saved results only. No setup or scientific calls.')
else:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if SOURCE_DIRECTORY is None:
        workspace = Path(tempfile.mkdtemp(prefix='SC-SSTW-b-terminal-three-group-v1-', dir='/content'))
        archive = workspace/'source.zip'
        url = 'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/'+SOURCE_REF
        urllib.request.urlretrieve(url, archive)
        destination = (workspace/'source').resolve()
        destination.mkdir()
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (destination/member.filename).resolve()
                if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('unsafe source archive entry: '+member.filename)
            source.extractall(destination)
        entries = list(destination.glob('*/experiments/wan_state_clock/b_terminal_three_group_v1_run.py'))
        if len(entries) != 1:
            raise FileNotFoundError('New B runner unavailable in downloaded source. Publish the prepared B research-branch changes first.')
        SOURCE_ROOT = entries[0].parents[2]
    else:
        SOURCE_ROOT = Path(SOURCE_DIRECTORY).expanduser().resolve()
    required = ['experiments/wan_state_clock/b_terminal_three_group_v1_run.py',
                'main/tube_state/b_local_identity_dct_v1.py', 'main/tube_state/b_m3_time_path_v1.py',
                'main/tube_state/b_cdc32_aggregation_v1.py', 'runtime/wan/b_cdc32_bundle_v1.py']
    for relative in required:
        if not (SOURCE_ROOT/relative).is_file():
            raise FileNotFoundError('New B source file unavailable: '+str(SOURCE_ROOT/relative))
    print('Actual editable source folder:', SOURCE_ROOT)
'''

RUN = r'''
if not EXISTING_ARTIFACTS:
    import importlib, re, shutil, time
    log_path = OUTPUT/'setup.log'
    def command(args, label):
        print(label, flush=True)
        with log_path.open('a') as stream:
            stream.write(label+'\n'); stream.flush()
            value = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
        if value.returncode:
            raise RuntimeError(label+' failed; see '+str(log_path))
    mapping = {'numpy':'numpy', 'torch':'torch', 'diffusers':'diffusers', 'accelerate':'accelerate',
               'transformers':'transformers', 'safetensors':'safetensors', 'huggingface_hub':'huggingface-hub',
               'sentencepiece':'sentencepiece', 'ftfy':'ftfy'}
    repaired = set()
    while True:
        probe = subprocess.run([sys.executable, '-c',
            'import numpy,torch,accelerate,transformers,safetensors,huggingface_hub,sentencepiece,ftfy; from diffusers import WanPipeline,AutoencoderKLWan'],
            capture_output=True, text=True)
        if probe.returncode == 0:
            break
        missing = re.search("ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", probe.stderr)
        name = missing.group(1).split('.')[0] if missing else None
        if name not in mapping or name in repaired:
            raise RuntimeError('Actual dependency import failure:\n'+probe.stderr)
        repaired.add(name)
        command([sys.executable, '-m', 'pip', 'install', mapping[name]], 'Install missing '+name)
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        command(['apt-get','-qq','update'], 'Prepare codec dependency')
        command(['apt-get','-qq','install','-y','ffmpeg'], 'Install FFmpeg')
    # Probe the actual new imports in a fresh process rooted at this source.
    import_probe = "from pathlib import Path; import importlib; root=Path.cwd().resolve(); names=['runtime.wan.b_terminal_three_group_v1','runtime.wan.b_terminal_three_group_backend_v1','main.tube_state.b_local_identity_dct_v1','main.tube_state.b_m3_time_path_v1','main.tube_state.b_cdc32_aggregation_v1','runtime.wan.b_cdc32_bundle_v1']; modules=[importlib.import_module(n) for n in names]; assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules); print(*[m.__file__ for m in modules],sep='\\n')"
    subprocess.run([sys.executable, '-c', import_probe], cwd=SOURCE_ROOT, check=True)
    with (OUTPUT/'runner.log').open('w') as stream:
        worker = subprocess.Popen([sys.executable, '-u', '-m',
            'experiments.wan_state_clock.b_terminal_three_group_v1_run', '--output', str(OUTPUT)],
            cwd=SOURCE_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    try:
        while worker.poll() is None:
            time.sleep(30)
            try:
                value = json.loads((OUTPUT/'progress.json').read_text())
                print(value.get('active'), value.get('counts'), flush=True)
            except (OSError, ValueError):
                print('Same worker active; progress record not available yet.', flush=True)
    except KeyboardInterrupt:
        worker.terminate()
        try: worker.wait(timeout=30)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()
        print('Stopped. Attempt remains spent; later Run all reads saved records.')
        raise
    print('Runner exit:', worker.returncode, 'Saved output:', OUTPUT)
'''

DISPLAY = r'''
for name in ('result.json', 'thresholds.json', 'evaluation.json', 'planned_outputs.json'):
    path = OUTPUT/name
    if not path.is_file():
        print('Unavailable:', name); continue
    try: value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        print('Read failure:', name, str(error)); continue
    if name == 'result.json': print('Execution', value.get('status'), value.get('counts'), value.get('denominators'))
    elif name == 'thresholds.json': print('New DCT2-chain tau', value.get('status'), value.get('tau'), 'planned null families', value.get('planned_families'))
    elif name == 'evaluation.json':
        print('Reader totals', value.get('totals'))
        for label in ('media_P0_PM_pair', 'native_P0_PM_pair'):
            print(label, {k:v for k,v in value.get(label,{}).items() if k!='rows'})
        print('Quality pairs', len(value.get('quality',{})), 'conditional bundles',len(value.get('bundles',[])))
        print('Evidence ceiling', value.get('evidence_ceiling'))
    else: print('Fixed planned tasks', len(value.get('tasks',{})), 'bundles', len(value.get('bundles',[])))
print('R0/R1 local bits, paths, common-bit BER, coverage, false attribution and quality remain separate. No scientific or quality PASS.')
print('Failures, NOT_ENTERED, full candidate families and unresolved mappings remain at', OUTPUT)
'''


def build(root=ROOT):
    root = Path(root)
    cells = []
    def markdown(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(keepends=True), execution_count=None, outputs=[]))
    markdown('''# B：DCT2 局部身份、M3-151 时间候选、CDC32 条件聚合

本地交付使用已采用共同r1/c0与B1/B2/B3组件。GitHub研究分支的新入口尚待发布；发布后默认Run all取可用源码。也可在第二代码cell填写普通可编辑源码包目录。此notebook不会用旧runner替代缺失的新实现。

固定4新来源（2cal/2eval）、OFF/P0/PM三臂、三fixture。DCT2 k25..49，原176状态在解码前最终FP32 latent写入；完整native50、CFG5不变。12 D181、432 E41、codec各48；M3/E2共享全部观察；B3没有新增物理调用。先用新链90个完整null族一次冻结tau，再进入eval。校准缺失不缩分母或回退旧tau。

共同局部C不被路径反剪；R0/R1分别记录；CDC使用全部计划副本，冲突和技术缺失保留。每个两片段bundle独立冻结匿名计划后读缓存；SEP与条件JOIN并列，不认证同源。Rc仅诊断。净32位、三层支撑和失败分母保留。

本轮真实模型/GPU/VAE/媒体工作由用户运行。首代码cell独立挂载Drive。存在任何执行残留时，后续Run all仅读结果；失败/中断消耗计划单位，不自动续算、换源或重试。不要求GPU型号、源码摘要、精确依赖版本或Git状态。

保存盲计划与决定并回读后才开放真值与质量后评。两eval来源及d=0/8固定协议不能证明一般编辑、总体FPR、源认证、新旧载体优劣或原生成期时间控制承诺。
''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    runner = (root/f'runtime/wan/{STEM}.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(runner).body if isinstance(n, ast.FunctionDef) and n.name == 'execution_artifacts')
    code(BOOTSTRAP+'\n'+ast.get_source_segment(runner, node)+'\n'+DOWNLOAD)
    code(RUN)
    code(DISPLAY)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')), nbformat=4, nbformat_minor=5)
    dest = root/'notebooks'
    dest.mkdir(exist_ok=True)
    path = dest/(STEM+'.ipynb')
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    files = source_files(root)
    archive = dest/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for rel in files:
            bundle.write(root/rel, rel)
        bundle.write(path, 'notebooks/'+path.name)
    return dict(notebook=str(path), source_zip=str(archive), source_files=files,
                github_new_entry_status='PUBLICATION_PENDING', physical_execution=False)


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
