"""Build the actual user notebook only. Never execute notebook experiment cells."""
import ast
import json
import textwrap
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_framewise_qim_v1'
ENTRY = f'experiments/wan_state_clock/{STEM}_run.py'
CONFIG = f'experiments/wan_state_clock/configs/{STEM}.json'
RECOVERY_FUNCTIONS = ('load','plan','denominators','has_existing_output','recover','recovery_summary')
RECOVERY_CONSTANTS = ('ARMS','LAYERS','BUDGETS')


def recovery_source(root=ROOT):
    source = (Path(root)/f'runtime/wan/{STEM}.py').read_text()
    pieces = []
    for node in ast.parse(source).body:
        if (isinstance(node,ast.FunctionDef) and node.name in RECOVERY_FUNCTIONS or
            isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in RECOVERY_CONSTANTS for t in node.targets)):
            pieces.append(ast.get_source_segment(source,node))
    return 'from pathlib import Path\nimport json\n\n'+'\n\n'.join(pieces)+'\n'


def setup_source(root=ROOT):
    # Reuse the actually exercised dependency repairs, without old scientific entry points.
    previous = ast.parse((Path(root)/'scripts/build_b_terminal_three_group_v1_notebook.py').read_text())
    run = next(ast.literal_eval(n.value) for n in previous.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='RUN' for t in n.targets))
    setup = textwrap.dedent(run.split('# Probe the actual new imports',1)[0].split('if not EXISTING_ARTIFACTS:\n',1)[1])
    setup = setup.replace("LOG = OUTPUT/'setup.log'", "LOG = LOG_DIRECTORY/'setup.log'")
    setup = setup.replace('subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)',
                          'subprocess.run(args, cwd=SOURCE_ROOT, stdout=stream, stderr=subprocess.STDOUT)')
    return setup


def source_files(root=ROOT):
    """Static local dependency inventory for publication review; no source bundle or runtime gate."""
    root = Path(root)
    pending = [ENTRY,CONFIG,f'scripts/build_{STEM}_notebook.py',f'docs/{STEM}.md',
               'scripts/build_b_terminal_three_group_v1_notebook.py']
    found = set()
    while pending:
        rel = pending.pop()
        if rel in found: continue
        path = root/rel
        if not path.is_file(): raise FileNotFoundError('delivery dependency missing: '+str(path))
        found.add(rel)
        for parent in path.relative_to(root).parents:
            init = (parent/'__init__.py').as_posix()
            if (root/init).is_file() and init not in found: pending.append(init)
        if path.suffix != '.py': continue
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node,ast.Import): names = [n.name.split('.') for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix = (package[:len(package)-node.level+1] if node.level else [])+(node.module.split('.') if node.module else [])
                names = [prefix]+[prefix+n.name.split('.') for n in node.names if n.name!='*']
            for parts in names:
                if not parts or parts[0] not in ('main','runtime','experiments','scripts'): continue
                for candidate in (Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py'):
                    if (root/candidate).is_file() and candidate.as_posix() not in found: pending.append(candidate.as_posix())
    return sorted(found)


SETTINGS = '''
import os, stat, subprocess, sys, tempfile, time, urllib.request, zipfile
SOURCE = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1')
LOG_DIRECTORY = OUTPUT.with_name(OUTPUT.name+'-logs')
SOURCE_REF = 'dev/local-joint-state-payload-v1'
SOURCE_DIRECTORY = None  # Optional ordinary editable source folder.
SCRATCH = Path('/content/video-wm-framewise-qim-v1')
CAN_START = False
PREPARATION_ERROR = None
SOURCE_ROOT = None
try:
    EXISTING = has_existing_output(OUTPUT)
    if EXISTING:
        print('已有输出：只读恢复；跳过下载、setup、模型和科学调用。')
    else:
        needed = [SOURCE/'private/b2-m3-e01/P0_terminal.pt', SOURCE/'protocol.json']
        missing = [str(path) for path in needed if not path.is_file()]
        if missing:
            PREPARATION_ERROR = 'INPUT_MISSING: ' + ', '.join(missing)
            print(PREPARATION_ERROR)
            print('未启动。SOURCE 可指向同一既定运行的正确目录；不生成或替换输入。')
        else:
            CAN_START = True
except (OSError, ValueError) as error:
    PREPARATION_ERROR = f'{type(error).__name__}: {error}'
    print('输出/输入读取未决，未启动：', PREPARATION_ERROR)
print('固定分母：', denominators())
'''

PREPARE = '''
if CAN_START:
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        WORKSPACE = Path(tempfile.mkdtemp(prefix='run-', dir=SCRATCH))
        if SOURCE_DIRECTORY is None:
            url = 'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/' + SOURCE_REF
            archive = WORKSPACE/'source.zip'
            urllib.request.urlretrieve(url, archive)
            destination = (WORKSPACE/'source').resolve()
            destination.mkdir()
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    name = member.filename.replace('\\\\','/')
                    target = (destination/name).resolve()
                    if name.startswith('/') or any(':' in part for part in name.split('/')) or not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError('unsafe source archive path: '+member.filename)
                source.extractall(destination)
            entries = list(destination.glob('*/experiments/wan_state_clock/b_framewise_qim_v1_run.py'))
            if len(entries) != 1:
                raise FileNotFoundError('available branch source lacks the QIM runner: '+url)
            SOURCE_ROOT = entries[0].parents[2]
        else:
            SOURCE_ROOT = Path(SOURCE_DIRECTORY).expanduser().resolve()
        CONFIG = SOURCE_ROOT/'experiments/wan_state_clock/configs/b_framewise_qim_v1.json'
        for path in (CONFIG, SOURCE_ROOT/'experiments/wan_state_clock/b_framewise_qim_v1_run.py', SOURCE_ROOT/'runtime/wan/b_framewise_qim_v1_backend.py'):
            if not path.is_file(): raise FileNotFoundError('usable QIM source missing: '+str(path))
        print('使用源码：', SOURCE_ROOT)
    except Exception as error:
        CAN_START = False
        PREPARATION_ERROR = f'{type(error).__name__}: {error}'
        print('源码准备失败，未启动：', PREPARATION_ERROR)
'''

WORKER = '''
probe = 'import runtime.wan.b_framewise_qim_v1; import runtime.wan.b_framewise_qim_v1_backend; import experiments.wan_state_clock.b_framewise_qim_v1_run'
command([sys.executable, '-c', probe], 'Probe actual QIM runner/backend imports')
with (LOG_DIRECTORY/'runner.log').open('a') as stream:
    worker = subprocess.Popen([sys.executable, '-u', '-m',
        'experiments.wan_state_clock.b_framewise_qim_v1_run',
        '--source', str(SOURCE), '--output', str(OUTPUT), '--config', str(CONFIG)],
        cwd=SOURCE_ROOT, stdout=stream, stderr=subprocess.STDOUT)
try:
    while worker.poll() is None:
        time.sleep(30)
        try:
            snapshot = recovery_summary(recover(OUTPUT))
            print('记录状态:', snapshot['state_counts'], '求解:', snapshot['solver'].get('status'),
                  '迭代:', snapshot['solver'].get('iterations'), flush=True)
        except (OSError, ValueError) as error:
            print('当前记录读取未决:', str(error), flush=True)
except KeyboardInterrupt:
    worker.terminate()
    try: worker.wait(timeout=30)
    except subprocess.TimeoutExpired: worker.kill(); worker.wait()
    print('子进程已停止；已有尝试保持消耗，下次 Run all 只读。')
    print('恢复摘要:', recovery_summary(recover(OUTPUT)))
    raise
print('worker exit:', worker.returncode, '日志:', LOG_DIRECTORY/'runner.log')
'''

DISPLAY = '''
if has_existing_output(OUTPUT):
    summary = recovery_summary(recover(OUTPUT))
    print('恢复状态:', summary['status'], '保存的最终状态:', summary['saved_result_status'])
    print('停止原因:', summary['stop'], '后评:', summary['evaluation_status'])
    print('固定分母:', summary['denominators'], '逐项状态:', summary['state_counts'])
    print('调用记录（None = 未知）:')
    for name, row in summary['calls'].items(): print(name, row)
    print('求解:', summary['solver'], '原生物化:', summary['native_status'])
    print('逐层保存读量（不是重评分；None = 未知）:')
    for row in summary['time_layers']: print(row)
    print('身份支持/错误/擦除:')
    for row in summary['identity']: print(row)
    print('共同完整cell比较:', summary['identity_comparison'])
    for name, error in summary['read_errors'].items(): print('记录不可用:', name, error)
else:
    print('没有尝试记录。准备状态:', PREPARATION_ERROR or '尚未进入运行cell')
    print('固定分母:', denominators())
print('这里只显示已有记录。真实成本、媒体存留、身份保持、盲同步、存在性/null标定及画质结论没有由工程检查证明。')
print('输出目录:', OUTPUT)
'''


def build(root=ROOT):
    root = Path(root)
    cells = []
    def markdown(text): cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):
        ast.parse(text); cells.append(nbformat.v4.new_code_cell(text,execution_count=None,outputs=[]))
    markdown('''# 逐帧 RGB DCT-QIM：固定一次最小诊断

本 notebook 默认 Run all 执行已固定的一次物理包：同一个既有 DCT2 P0 latent，BASE/QIM 两臂、完整181帧；无新 DiT 生成、训练、R2 或参数扫描。源码来自研究分支 dev/local-joint-state-payload-v1，也可填写普通源码目录。

**这是交付给用户运行的入口，发布准备时未执行实验 cells。没有预填实验结果。** 首次运行需要原三组输出中的 b2-m3-e01/P0 和 protocol.json。SOURCE 可改为同一既定输入的实际目录。已有 OUTPUT 时只读记录，不重新下载/setup/调用模型或重试失败。

固定读量：2臂×4层×181帧×5lane×2副本×63位＝912240槽；身份16个E41、2560局部位槽。物理上限为 D181一次，原生head46+46次，K/KT各4201次，H264 encode/probe/decode各2次；CPU FP64求解上限4000步/7200秒。失败、缺失、未决均保留。

已知位置诊断不等于盲同步或身份认证，软分数不是校准LLR/存在概率。DCT2、E41与CDC32原语义保留；本包不构造新时间族到CDC32的资格适配。''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    markdown('## 输入与只读恢复\n先看 OUTPUT 是否已有记录；新尝试只检查既定输入路径。源码与依赖准备在下一步。')
    code(recovery_source(root)+SETTINGS)
    markdown('## 加载可用源码\n已有尝试跳过本步骤。下载普通源码或使用可编辑目录，不校验摘要或固定版本。')
    code(PREPARE)
    markdown('## 执行固定一次包\n新尝试复用已跑通的依赖修复并启动独立子进程。日志放在 OUTPUT 同级的独立日志目录，避免提前占用 OUTPUT。中断后再次 Run all 仅读收据。')
    code('if CAN_START:\n'+textwrap.indent(setup_source(root)+'\n'+WORKER,'    '))
    markdown('## 查看保存结果\n以下展示求解、分层读量与身份记录，不运行解码器/评分器，不把未知补成零。完整逐项证据保留在 OUTPUT。')
    code(DISPLAY)
    for index,cell in enumerate(cells): cell['id']=f'qim-{index:02}'
    notebook = nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'colab':{'name':STEM+'.ipynb'},'accelerator':'GPU'})
    nbformat.validate(notebook)
    path=root/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(exist_ok=True)
    nbformat.write(notebook,path)
    return dict(notebook=str(path),source_files=source_files(root),physical_execution=False,source_zip_created=False)


if __name__=='__main__': print(json.dumps(build(),ensure_ascii=False,indent=2))
