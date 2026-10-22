"""Build the Card101 real notebook and ordinary editable source ZIP; never run it."""
from pathlib import Path
import ast
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_background_normalized_match_v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
ENTRY = 'experiments/wan_state_clock/'+STEM+'_run.py'
NEW_FILES = [
    'main/tube_state/b_background_normalized_match_v1.py',
    'runtime/wan/b_background_normalized_cpu_v1.py',
    'runtime/wan/b_background_normalized_confirmation_v1.py',
    'runtime/wan/b_background_normalized_backend_v1.py',
    'runtime/wan/b_background_normalized_evaluation_v1.py',
    'experiments/wan_state_clock/b_background_normalized_cpu_v1_run.py',
    ENTRY,'experiments/wan_state_clock/configs/'+STEM+'.json',
    'experiments/wan_state_clock/configs/'+STEM+'_budget.json',
    'scripts/build_'+STEM+'_notebook.py','tests/test_'+STEM+'.py','docs/'+STEM+'.md']


def source_files(root=ROOT):
    """Find ordinary Python import dependencies for delivery; no identity validation."""
    root = Path(root)
    pending, included = list(NEW_FILES), set()
    while pending:
        name = pending.pop()
        if name in included:
            continue
        p = root/name
        if not p.is_file():
            raise FileNotFoundError('delivery file unavailable: '+str(p))
        included.add(name)
        for parent in p.relative_to(root).parents:
            init = parent/'__init__.py'
            if (root/init).is_file() and str(init) not in included:
                pending.append(str(init))
        if p.suffix != '.py':
            continue
        package = list(p.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(p.read_text())):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [x.name.split('.') for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (package[:len(package)-node.level+1] if node.level else []) + (node.module.split('.') if node.module else [])
                candidates = [base]+[base+x.name.split('.') for x in node.names if x.name!='*']
            for parts in candidates:
                if not parts or parts[0] not in ('main', 'runtime', 'experiments', 'scripts'):
                    continue
                for candidate in [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']:
                    if (root/candidate).is_file() and str(candidate) not in included:
                        pending.append(str(candidate))
    return sorted(included)


SETUP = r'''
import subprocess, sys, time
if not EXISTING_ARTIFACTS:
    LOG = OUTPUT/'setup.log'
    def command(args, label, check=True):
        print(label, flush=True)
        with LOG.open('a') as stream:
            stream.write('COMMAND '+repr(args)+'\n'); stream.flush()
            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode and check:
            raise RuntimeError(label+' failed; see '+str(LOG))
        return completed.returncode
    def ensure_imports(probe, packages, label):
        import re
        mapping = {p.replace('-', '_'):p for p in packages}
        repaired = set()
        while True:
            offset = LOG.stat().st_size if LOG.exists() else 0
            if not command([sys.executable, '-c', probe], 'Probe '+label, check=False):
                return
            with LOG.open('rb') as stream:
                stream.seek(offset); failure = stream.read().decode('utf-8', errors='replace')
            missing = re.search("ModuleNotFoundError: No module named [\'\"]([^\'\"]+)[\'\"]", failure)
            if "cannot import name 'FqnToConfig'" in failure and 'torchao.quantization' in failure:
                action = 'remove incompatible optional torchao'
                args = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao']
            elif missing and missing.group(1).split('.')[0] in mapping:
                package = mapping[missing.group(1).split('.')[0]]
                action = 'install missing '+package
                args = [sys.executable, '-m', 'pip', 'install', package]
            else:
                raise RuntimeError(label+' actual import failure; see '+str(LOG))
            if action in repaired:
                raise RuntimeError(label+' still fails after '+action+'; see '+str(LOG))
            repaired.add(action)
            command(args, action)
    ensure_imports('import numpy,torch', ['numpy','torch'], 'tensor dependencies')
    ensure_imports('import accelerate,safetensors,huggingface_hub,transformers,sentencepiece,ftfy; from diffusers import WanPipeline,AutoencoderKLWan',
        ['diffusers','accelerate','safetensors','huggingface-hub','transformers','sentencepiece','ftfy'], 'Wan pipeline and VAE')
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        command(['apt-get','-qq','update'], 'Prepare media dependency')
        command(['apt-get','-qq','install','-y','ffmpeg'], 'Install FFmpeg')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('ffmpeg/ffprobe unavailable; see '+str(LOG))
    with (OUTPUT/'runner.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.wan_state_clock.b_background_normalized_match_v1_run', '--output', str(OUTPUT)],
            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        while process.poll() is None:
            time.sleep(30)
            try:
                progress = json.loads((OUTPUT/'progress.json').read_text())
                print('Units', progress['returned'], '/', progress['planned'], 'active', progress['active'], flush=True)
            except (OSError,ValueError,KeyError,TypeError) as error:
                print('Progress read unavailable; observing same worker:',type(error).__name__,str(error),flush=True)
    except KeyboardInterrupt:
        process.terminate()
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        print('Stopped; saved attempts consumed. Later Run all is read-only.')
        raise
    print('Runner return code',process.returncode,'; inspect saved results and runner.log. No retry.')
else:
    print('Existing attempt: source/setup/physical execution skipped.')
'''

DISPLAY = r'''
for name in ('result.json','roles.json','background.json','thresholds.json','evaluation.json'):
    path=OUTPUT/name
    if not path.is_file(): print('Not available:',name); continue
    try: value=json.loads(path.read_text())
    except (OSError,ValueError) as error: print('Read failure',name,str(error)); continue
    if name=='result.json':print('Execution',value.get('status'),'thresholds',value.get('threshold_status'),'counts',value.get('counts'))
    elif name=='roles.json':print('Saved role assignment',value.get('assignment'))
    elif name=='background.json':print('Background',value.get('status'),{k:{n:v.get(n) for n in ('dimension','host_groups','trace_scale','covariance_condition')} for k,v in value.get('models',{}).items()})
    elif name=='thresholds.json':print('Null calibration',value.get('status'),'tau_N',value.get('tau_N'))
    else:
        print('Confirmation fixed denominators',value.get('fixed_confirmation_denominators'))
        print('Confirmation results',value.get('confirmation'))
        for host,row in value.get('hosts',{}).items():
            print(host,row.get('totals'))
            print('Physical P0-PM true-phase bit pairs',row.get('paired_P0_PM_same_true_phase_physical_payload'))
ledger={}
try:
    if (OUTPUT/'ledger.json').is_file():ledger=json.loads((OUTPUT/'ledger.json').read_text())
except (OSError,ValueError) as error:print('Ledger read failure',str(error))
for path in (OUTPUT/'receipts').rglob('*.json') if (OUTPUT/'receipts').is_dir() else ():
    try:
        row=json.loads(path.read_text());ledger[row['slot']]=row
    except (OSError,ValueError,KeyError) as error:print('Receipt read failure',path,str(error))
unfinished=[(k,v.get('status'),v.get('error'),v.get('unavailable_from')) for k,v in ledger.items() if v.get('status')!='RETURNED']
print('Unreturned fixed slots',len(unfinished),'/',len(ledger),'first20',unfinished[:20])
print('Conditional alternatives are not multiple true paths. Common support, actual selected cell, message coverage and physical BER remain separate.')
print('Background fitted on assigned calibration hosts; no role-rank, CFAR, population FPR, conditional-on-acceptance or coverage guarantee. All reject/erase is not success.')
print('Full failures/geometry/quality and original bit denominators remain saved at',OUTPUT)
'''


def build(root=ROOT):
    root = Path(root)
    cells = []
    def md(s):
        cells.append(dict(cell_type='markdown', metadata={}, source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s)
        cells.append(dict(cell_type='code', metadata={}, source=s.splitlines(keepends=True), execution_count=None, outputs=[]))
    md("""# Card101：背景归一化Q匹配固定用户任务

已见四源一次CPU开发为混合信号：正例136/384正确、12真态入C，但null检出16/60、OFF输出49、缺失组假覆盖25；相对Card100保留88/损失8/新增48。没有调参补救，不替默认，不称科学改善。65个null输出由OFF/key00的48位、OFF/key01的1位、P0/key00的16位组成；P0/PM错钥输出0。缺失组25与wrong-group33交叉，不相加。下列四新tuple尚未运行，不能把开发结果当独立确认。

Run all执行固定四个未生成tuple，一次均匀随机2校准+2确认；角色先持久化。writer/gamma1/净32/默认/A不变。
先保存两cal全部Q。只用其OFF固定子集拟合双钥40维背景均值/协方差与固定1/2球形项；模型仅工作近似，不保证全Q白化/共同gain。训练OFF也用于null标定，有训练内乐观偏差；host单位2，位置/phase不是独立null。
实际Q每h取10row×4ROI，共40维，做正向Mahalanobis归一化匹配；不使用旧min(F,V)、旧L、正例floor、P0差分、真phase或消息选位置。60null family任一137分不完整则tau不可用、确认不进入。全部cal数值/模型/tau保存后才开始任何confirm sourceprep。
C={S>=tau_N}是可检测证据集合，不是认证confidence region；M>tau_N检出。全C共同最早canonical cell先定，再逐h取自己的既有单cellbit，一致且可用才读。不换j、删C、回退max。singleton可能错；名义cell不是VAE感受野。

完整物理：pipeline4/DiT800/native400/payload grad100（writerFFT100/autograd100）；VAE4/D12/E288、receiverFFT144，codec encode/probe/decode各12，Q576。双钥背景fit1次、matched stats19728；确认原位Counter23040，matched unique-max B0最多72候选/1152组Counter、ungated/gated共享一次且truth前双保存。
固定确认72query/60null/12正例，9864H/2304消息；正例384槽=160存在+224缺失，120cell-view/960原位bit，84unique sourcecell/672bit，净32/source。保留失败/技术未决/未进入、错钥、错定位/假覆盖和全部读擦代价。

首代码单元仅Drive mount。既存attempt/roles/background/numeric/收据等执行残留时，后续Run all仅读自身记录，不下载/安装/抽角色/模型/续算。新attempt直接取可用GitHub dev源码，无manual ZIP、巨量源码内嵌、hash/manifest/version/Git-clean门禁或资源硬停机。
当前准备/旧四源CPU开发不是新独立运行；真实模型/GPU/Colab由用户执行。无CFAR、总体FPR、位置覆盖、角色随机秩界或accepted条件风险保证；全拒/全擦除不称科学成功。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    source = (root/'runtime/wan/b_background_normalized_confirmation_v1.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper = ast.get_source_segment(source,node)
    bootstrap = 'from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper
    bootstrap += """
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-background-normalized-match-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch',
            'dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_background_normalized_match_v1_run.py').is_file():
        raise FileNotFoundError('Card101 runner unavailable in actual source directory: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Available source',SOURCE_ROOT)
else:
    print('Existing attempt: saved records only; no source/setup/model reentry')
"""
    code(bootstrap)
    md('一次固定执行：只修复实际缺失导入或媒体工具；进程异常／中断不重启，所有已尝试槽保留。')
    code(SETUP)
    md('读取已保存结果；逐帧／相邻帧残差、源身份、消息、错误归因和擦除均在原文件中保留。')
    code(DISPLAY)
    for i, cell in enumerate(cells):
        cell['id'] = 'b-absolute-confirmation-'+str(i)
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    target = root/'notebooks'
    target.mkdir(exist_ok=True)
    notebook = target/(STEM+'.ipynb')
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive = target/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in source_files(root)+[str(notebook.relative_to(root))]:
            bundle.write(root/name,name)
    return notebook, archive


if __name__ == '__main__':
    for path in build():
        print(path)
