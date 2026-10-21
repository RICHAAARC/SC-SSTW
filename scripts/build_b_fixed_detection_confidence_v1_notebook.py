"""Build the Card94 real notebook and ordinary editable source ZIP; never run it."""
from pathlib import Path
import ast
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_fixed_detection_confidence_v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Fixed-Detection-Confidence-V1/fixed-2cal-2confirm-v1'
ENTRY = 'experiments/wan_state_clock/'+STEM+'_run.py'
NEW_FILES = [
    'main/tube_state/'+STEM+'.py',
    'runtime/wan/'+STEM+'.py',
    'runtime/wan/b_fixed_detection_confidence_backend_v1.py',
    'runtime/wan/b_fixed_detection_confidence_evaluation_v1.py',
    ENTRY, 'experiments/wan_state_clock/configs/'+STEM+'.json',
    'experiments/wan_state_clock/configs/'+STEM+'_budget.json',
    'scripts/build_'+STEM+'_notebook.py', 'tests/test_'+STEM+'.py', 'docs/'+STEM+'.md']


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
        process = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.wan_state_clock.b_fixed_detection_confidence_v1_run', '--output', str(OUTPUT)],
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
for name in ('result.json','thresholds.json','evaluation.json'):
    path = OUTPUT/name
    if not path.is_file():
        print('Not available:',name); continue
    try: value=json.loads(path.read_text())
    except (OSError,ValueError) as error:
        print('Read failure',name,type(error).__name__,str(error)); continue
    if name=='result.json':
        print('Execution',value.get('status'),'threshold state',value.get('thresholds_status'),'counts',value.get('counts'))
    elif name=='thresholds.json':
        print('Calibration',value.get('status'),'tau',value.get('tau'),'lambda',value.get('lambda'))
    else:
        print('Confirmation',value.get('confirmation'))
        for host,row in value.get('hosts',{}).items():
            print(host,row['role'],'null detections',row['null_detections'],'wrong singleton',row['positive_singleton_wrong_origin'],
                  'physical present/missing',row['positive_geometry_present'],row['positive_geometry_missing'],
                  'nominal/unobserved',row['positive_nominal_supported'],row['positive_present_unobserved'],
                  'readable',row['positive_readable_present'],'utility',row['utility_status'])
ledger={}
try:
    if (OUTPUT/'ledger.json').is_file(): ledger=json.loads((OUTPUT/'ledger.json').read_text())
except (OSError,ValueError) as error: print('Ledger snapshot read failure',type(error).__name__,str(error))
for path in (OUTPUT/'receipts').rglob('*.json') if (OUTPUT/'receipts').is_dir() else ():
    try:
        row=json.loads(path.read_text()); ledger[row['slot']]=row
    except (OSError,ValueError,KeyError) as error: print('Receipt read failure',path,type(error).__name__,str(error))
unfinished=[(slot,row.get('status'),row.get('error'),row.get('unavailable_from')) for slot,row in ledger.items() if row.get('status')!='RETURNED']
print('Retained unreturned slots',len(unfinished),'/',len(ledger),'first20',unfinished[:20])
print('Fixed hand-chosen sources: empirical only; no population FPR, certified location or scientific PASS.')
print('Full failures: ledger.json / receipts. P0-PM same true-phase raw BER, coverage and all erasures: evaluation.json. Output',OUTPUT)
'''


def build(root=ROOT):
    root = Path(root)
    cells = []
    def md(s):
        cells.append(dict(cell_type='markdown', metadata={}, source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s)
        cells.append(dict(cell_type='code', metadata={}, source=s.splitlines(keepends=True), execution_count=None, outputs=[]))
    md("""# B固定检出与定位拒读：Card94用户运行包

Run all执行已采纳固定2校准＋2确认新来源，三臂OFF/P0/PM、六裁剪、双钥、四实际E41、全137状态。
先保存两校准宿主盲结果，再监督计算tau/lambda并持久冻结；确认任何sourceprep均在冻结之后。
检出与候选集定位分开；集合歧义只允许全候选一致支持的组位读出。OFF无目标、错钥归属、缺失组假读分别保留。

完整成功路径：生成pipeline4、DiT800、native400、局部FFT梯度100；VAEload4、D12/E288、接收FFT288、MP4编码/探测/解码各12。
writer FFT前向100＋autograd100另计。固定手选名单只支持经验结果，不是总体FPR或认证位置保证；全拒读不是成功。
每PM正确host几何存在80／缺失112，名义支持72／存在未观测8，净消息仍32。

首代码单元仅mount Drive。已有attempt/执行残留时后续全部只读，跳过源码/依赖/模型，不续算、不补失败。
新attempt直接用本地可用源码或GitHub现有dev分支，无手工ZIP上传、源码内嵌或摘要/版本/Git状态准入。
当前包仅CPU tiny/fake与静态检查；完整Colab运行及真实tau/lambda尚未执行。发布后才可从远端取得本候选。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    source = (root/'runtime/wan'/ (STEM+'.py')).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper = ast.get_source_segment(source,node)
    bootstrap = 'from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper
    bootstrap += """
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-fixed-detection-confidence-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch',
            'dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_fixed_detection_confidence_v1_run.py').is_file():
        raise FileNotFoundError('Card94 runner unavailable in actual source directory: '+str(SOURCE_ROOT))
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
        cell['id'] = 'b-fixed-confidence-'+str(i)
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
