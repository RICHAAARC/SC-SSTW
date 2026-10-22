"""Build the Card99 real notebook and ordinary editable source ZIP; never run it."""
from pathlib import Path
import ast
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_absolute_confirmation_v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Absolute-Compatibility-V1/fixed-four-unseen-v1'
ENTRY = 'experiments/wan_state_clock/'+STEM+'_run.py'
NEW_FILES = [
    'main/tube_state/b_absolute_confirmation_v1.py',
    'main/tube_state/b_absolute_compatibility_v1.py',
    'runtime/wan/b_absolute_confirmation_v1.py',
    'runtime/wan/b_absolute_confirmation_backend_v1.py',
    'runtime/wan/b_absolute_confirmation_evaluation_v1.py',
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
        process = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.wan_state_clock.b_absolute_confirmation_v1_run', '--output', str(OUTPUT)],
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
for name in ('result.json','roles.json','thresholds.json','evaluation.json'):
    path=OUTPUT/name
    if not path.is_file(): print('Not available:',name); continue
    try: value=json.loads(path.read_text())
    except (OSError,ValueError) as error: print('Read failure',name,str(error)); continue
    if name=='result.json':print('Execution',value.get('status'),'thresholds',value.get('threshold_status'),'counts',value.get('counts'))
    elif name=='roles.json':print('Saved role assignment',value.get('assignment'))
    elif name=='thresholds.json':print('Fresh calibration',value.get('status'),'tau',value.get('tau'),'L',value.get('L'))
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
print('Random roles on a fixed four-source roster: no population FPR, conditional-on-acceptance guarantee or scientific PASS. All reject/erase is not success.')
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
    md("""# B绝对兼容集合：Card99四新来源确认运行包

Run all对固定四个未见prompt/seed，仅一次均匀选六个无序校准pair之一，在任何生成前保存角色；其余两个确认。没有重抽、扫描、失败重试或旧D1补救。
两个校准源只做三臂媒体/四实际E41/Q/T，不做接收FFT或载荷读；tau=max(0,N1,N2)、L=min(K1,K2)保存后才准备确认来源。没有继承Card98阈值。
确认C={T>=L}，检出M>tau；每组固定最早共同canonical cell，逐候选取自己的单cell译值，逐bit一致才条件读。无max回退、换cell或删候选。名义完整四帧cell不是VAE感受野保证。

完整路径：pipeline4、DiT800/native400、局部payload梯度100；writer FFT前向100/autograd100；VAE4、D12/E288、接收FFT144，MP4编码/探测/解码各12。Q576、T19728。校准不FFT；确认原位Counter23040，B0至多72个unique-max归约/1152个组位Counter，新规则不重投票。
固定确认72查询/60null/12正例、9864状态、2304消息槽；正例384=物理存在160+缺失224，120原位cell-view/960bit，84唯一源cell/672bit。净32不变。所有失败/未决/未进入保留。

全部新决定与B0 ungated/同tau gated结果落盘后才打开确认语义真值后评。物理P0-PM共同位BER与新规则/B0逻辑比较分开；随机角色只支持有限名单解释，无总体FPR或接收条件保证，不宣布科学PASS。
首代码单元仅Drive mount。已有attempt/执行残留时后续Run all只读自身记录，跳过下载/依赖/角色/模型，不续算。新运行直接下载可用GitHub源码，无手工ZIP上传或巨量源码嵌入，无摘要/版本/Git-clean门禁。
本地候选尚未发布，当前远端不能声称可取得本候选；发布需另行授权。准备验证仅CPU/static/stub，真实角色尚未抽、来源NOT_GENERATED、Colab/物理任务未执行。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    source = (root/'runtime/wan'/ (STEM+'.py')).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper = ast.get_source_segment(source,node)
    bootstrap = 'from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper
    bootstrap += """
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-absolute-confirmation-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch',
            'dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_absolute_confirmation_v1_run.py').is_file():
        raise FileNotFoundError('Card99 runner unavailable in actual source directory: '+str(SOURCE_ROOT))
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
