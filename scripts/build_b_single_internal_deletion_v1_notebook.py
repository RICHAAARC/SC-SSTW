"""Build Card103 notebook/ordinary editable ZIP; never execute the user task."""
from pathlib import Path
import ast,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
STEM='b_single_internal_deletion_v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Single-Internal-Deletion-V1/fixed-viewed-development-v1'
NEW_FILES=[
 'main/tube_state/'+STEM+'.py',
 'runtime/wan/'+STEM+'.py',
 'runtime/wan/b_single_internal_deletion_backend_v1.py',
 'runtime/wan/b_single_internal_deletion_evaluation_v1.py',
 'experiments/wan_state_clock/'+STEM+'_run.py',
 'experiments/wan_state_clock/configs/'+STEM+'.json',
 'experiments/wan_state_clock/configs/'+STEM+'_budget.json',
 'scripts/build_'+STEM+'_notebook.py',
 'tests/test_'+STEM+'.py','docs/'+STEM+'.md']

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

SETUP='\nimport subprocess, sys, time\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub,transformers,sentencepiece,ftfy; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan pipeline and VAE\')\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'-qq\',\'update\'], \'Prepare media dependency\')\n        command([\'apt-get\',\'-qq\',\'install\',\'-y\',\'ffmpeg\'], \'Install FFmpeg\')\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        raise RuntimeError(\'ffmpeg/ffprobe unavailable; see \'+str(LOG))\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_single_internal_deletion_v1_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress[\'returned\'], \'/\', progress[\'planned\'], \'active\', progress[\'active\'], flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        process.terminate()\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n        print(\'Stopped; saved attempts consumed. Later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'

DISPLAY=r'''
for name in ('result.json','thresholds.json','evaluation.json'):
    p=OUTPUT/name
    if not p.is_file(): print('Not available:',name); continue
    try: value=json.loads(p.read_text())
    except (OSError,ValueError) as error: print('Read failure',name,str(error));continue
    if name=='result.json':print('Execution',value.get('status'),'counts',value.get('counts'),'unresolved numeric',value.get('numeric_unresolved_files'))
    elif name=='thresholds.json':print('New complete-domain tau103',value.get('status'),value.get('tau'),'legacy reference',value.get('legacy_tau_reference'),'hosts',value.get('hosts'))
    else:
        print('VIEWED development fixed denominator',value.get('fixed_evaluation'))
        print('Observed totals',value.get('totals'))
        print('Primary POSITIVE-ONLY supported-canonical coverage (2queries/64slots); literal separately',value.get('comparisons_positive_only'))
        print('All-query comparisons include payload-bearing P0 nulls; not positive benefit',value.get('comparisons_all_queries'))
        print('Fixed query-class denominators',{k:{n:r.get(n) for n in ('query_slots','message_slots')} for k,r in value.get('comparisons_by_query_class',{}).items()})
        for host,row in value.get('hosts',{}).items():
            print(host,'physical P0/PM oracle support pair',row.get('physical_P0_PM_true_phase_pair',{}).get('counts'))
            for q,r in row.get('queries',{}).items():
                print(host,q,'positive',r.get('positive'),'detected',r.get('detected'),'true-path-in-C',r.get('true_path_in_C'),'C pairs/paths',r.get('C_pairs'),r.get('C_paths'),'ranges',r.get('ranges'))
for p in sorted((OUTPUT/'quality').rglob('*.json')) if (OUTPUT/'quality').is_dir() else ():
    try:
        r=json.loads(p.read_text());print('Quality',str(p.relative_to(OUTPUT)),r.get('whole'),r.get('psnr_db_range1'),'adjacent',r.get('adjacent_residual'),'quality_PASS',r.get('quality_pass'))
    except (OSError,ValueError) as error:print('Read failure',str(p),str(error))
if (OUTPUT/'ledger.json').is_file():
    try:
        ledger=json.loads((OUTPUT/'ledger.json').read_text())
        unfinished=[(k,v.get('status'),v.get('error'),v.get('unavailable_from')) for k,v in ledger.items() if v.get('status')!='RETURNED']
        print('Fixed unreturned slots',len(unfinished),'/',len(ledger),'first20',unfinished[:20])
    except (OSError,ValueError) as error:print('Ledger read failure',str(error))
print('Coarse intact-anchor task only; middle seam unobserved, source hosts viewed, no CFAR/independent confirmation/quality PASS. All reject/erase is not success.')
'''
def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[]))
    md("""# Card103：一次真实内部连续删帧，固定双端观测

四个Card102来源都已看过，本轮是viewed-source development，不是新host确认。复用现有12份MP4解码RGB缓存；在像素上固定单段删除，再单次x264编解码，绝不删Q/latent冒充。
公共121帧收到序列只取首尾45帧anchor，各四phase E41；公共安全域限制删缝不穿anchor，只能检验coarse jump。中间未观察，不能宣称精确接缝或所有删帧鲁棒。anchor像素身份/顺序完整也不保证二次codec/局部因果VAE响应与缝独立。
完整域每query1891端点对/58621含缝路径；同80个同步特征取min，两cal来源10null全族最大值冻结新tau后才处理eval来源。旧tau .5368688672仅独立anchor参考；同观察continuous/unaligned与左右独立137候选参考全部保留。
先冻结全路径共同cell，再查原缓存位，一致才读；不按消息择路或用true seam选窗。支持正确归位与字面吻合分列，裁剪边界与内部删段缺失分列。

固定4例(a,b,d)=(1,74,59)/(18,53,23)/(2,74,58)/(29,57,25)，只给私有媒体生成器，receiver只收像素/key/公共协议/冻结背景。两eval正例64槽=56物理存在+8整组缺失；其中40名义可观察、16存在未观察。32seam别名不是32独立试验。
预算706槽：generation/DiT/native/gradient/D/新fit均0；VAE1/E96/FFT48/Q192/votes96，codec encode/probe/decode各12，原单cellCounter7680，新groupCounter0，6576S/45384pair，质量16对。GPU由用户自己Run all，本包未执行真实观测。
保留旧136/65开发负例、Card102 12/12 trueC与0/60null、gatedB0无新增覆盖、物理0/960→0/960及s04 35.51dB/Linf.918代价。没有调参救分、质量PASS或科学PASS。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    source=(root/'runtime/wan/b_single_internal_deletion_v1.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper=ast.get_source_segment(source,node)
    bootstrap='from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper
    bootstrap+="""
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-single-internal-deletion-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1',
            'https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_single_internal_deletion_v1_run.py').is_file():
        raise FileNotFoundError('Available source lacks Card103 runner: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Available source',SOURCE_ROOT)
else:
    print('Existing attempt: read-only; no source/setup/physical reentry')
"""
    code(bootstrap)
    md('一次持久化尝试；中断/失败消耗预算，后续Run all只读取保存结果。无hash、B64、manifest、版本相等或GPU型号门禁。')
    code(SETUP)
    md('只读已保存输出。整组缺失、存在未观察、部分cell、定位未决、技术失败分别保留；所有引用是条件替代，不是多条独立真路径。')
    code(DISPLAY)
    for i,c in enumerate(cells):c['id']='b-single-internal-deletion-'+str(i)
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    folder=root/'notebooks';folder.mkdir(exist_ok=True);notebook=folder/(STEM+'.ipynb')
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive=folder/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in source_files(root)+[str(notebook.relative_to(root))]:z.write(root/name,name)
    return notebook,archive
if __name__=='__main__':
    for p in build():print(p)
