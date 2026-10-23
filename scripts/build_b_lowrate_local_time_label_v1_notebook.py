"""Build actual notebook from fixed runner; available editable GitHub source, no integrity gates."""
from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[1]
OUTPUT='/content/drive/MyDrive/Video-WM/B-Lowrate-Local-Time-Label-V1/fixed-viewed-development-v1'
SETUP = '\nimport subprocess, sys, time\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'update\'], \'refresh package metadata for missing FFmpeg\')\n        command([\'apt-get\',\'install\',\'-y\',\'ffmpeg\'], \'install missing ffmpeg/ffprobe\')\n        if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n            raise RuntimeError(\'ffmpeg/ffprobe unavailable after dependency installation; see \'+str(LOG))\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE dependencies\')\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_lowrate_local_time_label_v1_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress.get(\'counts\',{}).get(\'RETURNED\',0), \'/\', progress[\'planned\'], \'counts\', progress.get(\'counts\',{}), \'active\', progress.get(\'active\'), flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        process.terminate()\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n        print(\'Stopped; saved attempts consumed. Later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'
NEW_FILES=["main/tube_state/b_lowrate_local_time_label_v1.py","main/tube_state/b_lowrate_local_time_label_plan_v1.py","runtime/wan/b_lowrate_local_time_label_backend_v1.py","runtime/wan/b_lowrate_local_time_label_v1.py","runtime/wan/b_lowrate_local_time_label_evaluation_v1.py","experiments/wan_state_clock/b_lowrate_local_time_label_v1_run.py","experiments/wan_state_clock/configs/b_lowrate_local_time_label_v1.json","experiments/wan_state_clock/configs/b_lowrate_local_time_label_v1_budget.json","scripts/build_b_lowrate_local_time_label_v1_notebook.py","tests/test_b_lowrate_local_time_label_v1.py","docs/b_lowrate_local_time_label_v1.md"]
def source_files(root=ROOT):
    root=Path(root);pending=list(NEW_FILES);included=set()
    while pending:
        name=pending.pop()
        if name in included:continue
        p=root/name
        if not p.is_file():raise FileNotFoundError('delivery file unavailable: '+str(p))
        included.add(name)
        for parent in p.relative_to(root).parents:
            init=parent/'__init__.py'
            if (root/init).is_file() and str(init) not in included:pending.append(str(init))
        if p.suffix!='.py':continue
        package=list(p.relative_to(root).parts[:-1])
        for n in ast.walk(ast.parse(p.read_text())):
            candidates=[]
            if isinstance(n,ast.Import):candidates=[a.name.split('.') for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                base=(package[:len(package)-n.level+1] if n.level else [])+(n.module.split('.') if n.module else [])
                candidates=[base]+[base+a.name.split('.') for a in n.names if a.name!='*']
            for parts in candidates:
                if not parts or parts[0] not in ('main','runtime','experiments','scripts'):continue
                for candidate in (Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py'):
                    if (root/candidate).is_file() and str(candidate) not in included:pending.append(str(candidate))
    return sorted(included)


def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[]))
    md('''# 低净码率局部时间标签 V1 — 固定开发验证
6bit ID1..21分别表示8帧名义单元，32chip RM(1,5)/r7 +独立16chip周期pilot。normalized LLL ch4..15/g1..21，alpha=.15，balanced orthogonal Walsh PN，新方法补定，不是旧论文结果修复。原净32位不重写。只写一次delta，不加幅、补偿、扫描或训练；native保护user ch0..3及边界，实际VAE耦合仍须检验。
局部52真实pixel帧→固定8相位各E45(T12)，读Haar j1..4。pilot全部64signed correlation均值唯一最大排序；该候选还须>=3行all16sign通过，RM不救phase。码字正确不是检测，>=2一致行的origin仅为CONDITIONAL_UNCALIBRATED临时提议；保留并列、未知、混合、缺失与错误合法ID。
4已见开发源，不是独立确认；私有crop origin48,49,50,51由制备器使用，receiver不收真origin/ID。所有raw、blind、payload保存/回读后才postjoin已制备几何与message。真实局部E不由full-latent切窗替代。
P0/TAG同runtime D8/quant8/MP4各8，fullE8，OFF历史actual MP4仅背景；localE45=96，E总104。FFT72/keyed payload80/Counter11776；1608raw+24family/96final slots。localTAG/correctkey128行仅16nominalFULL、112MIXED，不能事后删混合。原32退步、错钥背景、missing-ID误接受、全图/时序质量另报，无PASS/FPR或完成时间路径的主张。
Notebook直接从RICHAAARC/SC-SSTW现有dev/local-joint-state-payload-v1分支加载可用源码，Run all执行固定任务，无需手动上传ZIP，无摘要、版本或Git-clean门禁。GPU由用户运行；代理只做CPU/静态/stub。
一次attempt，STARTED先持久化，失败/中断消费；后续Run all仅保存JSON只读，不重试物理。''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    text=(root/'runtime/wan/b_lfg_posthoc_reference_v1.py').read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts');helper=ast.get_source_segment(text,node)
    code('from pathlib import Path\nimport json,subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper+'''
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT) or (OUTPUT/'families').exists()
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-lowrate-local-time-label-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_lowrate_local_time_label_v1_run.py').is_file():
        raise FileNotFoundError('Available source lacks low-rate entry; publish authorized source or use editable SOURCE_ROOT: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
else:
    print('Existing scientific attempt: skip source/setup/model/physical; saved JSON read-only.')
''')
    md('固定执行。仅修复真实缺失依赖和导入故障，不绑定精确版本或GPU型号。')
    code(SETUP)
    md('保存结果只读，缺失与失败仍属于固定分母。')
    code('''
for name in ('result.json','evaluation.json'):
    path=OUTPUT/name
    if not path.is_file():print('Unavailable',name);continue
    try:value=json.loads(path.read_text())
    except (OSError,ValueError) as e:print('Saved read error',name,str(e));continue
    if name=='result.json':print('Execution',value.get('status'),'physical calls',value.get('kinds_started'),'fixed',value.get('fixed_denominators'))
    else:
        print('Tag',value.get('tag_summary'));print('Local final',value.get('local_final'));print('Families',value.get('families'));print('Original32',value.get('payload'));print('Truth issues',value.get('issues'))
print('Costs: construction/. Quality/temporal/clipping: quality/. All failure receipts retained. Output:',OUTPUT)
print('Conditional provisional origin is not authentication or completed time path; four sources are seen development.')
''')
    for i,c in enumerate(cells):c['id']='lowrate-local-'+str(i)
    path=root/'notebooks/b_lowrate_local_time_label_v1.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'))),ensure_ascii=False,indent=1)+'\n')
    return path
if __name__=='__main__':print(build())
