"""Build the actual fixed user notebook from available GitHub source, no embedded source."""
from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[1]
STEM='b_block2_carrier_v1'
NEW_FILES=['main/tube_state/b_block2_carrier_v1.py','runtime/wan/b_block2_carrier_tensor_v1.py','runtime/wan/b_block2_carrier_backend_v1.py','runtime/wan/b_block2_carrier_v1.py','runtime/wan/b_block2_carrier_evaluation_v1.py','experiments/wan_state_clock/b_block2_carrier_v1_run.py','experiments/wan_state_clock/configs/b_block2_carrier_v1.json','experiments/wan_state_clock/configs/b_block2_carrier_v1_budget.json','scripts/build_b_block2_carrier_v1_notebook.py','tests/test_b_block2_carrier_v1.py','docs/b_block2_carrier_v1.md']
OUTPUT='/content/drive/MyDrive/Video-WM/B-Block2-Carrier-V1/fixed-viewed-full-v1'
SETUP = '\nimport subprocess, sys, time\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'update\'], \'refresh package metadata for missing FFmpeg\')\n        command([\'apt-get\',\'install\',\'-y\',\'ffmpeg\'], \'install missing ffmpeg/ffprobe\')\n        if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n            raise RuntimeError(\'ffmpeg/ffprobe unavailable after dependency installation; see \'+str(LOG))\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE dependencies\')\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_block2_carrier_v1_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress.get(\'counts\',{}).get(\'RETURNED\',0), \'/\', progress[\'planned\'], \'counts\', progress.get(\'counts\',{}), \'active\', progress.get(\'active\'), flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        process.terminate()\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n        print(\'Stopped; saved attempts consumed. Later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'

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
    md('''# 固定2×2载波与随机载波：full-only用户运行包
四个已见host、原P0独立分叉RANDOM/BLOCK2；只换同步carrier vectors，同key dither/pilot/mask及QIM/RM/pilot/强度不变。OFF复用旧实际readback只新E作null；P0/R/B新同runtime D/量化/MP4/E。净32用户payload与A/default不变。
固定VAE1、D12、MP4 encode/probe/decode各12、E181×16、FFT12供两钥24读/Counter9216；预计552 decoder/736 encoder chunks。4224 tag行全保留，读出202752个64D dots，构造另33792 dots；仅同family同chip错钥96 dots。不做E41、phase/path、加幅/缩放、阈值调参、方向筛选或重试。
本包仅完成CPU/合成/stub工程核验；没有真实模型/GPU/Colab/媒体结果。四源皆VIEWED，无独立确认、FPR认证或科学PASS；旧tau仅描述。
运行源码来自GitHub的dev/local-joint-state-payload-v1分支。源码发布不等于实验已执行；GPU仍由用户Run all启动。Notebook直接clone可用源码，不要求手动ZIP、B64、摘要/manifest/版本/Git-clean门禁。
一次attempt和每call STARTED先落盘，失败/中断消费；后续Run all只读已有结果，不下载源码、装依赖、续算或重试。缺输入/模型失败留槽，不用旧媒体代替新P0或自动decode OFF。全部raw/payload/final保存并回读后才打开writer bits及quality后评；tag失败不跳payload。
''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    tree=ast.parse((root/'runtime/wan/b_block2_carrier_v1.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper=ast.get_source_segment((root/'runtime/wan/b_block2_carrier_v1.py').read_text(),node)
    bootstrap='from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper+'''
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-block2-carrier-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_block2_carrier_v1_run.py').is_file():
        raise FileNotFoundError('Available source lacks the block2 entry: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Available editable source',SOURCE_ROOT)
else:
    print('Existing attempt/artifacts: saved JSON read-only, no setup/model/physical reentry.')
'''
    code(bootstrap)
    md('运行固定实验。导入失败只修具体依赖；失败/中断的科学调用不重试。GPU由用户Run all启动。')
    code(SETUP)
    md('只读结果：标签、实际扰动、载荷退步与画质分别报告，不据此自动选方案。')
    code('''
for name in ('result.json','evaluation.json'):
    p=OUTPUT/name
    if not p.is_file():print('Unavailable',name);continue
    try:v=json.loads(p.read_text())
    except (OSError,ValueError) as e:print('Saved read failure',name,str(e));continue
    if name=='result.json':print('Execution',v.get('status'),'calls',v.get('kinds_started'),'technical tag rows',v.get('technical_tag_rows'))
    else:
        print('Fixed denominators',v.get('fixed_denominators'))
        for k,x in v.get('tag_summary',{}).items():print('Tag',k,x)
        print('Payload',v.get('payload_total'),'truth issues',v.get('reference_issues'))
print('Construction receipts: actual FP32 energy/RMS/Linf, coefficients and projection errors in construction/.')
print('Quality24 paired preclip/RGB8/MP4 outputs and clipping in quality/; no adopted quality PASS threshold.')
print('All failures remain; full clean-span payload is conditional, not tag-guided. No local/unknownphase/FPR/independent confirmation claim. Output:',OUTPUT)
''')
    for i,c in enumerate(cells):c['id']='b-block2-'+str(i)
    path=root/'notebooks'/('b_block2_carrier_v1.ipynb');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'))),ensure_ascii=False,indent=1)+'\n')
    return path
if __name__=='__main__':print(build())
