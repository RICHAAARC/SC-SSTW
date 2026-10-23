"""Build the actual local stage1 notebook; direct available GitHub source, no embedded source."""
from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[1]
STEM='b_lfg_posthoc_reference_v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-LFG-Posthoc-Reference-V1/fixed-viewed-stage1-v1'
NEW_FILES=['main/tube_state/b_lfg_bch_reference_v1.py','main/tube_state/b_lfg_posthoc_reference_v1.py','main/tube_state/b_lfg_posthoc_plan_v1.py','runtime/wan/b_lfg_posthoc_fixture_v1.py','runtime/wan/b_lfg_posthoc_tensor_v1.py','runtime/wan/b_lfg_posthoc_backend_v1.py','runtime/wan/b_lfg_posthoc_reference_v1.py','runtime/wan/b_lfg_posthoc_evaluation_v1.py','experiments/wan_state_clock/b_lfg_posthoc_reference_v1_run.py','experiments/wan_state_clock/configs/b_lfg_posthoc_reference_v1.json','experiments/wan_state_clock/configs/b_lfg_posthoc_reference_v1_budget.json','scripts/build_b_lfg_posthoc_reference_v1_notebook.py','tests/test_b_lfg_posthoc_reference_v1.py','docs/b_lfg_posthoc_reference_v1.md']
SETUP = '\nimport subprocess, sys, time\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'update\'], \'refresh package metadata for missing FFmpeg\')\n        command([\'apt-get\',\'install\',\'-y\',\'ffmpeg\'], \'install missing ffmpeg/ffprobe\')\n        if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n            raise RuntimeError(\'ffmpeg/ffprobe unavailable after dependency installation; see \'+str(LOG))\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE dependencies\')\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_lfg_posthoc_reference_v1_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress.get(\'counts\',{}).get(\'RETURNED\',0), \'/\', progress[\'planned\'], \'counts\', progress.get(\'counts\',{}), \'active\', progress.get(\'active\'), flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        process.terminate()\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n        print(\'Stopped; saved attempts consumed. Later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'

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
    md('''# LFG公开公式 Post-hoc K=0：阶段1固定消息通道
本包只做Wan终态一次正交3D Haar LLL扩频、相关符号和BCH。不是时间ID/局部化/状态空间或真实多步同步。没有作者源码，仅称公开公式参考实现，不宣称论文等价。原用户净32位仍另读退步，不把新固定432位诊断消息叫增容。
四个已见pre-tag P0来源、181frames320x512。只选alpha .15一次；W除sqrt512，LLL每channel population sigma。全部16channel和旧guards参与，不能套旧QIM保护掩码；保存actual FP32扰动与7高子带roundoff，不加幅/扫描/投影补偿。BCH母码1023/943/t8 GF0x409缩短511信息0→512/432，全MSB-first，是参考补定。
P0/MSG同runtime新D→RGB8；未压缩RGB8与同份RGB8的H264 CRF23 MP4分别E。VAE1/D8/codec各8/E16、FFT16供两钥32/Counter12288；48消息盲表，原32位固定256group/2816cell配对；16质量+8clipping。无float-clean E/裁片/phase/path/新DiT/梯度/训练。
先保存并回读全部scores/raw512、BCH decoded432与原payload，再开真值。BCH_SUCCESS不是完整消息正确或认证；>8可误纠，失败432位保留未决、不偷用raw fallback。对照输出和全部失败保留，四来源非独立确认。
本地仅必要CPU/合成/stub核验，未真实运行模型、GPU、媒体或Colab。新源码/Notebook尚未发布，需另有发布授权后GitHub分支实际包含入口及依赖；本地包不冒称远程已可运行。直接下载可用源码，无巨量SOURCE_TEXT/B64/hash/manifest/版本/Git-clean门禁或手工ZIP。
一次attempt/每call STARTED先持久化，失败/中断消费，后续Run all只读已有JSON，不重试/续算/fallback。GPU由用户运行。
''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    text=(root/'runtime/wan/b_lfg_posthoc_reference_v1.py').read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts');helper=ast.get_source_segment(text,node)
    bootstrap='from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper+'''
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-lfg-posthoc-reference-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_lfg_posthoc_reference_v1_run.py').is_file():
        raise FileNotFoundError('Available source lacks the LFG reference entry: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
else:
    print('Existing scientific artifacts: source/setup/model/physical execution skipped; saved JSON read-only.')
'''
    code(bootstrap)
    md('固定运行。缺依赖只修真实导入/ffmpeg故障；生产不依赖bchlib，也不运行CPU oracle。')
    code(SETUP)
    md('保存结果只读：raw coded BER、BCH状态、完整432消息和原32位退步分开。')
    code('''
for name in ('result.json','evaluation.json'):
    path=OUTPUT/name
    if not path.is_file():print('Unavailable',name);continue
    try:value=json.loads(path.read_text())
    except (OSError,ValueError) as e:print('Saved read error',name,str(e));continue
    if name=='result.json':print('Execution',value.get('status'),'physical calls',value.get('kinds_started'),'fixed',value.get('fixed_denominators'))
    else:
        for group,stats in value.get('message_summary',{}).items():print('Message',group,stats)
        print('Original net32 payload',value.get('payload_by_view'),'reference issues',value.get('reference_issues'))
print('Actual native/LLL/FP32/guards/user-channel costs: construction/. Global quality/temporal/clipping: quality/.')
print('BCH success is not exact recovery or authentication; failed decoded bits remain unresolved. No local/time-ID/sync or capacity claim. Output:',OUTPUT)
''')
    for i,c in enumerate(cells):c['id']='lfg-posthoc-'+str(i)
    path=root/'notebooks'/('b_lfg_posthoc_reference_v1.ipynb');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'))),ensure_ascii=False,indent=1)+'\n')
    return path
if __name__=='__main__':print(build())
