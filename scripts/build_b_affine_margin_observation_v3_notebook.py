"""Build the real V3 notebook using available GitHub source."""
from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[1]
OUTPUT='/content/drive/MyDrive/Video-WM/B-Affine-Margin-Observation-V3/fixed-s01-once-v1'
NEW_FILES=['main/tube_state/b_affine_margin_v3.py','main/tube_state/b_affine_margin_plan_v3.py','runtime/wan/b_affine_margin_backend_v3.py','runtime/wan/b_affine_margin_observation_v3.py','runtime/wan/b_affine_margin_evaluation_v3.py','experiments/wan_state_clock/b_affine_margin_observation_v3_run.py','scripts/build_b_affine_margin_observation_v3_notebook.py','docs/b_affine_margin_observation_v3.md','tests/test_b_affine_margin_observation_v3.py']
SETUP='\nimport subprocess, sys, time, os, signal\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'update\'], \'refresh package metadata for missing FFmpeg\')\n        command([\'apt-get\',\'install\',\'-y\',\'ffmpeg\'], \'install missing ffmpeg/ffprobe\')\n        if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n            raise RuntimeError(\'ffmpeg/ffprobe unavailable after dependency installation; see \'+str(LOG))\n    ensure_imports(\'import numpy,torch,cvxpy,clarabel\', [\'numpy\',\'torch\',\'cvxpy\',\'clarabel\'], \'tensor and fixed solver dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE dependencies\')\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_affine_margin_observation_v3_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress.get(\'counts\',{}).get(\'RETURNED\',0), \'/\', progress[\'planned\'], \'counts\', progress.get(\'counts\',{}), \'active\', progress.get(\'active\'), flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        try: os.killpg(process.pid, signal.SIGTERM)\n        except ProcessLookupError: pass\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            try: os.killpg(process.pid, signal.SIGKILL)\n            except ProcessLookupError: pass\n            process.wait()\n        print(\'Worker group signalled. STARTED completion may remain unknown; attempt consumed and later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'
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
 def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
 def code(text):
  ast.parse(text);cells.append(dict(cell_type='code',metadata={},source=text.splitlines(keepends=True),execution_count=None,outputs=[]))
 md("""# V3 s01 三臂整链观察 — 用户一次 Run all
预选s01，P0／旧／新、两钥，O33 crop104与delete13 seam52。一次CPU max-t构造；VAE1、D181×3、E181×3+E45×144=147、FFT147、读取294，codec编码/探测/解码各9；源/DiT/旧梯度0。A、原净32位及逐g/channel能量/峰值预算不变，无加幅/扫描/放rho/自动重试。
Notebook直接从RICHAAARC/SC-SSTW的dev/local-joint-state-payload-v1分支获取可用源码，Run all执行上述固定任务，无需手动上传ZIP。也支持已放置的可编辑SOURCE_ROOT目录。没有B64、大内嵌、hash/manifest/版本/Git-clean门禁。
E181三次仅full MP4 readback；RGB8可保存/量质但不单独再编码读取。局部E45来自真实编辑104帧再次MP4/readback；此预算不能拆清VAE/codec/reset全部因果。
真实域始终MODEL_DOMAIN_UNVALIDATED，toy tau/rho只是固定诊断参考，不能认证/PASS或由s01重调。native label reserve7不等于整链容7错；C3/C4/C7仍全拒的历史缺口保留。历史4host CPU分母/两host未决不伪造为本包物理执行。
attempt/STARTED先保存，失败或中断消耗；同一Drive此固定输出后续Run all仅保存结果只读，不重新构造/物理调用。盲的完整矩阵/全部scores/H/c_hat/校正证据/来源及payload先保存回读，再读truth/P0配对量质。GPU/真实构造由用户执行，代理仅静态和stub核验。
""")
 code("from google.colab import drive\ndrive.mount('/content/drive')\n")
 text=(root/'runtime/wan/b_lfg_posthoc_reference_v1.py').read_text();tree=ast.parse(text);helper=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts'))
 code('from pathlib import Path\nimport json,subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper+"""
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-affine-margin-observation-v3')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1','https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_affine_margin_observation_v3_run.py').is_file():
        raise FileNotFoundError('Available source lacks the V3 entry; inspect the actual source download or use editable SOURCE_ROOT: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
else:
    print('Existing attempt: source/setup/construct/model/physical skipped. Saved JSON read-only.')
""")
 md('固定一次执行。只处理真实依赖/导入故障，不绑定版本或GPU型号；构造/物理失败不重试。')
 code(SETUP)
 md('只读保存结果；失败、未决和拒读保留。实际认证域仍未验证。')
 code("""
for name in ('result.json','evaluation.json'):
    path=OUTPUT/name
    if not path.is_file():
        print('Unavailable',name);continue
    try: value=json.loads(path.read_text())
    except (OSError,ValueError) as error:
        print('Saved read error',name,str(error));continue
    if name=='result.json':
        print('Execution:',value.get('status'),'calls:',value.get('kinds_started'),'fixed:',value.get('fixed_denominators'))
    else:
        print('Domain:',value.get('model_domain'),'alignment:',value.get('alignment'),'full32:',value.get('full_group'),'issues:',value.get('issues'))
print('Raw/blind full matrices, all H/c_hat and corrected rows: raw/ blind/. Quality/temporal: quality/. Output:',OUTPUT)
print('No calibration/PASS: describe saved observations only. This notebook has not been run on GPU by the agent.')
""")
 for i,c in enumerate(cells):c['id']='affine-observation-v3-'+str(i)
 path=root/'notebooks/b_affine_margin_observation_v3.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'))),ensure_ascii=False,indent=1)+'\n')
 return path
if __name__=='__main__':print(build())
