"""Build adopted Card106 user Notebook/ordinary source ZIP, never execute it."""
from pathlib import Path
import ast,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
STEM='b_time_label_local_channel_v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Time-Label-Local-Channel-V1/fixed-viewed-stage1-v1'
NEW_FILES=[
 'main/tube_state/b_detectable_time_label_v1.py','main/tube_state/'+STEM+'.py',
 'runtime/wan/b_time_label_tensor_v1.py','runtime/wan/b_time_label_local_channel_backend_v1.py',
 'runtime/wan/'+STEM+'.py','runtime/wan/b_time_label_local_channel_evaluation_v1.py',
 'experiments/wan_state_clock/'+STEM+'_run.py',
 'experiments/wan_state_clock/configs/'+STEM+'.json','experiments/wan_state_clock/configs/'+STEM+'_budget.json',
 'scripts/build_'+STEM+'_notebook.py','tests/test_'+STEM+'.py','tests/test_b_detectable_time_label_v1.py','docs/'+STEM+'.md']

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

SETUP='\nimport subprocess, sys, time\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE dependencies\')\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'-qq\',\'update\'], \'Prepare media dependency\')\n        command([\'apt-get\',\'-qq\',\'install\',\'-y\',\'ffmpeg\'], \'Install FFmpeg\')\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        raise RuntimeError(\'ffmpeg/ffprobe unavailable; see \'+str(LOG))\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_time_label_local_channel_v1_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress[\'returned\'], \'/\', progress[\'planned\'], \'active\', progress[\'active\'], flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        process.terminate()\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n        print(\'Stopped; saved attempts consumed. Later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'

DISPLAY="\nfor name in ('result.json','thresholds.json','evaluation.json'):\n    p=OUTPUT/name\n    if not p.is_file():print('Not available:',name);continue\n    try:v=json.loads(p.read_text())\n    except (OSError,ValueError) as e:print('Read failure',name,str(e));continue\n    if name=='result.json':print('Execution',v.get('status'),'units',v.get('counts'),'fixed tag rows',v.get('row_status_counts'),'technical rows',v.get('technical_tag_rows'))\n    elif name=='thresholds.json':\n        print('Complete unfiltered pilot-null calibration',v.get('status'),'tau',v.get('tau'),'ID screening',v.get('ID_screening_used'))\n        for host,stats in v.get('hosts',{}).items():\n            maxima=stats.get('all_maxima',[])\n            coords=[{k:row.get(k) for k in ('class','view','phase','row')} for row in maxima[:3]]\n            print('Calibration host',host,'N',stats.get('N'),'null_rows',stats.get('null_rows'),'tied_maxima',len(maxima),'first3_coordinates',coords)\n    else:\n        print('Fixed denominator',v.get('fixed_denominators'))\n        for scope,row in v.get('summary',{}).items():print('Tag scope',scope,row)\n        for scope,row in v.get('strata',{}).items():\n            if scope.startswith('evaluation/local/'):print('Local eval stratum',scope,row)\n        print('Original payload side effect, no tag-selected path',v.get('payload_summary'))\nfor p in sorted((OUTPUT/'quality').rglob('*.json')) if (OUTPUT/'quality').is_dir() else ():\n    if 'OFF_to_P0_reference' in str(p):continue\n    try:\n        v=json.loads(p.read_text());print('Quality',str(p.relative_to(OUTPUT)),v.get('whole'),v.get('psnr_db_range1'),'adjacent',v.get('adjacent_residual'),'PASS',v.get('quality_pass'))\n    except (OSError,ValueError) as e:print('Read failure',str(p),str(e))\nledger={}\ntry:\n    if (OUTPUT/'ledger.json').is_file():ledger=json.loads((OUTPUT/'ledger.json').read_text())\nexcept (OSError,ValueError) as e:print('Ledger read failure',str(e))\nfor p in (OUTPUT/'receipts').rglob('*.json') if (OUTPUT/'receipts').is_dir() else ():\n    try:\n        v=json.loads(p.read_text());ledger[v['slot']]=v\n    except (OSError,ValueError,KeyError,TypeError) as e:print('Receipt read failure',str(p),str(e))\nbad=[(k,v.get('status'),v.get('error'),v.get('unavailable_from')) for k,v in ledger.items() if v.get('status')!='RETURNED']\nprint('Unreturned fixed units',len(bad),'/',len(ledger),'first20',bad[:20])\nprint('VIEWED local development only. Native exact code is not media evidence. Mixed/guard/crossseam legal labels are not correct temporal support.')\nprint('Logical E288 = old cached80 + newly run208. Old6400 decoded bits are reuse, not new observations or recovered capacity.')\nprint('No identity authentication/state-space/independent confirmation/quality PASS. All reject is not success. Full failures stay at',OUTPUT)\n"
def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[]))
    md("""# Card106：固定局部时间标签通道（用户运行）

已采用单一方案：原生44cell的6bit时间ID编码成RM32chip＋16检测pilot；48个不交64D块QIM，δ=1/√12/α1，半径3/Hp≤1。channels4..15只写同步metadata，保留原多步4×8净32归因载荷，不终态重写payload。ECC/唯一解/pilot都不是身份认证。
四个VIEWED宿主2cal2eval；每臂OFF/P0/PT双钥，5个clean45窗及真实pixel删帧后center跨缝窗。完整4phase与10row都留；mixed/guard/crossseam合法ID不能当正确FULL单元。未知gain/offset不拟合救读，不按标签/消息择窗口/路径。
先保存cal全部P/64码距离；2400null P不按Hp/RM/真phase筛选，max阈值冻结后才处理eval。源信息只用于私有媒体形成/后评，blind只拿normalized latent的48投影/key/τ。先保存读回600tag表与288payload表，再开几何/消息真值。
实际最小完整预算：VAE1/D4；逻辑E288=旧80＋新208，encoder新2288；FFT104/vote208/新Counter16640，旧6400译值复用；PT full/cross encode/probe/decode各8。新source/DiT/gradient0，不读旧Q作新tag，不把复用当收益。5760局部＋1056原生row固定，native诊断不是媒体成功。
新P0→PT质量20对（3140帧/3120邻对），原OFF→P0质量20对复用。eval原载荷3840计划pair bits中944FULL，其他2896保留未定义cell目标（cross496）；不以tag筛payload或预测填缺失。
旧136/65负例、Card102有限正例及Card104 s02有限coarse/s04误检宽C/19vs40损失和35.51/34.74dB等质量成本保留。真实局部结果尚未知；合成7项core＋7项新接线检查只属工程。无状态空间、fresh来源或VAE训练。

旧cache不可读直接保留失败，无自动E/FFT重算、重试或强度/阈值scan。一次attempt先持久化，后续Run all只读已存JSON，不重新下载/安装/模型或续算。无B64/hash/manifest/版本/Gitclean门禁，无GPU型号硬门槛。GPU/模型/媒体真实调用由用户执行。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    text=(root/'runtime/wan/b_time_label_local_channel_v1.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper=ast.get_source_segment(text,node)
    bootstrap='from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper
    bootstrap+="""
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-time-label-local-channel-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1',
            'https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_time_label_local_channel_v1_run.py').is_file():
        raise FileNotFoundError('Available source lacks fixed Stage1 runner: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True);print('Available editable source',SOURCE_ROOT)
else:
    print('Existing attempt/artifacts: read saved results only; no source/setup/physical reentry')
"""
    code(bootstrap)
    md('固定执行，不选模式或试参数。仅修复实际缺失导入/媒体工具；失败或中断不重启worker。')
    code(SETUP)
    md('只读保存结果：检出、正确标签、错误合法标签、拒读、载荷退步和质量分别报告。')
    code(DISPLAY)
    for i,c in enumerate(cells):c['id']='b-time-label-local-channel-'+str(i)
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')))
    folder=root/'notebooks';folder.mkdir(exist_ok=True)
    notebook=folder/(STEM+'.ipynb');notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    archive=folder/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in source_files(root)+[str(notebook.relative_to(root))]:z.write(root/name,name)
    return notebook,archive
if __name__=='__main__':
    for p in build():print(p)
