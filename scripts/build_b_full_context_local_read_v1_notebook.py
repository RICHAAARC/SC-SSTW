"""Build local pending-publication full-E4 Notebook and ordinary editable bundle."""
from pathlib import Path
import ast,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
STEM='b_full_context_local_read_v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Full-Context-Local-Read-V1/fixed-viewed-clean-v1'
NEW_FILES=[
 'runtime/wan/'+STEM+'.py','runtime/wan/b_full_context_local_read_backend_v1.py','runtime/wan/b_full_context_local_read_evaluation_v1.py',
 'experiments/wan_state_clock/'+STEM+'_run.py','experiments/wan_state_clock/configs/'+STEM+'.json',
 'experiments/wan_state_clock/configs/'+STEM+'_budget.json','scripts/build_'+STEM+'_reference.py',
 'scripts/build_'+STEM+'_notebook.py','tests/test_'+STEM+'.py','docs/'+STEM+'.md',
 'data/'+STEM+'_reference.json.gz']

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

SETUP='\nimport subprocess, sys, time\nif not EXISTING_ARTIFACTS:\n    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan VAE dependencies\')\n    with (OUTPUT/\'runner.log\').open(\'w\') as log:\n        process = subprocess.Popen([sys.executable, \'-u\', \'-m\', \'experiments.wan_state_clock.b_full_context_local_read_v1_run\', \'--output\', str(OUTPUT)],\n            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)\n    try:\n        while process.poll() is None:\n            time.sleep(30)\n            try:\n                progress = json.loads((OUTPUT/\'progress.json\').read_text())\n                print(\'Units\', progress.get(\'counts\',{}).get(\'RETURNED\',0), \'/\', progress[\'planned\'], \'counts\', progress.get(\'counts\',{}), \'active\', progress.get(\'active\'), flush=True)\n            except (OSError,ValueError,KeyError,TypeError) as error:\n                print(\'Progress read unavailable; observing same worker:\',type(error).__name__,str(error),flush=True)\n    except KeyboardInterrupt:\n        process.terminate()\n        try: process.wait(timeout=30)\n        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n        print(\'Stopped; saved attempts consumed. Later Run all is read-only.\')\n        raise\n    print(\'Runner return code\',process.returncode,\'; inspect saved results and runner.log. No retry.\')\nelse:\n    print(\'Existing attempt: source/setup/physical execution skipped.\')\n'

DISPLAY=r'''
for name in ('result.json','evaluation.json'):
    p=OUTPUT/name
    if not p.is_file():print('Not available:',name);continue
    try:value=json.loads(p.read_text())
    except (OSError,ValueError) as e:print('Read failure',name,str(e));continue
    if name=='result.json':
        print('Execution',value.get('status'),'units',value.get('ledger_counts'),'technical full rows',value.get('technical_full_rows'))
        print('Post unresolved',value.get('post_unresolved_counts'))
    else:
        print('Fixed denominator',value.get('fixed_denominators'))
        for k,v in value.get('full',{}).items():print('Unique full',k,v)
        for k in ('all','pilot','data','PREFIX40','OTHER160'):print('Bit truth transitions',k,value.get('bit_truth_transitions',{}).get(k))
        for k in ('all','PREFIX40','OTHER160'):print('Row tag and true-code transitions',k,value.get('row_transitions',{}).get(k))
        print('Reference issues',len(value.get('reference_issues',[])),value.get('reference_issues',[])[:3])
ledger={}
try:
    if (OUTPUT/'ledger.json').is_file():ledger=json.loads((OUTPUT/'ledger.json').read_text())
except (OSError,ValueError) as e:print('Ledger read issue',str(e))
for p in (OUTPUT/'receipts').rglob('*.json') if (OUTPUT/'receipts').is_dir() else ():
    try:
        row=json.loads(p.read_text());ledger[row['slot']]=row
    except (OSError,ValueError,KeyError,TypeError) as e:print('Receipt read issue',str(p),str(e))
bad=[(k,v.get('status'),v.get('error'),v.get('unavailable_from')) for k,v in ledger.items() if v.get('status')!='RETURNED']
print('Unreturned units',len(bad),'/',len(ledger),'first10',bad[:10])
print('Bit agreement is not correctness: BOTH_WRONG stays separate. Row distance/correct ID/old-rule pass are separate.')
print('176 unique full rows, 200 historical pairs (40 same-prefix /160 other starts); no reset-only or independent-confirmation claim.')
print('Old tau/local null are historical descriptions; new full null=0, no full FPR/authentication validation. Saved output:',OUTPUT)
'''
def build(root=ROOT):
    root=Path(root);cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[]))
    md("""# 同保存PT媒体：整段E4正例读取工程包
此Notebook直接加载可用GitHub源码与普通gzip参考资产，不要求手动ZIP。交付检查只覆盖工程桩与静态接线；本工程包未执行用户真实E4，也不提供实际full结果或科学结论。
固定四个VIEWED宿主：复用Card106 PT full/received.pt，各181帧；VAE load1、E4，计划184 encoder chunks。新source/D/codec/E41/E121/FFT/payload/quality/path0。key00新full176行，8448个64D点积、11264距离/360448位Hamming、2816pilotcos；后评额外cos0，不新增key01读或full null。
176唯一full结果与历史clean200配对：clean0/p0、clean1/p1、clean2/p2、clean3/p3、clean4/p0，实际起点0/36/72/108/136；40同前缀anchor和160其他起点分列，不选最佳窗口。两次runtime不同，不能单归reset/长度/VAE/codec。
参考gzip约1.23MB，只复制旧8JSON中native176/clean200/4800旧null及已存字段，不含tensor/像素/视频，无cos/码距重算。它在176 raw全存/回读、旧tau描述状态全存/回读之后才由post解析；缺参考也不阻断盲E4，后评固定未决。
逐bit四向：都正确/仅full正确新增/仅local正确损失/都错误/未定义；不能将F/L相同bit当都正确。行级真码距、正确ID、旧联合规则通过分开。旧local tau与4800null不是新full FPR/错钥/认证验证，944oracle payload保留不是tag引导恢复。
一次attempt及每个物理call STARTED先持久化；失败/中断消费、不重试或续算。以后Run all只读已有JSON，不装依赖/下载源码/载入模型；缓存不可读不临时decode MP4或补local E。所有失败保留176full/200pair/9600chip槽。无B64/hash/manifest/版本/Git-clean或GPU型号门禁。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    text=(root/'runtime/wan/b_full_context_local_read_v1.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='execution_artifacts')
    helper=ast.get_source_segment(text,node)
    bootstrap='from pathlib import Path\nimport json, subprocess\nOUTPUT=Path('+repr(OUTPUT)+')\n'+helper
    bootstrap+="""
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    SOURCE_ROOT=Path('/content/SC-SSTW-b-full-context-local-read-v1')
    if not SOURCE_ROOT.exists():
        subprocess.run(['git','clone','--depth','1','--single-branch','--branch','dev/local-joint-state-payload-v1',
            'https://github.com/RICHAAARC/SC-SSTW.git',str(SOURCE_ROOT)],check=True)
    if not (SOURCE_ROOT/'experiments/wan_state_clock/b_full_context_local_read_v1_run.py').is_file():
        raise FileNotFoundError('Available source lacks the fixed full-E4 runner: '+str(SOURCE_ROOT))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Available editable source',SOURCE_ROOT,'; reference data is parsed only after full blind products.')
else:
    print('Existing attempt/artifacts: saved JSON read-only; no source/setup/model/physical reentry.')
"""
    code(bootstrap)
    md('固定运行，不选参数或模式。只处理真实导入/依赖错误；不会为缺参考数据重算旧科学量。')
    code(SETUP)
    md('只读结果。工程桩验证不是用户真实E4结果；实际标签、误码、拒读和未决分别报告。')
    code(DISPLAY)
    for i,c in enumerate(cells):c['id']='b-full-context-local-'+str(i)
    notebook=root/'notebooks'/(STEM+'.ipynb')
    notebook.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'))),ensure_ascii=False,indent=1)+'\n')
    bundle=root/'notebooks'/(STEM+'_user_bundle.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in source_files(root)+[str(notebook.relative_to(root))]:z.write(root/name,name)
    return notebook,bundle
if __name__=='__main__':
    for p in build():print(p)
