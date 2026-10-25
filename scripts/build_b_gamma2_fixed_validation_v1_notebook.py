"""Build the direct B-branch source notebook; no physical execution."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STEM='b_gamma2_fixed_validation_v1'
SOURCE_FILES=('runtime/wan/b_gamma2_fixed_validation_v1.py',
 'runtime/wan/b_gamma2_fixed_validation_backend_v1.py',
 'runtime/wan/b_gamma2_fixed_validation_evaluation_v1.py',
 'experiments/wan_state_clock/b_gamma2_fixed_validation_v1_run.py')

SETUP='    LOG = OUTPUT/\'setup.log\'\n    def command(args, label, check=True):\n        print(label, flush=True)\n        with LOG.open(\'a\') as stream:\n            stream.write(\'COMMAND \'+repr(args)+\'\\n\'); stream.flush()\n            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)\n        if completed.returncode and check:\n            raise RuntimeError(label+\' failed; see \'+str(LOG))\n        return completed.returncode\n    def ensure_imports(probe, packages, label):\n        import re\n        mapping = {p.replace(\'-\', \'_\'):p for p in packages}\n        repaired = set()\n        while True:\n            offset = LOG.stat().st_size if LOG.exists() else 0\n            if not command([sys.executable, \'-c\', probe], \'Probe \'+label, check=False):\n                return\n            with LOG.open(\'rb\') as stream:\n                stream.seek(offset); failure = stream.read().decode(\'utf-8\', errors=\'replace\')\n            missing = re.search("ModuleNotFoundError: No module named [\\\'\\"]([^\\\'\\"]+)[\\\'\\"]", failure)\n            if "cannot import name \'FqnToConfig\'" in failure and \'torchao.quantization\' in failure:\n                action = \'remove incompatible optional torchao\'\n                args = [sys.executable, \'-m\', \'pip\', \'uninstall\', \'-y\', \'torchao\']\n            elif missing and missing.group(1).split(\'.\')[0] in mapping:\n                package = mapping[missing.group(1).split(\'.\')[0]]\n                action = \'install missing \'+package\n                args = [sys.executable, \'-m\', \'pip\', \'install\', package]\n            else:\n                raise RuntimeError(label+\' actual import failure; see \'+str(LOG))\n            if action in repaired:\n                raise RuntimeError(label+\' still fails after \'+action+\'; see \'+str(LOG))\n            repaired.add(action)\n            command(args, action)\n    ensure_imports(\'import numpy,torch\', [\'numpy\',\'torch\'], \'tensor dependencies\')\n    ensure_imports(\'import accelerate,safetensors,huggingface_hub,transformers,sentencepiece,ftfy; from diffusers import WanPipeline,AutoencoderKLWan\',\n        [\'diffusers\',\'accelerate\',\'safetensors\',\'huggingface-hub\',\'transformers\',\'sentencepiece\',\'ftfy\'], \'Wan pipeline and VAE\')\n    import shutil\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        command([\'apt-get\',\'-qq\',\'update\'], \'Prepare media dependency\')\n        command([\'apt-get\',\'-qq\',\'install\',\'-y\',\'ffmpeg\'], \'Install FFmpeg\')\n    if not shutil.which(\'ffmpeg\') or not shutil.which(\'ffprobe\'):\n        raise RuntimeError(\'ffmpeg/ffprobe unavailable; see \'+str(LOG))\n'

BOOTSTRAP=r'''
from pathlib import Path
import json, os, shutil, stat, subprocess, sys, tempfile, urllib.request, zipfile
INPUT = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Gamma2-Fixed-Validation-V1/fixed-four-source-v1')
SOURCE_REF = 'dev/local-joint-state-payload-v1'
SOURCE_DIRECTORY = None  # Optional usable local tree; copied into an isolated directory.
{RECOVERY_HELPERS}
EXISTING_ARTIFACTS = execution_artifacts(OUTPUT)
SOURCE_ROOT = None
if EXISTING_ARTIFACTS:
    print('已有一次执行记录：仅恢复读取，不下载源码、不安装、不加载模型。')
else:
    workspace=Path(tempfile.mkdtemp(prefix='b-gamma2-source-')).resolve()
    if SOURCE_DIRECTORY is None:
        archive=workspace/'source.zip'
        url='https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/'+SOURCE_REF
        urllib.request.urlretrieve(url,archive)
        destination=workspace/'source';destination.mkdir()
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                normalized=member.filename.replace('\\','/')
                target=(destination/normalized).resolve()
                if normalized.startswith('/') or any(':' in p for p in normalized.split('/')) or not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError('unsafe source archive entry: '+member.filename)
            source.extractall(destination)
        entries=list(destination.glob('*/experiments/wan_state_clock/b_gamma2_fixed_validation_v1_run.py'))
        if len(entries)!=1:raise FileNotFoundError('Gamma2 source entry unavailable: '+url)
        SOURCE_ROOT=entries[0].parents[2]
    else:
        original=Path(SOURCE_DIRECTORY).expanduser().resolve()
        if not (original/'experiments/wan_state_clock/b_gamma2_fixed_validation_v1_run.py').is_file():
            raise FileNotFoundError('Gamma2 entry missing in SOURCE_DIRECTORY')
        SOURCE_ROOT=workspace/'source'
        shutil.copytree(original,SOURCE_ROOT,ignore=shutil.ignore_patterns('.git','__pycache__'))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('隔离源码目录:',SOURCE_ROOT)
    print('原始输入:',INPUT,'新输出:',OUTPUT)
'''


RUN="if EXISTING_ARTIFACTS:\n    print('只读已有执行记录。')\nelse:\n"+SETUP+r'''
    probe="from pathlib import Path; import importlib; root=Path.cwd().resolve(); names=['runtime.wan.b_gamma2_fixed_validation_v1','runtime.wan.b_gamma2_fixed_validation_backend_v1','runtime.wan.b_gamma2_fixed_validation_evaluation_v1','experiments.wan_state_clock.b_gamma2_fixed_validation_v1_run']; modules=[importlib.import_module(n) for n in names]; assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules); print(*[m.__file__ for m in modules],sep='\\n')"
    subprocess.run([sys.executable,'-c',probe],cwd=SOURCE_ROOT,check=True)
    args=[sys.executable,'-u','-m','experiments.wan_state_clock.b_gamma2_fixed_validation_v1_run',
          '--input',str(INPUT),'--output',str(OUTPUT)]
    with (OUTPUT/'runner.log').open('a') as stream:
        worker=subprocess.Popen(args,cwd=SOURCE_ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress=OUTPUT/'progress.json'
                    try:
                        value=json.loads(progress.read_text());print(value.get('active'),value.get('counts'),flush=True)
                    except (OSError,ValueError):print('同一worker运行中；详见runner.log。',flush=True)
        except BaseException:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
            raise
    print('Runner exit:',worker.returncode,'保存目录:',OUTPUT)
'''

DISPLAY=r'''
recovery=recover_ledger(OUTPUT)
print('恢复后的执行状态:',recovery['counts'])
print('读取异常数量:',len(recovery['read_errors']))
for name in ('result.json','thresholds.json','evaluation.json'):
    try:value=json.loads((OUTPUT/name).read_text())
    except (OSError,ValueError) as error:
        print(name,'尚不可读:',type(error).__name__);continue
    if name=='result.json':
        print('状态:',value.get('status'),'固定分母:',value.get('denominators'))
        print('物理调用:',{k:{n:v.get(n) for n in ('planned','started','returned','failed')} for k,v in value.get('physical_counts',{}).items()})
        print('停止原因:',value.get('stop'))
    elif name=='thresholds.json':print('共同校准:',{k:value.get(k) for k in ('status','tau1','tau2','tau','planned_families')})
    else:
        print('开发性结论:',value.get('development_support'),'规则:',value.get('support_checks'))
        print('CDC_M正例:',value.get('totals',{}).get('CDC_M/positive'),'null:',value.get('totals',{}).get('CDC_M/null'))
        for name,pair in value.get('paired',{}).items():
            print(name,'新增实际精确支持正确槽:',pair.get('added_actual_supported_correct'),
                  '丢失:',pair.get('lost_actual_supported_correct'),'共同输出数:',pair.get('common_count'))
print('完整计划、引用、失败明细和质量记录保留在:',OUTPUT)
print('仅固定已见来源的开发性配对证据；未宣称总体FPR、普遍鲁棒性或感知质量PASS。')
'''


def build(root=ROOT):
    root=Path(root)
    sources={rel:(root/rel).read_text(encoding='utf-8') for rel in SOURCE_FILES}
    for text in sources.values():ast.parse(text)
    runner=sources[SOURCE_FILES[0]]
    helpers='\n\n'.join(ast.get_source_segment(runner,n) for n in ast.parse(runner).body
        if (isinstance(n,ast.FunctionDef) and n.name in ('execution_artifacts','recover_ledger','plan_units','denominators')) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('HOSTS','ARMS','KEYS','WINDOWS','FIXTURES') for t in n.targets)))
    bootstrap=BOOTSTRAP.replace('{RECOVERY_HELPERS}',helpers)
    sources=["from google.colab import drive\ndrive.mount('/content/drive')\n",bootstrap,RUN,DISPLAY]
    notes=[
        '# gamma=2：固定配对验证\n\n在Colab打开本Notebook后Run all即可；无需手动上传源码ZIP。默认直接下载公开B研究分支dev/local-joint-state-payload-v1中的gamma2入口及可用源码，放入隔离目录。第一代码单元仅挂载Drive。',
        '## 1. 输入与隔离源码\n\n复用旧fixed-four-new-v1的P0终态、原始families及缓存。新输出B-Gamma2-Fixed-Validation-V1/fixed-four-source-v1。已有attempt时，在下载/setup前只读恢复。可选SOURCE_DIRECTORY先复制到隔离目录，原目录不会被覆盖。',
        '## 2. 固定一次预算\n\n仅PM2新增4D181、144E41，codec encode/probe/decode各16；无生成、DiT或E181。校准只用wrong key；双90族最大值得到共同108族阈值。首失败停止，未进入行保留，不补跑。四arm的48项新盲计划全部保存回读后才读bit值，最后开放真值。',
        '## 3. 结果摘要\n\n每gamma192正例、960null；双gamma唯一null1152。主读法CDC_M；实际supporting_refs精确同组支持单独后评。只新增36份P0→PM2质量记录，不加入跨片bundle。完整明细存Drive，不把巨型计划/引用输出到Notebook。']
    cells=[]
    for index,(note,code) in enumerate(zip(notes,sources)):
        ast.parse(code)
        cells.extend([dict(cell_type='markdown',id=f'gamma2-note-{index+1}',metadata={},source=(note+'\n').splitlines(keepends=True)),
            dict(cell_type='code',id=f'gamma2-code-{index+1}',metadata={},source=code.splitlines(keepends=True),execution_count=None,outputs=[])])
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    dest=root/'notebooks'/(STEM+'.ipynb');dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(dest),delivery='DIRECT_B_BRANCH_SOURCE',source_ref='dev/local-joint-state-payload-v1',source_files=list(SOURCE_FILES),code_cells=4)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
