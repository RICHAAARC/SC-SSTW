"""Build the fixed native VideoSeal reference entry; no model executes here."""
import ast,json,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STEM='b_m4_videoseal_reference_v1'
SOURCE_FILES=['runtime/wan/'+STEM+'.py','runtime/wan/b_m4_videoseal_reference_storage_v1.py','experiments/wan_state_clock/'+STEM+'_run.py','experiments/wan_state_clock/configs/'+STEM+'.json']

def build(root=ROOT):
    root=Path(root)
    old=ast.parse((root/'scripts/build_b_rank3_four_source_confirmation_v1_notebook.py').read_text())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SETUP','BOOTSTRAP')}
    storage=(root/'runtime/wan/b_m4_videoseal_reference_storage_v1.py').read_text();ast.parse(storage)
    runtime=(root/'runtime/wan'/f'{STEM}.py').read_text();tree=ast.parse(runtime)
    helpers='\n\n'.join(ast.get_source_segment(runtime,n) for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in ('plan_units','execution_artifacts','recover_ledger','read')) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('HOSTS','FIXTURES','SCOPES','ARMS') for t in n.targets)))
    preflight=r"""
from pathlib import Path
from collections import Counter
import json,os,shutil,stat,subprocess,sys,tempfile,urllib.request,zipfile
os.environ.update(HF_HOME='/content/video-wm-m4/cache/huggingface',HF_HUB_CACHE='/content/video-wm-m4/cache/huggingface/hub',TORCH_HOME='/content/video-wm-m4/cache/torch')
INPUT=Path('/content/drive/MyDrive/Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1')
OUTPUT=Path('/content/drive/MyDrive/Video-WM/B-M4-VideoSeal-Reference-V1/fixed-four-five-v1')
SOURCE_REF='dev/local-joint-state-payload-v1'
SOURCE_DIRECTORY=None
"""+storage+'\n'+helpers+r"""
CLOUD_CLIENT=None
SOURCE_ROOT=None
PREPARATION_ERROR=None
try:
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ModuleNotFoundError as error:
        if error.name not in ('google','google.auth','google.auth.transport','google.auth.transport.requests','requests'):raise
        subprocess.run([sys.executable,'-m','pip','install','google-auth','requests'],check=True)
    CLOUD_CLIENT=authenticate_drive()
except Exception as error:
    PREPARATION_ERROR=f'{type(error).__name__}: {error}'
    print('Drive API不可用，本次不启动:',PREPARATION_ERROR)
PREFLIGHT=preflight(CLOUD_CLIENT,OUTPUT)
CAN_START=PREFLIGHT['safe_to_start']
print('云端/挂载执行痕迹:',PREFLIGHT['status'],'实时quota:',PREFLIGHT.get('quota'))
print('具体问题:',PREFLIGHT.get('errors'),PREFLIGHT.get('quota_error'))
"""
    loader=constants['BOOTSTRAP'][constants['BOOTSTRAP'].index('    workspace='):].replace('b_rank3_four_source_confirmation_v1_run',STEM+'_run').replace("tempfile.mkdtemp(prefix='b-rank3-source-')","tempfile.mkdtemp(prefix='source-',dir='/content/video-wm-m4')")
    source="if CAN_START:\n    try:\n        Path('/content/video-wm-m4').mkdir(parents=True,exist_ok=True)\n"+textwrap.indent(textwrap.dedent(loader),'        ')+"\n    except Exception as error:\n        CAN_START=False\n        PREPARATION_ERROR=f'{type(error).__name__}: {error}'\n        print('源码准备失败:',PREPARATION_ERROR)\nelse:\n    print('只读或云端未知：跳过源码下载。')\n"
    setup=textwrap.dedent(constants['SETUP'])
    setup=setup.replace("LOG = OUTPUT/'setup.log'","LOG = Path('/content/video-wm-m4/setup.log')")
    setup=setup.replace("completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)","completed = subprocess.run(args,cwd=VENDOR,env=environment,stdout=stream,stderr=subprocess.STDOUT)")
    setup=setup.replace("mapping = {p.replace('-', '_'):p for p in packages}","mapping = {p.replace('-', '_'):p for p in packages}; mapping.update({'cv2':'opencv-python','pywt':'PyWavelets','skimage':'scikit-image','yaml':'PyYAML'})")
    start=setup.index("ensure_imports('import numpy,torch'")
    end=setup.index('import shutil',start)
    setup=setup[:start]+"ensure_imports('import numpy,torch,torchvision; import videoseal.utils.cfg', ['numpy','torch','torchvision','pandas','omegaconf','einops','lpips','timm','pycocotools','tqdm','pytorch-msssim','tensorboard','calflops','transformers','safetensors','scipy','ffmpeg-python','av','huggingface-hub','ftfy','sentencepiece','accelerate'], 'official VideoSeal cfg and tensor imports (no model load)')\n"+setup[end:]
    vendor=r"""
VENDOR=Path('/content/video-wm-m4/vendor/videoseal')
VENDOR.parent.mkdir(parents=True,exist_ok=True)
if not (VENDOR/'videoseal/utils/cfg.py').is_file():
    if VENDOR.exists():raise FileNotFoundError('Existing vendor directory lacks readable videoseal/utils/cfg.py')
    stage=Path(tempfile.mkdtemp(prefix='official-',dir='/content/video-wm-m4'))
    archive=stage/'source.zip'
    urllib.request.urlretrieve('https://codeload.github.com/facebookresearch/videoseal/zip/refs/heads/main',archive)
    destination=stage/'source';destination.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            normalized=member.filename.replace('\\','/')
            target=(destination/normalized).resolve()
            if normalized.startswith('/') or any(':' in p for p in normalized.split('/')) or not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr>>16):raise ValueError('unsafe official source archive path')
        zipped.extractall(destination)
    entries=list(destination.glob('*/videoseal/utils/cfg.py'))
    if len(entries)!=1:raise FileNotFoundError('available official VideoSeal cfg missing')
    shutil.move(str(entries[0].parents[2]),str(VENDOR))
environment=os.environ.copy()
environment['PYTHONPATH']=os.pathsep.join([str(SOURCE_ROOT),str(VENDOR)])
"""
    worker=r"""
probe="from pathlib import Path; import importlib; root=Path(__import__('sys').argv[1]).resolve(); names=['runtime.wan.b_m4_videoseal_reference_v1','runtime.wan.b_m4_videoseal_reference_storage_v1','experiments.wan_state_clock.b_m4_videoseal_reference_v1_run']; modules=[importlib.import_module(n) for n in names]; assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules); print(*[m.__file__ for m in modules],sep='\n')"
subprocess.run([sys.executable,'-c',probe,str(SOURCE_ROOT)],cwd=VENDOR,env=environment,check=True)
START_RECEIPT=reserve_start(CLOUD_CLIENT,OUTPUT)
print('启动收据:',START_RECEIPT)
if START_RECEIPT['status']=='START_RECEIPT_CLOUD_READ_BACK':
    args=[sys.executable,'-u','-c','from experiments.wan_state_clock.b_m4_videoseal_reference_v1_run import _notebook_reserved_main; raise SystemExit(_notebook_reserved_main())','--input',str(INPUT),'--output',str(OUTPUT)]
    with Path('/content/video-wm-m4/runner.log').open('a') as stream:
        worker=subprocess.Popen(args,cwd=VENDOR,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        progress=json.loads((OUTPUT/'progress.json').read_text());print(progress.get('active'),progress.get('counts'),flush=True)
                    except (OSError,ValueError):print('同一worker运行中；详见/content/video-wm-m4/runner.log',flush=True)
        except BaseException:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
            raise
    print('Runner exit:',worker.returncode,'保存目录:',OUTPUT)
else:print('启动收据未读回，未运行worker。')
"""
    run="if CAN_START:\n    try:\n"+textwrap.indent(textwrap.dedent(vendor)+setup+textwrap.dedent(worker),'        ')+"\n    except BaseException as error:\n        PREPARATION_ERROR=f'{type(error).__name__}: {error}'\n        print('准备或worker未完成，保留原记录:',PREPARATION_ERROR)\n    CAN_START=False\nelse:\n    print('只读或云端未知：跳过setup/vendor/model/worker。')\n"
    display=r"""
if local_artifacts(OUTPUT):
    recovery=recover_ledger(OUTPUT)
    print('一次尝试记录:',recovery['counts'],'读取问题:',len(recovery['read_errors']))
    for name in ('result.json','evaluation.json'):
        try:value=json.loads((OUTPUT/name).read_text())
        except (OSError,ValueError) as error:
            print(name,'尚不可读:',type(error).__name__);continue
        if name=='result.json':print('计算状态:',value.get('status'),'固定分母:',value.get('denominators'),'调用:',value.get('physical_counts'),'内部:',value.get('observed_internal'),'停止:',value.get('stop'))
        else:print('外部参考状态:',value.get('reference_status'),'原生256位摘要:',value.get('summary'))
print('完整原始logits、固定未决、质量和B历史负结果:',OUTPUT)
"""
    saving=r"""
STORAGE_REPORT=confirm_and_backup(CLOUD_CLIENT,OUTPUT,backup_directory='/content',progress=lambda text:print(text,flush=True))
print(json.dumps(storage_summary(STORAGE_REPORT),ensure_ascii=False,indent=2))
backup_path=STORAGE_REPORT.get('backup_zip')
if backup_path and Path(backup_path).is_file() and zipfile.is_zipfile(backup_path):
    print('本地备份已生成:',backup_path)
    try:
        from google.colab import files
        files.download(backup_path)
        print('已发起下载，请确认浏览器下载完成。')
    except Exception as error:print('可在Colab文件面板下载:',backup_path,type(error).__name__,str(error))
"""
    notes=[
        '# M4：VideoSeal固定外部参考\n\n随B分支发布的直接Run all入口。官方videoseal_1.0，原生256位共同owner消息、默认强度0.2/JND、chunk32/step4、FP32。当前仅完成实现与CPU检查，尚无真实M4结果。原B M3负结论保留。',
        '## 1. 一次尝试与云端预检\n\n已有attempt/启动收据/执行痕迹即只读，云端未知不启动。quota只记录；完成准备后、worker前才写读普通启动收据，不是并发原子锁。失败或中断消费一次预算，不补跑。',
        '## 2. 可用源码与固定输入\n\n直接下载普通B分支，无需手上传ZIP。4份OFF precodec RGB24和24个OFF MP4均为已有M3输入；缺失不替换、不重新生成。模型、官方源码、临时RGB和日志在/content；Drive仅必要JSON与24个WM MP4。',
        '## 3. 原生模型与固定调用\n\n只补实际导入缺失，深层cfg探针不加载模型；复用ffmpeg媒体链。4embed、24encode、48probe/decode、20edit、48detect。预计8 embedder/248 extractor网络前向，另8个非网络JND块。embed antialias=True、detect=False；一次detect原生FP32帧均值后>0，不另调extract_message。',
        '## 4. 结果口径\n\n48提取记录或固定未决全部落盘回读后才消息后评，12288位全分母。28质量/4468帧/4440相邻对，无质量PASS。OFF字面匹配不是误检率；不定义DETECT/FPR/wrong-key。B32与VideoSeal256异容量分列，B只引用原M3三读法及已有full质量。',
        '## 5. 保存确认与本地备份\n\nGoogle Drive API完整读取核心JSON与24个WM MP4，计算状态和保存状态分开。备份包含本次全部可读JSON和24媒体，缺失明确列出；发起下载不代表用户电脑已保存。保存检查不触发模型或codec。']
    codes=["from google.colab import drive\ndrive.mount('/content/drive')\n",preflight,source,run,display,saving]
    cells=[]
    for i,(note,code) in enumerate(zip(notes,codes),1):
        ast.parse(code)
        cells.extend([dict(cell_type='markdown',id=f'm4-note-{i}',metadata={},source=(note+'\n').splitlines(keepends=True)),dict(cell_type='code',id=f'm4-code-{i}',metadata={},source=code.splitlines(keepends=True),execution_count=None,outputs=[])])
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    destination=root/'notebooks'/(STEM+'.ipynb');destination.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(destination),code_cells=6,publication_status='B_BRANCH_RELEASE_ENTRY',actual_M4_run=False)

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
