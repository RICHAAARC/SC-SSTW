"""Build the B-branch Run-all entry. This script never runs the experiment."""
import ast
import json
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STEM='b_local_vae_head_v1'

def build(root=ROOT):
    root=Path(root)
    previous=ast.parse((root/'scripts/build_b_rank3_four_source_confirmation_v1_notebook.py').read_text())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in previous.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SETUP','BOOTSTRAP')}
    vendor='''
sys.path.insert(0,str(SOURCE_ROOT))
from runtime.wan.b_local_vae_head_setup_v1 import prepare_vendor,environment as source_environment,check_dependencies
VENDOR=prepare_vendor(SCRATCH)
environment=source_environment(SOURCE_ROOT,VENDOR)
'''
    storage=(root/'runtime/wan/b_local_vae_head_storage_v1.py').read_text();ast.parse(storage)
    bootstrap=r'''
from pathlib import Path
import json,os,shutil,stat,subprocess,sys,tempfile,urllib.request,zipfile,traceback
SCRATCH=Path('/content/video-wm-local-vae-head-v1')
LOCAL=SCRATCH/'results'
OUTPUT=Path('/content/drive/MyDrive/Video-WM/B-Local-VAE-Head-V1/fixed-six-source-v1')
SOURCE_REF='dev/local-joint-state-payload-v1'
SOURCE_DIRECTORY=None
SOURCE_ROOT=None
PREPARATION_ERROR=None
CLIENT=None
CAN_START=False
os.environ.update(HF_HOME='/content/video-wm-cache/huggingface',HF_HUB_CACHE='/content/video-wm-cache/huggingface/hub',TORCH_HOME='/content/video-wm-cache/torch')
'''+storage+r'''
try:
    CLIENT=authenticate()
    STATE=preflight(CLIENT,OUTPUT)
    local_existing=LOCAL.exists() and any(LOCAL.iterdir())
    CAN_START=STATE['can_start'] and not local_existing
    print('一次尝试检查:',STATE['status'],'本地执行记录:',bool(local_existing))
except Exception as error:
    PREPARATION_ERROR=f'{type(error).__name__}: {error}'
    print('Drive读取实际失败，未启动:',PREPARATION_ERROR)
'''
    source=r'''
if CAN_START:
    try:
        SCRATCH.mkdir(parents=True,exist_ok=True)
        workspace=Path(tempfile.mkdtemp(prefix='source-',dir=SCRATCH)).resolve()
        if SOURCE_DIRECTORY is None:
            url='https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/'+SOURCE_REF
            archive=workspace/'source.zip';urllib.request.urlretrieve(url,archive)
            destination=workspace/'source';destination.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    normalized=member.filename.replace('\\','/')
                    target=(destination/normalized).resolve()
                    if normalized.startswith('/') or any(':' in p for p in normalized.split('/')) or not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr>>16):raise ValueError('unsafe source archive path')
                zipped.extractall(destination)
            entries=list(destination.glob('*/experiments/wan_state_clock/b_local_vae_head_v1_run.py'))
            if len(entries)!=1:raise FileNotFoundError('available B source lacks local VAE entry: '+url)
            SOURCE_ROOT=entries[0].parents[2]
        else:
            SOURCE_ROOT=workspace/'source'
            shutil.copytree(Path(SOURCE_DIRECTORY).expanduser(),SOURCE_ROOT,ignore=shutil.ignore_patterns('.git','__pycache__'))
        CONFIG=SOURCE_ROOT/'experiments/wan_state_clock/configs/b_local_vae_head_v1.json'
        CFG=json.loads(CONFIG.read_text())
        print('可用源码:',SOURCE_ROOT,'固定预算:',CFG['physical_budget'])
    except Exception as error:
        CAN_START=False;PREPARATION_ERROR=f'{type(error).__name__}: {error}'
        print('源码准备失败:',PREPARATION_ERROR)
else:print('已有记录或云端读取未决：跳过源码准备与真实执行。')
'''
    setup=textwrap.dedent(constants['SETUP'])
    setup=setup.replace("LOG = OUTPUT/'setup.log'","LOG = SCRATCH/'setup.log'")
    setup=setup.replace('subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)','subprocess.run(args,cwd=VENDOR,env=environment,stdout=stream,stderr=subprocess.STDOUT)')
    setup=setup.replace("mapping = {p.replace('-', '_'):p for p in packages}","mapping = {p.replace('-', '_'):p for p in packages}; mapping.update({'cv2':'opencv-python','pywt':'PyWavelets','skimage':'scikit-image','yaml':'PyYAML'})")
    setup+="\nensure_imports('import numpy,torch,torchvision; import videoseal.utils.cfg', ['numpy','torch','torchvision','pandas','omegaconf','einops','lpips','timm','pycocotools','tqdm','pytorch-msssim','tensorboard','calflops','transformers','safetensors','scipy','ffmpeg-python','av','huggingface-hub','ftfy','sentencepiece','accelerate'], 'VideoSeal cfg import; no model loading')\n"
    worker=r'''
probe='import main.tube_state.b_local_vae_head_v1; import runtime.wan.b_local_vae_head_backend_v1; import experiments.wan_state_clock.b_local_vae_head_v1_run'
command([sys.executable,'-c',probe],'Probe actual writer/backend/runner imports')
check_dependencies(SOURCE_ROOT,VENDOR)
INPUTS=download_training(CLIENT,CFG,SCRATCH)
INPUT_FILE=SCRATCH/'downloaded_inputs.json';INPUT_FILE.write_text(json.dumps(INPUTS))
START=reserve(CLIENT,OUTPUT)
print('启动收据:',START)
if START['status']=='START_RECEIPT_CLOUD_READ_BACK':
    args=[sys.executable,'-u','-c','from experiments.wan_state_clock.b_local_vae_head_v1_run import _notebook_reserved_main; raise SystemExit(_notebook_reserved_main())','--config',str(CONFIG),'--local-output',str(LOCAL),'--mirror',str(OUTPUT),'--inputs',str(INPUT_FILE)]
    with (SCRATCH/'runner.log').open('a') as stream:
        worker=subprocess.Popen(args,cwd=SOURCE_ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:print(json.loads((LOCAL/'progress.json').read_text()),flush=True)
                    except (OSError,ValueError):print('同一 worker 运行中；详见',SCRATCH/'runner.log',flush=True)
        except BaseException:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
            raise
    print('Runner exit:',worker.returncode)
    if worker.returncode:
        print('\n'.join((SCRATCH/'runner.log').read_text(errors='replace').splitlines()[-40:]))
else:print('没有取得可读启动收据，本次未启动。')
'''
    run="if CAN_START:\n    try:\n"+textwrap.indent(textwrap.dedent(vendor)+setup+textwrap.dedent(worker),'        ')+r'''
    except BaseException as error:
        PREPARATION_ERROR=f'{type(error).__name__}: {error}'
        print('准备或 worker 未完成:',PREPARATION_ERROR)
        if (SCRATCH/'setup.log').exists():print('\n'.join((SCRATCH/'setup.log').read_text(errors='replace').splitlines()[-40:]))
    CAN_START=False
else:print('只读已有尝试；不安装、不加载模型、不补跑。')
'''
    display=r'''
if PREPARATION_ERROR:print('实际错误:',PREPARATION_ERROR)
if CLIENT is not None:
    if LOCAL.is_dir() and any(LOCAL.iterdir()):
        SAVE=sync_results(CLIENT,LOCAL,OUTPUT)
        print('云端保存:',{k:SAVE[k] for k in ('status','uploaded','readable','media','fixed_media_missing','errors','missing','physical_calls')})
    SAVED=read_existing(CLIENT,OUTPUT)
    print('云端读取错误:',SAVED['errors'])
    result=SAVED.get('result.json',{});evaluation=SAVED.get('evaluation.json',{})
    print('计算状态:',result.get('status'),'停止:',result.get('stop'))
    print('固定分母:',result.get('denominators'))
    print('实际调用:',result.get('physical_counts'))
    print('方法:',evaluation.get('method_status'),'推荐:',evaluation.get('recommendation'))
    print('中断恢复固定逐项状态:',{k:SAVED['fixed_recovery'][k] for k in ('status','missing_state_rows','scientific_calls_this_recovery')})
    print('逐源:',json.dumps(evaluation.get('per_source'),ensure_ascii=False,indent=2))
    print('失败/未决数量:',len(evaluation.get('failures',[])))
print('完整结果目录:',OUTPUT)
print('一次训练/固定已见来源证据；ECC不是来源认证或逐位物理支持。未宣称总体FPR、感知质量PASS或新源独立确认。')
'''
    notes=[
        '# 局部 VAE 条件写入头＋软状态空间同步＋32位纠错载荷\n\nB 线直接 Run all 入口，自动加载可用源码，无需上传源码 ZIP。此 Notebook 会执行用户确认的唯一固定训练与 CAL/DEV 验证；当前交付只经过 CPU/static 与独审，真实效果待本次运行。',
        '## 一次尝试与固定预算\n\n已有云端或本地执行记录即只读，失败或中断不重启训练。模型、原视频缓存、VAE 特征和 terminal 仅存 /content；Drive 保留小头、必要媒体、原始 logits、固定任务和失败记录。不得同时开两个运行时启动这个目录。',
        '## 直接加载 B 线源码\n\n固定 TRAIN n01–n04 既存 OFF/full；CAL m01/m02，DEV m03/m04/v02/v04。保留旧 prompt/seed。代码和依赖的来源记录不构成摘要或版本门禁。',
        '## 复用环境，执行一次\n\n沿用 Wan/VideoSeal 导入修复、FP32 VAE 和 H264 CRF18 媒体链。3000 Adam 步（首20步资源观察已包含），batch8、lr1e-4。E181=4、D181=16；编码3072、解码3076、probe76，训练与验证分账。CAL10正例全部正确且40null零检出/输出才启动DEV。没有参数扫描或自动追加新源实验。',
        '## 固定结果与保存核验\n\n主读法150任务/4800槽，I独立基线另150/4800；后评O/W_GROUP/U另90/2880。全部计划保存后读payload，全部blind记录后才后评。42质量对/6702帧/6660相邻对，无感知PASS。原始逐帧与8帧窗口logits、直接观察/ECC修复、错误时间引用、未决项全部保留。云端可读与方法通过分别报告。']
    codes=["from google.colab import drive\ndrive.mount('/content/drive')\n",bootstrap,source,run,display]
    cells=[]
    for i,(note,source_code) in enumerate(zip(notes,codes),1):
        ast.parse(source_code)
        cells.extend([dict(cell_type='markdown',id=f'vae-note-{i}',metadata={},source=(note+'\n').splitlines(keepends=True)),
            dict(cell_type='code',id=f'vae-code-{i}',metadata={},source=source_code.splitlines(keepends=True),execution_count=None,outputs=[])])
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    target=root/'notebooks'/(STEM+'.ipynb');target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(target),source_ref='dev/local-joint-state-payload-v1',real_experiment_executed=False)

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False))
