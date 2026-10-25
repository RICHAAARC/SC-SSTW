"""Build the separate repeat entry; never change the original result destination."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
STEM='b_rank3_four_source_repeat_20261025_v1'
STORAGE='runtime/wan/b_rank3_repeat_storage_v1.py'


def build(root=ROOT):
    root=Path(root)
    original=(root/'scripts/build_b_rank3_four_source_confirmation_v1_notebook.py').read_text()
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(original).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SETUP','BOOTSTRAP','RUN','DISPLAY')}
    storage=(root/STORAGE).read_text();ast.parse(storage)
    runtime=(root/'runtime/wan/b_rank3_four_source_confirmation_v1.py').read_text()
    helpers='\n\n'.join(ast.get_source_segment(runtime,n) for n in ast.parse(runtime).body
        if (isinstance(n,ast.FunctionDef) and n.name in ('execution_artifacts','recover_ledger','plan_units','denominators','tasks','task_id','quality_specs')) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SOURCES','ARMS','KEYS','WINDOWS','FIXTURES') for t in n.targets)))
    preflight=r"""
from pathlib import Path
import json, os, shutil, stat, subprocess, sys, tempfile, urllib.request, zipfile
os.environ.update(HF_HOME='/content/video-wm-cache/huggingface', HF_HUB_CACHE='/content/video-wm-cache/huggingface/hub', TRANSFORMERS_CACHE='/content/video-wm-cache/huggingface/hub', TORCH_HOME='/content/video-wm-cache/torch')
INPUT = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1')
SOURCE_REF = 'dev/local-joint-state-payload-v1'
SOURCE_DIRECTORY = None
"""+storage+'\n'+helpers+r"""
CLOUD_CLIENT=None
PREPARATION_ERROR=None
SOURCE_ROOT=None
try:
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ModuleNotFoundError as error:
        if error.name not in ('google','google.auth','google.auth.transport','google.auth.transport.requests','requests'):
            raise
        subprocess.run([sys.executable,'-m','pip','install','google-auth','requests'],check=True)
    CLOUD_CLIENT=authenticate_drive()
except Exception as error:
    PREPARATION_ERROR=f'{type(error).__name__}: {error}'
    print('Drive API认证不可用，本次不启动物理执行:',PREPARATION_ERROR)
PREFLIGHT=preflight(CLOUD_CLIENT,OUTPUT)
CAN_START=PREFLIGHT['safe_to_start']
print('云端/挂载执行痕迹预检:',PREFLIGHT['status'])
print('Drive实时storageQuota（未设容量阈值）:',PREFLIGHT.get('quota'))
print('具体读取问题:',PREFLIGHT.get('errors'),PREFLIGHT.get('quota_error'))
if not CAN_START:print('只读已有结果或报告云端未知状态；不下载源码、不setup、不运行模型。')
"""
    # Reuse the exact successful isolated source loader, with its original safe ZIP checks.
    loader=constants['BOOTSTRAP'][constants['BOOTSTRAP'].index('    workspace='):]
    source="if CAN_START:\n    try:\n"+textwrap.indent(textwrap.dedent(loader),'        ')+"\n    except Exception as error:\n        CAN_START=False\n        PREPARATION_ERROR=f'{type(error).__name__}: {error}'\n        print('源码准备失败，未启动物理执行:',PREPARATION_ERROR)\nelse:\n    print('跳过源码下载：只读结果或云端状态未知。')\n"
    original_run=constants['RUN']
    probe=original_run[original_run.index('    probe='):original_run.index('    args=')]
    worker=original_run[original_run.index('    args='):]
    run="if CAN_START:\n    try:\n"+textwrap.indent(textwrap.dedent(constants['SETUP']),'        ')+textwrap.indent(textwrap.dedent(probe),'        ')+r"""
        START_RECEIPT=reserve_start(CLOUD_CLIENT,OUTPUT)
        print('启动收据:',START_RECEIPT)
        if START_RECEIPT['status']=='START_RECEIPT_CLOUD_READ_BACK':
"""+textwrap.indent(textwrap.dedent(worker),'            ')+r"""
        else:
            print('没有启动worker；云端已执行、目录未可见或收据状态未知。')
        CAN_START=False
    except BaseException as error:
        CAN_START=False
        PREPARATION_ERROR=f'{type(error).__name__}: {error}'
        print('准备或worker中断，保留原记录，仍可执行下方保存检查:',PREPARATION_ERROR)
else:
    print('跳过setup与worker：只读结果或云端状态未知。')
"""
    display=constants['DISPLAY'].replace("print('仅固定四新源和151帧crop/声明的单次删帧；不宣称任意短片、精确seam、来源认证、总体FPR或感知质量PASS。')", "print('本次是同四源、同seed与固定方法的独立执行记录，不是新样本确认；保存状态不改method_status。')")
    display="if local_artifacts(OUTPUT):\n"+textwrap.indent(display,'    ')+"\nelse:\n    print('挂载端无执行记录；不能据此推断云端NOT_ENTERED。下一单元读取云端核心结果。')\n"
    saving=r"""
STORAGE_REPORT=confirm_and_backup(CLOUD_CLIENT,OUTPUT,backup_directory='/content',progress=lambda text:print(text,flush=True))
print(json.dumps(storage_summary(STORAGE_REPORT),ensure_ascii=False,indent=2))
print('具体缺失/读取错误列表:',STORAGE_REPORT.get('report_path'))
backup_path=STORAGE_REPORT.get('backup_zip')
if backup_path and Path(backup_path).is_file() and zipfile.is_zipfile(backup_path):
    print('本地结果备份已生成:',backup_path)
    try:
        from google.colab import files
        files.download(backup_path)
        print('已发起下载；请自行确认浏览器下载完成。')
    except Exception as error:
        print('未能发起浏览器下载，可从Colab文件面板下载该/content文件:',type(error).__name__,str(error))
else:print('未生成可打开的ZIP，具体磁盘/读取问题见保存报告。')
"""
    notes=[
        '# B-RANK3：独立复跑 repeat-20261025-v1\n\n本入口使用相同四prompt/seed/a、γ=1、旧背景与τ、五读法和原一次预算，是同样本重复执行，不是独立新样本确认。旧确认Notebook继续只读旧输出；本入口不清空、覆盖或补跑任何既有attempt。',
        '## 1. 云端预检与普通启动收据\n\nColab账户授权用于Drive API读取，以及worker前单次写入/读回普通repeat_start.json。先按MyDrive路径唯一定位新目录；API未知或云端已有执行痕迹即不启动，即使挂载端暂时看不到文件。实时storageQuota只作记录，不设猜测容量阈值。不宣称该收据对同时启动会话提供原子互斥。',
        '## 2. 可用源码\n\n默认下载公开B分支可用源码到隔离目录，无需上传源码ZIP。已有执行时跳过下载。INPUT只提供旧背景和阈值；新OUTPUT固定为B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1。',
        '## 3. 原安装/媒体链与一次预算\n\n沿用成功SETUP及ftfy/torchao定点修复，不改torch/diffusers环境方案。模型/HF和临时张量留在本次/content会话；JSON与最终视频写新Drive目录。8轨迹/400步/800DiT/100localgradient/4state，D181=12、E41=432、codec各48；72任务与五读法固定分母不变。准备完成后再查云端并单次写启动收据；创建响应不确定不重试、不启动worker。首物理失败停止；后续Run all只读与备份。',
        '## 4. 计算结果\n\n显示原runtime记录的计算状态和method_status。缺文件或云端保存不确定不能触发重算，也不能改写方法结论。全部旧失败/未进入分母保留。',
        '## 5. 云端保存确认与/content备份\n\n实际API完整读取解析15个核心根JSON，流式读完全部48MP4。成功仅标CORE_RESULTS_AND_48_MEDIA_CLOUD_READABLE，不声称逐一云端验证所有附属JSON。备份包含本地全部JSON、API已读核心与48视频；云端仅有而挂载缺失的附属JSON列缺项，整体不足标不完整。ZIP及具体错误清单放/content，供files.download；发起下载不等于用户电脑已保存。挂载文件存在、close或fsync都不代表云端保存完成。保存检查可重复，不执行模型/codec、不修改方法状态。']
    codes=["from google.colab import drive\ndrive.mount('/content/drive')\n",preflight,source,run,display,saving]
    cells=[]
    for i,(note,code) in enumerate(zip(notes,codes),1):
        ast.parse(code)
        cells.extend([dict(cell_type='markdown',id=f'rank3-repeat-note-{i}',metadata={},source=(note+'\n').splitlines(keepends=True)),
            dict(cell_type='code',id=f'rank3-repeat-code-{i}',metadata={},source=code.splitlines(keepends=True),execution_count=None,outputs=[])])
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    destination=root/'notebooks'/(STEM+'.ipynb');destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(destination),code_cells=6,publication_status='LOCAL_UNPUBLISHED',runtime='b_rank3_four_source_confirmation_v1',output='B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1')

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
