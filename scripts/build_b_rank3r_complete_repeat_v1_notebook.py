"""Build the separate repeat entry; never change the original result destination."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
STEM='b_rank3r_complete_repeat_v1'
STORAGE='runtime/wan/b_rank3r_complete_repeat_storage_v1.py'

SOURCE_FILES=['main/tube_state/b_rank3r_complete_repeat_v1.py', 'runtime/wan/b_rank3r_complete_repeat_v1.py', 'runtime/wan/b_rank3r_complete_repeat_backend_v1.py', 'runtime/wan/b_rank3r_complete_repeat_evaluation_v1.py', 'runtime/wan/b_rank3r_complete_repeat_storage_v1.py', 'experiments/wan_state_clock/b_rank3r_complete_repeat_v1_run.py', 'experiments/wan_state_clock/configs/b_rank3r_complete_repeat_v1.json']


def build(root=ROOT):
    root=Path(root)
    original=(root/'scripts/build_b_rank3_four_source_confirmation_v1_notebook.py').read_text()
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(original).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SETUP','BOOTSTRAP','RUN','DISPLAY')}
    storage=(root/STORAGE).read_text();ast.parse(storage)
    runtime=(root/'runtime/wan/b_rank3r_complete_repeat_v1.py').read_text()
    helpers='\n\n'.join(ast.get_source_segment(runtime,n) for n in ast.parse(runtime).body
        if (isinstance(n,ast.FunctionDef) and n.name in ('execution_artifacts','recover_ledger','plan_units','denominators','tasks','task_id','quality_specs')) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SOURCES','ARMS','KEYS','WINDOWS','FIXTURES','READERS','BUDGET') for t in n.targets)))
    preflight=r"""
from pathlib import Path
import json, os, shutil, stat, subprocess, sys, tempfile, urllib.request, zipfile
os.environ.update(HF_HOME='/content/video-wm-cache/huggingface', HF_HUB_CACHE='/content/video-wm-cache/huggingface/hub', TRANSFORMERS_CACHE='/content/video-wm-cache/huggingface/hub', TORCH_HOME='/content/video-wm-cache/torch')
INPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3R-Complete-Repeat-V1/fixed-four-v1')
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
    loader=constants['BOOTSTRAP'][constants['BOOTSTRAP'].index('    workspace='):].replace('b_rank3_four_source_confirmation_v1_run','b_rank3r_complete_repeat_v1_run')
    source="if CAN_START:\n    try:\n"+textwrap.indent(textwrap.dedent(loader),'        ')+"\n    except Exception as error:\n        CAN_START=False\n        PREPARATION_ERROR=f'{type(error).__name__}: {error}'\n        print('源码准备失败，未启动物理执行:',PREPARATION_ERROR)\nelse:\n    print('跳过源码下载：只读结果或云端状态未知。')\n"
    original_run=constants['RUN'].replace('b_rank3_four_source_confirmation','b_rank3r_complete_repeat')
    probe=original_run[original_run.index('    probe='):original_run.index('    args=')]
    worker=original_run[original_run.index('    args='):]
    worker=worker.replace("'-m','experiments.wan_state_clock.b_rank3r_complete_repeat_v1_run'","'-c',\"from experiments.wan_state_clock.b_rank3r_complete_repeat_v1_run import _notebook_reserved_main; raise SystemExit(_notebook_reserved_main())\"")
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
    display=r"""
if local_artifacts(OUTPUT):
    recovery=recover_ledger(OUTPUT)
    print('恢复执行状态:',recovery['counts'],'读取问题:',len(recovery['read_errors']))
    for name in ('result.json','evaluation.json'):
        try:value=json.loads((OUTPUT/name).read_text())
        except (OSError,ValueError) as error:
            print(name,'尚不可读:',type(error).__name__);continue
        if name=='result.json':print('计算状态:',value.get('status'),'固定分母:',value.get('denominators'),'物理调用:',value.get('physical_counts'),'停止:',value.get('stop'))
        else:print('方法状态:',value.get('method_status'),'判据:',value.get('checks'),'逐源:',value.get('per_source'),'相对原RANK3支持增益:',value.get('paired_gain'))
else:print('挂载端无执行记录；不能据此推断云端未开始。下方只读保存检查。')
print('完整计划/失败/质量:',OUTPUT,'本次只验证r8左右完整重播，不是新增来源确认。')
"""
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
        '# B-RANK3R：完整重播固定M2\n\n原RANK3与独立RANK3R两读法，复用已保存四来源γ1的12个完整MP4，固定r8、b49/102；输出24个151帧片段。本入口随B分支源码发布，可直接Run all，无需手动上传源码ZIP。沿用已跑通的环境与媒体链；本M2入口尚未执行真实实验。',
        '## 1. 一次预算与云端预检\n\n先用Drive API和挂载记录检查独立输出，已有attempt/启动收据即只读。实时quota只记录。准备成功后worker前单次写读普通启动收据，不宣称原子互斥；不确定时停止，不重建attempt。',
        '## 2. 源码与既有输入\n\n默认普通B分支源码，安全解压到本次会话临时隔离目录；无需源码ZIP或摘要。INPUT是成功repeat-20261025-v1完整结果目录，包含12 full MP4和旧72结果。缺文件不重新生成。',
        '## 3. 成功环境与固定物理预算\n\n沿用原成功SETUP、ftfy/torchao定点修复和FP32 VAE/cache；模型/HF缓存在/content。先完成旧72 CPU回归，随后codec encode24/probe36/decode36、E41=288、Q=576、DCT=288；生成/DiT/gradient/state/D181/E181/native均0。两读法各48任务，全部96 plans回读后才读位，96 decisions回读后才打开真值。首失败停止，剩余固定槽保留，不补跑。',
        '## 4. 方法与计算状态\n\n完整观测未达判据是方法未支持；真实缺测/调用失败是TECHNICAL_UNRESOLVED。null DETECT即候选失败，不因无bit输出豁免；只可完成既有保存数据诊断，不追加物理调用。',
        '## 5. 保存确认与备份\n\nAPI实读核心JSON和24最终MP4，保存状态与方法状态分开。/content ZIP备份含可读JSON及24视频，不足明确列缺项；可发起下载，不声称用户电脑已保存。保存检查可重复，不补VAE/codec。历史τ仅记录，缺失/差异不阻断且两排名读法不用τ；旧76质量保留固定缺项及可用原记录、不阻断；新24条是整幅edited→codec readback距离，无感知PASS阈值。']
    codes=["from google.colab import drive\ndrive.mount('/content/drive')\n",preflight,source,run,display,saving]
    cells=[]
    for i,(note,code) in enumerate(zip(notes,codes),1):
        ast.parse(code)
        cells.extend([dict(cell_type='markdown',id=f'rank3-repeat-note-{i}',metadata={},source=(note+'\n').splitlines(keepends=True)),
            dict(cell_type='code',id=f'rank3-repeat-code-{i}',metadata={},source=code.splitlines(keepends=True),execution_count=None,outputs=[])])
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    destination=root/'notebooks'/(STEM+'.ipynb');destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(destination),code_cells=6,publication_status='B_BRANCH_RELEASE_ENTRY',runtime='b_rank3r_complete_repeat_v1',output='B-Rank3R-Complete-Repeat-V1/fixed-four-v1')

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
