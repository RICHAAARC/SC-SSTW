"""Build the fixed M3 new-source entry; preserve all earlier result destinations."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
STEM='b_adjacent_origin_confirmation_v1'
STORAGE='runtime/wan/b_adjacent_origin_confirmation_storage_v1.py'

SOURCE_FILES=['main/tube_state/b_rank3r_adjacent_origin_v1.py', 'main/tube_state/b_adjacent_origin_confirmation_v1.py', 'runtime/wan/b_adjacent_origin_confirmation_v1.py', 'runtime/wan/b_adjacent_origin_confirmation_evaluation_v1.py', 'runtime/wan/b_adjacent_origin_confirmation_storage_v1.py', 'experiments/wan_state_clock/b_adjacent_origin_confirmation_v1_run.py', 'experiments/wan_state_clock/configs/b_adjacent_origin_confirmation_v1.json']


def build(root=ROOT):
    root=Path(root)
    original=(root/'scripts/build_b_rank3_four_source_confirmation_v1_notebook.py').read_text()
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(original).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SETUP','BOOTSTRAP','RUN','DISPLAY')}
    storage=(root/STORAGE).read_text();ast.parse(storage)
    runtime=(root/'runtime/wan/b_adjacent_origin_confirmation_v1.py').read_text()
    helpers='\n\n'.join(ast.get_source_segment(runtime,n) for n in ast.parse(runtime).body
        if (isinstance(n,ast.FunctionDef) and n.name in ('execution_artifacts','recover_ledger','plan_units','denominators','tasks','task_id','quality_specs')) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SOURCES','ARMS','KEYS','WINDOWS','FIXTURES','READERS','BUDGET','DENOMINATORS') for t in n.targets)))
    preflight=r"""
from pathlib import Path
import json, os, shutil, stat, subprocess, sys, tempfile, urllib.request, zipfile
os.environ.update(HF_HOME='/content/video-wm-cache/huggingface', HF_HUB_CACHE='/content/video-wm-cache/huggingface/hub', TRANSFORMERS_CACHE='/content/video-wm-cache/huggingface/hub', TORCH_HOME='/content/video-wm-cache/torch')
INPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Adjacent-Origin-Confirmation-V1/fixed-four-five-v1')
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
    loader=constants['BOOTSTRAP'][constants['BOOTSTRAP'].index('    workspace='):].replace('b_rank3_four_source_confirmation_v1_run','b_adjacent_origin_confirmation_v1_run')
    source="if CAN_START:\n    try:\n"+textwrap.indent(textwrap.dedent(loader),'        ')+"\n    except Exception as error:\n        CAN_START=False\n        PREPARATION_ERROR=f'{type(error).__name__}: {error}'\n        print('源码准备失败，未启动物理执行:',PREPARATION_ERROR)\nelse:\n    print('跳过源码下载：只读结果或云端状态未知。')\n"
    original_run=constants['RUN'].replace('b_rank3_four_source_confirmation','b_adjacent_origin_confirmation').replace('runtime.wan.b_adjacent_origin_confirmation_backend_v1','runtime.wan.b_rank3_four_source_confirmation_backend_v1')
    probe=original_run[original_run.index('    probe='):original_run.index('    args=')]
    worker=original_run[original_run.index('    args='):]
    worker=worker.replace("'-m','experiments.wan_state_clock.b_adjacent_origin_confirmation_v1_run'","'-c',\"from experiments.wan_state_clock.b_adjacent_origin_confirmation_v1_run import _notebook_reserved_main; raise SystemExit(_notebook_reserved_main())\"")
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
        else:print('系统方法:',value.get('method_status'),'贡献:',value.get('contribution_status'),'判据:',value.get('checks'),'逐源:',value.get('per_source'),'配对:',value.get('paired_summary'),'ADJ纠错证据:',value.get('adj_incremental'))
else:print('挂载端无执行记录；不能据此推断云端未开始。下方只读保存检查。')
print('完整计划/失败/质量:',OUTPUT,'四个新登记v来源，固定五种片段；历史/当前阶段分列，不能称IID或一般重复鲁棒。')
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
    notes=['# ADJ：四个新来源、五片段固定确认\n\nγ=1；ADJ、原RANK3R、CDC_M共享观测。仅原几何拒绝时尝试单窗唯一第二名，先限制origin相差1，再看合法路径并以唯一最小分差选择。已有CPU开发结果不等于新源确认。本入口默认直接读取B分支可用源码，用户Run all执行一次新验证。', '## 1. Drive与一次尝试\n\n已有attempt/启动收据或执行痕迹时只读；失败、中断和未进入项保留，不自动补跑。云端预检未知不启动；quota仅记录。普通启动收据不宣称并发原子锁。', '## 2. 源码与旧背景\n\n普通B分支直接下载，无需手动上传源码ZIP。复用成功安装、ftfy/torchao定点修复、ffmpeg和FP32 VAE。旧background.json仅作背景输入；新输出独立，不覆盖任何旧阶段。', '## 3. 固定四源与预算\n\n湖面红帆船、日间果蔬摊、雪景红电车、陶艺工坊慢速水平pan；seed2026102701–04，a=2/11/17/6。8轨迹、400步、800DiT、100局部梯度、4构造、D18112、codec各72、E41720、Q1440；native Q8/DCT8/keybits16，观测DCT720/keybits1440，108质量记录。360计划全部回读后读取位，360决定回读后评价。模型与临时tensor在/content。', '## 4. 系统与ADJ两层判读\n\n系统沿原M3总体/重复/裁剪删除阈值，null检测/输出、错误和无实际支持均须为0；对RANK3R及CDC_M逐槽零丢失。系统方法达标后，相对CDC_M净增至少32才贡献达标。ADJ对RANK3R的新增/丢失和纠错分支单列；无新增则写本批未提供新增纠错证据，不自动追加来源。共同BER分母为0时未定义。', '## 5. 保存与证据范围\n\nDrive仅保留必要JSON和72个最终MP4，不新增未来任务的OFF原始大体积归档。保存状态与方法状态分列；下载发起不等于浏览器已经保存。本次固定来源仅描述性确认，不给IID、总体FPR、全攻击或感知质量PASS。代理未运行GPU。']
    codes=["from google.colab import drive\ndrive.mount('/content/drive')\n",preflight,source,run,display,saving]
    cells=[]
    for i,(note,code) in enumerate(zip(notes,codes),1):
        ast.parse(code)
        cells.extend([dict(cell_type='markdown',id=f'rank3-repeat-note-{i}',metadata={},source=(note+'\n').splitlines(keepends=True)),
            dict(cell_type='code',id=f'rank3-repeat-code-{i}',metadata={},source=code.splitlines(keepends=True),execution_count=None,outputs=[])])
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    destination=root/'notebooks'/(STEM+'.ipynb');destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=str(destination),code_cells=6,publication_status='B_BRANCH_RELEASE_ENTRY',runtime='b_adjacent_origin_confirmation_v1',output='B-Adjacent-Origin-Confirmation-V1/fixed-four-five-v1')

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
