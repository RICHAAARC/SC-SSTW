"""Thin user-run Colab orchestration; CPU/GPU algorithms stay in source modules."""
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parents[1]
STEM='b_joint_time_packet_a_physical_v1'
DESCRIPTION="""# 联合时间包 A：一次 Run all 完成物理诊断
选择 Colab GPU 运行时，直接 Run all。按顺序执行尚未消费的 GPU-1 → 同 Colab 的 CPU 媒体与完整域时间盲读 → GPU-2 身份读取 → 一次后评，并显示完整紧凑汇总。当前 Drive 已完成 GPU-1，会直接复用；CPU handoff 与两份旧输入包已有默认 Drive 位置，无需用户拼装或搬运。

CPU 媒体仍用原 CPU 算法：固定 encode12/probe17/decode27、30 个完整域时间诊断；不重新求解。GPU-1 至多一次 VAE load、e02 一次 D181 capture、两源92次 head replay；GPU-2 至多一次 VAE load、180次 E41/project/bits，另复用 n01 OLD60 窗。所有模型均在独立阶段进程中释放。

只改变执行地点和编排。固定 NEW msg=8*j+age、两完整 BCH copy、旧 DCT2＋CDC32、H/L、原结果、方法预算均不变。两项盲终态先保存，之后才读取历史 H/L、writer bits 和映射做后评。实际操作开始/返回、score 分块进度、日志位置与阶段终态实时显示。

各阶段开始即消费，失败或中断也不自动重跑；已消费阶段只读元数据。未开始阶段只要求上游真实终态，不要求所有来源成功。缺失终态如实等待，不伪造成功。全部阶段已消费后，再 Run all 仅展示已存元数据与汇总。

时间分母30视图/4932节点/4902边；身份240窗/38400原始位/307200 coefficient票。startup、失败、缺失、合法错误、copy/F冲突和逆序全部保留。scores 只作诊断，formal_output=null；两源为已见来源，不承诺消息成功、通用FPR、来源认证或长视频恢复。源码可编辑，无摘要、固定版本或 Git 状态门。
"""

SETTINGS="""CPU_HANDOFF = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/inputs/cpu-p-run01.zip'
INPUT_PACKAGES = [
    '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/inputs/colab-inputs-run01-part1.zip',
    '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/inputs/colab-inputs-run01-part2.zip',
]
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/fixed-n01-e02-v1'
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
"""

RUN="""from pathlib import Path
import importlib.util, json, tempfile, urllib.request
from IPython.display import Markdown, display
stages = ('gpu1', 'cpu_media', 'gpu2', 'post')
if all((Path(OUTPUT)/stage/'attempt.json').exists() for stage in stages):
    result = {'status': 'READ_ONLY_CONSUMED_STAGES', 'physical_calls_this_invocation': 0}
    print('各阶段已消费；本次仅展示已保存元数据。')
else:
    try:
        with tempfile.TemporaryDirectory(prefix='joint-a-entry-') as temporary:
            path = Path(SOURCE_DIRECTORY)/'scripts/b_joint_time_packet_a_physical_v1_bootstrap.py' if SOURCE_DIRECTORY else Path(temporary)/'entry.py'
            if SOURCE_DIRECTORY is None:
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/b_joint_time_packet_a_physical_v1_bootstrap.py', path)
            spec = importlib.util.spec_from_file_location('joint_a_notebook', path)
            entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
            result = entry.launch(CPU_HANDOFF, INPUT_PACKAGES, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF, MODEL_ID)
    except Exception as error:
        result = {'status': 'PREPARATION_FAILED', 'error': str(error)}
if result.get('error'):
    print(result['status']+': '+result['error'])
current = {}
for stage in stages:
    folder = Path(OUTPUT)/stage
    item = dict(result.get('stages', {}).get(stage, {}))
    item.setdefault('status', 'READ_ONLY_CONSUMED_STAGE' if (folder/'attempt.json').exists() else 'NOT_STARTED')
    item.update(saved={}, read_errors={})
    for name in ('attempt', 'state', 'counts', 'resources', 'blind_terminal', 'result'):
        try:
            value = json.loads((folder/(name+'.json')).read_text())
            if not isinstance(value, dict): raise ValueError('metadata object required')
            item['saved'][name] = value
        except (OSError, ValueError) as error:
            item['read_errors'][name] = str(error)
    current[stage] = item
result['stages'] = current
try:
    path = Path(OUTPUT)/'preparation/b_joint_time_packet_a_physical_v1_report.py'
    spec = importlib.util.spec_from_file_location('joint_a_saved_report', path)
    report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
    result['summary'] = report.build(OUTPUT, current)
    display(Markdown((Path(OUTPUT)/'run_all_summary.md').read_text()))
except Exception as error:
    print('展示 helper 不可用；显示当前元数据，不展示旧汇总：', str(error))
    for stage, item in current.items():
        meta = item['saved']; value = meta.get('result', meta.get('blind_terminal', {}))
        terminal = meta.get('blind_terminal', {}).get('stage_terminal') is True
        status = 'CONSUMED_WITHOUT_TERMINAL' if (Path(OUTPUT)/stage/'attempt.json').exists() and not terminal else value.get('status', item['status'])
        print(stage, status, 'terminal=', terminal, 'errors=', value.get('errors', {}))
        for name in ('blind_terminal', 'result'):
            if name in item['read_errors']: print(stage, name+'.json', 'MISSING_OR_UNREADABLE', item['read_errors'][name])
print('输出目录：', OUTPUT)
"""


def build(root=ROOT):
    cells=[]
    for i,(kind,source) in enumerate((('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),
                                    ('markdown',DESCRIPTION),('code',SETTINGS),('code',RUN))):
        cell=dict(cell_type=kind,id=f'joint-a-physical-{i:02}',metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    book=dict(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n')
    return dict(notebook=str(path),scientific_cells_executed=False)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False))
