"""Thin staged user-run notebook; no science in cells, no automatic CPU execution."""
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parents[1]
STEM='b_joint_time_packet_a_physical_v1'
DESCRIPTION="""# 联合时间包 A：两源分阶段物理诊断
仅采用固定 NEW `msg=8*j+age`、两完整 BCH copy；旧 DCT2＋CDC32 和 H/L 保持原义。n01/e02 均是已见来源。本 notebook 由用户运行 GPU。

首次 Run all 执行 GPU-1：一次加载 VAE，e02 一次 D181 capture，n01 复用原 46 块 host feature，两源共至多 92 次 head replay。CPU handoff 已由本地求解准备成普通 zip，无需手动拼装；也可改为已解压目录。缺终态输入返回 WAITING_INPUT，不在 Colab 求解或做 codec。

GPU-1 保存后结束。后续由本地 CPU 完成固定 codec encode12/probe17/decode27、30 个完整域时间诊断和接收 RGB，再提交 cpu-media-run01.zip。再次 Run all 只启动从未开始的 GPU-2：一次 VAE load，至多180次 E41/project/bits，另复用 n01 OLD60 窗。部分来源失败仍保留全部240窗，继续可用行。GPU-2结束后由本地做一次盲后评。

各阶段开始即消费，失败或中断也不自动重跑；已消费阶段只读元数据。尚未开始阶段等待上游终态收据，不要求所有源成功。全部阶段完成后 Run all 仅报告收据。当前交付不表示 NEW 媒体、身份读取或后评已经产生。

时间分母30视图/4932节点/4902边；身份240窗/38400原始位/307200 coefficient票。startup、失败、缺失、合法错误、copy/F冲突和逆序全部保留。完整域 scores 只作诊断，formal_output=null；有限数值未决候选可被测量，不保证消息成功、通用FPR、来源认证或长视频恢复。具体输入/数值/文件错误按原样记录；源码位置可编辑，无摘要、固定版本或 Git 状态门。
"""
SETTINGS="""CPU_HANDOFF = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/inputs/cpu-p-run01.zip'
CPU_MEDIA = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/inputs/cpu-media-run01.zip'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1/fixed-n01-e02-v1'
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
"""
RUN="""from pathlib import Path
import importlib.util, json, tempfile, urllib.request
pending = any(not (Path(OUTPUT)/stage/'attempt.json').exists() and Path(source).exists()
              for stage, source in (('gpu1', CPU_HANDOFF), ('gpu2', CPU_MEDIA)))
if not pending:
    result = {'status': 'STAGED_STATUS', 'stages': {}}
    for stage in ('gpu1', 'gpu2'):
        folder = Path(OUTPUT)/stage
        item = {'status': 'READ_ONLY_CONSUMED_STAGE' if (folder/'attempt.json').exists() else 'WAITING_INPUT', 'physical_calls_this_invocation': 0, 'saved': {}, 'read_errors': {}}
        if (folder/'attempt.json').exists():
            for name in ('attempt', 'state', 'counts', 'resources', 'blind_terminal', 'result'):
                try: item['saved'][name] = json.loads((folder/(name+'.json')).read_text())
                except (OSError, ValueError) as error: item['read_errors'][name] = str(error)
        result['stages'][stage] = item
else:
    try:
        with tempfile.TemporaryDirectory(prefix='joint-a-entry-') as temporary:
            path = Path(SOURCE_DIRECTORY)/'scripts/b_joint_time_packet_a_physical_v1_bootstrap.py' if SOURCE_DIRECTORY else Path(temporary)/'entry.py'
            if SOURCE_DIRECTORY is None:
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/b_joint_time_packet_a_physical_v1_bootstrap.py', path)
            spec = importlib.util.spec_from_file_location('joint_a_notebook', path)
            entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
            result = entry.launch(CPU_HANDOFF, CPU_MEDIA, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF, MODEL_ID)
    except Exception as error:
        result = {'status': 'PREPARATION_FAILED', 'error': str(error), 'physical_calls_this_invocation': 0}
print(json.dumps({k: v.get('saved', {}).get('result', {'status': v.get('status'), 'error': v.get('error')})
                  for k,v in result.get('stages', {}).items()} or result, ensure_ascii=False, indent=2))
print('完整阶段元数据保留在 result；输出目录：', OUTPUT)
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
