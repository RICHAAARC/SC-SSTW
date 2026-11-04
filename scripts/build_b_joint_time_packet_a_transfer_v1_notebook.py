"""Build the separate CPU-only, four-cell GPU-1 file-transfer notebook."""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_joint_time_packet_a_transfer_v1'
DESCRIPTION = """# 联合时间包 A：GPU-1 原文件传输
使用 CPU 运行时即可。Run all 将现有 `fixed-n01-e02-v1/gpu1/{n01,e02}/NEW_preclip.npy` 逐字节复制成每块最多 96 MiB 的分块，写入同项目 `transport` 下的新子目录。每次运行创建新的运输目录，请运行一次并回传末尾显示的目录名和结果。

原 GPU-1 文件只读，原结果及预算保持。这个独立入口不加载 NumPy、Torch 或模型，不执行 GPU、codec、reader、solver 或训练，也不消费 CPU-media/GPU 阶段。原 GPU-1 无需再运行。

输出包含实际目录、分块文件名、字节数和具体 I/O 错误；`transfer_receipt.json` 只是运输定位记录，不是 hash、manifest 或版本校验门。源文件缺失或读取失败会如实显示，另一源继续复制。SOURCE_DIRECTORY 和 SOURCE_REF 可直接编辑，使用可用源码。
"""
SETTINGS = """PROJECT_DIRECTORY = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1'
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
"""
RUN = """from pathlib import Path
import importlib.util, json, tempfile, urllib.request
try:
    with tempfile.TemporaryDirectory(prefix='joint-a-transfer-entry-') as temporary:
        path = Path(SOURCE_DIRECTORY)/'scripts/b_joint_time_packet_a_transfer_v1_bootstrap.py' if SOURCE_DIRECTORY else Path(temporary)/'entry.py'
        if SOURCE_DIRECTORY is None:
            print('Loading transfer entry from branch:', SOURCE_REF, flush=True)
            urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/b_joint_time_packet_a_transfer_v1_bootstrap.py', path)
        spec = importlib.util.spec_from_file_location('joint_a_transfer_notebook', path)
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        result = entry.launch(PROJECT_DIRECTORY, SOURCE_DIRECTORY, SOURCE_REF)
except Exception as error:
    result = {'status': 'TRANSFER_PREPARATION_FAILED', 'error': type(error).__name__+': '+str(error), 'scientific_calls': 0}
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
"""


def build(root=ROOT):
    cells = []
    for i, (kind, source) in enumerate((('code', "from google.colab import drive\ndrive.mount('/content/drive')\n"),
                                       ('markdown', DESCRIPTION), ('code', SETTINGS), ('code', RUN))):
        cell = dict(cell_type=kind, id=f'joint-a-transfer-{i:02}', metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    book = dict(cells=cells, metadata=dict(kernelspec=dict(name='python3', language='python', display_name='Python 3')),
                nbformat=4, nbformat_minor=5)
    path = Path(root) / 'notebooks' / (STEM + '.ipynb')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + '\n')
    return dict(notebook=str(path), scientific_cells_executed=False)


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False))
