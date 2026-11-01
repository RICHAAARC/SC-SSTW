"""Build the authorized c02 rerun entry from the exercised full-chain notebook."""
from pathlib import Path
import importlib.util
import json
import tempfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAME = 'b_new_source_chain_confirmation_c02_rerun01'
OLD_OUTPUT = '/content/drive/MyDrive/Video-WM/B-New-Source-Chain-Confirmation-V1/fixed-c02-v1'
NEW_OUTPUT = '/content/drive/MyDrive/Video-WM/B-New-Source-Chain-Confirmation-V1/fixed-c02-rerun01-v1'


def build(root=ROOT):
    spec = importlib.util.spec_from_file_location('c02_original_notebook_builder',
        ROOT/'scripts/build_b_new_source_chain_confirmation_v1_notebook.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with tempfile.TemporaryDirectory(prefix='c02-rerun-notebook-') as temporary:
        built = original.build(Path(temporary))
        notebook = nbformat.read(built['notebook'], as_version=4)
    notebook.cells[2].source = notebook.cells[2].source.replace(OLD_OUTPUT, NEW_OUTPUT)
    original_description = original.DESCRIPTION.split('\n', 1)[1]
    original_description = original_description.replace(
        '本轮仅人工 CPU 工程检查；真实 P0 格式/有限性、新时间媒体及整链结果待未来授权执行。',
        '本地交付仅做工程检查；新目录的真实整链结果由本次用户运行产生。')
    notebook.cells[1].source = (
        '# c02 新目录重跑 01\n\n'
        '本入口用于用户已明确授权的新增一次完整验证。原 fixed-c02-v1 已产出中断失败记录，继续保留。'
        '本次固定写入 fixed-c02-rerun01-v1，首次运行执行原 A/B 物化、时间接收和 DCT2＋CDC32 载荷读取；'
        '此新目录一旦开始即消费本次尝试，之后再次 Run all 只读本次保存结果。\n\n'
        'SOURCE 已填好，选择 GPU 运行时后运行全部。复用现有整链入口及容量错误诊断，'
        'FP32、缓存位置、求解器、方法、固定分母和每次预算不变；本入口不表示容量故障已得到真实验证。\n'
        + original_description)
    for i, cell in enumerate(notebook.cells):
        cell['id'] = f'c02-rerun01-{i:02}'
    nbformat.validate(notebook)
    target = Path(root)/'notebooks'/(NAME+'.ipynb')
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    return dict(notebook=str(target), output=NEW_OUTPUT, scientific_cells_executed=False)


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False))
