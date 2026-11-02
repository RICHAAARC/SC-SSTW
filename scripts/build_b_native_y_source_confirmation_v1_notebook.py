"""Build the thin fixed n01 user-run Colab notebook."""
from pathlib import Path
import ast
import json
import nbformat
ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_native_y_source_confirmation_v1'
SETTINGS = """SOURCE = '/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/fixed-four-new-v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Native-Y-Source-Confirmation-V1/fixed-n01-v1'
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
MODEL_ID = None
"""
RUN = """from pathlib import Path
import importlib.util, json, tempfile, urllib.request
if (Path(OUTPUT)/'attempt.json').exists():
    result = {'status': 'READ_ONLY_EXISTING_PACKAGE', 'physical_calls_this_invocation': 0, 'post_calls_this_invocation': 0, 'saved': {}, 'read_errors': {}}
    for name in ('attempt', 'state', 'counts', 'resources', 'fixed_slots', 'blind_summary', 'post_status', 'result'):
        try: result['saved'][name] = json.loads((Path(OUTPUT)/(name+'.json')).read_text())
        except (OSError, ValueError) as error: result['read_errors'][name] = str(error)
else:
    try:
        with tempfile.TemporaryDirectory(prefix='n01-notebook-entry-') as temporary:
            path = Path(SOURCE_DIRECTORY)/'scripts/b_native_y_source_confirmation_v1_bootstrap.py' if SOURCE_DIRECTORY else Path(temporary)/'entry.py'
            if SOURCE_DIRECTORY is None:
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/b_native_y_source_confirmation_v1_bootstrap.py', path)
            spec = importlib.util.spec_from_file_location('n01_notebook_entry', path)
            entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
            result = entry.launch(SOURCE, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF, MODEL_ID)
    except Exception as error:
        result = {'status': 'PREPARATION_FAILED', 'error': str(error), 'physical_calls_this_invocation': 0}
saved_result = result.get('saved', {}).get('result', result)
print(json.dumps({k: saved_result.get(k) for k in ('status', 'evaluation_status', 'errors', 'phases', 'stage_results', 'known_counterexamples')}, ensure_ascii=False, indent=2))
print('完整元数据保留在 result；输出目录：', OUTPUT)
"""
DESCRIPTION = """# n01：原生 Y 时间链与 DCT2＋CDC32 一源确认
固定 b-rank3-n01（林荫大道红色电车，seed2026102501）的既存 P0。SOURCE 与独立 OUTPUT 已填写，选择可用 GPU 运行时后运行全部。源码目录与分支均可编辑，直接加载可用源码。不会重新生成 P0；源实际不可用时保留失败，不换源。

一次执行原 A/B 时间物化、同源 BASE/QIM 各五 received 视图、十个 native-Y/H/原 C/L/submit，再仅对五 QIM 冻结全部 B/W/U refs 并共享60次 E41。全部盲终态后才读取原 protocol.writer.bits 与攻击映射进行后评，无默认消息。BASE 有身份水印、无时间写入，并非完全无水印控制；BASE 载荷读取0。

固定异质逻辑上界8866（A8457+B57+time96+payload256），不是模型推理次数；codec50（encode12/probe19/decode19），后评3另列。时间1644节点/1634边、8220lane/1035720位槽；载荷raw9600/vote76800/local1440/final480，每臂160。固定失败、未决、缺失均保留。原 A/features FP32 缓存、CPU FP64 solver4000步及各阶段资源限制沿用；实际输出文件系统容量检查保留。

开始即消费整包；失败/中断也不补跑。再次 Run all 同 OUTPUT 只读取元数据，模型、codec、缓存和后评均不重开。底层长循环在源码中，详细回执在 OUTPUT；入口缓存与源码定位信息仅辅助排错。

时间全部合法输出与精确事件逐项核对，启动未决与非启动覆盖单列。BASE 任一 SUPPORTED_RELATION_SET 即有限错误接受，即便精确输出0。B 各视图32/32字面正确且实际归属/物理支持全正确才达到载荷目标；B/U共同位BER、新输出和支持、归属修正分开报告。无新源RGB/Y对照，不宣称Y收益。

该源曾用于旧 RANK3/RANK3R，仅称当前机制的一源迁移；不推断总体FPR、来源认证、跨消息或长视频能力。profile未知、原native精确目标失败、solver最优性未证继续保留。此 notebook 由用户自行运行；交付与人工工程检查不是真实新源结果。
"""


def build(root=ROOT):
    cells = [nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')\n"),
             nbformat.v4.new_markdown_cell(DESCRIPTION), nbformat.v4.new_code_cell(SETTINGS), nbformat.v4.new_code_cell(RUN)]
    for i, cell in enumerate(cells):
        cell['id'] = f'native-y-n01-{i:02}'
        if cell.cell_type == 'code':
            ast.parse(cell.source); cell.execution_count = None; cell.outputs = []
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(name='python3', language='python', display_name='Python 3'), accelerator='GPU'))
    nbformat.validate(notebook)
    path = Path(root)/'notebooks'/(STEM+'.ipynb')
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return dict(notebook=str(path), scientific_cells_executed=False)


if __name__ == '__main__': print(json.dumps(build(), ensure_ascii=False))
