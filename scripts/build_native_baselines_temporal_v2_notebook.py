"""Build thin native WAM/PixelSeal user-run Colab entry. No execution."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    cells = []
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type="code", id=f"code-{len(cells)}", metadata={}, execution_count=None,
                          outputs=[], source=text.splitlines(keepends=True)))
    def note(text):
        cells.append(dict(cell_type="markdown", id=f"note-{len(cells)}", metadata={}, source=text.splitlines(keepends=True)))
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    note("# WAM32 / PixelSeal256：时间域开发 baseline\n\nGPU由用户Run all。四个已见OFF来源；WAM中央区域逐帧图像扩展，PixelSeal全帧原生视频。11个固定时间条件×G1直接RGB/G2再次H264 CRF18，共352次实际读取、53600帧行；本批均全幅接收。容量、冗余、空间范围分列；不代表独立论文测试、等净载荷或时间恢复。固定attempt01失败或中断也消费，重入只读。\n")
    code("SOURCE_REF = 'dev/local-joint-state-payload-v1'\nSOURCE_DIRECTORY = None  # 可选：已有可编辑源码目录；默认直接下载可用源码\n")
    note("## 加载入口与执行\n\n源码可编辑，无摘要或精确版本门禁。运行入口检查Drive已存attempt后才准备模型依赖。输出固定在MyDrive/Video-WM_Result/baselines/native-wam-pixelseal-temporal-dev-v2/attempt01；失败包也保留。模型/缓存只在/content。请勿同时打开两个实例运行同一attempt。\n")
    code("from pathlib import Path\nimport urllib.request\nbootstrap_path = 'runtime/wan/native_baseline_temporal_colab_v2.py'\nif SOURCE_DIRECTORY:\n    bootstrap_text = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()\nelse:\n    url = f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}'\n    bootstrap_text = urllib.request.urlopen(url).read().decode('utf-8')\nentry = {'__name__': 'native_baseline_colab_entry'}\nexec(compile(bootstrap_text, bootstrap_path, 'exec'), entry)\nRUN = entry['run_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF)\n")
    note("## 结果\n\n盲输出先保存再读取消息真值评价。分母含所有失败和未启动；NOT_RECEIVED源帧与接收器UNKNOWN分开。空mask未知、可读位BER、coverage、完整消息、同代画质和实际成本分别报告。两代IDENTITY对照保留，重复收到帧不去重。没有科学PASS字段赋真；两方法时间位置、GAP/REPEAT/REPLAY及联合身份时间均为N/A。原生32/256位不截前缀作公平性。输出目录不参加常规清理。\n")
    code("print('入口执行状态:', RUN['status'])\nprint('保留结果目录:', RUN['output'])\nentry['show_results']()\n")
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                                                       accelerator="GPU"), cells=cells)
    destination = Path(root) / "notebooks/native_baselines_temporal_v2.ipynb"
    destination.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build())
