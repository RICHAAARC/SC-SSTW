"""Assemble the thin user notebook. No recovery extraction or setup code generation."""
import ast
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_framewise_qim_v1'

SETTINGS = '''from pathlib import Path
SOURCE = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1')
SOURCE_REF = 'dev/local-joint-state-payload-v1'
SOURCE_DIRECTORY = None  # 可用的普通源码目录；SOURCE 仍是同一既定旧输入。
'''

BOOTSTRAP = '''import importlib.util, sys, urllib.request
CACHE = OUTPUT.with_name(OUTPUT.name+'-entry-cache')
entry_path = CACHE/'scripts/b_framewise_qim_v1_entry.py'
entry = None
existing = OUTPUT.exists() and any(OUTPUT.iterdir())
missing = [] if existing else [str(p) for p in (SOURCE/'private/b2-m3-e01/P0_terminal.pt', SOURCE/'protocol.json') if not p.is_file()]
if not existing and missing:
    print('INPUT_MISSING，未启动：', missing)
else:
    try:
        if not existing:
            if SOURCE_DIRECTORY is not None:
                entry_path = Path(SOURCE_DIRECTORY)/'scripts/b_framewise_qim_v1_entry.py'
            else:
                entry_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/b_framewise_qim_v1_entry.py', entry_path)
        if entry_path.is_file():
            spec = importlib.util.spec_from_file_location('qim_notebook_entry', entry_path)
            entry = importlib.util.module_from_spec(spec)
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try: spec.loader.exec_module(entry)
            finally: sys.dont_write_bytecode = previous
    except Exception as error:
        entry = None
        print('入口代码不可用：', str(error))
'''

RUN = '''if entry is not None:
    session = entry.Session(SOURCE, OUTPUT, SOURCE_REF, SOURCE_DIRECTORY)
    session.execute()
'''

DISPLAY = '''if entry is not None:
    session.display()
elif existing:
    import json
    print('详细报告代码不可用；仅显示已有原始状态，不下载、不安装、不重跑。')
    for name in ('attempt', 'result'):
        try: print(name, json.loads((OUTPUT/(name+'.json')).read_text()))
        except (OSError, ValueError) as error: print(name, '未知：', str(error))
else:
    print('没有启动物理尝试，请查看上方具体路径或入口错误。')
'''


def build(root=ROOT):
    cells = []
    def markdown(text): cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):
        ast.parse(text)
        cells.append(nbformat.v4.new_code_cell(text, execution_count=None, outputs=[]))
    markdown('''# 逐帧 RGB DCT-QIM：固定一次最小诊断

默认 Run all 使用既有 b2-m3-e01/P0 和 protocol.json，运行 BASE/QIM 两臂、完整181帧的固定包。没有新 DiT 生成、训练、R2、参数扫描或失败重试。SOURCE 可指向同一既定输入的实际目录。

固定分母：时间读量912240槽，身份16个E41、2560局部位槽。物理上限：D181一次、head46+46次、K/KT各4201次、H264 encode/probe/decode各2次；CPU FP64求解上限4000步/7200秒。所有失败、缺失和未决保留。已知位置诊断不等于盲同步、存在性或身份认证。

**本 notebook 发布准备时未执行实验 cells，没有预填结果。** 科学结果需由用户实际运行后判断。''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    markdown('## 输入与输出\n已有 OUTPUT 时只读，不下载、安装或运行模型。')
    code(SETTINGS)
    markdown('## 加载入口\n新尝试从普通源码加载入口；已用输出只加载同级 entry-cache。旧输出没有缓存时，最后一格显示原始状态。缓存不是读取门禁。')
    code(BOOTSTRAP)
    markdown('## 执行固定包\n入口 helper 完成源码准备、已验证的依赖修复和受控 worker。日志、入口缓存和临时源码均在 OUTPUT 外；中断后不恢复物理计算。')
    code(RUN)
    markdown('## 查看已有记录\n报告模块只读 JSON，不重评分；未知不补零。完整求解、分层读量与身份证据保留在 OUTPUT。')
    code(DISPLAY)
    for index, cell in enumerate(cells): cell['id'] = f'qim-{index:02}'
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'colab':{'name':STEM+'.ipynb'}, 'accelerator':'GPU'})
    nbformat.validate(notebook)
    path = Path(root)/'notebooks'/(STEM+'.ipynb'); path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return dict(notebook=str(path), code_cells=sum(c.cell_type=='code' for c in cells),
                code_lines=sum(len(c.source.splitlines()) for c in cells if c.cell_type=='code'),
                physical_execution=False)


if __name__ == '__main__': print(json.dumps(build(), ensure_ascii=False, indent=2))
