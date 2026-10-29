"""Build the actual thin saved-candidate notebook; never run its experiment cells."""
import ast
import json
from pathlib import Path
import nbformat
ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_framewise_qim_saved_candidate_v1'
SETTINGS = '''from pathlib import Path
SAVED_ATTEMPT = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1')
TRUTH_SOURCE = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
NEW_OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1-saved-candidate-native-v1')
MODEL_ID = None  # 可用模型 id/目录；None 使用原 protocol 的模型位置。
SOURCE_DIRECTORY = None  # 可选普通源码目录。
SOURCE_REF = 'dev/local-joint-state-payload-v1'
'''
BOOTSTRAP = '''import importlib.util, sys, urllib.request
entry = None
SAVED_ATTEMPT, TRUTH_SOURCE, NEW_OUTPUT = SAVED_ATTEMPT.resolve(), TRUTH_SOURCE.resolve(), NEW_OUTPUT.resolve()
cache = NEW_OUTPUT.with_name(NEW_OUTPUT.name+'-entry-cache')
consumed = (NEW_OUTPUT/'attempt.json').exists()
conflict = NEW_OUTPUT.exists() and (not NEW_OUTPUT.is_dir() or any(NEW_OUTPUT.iterdir())) and not consumed
if any(NEW_OUTPUT == source or NEW_OUTPUT.is_relative_to(source) for source in (SAVED_ATTEMPT, TRUTH_SOURCE)):
    raise ValueError('NEW_OUTPUT 必须在只读输入目录之外')
try:
    folder = cache/'scripts'
    if not consumed and not conflict:
        writable = [NEW_OUTPUT, cache, NEW_OUTPUT.with_name(NEW_OUTPUT.name+'-preflight-v1'), NEW_OUTPUT.with_name(NEW_OUTPUT.name+'-logs')]
        writable += [Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY is not None else Path('/content/video-wm-framewise-qim-v1')]
        if any(path.resolve() == source or path.resolve().is_relative_to(source) for path in writable for source in (SAVED_ATTEMPT, TRUTH_SOURCE)):
            raise ValueError('写入目录与旧输入重合；未下载、未预检、未启动')
        if SOURCE_DIRECTORY is not None: folder = Path(SOURCE_DIRECTORY)/'scripts'
        else:
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('b_framewise_qim_v1_entry.py', 'b_framewise_qim_saved_candidate_v1_entry.py'):
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/'+filename, folder/filename)
    path = folder/'b_framewise_qim_saved_candidate_v1_entry.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('qim_saved_entry', path)
        entry = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
        try: spec.loader.exec_module(entry)
        finally: sys.dont_write_bytecode = previous
except Exception as error:
    entry = None
    print('入口代码不可用：', str(error))
'''
RUN = '''if entry is not None:
    session = entry.Session(SAVED_ATTEMPT, TRUTH_SOURCE, NEW_OUTPUT, MODEL_ID, SOURCE_DIRECTORY, SOURCE_REF)
    session.execute()
'''
DISPLAY = '''if entry is not None:
    session.display()
elif consumed or conflict:
    import json
    print('只读原始状态；详细入口不可用或输出冲突，不下载、不修复缓存、不重跑。')
    for name in ('attempt', 'result'):
        try: print(name, json.loads((NEW_OUTPUT/(name+'.json')).read_text()))
        except (OSError, ValueError) as error: print(name, '未知：', str(error))
else:
    print('未进入物理尝试。请查看入口错误；正式预检结果在 NEW_OUTPUT 同级 -preflight-v1 目录。')
'''


def build(root=ROOT):
    cells=[]
    def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):
        ast.parse(text); cells.append(nbformat.v4.new_code_cell(text,execution_count=None,outputs=[]))
    md('''# 保存 QIM 候选：独立原生物化与分层诊断
同一保存 coefficients/targets/features/BASE；原求解未决与最优性未证保持。不重解、不调参、不增加 baseline 物理观测。默认 Run all 是未来用户执行的固定一次包；发布准备未执行实验 cells。

新调用上限：模型1生命周期，K1（内部136卷积），BASE与Aa约束P各1，head46，QIM四层456120槽，E41×8/身份1280槽，codec encode/probe/decode各1。D181、baseline head、KT、solver、DiT、training、BASE新物理均0。paired1280与完整cell eligible320分列，未测不算损伤。

只读原attempt；预检写同级 -preflight-v1。预留新attempt后失败/中断消耗，再次 Run all 只读。线性约束失败停止head；有限完整native即使不达标仍完成固定读量；非法/非有限停止依赖步骤。记录不构成最优性、盲同步、认证或画质PASS。''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    md('## 既有输入、新输出与可用模型位置')
    code(SETTINGS)
    md('## 加载普通入口\n已有attempt只读同级入口缓存；缓存不可用仍可显示原始状态，不自动下载。')
    code(BOOTSTRAP)
    md('## 预检并执行固定一次包\n预检缺项可修复后重查，尚不消耗物理attempt。日志、预检和入口缓存均不写入 NEW_OUTPUT。')
    code(RUN)
    md('## 查看已有记录\n报告只读JSON；同一旧输入不改写，未知不补零。truth只在全部固定读取终止后打开。')
    code(DISPLAY)
    for index,cell in enumerate(cells): cell['id']=f'saved-qim-{index:02}'
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':STEM+'.ipynb'},'accelerator':'GPU'})
    nbformat.validate(notebook)
    path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,path)
    return dict(notebook=str(path),code_lines=sum(len(c.source.splitlines()) for c in cells if c.cell_type=='code'),physical_execution=False)

if __name__=='__main__': print(json.dumps(build(),indent=2))
