"""Build the actual CPU notebook. Never execute its experiment cells here."""
import ast
import json
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
STEM='b_framewise_qim_time_recovery_v1'
SETTINGS='''from pathlib import Path
QIM_MEDIA = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1-saved-candidate-native-v1/QIM.mp4')
BASE_MEDIA = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-V1/fixed-e01-181-v1/BASE.mp4')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-Time-Recovery-V1/fixed-e01-ten-views-v1')
SOURCE_DIRECTORY = None  # 可选普通源码目录。
SOURCE_REF = 'dev/local-joint-state-payload-v1'
'''
BOOTSTRAP='''import importlib.util, sys, urllib.request
entry = None
OUTPUT = OUTPUT.expanduser().resolve()
cache = OUTPUT.with_name(OUTPUT.name+'-entry-cache')
consumed = (OUTPUT/'attempt.json').exists()
conflict = OUTPUT.exists() and (not OUTPUT.is_dir() or any(OUTPUT.iterdir())) and not consumed
try:
    folder = cache/'scripts'
    if not consumed and not conflict:
        protected = [QIM_MEDIA.expanduser().resolve().parent, BASE_MEDIA.expanduser().resolve().parent]
        writes = [OUTPUT, cache, OUTPUT.with_name(OUTPUT.name+'-preflight-v1'), OUTPUT.with_name(OUTPUT.name+'-logs')]
        writes += [Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY is not None else Path('/content/video-wm-time-recovery-v1')]
        if any(p.resolve()==s or p.resolve().is_relative_to(s) or s.is_relative_to(p.resolve()) for p in writes for s in protected):
            raise ValueError('写入路径与旧输入目录重合，未下载或启动')
        if SOURCE_DIRECTORY is not None: folder = Path(SOURCE_DIRECTORY)/'scripts'
        else:
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('b_framewise_qim_v1_entry.py', 'b_framewise_qim_time_recovery_v1_entry.py'):
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/'+filename, folder/filename)
    path = folder/'b_framewise_qim_time_recovery_v1_entry.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('time_recovery_entry', path)
        entry = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
        try: spec.loader.exec_module(entry)
        finally: sys.dont_write_bytecode = previous
except Exception as error:
    entry = None
    print('入口不可用:', str(error))
'''
RUN='''if entry is not None:
    session = entry.Session(QIM_MEDIA, BASE_MEDIA, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF)
    session.execute()
'''
DISPLAY='''if entry is not None:
    session.display()
elif consumed or conflict:
    import json
    print('只读原始状态；不下载、不安装、不重跑。')
    for name in ('attempt', 'result'):
        try:
            value = json.loads((OUTPUT/(name+'.json')).read_text())
            if not isinstance(value, dict): raise ValueError('收据应为JSON对象')
            print(name, {k:v for k,v in value.items() if k not in ('plan',)})
        except (OSError, ValueError) as error: print(name, '未知:', str(error))
else:
    print('未进入物理尝试；请查看预检/入口错误。')
'''


def build(root=ROOT):
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s,execution_count=None,outputs=[]))
    md('''# 剪辑后时间关系：固定十视图 CPU 验证
普通 CPU Colab，Run all 执行一次固定包。输入两个既有 MP4，不加载模型、不重写水印、不运行 E41。发布准备仅做人工工程验证，真实媒体由用户运行。

QIM/BASE 各 FULL181、CROP101、DELETE1 180、REPEAT1 182、DELETE3 178，全部同二代 H264 CRF18/yuv420p/8fps。encode10、probe12、decode12、时间读10、接收10、码表1；A/模型/VAE/head/E41/K/KT/solver/DiT/训练/writer均0。固定1644帧、1035720位槽、8220lane槽、1634关系。

全域92672码字，soft副本和、2e+u≤14、非冲突观测精确非降区间路径。启动歧义、局部冲突、错误与未执行均保留；BASE支撑输出计有限负例错误接受。所有盲输出终止后才后评。

码表300秒、每view接收600秒、接收累计6300秒、实际峰值RSS8GiB；每codec300秒，超时terminate等待最多30秒后kill。失败消耗不补领，再次 Run all 只读。原native精确落点失败与最优性未证保持，结果不证明一般编辑/拒绝或剪辑后身份。''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    md('## 输入与独立新输出');code(SETTINGS)
    md('## 加载普通入口\n已消费attempt优先只读缓存；没有缓存仍可显示原始状态。');code(BOOTSTRAP)
    md('## 预检、CPU依赖与固定一次执行\n预检、日志、缓存写同级目录；只有NumPy、psutil、FFmpeg按缺失安装。');code(RUN)
    md('## 紧凑结果\n细表、raw、区间与每条关系留在输出文件。');code(DISPLAY)
    for i,cell in enumerate(cells):cell['id']=f'time-recovery-{i:02}'
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':STEM+'.ipynb'}})
    nbformat.validate(notebook);path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,path)
    return dict(notebook=str(path),code_cells=5,code_lines=sum(len(c.source.splitlines()) for c in cells if c.cell_type=='code'),experiment_cells_executed=False)


if __name__=='__main__': print(json.dumps(build(),indent=2))
