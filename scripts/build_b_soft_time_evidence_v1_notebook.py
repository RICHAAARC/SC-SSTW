"""Build a thin CPU L/G diagnostic notebook; never execute scientific cells."""
import ast
import json
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
STEM='b_soft_time_evidence_v1'
SETTINGS='''from pathlib import Path
SOURCE_ATTEMPT = Path('/content/drive/MyDrive/Video-WM/B-Framewise-QIM-Time-Recovery-V1/fixed-e01-ten-views-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Soft-Time-Evidence-V1/fixed-e01-lg-v1')
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
        source = SOURCE_ATTEMPT.expanduser().resolve()
        writes = [OUTPUT, cache, OUTPUT.with_name(OUTPUT.name+'-preflight-v1'), OUTPUT.with_name(OUTPUT.name+'-logs')]
        writes += [Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY is not None else Path('/content/video-wm-soft-time-evidence-v1')]
        if any(p.resolve()==source or p.resolve().is_relative_to(source) or source.is_relative_to(p.resolve()) for p in writes):
            raise ValueError('写入路径与旧attempt重合，未下载或启动')
        if SOURCE_DIRECTORY is not None: folder = Path(SOURCE_DIRECTORY)/'scripts'
        else:
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('b_framewise_qim_v1_entry.py', 'b_soft_time_evidence_v1_entry.py'):
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/'+filename, folder/filename)
    path = folder/'b_soft_time_evidence_v1_entry.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('soft_time_entry', path)
        entry = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
        try: spec.loader.exec_module(entry)
        finally: sys.dont_write_bytecode = previous
except Exception as error:
    entry = None
    print('入口不可用:', str(error))
'''
RUN='''if entry is not None:
    session = entry.Session(SOURCE_ATTEMPT, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF)
    session.execute()
'''
DISPLAY='''if entry is not None:
    session.display()
elif consumed or conflict:
    import json
    print('只读原始状态；不下载、不安装、不重算。')
    for name in ('attempt', 'result', 'resources'):
        try:
            value = json.loads((OUTPUT/(name+'.json')).read_text())
            if not isinstance(value, dict): raise ValueError('收据应为JSON对象')
            print(name, {k:v for k,v in value.items() if k!='plan'})
        except (OSError, ValueError) as error: print(name, '未知:', str(error))
else:
    print('未进入科学尝试；请查看预检/入口错误。')
'''


def build(root=ROOT):
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s,execution_count=None,outputs=[]))
    md('''# L/G 离线软信息诊断（CPU）
未来用户 Run all：读取已有十视图 soft/mask 和旧 blind，每份一次，写入独立诊断目录。当前发布准备只做人工工程检查，未回放真实数据。

逐副本hinge loss后平均，同支持C；L局部排名，G同C的非降路径节点min-marginal。保留C/dL/Phi/delta/dG_C、G_native与G_aligned全部集合。epsilon=1e-12是数值并列约定，不是置信阈值；微负残差保留，不能排除数值解释的变化不计图收益。startup候选含n<4整行未定义；冲突/无观测保留。

旧DCT2＋CDC32、writer、正式对应/资格/关系保持，只增加模型排名。新增正式确定=0，新关系/事件=0。45非启动/16启动仅后评；旧761帧/715关系机械保留不是新证据，BASE旧假j0与拒绝事实仍保留。

预算：raw10、old blind10、码表1、C/L/G各10；旧BCH/receiver、所有媒体/模型/E41等0。固定1644帧/1035720位槽/8220lane槽/1634关系。码表300秒，每view输入至盲写600秒，累计6300秒，实际RSS8GiB。失败消耗不补领，重开只读；十视图盲结果终止后才读取truth。''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    md('## 旧输入与独立输出');code(SETTINGS)
    md('## 普通源码入口');code(BOOTSTRAP)
    md('## 固定一次诊断\n仅按缺失安装NumPy/psutil；预检、日志、缓存保持同级。');code(RUN)
    md('## 只读紧凑结果\n全候选数值、掩码、offsets与旧正式结果在文件中，不打印巨型数组。');code(DISPLAY)
    for i,c in enumerate(cells):c['id']=f'soft-time-{i:02}'
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':STEM+'.ipynb'}})
    nbformat.validate(notebook);path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,path)
    return dict(notebook=str(path),code_cells=5,code_lines=sum(len(c.source.splitlines()) for c in cells if c.cell_type=='code'),scientific_cells_executed=False)


if __name__=='__main__':print(json.dumps(build(),indent=2))
