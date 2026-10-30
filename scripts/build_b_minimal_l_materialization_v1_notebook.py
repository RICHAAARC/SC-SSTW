"""Build the thin e02 A/B GPU notebook without executing its scientific cells."""
import ast
import json
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
STEM='b_minimal_l_materialization_v1'
SETTINGS='''from pathlib import Path
SOURCE = Path('/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Minimal-L-V1/fixed-e02-materialization-v1')
MODEL_ID = None
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
'''
BOOTSTRAP='''import importlib.util, json, sys, urllib.request
entry = None
SOURCE, OUTPUT = SOURCE.expanduser().resolve(), OUTPUT.expanduser().resolve()
cache = OUTPUT.with_name(OUTPUT.name+'-entry-cache')
a_used = (OUTPUT/'A').exists() and (not (OUTPUT/'A').is_dir() or any((OUTPUT/'A').iterdir()))
b_used = (OUTPUT/'B').exists() and (not (OUTPUT/'B').is_dir() or any((OUTPUT/'B').iterdir()))
can_start = not a_used and not b_used
if a_used and not b_used:
    try:
        receipt = json.loads((OUTPUT/'A/result.json').read_text())
        names = ['coefficients.pt','fixed_targets.npy','BASE_preclip.npy']+[f'features/{s:03}.pt' for s in (0,*range(1,181,4))]
        can_start = isinstance(receipt,dict) and receipt.get('stage_complete') is True and receipt.get('finite_candidate_saved') is True and all((OUTPUT/'A'/name).is_file() for name in names)
    except (OSError, ValueError): can_start = False
try:
    folder = cache/'scripts'
    if can_start:
        writes = [OUTPUT,cache,OUTPUT.with_name(OUTPUT.name+'-logs'),OUTPUT.with_name(OUTPUT.name+'-preflight-v1'),Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY is not None else Path('/content/video-wm-minimal-l-v1')]
        if any(p.resolve()==SOURCE or p.resolve().is_relative_to(SOURCE) or SOURCE.is_relative_to(p.resolve()) for p in writes): raise ValueError('写入目录与只读输入重合')
        if SOURCE_DIRECTORY is not None: folder = Path(SOURCE_DIRECTORY)/'scripts'
        else:
            folder.mkdir(parents=True,exist_ok=True)
            for name in ('b_framewise_qim_v1_entry.py','b_minimal_l_materialization_v1_entry.py'):
                urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/'+name,folder/name)
    path = folder/'b_minimal_l_materialization_v1_entry.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('minimal_l_entry',path)
        entry = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
        try: spec.loader.exec_module(entry)
        finally: sys.dont_write_bytecode = previous
except Exception as error:
    entry = None
    print('入口不可用:',str(error))
'''
RUN='''if entry is not None:
    session = entry.Session(SOURCE,OUTPUT,MODEL_ID,SOURCE_DIRECTORY,SOURCE_REF)
    session.execute()
'''
DISPLAY='''if entry is not None:
    session.display()
else:
    print('未启动；已有阶段仅显示原始状态，不下载或补领。')
    for stage in ('A','B'):
        try: print(stage,json.loads((OUTPUT/stage/'result.json').read_text()))
        except (OSError,ValueError) as error: print(stage,'未知:',str(error))
'''


def build(root=ROOT):
    cells=[nbformat.v4.new_markdown_cell('''# e02 最小 L：时间媒体两阶段物化
固定 e02 已有 P0，保留 DCT2+CDC32；不训练、不重新 DiT、不增加身份结论。用户 Run all 顺序执行 A capture/CPU FP64 solve 与 B 保存候选直接检查/原生物化。A 求解 4000 次、每20检查、7200秒、16GiB；数值未决和最优性未证保持。目标1e-3、预算1e-7不放宽。

A/B 各自先预约再调用，失败/中断消费。已有 A 只读；正常结束且缓存完整、有限系数时允许首次 B，即使求解数值未决。B 消费后只读。直接线性失败保留已用 model/K/P，停止下游 head/media；有限完整 native 不达标仍完成固定层读。

总上限：model_load2、D1811、BASE head46、QIM head46、solver1、targets1、K4201/KT4200、BASE/Aa约束P各1、四层时间读8/912240位槽、codec encode/probe/decode各2共6次，E41/DiT/training0。codec每调用300秒，terminate等待30秒后kill，不重试。

本 notebook 不自动运行独立 CPU 接收。媒体齐后用本地 b_minimal_l_v1_run 处理固定十视图：codec34次、time_read10/hard10/C10/L10/码表1/G0，固定1644帧。6300秒为CPU接收累计，不含codec；全链codec计划40次。候选不升级formal或关系。''')]
    for source in ("from google.colab import drive\ndrive.mount('/content/drive')\n",SETTINGS,BOOTSTRAP,RUN,DISPLAY):
        ast.parse(source);cells.append(nbformat.v4.new_code_cell(source,execution_count=None,outputs=[]))
    for i,cell in enumerate(cells): cell['id']=f'minimal-l-{i:02}'
    notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        accelerator='GPU',colab=dict(name=STEM+'.ipynb')))
    nbformat.validate(notebook)
    path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,path)
    return dict(notebook=str(path),code_lines=sum(len(c.source.splitlines()) for c in cells if c.cell_type=='code'),scientific_cells_executed=False)


if __name__=='__main__': print(json.dumps(build(),indent=2))
