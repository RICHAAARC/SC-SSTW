"""Build, never execute, the thin t12 saved-candidate user-run notebook."""
from pathlib import Path
import ast
import nbformat

ROOT=Path(__file__).resolve().parents[1]
STEM='b_t12_saved_candidate_media_v1'
SETTINGS="""CONFIG = dict(
    candidate='/content/drive/MyDrive/Video-WM/T12-Saved-Candidate-Media-V1/inputs/run02',
    original='/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1/paper-joint-t12/attempt01',
    host='/content/drive/MyDrive/Video-WM_Cache/paper-checkpoints-v1/cohort/paper-joint-t12/prepare-b3ccfc5ef961476abdedb6607b64a736/host',
    output='/content/drive/MyDrive/Video-WM/T12-Saved-Candidate-Media-V1/attempt01',
    model_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
"""
BOOTSTRAP="""from pathlib import Path
import importlib.util, json, urllib.request
output = Path(CONFIG['output'])
finished = False
if (output/'operations.json').is_file():
    finished = json.loads((output/'operations.json').read_text()).get('post', {}).get('status') != 'NOT_STARTED'
entry = None
if not finished:
    if SOURCE_DIRECTORY is None:
        folder = Path('/content/video-wm-t12-entry'); folder.mkdir(parents=True, exist_ok=True)
        path = folder/'b_t12_saved_candidate_media_v1_entry.py'
        urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/'+path.name, path)
    else:
        path = Path(SOURCE_DIRECTORY)/'scripts/b_t12_saved_candidate_media_v1_entry.py'
    spec = importlib.util.spec_from_file_location('t12_entry', path)
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
"""
RUN="""if finished:
    print('固定包已收束：只读现存记录，不重新调用模型、媒体或接收器。')
    for name in ('result.json', 'evaluation_summary.json'):
        path = output/name
        if path.is_file():
            value = json.loads(path.read_text())
            print(name, {k:v for k,v in value.items() if k not in ('native','cpu_saved_result','startup_nodes','startup_edges','old_wrong_all')})
else:
    entry.execute(CONFIG, SOURCE_DIRECTORY, SOURCE_REF)
"""


def build(root=ROOT):
    texts=[('markdown','''# t12 已保存越预算候选：原生与媒体观察
**DECLARED_OVER_BUDGET_EXPLORATORY_CANDIDATE**。唯一 run02 终点保持不变；CPU实际预算失败、数值未决、最优性/包络未证全部保留。native另记原8/255、2/255、1e-7预算；不设新容差、不重解、不投影缩放。

选择GPU后Run all由用户运行。前提：私有候选输入已传递成功，原46缓存/BASE/head仍可读。预测残差若以两段原字节传输，入口自动拼回后按原数组格式读取。输入缺失会报告具体路径并停止依赖，不重新生成t12；没有手动上传代码/文件步骤。

本包保留全五视图822节点/817关系、8220物理字/517860位；单副本未决由原盲接收逻辑处理，不指定正确副本。先存盲输出再后评。局部结果不代表通用修复、时间认证或科学PASS。'''),
      ('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),
      ('markdown','## 固定输入与新诊断输出\n正式Video-WM_Result和旧缓存只读，旧失败包全部保留。'),('code',SETTINGS),
      ('markdown','## 普通源码入口\n无摘要、固定版本或Git状态门禁。已收束包直接只读；中断包只续尚未消费且依赖完整的操作。'),('code',BOOTSTRAP),
      ('markdown','''## 执行固定一次观察包
原head46；候选codec encode/probe各6、RGB decode6、native-Y decode5；时间read/hard/relative各5。原NEW载荷60 E41、60投影、双key120读位、45局部/15聚合、20 QMatch。二代RGB共享，不重复解码。

质量REF固定旧P0 input5..9，probe5/RGBdecode最多5另列；PSNR/MSE/maxabs与旧NEW_P0同口径，SSIM/LPIPS未测。不补生成/补接收旧臂。模型实际加载生命周期单列，K/KT/solver/DiT/D181/训练均0。每次physical调用前落盘预约，失败或中断不重试；无时间/RSS截停。'''),('code',RUN)]
    cells=[]
    for index,(kind,text) in enumerate(texts):
        if kind=='code': ast.parse(text)
        cell=nbformat.v4.new_code_cell(text,execution_count=None,outputs=[]) if kind=='code' else nbformat.v4.new_markdown_cell(text)
        cell['id']=f't12-saved-{index:02}';cells.append(cell)
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU','colab':{'name':STEM+'.ipynb'}})
    nbformat.validate(nb);path=Path(root)/'notebooks'/(STEM+'.ipynb');nbformat.write(nb,path);return path


if __name__=='__main__': print(build())
