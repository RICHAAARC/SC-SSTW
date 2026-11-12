"""Build the thin user-run supplement notebook; never execute a quality model."""
from pathlib import Path
import ast
import nbformat

ROOT=Path(__file__).resolve().parents[1]
STEM='paper_evidence_supplement_v1'
SETTINGS="""CONFIG = dict(
    result_root='/content/drive/MyDrive/Video-WM_Result',
    output='/content/drive/MyDrive/Video-WM_Result/paper_evidence_supplement_v1/attempt01/quality',
    cache='/content/paper-evidence-supplement-cache')
SOURCE_DIRECTORY = None
SOURCE_REF = 'main'
"""
BOOTSTRAP="""from pathlib import Path
import importlib.util, json, urllib.request
output = Path(CONFIG['output'])
consumed = (output/'attempt.json').exists()
entry = None
if not consumed:
    if SOURCE_DIRECTORY is None:
        folder = Path('/content/paper-evidence-supplement-entry')
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'paper_evidence_supplement_v1_entry.py'
        urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/scripts/'+path.name, path)
    else:
        path = Path(SOURCE_DIRECTORY)/'scripts/paper_evidence_supplement_v1_entry.py'
    spec = importlib.util.spec_from_file_location('supplement_entry', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
"""
RUN="""if consumed:
    print('固定包已消耗：只读现存结果，不重新加载模型、解码或补评分。')
    summary = output/'quality_summary.json'
    if summary.is_file():
        print(summary.read_text(encoding='utf-8'))
    else:
        print('摘要尚缺；中断/失败仍消耗本包，保留未知槽。')
    print('完整记录：', output)
else:
    entry.execute(CONFIG, SOURCE_DIRECTORY, SOURCE_REF)
"""


def build(root=ROOT):
    sections=[('markdown','''# 固定16帧采样的MUSIQ-SPAQ成像质量
选择GPU后由用户 **Run all**。自动准备依赖和源码，无需上传本地代码或中间文件。

前提：Drive中十二源原始最终二代FULL媒体仍可读。固定84逻辑流：每源main OFF/P0/NEW和四baseline WM。缺文件、缺帧和评分失败保留，不重生成、不补抽。

此notebook交付时尚未运行真实模型；新质量分尚不存在。只提供抽样成像代理，不声称已测时间连贯、闪烁、短暂伪影、审美或提示词。'''),
        ('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),
        ('markdown','## 固定输入与输出\n新质量保存在attempt01/quality；保存证据聚合使用兄弟目录saved_evidence，互不占用一次执行目录。旧正式结果只读保留。'),
        ('code',SETTINGS),
        ('markdown','## 加载可用源码\n普通源码URL或可编辑目录均可；缺文件或实际导入失败会直接报告。已消耗包在下载和依赖安装前转只读。'),
        ('code',BOOTSTRAP),
        ('markdown','''## 执行与读取结果
预算：1次模型加载、最多84次媒体读取、1344个计划帧评分槽。索引固定为0,12,…,180；缺帧不替换末帧。MUSIQ-SPAQ使用FP32、eval/inference；长边超过512才缩小，bilinear、antialias=False，不裁剪、不放大。

只在16帧完整时计算完整clip均值。NEW−P0、P0−OFF、NEW−OFF要求同源同索引完整；baseline的WM−OFF还要求已有明确OFF路径相符，否则差值未知。部分均值和失败均保留。分数不设置PASS或领先阈值，重复Run all只读。

输出：逐帧quality_frames、逐流quality_clips、配对quality_pairs（CSV和JSON），quality_summary.json、REPORT.md和操作记录。'''),
        ('code',RUN),
        ('markdown','''## 解释范围
原始MUSIQ-SPAQ分不等于VBench官方全帧/总分。保存证据表、固定四态和条件随机查询界由独立只读CLI汇总，不在此重新运行水印或随机查询。

官方口径：[VBench imaging_quality](https://github.com/Vchitect/VBench/blob/master/vbench/imaging_quality.py)、[VBench preprocessing](https://github.com/Vchitect/VBench/blob/master/vbench/utils.py)。''')]
    cells=[]
    for index,(kind,text) in enumerate(sections):
        if kind=='code': ast.parse(text)
        cell=nbformat.v4.new_code_cell(text,execution_count=None,outputs=[]) if kind=='code' else nbformat.v4.new_markdown_cell(text)
        cell['id']=f'supplement-{index:02}';cells.append(cell)
    notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        accelerator='GPU',colab=dict(name=STEM+'.ipynb')))
    nbformat.validate(notebook);path=Path(root)/'notebooks'/(STEM+'.ipynb')
    path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,path);return path


if __name__=='__main__': print(build())
