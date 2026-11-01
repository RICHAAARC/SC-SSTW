"""Build the fixed CPU-only nine-raw diagnostic notebook; no scientific cell runs."""
from pathlib import Path
import ast
import json
import nbformat

ROOT=Path(__file__).resolve().parents[1]
STEM='b_time_layer_diagnostic_v1'
SETTINGS="""SOURCE_ATTEMPT = '/content/drive/MyDrive/Video-WM/B-New-Source-Chain-Confirmation-V1/fixed-c02-rerun01-v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Time-Layer-Diagnostic-V1/fixed-c02-rerun01-v1'
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
"""
DESCRIPTION="""# c02 固定9raw CPU分层诊断
一次 Run all 读取既有五view received raw 与 QIM preclip/clip/rgb8/h264 四层 raw，不重生媒体。CPU，仅 numpy/psutil；不加载torch/模型/GPU/codec，不重读payload。源码普通下载或可编辑目录，无摘要/版本门禁。

固定67逻辑项：raw_load9、saved_view_load5、codebook1、arithmetic9、fusion9、compatibility27、path5、blind_save1、post1。接收822位置/817边/4110lane/517860原始bit；辅助724位置/3620lane/456120原始bit，无图。总1546位置/7730lane/973980原始bit，F/R0/R1共23190lane-role、1460970bit决策槽；兼容比较上界429812736不是物理采样数。内部复用fuse可能重复加法，逻辑fusion9不是浮点操作计数。

原FP64低tie算式只复算并对比保存量，本轮三角色兼容诊断仍用保存soft/mask，全扫公域92672、lane模5、2e+u<=14、block1024。只对五接收F运行原interval_paths；不改H/L，不重算C/L/G，不选好副本或阈值。四层不是可选追加，也不是H接受门。

所有盲派生和失败账本先保存，唯一post才读原time_evaluation的truth。启动未写lane保持UNWRITTEN。全817边检查精确端点和缺失/重复索引；真假GAP/REPEAT均保留。缺层的首见/转变为UNKNOWN；preclip已错不证明唯一物理根因。旧H/原融合L、B160及native/solver失败不被诊断覆盖。

首计算前消费固定attempt，失败/中断不补跑，再Run all只读本次记录、不重post。RSS8GiB、总墙钟6300秒（含post/主要保存）、码本300秒、每raw600秒；阻塞调用只能返回后观察超限，不保证硬中断。全局资源停止不暗算剩余项，局部失败仍保留分母并可继续独立项。

关闭资源监视器的迟报错误也进入总状态。恢复保留STARTED中已知started/returned回执计数，未持久返回标为UNFINISHED_RECEIPT；缺失/非法回执为UNRESOLVED_RECEIPT。默认摘要显示已保存post/进度中的诊断统计、读盘错误和67项状态计数，失败不抹掉已知发现。

默认仅打印紧凑摘要；逐bit数组、全部hit/距离、旧/重算路径、完整post逐行记录在OUTPUT。本地工程交付不等于已完成真实诊断；发布后由用户自行运行。完整writer/reader92672码字核对仅交付前一次工程检查，不是用户第68项。
"""
RUN="""from pathlib import Path
import importlib.util, json, sys, tempfile, urllib.request
output = Path(OUTPUT)
def entry_at(root):
    spec = importlib.util.spec_from_file_location('time_layer_entry', root/'scripts/b_time_layer_diagnostic_v1_entry.py')
    entry = importlib.util.module_from_spec(spec); previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try: spec.loader.exec_module(entry)
    finally: sys.dont_write_bytecode = previous
    return entry
summary = None; helper_errors = []
if (output/'attempt.json').exists():
    for root in ([Path(SOURCE_DIRECTORY)] if SOURCE_DIRECTORY else []) + [output.with_name(output.name+'-entry-cache')]:
        try:
            entry = entry_at(root); result = entry.saved_status(output); summary = entry.summarize(result, output); break
        except Exception as error: helper_errors.append(str(error)[:240])
    if summary is None:  # No usable source/cache: direct metadata plus a small fixed plan, no downloads.
        saved, errors = {}, {}
        for name in ('attempt','state','counts','resources','evaluation_summary','post_progress','result'):
            try:
                value = json.loads((output/(name+'.json')).read_text())
                if not isinstance(value, dict): raise ValueError('saved object required')
                saved[name] = value
            except (OSError, ValueError) as error: errors[name] = str(error)
        units = [('received/'+v,n) for v,n in zip(('FULL','CROP','DELETE1','REPEAT1','DELETE3'),(181,101,180,182,178))] + [('material/'+v,181) for v in ('preclip','clip','rgb8','h264')]
        plan = {'codebook':'codebook','blind_save':'blind_save','post':'post'}
        for uid,n in units:
            plan.update({uid+'/'+op:op.split('/')[-1] for op in ['raw_load','arithmetic','fusion','F/compatibility','R0/compatibility','R1/compatibility'] + (['saved_view_load','path'] if uid.startswith('received/') else [])})
        rows = {key:dict(kind=kind, planned=1, status='UNRESOLVED_RECEIPT', saved_receipt=saved.get('state',{}).get(key)) for key,kind in plan.items()}
        for row in rows.values():
            receipt = row['saved_receipt']
            if isinstance(receipt,dict): row.update(started=receipt.get('started'), returned=receipt.get('returned'), status='UNFINISHED_RECEIPT' if receipt.get('kind')==row['kind'] and receipt.get('status')=='STARTED' and all(type(receipt.get(k)) is int for k in ('started','returned')) and (receipt.get('started'),receipt.get('returned'))==(1,0) else 'SAVED_RECEIPT_UNVALIDATED')
        slots = {uid:dict(positions=n,edges=n-1 if uid.startswith('received/') else 0,roles={role:dict(lane_slots=n*5,bit_slots=n*315) for role in ('F','R0','R1')}) for uid,n in units}
        result = dict(status='READ_ONLY_METADATA_FALLBACK',computations_this_invocation=0,post_calls_this_invocation=0,rows=rows,fixed_slots=slots,saved=saved,read_errors=errors)
        diagnostic = saved.get('evaluation_summary') or saved.get('post_progress') or {}
        summary = dict(status=result['status'],output=str(output),computations_this_invocation=0,post_calls_this_invocation=0,fixed_operations=len(rows),fixed_raws=len(slots),counts=saved.get('counts'),unfinished={k:r['saved_receipt'] for k,r in rows.items() if r['status']=='UNFINISHED_RECEIPT'},unresolved=sum(r['status']=='UNRESOLVED_RECEIPT' for r in rows.values()),unvalidated=sum(r['status']=='SAVED_RECEIPT_UNVALIDATED' for r in rows.values()),read_errors=errors,helper_errors=helper_errors,evaluation_status=saved.get('result',{}).get('evaluation_status'),resource_receipt=saved.get('resources'),errors=saved.get('result',{}).get('errors'),saved_diagnostic_source='evaluation_summary' if saved.get('evaluation_summary') else 'post_progress',saved_diagnostic={k:diagnostic.get(k) for k in ('status','known_lane_findings','node_status','edge_status','events')},note='Direct saved metadata; unvalidated receipts and all67 rows remain in result. No compute or post.')
        summary['recorded_evaluation_status'] = summary['evaluation_status']
        if saved.get('resources',{}).get('completion_errors') or saved.get('resources',{}).get('error'): summary['evaluation_status'] = 'EVIDENCE_INCOMPLETE'
elif output.exists() and (not output.is_dir() or any(output.iterdir())):
    result = summary = dict(status='OUTPUT_CONFLICT',computations_this_invocation=0)
else:
    try:
        with tempfile.TemporaryDirectory(prefix='time-layer-bootstrap-') as temporary:
            root = Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY else Path(temporary)
            if SOURCE_DIRECTORY is None:
                for relative in ('scripts/b_time_layer_diagnostic_v1_entry.py','scripts/b_conditional_time_output_v1_entry.py','scripts/b_framewise_qim_v1_entry.py','experiments/wan_state_clock/b_time_layer_diagnostic_v1_records.py','experiments/wan_state_clock/b_minimal_l_v1_records.py','experiments/wan_state_clock/configs/b_time_layer_diagnostic_v1.json'):
                    path = root/relative; path.parent.mkdir(parents=True,exist_ok=True)
                    urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/'+relative,path)
            entry = entry_at(root); result = entry.launch(SOURCE_ATTEMPT,OUTPUT,SOURCE_DIRECTORY,SOURCE_REF); summary = entry.summarize(result,OUTPUT)
    except Exception as error: result = summary = dict(status='PREPARATION_FAILED',error=str(error),computations_this_invocation=0)
print(summary)
"""


def build(root=ROOT):
    cells=[nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')\n"),
           nbformat.v4.new_markdown_cell(DESCRIPTION),nbformat.v4.new_code_cell(SETTINGS),nbformat.v4.new_code_cell(RUN)]
    for i,cell in enumerate(cells):
        cell['id']=f'time-layer-{i:02}'
        if cell.cell_type=='code':ast.parse(cell.source);cell.execution_count=None;cell.outputs=[]
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')))
    nbformat.validate(nb);path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,path)
    return dict(notebook=str(path),scientific_cells_executed=False)


if __name__=='__main__':print(json.dumps(build()))
