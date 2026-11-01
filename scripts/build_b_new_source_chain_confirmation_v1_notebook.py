"""Build the thin existing-c01-received-media user-run notebook."""
from pathlib import Path
import ast
import json
import nbformat

ROOT=Path(__file__).resolve().parents[1]
STEM='b_new_source_chain_confirmation_v1'
SETTINGS="""SOURCE = '/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-New-Source-Chain-Confirmation-V1/fixed-c02-v1'
SOURCE_DIRECTORY = None
SOURCE_REF = 'dev/local-joint-state-payload-v1'
MODEL_ID = None
"""
RUN="""from pathlib import Path
import importlib.util, json, sys, tempfile, urllib.request
if (Path(OUTPUT)/'attempt.json').exists():
    result = {'status': 'READ_ONLY_EXISTING_PACKAGE', 'physical_calls_this_invocation': 0, 'saved': {}, 'read_errors': {}}
    for name in ('attempt', 'state', 'counts', 'resources', 'evaluation', 'result'):
        try:
            saved = json.loads((Path(OUTPUT)/(name+'.json')).read_text())
            if not isinstance(saved, dict): raise ValueError('saved receipt object required')
            result['saved'][name] = saved
        except (OSError, ValueError) as error: result['read_errors'][name] = str(error)
    # Record-only display uses the fixed skeleton saved before the first operation.
    # No source, planner, bit cache, truth, model or evaluator is loaded here.
    import copy
    result['recovered_artifacts'] = {}
    states = {}
    for item in result['saved'].get('attempt', {}).get('recovery_layout', []):
        value, error, unconfirmed = copy.deepcopy(item['fallback']), None, None
        state_path = item['state_path']
        if state_path not in states:
            try: states[state_path] = json.loads((Path(OUTPUT)/state_path).read_text())
            except (OSError, ValueError): states[state_path] = {}
        state = states[state_path] if isinstance(states[state_path], dict) else {}
        try:
            path = (Path(OUTPUT)/item['path']).resolve()
            if not path.is_relative_to(Path(OUTPUT).resolve()): raise ValueError('receipt path outside output')
            saved = json.loads(path.read_text())
            if item['kind'] == 'raw_bank':
                if not isinstance(saved, dict): raise ValueError('raw metadata object required')
                for key in value:
                    fixed, row = value[key], saved.get(key)
                    statuses = [state[op].get('status') if isinstance(state.get(op), dict) else None for op in fixed['operations']]
                    known = 'RETURNED' if statuses[-1] == 'RETURNED' else next((s for s in reversed(statuses) if s in ('FAILED', 'INTERRUPTED', 'RESOURCE_UNRESOLVED')), None)
                    if known is None: known = 'DEPENDENCY_UNRESOLVED' if all(s in ('NOT_STARTED', 'DEPENDENCY_UNRESOLVED') for s in statuses) else 'UNRESOLVED_RECEIPT'
                    valid = isinstance(row, dict) and known != 'UNRESOLVED_RECEIPT' and row.get('status') == known and all(row.get(k) == fixed[k] for k in ('raw_bit_slots', 'vote_slots', 'encoder_input_support'))
                    if valid: value[key] = dict(row, **{k: fixed[k] for k in ('raw_bit_slots', 'vote_slots', 'encoder_input_support', 'operations')})
                    else:
                        fixed.update(operation_status=known, unconfirmed_raw_metadata=row)
                        error = 'raw metadata incomplete or unconfirmed; fixed unit fields retained'
            elif not isinstance(state.get(item['operation']), dict) or state[item['operation']].get('status') != 'RETURNED':
                unconfirmed, error = saved, 'operation receipt not returned; saved artifact retained as unconfirmed'
            elif item['kind'] == 'time':
                if not isinstance(saved, dict): raise ValueError('time submission object required')
                for key, coordinate in (('final_nodes', 'received_frame'), ('final_relations', 'received_left')):
                    rows = saved.get(key)
                    if not isinstance(rows, list): raise ValueError('fixed time rows missing')
                    for r in range(len(value[key])):
                        if r < len(rows) and isinstance(rows[r], dict) and rows[r].get(coordinate) == r: value[key][r] = rows[r]
                        else: error = 'incomplete time receipt; fixed rows retained'
                    if len(rows) != len(value[key]): error = 'time row count mismatch'
            else:
                rows = saved.get('slots') if isinstance(saved, dict) else None
                if not isinstance(rows, list): raise ValueError('slot receipt list required')
                for i, row in enumerate(rows[:32]):
                    if isinstance(row, dict) and type(row.get('slot')) is int and row['slot'] == i and 'value' in row and (row['value'] is None or type(row['value']) is int and row['value'] in (0, 1)) and isinstance(row.get('status'), str): value['slots'][i] = row
                    else: error = 'invalid slot record retained as unresolved'
                if len(rows) != 32: error = 'incomplete slot receipt; fixed32 retained'
        except (OSError, ValueError, TypeError) as exc: error = str(exc)
        result['recovered_artifacts'][item['path']] = {'kind': item['kind'], 'value': value, 'error': error, 'unconfirmed_artifact': unconfirmed}
    if not result['recovered_artifacts']: result['read_errors']['fixed_slots'] = 'saved attempt skeleton unreadable; use the standalone read-only CLI to reconstruct fixed unresolved slots'
elif Path(OUTPUT).exists() and (not Path(OUTPUT).is_dir() or any(Path(OUTPUT).iterdir())):
    result = {'status': 'OUTPUT_CONFLICT', 'physical_calls_this_invocation': 0}
else:
    try:
        with tempfile.TemporaryDirectory(prefix='time-payload-bootstrap-') as temporary:
            root = Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY is not None else Path(temporary)
            if SOURCE_DIRECTORY is None:
                files = ('scripts/b_new_source_chain_confirmation_v1_entry.py', 'scripts/b_conditional_time_output_v1_entry.py', 'scripts/b_framewise_qim_v1_entry.py', 'experiments/wan_state_clock/b_new_source_chain_confirmation_v1_records.py', 'experiments/wan_state_clock/b_minimal_l_v1_records.py', 'experiments/wan_state_clock/b_minimal_l_materialization_v1_records.py', 'experiments/wan_state_clock/b_framewise_qim_v1_records.py', 'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_records.py', 'experiments/wan_state_clock/b_conditional_time_validation_v1_records.py', 'experiments/wan_state_clock/b_conditional_time_output_v1_records.py', 'experiments/wan_state_clock/b_time_payload_cdc32_v1_records.py', 'experiments/wan_state_clock/configs/b_new_source_chain_confirmation_v1.json')
                for relative in files:
                    path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/'+relative, path)
            spec = importlib.util.spec_from_file_location('time_payload_entry', root/'scripts/b_new_source_chain_confirmation_v1_entry.py')
            entry = importlib.util.module_from_spec(spec)
            previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
            try: spec.loader.exec_module(entry)
            finally: sys.dont_write_bytecode = previous
            result = entry.launch(SOURCE, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF, MODEL_ID)
    except Exception as error:
        result = {'status': 'PREPARATION_FAILED', 'error': str(error), 'physical_calls_this_invocation': 0}
# Bounded display; complete recovery records remain in result and OUTPUT.
saved = result.get('saved', {})
for name in ('evaluation', 'result', 'resources'):
    if name not in saved:
        try: saved[name] = json.loads((Path(OUTPUT)/(name+'.json')).read_text())
        except (OSError, ValueError): pass
evaluation = saved.get('evaluation', {})
transfer = evaluation.get('payload_transfer', {})
summary = {'status': result.get('status'), 'output': str(OUTPUT),
           'evaluation_status': evaluation.get('status', result.get('evaluation_status')),
           'B_correct_supported': transfer.get('correct_supported'), 'B_fixed_slots': 160,
           'five_view_target': transfer.get('full_five_view_target_observed'),
           'time_counts': evaluation.get('time', {}).get('counts'),
           'synchronization_effect': evaluation.get('synchronization_effect', {}).get('status'),
           'fixed_denominators': {'time_nodes': 822, 'time_edges': 817, 'raw_bits': 9600, 'votes': 76800, 'local': 1440, 'final': 480},
           'arithmetic_upper_bound': 8813, 'not_model_inference_count': True,
           'codec': {'encode': 7, 'probe': 13, 'decode': 13},
           'total_wall_seconds': saved.get('resources', {}).get('total_wall_seconds'),
           'recovered_records': len(result.get('recovered_artifacts', {})),
           'errors': {str(k)[:80]: str(v)[:240] for k, v in list(result.get('errors', result.get('read_errors', {})).items())[:8]},
           'full_records': 'Full result and saved receipts remain in OUTPUT; no post on recovery.'}
print(summary)
"""
DESCRIPTION='# c02 新内容同链有限确认\n固定旧 CAL 玻璃工坊 c02 的 P0，一次执行原 A/B 物化、43 项时间接收、256 项载荷读取。time global_start=0；旧 a=11 仅管理/后评。不是全项目未见或全部组件独立留出。SOURCE、OUTPUT 已填写；普通源码位置可编辑，无摘要或版本门禁。默认远程入口从项目开发分支加载本次整包源码；也可填写已有源码目录。\n\n全部 B/W/U 计划先冻结，再共享 60 个实际 E41。所有时间和载荷盲终态保存后才统一后评，读取原 SOURCE/protocol.json 的 config.writer.bits 与 c02 记录；不可读或非法不回退常数。旧算法、hard/L/submit、DCT2/CDC32、冲突擦除和 native 有限候选规则原样复用。\n\n固定未来预算 A8457、B57、time43、payload256；异类算术上界8813不是模型推理次数。codec33（encode7/probe13/decode13）；post 时间1、载荷1、总汇总1单列，不增加观察。固定时间822节点/817边、载荷raw9600位/vote76800/local1440/final480，每臂160。各原阶段资源限制继续在实际路径执行，记录分阶段与总耗时，不把载荷1800秒套整包。\n\n第一次真实调用前整包预约并保存纯数据恢复骨架；失败/中断连未进入阶段一起消费。再次 Run all 仅展示已存回执，不补物化/读取/评分、不打开位缓存或 truth。首轮记录保存失败时不后评。完整文件和 result 变量保留；默认只显示有界摘要。\n\n五个 view 各 B32/32 且所有支持精确正确才达到 B160/160 目标，不能跨 view 拼凑。所有已提交时间位置/精确边包括旧 H 错误均保留；逐帧覆盖另报。B/U 共同位纠错、新输出支持、归属修正、丢失分别报告；B=U 同好可支持载荷迁移，但不能证明同步作用。仅归属修正不称解码纠错或 BER 下降。\n\n本包没有新增拒绝阈值或负例预算；W 错组不是 Ø。existence/null/FPR/隐藏错时间/来源认证未评估。原 native181/181 精确目标失败、4000步最优性及画质未证保留；旧失败不被新读回覆盖。本轮仅人工 CPU 工程检查；真实 P0 格式/有限性、新时间媒体及整链结果待未来授权执行。\n'

def build(root=ROOT):
    cells=[nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')\n"),
           nbformat.v4.new_markdown_cell(DESCRIPTION),nbformat.v4.new_code_cell(SETTINGS),nbformat.v4.new_code_cell(RUN)]
    for i,cell in enumerate(cells):
        cell['id']=f'new-source-chain-{i:02}'
        if cell.cell_type=='code':ast.parse(cell.source);cell.execution_count=None;cell.outputs=[]
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'),accelerator='GPU'))
    nbformat.validate(nb);path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,path)
    return dict(notebook=str(path),scientific_cells_executed=False)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False))
