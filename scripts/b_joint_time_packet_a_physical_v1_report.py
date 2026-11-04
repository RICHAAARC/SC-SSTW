"""Compact display of saved stage/post metadata; never reads scores or raw arrays."""
from pathlib import Path
import json

STAGES=('gpu1','cpu_media','gpu2','post')


def build(output,stages):
    output=Path(output)
    summary=dict(stages={},native={},time={},identity={},scientific_pass=None,
                 claim_ceiling='Two seen sources; uncalibrated time diagnostics. No general FPR, source authentication or long-video recovery.')
    for stage in STAGES:
        item=stages.get(stage,{})
        meta=item.get('saved',{});result=meta.get('result',meta.get('blind_terminal',{}))
        is_terminal=meta.get('blind_terminal',{}).get('stage_terminal') is True
        status=result.get('status',item.get('status','NOT_STARTED'))
        if (output/stage/'attempt.json').exists() and not is_terminal:status='CONSUMED_WITHOUT_TERMINAL'
        summary['stages'][stage]=dict(status=status,
            terminal=is_terminal,seconds=result.get('wall_seconds'),
            errors=result.get('errors',{}),entry_error=item.get('error'),dependencies=item.get('dependencies',[]),
            metadata_read_errors=item.get('read_errors',{}),counts=meta.get('counts',{}),
            resources=meta.get('resources',{}),log=str(output/'preparation'/(stage+'.console.log')))
        if stage=='gpu1':summary['native']=result.get('sources',{})
        if stage=='post':
            summary['time']=result.get('results',{}).get('time',{})
            summary['identity']=result.get('results',{}).get('identity',{})
            summary['denominators']=result.get('denominators',{})
    lines=['# 联合时间包 A：Run all 已保存结果','',
           '| 阶段 | 状态 | 终态 | 秒 | 错误数 |','|---|---|---|---:|---:|']
    def show(value):
        if value is None:return '—'
        return f'{value:.6g}' if isinstance(value,float) else str(value)
    def row(values):lines.append('| '+' | '.join(show(x).replace('|','/') for x in values)+' |')
    for stage,item in summary['stages'].items():
        row((stage,item['status'],item['terminal'],item['seconds'],len(item['errors'])))
    for stage,item in summary['stages'].items():
        if item['entry_error']:lines.extend(['',stage+': '+item['entry_error'],''])
        if item['dependencies']:lines.extend(['',stage+' 等待终态：'+', '.join(item['dependencies']),''])
        missing=[name for name in ('blind_terminal','result') if name in item['metadata_read_errors']]
        if missing:lines.extend(['',stage+' 当前缺失或不可读：'+', '.join(name+'.json' for name in missing),''])
    lines.extend(['','固定时间分母：30 视图 / 4932 节点 / 4902 边 / 3107160 time bits。',
                  '固定身份分母：240 窗 / 38400 原始位 / 307200 coefficient 票。失败和缺失均保留。','',
                  '| 来源 | native 状态 | target max | 最大预算超出 |','|---|---|---:|---:|'])
    for source in ('n01','e02'):
        item=summary['native'].get(source,{})
        row((source,item.get('native_residual_status',item.get('status','MISSING')),item.get('target_max'),item.get('maximum_budget_excess')))
    time=summary['time'];identity=summary['identity']
    if time:
        lines.extend(['','时间后评：OLD 原正式输出与 NEW 未校准局部诊断分开保留；BASE 仅描述。',
                      '下表各计数依次为正确 / 错误 / 未决 / 未知；OLD-j 列是仅有局部 j 诊断的节点数。','',
                      '| 时间单元 | 节点计数 | OLD-j | 边计数 | 假 GAP / REPEAT |',
                      '|---|---|---:|---|---|'])
        for uid,unit in time.get('units',{}).items():
            counts=unit.get('counts',{})
            def count(prefix,status):
                return counts.get(prefix+status,0)+counts.get(prefix+'DESCRIPTIVE_GEOMETRY_'+status,0)
            row((uid,' / '.join(str(count('local/',s)) for s in ('CORRECT','WRONG','UNRESOLVED','UNKNOWN')),
                 count('local/','OLD_J_LOCAL_ONLY'),
                 ' / '.join(str(count('edge/',s)) for s in ('CORRECT','WRONG','UNRESOLVED','UNKNOWN')),
                 f"{counts.get('false_GAP',0)} / {counts.get('false_REPEAT',0)}"))
        lines.extend(['','| 来源 | 旧正确 / 错误 | 旧基线缺失 | 修正旧错 | 损坏旧对 | 丢失旧对 | NEW 未知 |',
                      '|---|---|---:|---:|---:|---:|---:|'])
        for source,item in time.get('preservation',{}).items():
            row((source,f"{item.get('old_correct')} / {item.get('old_wrong')}",item.get('baseline_unavailable'),
                 item.get('corrected_old_wrong'),item.get('damaged_old_correct'),item.get('lost_old_correct'),item.get('new_local_unknown')))
    else:lines.extend(['','时间后评尚无已保存汇总；不解释为零错误。'])
    if identity:
        lines.extend(['','身份后评：每行固定 1920 OLD/NEW raw pairs；共同可比 BER 与新增可读覆盖分别报告。','',
                      '| 来源/视图 | 共同可读 | 共同可比 | OLD 错误 / BER | NEW 错误 / BER | 新增 / 丢失 | 几何未决 |',
                      '|---|---:|---:|---|---|---|---:|'])
        for uid,item in identity.get('comparisons',{}).items():
            row((uid,item.get('common_readable'),item.get('common_comparable'),
                 f"{show(item.get('OLD_common_errors'))} / {show(item.get('OLD_common_BER'))}",
                 f"{show(item.get('NEW_common_errors'))} / {show(item.get('NEW_common_BER'))}",
                 f"{item.get('added_readable')} / {item.get('lost_readable')}",item.get('common_geometry_unresolved')))
        lines.extend(['','false attribution：盲读器未作 group 归属；新增覆盖不当作 BER 改善。'])
    else:lines.extend(['','身份后评尚无已保存汇总；不解释为零错误。'])
    for stage,item in summary['stages'].items():
        if item['errors']:
            lines.extend(['',stage+' 已保存错误：','```json',json.dumps(item['errors'],ensure_ascii=False,indent=2),'```'])
    lines.extend(['','formal_output=null；scientific_pass=null。startup、冲突、逆序、未知及失败保留在完整逐行结果中。',
                  '结果范围：两个已见来源及固定媒体；不作通用 FPR、来源认证或长视频恢复结论。','',
                  '完整结果：`'+str(output/'post')+'`（evaluation / time_nodes / time_edges / identity_rows）。',
                  '逐阶段日志与物理计数：各阶段目录及 `'+str(output/'preparation')+'`。'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'run_all_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'run_all_summary.md').write_text('\n'.join(lines)+'\n')
    return summary
