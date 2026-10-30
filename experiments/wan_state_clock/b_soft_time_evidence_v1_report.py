"""Compact stdlib-only saved diagnosis display. No array load, score or retry."""
from collections import Counter
from .b_soft_time_evidence_v1_records import recover


def recovery_summary(value):return dict(state_counts=dict(Counter(r['status'] for r in value['rows'].values())),solver=dict(status='NOT_RUN_OFFLINE_NODE_DIAGNOSTIC'))


def display(output):
    value=recover(output);print('只读记录:',recovery_summary(value)['state_counts'],'固定分母:',value['denominators'])
    for name in ('resources','result'):
        record=value['saved'].get(name)
        print(name,record if record is not None else '未知: '+value['read_errors'].get(name,'missing'))
    evaluation=value['saved'].get('evaluation')
    if evaluation:
        for name in ('status','cohorts','transitions','old_mechanical_counts','numeric_attribution_unresolved','tolerance_unit_changes','constraint_ranking_differences_at_stored_C','attributed_model_transitions','truth_error'):
            print(name,evaluation.get(name,'未知'))
    else:print('后评未知:',value['read_errors'].get('evaluation','missing'))
    print('新增正式确定=0；无新关系/事件。L/G仅模型排名，数值归因未决不能记图收益。旧761/715保留不作新证据。')
    print('完整C/dL/Phi/delta/dG_C及所有首选掩码和旧正式结果保存在:',output)
    return recovery_summary(value)
