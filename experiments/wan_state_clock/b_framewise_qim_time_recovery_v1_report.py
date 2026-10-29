"""Optional compact stdlib saved display; never launches or repairs anything."""
from collections import Counter
from .b_framewise_qim_time_recovery_v1_records import recover


def recovery_summary(value):
    return dict(state_counts=dict(Counter(r['status'] for r in value['rows'].values())),
                solver=dict(status='NOT_RUN_CPU_TIME_RECEIVER'))


def display(output):
    value=recover(output)
    print('只读记录:',recovery_summary(value)['state_counts'],'固定分母:',value['denominators'])
    evaluation=value['saved'].get('evaluation',{})
    print('结果:',evaluation.get('outcome','UNRESOLVED'),'失败事实:',evaluation.get('failure_facts','UNRESOLVED'))
    print('最终资源:',value['saved'].get('resources','UNRESOLVED'))
    for row in evaluation.get('views',[]):
        if not isinstance(row,dict):continue
        print(row.get('arm'),row.get('view'),'接收:',row.get('receiver_status'),
              '对应:',row.get('frame_counts'),'关系:',row.get('relation_counts'),
              '支撑:',row.get('supported'),'BASE错误接受:',row.get('base_false_accept'),
              '局部位:',row.get('local_bits'),'标签:',row.get('local_labels'),
              '后评错误:',row.get('evaluation_error'))
    print('原native精确落点失败、最优性未证保持；仅单源固定剪辑同二代H264时间关系，不含剪辑后身份。')
    print('细表和原始读量:',output)
    return recovery_summary(value)
