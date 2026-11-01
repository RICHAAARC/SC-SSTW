"""One post-only complete fixed-row diagnostic. Never called by the blind receiver."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from . import b_time_layer_diagnostic_v1_records as rec
from .b_conditional_time_output_v1_evaluation import _edge_status


def at(rows,i):return rows[i] if isinstance(rows,list) and i<len(rows) else None
def obj(value):return value if isinstance(value,dict) else {}
def contains(bounds,n):
    return bool(bounds[0]<=n<=bounds[1]) if type(n) is int and isinstance(bounds,list) and len(bounds)==2 and all(type(v) is int for v in bounds) else None
def index_status(row,n,key='output_index'):
    if not isinstance(row,dict) or type(n) is not int:return 'UNKNOWN'
    value=row.get(key)
    return 'UNRESOLVED' if value is None else 'CORRECT' if value==n else 'WRONG'
def edge_report(row,a,b):
    if not isinstance(row,dict) or type(a) is not int or type(b) is not int:return dict(status='UNKNOWN')
    kind='REPEAT' if b==a else 'NORMAL' if b==a+1 else 'GAP'
    endpoints=row.get('output_endpoints')
    status='UNRESOLVED' if endpoints is None else _edge_status(row,a,b,old=isinstance(row.get('delta'),list))[0]
    return dict(status=status,true_kind=kind,saved_kind=row.get('kind'),saved=row,
                false_GAP=endpoints is not None and row.get('kind')=='GAP' and status!='CORRECT',
                false_REPEAT=endpoints is not None and row.get('kind')=='REPEAT' and status!='CORRECT',
                missed_GAP=kind=='GAP' and (status!='CORRECT' or row.get('kind')!='GAP'),
                missed_REPEAT=kind=='REPEAT' and (status!='CORRECT' or row.get('kind')!='REPEAT'))


def expected_bits(table,j,lane):
    if table is None or j is None or j<0:return None
    js,codes=table[lane];idx=int(np.searchsorted(js,j))
    if idx>=len(js) or int(js[idx])!=j:return None
    word=int(codes[idx]);return np.array([(word>>k)&1 for k in range(62,-1,-1)],dtype=bool)


def lane_report(unit,r,lane,n,table):
    j=None if n is None else n-((n-lane)%5)
    expected=expected_bits(table,j,lane);roles={}
    for role in rec.ROLES:
        trace=unit.get('fusion',{}).get(role);compatible=unit.get('compat',{}).get(role)
        row=at(at(obj(compatible).get('labels'),r),lane)
        if trace is None or not isinstance(row,dict):
            roles[role]=dict(status='UNKNOWN',e=None,u=None,expected_compatible=None,actual_hits=None,bit_slots=63);continue
        bits=trace['bits'][r,lane];known=trace['known'][r,lane];u=int(63-known.sum())
        e=None if expected is None else int(np.count_nonzero(known&(bits!=expected)))
        hits=row['labels']
        status=('UNWRITTEN' if j is not None and j<0 else 'UNKNOWN_EXPECTATION' if expected is None else
                'EXPECTED_COMPATIBLE' if hits==[j] else 'NO_COMPATIBLE_CODE' if not hits else 'WRONG_LABEL' if len(hits)==1 else 'AMBIGUOUS')
        roles[role]=dict(status=status,e=e,u=u,expected_compatible=None if e is None else 2*e+u<=14,
                        actual_hits=hits,hit_distances=row.get('hit_distances'),bit_slots=63)
    pair=dict(status='UNKNOWN')
    traces=unit.get('fusion',{})
    if all(role in traces for role in rec.ROLES):
        a,b=traces['R0'],traces['R1'];known=a['known'][r,lane]&b['known'][r,lane]
        x,y=a['bits'][r,lane],b['bits'][r,lane]
        pair=dict(common_known=int(known.sum()),conflicting_bits=np.flatnonzero(known&(x!=y)).tolist(),
            same_wrong_bits=None if expected is None else np.flatnonzero(known&(x==y)&(x!=expected)).tolist(),
            R0_right_R1_wrong=None if expected is None else np.flatnonzero(known&(x==expected)&(y!=expected)).tolist(),
            R1_right_R0_wrong=None if expected is None else np.flatnonzero(known&(y==expected)&(x!=expected)).tolist(),
            same_wrong_label=None if j is None or j<0 or any(roles[x]['actual_hits'] is None for x in ('R0','R1')) else bool(
                roles['R0']['actual_hits']==roles['R1']['actual_hits'] and len(roles['R0']['actual_hits'])==1 and roles['R0']['actual_hits']!=[j]),
            fusion_hits_differ_from_R0=None if any(roles[x]['actual_hits'] is None for x in ('F','R0')) else roles['F']['actual_hits']!=roles['R0']['actual_hits'],
            fusion_hits_differ_from_R1=None if any(roles[x]['actual_hits'] is None for x in ('F','R1')) else roles['F']['actual_hits']!=roles['R1']['actual_hits'])
    return dict(received_frame=r,lane=lane,truth=n,expected_j=j,schedule_status='UNKNOWN' if j is None else 'UNWRITTEN' if j<0 else 'WRITTEN',
                expected_word=None if expected is None else expected.astype(int).tolist(),roles=roles,replica_comparison=pair)


def transition(left,right,a,b,lane,role):
    x=left.get('fusion',{}).get(role);y=right.get('fusion',{}).get(role)
    if x is None or y is None:return dict(status='UNKNOWN')
    lx=at(at(obj(left.get('compat',{}).get(role)).get('labels'),a),lane)
    ly=at(at(obj(right.get('compat',{}).get(role)).get('labels'),b),lane)
    if lx is None or ly is None:return dict(status='UNKNOWN')
    raw_delta={}
    if 'raw' in left and 'raw' in right:
        for key in ('q','d0','d1','soft'):
            u=left['raw'][key][a,lane];v=right['raw'][key][b,lane]
            if role!='F':u=u[int(role[1])];v=v[int(role[1])]
            raw_delta[key+'_max_abs_change']=float(np.abs(u-v).max())
    return dict(status='OBSERVED',raw_observation_changes=raw_delta,bit_changes=np.flatnonzero(x['bits'][a,lane]!=y['bits'][b,lane]).tolist(),
                known_changes=np.flatnonzero(x['known'][a,lane]!=y['known'][b,lane]).tolist(),
                left_hits=lx['labels'],right_hits=ly['labels'],hits_changed=lx['labels']!=ly['labels'],
                soft_max_abs_change=float(np.abs(x['soft'][a,lane]-y['soft'][b,lane]).max()))


def evaluate(output,source,data,table,rows,check=lambda:None):
    output=Path(output);terminal=rec.load(output/'blind_terminal.json')
    if terminal.get('status')!='ALL_BLIND_DIAGNOSTICS_TERMINAL':raise ValueError('saved whole blind terminal required before truth')
    expected=set(rec.plan())-{'blind_save','post'}
    if set(terminal.get('rows',{}))!=expected or any(v.get('status') not in ('RETURNED','FAILED','INTERRUPTED','RESOURCE_UNRESOLVED','DEPENDENCY_UNRESOLVED') for v in terminal['rows'].values()):
        raise ValueError('all fixed blind items must have saved terminal receipts before truth')
    if rows.get('blind_save',{}).get('status')!='RETURNED':raise ValueError('blind save did not return successfully')
    truth_error=None;truth={}
    try:
        old=rec.load(Path(source)/'time_evaluation.json')
        for view,n in zip(rec.VIEWS,rec.LENGTHS):
            source_nodes=obj(obj(old.get('views')).get(view)).get('nodes')
            mapping=[obj(at(source_nodes,r)).get('truth') for r in range(n)]
            truth[view]=[value if type(value) is int and 0<=value<181 else None for value in mapping]
    except (OSError,ValueError,KeyError,TypeError) as error:truth_error=str(error)
    report=dict(status='POST_IN_PROGRESS',denominators=rec.denominators(),truth_read_error=truth_error,
        known_lane_findings={},node_status={},edge_status={},events={},unit_errors={},scientific_pass=None,
        claim_ceiling='first visible deviation only; no new H rule, repaired recovery, unique physical cause, identity or FPR claim',
        prior_results_retained=dict(time_nodes='798 correct / 4 wrong / 20 unresolved',time_edges='793 correct / 4 wrong / 20 unresolved',L='4 correct nodes / 8 NORMAL edges',
                                    payload='B160/160; referenced adopted prior result, not reread or reevaluated',native='target failure and unproved solver optimality retained'))
    technical=bool(truth_error) or any(v['status']!='RETURNED' for k,v in rows.items() if k not in ('blind_save','post'))
    counters={};lane_results={};folder=output/'post';folder.mkdir(exist_ok=True)
    def line(stream,value):check();stream.write(json.dumps(value,allow_nan=False)+'\n')
    def progress():
        report.update(known_lane_findings=dict(counters))
        rec.dump(output/'post_progress.json',report)
    with (folder/'lanes.jsonl').open('w') as lanes,(folder/'nodes.jsonl').open('w') as nodes,(folder/'edges.jsonl').open('w') as edges:
        for descriptor in rec.units():
            uid=descriptor['id'];n=descriptor['frames'];unit=data.get(uid,{})
            mapping=truth.get(uid.split('/')[-1],[None]*n) if descriptor['kind']=='received' else list(range(181))
            if any(v is None for v in mapping):technical=True
            saved=unit.get('saved',{});technical |= bool(saved.get('errors'))
            technical |= bool(unit.get('path',{}).get('saved_fusion_comparison',{}).get('status')=='UNKNOWN')
            report['unit_errors'][uid]=unit.get('errors',{}) or saved.get('errors',{})
            lane_results[uid]=[]
            for r in range(n):
                frame=[]
                for lane in range(5):
                    value=lane_report(unit,r,lane,mapping[r],table);frame.append(value);line(lanes,dict(unit=uid,**value))
                    for role,row in value['roles'].items():
                        key=descriptor['kind']+'/'+role+'/'+row['status'];counters[key]=counters.get(key,0)+1
                lane_results[uid].append(frame)
            if descriptor['kind']=='received':
                hard=obj(obj(saved.get('semantic.json')).get('hard',saved.get('blind.json')))
                final=obj(saved.get('submission.json'));recomputed=obj(obj(unit.get('path')).get('recomputed'))
                node_counts={k:Counter() for k in ('old_H','final','recomputed_F')};edge_counts={k:Counter() for k in node_counts};events=Counter()
                for r in range(n):
                    original=at(hard.get('nodes'),r);new=at(recomputed.get('nodes'),r);ntruth=mapping[r]
                    value=dict(unit=uid,received_frame=r,truth=ntruth,old_H=original,recomputed_F=new,
                        final=at(final.get('final_nodes'),r),local_truth_inside=contains(obj(original).get('intersection'),ntruth),
                        global_truth_inside=contains(obj(original).get('interval'),ntruth),
                        old_C_bound=at(obj(saved.get('C.npz')).get('bounds'),r),
                        old_L=at(obj(saved.get('candidates.json')).get('nodes'),r),
                        reproduction=None if original is None or new is None else original==new)
                    value['truth_in_old_C_bound']=contains(value['old_C_bound'],ntruth)
                    value['statuses']={k:index_status(value[k],ntruth) for k in node_counts}
                    for k,status in value['statuses'].items():node_counts[k][status]+=1
                    line(nodes,value)
                for r in range(n-1):
                    a,b=mapping[r:r+2];value=dict(unit=uid,received_left=r,truth_endpoints=[a,b])
                    for key,source_rows in (('old_H',hard.get('relations')),('final',final.get('final_relations')),('recomputed_F',recomputed.get('relations'))):
                        value[key]=edge_report(at(source_rows,r),a,b);edge_counts[key][value[key]['status']]+=1
                        for flag in ('false_GAP','false_REPEAT','missed_GAP','missed_REPEAT'):
                            if value[key].get(flag):events[key+'/'+flag]+=1
                    if type(a) is int and type(b) is int:events['truth/'+('REPEAT' if b==a else 'NORMAL' if b==a+1 else 'GAP')]+=1
                    line(edges,value)
                report['node_status'][uid]={k:dict(v) for k,v in node_counts.items()};report['edge_status'][uid]={k:dict(v) for k,v in edge_counts.items()};report['events'][uid]=dict(events)
            progress()
    with (folder/'first_visible.jsonl').open('w') as first,(folder/'transitions.jsonl').open('w') as changes:
        for n in range(181):
            for lane in range(5):
                for role in rec.ROLES:
                    statuses=[lane_results['material/'+layer][n][lane]['roles'][role]['status'] for layer in rec.LAYERS]
                    earliest='NONE_OBSERVED';unknown=False
                    for layer,status in zip(rec.LAYERS,statuses):
                        if status.startswith('UNKNOWN'):unknown=True
                        elif status not in ('EXPECTED_COMPATIBLE','UNWRITTEN'):
                            earliest='UNKNOWN' if unknown else layer;break
                    else:
                        if unknown:earliest='UNKNOWN'
                    line(first,dict(source_frame=n,lane=lane,role=role,layer_statuses=dict(zip(rec.LAYERS,statuses)),
                                    first_visible_deviation=earliest,unique_physical_cause=False))
                    for a,b in zip(rec.LAYERS,rec.LAYERS[1:]):
                        line(changes,dict(left='material/'+a,right='material/'+b,source_frame=n,lane=lane,role=role,
                            **transition(data.get('material/'+a,{}),data.get('material/'+b,{}),n,n,lane,role)))
        for view,n in zip(rec.VIEWS,rec.LENGTHS):
            mapping=truth.get(view,[None]*n)
            for r,source_frame in enumerate(mapping):
                for lane in range(5):
                    for role in rec.ROLES:
                        change=dict(status='UNKNOWN') if source_frame is None else transition(data.get('material/h264',{}),data.get('received/'+view,{}),source_frame,r,lane,role)
                        line(changes,dict(left='material/h264',right='received/'+view,source_frame=source_frame,received_frame=r,lane=lane,role=role,**change))
    report['status']='EVIDENCE_INCOMPLETE' if technical else 'DIAGNOSTIC_COMPLETE'
    progress();rec.dump(output/'evaluation_summary.json',report);return report
