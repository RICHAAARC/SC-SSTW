"""Post-only truth and actual old-source support; no parameter or path selection."""
from pathlib import Path
from collections import Counter
import json
from . import b_native_y_diagnostic_v1_records as rec
from .b_time_layer_diagnostic_v1_evaluation import at,obj,index_status,edge_report,contains


def label_status(row,expected):
    if not isinstance(row,dict) or not isinstance(row.get('labels'),list):return 'UNKNOWN'
    if expected is None:return 'UNKNOWN_EXPECTATION'
    if expected<0:return 'UNWRITTEN'
    hits=row['labels']
    return 'EXPECTED_COMPATIBLE' if hits==[expected] else 'NO_COMPATIBLE_CODE' if not hits else 'WRONG_LABEL' if len(hits)==1 else 'IMPLEMENTATION_UNRESOLVED'


def support(old,new):
    if not isinstance(old,dict) or not isinstance(new,dict):return 'UNKNOWN'
    value=old.get('output_index')
    if value is None:return 'OLD_UNRESOLVED'
    if new.get('output_index')==value:return 'PRESERVED_EXACT'
    if new.get('output_index') is not None:return 'CHANGED_EXACT'
    return 'BECAME_UNRESOLVED'


def new_error(old_status,new_status,correct='CORRECT',unresolved='UNRESOLVED',wrong='WRONG'):
    known=(correct,unresolved,wrong)
    if old_status not in known or new_status not in known:return None
    return new_status==wrong and old_status in (correct,unresolved)


def added_correct(hard_status,final_status):
    if hard_status not in ('CORRECT','UNRESOLVED','WRONG') or final_status not in ('CORRECT','UNRESOLVED','WRONG'):return None
    return final_status=='CORRECT' and hard_status!='CORRECT'


def feasible_index(node,value):
    return contains(obj(node).get('interval'),value)


def l_node_status(old,new):
    if not isinstance(old,dict) or not isinstance(new,dict):return 'UNKNOWN'
    n=old.get('output_index')
    if type(n) is not int:return 'NOT_OLD_OUTPUT'
    if new.get('output_index')==n:return 'DETERMINED_BY_Y_H'
    inside=feasible_index(new,n)
    return 'UNKNOWN' if inside is None else 'FEASIBLE_UNDETERMINED' if inside else 'EXCLUDED_FROM_Y_GRAPH'


def old_source_rows(path):
    """Read only inside post. Duplicate/malformed slots stay UNKNOWN, not idealized."""
    rows={};bad=set();errors=[]
    try:
        with Path(path).open() as stream:
            for number,line in enumerate(stream,1):
                try:
                    value=json.loads(line)
                    if not isinstance(value,dict):raise ValueError('object required')
                    if value.get('unit')!='material/h264':continue
                    r,l=value.get('received_frame'),value.get('lane')
                    if type(r) is not int or type(l) is not int or not 0<=r<181 or not 0<=l<5:raise ValueError('source slot coordinate')
                    key=(r,l);role=obj(value.get('roles')).get('F')
                    if key in rows or key in bad:
                        rows.pop(key,None);bad.add(key);raise ValueError('duplicate source slot')
                    hits=obj(role).get('actual_hits');status=obj(role).get('status')
                    if not isinstance(hits,list) or any(type(j) is not int or not 0<=j<92672 or j%5!=l for j in hits):
                        bad.add(key);raise ValueError('invalid source F hits')
                    expected=r-((r-l)%5);computed=label_status(dict(labels=hits),expected)
                    # The original67 evaluator calls a multi-hit row AMBIGUOUS.
                    if computed!=status and not (computed=='IMPLEMENTATION_UNRESOLVED' and status=='AMBIGUOUS'):
                        bad.add(key);raise ValueError('inconsistent source F saved status')
                    rows[key]=dict(labels=hits,status=status)
                except (ValueError,TypeError,KeyError) as error:errors.append(dict(line=number,error=str(error)))
    except OSError as error:errors.append(dict(error=str(error)))
    return rows,errors


def evaluate(output,config,data,rows,check=lambda:None):
    output=Path(output);terminal=rec.load(output/'blind_terminal.json');expected=set(rec.plan())-{'blind_save','post'}
    if terminal.get('status')!='ALL_BLIND_DIAGNOSTICS_TERMINAL' or set(terminal.get('rows',{}))!=expected or rows['blind_save']['status']!='RETURNED':
        raise ValueError('saved complete fixed blind terminal required before truth')
    if any(r.get('status') not in ('RETURNED','FAILED','INTERRUPTED','RESOURCE_UNRESOLVED','DEPENDENCY_UNRESOLVED') for r in terminal['rows'].values()):
        raise ValueError('unfinished blind item before post')
    truth={};truth_error=None
    try:
        saved=rec.load(Path(config['source_chain'])/'time_evaluation.json')
        for view,n in zip(rec.VIEWS,rec.LENGTHS):
            source=obj(obj(saved.get('views')).get(view)).get('nodes')
            mapping=[obj(at(source,r)).get('truth') for r in range(n)]
            truth[view]=[v if type(v) is int and 0<=v<181 else None for v in mapping]
    except (OSError,ValueError,TypeError,KeyError) as error:truth_error=str(error)
    old_source,source_errors=old_source_rows(config['post_source_baseline'])
    technical=bool(truth_error or source_errors) or any(row['status']!='RETURNED' for k,row in rows.items() if k not in ('blind_save','post'))
    report=dict(status='POST_IN_PROGRESS',counts={},views={},source_control={},profiles={},denominators=rec.denominators(),
                truth_error=truth_error,source_baseline_errors=source_errors,source_baseline_slots=len(old_source),scientific_pass=None,
                prior_context=dict(H='794 correct / 4 wrong / 24 unresolved',final='798 correct / 4 wrong / 20 unresolved',
                    L='4 added correct nodes / 8 NORMAL edges; no Y-L ranking',payload='B160/160 retained, not reread',native='target tolerance failure / optimality unproved retained'),
                claim_ceiling='fixed M_Y shadow original H only; correlated c02 views; no formal replacement or identity/FPR/general recovery claim')
    folder=output/'post';folder.mkdir(exist_ok=True);totals=Counter();r5=[]
    def line(stream,value):check();stream.write(json.dumps(value,allow_nan=False)+'\n')
    def progress():report['counts']=dict(totals);rec.dump(output/'post_progress.json',report)
    with (folder/'lanes.jsonl').open('w') as lanes,(folder/'nodes.jsonl').open('w') as nodes,(folder/'edges.jsonl').open('w') as edges:
        for uid,n in rec.UNITS:
            unit=data.get(uid,{});public=obj(unit.get('public'));hard=obj(obj(public.get('semantic.json')).get('hard',public.get('blind.json')))
            final=obj(public.get('submission.json'));path=obj(unit.get('path'));compatible=obj(unit.get('compatible'))
            mapping=list(range(181)) if uid=='SOURCE' else truth.get(uid,[None]*n)
            counts=Counter();profile=obj(unit.get('probe')).get('profile');report['profiles'][uid]=profile
            complete=all(v is not None for v in mapping) and isinstance(compatible.get('labels'),list) and len(compatible['labels'])==n and profile is not None
            if uid=='SOURCE':complete &= len(old_source)==905 and not source_errors
            else:
                complete &= not public.get('errors')
                complete &= all(isinstance(obj(owner).get(key),list) and len(owner[key])==size for owner,key,size in
                    ((hard,'nodes',n),(hard,'relations',n-1),(hard,'labels',n),(final,'final_nodes',n),(final,'final_relations',n-1),(path,'nodes',n),(path,'relations',n-1)))
            for r,t in enumerate(mapping):
                for lane in range(5):
                    expected_j=None if t is None else t-((t-lane)%5)
                    new=at(at(compatible.get('labels'),r),lane)
                    old=old_source.get((r,lane)) if uid=='SOURCE' else at(at(hard.get('labels'),r),lane)
                    old_status=label_status(old,expected_j);new_status=label_status(new,expected_j)
                    preserved=None if old_status!='EXPECTED_COMPATIBLE' or new_status.startswith('UNKNOWN') else new_status=='EXPECTED_COMPATIBLE'
                    new_wrong=new_error(old_status,new_status,'EXPECTED_COMPATIBLE','NO_COMPATIBLE_CODE','WRONG_LABEL')
                    line(lanes,dict(unit=uid,received_frame=r,lane=lane,truth=t,expected_j=expected_j,
                        schedule_status='UNKNOWN' if t is None else 'UNWRITTEN' if expected_j<0 else 'WRITTEN',
                        old_F=old,Y_F=new,old_status=old_status,Y_status=new_status,old_correct_support_preserved=preserved,new_wrong_label=new_wrong))
                    counts['lane/old/'+old_status]+=1;counts['lane/Y/'+new_status]+=1
                    if old_status=='EXPECTED_COMPATIBLE':counts['lane/old_correct_to/'+new_status]+=1
                    if new_wrong:counts['lane/new_wrong_label']+=1
                    if new_status.startswith('UNKNOWN') or old_status.startswith('UNKNOWN'):complete=False
                if uid=='SOURCE':continue
                old=at(hard.get('nodes'),r);f=at(final.get('final_nodes'),r);new=at(path.get('nodes'),r)
                hs=index_status(old,t);fs=index_status(f,t);ys=index_status(new,t);preservation=support(old,new)
                added=added_correct(hs,fs);ls='UNKNOWN' if added is None else l_node_status(f,new) if added else 'NOT_L_ADDED_CORRECT'
                value=dict(view=uid,received_frame=r,truth=t,old_H=old,old_final=f,Y_H=new,
                    old_H_status=hs,old_final_status=fs,Y_H_status=ys,old_H_support=preservation,
                    truth_in_Y_graph=feasible_index(new,t),old_L_added_correct=added,old_L_status=ls,
                    new_wrong=new_error(hs,ys))
                if r==5 and uid in ('FULL','DELETE1','REPEAT1','DELETE3'):
                    wrong_possible=feasible_index(new,4)
                    disposition='UNKNOWN' if ys=='UNKNOWN' or wrong_possible is None else 'TRUE5_UNIQUE_WRONG4_EXCLUDED' if obj(new).get('output_index')==5 and wrong_possible is False else 'WRONG4_RETAINED' if obj(new).get('output_index')==4 else 'OTHER_WRONG' if ys=='WRONG' else 'UNRESOLVED'
                    value['r5']=dict(status=disposition,wrong4_possible=wrong_possible,true5_possible=feasible_index(new,5));r5.append(dict(view=uid,**value['r5']))
                    counts['r5/'+disposition]+=1
                line(nodes,value)
                for k,s in (('old_H',hs),('old_final',fs),('Y_H',ys)):counts['node/'+k+'/'+s]+=1
                counts['old_H_'+hs+'/'+preservation]+=1
                if added:counts['old_L_node/'+ls]+=1
                if value['new_wrong'] and hs=='CORRECT':counts['node/new_wrong_on_old_correct']+=1
                if value['new_wrong']:counts['node/new_wrong_on_old_not_wrong']+=1
                if 'UNKNOWN' in (hs,fs,ys):complete=False
            if uid!='SOURCE':
                for r in range(n-1):
                    a,b=mapping[r:r+2];old=edge_report(at(hard.get('relations'),r),a,b);f=edge_report(at(final.get('final_relations'),r),a,b);new=edge_report(at(path.get('relations'),r),a,b)
                    endpoints=obj(f.get('saved')).get('output_endpoints');possible=None
                    if isinstance(endpoints,list) and len(endpoints)==2 and all(type(x) is int for x in endpoints):
                        left=feasible_index(at(path.get('nodes'),r),endpoints[0]);right=feasible_index(at(path.get('nodes'),r+1),endpoints[1])
                        if left is not None and right is not None:possible=left and right and endpoints[0]<=endpoints[1]
                    added=added_correct(old['status'],f['status'])
                    l_status='DETERMINED_BY_Y_H' if new['status']=='CORRECT' else 'UNKNOWN' if possible is None else 'FEASIBLE_UNDETERMINED' if possible else 'EXCLUDED_FROM_Y_GRAPH'
                    value=dict(view=uid,received_left=r,truth_endpoints=[a,b],old_H=old,old_final=f,Y_H=new,
                        old_L_added_correct=added,old_final_pair_in_Y_graph=possible,old_L_status='UNKNOWN' if added is None else l_status if added else 'NOT_L_ADDED_CORRECT',
                        new_wrong=new_error(old['status'],new['status']),
                        old_correct_preserved=None if old['status']!='CORRECT' or new['status']=='UNKNOWN' else new['status']=='CORRECT')
                    line(edges,value)
                    for name,item in (('old_H',old),('old_final',f),('Y_H',new)):counts['edge/'+name+'/'+item['status']]+=1
                    if old['status']=='CORRECT':counts['edge/old_correct_to/'+new['status']]+=1
                    if old.get('false_GAP'):counts['old_false_GAP_to/'+new['status']+'/'+str(new.get('saved_kind'))]+=1
                    if old.get('false_REPEAT'):counts['old_false_REPEAT_to/'+new['status']+'/'+str(new.get('saved_kind'))]+=1
                    if new.get('true_kind'):counts['true_event/'+new['true_kind']+'/'+new['status']]+=1
                    if value['new_wrong']:counts['edge/new_wrong_on_old_not_wrong']+=1
                    if added:counts['old_L_edge/'+l_status]+=1
                    if any(x['status']=='UNKNOWN' for x in (old,f,new)):complete=False
                report['views'][uid]=dict(complete=bool(complete),counts=dict(counts),path_status=path.get('path_status'),qualification=path.get('qualification'),public_errors=public.get('errors',{}))
                totals.update(counts)
            else:report['source_control']=dict(complete=bool(complete),counts=dict(counts),scope='905 auxiliary lanes; excluded from received totals')
            technical |= not complete;progress()
    profiles=list(report['profiles'].values())
    report['model_applicability']='CONFLICT' if any(obj(p).get('status')=='MODEL_APPLICABILITY_CONFLICT' for p in profiles) else 'UNCONFIRMED' if any(obj(p).get('status')!='METADATA_CONSISTENT' for p in profiles) else 'METADATA_CONSISTENT_NOT_PROOF'
    report['r5']=r5
    report['all_four_r5_corrected']=len(r5)==4 and all(x['status']=='TRUE5_UNIQUE_WRONG4_EXCLUDED' for x in r5)
    report['status']='EVIDENCE_INCOMPLETE' if technical else 'DIAGNOSTIC_COMPLETE'
    progress();rec.dump(output/'evaluation_summary.json',report);return report
