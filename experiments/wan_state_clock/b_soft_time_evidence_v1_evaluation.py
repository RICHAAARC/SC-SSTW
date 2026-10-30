"""Post-only diagnostic counts. Truth classifications never feed ranking."""
from pathlib import Path
import zipfile
from .b_soft_time_evidence_v1_records import load,denominators,LENGTHS


def category(node,name,truth,missing='NOT_EXECUTED'):
    if node is None:return missing
    if node['model_status']!='DEFINED':return node['model_status']
    index=node[name+'_model_index']
    return 'MODEL_TIE' if index is None else 'MODEL_BEST_CORRECT' if index==truth else 'MODEL_BEST_WRONG'


def saved_local_nodes(folder,R):
    """Display already computed L. Only new output metadata/masks; no score or old input read."""
    import numpy as np
    cost=load(folder/'C.json');local=load(folder/'L.json')
    if not isinstance(cost,dict) or not isinstance(local,dict) or len(cost['states'])!=R or len(local['status'])!=R:raise ValueError('invalid saved local metadata')
    with np.load(folder/'C.npz',allow_pickle=False) as archive:
        bounds=archive['bounds'];offsets=archive['offsets']
    with np.load(folder/'L.npz',allow_pickle=False) as archive:best=archive['best']
    if bounds.shape!=(R,2) or bounds.dtype.kind not in 'iu' or offsets.shape!=(R+1,) or offsets.dtype.kind not in 'iu' or best.ndim!=1 or best.dtype!=np.bool_:
        raise ValueError('invalid saved local candidate/mask format')
    if offsets[0]!=0 or offsets[-1]!=len(best) or np.any(np.diff(offsets)<0):raise ValueError('invalid saved local offsets')
    nodes=[]
    for r,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        lo,hi=map(int,bounds[r]);state=cost['states'][r];status=local['status'][r]
        if not isinstance(state,str) or not isinstance(status,str):raise ValueError('invalid saved local state')
        if not ((0<=lo<=hi<92672 and b-a==hi-lo+1) or (lo==hi==-1 and a==b)):raise ValueError('invalid saved local candidate interval')
        mask=best[a:b];count=int(mask.sum())
        nodes.append(dict(received_frame=r,model_status=state,L_status=status,L_count=count,
                          L_model_index=int(lo+np.flatnonzero(mask)[0]) if state=='DEFINED' and count==1 else None))
    return nodes


def evaluate(source,output,rows):
    if any(row['status'] in ('STARTED','NOT_STARTED') for row in rows.values()):raise RuntimeError('blind work not terminal before truth')
    source,output=Path(source),Path(output);views=[];truth_error=None
    try:
        truth=load(source/'private/attack_truth.json')
        tasks=truth['views']
        if len(tasks)!=10 or {t['id'] for t in tasks}!={f'v{i:02}' for i in range(10)}:raise ValueError('fixed ten-view truth required')
        truth_by_id={t['id']:t for t in tasks}
    except (OSError,ValueError,KeyError,TypeError) as error:truth_by_id={};truth_error=str(error)
    cohorts={name:{branch:{} for branch in ('L','G_native','G_aligned')} for name in ('OLD_UNRESOLVED_NONSTARTUP','OLD_UNRESOLVED_STARTUP','OLD_DETERMINED','BASE','TRUTH_UNRESOLVED')}
    transitions={name:{} for name in ('L_to_G_native','L_to_G_aligned')}
    old_counts=dict(determined_correct=0,determined_wrong=0,unresolved=0,relations_correct=0,relations_wrong=0,relations_unresolved=0,
                    QIM_frames_unknown=0,QIM_relations_unknown=0,BASE_views_unknown=0,BASE_label_frames_unknown=0,BASE_j0_labels=0,BASE_supported_views=0)
    numeric_unresolved=scale_changes=constraint_changes=0;attributed_transitions={}
    for i,R in enumerate(LENGTHS):
        ident=f'v{i:02}';folder=output/'views'/ident;task=truth_by_id.get(ident);errors={};old=ranking=None;local_nodes=None
        try:
            old=load(folder/'old_blind.json')
            if not isinstance(old,dict) or len(old['nodes'])!=R or len(old['relations'])!=R-1:raise ValueError('invalid old formal result')
            if old.get('qualification') not in ('SUPPORTED_RELATION_SET','INSUFFICIENT_MULTILANE_EVIDENCE','IMPLEMENTATION_UNRESOLVED'):raise ValueError('invalid old qualification')
            for node in old['nodes']:
                if not isinstance(node,dict) or 'output_index' not in node:raise ValueError('old formal node must be an object with output_index')
                index=node.get('output_index')
                if index is not None and (type(index)!=int or not 0<=index<92672):raise ValueError('invalid old formal index')
            for relation in old['relations']:
                if not isinstance(relation,dict) or 'output_endpoints' not in relation:raise ValueError('old formal relation must be an object with output_endpoints')
                endpoints=relation.get('output_endpoints')
                if endpoints is not None and (not isinstance(endpoints,list) or len(endpoints)!=2 or any(type(n)!=int or not 0<=n<92672 for n in endpoints)):raise ValueError('invalid old formal relation')
        except (OSError,ValueError,KeyError,TypeError) as error:errors['old_formal']=str(error);old=None
        if rows[ident+'/G']['status']=='RETURNED':
            try:
                ranking=load(folder/'blind_ranking.json')
                if len(ranking['nodes'])!=R or ranking['formal_new_determined']!=0 or ranking['new_relation_outputs']!=0:raise ValueError('invalid node-only ranking')
                required={'model_status','attribution','native_aligned_differ','L_G_native_same_set','L_G_aligned_same_set'}
                required.update(branch+suffix for branch in ('L','G_native','G_aligned') for suffix in ('_model_index','_status'))
                for node in ranking['nodes']:
                    if not isinstance(node,dict) or not required<=node.keys():raise ValueError('invalid ranking node fields')
                    if any(type(node[k]) is not bool for k in ('native_aligned_differ','L_G_native_same_set','L_G_aligned_same_set')):raise ValueError('invalid ranking set-comparison flags')
                    if not isinstance(node['model_status'],str) or not isinstance(node['attribution'],str):raise ValueError('invalid ranking status')
                    for branch in ('L','G_native','G_aligned'):
                        index=node[branch+'_model_index']
                        if not isinstance(node[branch+'_status'],str) or (index is not None and (type(index)!=int or not 0<=index<92672)):raise ValueError('invalid ranking model index/status')
            except (OSError,ValueError,KeyError,TypeError) as error:errors['ranking']=str(error);ranking=None
        if rows[ident+'/L']['status']=='RETURNED':
            if ranking is not None:local_nodes=ranking['nodes']
            else:
                try:local_nodes=saved_local_nodes(folder,R)
                except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile) as error:errors['local_ranking']=str(error)
        mapping=None;arm=None
        try:
            if task is not None:
                mapping=task['mapping'];arm=task['arm']
                if arm!=('QIM' if i<5 else 'BASE') or len(mapping)!=R or any(type(n)!=int or n<0 or n>=92672 for n in mapping):raise ValueError('invalid post-only source mapping')
        except (ValueError,KeyError,TypeError) as error:errors['truth']=str(error);mapping=None;arm=None
        if old is None or arm is None:
            if i<5:old_counts['QIM_frames_unknown']+=R;old_counts['QIM_relations_unknown']+=R-1
            else:old_counts['BASE_views_unknown']+=1;old_counts['BASE_label_frames_unknown']+=R
        details=[]
        for r in range(R):
            node=ranking['nodes'][r] if ranking is not None else None
            local_node=local_nodes[r] if local_nodes is not None else None
            formal=old['nodes'][r].get('output_index') if old is not None else None
            n=mapping[r] if mapping is not None else None
            cohort='TRUTH_UNRESOLVED' if arm is None or old is None else 'BASE' if arm=='BASE' else 'OLD_DETERMINED' if formal is not None else 'OLD_UNRESOLVED_STARTUP' if n<4 else 'OLD_UNRESOLVED_NONSTARTUP'
            result=dict(received_frame=r,cohort=cohort,truth=n if arm=='QIM' else None,old_formal_index=formal,formal_new_determined=0)
            for branch in ('L','G_native','G_aligned'):
                branch_node=local_node if branch=='L' else node
                receipt=rows[ident+('/L' if branch=='L' else '/G')]['status']
                missing='SAVED_RESULT_UNRESOLVED' if receipt=='RETURNED' else 'FAILED_UNRESOLVED' if receipt in ('FAILED','INTERRUPTED','RESOURCE_UNRESOLVED','STARTED') else 'NOT_EXECUTED'
                label='TRUTH_UNRESOLVED' if cohort=='TRUTH_UNRESOLVED' else (missing if branch_node is None else branch_node[branch+'_status']) if arm=='BASE' else category(branch_node,branch,n,missing)
                result[branch]=label;counts=cohorts[cohort][branch];counts[label]=counts.get(label,0)+1
                result[branch+'_model_index']=branch_node[branch+'_model_index'] if branch_node else None
            if node is not None:
                result.update(attribution=node['attribution'],native_aligned_differ=node['native_aligned_differ'],
                              L_G_native_same_set=node['L_G_native_same_set'],L_G_aligned_same_set=node['L_G_aligned_same_set'])
                numeric_unresolved+=node['attribution']=='NUMERIC_ATTRIBUTION_UNRESOLVED';scale_changes+=node['native_aligned_differ']
                constraint_changes+=node['attribution']=='CONSTRAINT_RANKING_DIFFERENCE_AT_STORED_C'
            if arm=='QIM':
                for branch in ('G_native','G_aligned'):
                    a,b=result['L'],result[branch]
                    same_set=node['L_'+branch+'_same_set'] if node else None
                    change='UNRESOLVED_COMPARISON' if node is None or local_node is None else 'CORRECTION' if a=='MODEL_BEST_WRONG' and b=='MODEL_BEST_CORRECT' else 'DAMAGE' if a=='MODEL_BEST_CORRECT' and b=='MODEL_BEST_WRONG' else 'TIE_CHANGE' if (a!=b or not same_set) and 'MODEL_TIE' in (a,b) else 'OTHER_CHANGE' if a!=b or not same_set else 'UNCHANGED'
                    counts=transitions['L_to_'+branch];counts[change]=counts.get(change,0)+1
                    result['L_to_'+branch]=change
                    if branch=='G_aligned' and node is not None and node['attribution']=='CONSTRAINT_RANKING_DIFFERENCE_AT_STORED_C':attributed_transitions[change]=attributed_transitions.get(change,0)+1
                if old is not None:old_counts['unresolved' if formal is None else 'determined_correct' if formal==n else 'determined_wrong']+=1
            details.append(result)
        if old is not None and arm=='QIM':
            for r,relation in enumerate(old['relations']):
                endpoints=relation.get('output_endpoints')
                old_counts['relations_unresolved' if endpoints is None else 'relations_correct' if endpoints==mapping[r:r+2] else 'relations_wrong']+=1
        if old is not None and arm=='BASE':
            old_counts['BASE_supported_views']+=old.get('qualification')=='SUPPORTED_RELATION_SET'
            try:
                labels=old['labels']
                if not isinstance(labels,list) or len(labels)!=R or any(not isinstance(frame,list) or len(frame)!=5 or any(not isinstance(row,dict) or type(row.get('lane')) is not int or not isinstance(row.get('labels'),list) for row in frame) or {row['lane'] for row in frame}!=set(range(5)) for frame in labels):raise ValueError('invalid old BASE label records')
                old_counts['BASE_j0_labels']+=sum(row.get('lane')==0 and row.get('labels')==[0] for frame in labels for row in frame)
            except (ValueError,KeyError,TypeError) as error:errors['old_labels']=str(error);old_counts['BASE_label_frames_unknown']+=R
        views.append(dict(id=ident,arm=arm,view=task.get('view') if task else None,frames=R,relations=R-1,ranking_status=rows[ident+'/G']['status'],
                          L_receipt_status=rows[ident+'/L']['status'],G_receipt_status=rows[ident+'/G']['status'],
                          old_formal_preserved=old is not None,old_qualification=old.get('qualification') if old else None,errors=errors,nodes=details))
    try:
        resources=load(output/'resources.json')
        if not isinstance(resources,dict):raise ValueError('resources must be an object')
    except (OSError,ValueError) as error:resources=dict(status='UNRESOLVED_RECEIPT',error=str(error))
    resource_failure=resources.get('status')!='CLOSED' or bool(resources.get('error'))
    incomplete=any(row['status']!='RETURNED' for row in rows.values()) or bool(truth_error) or any(v['errors'] for v in views) or resource_failure
    return dict(status='DIAGNOSTIC_WITH_UNRESOLVED_EXECUTION' if incomplete else 'NODE_DIAGNOSTIC_COMPLETE',denominators=denominators(),views=views,
                cohorts=cohorts,transitions=transitions,old_mechanical_counts=old_counts,truth_error=truth_error,resources=resources,
                numeric_attribution_unresolved=numeric_unresolved,tolerance_unit_changes=scale_changes,
                constraint_ranking_differences_at_stored_C=constraint_changes,attributed_model_transitions=attributed_transitions,
                formal_new_determined=0,new_relations=0,
                interpretation='Transitions are model-preference observations. Uncertified numeric differences are not graph gains; old outputs are mechanical retention, not new evidence.')
