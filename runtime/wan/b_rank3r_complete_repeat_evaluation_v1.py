"""Post-only two-reader accounting. No model, media or threshold calibration."""
from collections import Counter
from pathlib import Path
import time
from runtime.wan import b_rank3r_complete_repeat_v1 as run
from runtime.wan.b_rank3_four_source_confirmation_evaluation_v1 import assess as original_assess,paired

def assess(decision,bits,positive,frames):
    result=original_assess(decision,bits,positive,frames)
    for row in result['slots']:
        aliases={}
        for ref in row['actual_supporting_refs']:
            if ref['actual_full']:
                aliases.setdefault(str(ref['actual_cell']),set()).add(ref['evidence_id'])
        row['source_cell_aliases']={cell:sorted(evidence) for cell,evidence in aliases.items() if len(evidence)>1}
        row['coverage_count']=int(row['actual_supported_correct'])
        row['references_are_not_independent_votes']=True
    return result

def summarize(tasks,regression,complete):
    totals={};detection={};per_source={};comparisons={}
    for reader in run.READERS:
        for scope in ('positive','null','wrong_key_PM'):
            rows=[t for name,t in tasks.items() if (t['positive'] if scope=='positive' else not t['positive'] if scope=='null' else name.split('/')[1]=='PM' and name.endswith('/key01'))]
            counts=[t['readers'][reader]['counts'] for t in rows]
            totals[reader+'/'+scope]={k:sum(x[k] for x in counts) for k in counts[0]}
            detection[reader+'/'+scope]=dict(planned=len(rows),statuses=dict(Counter(t['readers'][reader]['detection_status'] for t in rows)))
    for task,t in tasks.items():
        if t['positive']:comparisons[task]=paired(t['readers']['RANK3'],t['readers']['RANK3R'])
    for host in run.SOURCES:
        h=host['id'];rows=[t for name,t in tasks.items() if name.startswith(h+'/') and t['positive']]
        per_source[h]=dict(planned=64,positive_tasks=2,readers={r:dict(
            supported_correct=sum(t['readers'][r]['counts']['actual_supported_correct'] for t in rows),
            detect=sum(t['readers'][r]['detection_status']=='DETECT' for t in rows)) for r in run.READERS})
    positive,null=totals['RANK3R/positive'],totals['RANK3R/null']
    technical=[t for t,row in tasks.items() if any(v['technical_detail_slots'] or v['detection_status']=='TECHNICAL_UNRESOLVED' for v in row['readers'].values())]
    complete=bool(complete and not technical)
    checks=dict(positive_detect_at_least6=detection['RANK3R/positive']['statuses'].get('DETECT',0)>=6,
        each_source_detect_at_least1=all(p['readers']['RANK3R']['detect']>=1 for p in per_source.values()),
        supported_correct_at_least192=positive['actual_supported_correct']>=192,
        each_source_supported_at_least32=all(p['readers']['RANK3R']['supported_correct']>=32 for p in per_source.values()),
        old384_zero_loss=regression.get('preserved_supported_correct')==384 and regression.get('lost_supported')==0,
        old60_null_detect_zero=regression.get('null_detect')==0,new40_null_detect_zero=detection['RANK3R/null']['statuses'].get('DETECT',0)==0,
        literal_errors_zero=positive['literal_errors']+null['literal_errors']==0,
        false_attribution_zero=positive['false_attribution']+null['false_attribution']==0,
        unsupported_outputs_zero=positive['unsupported_positive_outputs']==0 and null['emitted']==0,
        required_observations_complete=complete)
    observed_failure=not all(checks[k] for k in ('old384_zero_loss','old60_null_detect_zero','new40_null_detect_zero','literal_errors_zero','false_attribution_zero','unsupported_outputs_zero'))
    method='TECHNICAL_UNRESOLVED' if not complete else 'M2_METHOD_CRITERIA_MET' if all(checks.values()) else 'METHOD_NOT_SUPPORTED'
    return dict(method_status=method,observed_method_violation=observed_failure,checks=checks,totals=totals,detection=detection,per_source=per_source,
        comparisons_to_original=comparisons,paired_gain=positive['actual_supported_correct']-totals['RANK3/positive']['actual_supported_correct'],
        technical_tasks=technical,next_candidate_allowed=False,domain_shrink_retry=False)

def evaluate(out,journal):
    out=Path(out);marker=run.read(out/'blind_complete.json')
    if marker.get('status')!='READ_BACK' or marker.get('reader_decisions')!=96:raise ValueError('all96 blind decisions must be read back before truth')
    run.dump(out/'post_truth_access.json',dict(opened_unix=time.time(),receiver_truth_access='after_all96_blind_decisions_read_back'))
    protocol=run.read(out/'protocol.json')['config'];bits=protocol['writer']['bits']
    if len(bits)!=32 or any(type(v) is not int or v not in (0,1) for v in bits):raise ValueError('fixed32 payload truth required')
    tasks={}
    for task in run.tasks():
        h,arm,fixture,key=task.split('/');host=next(x for x in protocol['sources'] if x['id']==h)
        frames=run.source_frames(host,fixture);positive=arm=='PM' and key=='key00'
        decision=run.read(out/'blind'/task/'decision.json');readers={}
        for reader in run.READERS:
            raw=decision['readers'][reader]['aggregate'];value=assess(raw,bits,positive,frames)
            value['detection_status']=decision['readers'][reader]['selection']['status']
            value['technical_detail_slots']=[i for i,row in enumerate(raw['slots']) if row.get('missing_refs') or any(s in row['status'] for s in ('TECHNICAL','UNRESOLVED','UNSUPPORTED','NOT_ENTERED'))]
            readers[reader]=value
        tasks[task]=dict(positive=positive,source=h,fixture=fixture,actual_source_frames=frames,readers=readers)
    quality={slot:run.read(out/'quality'/(slot+'.json')) for slot in run.quality_specs()}
    regression=run.read(out/'old72_regression.json')
    complete=all(row['status']=='RETURNED' for slot,row in journal.data.items() if slot!='post')
    return dict(status='POST_RECORDED',denominators=run.denominators(),tasks=tasks,quality=quality,
        historical_quality=run.read(out/'historical_quality.json'),old72_regression=regression,
        **summarize(tasks,regression,complete),quality_pass=None,population_FPR=None,source_authentication=False,
        exact_seam_certified=False,bundles=[],claim_ceiling='Fixed r8 left/right complete replay on four existing sources; no new independent source confirmation, no arbitrary repeats or speed changes',
        stop_interpretation='null DETECT fails candidate even with zero outputs; no later physical work or search; saved diagnostics only')
