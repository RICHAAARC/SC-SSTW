"""Card77 read-only archived event-score arithmetic; no old model or detector run."""
import argparse,csv,io,json,zipfile
from decimal import Decimal
from collections import Counter,defaultdict
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,required=True,help='available SyncTube_results.zip')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUTPUT=args.output
OUTPUT.mkdir(parents=True,exist_ok=True)
counts=defaultdict(Counter)
groups=defaultdict(Counter)
gap_counts=Counter()
branch_counts=Counter()
geometry=defaultdict(Counter)
issues=[]
records=0
clip_rows=0
with zipfile.ZipFile(args.archive) as outer:
    member=next(n for n in outer.namelist() if n.endswith('packages/real_video_vae_latent_probe_shard_aggregated.zip'))
    with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
        prefix='real_video_vae_latent_probe_shard_aggregated/'
        thresholds=json.loads(inner.read(prefix+'thresholds/thresholds.json'))
        threshold={x['method_variant']:x['threshold_value'] for x in thresholds}
        for f in ['thresholds/thresholds.json','artifacts/runtime_config.json','tables/main_tpr_fpr_table.csv','tables/ablation_table.csv','tables/stage2_sync_gain_table.csv','reports/stage2_mechanism_audit_report.md']:
            p=OUTPUT/'source_records'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(inner.read(prefix+f))
        with inner.open(prefix+'records/event_scores.jsonl') as stream,(OUTPUT/'EVENT_SCORE_REPLAY.jsonl').open('w') as saved:
            for line in stream:
                obj=json.loads(line);records+=1
                method=obj['method_variant'];trace=obj['mechanism_trace'];score=obj['evidence_scores']
                before,after=trace.get('mean_projection_before'),trace.get('mean_projection_after')
                support=0. if before is None or after is None else max(0.,min(1.,float(after)-float(before)))
                weight=0. if method=='frame_prc' else .09
                bonus=round(support*weight,6)
                st=float(score['S_tubelet']);sf=float(score['S_final'])
                if abs(st)>=1 or abs(float(trace['S_payload_aligned']))>=1:clip_rows+=1
                unaligned=float(trace['S_payload_unaligned']);aligned=float(trace['S_payload_aligned'])
                gate=bool(trace['sync_rescue_applied']);gain=float(trace['S_payload_rescue_gain'])
                reproduced=round(st+(max(0.,gain) if gate else 0.),6)
                if reproduced!=sf:issues.append({'event':obj['event_id'],'issue':'fusion_mismatch'})
                if method=='tubelet_sync':
                    gap=round(aligned-float(trace['sync_candidate_score_penalized']),6)
                    gap_counts[(obj['sample_role'],gap)]+=1
                    branch_counts[(obj['attack_name'],trace.get('sync_alignment_mode'),trace.get('sync_estimated_scale'),trace.get('sync_search_score_rule'),trace.get('sync_search_space_size'))]+=1
                    if abs(Decimal(str(gap))-Decimal(str(bonus)))>Decimal('0.000001'):issues.append({'event':obj['event_id'],'issue':'aligned_minus_penalized_not_bonus','gap':gap,'bonus':bonus})
                # This is a saved-score counterfactual with the archived test
                # threshold frozen, NOT a rerun/recalibration or new baseline.
                no_trace_held_gate=round(sf-bonus,6)
                ua0=round(unaligned-bonus,6);al0=round(aligned-bonus,6)
                gate0=False
                if method=='tubelet_sync':
                    gate0=(gain>=trace['sync_confidence_min_payload_rescue_gain'] and al0>=trace['sync_confidence_min_aligned_payload_score'] and trace['sync_alignment_coverage_ratio']>=trace['sync_confidence_min_coverage_ratio'] and trace['sync_alignment_matched_count']>=trace['sync_confidence_min_matched_count'])
                no_trace_regated=round(ua0+(gain if gate0 else 0.),6)
                if method!='tubelet_sync':no_trace_regated=round(st-bonus,6)
                decision=sf>=threshold[method]
                row=dict(event_id=obj['event_id'],sample_id=obj['sample_id'],method=method,split=obj['split'],role=obj['sample_role'],attack=obj['attack_name'],threshold=threshold[method],saved_decision=obj['decision'],score=sf,S_tubelet=st,unaligned=unaligned,aligned=aligned,rescue_gain=gain,rescue_applied=gate,projection_before=before,projection_after=after,embedding_support=support,weight_assumed_from_archived_method_config=weight,bonus=bonus,aligned_penalized_gap=None if method!='tubelet_sync' else gap,without_trace_held_gate=no_trace_held_gate,without_trace_regated=no_trace_regated,regated_rescue_applied=gate0,source_backend=trace.get('vae_backend_name'),reference_shape_source=trace.get('reference_latent_shape_source'))
                saved.write(json.dumps(row,ensure_ascii=False)+'\n')
                key=(method,obj['split'],obj['sample_role'])
                c=counts[key];c['n']+=1;c['with_before_after']+=before is not None and after is not None;c['bonus_nonzero']+=bonus!=0;c['saved_threshold_positive']+=decision;c['record_decision_positive']+=obj['decision'] is True;c['record_threshold_mismatch']+=obj['decision']!=decision;c['no_trace_held_gate_positive']+=no_trace_held_gate>=threshold[method];c['no_trace_regated_positive']+=no_trace_regated>=threshold[method];c['rescue_applied']+=gate;c['regated_rescue_applied']+=gate0
                if obj['split']=='test' and method=='tubelet_sync' and obj['sample_role']=='attacked_positive':
                    ge=geometry[obj['attack_name']];ge['n']+=1;ge['detected']+=decision
                    error=trace.get('sync_alignment_error')
                    if error is not None:ge['error_present']+=1;ge['exact']+=error==0;ge['absolute_error_sum']+=error
                if obj['split']=='test':
                    g=groups[(method,obj['sample_role'],obj['attack_name'])];g['n']+=1;g['historical_positive']+=decision;g['without_trace_held_gate_positive']+=no_trace_held_gate>=threshold[method];g['without_trace_regated_positive']+=no_trace_regated>=threshold[method]
result=dict(scope='old framewise saved score replay and writer-trace contribution at frozen old thresholds; not current blind baseline',records=records,thresholds=threshold,clip_boundary_rows=clip_rows,issues=issues,group_counts=[dict(method=k[0],split=k[1],role=k[2],**v) for k,v in counts.items()],test_attack_counts=[dict(method=k[0],role=k[1],attack=k[2],**v) for k,v in groups.items()],direct_aligned_penalized_gap_counts=[dict(role=k[0],gap=k[1],n=v) for k,v in gap_counts.items()],actual_search_branches=[dict(attack=k[0],alignment_mode=k[1],estimated_scale=k[2],search_rule=k[3],candidate_count=k[4],n=v) for k,v in branch_counts.items()],test_geometry=[dict(attack=k,**v) for k,v in geometry.items()],physical_calls=0)
(OUTPUT/'SAVED_LEGACY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['test_attack_counts','group_counts']},ensure_ascii=False,indent=2))
print('TEST_GROUPS',json.dumps([x for x in result['group_counts'] if x['split']=='test'],ensure_ascii=False))
