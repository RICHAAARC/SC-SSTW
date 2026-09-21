"""Fixed existing-run CPU-only audit. Reads JSON, never media/tensors/models."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
from . import flow_tube_sync_diagnostics as analysis

SOURCE_RUN='flow_tube_detection_20260920T113113319009Z'
SOURCE_COMMIT='f561fd58f01c8cc44a05885c0eb9ff42017465c7'
CASES=('eval01','eval02')
ARMS=('OFF','LAST_A','LAST_B')
VIEWS=('full181','crop0','crop4','crop5','delete90','speed125','resaved')

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def read_json(path,inputs,expected=None):
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    inputs.append(dict(path=str(path),bytes=len(raw),sha256=sha,expected_sha256=expected,hash_matches=None if expected is None else sha==expected))
    if expected is not None and sha!=expected:raise ValueError('input hash mismatch: '+str(path))
    return json.loads(raw)

def run(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:raise ValueError('source/output must be separate nonnested directories')
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();inputs=[];rows=[]
    result=dict(status='RUNNING',source_run=SOURCE_RUN,expected_source_commit=SOURCE_COMMIT,view_denominator=42,source_denominator=6,
                marked_source_denominator=4,OFF_source_denominator=2,windows_per_view=11,window_denominator=462,
                calibration='nine calibration sources excluded; original evaluation decisions/thresholds copied unchanged',
                rows=rows,inputs=inputs,failures=[],source_path=str(source),scientific_new_samples=0,original_sequences=[],
                runtime_calls=dict(transformer=0,vae_encode=0,vae_decode=0,mp4_save=0,backward=0))
    try:result['audit_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:result['audit_source_commit']=None
    for case in CASES:
        original=None;key=None;case_error=None
        try:
            original=read_json(source/case/'result.json',inputs)
            if original.get('source_commit')!=SOURCE_COMMIT:raise ValueError('wrong original model-run source commit')
            threshold=read_json(source/'threshold.json',inputs,original['threshold_sha256'])
            result['original_threshold']=threshold
            config=read_json(source/case/'config.json',inputs,original['file_sha256']['config.json'])
            key=config['key_utf8'].encode()
        except Exception as exc:
            case_error=repr(exc);result['failures'].append(dict(case=case,stage='case_inputs',error=case_error))
        for arm in ARMS:
            result['original_sequences'].append(dict(case=case,arm=arm,original_sequence=None if original is None else original.get('sources',{}).get(arm,{}).get('sequence'),
                                                      meaning='one of six original source aggregates; separate from 42 views, no new independent samples'))
            for view_name in VIEWS:
                row=dict(case=case,arm=arm,view=view_name,status='MISSING_OR_INVALID',window_denominator=11)
                rows.append(row);view={} if original is None else original.get('sources',{}).get(arm,{}).get('views',{}).get(view_name,{})
                row['original_threshold_sha256']=None if original is None else original.get('threshold_sha256')
                try:
                    # Geometry available in result survives missing full receiver detail, but no invented candidates.
                    row['diagnostics']=analysis.analyze(view_name,view)
                    if case_error is not None:raise ValueError(case_error)
                    relative='views/'+arm+'/'+view_name+'/blind.json'
                    expected=original['file_sha256'].get(relative)
                    if expected is None:raise ValueError('no recorded SHA for full receiver JSON')
                    blind=read_json(source/case/relative,inputs,expected)
                    row['diagnostics']=analysis.analyze(view_name,view,blind,key)
                    row['status']='CHECKED'
                    checks=row['diagnostics']
                    consistency=[checks['mapping_matches_saved'],checks['best_geometry_matches_saved'],checks['message_scores_match_saved'],checks['best_statistic_matches_saved'],checks['candidate_diagnostics']['original_protocol_paths_match'],checks['candidate_diagnostics']['recorded_counts_consistent']]
                    consistency += [w['allocation_matches_record'] for m in checks['candidate_diagnostics']['modes'].values() for w in m['selected_path_windows']]
                    consistency += [m['best_observer_matches_class'] is True and m['best_score_class_residual']<=1e-12 for m in checks['candidate_diagnostics']['modes'].values() if m['original_ranking'].get('best') is not None]
                    if checks['observer_diagnostics'] is not None:consistency += [r['consistent_with_saved'] for r in checks['observer_diagnostics']['messages']]
                    if any(v is not True for v in consistency):row['status']='WITH_INCONSISTENCIES'
                    blind=None
                except Exception as exc:
                    row['error']=repr(exc);result['failures'].append(dict(case=case,arm=arm,view=view_name,error=repr(exc)))
                dump(output/'result.json',result)
    result['by_view']={}
    for view in VIEWS:
        subset=[r for r in rows if r['view']==view]
        result['by_view'][view]=dict(view_denominator=6,checked=sum(r['status']=='CHECKED' for r in subset),
            marked_denominator=4,OFF_denominator=2,
            summaries=[dict(case=r['case'],arm=r['arm'],status=r['status'],
                original_decision=r.get('diagnostics',{}).get('original_decision'),original_attribution=r.get('diagnostics',{}).get('original_attribution'),
                valid_windows=r.get('diagnostics',{}).get('selected_geometry',{}).get('valid_windows'),
                equal_nominal_support_windows=r.get('diagnostics',{}).get('selected_geometry',{}).get('equal_nominal_support_windows'),
                nominal_fully_available_windows=r.get('diagnostics',{}).get('selected_geometry',{}).get('nominal_fully_available_windows'),exact_sync=None) for r in subset])
    result['status']='AUDIT_COMPLETE' if all(r['status']=='CHECKED' for r in rows) else 'WITH_RETAINED_MISSING_OR_INCONSISTENCIES'
    result['elapsed_seconds']=time.perf_counter()-started
    result['claim']='CPU re-analysis of one existing run, no new validation; nominal RGB support equality is not VAE equivalence, sync accuracy or exact edit localization'
    dump(output/'result.json',result);dump(output/'input_manifest.json',inputs)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.source,args.output)
    print(json.dumps({k:result[k] for k in ('status','source_denominator','view_denominator','window_denominator','by_view','claim')},indent=2))
