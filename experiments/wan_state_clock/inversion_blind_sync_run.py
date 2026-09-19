"""CPU saved-tensor blind search. Source metadata is read only after all searches persist."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import torch
from main.tube_state import inversion_blind_sync as method,inversion_state as state

SOURCE_RUN='inversion_crop_observation_20260918T171553283563Z'
SOURCE_COMMIT='8ad796e1bfa5d3f5dc697c3f14f62df0f24f0d29'
CASES=('holdout_p0_s0','holdout_p1_s0')
IDS=tuple(f'r{i:02d}' for i in range(9))
MANIFEST=Path(__file__).parent/'configs/inversion_blind_sync.json'
PUBLIC_CONFIG=MANIFEST.with_name('inversion_state.json')


def load(path):return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def missing():
    return dict(status='MISSING_OR_FAILED',candidate_denominator=28,
        candidates=[dict(shift=s,message=m,status='NOT_SCORED',score=None) for s in method.SHIFTS for m in (0,1)])


def posthoc(search,association):
    """Truth/time join occurs after immutable blind output; never reranks."""
    start=association['source_start'];arm=association['source_arm']
    if not isinstance(start,int) or not 0<=start<=52 or arm not in ('OFF','A','B'):raise ValueError('invalid reporting-only source metadata')
    true=None if arm=='OFF' else ('A','B').index(arm)
    floor=start//4;ceil=(start+3)//4
    def evidence(shift):
        return dict(shift=shift,candidates=[dict(r) for r in search['candidates'] if r['shift']==shift],
            neighboring_candidates=[dict(r) for r in search['candidates'] if r['shift'] in (shift-1,shift+1)])
    report=dict(source_start_reporting_only=start,truth_reporting_only=true,source_arm_reporting_only=arm,
        phase_aligned=start%4==0,nominal_floor=evidence(floor),nominal_ceil=evidence(ceil) if ceil!=floor else None,
        phase_claim='no within-latent frame phase estimate; nonaligned floor and ceil are separate diagnostics')
    ranking=search['ranking']
    if true is not None:
        profiles=ranking['message_profiles']
        report.update(unique_message_correct=ranking['selected_message']==true,
            true_wrong_message_margin=profiles[true]['max_score']-profiles[1-true]['max_score'])
        if start%4==0:
            report.update(unique_aligned_shift_correct=ranking['selected_shift']==floor,
                unique_aligned_pair_correct=ranking['selected_pair']=={'shift':floor,'message':true})
    return report


def summarize(records):
    result=dict(clip_denominator=18,candidate_denominator=504,marked_denominator=12,off_denominator=6,
        complete_blind_clips=sum(r['blind']['status']=='COMPLETE' for r in records),
        complete_posthoc_marked=0,unique_correct_messages=0,aligned_marked_denominator=8,
        observed_aligned_marked=0,unique_correct_aligned_shifts=0,unique_correct_aligned_pairs=0,
        nonaligned_marked_denominator=4,per_clip={},scientific_pass=None)
    for row in records:
        entry=dict(blind_status=row['blind']['status'],posthoc_status=row['posthoc']['status'])
        if row['blind']['status']=='COMPLETE':
            r=row['blind']['ranking'];entry.update(selected_pair=r['selected_pair'],selected_message=r['selected_message'],
                top_ties=r['top_ties'],top_runner_gap=r['top_runner_gap'],message_profiles=r['message_profiles'],message_margin=r['message_margin'])
            entry.update(offset_profiles=r['offset_profiles'],offset_top_ties=r['offset_top_ties'],
                different_offset_top_runner_gap=r['different_offset_top_runner_gap'])
        if row['posthoc']['status']=='COMPLETE':
            report=row['posthoc']['report'];entry['report']=report
            if report['truth_reporting_only'] is not None:
                result['complete_posthoc_marked']+=1;result['unique_correct_messages']+=report['unique_message_correct']
                if report['phase_aligned']:
                    result['observed_aligned_marked']+=1
                    result['unique_correct_aligned_shifts']+=report['unique_aligned_shift_correct']
                    result['unique_correct_aligned_pairs']+=report['unique_aligned_pair_correct']
        result['per_clip'][row['case']+'/'+row['receiver_id']]=entry
    result['unmeasured_marked']=12-result['complete_posthoc_marked']
    result['unmeasured_aligned_marked']=8-result['observed_aligned_marked']
    result['claim']='ranking only; OFF has no detection count; nominal latent shift is not exact RGB-frame synchronization; same-source clips correlated'
    return result


def run(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if source.name!=SOURCE_RUN:raise ValueError('fixed crop source run required')
    if output==source or source in output.parents:raise ValueError('independent output directory required')
    output.mkdir(parents=True,exist_ok=False)
    protocol=load(MANIFEST)
    if tuple(protocol['nominal_shifts'])!=method.SHIFTS or tuple(protocol['local_indices'])!=method.LOCAL or protocol['tie_epsilon']!=method.EPS:
        raise ValueError('manifest differs from fixed search protocol')
    dump(output/'protocol.json',protocol)
    book=state.codebook(load(PUBLIC_CONFIG)['base_config']['key_utf8'].encode())
    torch.save(book,output/'public_book.pt')
    records=[]
    # No source metadata or attacker manifest reads occur in this entire phase.
    for case in CASES:
        for rid in IDS:
            path=source/case/'receiver'/f'{rid}_recovered.pt'
            record=dict(case=case,receiver_id=rid,blind=missing(),posthoc={'status':'NOT_RUN'})
            try:
                record['input_sha256']=sha(path)
                recovered=torch.load(path,map_location='cpu',weights_only=True)
                record['blind']=method.search(recovered,book);del recovered
            except Exception as exc:record['blind']['error']=repr(exc)
            target=output/'blind'/case/(rid+'.json')
            dump(target,record['blind']);record['blind_sha256']=sha(target)
            records.append(record)
            print(case,rid,record['blind']['status'],flush=True)
    dump(output/'search_phase_complete.json',dict(clip_denominator=18,
        records=[{k:r[k] for k in ('case','receiver_id','blind_sha256')} for r in records],
        claim='all blind outputs persisted before metadata join'))
    # Only now load old result/attacker metadata, for integrity and reporting.
    metadata_hashes={}
    try:
        result_path=source/'result.json';original=load(result_path);metadata_hashes['result.json']=sha(result_path)
    except Exception as exc:original={};metadata_hashes['result_error']=repr(exc)
    for case in CASES:
        try:
            path=source/case/'attacker_manifest.json';attacker=load(path);metadata_hashes[case+'/attacker_manifest.json']=sha(path)
            associations={a['receiver_id']:a for a in attacker['associations']}
        except Exception as exc:associations={};metadata_hashes[case+'/attacker_error']=repr(exc)
        for record in [r for r in records if r['case']==case]:
            if record['blind']['status']!='COMPLETE':record['posthoc']={'status':'MISSING_BLIND_RESULT'};continue
            try:
                rid=record['receiver_id']
                if original['cases'][case].get('source_commit')!=SOURCE_COMMIT:raise ValueError('source crop case commit differs from frozen run provenance')
                expected=original['cases'][case]['clips'][rid]['receiver']['recovered_sha256']
                if record['input_sha256']!=expected:raise ValueError('saved recovered tensor differs from same-run result hash')
                record['posthoc']=dict(status='COMPLETE',input_hash_matches_same_run=True,
                    report=posthoc(record['blind'],associations[rid]))
            except Exception as exc:record['posthoc']=dict(status='MISSING_OR_FAILED',error=repr(exc))
            if sha(output/'blind'/case/(record['receiver_id']+'.json'))!=record['blind_sha256']:raise RuntimeError('posthoc modified blind output')
    result=dict(status='EXECUTION_COMPLETE' if all(r['blind']['status']=='COMPLETE' and r['posthoc']['status']=='COMPLETE' for r in records) else 'WITH_RETAINED_FAILURES',
        source_run=SOURCE_RUN,clip_denominator=18,candidate_denominator=504,records=records,summary=summarize(records),
        source_commit=subprocess.check_output(['git','-C',str(Path(__file__).parents[2]),'rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','-C',str(Path(__file__).parents[2]),'status','--porcelain'],text=True).strip()),
        fixed_model_calls=0,actual_model_calls=0,metadata_hashes=metadata_hashes,
        source_sha256={str(p):sha(p) for p in (Path(__file__),Path(method.__file__),Path(state.__file__),MANIFEST,PUBLIC_CONFIG)},
        environment=dict(python=platform.python_version(),torch=str(torch.__version__)),scientific_pass=None)
    dump(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);result=run(args.input,args.output)
    print(json.dumps(result['summary'],indent=2))
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
