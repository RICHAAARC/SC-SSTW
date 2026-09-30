"""CPU-only saved-window development diagnostic; raw path evidence precedes truth joins."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
from main.tube_state import video_temporal_sync_path as method
from main.tube_state import video_temporal_sync_multi as windows

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent / 'configs/video_temporal_sync_path_v1.json'
CONFIG_SHA256 = '38a39b2f96f63a807dd0d8e48cf550912aad33f3dc888a66480bb513176793e1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    if path.suffix == '.gz':
        data = (json.dumps(value,separators=(',', ':'),allow_nan=False)+'\n').encode()
        with temp.open('wb') as f:
            f.write(gzip.compress(data,compresslevel=6,mtime=0)); f.flush(); os.fsync(f.fileno())
    else:
        with temp.open('w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def load_config():
    if sha(CONFIG) != CONFIG_SHA256:
        raise ValueError('pre-score mechanism config bytes changed')
    return json.loads(CONFIG.read_text())


def roster(cfg):
    return [dict(id=f'condition_{i:03d}', arm=arm, view=view, key_id=k, length=v['length'])
            for i, (arm, view, v, k) in enumerate((a, vn, vv, k) for a in cfg['arms']
                 for vn, vv in cfg['views'].items() for k in cfg['keys'])]


def validate_artifact(data, length, g, key):
    expected_keys = {'schema','phase_g','received_frame_count','used_received_frame_interval','public_key_sha256',
        'public_layout_sha256','nominal_stride4','grid_meaning','first_latent_excluded','primary_regular_count','window_count',
        'payload_bit_count','pilot_coefficient_count','transform','statistic_reduction','time_dependent_payload','windows'}
    if set(data) != expected_keys:
        raise ValueError('unexpected public window schema fields')
    coords, pn = method.carrier.pilot_layout(key)
    layout = method.carrier.digest(dict(payload_coordinates=method.carrier.payload_coordinates(key),pilot_coordinates=coords,pilot_pn=pn))
    R = method.carrier.candidates(length)[0]['R']
    n = windows.window_row_count(length, g)
    expected = dict(schema=windows.WINDOW_SCHEMA,phase_g=g,received_frame_count=length,
        used_received_frame_interval=list(method.carrier.phase_slice(length,g)),public_key_sha256=hashlib.sha256(key.encode()).hexdigest(),
        public_layout_sha256=layout,nominal_stride4=4,grid_meaning='nominal newly supplied RGB frames, not VAE receptive field',
        first_latent_excluded=True,primary_regular_count=R,window_count=n,payload_bit_count=32,pilot_coefficient_count=64,
        transform='fft2_ortho_real_float32',statistic_reduction='float64',time_dependent_payload='disabled_unverified')
    if any(data.get(k) != v for k,v in expected.items()) or len(data['windows']) != n:
        raise ValueError('public window metadata/length/layout mismatch')
    for j, row in enumerate(data['windows'],1):
        if set(row) != {'phase_g','observed_regular_j','nominal_stride4','nominal_received_new_frame_interval','pilot_fft','payload_stats'}:
            raise ValueError('unexpected window fields')
        if (row['phase_g'],row['observed_regular_j'],row['nominal_stride4'],row['nominal_received_new_frame_interval']) != (g,j,4,[g+4*j-3,g+4*j+1]):
            raise ValueError('window coordinate/grid mismatch')
        if len(row['pilot_fft']) != 64 or any(type(x) not in (int,float) or not math.isfinite(x) for x in row['pilot_fft']):
            raise ValueError('nonfinite/wrong-size pilot coefficients')
        if len(row['payload_stats']) != 32:
            raise ValueError('32 public payload statistics required')
        for s in row['payload_stats']:
            if set(s) != {'sum','sumsq','positive_count','negative_count','zero_count','support_count','first_bit'}:
                raise ValueError('unexpected payload statistic fields')
            counts = [s[k] for k in ('positive_count','negative_count','zero_count','support_count')]
            if any(type(x) is not int or x<0 for x in counts) or sum(counts[:3]) != 30 or counts[3] != 30:
                raise ValueError('invalid payload statistic counts')
            if type(s['first_bit']) is not int or s['first_bit'] not in (0,1) or not all(math.isfinite(s[k]) for k in ('sum','sumsq')) or s['sumsq']<0:
                raise ValueError('invalid payload soft statistic/first bit')
    return [row['pilot_fft'] for row in data['windows'][:R]]


def read_input(source, source_result, item, g, key):
    slot = f"{item['arm']}/{item['view']}/{g}/{item['key_id']}"
    path = Path(source) / item['arm'] / f"{item['view']}.phase{g}.{item['key_id']}.windows.json"
    row = dict(slot=slot,path=str(path),status='FAILED',expected_sha256=None,sha256=None,primary_rows=45 if item['length']==181 else 31)
    try:
        ref = source_result['windows'][slot]
        row['expected_sha256'] = ref['sha256']
        if ref['status'] != 'SAVED' or ref['schema'] != windows.WINDOW_SCHEMA or ref['row_count'] != windows.window_row_count(item['length'],g):
            raise ValueError('saved input reference is not complete')
        row['sha256'] = sha(path)
        if row['sha256'] != ref['sha256']:
            raise ValueError('saved window hash mismatch')
        data = json.loads(path.read_text())
        pilot = validate_artifact(data,item['length'],g,key)
        row.update(status='VERIFIED',bytes=path.stat().st_size,window_rows=len(data['windows']))
        return row,pilot
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row,None


def legacy_baseline(pilot, length, key, analysis):
    rows = []
    for candidate in method.carrier.candidates(length):
        if candidate['g'] in pilot:
            rows.append(method.carrier.score_candidate({'pilot_fft':pilot[candidate['g']]},key,candidate))
        else:
            rows.append(dict(**candidate,status='FAILED',score=None,error='required verified phase missing'))
    decision = method.carrier.decide(rows,length)
    zero = {r['b']:r for r in analysis['catalog'] if r['event_type']=='ZERO_EDIT'}
    errors = [abs(r['score']-zero[r['b']]['score']) for r in rows if r['score'] is not None and zero[r['b']]['score'] is not None]
    return dict(candidates=rows,decision=decision,zero_edit_score_max_abs_error=max(errors,default=None),
                threshold_scope='Old global diagnostic only; not a path threshold')


def evaluate(output, result, cfg):
    """Only called after every raw file has been committed. Reopen raw evidence."""
    before = {cid:sha(Path(output)/r['raw_path']) for cid,r in result['conditions'].items() if r.get('raw_path')}
    if before != {cid:r['sha256'] for cid,r in result['conditions'].items() if r.get('raw_path')}:
        raise RuntimeError('raw evidence hash differs from committed index before posthoc')
    reports = []
    for item in roster(cfg):
        ref = result['conditions'][item['id']]
        raw = read_json(Path(output)/ref['raw_path']) if ref.get('raw_path') else None
        complete = raw is not None and raw['summary']['status']=='COMPLETE'
        payload = None
        info = dict(condition_id=item['id'],arm=item['arm'],view=item['view'],key_id=item['key_id'],
                    known_input_edit='NONE; only full/crop views of saved source',accepted_payload=False,
                    interpretation='Development posthoc; a canonical event is a fitted hypothesis, not an observed edit')
        if complete:
            rows = {r['id']:r for r in raw['catalog']}
            canonical = rows[raw['summary']['canonical_path_id']]
            true_b = cfg['views'][item['view']]['posthoc_start']
            true_id = method.path_id(true_b,'ZERO_EDIT',None)
            true = rows[true_id]
            info.update(summary=raw['summary'],canonical_b=canonical['b'],canonical_g=canonical['g'],
                        canonical_taus=list(method.path_taus(canonical)),posthoc_true_b=true_b,
                        true_zero_edit_path_id=true_id,true_zero_edit_score=true['score'],
                        true_zero_edit_in_top= true_id in raw['summary']['top_path_ids'],
                        canonical_event_on_known_unedited_input=canonical['event_type']!='ZERO_EDIT',
                        canonical_start_matches_truth=canonical['b']==true_b,
                        path_minus_true_zero_edit=raw['summary']['max_score']-true['score'])
            slot=f"{item['arm']}/{item['view']}/{canonical['g']}/{item['key_id']}"
            artifact=result['inputs'][slot]
            try:
                if sha(artifact['path']) != artifact['sha256']:
                    raise ValueError('posthoc input hash changed')
                data=json.loads(Path(artifact['path']).read_text())
                R=method.carrier.candidates(item['length'])[0]['R']
                payload=windows.aggregate_ordered_windows(data['windows'][:R])['decoded_bits']
                info['payload_diagnostic']='Same saved primary R windows in received order; no tau-based selection/remapping'
            except Exception as exc:
                info['payload_error']=f'{type(exc).__name__}: {exc}'
        for target,value in cfg['targets'].items():
            eid=f"{item['id']}/{target}"
            bits=method.carrier.message_bits(value)
            result['evaluations'][eid]=dict(status='EVALUATED' if complete and payload is not None else 'MISSING',
                accepted_payload=False,bit_errors=None,exact_bits=None,
                raw_phase_bit_errors=sum(x!=y for x,y in zip(payload,bits)) if payload is not None else None,
                raw_phase_exact=payload==bits if payload is not None else None,
                payload_diagnostic_only=True,condition_id=item['id'],target_id=target)
        reports.append(info)
    after = {cid:sha(Path(output)/r['raw_path']) for cid,r in result['conditions'].items() if r.get('raw_path')}
    if before != after:
        raise RuntimeError('raw blind evidence changed during posthoc evaluation')
    dump(Path(output)/'posthoc.json',dict(raw_hashes_unchanged=True,raw_sha256=after,conditions=reports,evaluations=result['evaluations']))
    return after


def execute(source, output, cfg):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    items=roster(cfg)
    result=dict(status='RUNNING',stage='VERIFY_INPUTS_AND_RAW_PATHS',fixed_denominator=cfg['fixed_denominator'],
        conditions={x['id']:dict(status='PENDING',raw_path=None) for x in items},
        evaluations={f"{x['id']}/{target}":dict(status='PENDING',accepted_payload=False,bit_errors=None,exact_bits=None) for x in items for target in cfg['targets']},
        inputs={},failures=[],source_result_sha256=None,source_run=cfg['source_run'],score_status='UNCALIBRATED_DIAGNOSTIC',
        accepted_payload=False,calls=dict(model=0,gpu=0,vae=0,fft=0,media_read=0,media_write=0),
        evidence='Saved already-observed run, CPU development diagnostic, not independent confirmation or real edit validation')
    for item in items:
        for g in method.carrier.phases(item['length']):
            slot=f"{item['arm']}/{item['view']}/{g}/{item['key_id']}"
            result['inputs'][slot]=dict(slot=slot,status='PENDING')
    dump(output/'result.json',result)
    try:
        source_result=json.loads((Path(source)/'result.json').read_text())
        result['source_result_sha256']=sha(Path(source)/'result.json')
        if result['source_result_sha256']!=cfg['source_result_sha256']:
            raise ValueError('source result hash differs from pre-score config')
        if set(source_result['windows'])!=set(result['inputs']):
            raise ValueError('source window reference denominator differs')
    except Exception as exc:
        source_result={}
        result['failures'].append(dict(stage='SOURCE_RESULT',error=f'{type(exc).__name__}: {exc}'))
    for item in items:
        pilot={}
        for g in method.carrier.phases(item['length']):
            row,X=read_input(source,source_result,item,g,cfg['keys'][item['key_id']])
            result['inputs'][row['slot']]=row
            if X is not None:pilot[g]=X
        try:
            raw=method.analyze(pilot,item['length'],cfg['keys'][item['key_id']])
            raw['legacy_global_baseline']=legacy_baseline(pilot,item['length'],cfg['keys'][item['key_id']],raw)
            raw['input_sha256']={str(g):result['inputs'][f"{item['arm']}/{item['view']}/{g}/{item['key_id']}"].get('sha256') for g in method.carrier.phases(item['length'])}
            rawpath=Path('raw')/(item['id']+'.json.gz')
            dump(output/rawpath,raw)
            result['conditions'][item['id']]=dict(status=raw['summary']['status'],raw_path=str(rawpath),sha256=sha(output/rawpath),counts=raw['counts'])
        except Exception as exc:
            # Retain complete parameter catalog and classes even on engineering failure.
            rows=method.catalog(item['length']);classes=method.equivalence_classes(rows)
            for row in rows:
                if row['structurally_valid']:row['status']='FAILED'
            for cls in classes:cls['status']='FAILED'
            counts=dict(catalog=len(rows),structurally_excluded=sum(not r['structurally_valid'] for r in rows),
                        scorable=sum(r['structurally_valid'] for r in rows),scored=0,zero_edit=sum(r['event_type']=='ZERO_EDIT' for r in rows),equivalence_classes=len(classes))
            raw=dict(summary=dict(status='INCOMPLETE',score_status='UNCALIBRATED_DIAGNOSTIC',accepted_payload=False),
                     catalog=rows,equivalence_classes=classes,dp=None,counts=counts,error=f'{type(exc).__name__}: {exc}')
            rawpath=Path('raw')/(item['id']+'.json.gz');dump(output/rawpath,raw)
            result['conditions'][item['id']]=dict(status='INCOMPLETE',raw_path=str(rawpath),sha256=sha(output/rawpath),counts=counts,error=raw['error'])
        dump(output/'result.json',result)
    result['stage']='RAW_COMPLETE_BEFORE_TRUTH'
    result['raw_sha256_before_truth']={k:v['sha256'] for k,v in result['conditions'].items()}
    dump(output/'result.json',result)
    result['raw_sha256_after_truth']=evaluate(output,result,cfg)
    result['counts']=dict(inputs_verified=sum(r['status']=='VERIFIED' for r in result['inputs'].values()),
        conditions_complete=sum(r['status']=='COMPLETE' for r in result['conditions'].values()),
        evaluations_complete=sum(r['status']=='EVALUATED' for r in result['evaluations'].values()),
        catalog_paths=sum(r.get('counts',{}).get('catalog',0) for r in result['conditions'].values()),
        scored_paths=sum(r.get('counts',{}).get('scored',0) for r in result['conditions'].values()),
        scorable_paths=sum(r.get('counts',{}).get('scorable',0) for r in result['conditions'].values()),
        zero_edit_paths=sum(r.get('counts',{}).get('zero_edit',0) for r in result['conditions'].values()),
        structurally_excluded=sum(r.get('counts',{}).get('structurally_excluded',0) for r in result['conditions'].values()),
        equivalence_classes=sum(r.get('counts',{}).get('equivalence_classes',0) for r in result['conditions'].values()))
    expected=dict(inputs_verified=102,conditions_complete=30,evaluations_complete=60,catalog_paths=78126,
                  scored_paths=77862,scorable_paths=77862,zero_edit_paths=1278,structurally_excluded=264)
    result['status']='EXECUTION_COMPLETE' if all(result['counts'][k]==v for k,v in expected.items()) else 'INCOMPLETE'
    result['stage']='FINISHED'
    dump(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path)
    args=parser.parse_args();cfg=load_config()
    result=execute(args.source or cfg['source_directory'],args.output,cfg)
    source_files=['main/tube_state/video_temporal_sync_path.py','main/tube_state/video_temporal_sync_bridge.py',
                  'main/tube_state/video_temporal_sync_multi.py','experiments/wan_state_clock/video_temporal_sync_path_run.py',
                  'experiments/wan_state_clock/configs/video_temporal_sync_path_v1.json']
    dump(args.output/'source_receipt.json',dict(base_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         files={p:sha(ROOT/p) for p in source_files},config_pre_score_sha256=CONFIG_SHA256,
         execution='CPU saved-window development diagnostic only',new_model_or_vae_execution=False))
    print(json.dumps(dict(status=result['status'],counts=result['counts']),indent=2))
    return 0 if result['status']=='EXECUTION_COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())
