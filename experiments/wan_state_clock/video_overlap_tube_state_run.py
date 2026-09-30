"""Fixed CPU same-tensor development experiment; no model/media imports."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from main.tube_state import video_overlap_tube_state as method
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_overlap_tube_state_v1.json'
CONFIG_SHA='4c45eec080679a8e8c9324a3ea8d2cd37fafbd2c3aad1e8dd250eed8fc444462'
KEYS={'correct':'watermark','wrong':'watermark-wrong'}
MASKS=('none','block0','row16')
COMPLETE={'COMPLETE','NO_ENERGY'}
SOURCE_PATHS=(CONFIG,Path(__file__),ROOT/'main/tube_state/video_overlap_tube_state.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def atom(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if path.suffix=='.gz':data=gzip.compress(data,mtime=0)
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(data);tmp.replace(path)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def read(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)
def load_config():
    if sha(CONFIG)!=CONFIG_SHA:raise ValueError('frozen mechanism config identity mismatch')
    return json.loads(CONFIG.read_text())


def edit_rows(array,kind,p):
    """Physical tensor edit; p is a one-based source row, never a receiver input."""
    if kind=='NO_EDIT':return array.copy()
    if kind=='DELETE':return np.delete(array,p-1,axis=0)
    if kind in ('REPEAT','EXACT_COPY_INSERT'):row=array[p-1:p]
    elif kind=='NULL_INSERT':row=np.zeros_like(array[p-1:p])
    elif kind=='MIDPOINT_INSERT':row=.5*(array[p-1:p]+array[p:p+1])
    else:raise ValueError(kind)
    return np.concatenate((array[:p],row,array[p:]),axis=0)


def availability(length,name):
    mask=np.ones((length,4),dtype=bool)
    if name=='block0':mask[:,0]=False
    elif name=='row16':mask[15,:]=False
    elif name!='none':raise ValueError(name)
    return mask


def roster(config):
    physical=[];conditions=[];counter=[]
    for clip in config['stage_B_plan']['clips']:
        L=clip['length'];ps=(2,(L+1)//2,L-1)
        edits=[('NO_EDIT',None)]+[(kind,p) for kind in ('DELETE','REPEAT') for p in ps]
        for kind,p in edits:
            for ni,noise in enumerate(config['stage_B_plan']['noise_cases']):
                for writer in ('OFF','OVERLAP_PILOT'):
                    pid=f'{clip["id"]}/{kind}{p or 0}/noise{ni}/{writer}'
                    item=dict(id=pid,clip=clip,edit=kind,p=p,noise=noise,writer=writer)
                    physical.append(item)
                    for missing in MASKS:
                        for key in KEYS:conditions.append(dict(id=f'{pid}/{missing}/{key}',physical_id=pid,missing=missing,key=key,group='primary'))
        p=(L+1)//2
        for kind in ('EXACT_COPY_INSERT','NULL_INSERT','MIDPOINT_INSERT'):
            pid=f'{clip["id"]}/{kind}{p}/counterexample/OVERLAP_PILOT'
            physical.append(dict(id=pid,clip=clip,edit=kind,p=p,noise=dict(sigma_over_alpha=0,seed=0),writer='OVERLAP_PILOT'))
            for key in KEYS:counter.append(dict(id=f'{pid}/none/{key}',physical_id=pid,missing='none',key=key,group='counterexample'))
    assert len(conditions)==1680 and len(counter)==24 and len(physical)==292
    return physical,conditions+counter


def failed_raw(length,error):
    rows=method.catalog(length);valid=[i for i,r in enumerate(rows) if r['structurally_valid']]
    return dict(summary=dict(status='INCOMPLETE',reason='ENGINEERING_FAILURE',error=error,score_status='UNCALIBRATED_DIAGNOSTIC',
         accepted_payload=False,state_path_accepted=False,canonical_catalog_index=None,top_catalog_indices=[],top_class_indices=[]),
         received_length=length,valid_catalog_indices=valid,path_costs=[None]*len(valid),class_costs=None,projection=None,
         counts=dict(catalog=len(rows),scorable=len(valid),structurally_excluded=len(rows)-len(valid),scored=0))


def posthoc(raw,condition,physical,catalog):
    summary=raw['summary'];clip=physical['clip'];kind=physical['edit'];p=physical['p']
    source=np.arange(clip['start'],clip['start']+clip['length'])
    supported=kind in ('NO_EDIT','DELETE','REPEAT')
    truth=edit_rows(source,kind,p).tolist() if supported else None
    top=summary.get('top_catalog_indices',[]);canonical=summary.get('canonical_catalog_index')
    true_ids=[] if truth is None else [i for i,r in enumerate(catalog) if r['structurally_valid'] and r['taus']==truth]
    return dict(id=condition['id'],group=condition['group'],writer=physical['writer'],reader_key=condition['key'],
      clip=clip['id'],edit=kind,p=p,noise=physical['noise'],missing=condition['missing'],
      status='EVALUATED' if summary['status'] in COMPLETE else 'INCOMPLETE',model_status=summary['status'],reason=summary['reason'],
      true_tau=truth,true_catalog_indices=true_ids,truth_in_top=None if not supported else bool(set(true_ids)&set(top)),
      top_path_count=len(top),top_class_count=len(summary.get('top_class_indices',[])),
      canonical_event=None if canonical is None else catalog[canonical]['event_type'],
      canonical_tau=None if canonical is None else catalog[canonical]['taus'],
      any_top_zero_edit=summary.get('any_top_contains_zero_edit'),all_top_require_event=summary.get('all_top_require_event'),
      min_cost=summary.get('min_cost'),zero_edit_minus_best=summary.get('zero_edit_minus_best'),
      unique_finite_model=summary.get('unique_model_hypothesis',False),accepted_payload=False,state_path_accepted=False,
      interpretation='unsupported generating operator; finite family fit is not physical edit identification' if not supported else 'finite-model diagnostic only')


def summarize(evaluations):
    primary=[r for r in evaluations if r['group']=='primary']
    positive=[r for r in primary if r['writer']=='OVERLAP_PILOT' and r['reader_key']=='correct']
    groups=[]
    for noise in (0,.25,.5):
        for missing in MASKS:
            rows=[r for r in positive if r['noise']['sigma_over_alpha']==noise and r['missing']==missing]
            noedit=[r for r in rows if r['edit']=='NO_EDIT']
            groups.append(dict(sigma_over_alpha=noise,missing=missing,total=len(rows),evaluated=sum(r['status']=='EVALUATED' for r in rows),
              truth_in_top=sum(r['truth_in_top'] is True for r in rows),unique=sum(r['unique_finite_model'] for r in rows),
              noedit_total=len(noedit),noedit_canonical_event=sum(r['canonical_event'] not in (None,'ZERO_EDIT') for r in noedit),
              noedit_all_top_require_event=sum(r['all_top_require_event'] is True for r in noedit)))
    return dict(positive_groups=groups,primary_status=dict(Counter(r['model_status'] for r in primary)),
      counterexample_status=dict(Counter(r['model_status'] for r in evaluations if r['group']=='counterexample')),
      control_interpretation='OFF and wrong-key are uncalibrated fits; accepted=False by design is not FPR evidence')


def verify_cost_integrity(index):
    """Read persisted costs; this never calls extraction or inference."""
    counts=dict(primary_path_costs=0,counterexample_path_costs=0,catalog_slots=0,structural_exclusions=0)
    failures=[];shared={}
    def shared_read(ref):
        if ref['path'] not in shared:
            if sha(ref['path'])!=ref['sha256']:raise ValueError('shared artifact hash mismatch')
            shared[ref['path']]=read(ref['path'])
        return shared[ref['path']]
    for cid,entry in index['conditions'].items():
        try:
            ref=entry['raw']
            if sha(ref['path'])!=ref['sha256']:raise ValueError('raw hash mismatch')
            envelope=read(ref['path']);raw=envelope['inference'];stats=raw['counts']
            cat=shared_read(envelope['catalog'])['rows'];fam=shared_read(envelope['equivalence_family'])
            valid=[i for i,r in enumerate(cat) if r['structurally_valid']]
            counts['catalog_slots']+=len(cat);counts['structural_exclusions']+=len(cat)-len(valid)
            if raw['valid_catalog_indices']!=valid:raise ValueError('valid catalog index mismatch')
            if len(raw['path_costs'])!=len(valid):raise ValueError('path cost slot mismatch')
            if raw['summary']['status'] in COMPLETE:
                if stats['scored']!=len(valid) or not np.isfinite(raw['path_costs']).all():raise ValueError('missing/nonfinite path costs')
                if len(raw['class_costs'])!=len(fam['classes']) or not np.isfinite(raw['class_costs']).all():raise ValueError('missing/nonfinite class costs')
                counts['primary_path_costs' if entry['group']=='primary' else 'counterexample_path_costs']+=len(valid)
            else:failures.append(dict(id=cid,error='condition incomplete'))
        except Exception as exc:failures.append(dict(id=cid,error=f'{type(exc).__name__}: {exc}'))
    expected=dict(primary_path_costs=1195500,counterexample_path_costs=16146,catalog_slots=5216400,structural_exclusions=4004754)
    return dict(status='PASS' if counts==expected and not failures else 'INCOMPLETE',expected=expected,actual=counts,failures=failures,
                new_extractions=0,new_scores=0)


def run(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('new output directory required; never overwrite evidence')
    output.mkdir(parents=True,exist_ok=True);start=time.monotonic();config=load_config()
    source_before={str(p.relative_to(ROOT)):sha(p) for p in SOURCE_PATHS}
    physical,conditions=roster(config);physical_by_id={r['id']:r for r in physical}
    index=dict(schema='overlap-state-development-v1',status='RUNNING',stage='RAW',config_sha256=CONFIG_SHA,
      score_status='UNCALIBRATED_DIAGNOSTIC',no_real_model_or_media=True,source_files_before=source_before,
      expected=dict(primary=1680,counterexamples=24,physical_tensors=292,catalog=20250,valid=3049,excluded=17201),
      conditions={r['id']:dict(status='PENDING',group=r['group']) for r in conditions},physical={},catalogs={},families={},failures=[])
    atom(output/'result.json',index)
    catalogs={L:method.catalog(L) for L in method.PUBLIC.observed_lengths}
    for L,rows in catalogs.items():index['catalogs'][str(L)]=atom(output/'catalogs'/f'{L}.json.gz',dict(length=L,rows=rows))
    for L in method.PUBLIC.observed_lengths:
        for missing in MASKS:
            for kn,key in KEYS.items():
                fid=f'{L}/{missing}/{kn}'
                index['families'][fid]=atom(output/'families'/f'{L}_{missing}_{kn}.json.gz',method.family_receipt(key,L,availability(L,missing)))
    totals=dict(catalog=sum(map(len,catalogs.values())),valid=sum(r['structurally_valid'] for rows in catalogs.values() for r in rows))
    totals['excluded']=totals['catalog']-totals['valid']
    assert totals==dict(catalog=20250,valid=3049,excluded=17201)
    marked=method.synthesize(KEYS['correct']);regular=marked[0,:,1:].transpose(1,0,2,3)
    source_path=output/'actual_source_tensor.npz';np.savez_compressed(source_path,marked=marked)
    index['source_tensor']=dict(path=str(source_path),sha256=sha(source_path),tensor_sha256=tensor_sha(marked),
      norm=float(np.linalg.norm(marked)),OFF='exact zero tensor with identical geometry')
    atom(output/'result.json',index)
    by_physical={pid:[] for pid in physical_by_id}
    for n,c in enumerate(conditions):by_physical[c['physical_id']].append((n,c))
    for pi,recipe in enumerate(physical):
        pid=recipe['id'];clip=recipe['clip'];base=regular[clip['start']-1:clip['start']-1+clip['length']]
        if recipe['writer']=='OFF':base=np.zeros_like(base)
        Y=edit_rows(base,recipe['edit'],recipe['p']);noise=recipe['noise']
        if noise['sigma_over_alpha']:
            Y+=method.PUBLIC.alpha*noise['sigma_over_alpha']*np.random.default_rng(noise['seed']).standard_normal(Y.shape)
        physical_record=dict(recipe=recipe,shape=list(Y.shape),dtype=str(Y.dtype),tensor_sha256=tensor_sha(Y),
          noise_recipe='NumPy default_rng(seed).standard_normal(full edited tensor shape), multiplied by alpha*sigma_over_alpha')
        if recipe['edit']=='EXACT_COPY_INSERT':physical_record['repeat_same_tensor_sha256']=tensor_sha(edit_rows(base,'REPEAT',recipe['p']))
        index['physical'][pid]=physical_record
        for n,c in by_physical[pid]:
            L=len(Y);fid=f'{L}/{c["missing"]}/{c["key"]}'
            try:raw=method.infer(Y,KEYS[c['key']],availability(L,c['missing']))
            except Exception as exc:
                raw=failed_raw(L,f'{type(exc).__name__}: {exc}');index['failures'].append(dict(id=c['id'],error=str(exc)))
            envelope=dict(condition_id=c['id'],physical_id=pid,input_tensor_sha256=physical_record['tensor_sha256'],
              catalog=index['catalogs'][str(L)],equivalence_family=index['families'][fid],inference=raw)
            ref=atom(output/'raw'/f'{n:04d}.json.gz',envelope)
            index['conditions'][c['id']]=dict(status=raw['summary']['status'],group=c['group'],raw=ref)
        atom(output/'result.json',index)
        if (pi+1)%50==0:print(f'physical tensors completed: {pi+1}/{len(physical)}',flush=True)
    # Freeze every raw output before this separate truth join. Never rerun inference here.
    raw_hashes={c['id']:index['conditions'][c['id']]['raw']['sha256'] for c in conditions}
    raw_manifest=atom(output/'raw_manifest.json',dict(conditions=raw_hashes,all_raw_committed_before_truth=True))
    index.update(stage='POSTHOC',raw_manifest=raw_manifest);atom(output/'result.json',index)
    evaluations=[]
    for c in conditions:
        ref=index['conditions'][c['id']]['raw'];raw=read(ref['path'])['inference']
        evaluations.append(posthoc(raw,c,physical_by_id[c['physical_id']],catalogs[raw['received_length']]))
    post_ref=atom(output/'posthoc.json.gz',evaluations)
    raw_unchanged=all(sha(index['conditions'][c['id']]['raw']['path'])==raw_hashes[c['id']] for c in conditions)
    counts=dict(primary=sum(r['status']=='EVALUATED' for r in evaluations if r['group']=='primary'),
      counterexamples=sum(r['status']=='EVALUATED' for r in evaluations if r['group']=='counterexample'),
      physical_tensors=len(index['physical']),**totals)
    integrity=verify_cost_integrity(index)
    source_after={str(p.relative_to(ROOT)):sha(p) for p in SOURCE_PATHS}
    source_unchanged=source_before==source_after
    index.update(status='EXECUTION_COMPLETE' if counts==index['expected'] and raw_unchanged and integrity['status']=='PASS' and source_unchanged else 'INCOMPLETE',stage='FINISHED',
      cost_integrity=integrity,source_unchanged=source_unchanged,
      completed=counts,posthoc=post_ref,summary=summarize(evaluations),raw_unchanged_after_truth=raw_unchanged,
      wall_seconds=time.monotonic()-start,software=dict(python=sys.version,numpy=np.__version__),
      source_files=source_after)
    atom(output/'result.json',index)
    files=[p for p in output.rglob('*') if p.is_file() and p.name!='output_manifest.json']
    manifest={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)}
    tree=hashlib.sha256(''.join(f'{p}  {v["sha256"]}\n' for p,v in manifest.items()).encode()).hexdigest()
    atom(output/'output_manifest.json',dict(files=manifest,tree_sha256=tree))
    return index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.output);print(json.dumps({k:result[k] for k in ('status','completed','wall_seconds','summary')},indent=2))
    return 0 if result['status']=='EXECUTION_COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
