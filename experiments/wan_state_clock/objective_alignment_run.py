"""CPU-only fixed 6-terminal/12-pair/24-update objective comparison."""
import argparse,hashlib,json,math,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import state_clock,projection_margin,velocity_coefficients,objective_alignment as method

CASES=('dev_p0_s0','dev_p1_s0');STARTS=('OFF','LOCAL_A','LOCAL_B')
EXPECTED_SOURCE='9fdfac97126fa681c42423089eb342ce85b676a7'
PROTOCOL=Path(__file__).with_name('OBJECTIVE_ALIGNMENT_PROTOCOL.md')
PROTOCOL_SHA='e23bfad77c4ec3bc9440ee5e2b43e2e04689bc5355c2fe8edbf0294845351d84'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def layer(start,message):return 'OFF' if start=='OFF' else ('same_target' if (start=='LOCAL_A')==(message==0) else 'cross_target')

def run(source,output,media_source=None):
    torch.set_num_threads(1)
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:raise ValueError('input/output overlap')
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('frozen protocol changed')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',source_path=str(source),source_commit=EXPECTED_SOURCE,protocol_sha256=PROTOCOL_SHA,
                terminal_denominator=6,pair_denominator=12,update_denominator=24,inputs=[],failures=[],rows=[],budgets={},
                calibration_or_tuning=False,updated_MP4_evidence=None,model_calls=0,VAE_calls=0,backward_scope='detached terminal leaf CPU only')
    result['analysis_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['analysis_file_sha256']={str(p):sha(p) for p in (Path(__file__),Path(method.__file__),Path(velocity_coefficients.__file__),PROTOCOL)}
    def save():dump(output/'result.json',result)
    # Verify the entire six-input roster before any candidate tensor calculation.
    configs={};valid={};original_hashes={}
    for case in CASES:
        valid[case]=False
        try:
            root=source/case;gp=root/'generation.json';generation=json.loads(gp.read_text())
            result['inputs'].append(dict(path=str(gp),sha256=sha(gp),type='local_original_generation_record'))
            if generation['source_commit']!=EXPECTED_SOURCE:raise ValueError('original source mismatch')
            original_hashes[case]=generation['file_sha256']
            for module in (state_clock,projection_margin):
                suffix='/main/tube_state/'+Path(module.__file__).name
                expected=next(v for k,v in generation['source_sha256'].items() if k.endswith(suffix))
                if sha(Path(module.__file__))!=expected:raise ValueError('original codebook algorithm source mismatch')
            cp=root/'config.json';actual=sha(cp);expected=generation['file_sha256']['config.json']
            result['inputs'].append(dict(path=str(cp),sha256=actual,expected_sha256=expected,matched=actual==expected))
            if actual!=expected:raise ValueError('original config mismatch')
            configs[case]=json.loads(cp.read_text())
            ok=True
            for start in STARTS:
                path=root/(start+'_terminal.pt');actual=sha(path);expected=generation['file_sha256'][path.name]
                result['inputs'].append(dict(case=case,start=start,path=str(path),sha256=actual,expected_sha256=expected,matched=actual==expected,type='original_terminal'))
                ok=ok and actual==expected
            if not ok:raise ValueError('original terminal SHA mismatch')
            valid[case]=True
        except Exception as exc:result['failures'].append(dict(case=case,stage='input_verification',error=repr(exc)))
    for case in CASES:
        tensors=None;directions=codes=None;budget={}
        try:
            if not valid[case]:raise ValueError('unverified inputs')
            tensors={s:torch.load(source/case/(s+'_terminal.pt'),map_location='cpu',weights_only=True) for s in STARTS}
            book=state_clock.codebook(configs[case]['key_utf8'].encode())
            directions=torch.from_numpy(book['directions']).double();codes=torch.from_numpy(book['codes']).double()
            result.setdefault('existing_codebook',{})[case]=dict(logical_directions_sha256=hashlib.sha256(book['directions'].tobytes()).hexdigest(),logical_codes_sha256=hashlib.sha256(book['codes'].tobytes()).hexdigest(),meaning='deterministic reconstruction of original key and verified unchanged state_clock/carrier source; no new book')
            budget={m:method.rms(tensors['LOCAL_'+('A' if m==0 else 'B')].double()-tensors['OFF'].double()) for m in (0,1)}
            result['budgets'][case]={str(m):b for m,b in budget.items()}
        except Exception as exc:result['failures'].append(dict(case=case,stage='load_budget',error=repr(exc)));tensors=None
        for start in STARTS:
            for message in (0,1):
                pair=dict(case=case,start=start,target=message,stratum=layer(start,message),updates={},status='INCOMPLETE',new_minus_old_hard_gap_gain=None)
                result['rows'].append(pair);controls={}
                for objective in method.OBJECTIVES:
                    try:
                        if tensors is None:raise ValueError('verified tensors unavailable')
                        row,controls[objective]=method.update(tensors[start],directions,codes,message,objective,budget[message])
                        pair['updates'][objective]=row
                    except Exception as exc:pair['updates'][objective]=dict(status='FAILED',error=repr(exc));result['failures'].append(dict(case=case,start=start,target=message,objective=objective,error=repr(exc)))
                if len(controls)==2:
                    pair['status']='COMPLETE';a=pair['updates']['hinge'];b=pair['updates']['tanh']
                    pair['new_minus_old_hard_gap_gain']=b['hard_gap_gain']-a['hard_gap_gain']
                    x=controls['hinge'].double().flatten();y=controls['tanh'].double().flatten()
                    denominator=float(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y))
                    cosine=None if denominator==0 else float(torch.dot(x,y)/denominator)
                    pair['actual_control_cosine']=cosine if cosine is not None and math.isfinite(cosine) else None
                    pair['control_cosine_status']='MEASURED' if pair['actual_control_cosine'] is not None else 'UNDEFINED_ZERO_OR_NONFINITE_ACTUAL_CONTROL'
                controls={};save()
        tensors=directions=codes=None
    result['summary']={}
    for stratum in ('OFF','same_target','cross_target'):
        rows=[p for p in result['rows'] if p['stratum']==stratum]
        summary=dict(pair_denominator=4,complete=sum(p['status']=='COMPLETE' for p in rows),methods={})
        for objective in method.OBJECTIVES:
            records=[p['updates'][objective] for p in rows if p['updates'][objective]['status']=='COMPLETE']
            gains=[r['hard_gap_gain'] for r in records]
            summary['methods'][objective]=dict(update_denominator=4,complete=len(records),positive_hard_gap_gain=sum(v>0 for v in gains),negative_hard_gap_gain=sum(v<0 for v in gains),zero_hard_gap_gain=sum(v==0 for v in gains),nominal_correct_after=sum(r['after']['nominal_rank']=='CORRECT' for r in records),nominal_correct_before=sum(r['before']['nominal_rank']=='CORRECT' for r in records),hard_gap_gains=gains)
        summary['new_minus_old_hard_gap_gains']=[p['new_minus_old_hard_gap_gain'] for p in rows]
        result['summary'][stratum]=summary
    result['unchanged_MP4_reference']=dict(status='UNAVAILABLE',updated_candidate_evidence=False)
    if media_source is not None:
        try:
            path=Path(media_source);original=json.loads(path.read_text());refs=[]
            for case in CASES:
                saved=original['cases'][case]
                if saved.get('source_commit')!=EXPECTED_SOURCE:raise ValueError('MP4 reference source differs')
                for start in STARTS:
                    name=start+'_terminal.pt'
                    if saved['file_sha256'].get(name)!=original_hashes[case].get(name):raise ValueError('MP4 reference terminal SHA differs')
                    v=saved['videos'][start]
                    refs.append(dict(case=case,start=start,original_status=v.get('status'),original_rankings=v.get('rankings')))
            result['unchanged_MP4_reference']=dict(status='COPIED_UNCHANGED_REFERENCE',path=str(path),sha256=sha(path),rows=refs,updated_candidate_evidence=False)
        except Exception as exc:result['unchanged_MP4_reference']['error']=repr(exc)
    result['status']='CPU_DIAGNOSTIC_COMPLETE' if all(p['status']=='COMPLETE' for p in result['rows']) else 'WITH_RETAINED_FAILURES'
    result['claim']='same latent conditional-budget diagnosis; hard nominal gap is not blind media detection, existence, early native transmission or new independent validation'
    save();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--media-source');args=p.parse_args()
    r=run(args.source,args.output,args.media_source);print(json.dumps({k:r[k] for k in ('status','terminal_denominator','pair_denominator','update_denominator','summary')},indent=2))
