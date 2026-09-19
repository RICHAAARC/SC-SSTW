"""Fixed CPU saved-result/tensor audit with torch; zero model/media calls."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
ARMS=('OFF','A','B')
LAYERS=('terminal','float_rgb','rgb8','mp4')
SOURCE_RUN='grow_temporal_difference_media_20260918T171544536253Z'

def sign(x):return (x>0)-(x<0)
EXPECTED_RESULT_SHA='723a596c51eb7f111c7df2e9ab241fbdcd191a0003e32e964184b73459627906'

def run(result_path,output,tensor_root):
    path=Path(result_path)
    output=Path(output)
    if output.resolve()==Path(tensor_root).resolve() or Path(tensor_root).resolve() in output.resolve().parents or output.resolve()==path.resolve():raise ValueError('output must be outside immutable tensor input')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_RESULT_SHA:raise ValueError('fixed media result SHA mismatch')
    source=json.loads(path.read_text());out={'source_run':SOURCE_RUN,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'case_denominator':4,'video_denominator':12,'layer_denominator':48,'marked_denominator':8,'fixed_bit_denominator':128,
        'records':[],'failures':[],'scientific_pass':None,'claim':'saved JSON recomputation, not rerun or new receiver validation; soft diagnostics do not replace fixed hard results'}
    for case in CASES:
        c=source.get('cases',{}).get(case,{})
        try:
            key=c['config']['key_utf8'].encode();digest=hashlib.sha256(key+b'/grow-video-v1/payload').digest()
            payload=[1 if digest[i//8]&(1<<(i%8)) else -1 for i in range(16)]
        except Exception as exc:
            payload=None;out['failures'].append({'case':case,'stage':'payload_metadata','error':repr(exc)})
        for arm in ARMS:
            target=payload if arm!='B' or payload is None else [-x for x in payload]
            for layer in LAYERS:
                row={'case':case,'arm':arm,'layer':layer,'status':'MISSING_OR_FAILED','pair_denominator':23,'bits':16}
                try:
                    old=c['videos'][arm]['layers'][layer]
                    if old['status']!='COMPLETE' or target is None:raise ValueError('source layer incomplete or payload absent')
                    pairs=old['per_pair'];agg=old['aggregate'];assert len(pairs)==23 and agg['votes_per_bit']==92
                    for k,p in enumerate(pairs):
                        assert p['pair']==k and p['times']==[2*k,2*k+1] and p['votes_per_bit']==4
                        assert p['signs']==[sign(x) for x in p['vote_sums']] and p['bit_erasures']==p['signs'].count(0)
                    votes=[sum(p['vote_sums'][i] for p in pairs) for i in range(16)]
                    soft=[statistics.mean(p['coefficient_mean_diagnostic'][i] for p in pairs) for i in range(16)]
                    assert votes==agg['vote_sums'] and [sign(x) for x in votes]==agg['signs']
                    assert agg['bit_erasures']==votes.count(0)
                    assert all(math.isfinite(x) and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12) for x,y in zip(soft,agg['coefficient_mean_diagnostic']))
                    comp=old['payload_comparisons_reporting_only']['aggregate']
                    for m,truth in enumerate((payload,[-x for x in payload])):
                        e=sum(sign(v)!=t for v,t in zip(votes,truth));assert comp[m]['bit_errors_including_erasures']==e and comp[m]['ber_including_erasures']==e/16 and comp[m]['exact_payload_match']==(e==0)
                    row.update(status='VERIFIED',signed_votes=[v*t for v,t in zip(votes,target)],signed_soft=[v*t for v,t in zip(soft,target)],
                        hard_errors=sum(sign(v)!=t for v,t in zip(votes,target)),erasures=votes.count(0),
                        soft_errors_diagnostic=sum(sign(v)!=t for v,t in zip(soft,target)),
                        per_pair_signed_votes=[[v*t for v,t in zip(p['vote_sums'],target)] for p in pairs],
                        per_pair_signed_soft=[[v*t for v,t in zip(p['coefficient_mean_diagnostic'],target)] for p in pairs],
                        reference_payload_only=arm=='OFF',truth=None if arm=='OFF' else ARMS.index(arm)-1,OFF_exact_coincidences=[p['message'] for p in comp if p['exact_payload_match']] if arm=='OFF' else None)
                except Exception as exc:out['failures'].append({'case':case,'arm':arm,'layer':layer,'error':repr(exc)})
                if arm=='OFF':row['hard_errors']=None;row['soft_errors_diagnostic']=None
                out['records'].append(row)
    out['layers']={};out['failed_mp4_bits']=[]
    for layer in LAYERS:
        rows=[x for x in out['records'] if x['layer']==layer and x['arm']!='OFF' and x['status']=='VERIFIED'];off=[x for x in out['records'] if x['layer']==layer and x['arm']=='OFF' and x['status']=='VERIFIED']
        out['layers'][layer]=dict(marked_denominator=8,completed=len(rows),missing=8-len(rows),hard_exact=sum(x['hard_errors']==0 for x in rows),
            hard_errors=sum(x['hard_errors'] for x in rows),erasures=sum(x['erasures'] for x in rows),soft_exact_diagnostic=sum(x['soft_errors_diagnostic']==0 for x in rows),
            soft_errors_diagnostic=sum(x['soft_errors_diagnostic'] for x in rows),mean_signed_soft=statistics.mean(v for x in rows for v in x['signed_soft']) if rows else None,
            OFF_denominator=4,OFF_completed=len(off),OFF_exact_coincidences=sum(bool(x['OFF_exact_coincidences']) for x in off))
    out['transitions']={}
    for left,right in zip(LAYERS,LAYERS[1:]):
        deltas=[];vote_deltas=[];before=[];after=[]
        for case in CASES:
            for arm in ('A','B'):
                pair=[next(x for x in out['records'] if (x['case'],x['arm'],x['layer'])==(case,arm,l)) for l in (left,right)]
                if any(x['status']!='VERIFIED' for x in pair):continue
                a,b=pair;before+=a['signed_soft'];after+=b['signed_soft'];deltas += [y-x for x,y in zip(a['signed_soft'],b['signed_soft'])];vote_deltas += [y-x for x,y in zip(a['signed_votes'],b['signed_votes'])]
        out['transitions'][left+'_to_'+right]=dict(observed_bits=len(deltas),mean_soft_delta=statistics.mean(deltas) if deltas else None,mean_signed_vote_delta=statistics.mean(vote_deltas) if deltas else None,ratio_of_mean_soft=sum(after)/sum(before) if before and sum(before)!=0 else None)
    for x in out['records']:
        if x['layer']=='mp4' and x['arm']!='OFF' and x['status']=='VERIFIED':
            for bit,v in enumerate(x['signed_votes']):
                if v<=0:
                    rows=[next(y for y in out['records'] if (y['case'],y['arm'],y['layer'])==(x['case'],x['arm'],l)) for l in LAYERS]
                    out['failed_mp4_bits'].append(dict(case=x['case'],arm=x['arm'],bit=bit,layers={y['layer']:dict(signed_votes=y['signed_votes'][bit],signed_soft=y['signed_soft'][bit]) if y['status']=='VERIFIED' else dict(status=y['status']) for y in rows}))
    out['soft_negative_bits_diagnostic']=[dict(case=x['case'],arm=x['arm'],layer=x['layer'],bit=i,signed_soft=v,signed_votes=x['signed_votes'][i]) for x in out['records'] if x['arm']!='OFF' and x['status']=='VERIFIED' for i,v in enumerate(x['signed_soft']) if v<=0]
    out['tensor_diagnostics']=tensor_diagnostics(source,tensor_root,out)
    out['actual_calls']={'transformer':0,'vae':0,'mp4_encode':0,'generation':0}
    out['tensor_summary']={'verified':out['tensor_diagnostics']['verified_tensors'],'denominator':48,'failure_count':len(out['failures']), 'transition_sign_flips':{name:sum(x.get('sign_flips',0) for x in out['tensor_diagnostics']['transitions'] if x['transition']==name and x['arm']!='OFF' and x['status']=='MEASURED') for name in ('terminal_to_float_rgb','float_rgb_to_rgb8','rgb8_to_mp4')}}
    out['status']='DIAGNOSTIC_COMPLETE' if not out['failures'] else 'WITH_RETAINED_FAILURES'
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('status','actual_calls','tensor_summary','layers','transitions','failed_mp4_bits')},indent=2));return out
def tensor_diagnostics(source,root,out):
    import torch
    from main.tube_state import grow_temporal_difference as method
    torch.set_num_threads(1)
    root=Path(root);result={'method_sha256':hashlib.sha256(Path(method.__file__).read_bytes()).hexdigest(),'spatial_dependency_sha256':hashlib.sha256(Path(method.spatial.__file__).read_bytes()).hexdigest(),'layer_tensor_denominator':48,'records':[],'transitions':[],'books':{},'configs':{},'source_commits':{}}
    for case in CASES:
        c=source.get('cases',{}).get(case,{})
        try:
            book=method.codebook(c['config']['key_utf8'].encode());result['books'][case]=book;result['configs'][case]=c['config'];result['source_commits'][case]=c['source_commit']
        except Exception as exc:
            out['failures'].append(dict(case=case,stage='tensor_book',error=repr(exc)));book=None
        for arm in ARMS:
            coefficients={}
            for layer in LAYERS:
                row=dict(case=case,arm=arm,layer=layer,status='MISSING_OR_FAILED',pair_denominator=23,coefficient_denominator=1472)
                try:
                    if book is None:raise ValueError('missing book')
                    path=root/case/'latents'/f'{arm}_{layer}.pt';actual=hashlib.sha256(path.read_bytes()).hexdigest()
                    assert actual==c['file_sha256'][f'latents/{arm}_{layer}.pt']
                    z=torch.load(path,map_location='cpu',weights_only=True)
                    decoded=method.read(z,book);expected=c['videos'][arm]['layers'][layer]['aggregate'];actual_read=decoded['aggregate']
                    for key in actual_read:
                        if key=='coefficient_mean_diagnostic':
                            err=max(abs(a-b) for a,b in zip(actual_read[key],expected[key]));assert err<=1e-12
                            row['recomputed_soft_maxabs_difference']=err
                        else:assert actual_read[key]==expected[key]
                    x=method.selected(z.double(),book).reshape(23,4,16);coefficients[layer]=x
                    row.update(status='VERIFIED',sha256=actual,coefficients_by_pair_repeat_bit=x.tolist())
                except Exception as exc:out['failures'].append(dict(case=case,arm=arm,layer=layer,stage='tensor',error=repr(exc)))
                result['records'].append(row)
            for left,right in zip(LAYERS,LAYERS[1:]):
                row=dict(case=case,arm=arm,transition=left+'_to_'+right,status='MISSING_OR_FAILED',bit_denominator=16)
                if left in coefficients and right in coefficients:
                    x,y=coefficients[left],coefficients[right];den=float(x.square().sum());k=float((x*y).sum())/den if den else None
                    residual=y-k*x if k is not None else y
                    row.update(status='MEASURED',zero_intercept_gain=k,fit_residual_rms=float(residual.square().mean().sqrt()),
                        input_rms=float(x.square().mean().sqrt()),output_rms=float(y.square().mean().sqrt()),
                        sign_flips=int((x.sign()!=y.sign()).sum()),per_pair_flips=(x.sign()!=y.sign()).sum((1,2)).tolist(),
                        per_repeat_bit_flips=(x.sign()!=y.sign()).sum(0).tolist(),per_bit=[])
                    if arm!='OFF':
                        target=torch.tensor(book['payloads'][ARMS.index(arm)-1],dtype=x.dtype);a=x*target;b=y*target
                        for bit in range(16):
                            v,w=a[:,:,bit],b[:,:,bit]
                            row['per_bit'].append(dict(bit=bit,correct_to_wrong_or_zero=int(((v>0)&(w<=0)).sum()),wrong_or_zero_to_correct=int(((v<=0)&(w>0)).sum()),
                                signed_input_mean=float(v.mean()),signed_output_mean=float(w.mean()),input_correct_count=int((v>0).sum()),output_correct_count=int((w>0).sum()),
                                flipped_input_absolute_mean=float(v[(v.sign()!=w.sign())].abs().mean()) if bool((v.sign()!=w.sign()).any()) else None))
                    row['claim']='same-coordinate descriptive transmission; fitted gain is not causal codec isolation or receiver correction'
                result['transitions'].append(row)
    result['verified_tensors']=sum(x['status']=='VERIFIED' for x in result['records'])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--output',required=True);p.add_argument('--tensor-root',required=True);a=p.parse_args();run(a.result,a.output,a.tensor_root)
