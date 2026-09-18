"""CPU-only known-time-coordinate diagnostic of saved inversion tensors.

No generation, inverse model, writer artifact, or receiver algorithm changes.
Run with python -m experiments.wan_state_clock.inversion_soft_observation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from main.tube_state import initial_noise as method

CASES = ('dev_p0_s0', 'dev_p0_s1', 'dev_p1_s0', 'dev_p1_s1')
ARMS = ('OFF', 'A', 'B')


def windows():
    return ([dict(group='short', start=t, stop=t+1) for t in range(46)]
            + [dict(group='medium_core', start=t, stop=t+4) for t in range(1,45,4)]
            + [dict(group='medium_boundary', start=t, stop=t+1) for t in (0,45)]
            + [dict(group='full', start=0, stop=46)])


PROTOCOL = dict(version='inversion_soft_observation_v1', cases=CASES, arms=ARMS,
    windows=windows(), shape=method.SHAPE, bits=16, repeats_per_slice=160,
    soft='mean recovered channel0 amplitude times pad over coordinates and window',
    baseline='sum sign(recovered channel0) times pad; original decoder unchanged',
    margin='mean(q * true) - mean(q * wrong); truth joined after observation',
    coordinate_assumption='known original latent-time index and pad; not blind crop recovery',
    independence='12 videos, 4 paired content-seed cases; windows and coordinates are correlated',
    boundary_policy='two singleton boundary windows retained separately; never pooled into core',
    scientific_pass=None)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def demodulate(recovered, book):
    method.check(recovered)
    order, pads = book['order'], book['pads']
    if order.shape != (2560,) or order.dtype != torch.int64 or not torch.equal(order.sort().values,torch.arange(2560)):
        raise ValueError('invalid coordinate permutation')
    if pads.shape != (46,2560) or not bool(((pads == 1) | (pads == -1)).all()):
        raise ValueError('invalid fixed time pads')
    values = recovered[0,0].reshape(46,2560)[:,order].double()
    return (values*pads).reshape(46,160,16), (values.sign()*pads).reshape(46,160,16)


def observe(recovered, book):
    """Payload-free amplitude and original sign-vote observations."""
    amplitude, votes = demodulate(recovered,book)
    rows = []
    for window in windows():
        a,b=window['start'],window['stop']
        soft=amplitude[a:b].mean((0,1)); hard=votes[a:b].sum((0,1))
        rows.append(dict(window, status='COMPLETE', coordinates_per_bit=(b-a)*160,
            soft_vector=soft.tolist(), vote_sums=hard.int().tolist(),
            soft_signs=soft.sign().int().tolist(), vote_signs=hard.sign().int().tolist(),
            soft_erasures=int((soft==0).sum()), vote_erasures=int((hard==0).sum())))
    return rows


def score(rows, payloads, truth):
    """Posthoc candidate comparison; never modifies observations or chooses windows."""
    candidates=torch.tensor(payloads,dtype=torch.float64)
    if candidates.shape != (2,16) or not bool(((candidates==1)|(candidates==-1)).all()):
        raise ValueError('two public 16-bit candidates required')
    if torch.equal(candidates[0],candidates[1]): raise ValueError('distinct candidates required')
    for row in rows:
        if row['status']!='COMPLETE': continue
        scoring={}
        for name,key in (('soft','soft_vector'),('vote','vote_sums')):
            q=torch.tensor(row[key],dtype=torch.float64)
            if name=='vote': q=q/row['coordinates_per_bit']
            scores=(candidates*q).mean(1)
            matches=(candidates==q.sign()).all(1)
            item=dict(candidate_scores=scores.tolist(),exact_candidate_matches=matches.tolist())
            if truth is not None:
                margins=q*candidates[truth]
                item.update(true_wrong_margin=float(scores[truth]-scores[1-truth]),
                    target_bit_margins=margins.tolist(),minimum_target_bit_margin=float(margins.min()),
                    bit_errors_including_erasures=int((q.sign()!=candidates[truth]).sum()),
                    exact_true_match=bool(matches[truth]))
            scoring[name]=item
        row['reporting_only']=scoring
    return rows


def baseline_check(recovered,book,original):
    decoded=method.read(recovered,book)
    fields=('signs','vote_sums','votes_per_bit','bit_erasures','zero_coordinate_votes')
    differences=[]
    for name,actual,expected in [('aggregate',decoded['aggregate'],original.get('aggregate',{}))]+[
        (f'per_time/{t}',row,(original.get('per_time',[])+[{}]*46)[t]) for t,row in enumerate(decoded['per_time'])]:
        for field in fields:
            if actual[field]!=expected.get(field): differences.append(name+'/'+field)
    return dict(status='MATCH' if not differences else 'MISMATCH',differences=differences)


def summarize(samples):
    summaries=[]
    for selection in [list(samples)]+[[s] for s in samples]:
        for group in ('short','medium_core','medium_boundary','full'):
            for arm in ('MARKED','OFF'):
                eligible=[s for s in selection if (s['arm']=='OFF')==(arm=='OFF')]
                rows=[r for s in eligible for r in s['windows'] if r['group']==group]
                good=[r for r in rows if r['status']=='COMPLETE']
                report=dict(scope='all' if len(selection)>1 else selection[0]['case']+'/'+selection[0]['arm'],
                    group=group,arm=arm,video_denominator=len(eligible),window_denominator=len(rows),
                    complete_windows=len(good),missing_or_failed_windows=len(rows)-len(good))
                for name in ('soft','vote'):
                    stats=dict(erasures=sum(r[name+'_erasures'] for r in good),bit_denominator=len(rows)*16)
                    if arm=='MARKED':
                        margins=[r['reporting_only'][name]['true_wrong_margin'] for r in good]
                        stats.update(exact_true_windows=sum(r['reporting_only'][name]['exact_true_match'] for r in good),
                            bit_errors_including_erasures=sum(r['reporting_only'][name]['bit_errors_including_erasures'] for r in good),
                            candidate_margin_min=min(margins) if margins else None,
                            candidate_margin_mean=sum(margins)/len(margins) if margins else None,
                            candidate_margin_max=max(margins) if margins else None)
                    else:
                        stats['exact_any_candidate_windows']=sum(any(r['reporting_only'][name]['exact_candidate_matches']) for r in good)
                    report[name]=stats
                if eligible: summaries.append(report)
    return summaries


def run(root,output):
    root,output=Path(root),Path(output)
    if output.exists(): raise FileExistsError('use a new independent output directory')
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    provenance={}; failures=[]
    def load(path,tensor=False):
        provenance[str(path.relative_to(root))]=sha(path)
        return torch.load(path,map_location='cpu',weights_only=True) if tensor else json.loads(path.read_text())
    manifest=load(root/'manifest.json'); original=load(root/'result.json')
    frozen=root/'FROZEN_PROTOCOL.txt'
    if frozen.exists(): provenance[frozen.name]=sha(frozen)
    if tuple(c['id'] for c in manifest['development'])!=CASES or tuple(manifest['arms'])!=ARMS:
        raise ValueError('manifest differs from fixed roster')
    payloads=manifest['payloads']; samples=[]
    expected_book=method.codebook(manifest['base_config']['key_utf8'].encode())
    for case in CASES:
        for arm in ARMS:
            sample=dict(case=case,arm=arm,truth_reporting_only=None if arm=='OFF' else ARMS.index(arm)-1,
                windows=[dict(w,status='NOT_RUN') for w in windows()])
            try:
                book=load(root/case/'codebook.pt',True)
                if set(book)!=set(expected_book) or any(not torch.equal(book[k],v) for k,v in expected_book.items()):
                    raise ValueError('saved codebook differs from manifest key')
                recovered=load(root/case/'receiver'/f'{arm}_recovered_noise.pt',True)
                rows=observe(recovered,book)
                sample['windows']=score(rows,payloads,sample['truth_reporting_only'])
                prior=original.get('cases',{}).get(case,{}).get('videos',{}).get(arm,{}).get('decoded',{})
                sample['baseline_check']=baseline_check(recovered,book,prior)
                full=rows[-1]
                sample['full_soft_vs_vote']=dict(sign_differences=sum(a!=b for a,b in zip(full['soft_signs'],full['vote_signs'])))
                sample['status']='COMPLETE' if sample['baseline_check']['status']=='MATCH' else 'BASELINE_MISMATCH'
            except Exception as exc:
                sample.update(status='MISSING_OR_FAILED',error=repr(exc))
                for row in sample['windows']: row['status']='MISSING_OR_FAILED'
            if sample['status']!='COMPLETE': failures.append(dict(case=case,arm=arm,status=sample['status'],error=sample.get('error')))
            samples.append(sample)
    result=dict(status='COMPLETE' if not failures else 'WITH_RETAINED_FAILURES',protocol=PROTOCOL,
        video_denominator=12,marked_video_denominator=8,off_video_denominator=4,
        input_root=str(root.resolve()),input_sha256=provenance,analysis_source_sha256=sha(Path(__file__)),
        failures=failures,samples=samples,summaries=summarize(samples),scientific_pass=None)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=run(args.input,args.output)
    print(json.dumps(dict(status=report['status'],video_denominator=12,failures=report['failures'])))
