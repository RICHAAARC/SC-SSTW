"""Prepared same-carrier hard/raw-soft comparison; never selects a winning decoder."""
import argparse,hashlib,json
from pathlib import Path
from main.tube_state.grow_readout_comparison import score,compare,candidate_scores
from main.tube_state import grow_readout_comparison as scoring
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
ARMS=('OFF','A','B');LAYERS=('terminal','float_rgb','rgb8','mp4')

def run(input_result,output):
    path=Path(input_result);output=Path(output)
    if output.resolve()==path.parent.resolve() or path.parent.resolve() in output.resolve().parents:raise ValueError('output must be outside saved input directory')
    data=json.loads(path.read_text());result=dict(status='PREPARING',video_denominator=12,layer_denominator=48,marked_denominator=8,fixed_bits_per_layer=128,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest(),actual_calls={'generation':0,'transformer':0,'vae':0,'media_encode':0},records=[],failures=[],scientific_pass=None,
        rule='hard sign(sum sign coefficients); soft sign(mean raw coefficients); threshold zero; both retained, no selection')
    for case in CASES:
        c=data.get('cases',{}).get(case,{})
        try:
            d=hashlib.sha256(c['config']['key_utf8'].encode()+b'/grow-video-v1/payload').digest();payload=[1 if d[i//8]&(1<<(i%8)) else -1 for i in range(16)]
        except Exception:payload=None
        for arm in ARMS:
            for layer in LAYERS:
                row=dict(case=case,arm=arm,layer=layer,status='MISSING_OR_FAILED',truth=None if arm=='OFF' else ARMS.index(arm)-1)
                try:
                    saved=c['videos'][arm]['layers'][layer]
                    if saved['status']!='COMPLETE' or payload is None:raise ValueError('missing complete observation/metadata')
                    scores=score(saved['aggregate']);candidates=candidate_scores(scores,[payload,[-x for x in payload]])
                    row.update(status='COMPLETE',scores=scores,candidates=candidates,source_commit=c.get('source_commit'))
                    if arm=='OFF':row['reference_payload_comparisons']=[compare(scores,payload),compare(scores,[-x for x in payload])]
                    else:
                        row['recovery']=compare(scores,payload if arm=='A' else [-x for x in payload])
                        row['candidate_attribution_reporting_only']={k:(v['top']==row['truth']) if v['top'] is not None else False for k,v in candidates.items()}
                except Exception as exc:result['failures'].append(dict(case=case,arm=arm,layer=layer,error=repr(exc)))
                result['records'].append(row)
    result['summary']={}
    for layer in LAYERS:
        rows=[r for r in result['records'] if r['layer']==layer and r['arm']!='OFF' and r['status']=='COMPLETE']
        summary=dict(marked_denominator=8,completed=len(rows),missing=8-len(rows),fixed_bit_denominator=128)
        for decoder in ('hard','soft'):
            summary[decoder]=dict(exact=sum(r['recovery'][decoder]['exact'] for r in rows),errors_observed=sum(r['recovery'][decoder]['bit_errors_including_erasures'] for r in rows),erasures_observed=sum(r['recovery'][decoder]['erasures'] for r in rows),candidate_attribution_correct=sum(r['candidate_attribution_reporting_only'][decoder] for r in rows),candidate_ties=sum(r['candidates'][decoder]['tie'] for r in rows),selected=False)
        summary['paired_message_outcomes']={k:0 for k in ('both','hard_only','soft_only','neither')}
        for r in rows:
            h,s=(r['recovery'][x]['exact'] for x in ('hard','soft'));key='both' if h and s else 'hard_only' if h else 'soft_only' if s else 'neither';summary['paired_message_outcomes'][key]+=1
        summary['OFF_completed']=sum(r['layer']==layer and r['arm']=='OFF' and r['status']=='COMPLETE' for r in result['records']);summary['OFF_denominator']=4
        result['summary'][layer]=summary
    result['status']='COMPLETE' if not result['failures'] else 'WITH_RETAINED_FAILURES'
    output.mkdir(parents=True,exist_ok=False);(output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-result',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input_result,a.output)
