"""Offline fixed-roster analysis; no model, generation, or parameter selection."""
import argparse,hashlib,json
from pathlib import Path
import torch
from main.tube_state import grow_frequency as method

def run(root,output):
    root=Path(root);r=json.loads((root/'result.json').read_text());out={'source_run':'grow_late_control_20260918T080310904782Z','video_denominator':12,'marked_denominator':8,'cases':{},'claim':'development diagnostic; OFF and AB are mechanism references, never receivers; soft is not a selected replacement'}
    for case,c in r['cases'].items():
        book=method.codebook(c['config']['key_utf8'].encode());p=torch.tensor(book['payloads'][0],dtype=torch.float64);values={};summary={}
        for arm in ('OFF','A','B'):
            path=root/case/(arm+'_terminal.pt');digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==c['file_sha256']['latents/'+arm+'_terminal.pt']
            z=torch.load(path,map_location='cpu',weights_only=True);coef=method.selected(z.double(),book).reshape(46,4,16);values[arm]=coef
            rd=method.read(z,book);assert rd['aggregate']==c['videos'][arm]['terminal']['aggregate']
            target=p if arm!='B' else -p
            votes=coef.sign().sum((0,1));soft=coef.mean((0,1));signed=coef*target
            summary[arm]={'sha256':digest,'coefficients_by_time_repeat_bit':coef.tolist(),'coordinates':book['coordinates'],
                'hard_errors':int((votes.sign()!=target).sum()) if arm!='OFF' else None,'hard_erasures':int((votes==0).sum()),
                'soft_errors_diagnostic':int((soft.sign()!=target).sum()) if arm!='OFF' else None,
                'hard_signed_vote_margin':(votes*target).tolist(),'soft_signed_mean':(soft*target).tolist(),
                'frequency_signed_mean':signed.mean(0).tolist(),'frequency_signed_sign_vote':signed.sign().sum(0).tolist(),
                'time_signed_vote_margin':signed.sign().sum(1).tolist(),
                'time_bit_erasures':int((coef.sign().sum(1)==0).sum()),
                'wrong_frequency_time_slots':int((signed<0).sum()),
                'nearzero_frequency_time_slots_1e_6':int((coef.abs()<1e-6).sum()),
                'direction_score':float(signed.mean())}
        for arm in ('A','B'):
            target=p if arm=='A' else -p;change=(values[arm]-values['OFF'])*target
            summary[arm]['marked_minus_OFF_signed_mean']=float(change.mean())
            summary[arm]['positive_response_fraction']=float((change>0).double().mean())
            summary[arm]['per_frequency_OFF_bias']=values['OFF'].mean(0).tolist()
            summary[arm]['per_frequency_response_signed_mean']=change.mean(0).tolist()
            steps=c['videos'][arm]['steps'];last=steps[-1]
            summary[arm]['last_control']={k:last[k] for k in ['loss_before','loss_after_local','control_induced_delta_rms','terminal_vs_last_controlled_clean_rms']}
        off=values['OFF'];mid=(values['A']+values['B'])/2;response=(values['A']-values['B'])/2
        pair=(off[::2]-off[1::2])/2**.5
        diagnostics={'OFF_temporal_mean_energy_fraction':float(off.mean(0).square().mean()/off.square().mean()),
            'common_bias_temporal_mean_energy_fraction':float(mid.mean(0).square().mean()/mid.square().mean()),
            'common_bias_exceeds_AB_half_fraction':float((mid.abs()>response.abs()).double().mean()),
            'OFF_coefficient_rms':float(off.square().mean().sqrt()),'OFF_fixed_adjacent_difference_rms':float(pair.square().mean().sqrt()),
            'note':'fixed adjacent pairing 0/1,...44/45 selected as a mechanism diagnostic, not a tuned receiver or simulated marked result'}
        ab=(values['A']-values['B'])*p
        out['cases'][case]={'arms':summary,'bias_diagnostics':diagnostics,'AB_payload_signed_mean':float(ab.mean()),'AB_positive_fraction':float((ab>0).double().mean()),'AB_per_frequency_signed_mean':ab.mean(0).tolist()}
    Path(output).write_text(json.dumps(out,indent=2)+'\n')
    for case,c in out['cases'].items():
        for arm in ('A','B'):
            a=c['arms'][arm];print(case,arm,'hard',a['hard_errors'],'erase',a['hard_erasures'],'soft',a['soft_errors_diagnostic'],'timeerase',a['time_bit_erasures'],'direction',round(a['direction_score'],6),'deltaOFF',round(a['marked_minus_OFF_signed_mean'],6),'positiveResponse',round(a['positive_response_fraction'],4),'wrongslots',a['wrong_frequency_time_slots'],'nearzero',a['nearzero_frequency_time_slots_1e_6'])
        print('AB',case,c['AB_payload_signed_mean'],c['AB_positive_fraction'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
