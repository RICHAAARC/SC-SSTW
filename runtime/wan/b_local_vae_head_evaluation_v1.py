"""Post-blind reporting only. Never imported by the pure receiver."""
import numpy as np

FIXTURES=('CROP','DELETE_LEFT_GAP','DELETE_RIGHT_GAP','REPEAT_LEFT_GAP','REPEAT_RIGHT_GAP')

def source_frames(a,fixture):
    if fixture not in FIXTURES:raise ValueError('fixed five fixtures only')
    r=np.arange(151);d=8 if fixture.startswith('DELETE') else -8 if fixture.startswith('REPEAT') else 0
    b=49 if 'LEFT' in fixture else 102
    values=a+r+d*(r>=b)
    if values.min()<0 or values.max()>180:raise ValueError('fixture outside source')
    return values.tolist()

def assess(row,plan,read,actual,message):
    """Call only after that phase's complete blind decisions are saved/read back."""
    support=plan.get('support',[]);used=[x['received_frame'] for x in support]
    references=[dict(received_frame=x['received_frame'],claimed=x['source_frame'],
        actual=actual[x['received_frame']],exact=x['source_frame']==actual[x['received_frame']]) for x in support]
    output=read.get('status')=='OUTPUT';bits=read.get('information_bits',[None]*32)
    raw=read.get('raw_information_bits',[None]*32)
    positive=row['kind']=='positive'
    ref_ok=bool(references) and all(x['exact'] for x in references)
    correct=output and bits==message
    wrong=sum(b!=t for b,t in zip(bits,message)) if positive and output else None
    raw_known=[i for i,b in enumerate(raw) if b is not None]
    bch=read.get('BCH',{});labels=bch.get('information_observation',[])
    return dict(**row,status=read.get('status','NOT_STARTED'),detected=plan.get('detected',False),
        full_message_correct=correct if positive else None,output=output,
        emitted_slots=32 if output else 0,erased_slots=32 if read.get('status')=='ERASURE' else 0,
        unresolved_slots=32 if read.get('status') not in ('OUTPUT','ERASURE') else 0,
        literal_bit_errors=wrong,raw_information_bits=raw,information_bits=bits,
        raw_observed_slots=len(raw_known),raw_bit_errors=sum(raw[i]!=message[i] for i in raw_known) if positive else None,
        raw_error_positions=[i for i in raw_known if raw[i]!=message[i]] if positive else None,
        raw_BER=(sum(raw[i]!=message[i] for i in raw_known)/len(raw_known)) if positive and raw_known else None,
        temporal_support_frames=len(support),exact_support_frames=sum(x['exact'] for x in references),
        erroneous_reference_frames=sum(not x['exact'] for x in references),all_used_references_exact=ref_ok,
        supported_correct_information_slots=32 if positive and correct and ref_ok else 0,
        direct_observed_information_slots=sum(x=='DIRECT_OBSERVED' for x in labels),
        ECC_reconstructed_information_slots=sum(x=='ECC_RECONSTRUCTED' for x in labels),
        source_span=plan.get('source_span',0),edit_status=plan.get('edit_status'),references=references,
        false_attribution=bool(positive and output and not ref_ok),
        null_detection=bool(not positive and plan.get('detected',False)),null_output=bool(not positive and output),
        wrong_key=row['key']=='key01',actual_per_bit_watermark_support_claim=False)

def summary(rows):
    pos=[r for r in rows if r['kind']=='positive'];null=[r for r in rows if r['kind']=='null']
    wrong=[r for r in null if r['wrong_key']]
    return dict(tasks=len(rows),positive_tasks=len(pos),null_tasks=len(null),information_slots=32*len(rows),
        positive_detected=sum(r['detected'] for r in pos),full_message_correct=sum(r['full_message_correct'] is True for r in pos),
        positive_emitted_slots=sum(r['emitted_slots'] for r in pos),positive_erased_slots=sum(r['erased_slots'] for r in pos),
        positive_unresolved_slots=sum(r['unresolved_slots'] for r in pos),
        supported_correct_information_slots=sum(r['supported_correct_information_slots'] for r in pos),
        literal_bit_errors=sum(r['literal_bit_errors'] or 0 for r in pos),
        erroneous_reference_frames=sum(r['erroneous_reference_frames'] for r in pos),
        false_attribution_tasks=sum(r['false_attribution'] for r in pos),
        null_detected=sum(r['null_detection'] for r in null),null_outputs=sum(r['null_output'] for r in null),
        wrong_key_tasks=len(wrong),wrong_key_detected=sum(r['null_detection'] for r in wrong),wrong_key_outputs=sum(r['null_output'] for r in wrong),
        unresolved_tasks=sum(r['status'] not in ('OUTPUT','ERASURE') for r in rows))

def criteria(rows,phase):
    pos=[r for r in rows if r['kind']=='positive'];null=[r for r in rows if r['kind']=='null']
    expected=(10,40) if phase=='CAL' else (20,80)
    null_complete=len(null)==expected[1] and all(r['status'] in ('OUTPUT','ERASURE') for r in null)
    checks=dict(fixed_tasks_present=(len(pos),len(null))==expected,
        all_full_messages_correct=all(r['full_message_correct'] is True for r in pos),
        null_existence_zero=all(not r['detected'] for r in null) if null_complete else None,
        null_output_zero=all(not r['output'] for r in null) if null_complete else None,
        observations_complete=all(r['status'] in ('OUTPUT','ERASURE') for r in rows),
        support_and_span=all(r['temporal_support_frames']>=24 and r['source_span']>=64 for r in pos))
    if phase=='DEV':checks['all_used_references_exact']=all(r['all_used_references_exact'] for r in pos)
    return dict(passed=all(v is True for v in checks.values()),checks=checks,summary=summary(rows))

def paired(main,baseline):
    """Compare whole recovered information slots; do not relabel ECC as observed bits."""
    other={r['task']:r for r in baseline};result=[]
    for row in main:
        if row['kind']!='positive':continue
        b=other[row['task']];n=row['supported_correct_information_slots'];old=b['supported_correct_information_slots']
        common=[i for i,(x,y) in enumerate(zip(row['raw_information_bits'],b['raw_information_bits'])) if x is not None and y is not None]
        main_errors=len(set(common)&set(row['raw_error_positions'] or []));baseline_errors=len(set(common)&set(b['raw_error_positions'] or []))
        result.append(dict(task=row['task'],source=row['source'],fixture=row['fixture'],
            added_supported_information_slots=max(n-old,0),lost_supported_information_slots=max(old-n,0),
            common_raw_information_slots=len(common),common_raw_information_indices=common,
            main_common_raw_errors=main_errors,I_common_raw_errors=baseline_errors,
            main_raw_BER=main_errors/len(common) if common else None,I_raw_BER=baseline_errors/len(common) if common else None,
            main_status=row['status'],I_status=b['status']))
    return result

def quality(off,wm,*,precodec):
    """42 fixed pairs. Numerical observations only, no perceptual admission gate."""
    import torch
    a=torch.as_tensor(off).detach().cpu();b=torch.as_tensor(wm).detach().cpu()
    if a.shape!=b.shape or a.ndim!=4 or tuple(a.shape[1:])!=(320,512,3):raise ValueError('quality full canvas shape mismatch')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise ValueError('nonfinite quality input')
    if a.dtype==torch.uint8:a=a.float()/255
    if b.dtype==torch.uint8:b=b.float()/255
    regions={'full':np.ones((320,512),bool),'L':np.zeros((320,512),bool),'R':np.zeros((320,512),bool)}
    regions['L'][32:288,:256]=True;regions['R'][32:288,256:]=True;regions['outside_L']=~regions['L']
    values={}
    for name,mask in regions.items():
        delta=b[:,mask,:]-a[:,mask,:];mse=float(delta.square().mean());adj=delta[1:]-delta[:-1]
        values[name]=dict(MSE=mse,PSNR=None if mse==0 else float(-10*np.log10(mse)),PSNR_infinite=mse==0,
            max_abs_delta=float(delta.abs().max()),changed_values=int(torch.count_nonzero(delta)),
            adjacent_delta_MSE=float(adj.square().mean()))
    equal=bool(torch.equal(a[:,regions['outside_L'],:],b[:,regions['outside_L'],:]))
    return dict(status='RECORDED',frames=len(a),adjacent=len(a)-1,regions=values,
        outside_L_exact_equal=equal,precodec=precodec,perceptual_PASS=None,
        codec_leakage_outside_L=None if precodec else values['outside_L'])
