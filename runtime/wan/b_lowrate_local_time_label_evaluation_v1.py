"""Post-only geometry, wrong-valid IDs, payload and coverage arithmetic; no receiver calls."""
from pathlib import Path
from collections import Counter
from main.tube_state import b_lowrate_local_time_label_plan_v1 as p
from runtime.wan.b_lowrate_local_time_label_v1 import read,save,raw_path,blind_path,pay_path,fam_path

def nominal(origin,phase,j):
    frames=list(range(origin+phase+8*j-3,origin+phase+8*j+5));labels={(t+3)//8 for t in frames}
    if min(frames)<5 or max(frames)>172:return dict(kind='UNWRITTEN_OR_GUARD',true_id=None,source_frames=frames,nominal_not_full_VAE_causal_support=True)
    return dict(kind='FULL' if len(labels)==1 else 'MIXED',true_id=next(iter(labels)) if len(labels)==1 else None,source_frames=frames,nominal_not_full_VAE_causal_support=True)

def id_support(origin,value):
    if value is None:return 'UNRESOLVED_ID'
    if type(value) is not int or value not in range(1,22):return 'INVALID_OR_NONFINITE_ID'
    left,right=8*value-3,8*value+4
    if right<origin or left>origin+51:return 'ABSENT'
    return 'FULL_PRESENT' if left>=origin and right<=origin+51 else 'PARTIAL_PRESENT'

def compare(a,b,target):
    common=a is not None and b is not None and target is not None
    return dict(baseline=a,marked=b,target_post_only=target,common=common,baseline_error=common and a!=target,marked_error=common and b!=target,correct_retained=target is not None and a==b==target,correct_lost=target is not None and a==target and b!=target,correct_added=target is not None and a!=target and b==target,added_coverage=a is None and b is not None,lost_coverage=a is not None and b is None,undefined=not common)

def pair_summary(rows):
    names=('common','baseline_error','marked_error','correct_retained','correct_lost','correct_added','added_coverage','lost_coverage','undefined');out={n:sum(bool(r[n]) for r in rows) for n in names};out['planned']=len(rows);n=out['common']
    out.update(baseline_common_BER=out['baseline_error']/n if n else None,marked_common_BER=out['marked_error']/n if n else None)
    return out

def evaluate(output,protocol):
    out=Path(output)
    if not (out/'markers/blind_complete.json').is_file():raise ValueError('blind products must be saved/read back first')
    origins=read(out/'private/preparation.json')['origins'];issues=[];message=[None]*32
    try:
        message=read(protocol)['config']['writer']['bits']
        if len(message)!=32 or any(type(x) is not int or x not in (0,1) for x in message):raise ValueError('original32 binary bits')
    except (OSError,ValueError,KeyError,TypeError) as e:issues.append('original payload truth: '+str(e));message=[None]*32
    observations=[];local_final=[];family_rows=[]
    families={t:read(out/fam_path(t)) for t in p.families()}
    for t in p.tag_tables():
        view,h,a,phase,k=t;raw=read(out/raw_path(t));family=families[(h,a,k)] if view=='local' else None
        blind=read(out/blind_path(t)) if view!='local' else None
        for n,r in enumerate(raw['rows']):
            geom=nominal(origins[h],phase,r['row']) if view=='local' else dict(kind='FULL',true_id=r['row'],nominal_not_full_VAE_causal_support=True)
            decoded=family['data_decodes'][str(phase)][n] if family else blind['rows'][n]['data']
            candidates=[x['rows'][n] for x in family['hypotheses'] if x['phase']==phase] if family else [blind['rows'][n]]
            accepted=any(x['accepted'] for x in candidates);pilot=any(x['pilot']['passed'] for x in candidates);value=decoded['decoded_id'];positive=a=='TAG' and k=='key00';true_id=geom['true_id'] if positive else None
            observations.append(dict(view=view,host=h,arm=a,phase=phase,key=k,row=r['row'],score_status=r['status'],material_positive=positive,geometry=geom,RM_status=decoded['status'],RM_id=value,RM_distance=decoded['nearest_distance'],candidate_pilot_present=pilot,candidate_accepted=accepted,decoded_ID_support_post_only=id_support(origins[h],value) if view=='local' else 'FULL_PRESENT' if value is not None else 'UNRESOLVED_ID',accepted_ID_support_post_only=id_support(origins[h],value) if view=='local' and accepted else 'FULL_PRESENT' if accepted else 'NOT_ACCEPTED',missing_valid_ID_candidate_accepted=accepted and positive and view=='local' and id_support(origins[h],value)=='ABSENT',partial_valid_ID_candidate_accepted=accepted and positive and view=='local' and id_support(origins[h],value)=='PARTIAL_PRESENT',correct_valid_RM=positive and true_id is not None and value==true_id,wrong_valid_RM=positive and true_id is not None and value is not None and value!=true_id,unsupported_RM=positive and true_id is None and value is not None,null_RM=value is not None and not positive,correct_candidate_accepted=accepted and positive and true_id is not None and value==true_id,wrong_ID_candidate_accepted=accepted and positive and true_id is not None and value!=true_id,unsupported_candidate_accepted=accepted and positive and true_id is None,null_candidate_accepted=accepted and not positive))
    for t,f in families.items():
        h,a,k=t;phase=f['selected_phase'];positive=a=='TAG' and k=='key00';origin=f['provisional_origin']
        family_rows.append(dict(host=h,arm=a,key=k,positive=positive,status=f['status'],selected_phase=phase,selected_clock=f['selected_clock'],provisional_origin=origin,true_origin_post_only=origins[h] if positive else None,correct_origin=positive and origin==origins[h],wrong_origin=positive and origin is not None and origin!=origins[h],null_origin=not positive and origin is not None,technical_missing=bool(f['missing_phases'])))
        for r in f['rows']:
            geom=nominal(origins[h],phase,r['row']) if phase is not None else dict(kind='NO_PHASE',true_id=None);value=r['accepted_id'];accepted=r['accepted'];target=geom['true_id']
            local_final.append(dict(host=h,arm=a,key=k,row=r['row'],positive=positive,accepted=accepted,accepted_id=value,geometry=geom,decoded_ID_support_post_only=id_support(origins[h],r.get('data',{}).get('decoded_id')),accepted_ID_support_post_only=id_support(origins[h],value) if accepted else 'NOT_ACCEPTED',correct=accepted and positive and target is not None and value==target,wrong_valid_ID=accepted and positive and target is not None and value!=target,unsupported_accepted=accepted and positive and target is None,null_accepted=accepted and not positive,missing_valid_ID_accepted=accepted and positive and id_support(origins[h],value)=='ABSENT',partial_valid_ID_accepted=accepted and positive and id_support(origins[h],value)=='PARTIAL_PRESENT',missing_phase=phase is None))
    group=[];cells=[];local=[]
    for h in p.HOSTS:
        base=read(out/pay_path(('mp4',h,'P0',-1,'key00')));tag=read(out/pay_path(('mp4',h,'TAG',-1,'key00')))
        for b in range(32):group.append(dict(host=h,bit=b,**compare(base['decoded_bits'][b],tag['decoded_bits'][b],message[b])))
        for t in range(44):
            for b in range(8):
                index=8*(t//11)+b;cells.append(dict(host=h,cell=t+1,bit=index,**compare(base['local_cells'][t]['bits'][b]['decoded'],tag['local_cells'][t]['bits'][b]['decoded'],message[index])))
        for phase in range(8):
            base=read(out/pay_path(('local',h,'P0',phase,'key00')));tag=read(out/pay_path(('local',h,'TAG',phase,'key00')))
            for t in range(1,12):
                frames=list(range(origins[h]+phase+4*t-3,origins[h]+phase+4*t+1));source_cells={(x+3)//4 for x in frames};full=len(source_cells)==1 and min(frames)>=1 and max(frames)<=176;cell=next(iter(source_cells)) if full else None
                for b in range(8):
                    index=8*((cell-1)//11)+b if cell is not None else None;target=message[index] if index is not None else None
                    local.append(dict(host=h,phase=phase,row=t,bit=b,kind='FULL' if full else 'MIXED',source_cell_post_only=cell,source_frames=frames,**compare(base['local_cells'][t-1]['bits'][b]['decoded'],tag['local_cells'][t-1]['bits'][b]['decoded'],target)))
    save(out/'tag_post_rows.json',observations);save(out/'local_final_post_rows.json',local_final);save(out/'family_post_rows.json',family_rows);save(out/'payload_group_pairs.json',group);save(out/'payload_cell_pairs.json',cells);save(out/'payload_local_pairs.json',local)
    flags=('correct_valid_RM','wrong_valid_RM','unsupported_RM','null_RM','correct_candidate_accepted','wrong_ID_candidate_accepted','unsupported_candidate_accepted','null_candidate_accepted','candidate_pilot_present','missing_valid_ID_candidate_accepted','partial_valid_ID_candidate_accepted')
    summary={}
    for r in observations:
        key='/'.join((r['view'],r['arm'],r['key'],r['geometry']['kind']));s=summary.setdefault(key,dict(planned=0,technical=0,**{f:0 for f in flags}));s['planned']+=1;s['technical']+=r['score_status']!='OBSERVED'
        for f in flags:s[f]+=bool(r[f])
    return dict(status='POST_TRUTH_UNRESOLVED' if issues else 'POST_DESCRIPTION_ONLY',issues=issues,tag_summary=summary,local_final=dict(planned=96,**{f:sum(bool(r[f]) for r in local_final) for f in ('correct','wrong_valid_ID','unsupported_accepted','null_accepted','missing_valid_ID_accepted','partial_valid_ID_accepted','missing_phase')}),families=dict(planned=24,**{f:sum(bool(r[f]) for r in family_rows) for f in ('correct_origin','wrong_origin','null_origin','technical_missing')}),local_ID_support_note='Geometry of the decoded nominal ID unit inside fragment only; FULL_PRESENT never establishes tag presence. Null arms/wrong keys remain null regardless of geometry.',local_ID_support_acceptance={kind:dict(material_positive_candidates=sum(r['view']=='local' and r['material_positive'] and r['candidate_accepted'] and r['accepted_ID_support_post_only']==kind for r in observations),null_candidates=sum(r['view']=='local' and not r['material_positive'] and r['candidate_accepted'] and r['accepted_ID_support_post_only']==kind for r in observations),material_positive_selected=sum(r['positive'] and r['accepted'] and r['accepted_ID_support_post_only']==kind for r in local_final),null_selected=sum(not r['positive'] and r['accepted'] and r['accepted_ID_support_post_only']==kind for r in local_final),raw_candidates=sum(r['view']=='local' and r['candidate_accepted'] and r['accepted_ID_support_post_only']==kind for r in observations),selected_final=sum(r['accepted'] and r['accepted_ID_support_post_only']==kind for r in local_final)) for kind in ('FULL_PRESENT','PARTIAL_PRESENT','ABSENT')},payload=dict(full_groups=pair_summary(group),full_cells=pair_summary(cells),local_raw=pair_summary(local),local_by_geometry={kind:pair_summary([r for r in local if r['kind']==kind]) for kind in ('FULL','MIXED')}),fixed=p.fixed(),scientific_PASS=None,independent_confirmation=False,false_positive_rate_claim=None,authentication=False,temporal_path_complete=False)
