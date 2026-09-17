"""Deterministic development-only selection; missing rows never shrink a roster."""
from __future__ import annotations
import math

METRICS=('rgb_mse','residual_temporal_mse')


def flow_name(index,suffix):return f'FLOW_r{index:02d}_{suffix}'


def finite(value):return isinstance(value,(int,float)) and math.isfinite(value)


def select_terminal(manifest,case_results):
    records=[]
    for index,rho in enumerate(manifest['strengths']):
        rows=[]
        for case in manifest['development']:
            result=case_results.get(case['id']) or {}
            for suffix in ('A','B'):
                row=result.get('conditions',{}).get(flow_name(index,suffix),{})
                zero=result.get('conditions',{}).get('ZERO_'+suffix,{})
                margin=row.get('correct_minus_competitor_margin')
                steps=row.get('steps',[])
                valid=terminal_valid(row,zero)
                rows.append({'case':case['id'],'message':suffix,'status':row.get('status','MISSING'),
                    'eligible':valid,'loss':row.get('loss'),'zero_loss':zero.get('loss'),
                    'loss_change':row['loss']-zero['loss'] if finite(row.get('loss')) and finite(zero.get('loss')) else None,
                    'margin':margin,'positive_margin_screen_only':margin>0 if finite(margin) else None,
                    'signed_projection_gain':row.get('correct_code_signed_projection_gain')})
        valid=all(row['eligible'] for row in rows)
        records.append({'index':index,'rho':rho,'rows':rows,'denominator':2*len(manifest['development']),
            'status':'ELIGIBLE' if valid else 'INELIGIBLE_MISSING_FAILED_OR_BUDGET',
            'worst_margin':min(row['margin'] for row in rows) if valid else None,
            'mean_margin':sum(row['margin'] for row in rows)/len(rows) if valid else None})
    eligible=sorted((r for r in records if r['status']=='ELIGIBLE'),key=lambda r:(-r['worst_margin'],-r['mean_margin'],r['rho']))
    selected=eligible[:manifest['max_media_candidates']]
    for row in eligible:row['status']='SELECTED_FOR_MEDIA' if row in selected else 'NOT_SELECTED'
    curves=[]
    for row_index in range(2*len(manifest['development'])):
        points=[{'rho':record['rho'],**record['rows'][row_index]} for record in records]
        points.sort(key=lambda point:point['rho'])
        previous=None
        for point in points:
            change=point['loss_change']
            gain=-change if change is not None else None
            point['loss_gain']=gain
            point['incremental_gain_per_rho']=None
            point['gain_reversal_vs_previous']=None
            if previous is not None and gain is not None and previous['loss_gain'] is not None:
                point['incremental_gain_per_rho']=(gain-previous['loss_gain'])/(point['rho']-previous['rho'])
                point['gain_reversal_vs_previous']=gain<previous['loss_gain']
            previous=point
        curves.append({'case':points[0]['case'],'message':points[0]['message'],'points':points})
    return {'strength_curves':curves,'status':'CANDIDATES_SELECTED' if selected else 'NO_SELECTION','candidates':records,
            'selected_indices':[r['index'] for r in selected],'selected_rhos':[r['rho'] for r in selected],
            'claim':'terminal margin is a screen, not a blind readout or FPR result'}


def quality_ratio(flow,terminal):
    if not finite(flow) or not finite(terminal) or flow<0 or terminal<0:return None
    return flow/terminal if terminal>0 else (0. if flow==0 else None)


def evaluate_media_case(videos,index,case_id):
    """Shared fixed criteria for dev and holdout; no ranking or selection here."""
    rows=[]
    controls=all(videos.get(a,{}).get('status')=='COMPLETE' for a in ('OFF','TERMINAL_A','TERMINAL_B'))
    terminal_read=all(videos.get('TERMINAL_'+s,{}).get('five_methods_unique_correct') is True for s in ('A','B'))
    for suffix in ('A','B'):
        flow=videos.get(flow_name(index,suffix),{});terminal=videos.get('TERMINAL_'+suffix,{})
        ratios={k:quality_ratio(flow.get('saved_quality_vs_off',{}).get(k),terminal.get('saved_quality_vs_off',{}).get(k)) for k in METRICS}
        valid=(controls and terminal_read and flow.get('status')=='COMPLETE' and
               flow.get('five_methods_unique_correct') is True and all(v is not None and v<=1.5 for v in ratios.values()))
        rows.append({'case':case_id,'message':suffix,'status':flow.get('status','MISSING'),
            'eligible':valid,'quality_ratios':ratios,'five_modes_unique_correct':flow.get('five_methods_unique_correct'),
            'controls_complete':controls,'terminal_five_modes_unique_correct':terminal_read})
    return rows


def terminal_valid(row,zero):
    steps=row.get('steps',[])
    return (row.get('status')=='COMPLETE' and zero.get('status')=='COMPLETE' and
        finite(row.get('correct_minus_competitor_margin')) and finite(row.get('loss')) and finite(zero.get('loss')) and
        len(steps)==3 and all(s.get('step_within_budget') is True and
        s.get('sum_within_budget') is True and s.get('response_valid') is True for s in steps))


def evaluate_holdout_case(terminal,media,case_id):
    """Evaluate the one frozen rho at local index 0; never calls a selector."""
    rows=evaluate_media_case((media or {}).get('videos',{}),0,case_id)
    conditions=(terminal or {}).get('conditions',{})
    for evaluation in rows:
        suffix=evaluation['message'];row=conditions.get(flow_name(0,suffix),{})
        zero=conditions.get('ZERO_'+suffix,{})
        valid=terminal_valid(row,zero)
        evaluation.update(terminal_status=row.get('status','MISSING'),terminal_eligible=valid,
            terminal_loss=row.get('loss'),zero_loss=zero.get('loss'),
            competition_margin=row.get('correct_minus_competitor_margin'),
            signed_projection_gain=row.get('correct_code_signed_projection_gain'),
            control_steps=row.get('steps',[]),fixed_criteria_met=valid and evaluation['eligible'])
    return rows


def select_media(manifest,terminal_selection,media_results):
    records=[]
    for index in terminal_selection['selected_indices']:
        rows=[]
        for case in manifest['development']:
            videos=(media_results.get(case['id']) or {}).get('videos',{})
            rows.extend(evaluate_media_case(videos,index,case['id']))
        valid=all(row['eligible'] for row in rows)
        ratios=[v for row in rows for v in row['quality_ratios'].values()]
        records.append({'index':index,'rho':manifest['strengths'][index],'rows':rows,'denominator':2*len(manifest['development']),
            'status':'ELIGIBLE' if valid else 'INELIGIBLE_READOUT_QUALITY_OR_MISSING',
            'worst_quality_ratio':max(ratios) if valid else None,'mean_quality_ratio':sum(ratios)/len(ratios) if valid else None})
    eligible=sorted((r for r in records if r['status']=='ELIGIBLE'),key=lambda r:(r['worst_quality_ratio'],r['mean_quality_ratio'],r['rho']))
    chosen=eligible[0] if eligible else None
    for row in eligible:row['status']='SELECTED' if row is chosen else 'NOT_SELECTED'
    return {'status':'SELECTED' if chosen else 'NO_SELECTION','rho':chosen['rho'] if chosen else None,
            'candidates':records,'holdout_used_for_selection':False,
            'claim':'frozen development choice only; quality rule is not a validated perceptual threshold; rank is not FPR'}
