"""Read-only paper tables from saved terminal/post records; never rerun an old post."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from main.tube_state import paper_suite_v1 as p
from main.baselines.public_compact12_v1 import OUTPUT as COMPACT_PUBLIC_ROOT
from . import paper_suite_records_v1 as rec


from .paper_suite_reporting_v1 import SavedReader,unique,fixed_positions,counts_for,payload_summary,query_tables


def build(cohort_root=p.COHORT_ROOT,wan_root=p.WAN_ROOT,query_root=p.QUERY_ROOT,
          public_root=COMPACT_PUBLIC_ROOT,report_root=p.REPORT_ROOT):
    from .b_paper_cohort_evaluation_v1 import cohort_summary
    folder=Path(report_root)/'reports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    reader=SavedReader();read=reader.load
    times=[];strata=[];payload=[];controls=[];qualities=[];costs=[];wan=[];source_families=[];component=[]
    for sid in p.SOURCES:
        source=Path(cohort_root)/sid/'attempt01';evaluation=read(source/'evaluation_summary.json')
        valid_post=read(source/'post_status.json').get('status')=='RETURNED'
        if not valid_post:evaluation={}
        nodes=read(source/'node_results.json',list) if valid_post else []
        edges=read(source/'edge_results.json',list) if valid_post else []
        for view,n in zip(p.VIEWS,p.LENGTHS):
            nr=[r for r in nodes if r.get('arm')=='NEW' and r.get('view')==view] if isinstance(nodes,list) else []
            er=[r for r in edges if r.get('arm')=='NEW' and r.get('view')==view] if isinstance(edges,list) else []
            nr=fixed_positions(nr,n,'received_frame');er=fixed_positions(er,n-1,'received_edge')
            for route,old in [('F42_saved',True),('relative_saved',False)]:
                times.append(dict(source=sid,view=view,route=route,planned_nodes=n,planned_edges=n-1,
                    nodes=counts_for(nr,old),edges=counts_for(er,old),recomputed_receiver=False))
                for dimension,field,values,items in [('node','startup',(True,False,None),nr),('edge','true_kind',('NORMAL','GAP','REPEAT',None),er)]:
                    for value in values:
                        subset=[r for r in items if r.get(field)==value]
                        strata.append(dict(source=sid,view=view,route=route,dimension=dimension,stratum_field=field,
                            stratum=value if value is not None else 'UNKNOWN',fixed_total=len(items),stratum_positions=len(subset),
                            counts=counts_for(subset,old),basis='saved post labels only; missing labels retained UNKNOWN'))
            absolute={}
            for strategy,offset in [('B_NEW',10),('U_NEW',10),('P_PAIRED',5)]:
                saved=read(source/'post'/f'input{offset+p.VIEWS.index(view)}'/(strategy+'.json'),list) if valid_post else []
                absolute[strategy]=payload_summary(saved,reader,source/'post'/f'input{offset+p.VIEWS.index(view)}'/(strategy+'.json'))
            payload.append(dict(source=sid,view=view,planned_B_slots=32,planned_U_slots=32,planned_P_PAIRED_slots=32,
                absolute=absolute,P_PAIRED_scope='borrowed NEW mapping; paired identity retention, not P0 blind recovery',
                comparisons=reader.obj(evaluation.get('views',{}),source/'evaluation_summary.json/views').get(view),
                status='SAVED_POST' if valid_post else 'TECHNICAL_UNKNOWN'))
        decisions=read(source/'decisions.json',list)
        if not isinstance(decisions,list):decisions=[]
        for arm in p.cohort.ARMS:
            for key in p.cohort.KEYS:
                for claim in p.cohort.CLAIMS:
                    group=[]
                    for view in p.VIEWS:
                        found=[r for r in decisions if r.get('arm')==arm and r.get('view')==view
                               and r.get('key_id')==key and r.get('claim_id')==claim]
                        row=found[0] if len(found)==1 else {}
                        verdict=row.get('verdict') if row.get('status')=='RETURNED' else 'INPUT_UNRESOLVED'
                        if verdict not in ('MATCH','MISMATCH','UNRESOLVED','INPUT_UNRESOLVED'):verdict='INPUT_UNRESOLVED'
                        group.append(verdict)
                        controls.append(dict(source=sid,arm=arm,view=view,key=key,claim=claim,verdict=verdict,
                            is_negative_identity_claim=arm=='OFF' or key=='key01' or claim=='q1',
                            unresolved_is_contradiction_rejection=False))
                    event=p.family_event([True if v=='MATCH' else None if v=='INPUT_UNRESOLVED' else False for v in group])
                    source_families.append(dict(source=sid,arm=arm,key=key,claim=claim,family='ANY5',event=event,
                        verdict_counts=dict(Counter(group)),is_negative_identity_claim=arm=='OFF' or key=='key01' or claim=='q1'))
        quality=read(source/'quality.json')
        for pair in ('NEW_P0','P0_OFF','NEW_OFF'):
            for view in p.VIEWS:
                qualities.append(dict(source=sid,method='main',pair=pair,view=view,
                    value=quality.get(pair+'/'+view) if isinstance(quality,dict) else None))
        costs.append(dict(source=sid,method='main',value=read(source/'chain_resources.json'),
            generation=read(source/'generation_resources.json'),identity=read(source/'identity_backend_resources.json'),
            stages={stage:{name:read(source/'stages'/stage/(name+'.json')) for name in ('resources','backend_resources','io_counts','result')}
                for stage in p.cohort.STAGES},scratch_release=read(source/'scratch_release.json'),
            operation_counts=read(source/'counts.json'),measurement_scope='saved measured fields only; missing measurements UNKNOWN'))
        main_result=read(source/'result.json')
        component.append(dict(component='main',source=sid,status=main_result.get('status','MISSING'),post_available=valid_post))
        base=Path(wan_root)/sid/'attempt01';result=read(base/'result.json')
        component.append(dict(component='wan_baselines',source=sid,status=result.get('status','MISSING'),post_count=result.get('posts_returned',0)))
        for method in p.METHODS:
            for arm in ('OFF','WM'):
                for view,n in zip(p.VIEWS,p.LENGTHS):
                    value=read(base/'post'/method/arm/(view+'.json'))
                    wan.append(dict(source=sid,method=method,arm=arm,view=view,planned_frames=n,
                        native_bits=32 if method=='wam' else 256,owner_bits=248 if method=='frameindex' else 32 if method=='wam' else 256,
                        status='SAVED_POST' if bool(value) else 'TECHNICAL_UNKNOWN',
                        result=value,time_method='ADAPTED_FRAME_INDEX' if method=='frameindex' else 'N/A',scope=p.COMPARISON_SCOPE))
        qrows=read(base/'quality.json',list)
        for method in p.METHODS:
            for view in p.VIEWS:
                matched=[row for row in qrows if row.get('method')==method and row.get('view')==view] if isinstance(qrows,list) else []
                qualities.append(dict(source=sid,method=method,view=view,pair='WM_OFF',value=matched[0] if len(matched)==1 else None))
        costs.append(dict(source=sid,method='wan_baselines',value=result,
            actual_network_hooks=read(base/'resources.json',list),operation_rows=read(base/'operations.json')))
    fixed_stats=[];fixed_panels=[]
    for panel,sids in [('ALL',p.SOURCES),('TEST-A',p.SOURCES[:4]),('TEST-B',p.SOURCES[4:])]:
        for arm in p.cohort.ARMS:
            for key in p.cohort.KEYS:
                for claim in p.cohort.CLAIMS:
                    group=[r for r in source_families if (r['arm'],r['key'],r['claim'])==(arm,key,claim) and r['source'] in sids]
                    row=dict(panel=panel,arm=arm,key=key,claim=claim,family='ANY5',**p.rate_summary([r['event'] for r in group]),
                        negative_identity_claim=group[0]['is_negative_identity_claim'],
                        verdict_counts=dict(sum((Counter(r['verdict_counts']) for r in group),Counter())),
                        interpretation='any of five accepts; not ALL5 robustness; nonacceptance includes unresolved')
                    fixed_panels.append(row)
                    if panel=='ALL':fixed_stats.append(row)
    wan_stats=[]
    for method in p.METHODS:
        for arm in ('OFF','WM'):
            for verification in ('ownership','wrong_query') if arm=='WM' else ('ownership',):
                events=[];collisions=[]
                for sid in p.SOURCES:
                    group=[r for r in wan if (r['source'],r['method'],r['arm'])==(sid,method,arm)]
                    values=[reader.obj((r['result'] or {}).get(verification,{}),f'wan/{sid}/{method}/{verification}') for r in group]
                    # A correct owner-query collision is N/A, never a false acceptance.
                    flags=[(r['result'] or {}).get('wrong_query_collision') for r in group]
                    collision=verification=='wrong_query' and any(v is True for v in flags)
                    collisions.append(collision)
                    decision=[v.get('decision') if type(v.get('decision')) is bool else None for v in values]
                    if verification=='wrong_query' and not collision and not all(v is False for v in flags):decision=[None]*5
                    events.append(p.family_event(decision))
                wan_stats.append(dict(method=method,arm=arm,verification=verification,family='ANY5',**p.rate_summary(events,collisions),
                    query_collisions=sum(collisions),
                    threshold_source='unchanged public v4 analytic rule, not tuned on Wan'))
    tables={'main_time':times,'main_payload_contributions':payload,'main_fixed_claims':controls,
        'main_fixed_source_statistics':fixed_stats,'main_fixed_panel_statistics':fixed_panels,'main_time_strata':strata,'wan_native_comparison':wan,'wan_source_statistics':wan_stats,
        'quality':qualities,'costs':costs,'components':component}
    tables.update(query_tables(query_root,reader))
    from main.tube_state.paper_fpr_claims_v1 import measurement_rows,protocol_rows
    tables['fpr_measurements']=measurement_rows(controls,tables['main_random_family'])
    tables['fpr_probability_models']=protocol_rows()
    public=read(Path(public_root)/'result.json')
    pb=read(Path(public_root)/'tables/B_ownership.json',list)
    tables['public_baseline_summary']=[r for r in pb if r.get('split')=='TEST' and r.get('channel') in ('FAMILY12','ALL12_ROBUST')]
    if not tables['public_baseline_summary']:tables['public_baseline_summary']=[dict(status='PENDING_OR_UNAVAILABLE',output=str(public_root))]
    query_status=read(Path(query_root)/'result.json')
    tables['read_errors']=reader.errors
    for name,rows in tables.items():rec.table(folder,name,rows)
    cohort=cohort_summary(cohort_root,'ALL');rec.write(folder/'twelve_source_summary.json',cohort)
    public_refs={name:str(Path(public_root)/'tables'/(name+suffix)) for name,suffix in (
        ('A_payload','.csv'),('B_ownership','.csv'),('C_time','.csv'),('D_quality','.csv'),('D_resources','.json'))}
    state=dict(status='SAVED_REPORT_SNAPSHOT',report=str(folder),source_denominator=12,
        public_baseline_status=public.get('status','PENDING'),public_tables=public_refs,
        old_public_v4_reference=p.PUBLIC_ROOT,old_public_v4_status_untouched=True,
        query_status=query_status,components=component,read_error_count=len(reader.errors),
        no_scientific_success_assumed=True,population_fpr_certified=False,
        fpr_increment=dict(derived_measurement_rows=48,added_scientific_calls=0,validation_sources=8,
            models_table=str(folder/'fpr_probability_models.csv'),measurements_table=str(folder/'fpr_measurements.csv')))
    rec.write(folder/'index.json',state);rec.write(Path(report_root)/'latest_report.json',state)
    text='# Formal paper measurement package\n\n'+p.COMPARISON_SCOPE+'\n\n'
    text+='Twelve fixed Wan sources. Existing C4/TEST-A evidence is retained; TEST-B is formal measurement. All failures, unresolved and missing rows remain.\n\n'
    text+='QMatch accepts only all32 matching valid bits; analytic random-query family risk and fixed-panel empirical rates are separate. No natural-population 0.1% certification. OFF/P0 empty paths are unresolved, not contradiction rejection.\n\n'
    text+='FPR increment:48 deterministic summaries, zero new scientific calls. TEST-B8 is the predeclared unobserved finite validation panel at increment design; TEST-A remains prior formal evidence. Random-query bounds do not certify fixed-identity population FPR. See fpr_probability_models and fpr_measurements.\n\n'
    text+='Public natural-video results remain a separate domain table. FrameIndex is ADAPTED, other native time columns are N/A. No equal-capacity, temporal-localization literature superiority, or perceptual superiority claim is added.\n\n'
    for name in tables:text+=f'- [{name}]({name}.csv)\n'
    text+='\nCompact public saved tables (old v4 remains a separate reference):\n'
    for name,path in public_refs.items():text+=f'- {name}: `{path}`\n'
    (folder/'REPORT.md').write_text(text)
    return state
