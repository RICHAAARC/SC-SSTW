"""Fixed existing-JSON CPU development read. No physical observation entry point."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
from main.tube_state import b_rank3r_adjacent_origin_v1 as candidate
from main.tube_state import b_rank3_four_source_confirmation_v1 as original
from main.tube_state import b_cdc32_aggregation_v1 as cdc
from runtime.wan.b_gamma2_fixed_validation_v1 import read_bit_header
from runtime.wan.b_rank3_four_source_confirmation_v1 import dump,view_metadata,flatten_cache,empty_slots
from runtime.wan.b_rank3r_complete_repeat_evaluation_v1 import assess
from runtime.wan.b_rank3_four_source_confirmation_evaluation_v1 import paired

WINDOWS=candidate.WINDOWS
ARMS=('OFF','P0','PM');KEYS=('key00','key01')
CROP_DELETE=('CROP','DELETE_LEFT_GAP','DELETE_RIGHT_GAP')
REPEAT=('REPEAT_LEFT_GAP','REPEAT_RIGHT_GAP')
GROUPS={
 'stage1':dict(hosts=('b2-m3-c01','b2-m3-c02','b2-m3-e01','b2-m3-e02'),fixtures=CROP_DELETE,read_hosts=('b2-m3-e01','b2-m3-e02')),
 'stage2':dict(hosts=tuple('b-rank3-n0'+str(i) for i in range(1,5)),fixtures=CROP_DELETE),
 'm2':dict(hosts=tuple('b-rank3-n0'+str(i) for i in range(1,5)),fixtures=REPEAT),
 'm3':dict(hosts=tuple('b-rank3r-m0'+str(i) for i in range(1,5)),fixtures=CROP_DELETE+REPEAT)}
DENOMINATORS=dict(detection_tasks=312,positive_detection=52,null_detection=260,read_tasks=276,
 positive_read=46,null_read=230,read_slots=8832,positive_slots=1472,null_slots=7360,
 stage1_cal_detection_only=36,wrong_key_PM_detection=52,wrong_key_PM_read=46,
 H137_families=936,candidate_score_slots=128232,membership_limit=1248,all_phase_bit_files_limit=3312,
 model_VAE_codec_E_Q_DCT_bit_recomputations=0)


def roster():
    rows=[]
    for group,spec in GROUPS.items():
        for host in spec['hosts']:
            for arm in ARMS:
                for fixture in spec['fixtures']:
                    for key in KEYS:
                        rows.append(dict(group=group,task='/'.join((host,arm,fixture,key)),key=key,
                            read_bits=host in spec.get('read_hosts',spec['hosts'])))
    return rows


def read(path):
    def bad(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=bad)


def resolve(group,value,catalog_directory):
    if not isinstance(value,str) or not value:raise ValueError('readable local file path required')
    p=Path(value)
    if p.is_absolute():return p
    root=Path(group.get('root','.'))
    if not root.is_absolute():root=catalog_directory/root
    return root/p


class Cache:
    def __init__(self):self.addresses={};self.headers={};self.loaded={};self.failures={};self.enabled=False;self.accesses=0
    def register(self,path,key,view):
        path=str(path)
        if path in self.headers and self.headers[path]!=(key,view):raise ValueError('contradictory file view/key metadata')
        self.headers[path]=(key,view)
        for row in range(1,11):
            for region in ('R0','R1'):
                address=(f'{view}/row{row}/{region}',key)
                if address in self.addresses and self.addresses[address]!=str(path):raise ValueError('contradictory physical evidence address')
                self.addresses[address]=str(path)
    def __call__(self,evidence,key,bit):
        if not self.enabled:raise RuntimeError('payload before all fixed plans were saved and read back')
        self.accesses+=1;path=self.addresses[evidence,key]
        if path in self.failures:raise ValueError(self.failures[path])
        if path not in self.loaded:
            try:
                data=read(path);expected_key,view=self.headers[path]
                if (data.get('status')!='AVAILABLE' or data.get('kind')!='E41' or data.get('key')!=expected_key
                    or data.get('view_id')!=view or data.get('local_rows')!=list(range(1,11))):
                    raise ValueError('actual E41 header/view/key/rows mismatch')
                self.loaded[path]=data
            except Exception as error:
                self.failures[path]=f'{type(error).__name__}: {error}';raise
        data=self.loaded[path];view=data['view_id']
        suffix=evidence[len(view)+4:];row_text,region=suffix.split('/')
        row_number=int(row_text)
        cells=data.get('regions',{}).get(region,{}).get('local_cells',[])
        byrow={c.get('local_row'):c for c in cells}
        if len(byrow)!=len(cells):raise ValueError('duplicate local cell address')
        cell=byrow.get(row_number,{})
        slc={'R0':[8,24,8,24],'R1':[8,24,40,56]}[region]
        if (cell.get('status')!='AVAILABLE' or cell.get('evidence_id')!=evidence
            or cell.get('observed_region_id')!=region or cell.get('carrier_region_id')!=region
            or cell.get('observed_region_slice')!=slc or cell.get('channels')!=[0,4]
            or cell.get('coordinate_space')!='normalized_latent' or cell.get('diagnostic_only') is not False):
            raise ValueError('actual requested cell/region/availability mismatch')
        bits=cell.get('bits',[]);bybit={v.get('local_bit'):v for v in bits}
        if len(bybit)!=len(bits):raise ValueError('duplicate bit address')
        value=bybit.get(bit,{})
        if (value.get('key')!=key or value.get('evidence_id')!=evidence
            or value.get('status')!='AVAILABLE' or value.get('planned_count')!=8
            or value.get('returned_count')!=8 or value.get('count')!=8
            or value.get('diagnostic_only') is not False):
            raise ValueError('actual requested bit key/evidence/availability mismatch')
        return value



def _address_locals(saved,reader):
    if 'readers' in saved:saved=saved['readers'][reader]
    locals_=saved['locals']
    if isinstance(locals_,dict):return locals_
    return {p['window']:p for p in locals_}


def views_and_headers(group,entry,row,catalog_directory,cache):
    """Only leading non-value bit metadata; old plan is used for anonymous IDs only."""
    errors=[];views={};addresses={}
    if 'address_plan' in entry:
        try:addresses=_address_locals(read(resolve(group,entry['address_plan'],catalog_directory)),entry.get('plan_reader','RANK3R'))
        except Exception as error:errors.append(dict(kind='address_plan',error=str(error)))
    for window,w in WINDOWS.items():
        supplied=entry.get('views',{}).get(window,[])
        by_phase={v.get('phase'):v for v in supplied if isinstance(v,dict)}
        address=addresses.get(window,{})
        views[window]=[]
        for phase in range(4):
            meta=by_phase.get(phase,{})
            input_id=meta.get('input_id',address.get('input_id'));query_id=meta.get('query_id',address.get('query_id'))
            valid_ids=all(isinstance(v,str) and v for v in (input_id,query_id))
            # Missing addresses stay explicitly unresolved; placeholders do not assert ownership.
            if not valid_ids:input_id='unresolved-input';query_id='unresolved-query-'+window
            status='MISSING_VIEW';view='unresolved-'+window+'-'+str(phase)
            try:
                filename=entry['bits'][window][str(phase)];path=resolve(group,filename,catalog_directory)
                header=read_bit_header(path)
                if header.get('status')!='AVAILABLE' or header.get('kind')!='E41' or header.get('key')!=row['key'] or header.get('local_rows')!=list(range(1,11)):raise ValueError('invalid E41 bit header')
                view=header.get('view_id')
                if not isinstance(view,str) or not view:raise ValueError('nonempty actual view_id required')
                if meta.get('view_id') not in (None,view):raise ValueError('catalog view_id disagrees with actual metadata')
                if not valid_ids:raise ValueError('missing anonymous input/query address')
                if meta.get('received_query_start',w)!=w or meta.get('encoder_input_support',list(range(w+phase,w+phase+41)))!=list(range(w+phase,w+phase+41)):raise ValueError('invalid declared phase/window support')
                cache.register(path,row['key'],view);status='RETURNED'
            except Exception as error:errors.append(dict(window=window,phase=phase,error=f'{type(error).__name__}: {error}'))
            views[window].append(view_metadata(input_id,query_id,phase,w,view,status))
    ids=[v['view_id'] for values in views.values() for v in values if v['status']=='RETURNED']
    if len(ids)!=len(set(ids)):
        errors.append(dict(kind='ownership',error='distinct phase/windows cannot share one view_id'))
        for values in views.values():
            for view in values:view['status']='UNRESOLVED_EVIDENCE_OWNERSHIP'
    return views,errors


def _decision(packet,cache):
    if not packet.get('locals'):
        return dict(selection=packet['selection'],aggregate=empty_slots('TECHNICAL_UNRESOLVED'),technical_errors=packet.get('read_errors',[]))
    memo={};locals_=[cdc.read_local(p,cache,memo=memo) for p in packet['locals']]
    aggregate=cdc.aggregate(locals_,path_context=packet['selection'])
    aggregate['mapping_scope']='RANK3R_SINGLETON_C' if packet['correction']['selected_window'] is None else candidate.NAME+'_SINGLETON_C'
    aggregate['selection_status']=packet['selection']['status'];aggregate['detection_status']=packet['selection']['status']
    return dict(selection=packet['selection'],aggregate=aggregate,correction=packet['correction'],
        original_branch_preserved=packet['correction']['status']=='ORIGINAL_BRANCH_PRESERVED',cache_accesses=cache.accesses)


def _existing(output):
    saved={};errors={}
    for name in ('result.json','ledger.json','planned_outputs.json','evaluation.json'):
        try:saved[name]=read(output/name)
        except Exception as error:errors[name]=str(error)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_calls=0,saved=saved,read_errors=errors)


def run(catalog_path,output,baseline_path=None):
    """One fixed CPU attempt; missing tasks remain unresolved, never shrink roster."""
    output=Path(output)
    if output.exists() and any(output.iterdir()):return _existing(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'attempt.json').open('x') as stream:json.dump(dict(status='STARTED',attempts=1,physical_calls=0,started_unix=time.time()),stream);stream.flush()
    fixed=roster();ledger={f"plan/{r['group']}/{r['task']}":dict(status='NOT_ENTERED',attempts=0) for r in fixed}
    ledger.update({f"read/{r['group']}/{r['task']}":dict(status='NOT_ENTERED',attempts=0) for r in fixed if r['read_bits']})
    ledger.update({f'post/{g}':dict(status='NOT_ENTERED',attempts=0) for g in GROUPS})
    outputs={r['group']+'/'+r['task']:dict(status='NOT_ENTERED',read_bits=r['read_bits'],slots=empty_slots()['slots'] if r['read_bits'] else []) for r in fixed}
    dump(output/'plan.json',dict(roster=fixed,denominators=DENOMINATORS,catalog_path=str(catalog_path)));dump(output/'ledger.json',ledger);dump(output/'planned_outputs.json',outputs)
    def call(slot,function):
        ledger[slot]=dict(status='STARTED',attempts=1);dump(output/'ledger.json',ledger)
        try:value=function()
        except BaseException as error:
            ledger[slot].update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}');dump(output/'ledger.json',ledger);raise
        ledger[slot]['status']='RETURNED';dump(output/'ledger.json',ledger);return value
    caches={g:Cache() for g in GROUPS};catalog_directory=Path(catalog_path).resolve().parent;catalog_errors=[];stop=None
    try:
        try:catalog=read(catalog_path)
        except Exception as error:catalog={};catalog_errors.append(str(error))
        groups=catalog.get('groups',{}) if isinstance(catalog,dict) else {}
        if set(groups)!=set(GROUPS):catalog_errors.append('catalog must address exactly stage1/stage2/m2/m3; fixed roster retained')
        for row in fixed:
            group=groups.get(row['group'],{});entry=group.get('tasks',{}).get(row['task'],{})
            def prepare(row=row,group=group,entry=entry):
                families={};errors=[]
                for window in WINDOWS:
                    try:raw=read(resolve(group,entry['families'][window],catalog_directory))
                    except Exception as error:raw={};errors.append(dict(window=window,kind='H137',error=f'{type(error).__name__}: {error}'))
                    families[window]=original.check_family(raw)
                selection=candidate.select(families)
                packet=dict(**selection,locals=[],read_errors=errors,read_bits=row['read_bits'],
                    family_status={w:{k:f[k] for k in ('status','issues','missing','unexpected','submitted')} for w,f in families.items()})
                if row['read_bits'] and selection['selection']['status']!='TECHNICAL_UNRESOLVED':
                    try:
                        views,header_errors=views_and_headers(group,entry,row,catalog_directory,caches[row['group']])
                        packet.update(candidate.plan_task(row['key'],families,views,selection));packet['header_errors']=header_errors
                    except Exception as error:packet['read_errors'].append(dict(kind='plan',error=f'{type(error).__name__}: {error}'))
                dump(output/'blind'/row['group']/row['task']/'plan.json',packet)
                return packet
            try:
                call(f"plan/{row['group']}/{row['task']}",prepare)
                if not row['read_bits']:outputs[row['group']+'/'+row['task']]['status']='DETECTION_ONLY_RECORDED'
            except Exception as error:
                packet=dict(selection=dict(status='TECHNICAL_UNRESOLVED',detected=None),original_selection={},locals=[],
                    correction=dict(status='TECHNICAL_UNRESOLVED',selected_window=None),membership_checks=0,read_errors=[str(error)],read_bits=row['read_bits'])
                dump(output/'blind'/row['group']/row['task']/'plan.json',packet)
        for row in fixed:read(output/'blind'/row['group']/row['task']/'plan.json')
        dump(output/'plans_complete.json',dict(status='READ_BACK',detection_plans=312,read_plans=276,payload_values_read=False))
        for cache in caches.values():cache.enabled=True
        for row in fixed:
            if not row['read_bits']:continue
            path=output/'blind'/row['group']/row['task']
            def consume(path=path,row=row):
                decision=_decision(read(path/'plan.json'),caches[row['group']]);dump(path/'decision.json',decision);return decision
            try:value=call(f"read/{row['group']}/{row['task']}",consume)
            except Exception as error:
                value=dict(selection=read(path/'plan.json')['selection'],aggregate=empty_slots('TECHNICAL_UNRESOLVED'),technical_errors=[str(error)])
                dump(path/'decision.json',value)
            outputs[row['group']+'/'+row['task']]=dict(status=ledger[f"read/{row['group']}/{row['task']}"]['status'],read_bits=True,slots=value['aggregate']['slots'])
        for row in fixed:
            if row['read_bits']:read(output/'blind'/row['group']/row['task']/'decision.json')
        dump(output/'blind_complete.json',dict(status='READ_BACK',decisions=276,truth_accessed=False,
            payload_files_loaded=sum(len(c.loaded) for c in caches.values()),payload_files_failed=sum(len(c.failures) for c in caches.values()),cache_accesses=sum(c.accesses for c in caches.values())))
        dump(output/'post_truth_access.json',dict(opened_unix=time.time(),all_decisions_read_back=True))
        evaluation=evaluate(output,groups,catalog_directory,call,catalog_errors,baseline_path)
        dump(output/'evaluation.json',evaluation)
    except BaseException as error:stop=dict(type=type(error).__name__,message=str(error),interrupted=not isinstance(error,Exception),retry=False)
    dump(output/'ledger.json',ledger);dump(output/'planned_outputs.json',outputs)
    result=dict(status='COMPLETED_FIXED_CPU_ATTEMPT' if stop is None else 'TECHNICAL_UNRESOLVED',stop=stop,
        denominators=DENOMINATORS,ledger_counts=dict(Counter(r['status'] for r in ledger.values())),physical_calls=0,
        catalog_errors=catalog_errors,output=str(output),original_M3_conclusion_unchanged=True)
    dump(output/'result.json',result);return result


def _saved_reader(evaluation,task,reader):
    tasks=evaluation.get('tasks',{})
    value=tasks.get(reader+'/'+task) if reader+'/'+task in tasks else tasks.get(task,{}).get('readers',{}).get(reader)
    if not isinstance(value,dict) or len(value.get('slots',[]))!=32:raise ValueError('saved control fixed32 missing: '+reader+'/'+task)
    if any(type(v.get('actual_supported_correct')) is not bool for v in value['slots']):raise ValueError('saved actual-support flags missing')
    return value


def _frames(config,task):
    host,_,fixture,_=task.split('/');a=next(h['a'] for h in config['sources'] if h['id']==host)
    if type(a) is not int or a not in range(31):raise ValueError('invalid saved source start')
    d=0 if fixture=='CROP' else -8 if fixture.startswith('REPEAT') else 8
    seam=49 if fixture.endswith('LEFT_GAP') else 102
    return [a+u+(d if u>=seam else 0) for u in range(151)]


def _technical(raw,selection):
    return selection.get('status')=='TECHNICAL_UNRESOLVED' or any(v.get('missing_refs') or any(s in v.get('status','') for s in ('TECHNICAL','UNRESOLVED','UNSUPPORTED','NOT_ENTERED')) for v in raw.get('slots',[]))


def evaluate(output,groups,catalog_directory,call,catalog_errors,baseline_path=None):
    if read(output/'blind_complete.json').get('decisions')!=276:raise ValueError('fixed276 blind decisions required')
    fixed=roster();results={};all_tasks={};errors=[];m3=None
    for group_id,spec in GROUPS.items():
        group=groups.get(group_id,{})
        def one_group():
            cfg=read(resolve(group,group['protocol'],catalog_directory))['config'];truth=cfg['writer']['bits']
            if len(truth)!=32 or any(type(x) is not int or x not in (0,1) for x in truth):raise ValueError('saved fixed32 truth required')
            historical=read(resolve(group,group['evaluation'],catalog_directory));tasks={};det=Counter();preservation=[];pairs={};technical=[]
            for row in [r for r in fixed if r['group']==group_id]:
                task=row['task'];host,arm,fixture,key=task.split('/');positive=arm=='PM' and key=='key00'
                packet=read(output/'blind'/group_id/task/'plan.json');selection=packet['selection'];scope='positive' if positive else 'null'
                det[scope+'/'+selection['status']]+=1
                if arm=='PM' and key=='key01':det['wrong_key_PM/'+selection['status']]+=1
                if not row['read_bits']:continue
                decision=read(output/'blind'/group_id/task/'decision.json');raw=decision['aggregate']
                value=assess(raw,truth,positive,_frames(cfg,task));value['detection_status']=selection['status']
                value['technical_detail_slots']=[i for i,v in enumerate(raw['slots']) if v.get('missing_refs') or any(s in v['status'] for s in ('TECHNICAL','UNRESOLVED','UNSUPPORTED','NOT_ENTERED'))]
                readers={'CANDIDATE':value};controls=('RANK3','RANK3R') if group_id=='m2' else ('CDC_M','RANK3') if group_id in ('stage1','stage2') else ('CDC_M','RANK3','RANK3R')
                for control in controls:
                    readers[control]=deepcopy(_saved_reader(historical,task,control))
                    readers[control].setdefault('technical_detail_slots',[])
                    if positive:pairs.setdefault(control,{})[task]=paired(readers[control],value)
                tasks[task]=dict(positive=positive,source=host,fixture=fixture,readers=readers,correction=packet['correction'])
                if _technical(raw,selection) or packet.get('read_errors'):technical.append(task)
                if packet['original_selection'].get('detected') is True:
                    preservation.append(dict(task=task,selection_unchanged=selection==packet['original_selection'],
                        untouched_original_plan_and_read_chain=decision.get('original_branch_preserved') is True,
                        scope='current original RANK3R branch; stage1/2 did not historically store RANK3R'))
            totals={}
            for scope in ('positive','null','wrong_key_PM'):
                selected=[t for name,t in tasks.items() if (t['positive'] if scope=='positive' else not t['positive'] if scope=='null' else name.split('/')[1]=='PM' and name.endswith('/key01'))]
                counts=[t['readers']['CANDIDATE']['counts'] for t in selected]
                totals[scope]=dict(planned_tasks=len(selected),planned_slots=32*len(selected),counts={k:sum(c[k] for c in counts) for k in counts[0]} if counts else {})
            return dict(status='POST_RECORDED',tasks=tasks,detection=dict(det),read_totals=totals,technical_tasks=technical,
                original_success_preservation=preservation,comparisons=pairs,
                CDC_M_scope='NOT_AVAILABLE_IN_SAVED_M2; no extra CDC read' if group_id=='m2' else 'saved historical control',
                historical_status={k:historical.get(k) for k in ('status','method_status','contribution_status')})
        try:results[group_id]=call('post/'+group_id,one_group)
        except Exception as error:
            results[group_id]=dict(status='TECHNICAL_UNRESOLVED',error=str(error),unresolved_tasks={r['task']:dict(planned_slots=32,status='TECHNICAL_UNRESOLVED') for r in fixed if r['group']==group_id and r['read_bits']});errors.append(group_id+':'+str(error))
        for task,value in results[group_id].get('tasks',{}).items():all_tasks[group_id+'/'+task]=value
    # Detection uses every fixed plan, including calibration tasks and missing nulls.
    detection={};membership=0;corrections=Counter();technical_detection=[]
    for row in fixed:
        packet=read(output/'blind'/row['group']/row['task']/'plan.json');positive=row['task'].split('/')[1]=='PM' and row['key']=='key00'
        scope='positive' if positive else 'null';selection=packet['selection']
        detection.setdefault(row['group'],{}).setdefault(scope,Counter())[selection['status']]+=1
        if row['task'].split('/')[1]=='PM' and row['key']=='key01':detection[row['group']].setdefault('wrong_key_PM',Counter())[selection['status']]+=1
        if selection['status']=='TECHNICAL_UNRESOLVED':technical_detection.append(row['group']+'/'+row['task'])
        membership+=packet['membership_checks'];corrections[packet['correction']['status']]+=1
    if results['m3'].get('status')=='POST_RECORDED':
        from runtime.wan.b_rank3r_new_source_confirmation_evaluation_v1 import summarize
        m3tasks={task:dict(row,readers=dict(RANK3R=row['readers']['CANDIDATE'],RANK3=row['readers']['RANK3'],CDC_M=row['readers']['CDC_M'])) for task,row in results['m3']['tasks'].items()}
        m3=summarize(m3tasks,not results['m3']['technical_tasks'])
    counts=[t['readers']['CANDIDATE']['counts'] for t in all_tasks.values()]
    totals={k:sum(c[k] for c in counts) for k in counts[0]} if counts else {}
    null_outputs=sum(t['readers']['CANDIDATE']['counts']['emitted'] for t in all_tasks.values() if not t['positive'])
    null_detect=sum(v.get('null',{}).get('DETECT',0) for v in detection.values())
    preservation=[r for v in results.values() for r in v.get('original_success_preservation',[])]
    complete=not(catalog_errors or errors or technical_detection or any(v.get('technical_tasks') for v in results.values())) and len(all_tasks)==276
    checks=dict(required_saved_inputs_complete=complete,null_detect_zero=null_detect==0,null_output_zero=null_outputs==0,
        literal_errors_zero=totals.get('literal_errors',0)==0,false_attribution_zero=totals.get('false_attribution',0)==0,
        unsupported_positive_outputs_zero=totals.get('unsupported_positive_outputs',0)==0,
        original_success_branch_preserved=all(r['selection_unchanged'] and r['untouched_original_plan_and_read_chain'] for r in preservation),
        M3_method_criteria_met=bool(m3 and m3['method_status']=='M3_METHOD_CRITERIA_MET'),
        M3_contribution_criteria_met=bool(m3 and m3['contribution_status']=='M3_CONTRIBUTION_CRITERIA_MET'))
    # Fixed ADJ criteria, post-only. The old failed candidate is used solely to
    # compare its untouched original-DETECT branches; its corrected outputs are not controls.
    preserved=[];preservation_errors=[]
    baseline=Path(baseline_path) if baseline_path is not None else None
    for row in fixed:
        task=row['task'];group=row['group']
        packet=read(output/'blind'/group/task/'plan.json')
        if packet['original_selection'].get('detected') is not True:continue
        item=dict(group=group,task=task,read_bits=row['read_bits'],
            selection_unchanged=packet['selection']==packet['original_selection'],
            plan_unchanged=None,aggregate_unchanged=None,actual_support_unchanged=None)
        try:
            if baseline is None:raise ValueError('saved untouched original-branch reference required')
            old_packet=read(baseline/'blind'/group/task/'plan.json')
            item['old_branch_was_original']=old_packet['correction']['status']=='ORIGINAL_BRANCH_PRESERVED'
            item['selection_unchanged'] &= packet['selection']==old_packet['selection']
            if row['read_bits']:
                item['plan_unchanged']=packet['locals']==old_packet['locals']
                old_decision=read(baseline/'blind'/group/task/'decision.json')
                decision=read(output/'blind'/group/task/'decision.json')
                item['aggregate_unchanged']=decision['aggregate']==old_decision['aggregate']
                # Identical actual references and values under the same source map imply
                # identical support; compare saved assessed slot rows as well.
                if 'baseline_evaluation' not in locals():baseline_evaluation=read(baseline/'evaluation.json')
                old_slots=baseline_evaluation['groups'][group]['tasks'][task]['readers']['CANDIDATE']['slots']
                new_slots=results[group]['tasks'][task]['readers']['CANDIDATE']['slots']
                item['actual_support_unchanged']=new_slots==old_slots
        except Exception as error:
            item['error']=f'{type(error).__name__}: {error}';preservation_errors.append(item)
        preserved.append(item)
    positive_detect=sum(v.get('positive',{}).get('DETECT',0) for v in detection.values())
    positive_support=sum(t['readers']['CANDIDATE']['counts']['actual_supported_correct'] for t in all_tasks.values() if t['positive'])
    unsupported=sum(s['value'] is not None and not s['actual_supported_correct']
                    for t in all_tasks.values() for s in t['readers']['CANDIDATE']['slots'])
    m04={}
    for fixture in ('DELETE_LEFT_GAP','REPEAT_RIGHT_GAP'):
        task='b-rank3r-m04/PM/'+fixture+'/key00'
        m04[fixture]=results['m3'].get('tasks',{}).get(task,{}).get('readers',{}).get('CANDIDATE',{}).get('counts',{})
    conflicts=[]
    for fixture in CROP_DELETE:
        task='b2-m3-e02/PM/'+fixture+'/key00'
        slots=results['stage1'].get('tasks',{}).get(task,{}).get('readers',{}).get('CANDIDATE',{}).get('slots',[])
        conflicts.append(dict(task=task,slot=30,retained=len(slots)==32 and slots[30]['value'] is None and 'CONFLICT' in slots[30]['status']))
    checks.update(positive_detection_52_of_52=positive_detect==52,
        actual_support_1469_of_1472=positive_support==1469,
        M3_detection_20_of_20=detection.get('m3',{}).get('positive',{}).get('DETECT',0)==20,
        M3_actual_support_640_of_640=results['m3'].get('read_totals',{}).get('positive',{}).get('counts',{}).get('actual_supported_correct')==640,
        m04_both_32_of_32=all(v.get('actual_supported_correct')==32 for v in m04.values()),
        no_actual_unsupported_output=unsupported==0,
        original_50_detect_selections_unchanged=len(preserved)==50 and all(r['selection_unchanged'] and r.get('old_branch_was_original') for r in preserved),
        original_read_plans_and_support_unchanged=not preservation_errors and all(r['plan_unchanged'] and r['aggregate_unchanged'] and r['actual_support_unchanged'] for r in preserved if r['read_bits']),
        stage1_three_slot30_conflicts_retained=all(r['retained'] for r in conflicts))
    complete=complete and not preservation_errors
    checks['required_saved_inputs_complete']=complete
    wrong_key_all_detection={g:dict(Counter(read(output/'blind'/r['group']/r['task']/'plan.json')['selection']['status']
        for r in fixed if r['group']==g and r['key']=='key01')) for g in GROUPS}
    wrong_key_all_read=[t for name,t in all_tasks.items() if name.endswith('/key01')]
    wrong_key_all=dict(detection_tasks=156,detection=wrong_key_all_detection,read_tasks=138,slots=4416,
        emitted=sum(t['readers']['CANDIDATE']['counts']['emitted'] for t in wrong_key_all_read),
        PM_subset=dict(detection_tasks=52,read_tasks=46,slots=1472))
    return dict(status='POST_RECORDED',candidate_status='TECHNICAL_UNRESOLVED' if not complete else 'FIXED_DEVELOPMENT_CRITERIA_MET' if all(checks.values()) else 'CANDIDATE_NOT_SUPPORTED',
        checks=checks,groups=results,detection=detection,M3_separate=m3,totals=totals,
        original_50_preservation=preserved,preservation_errors=preservation_errors,m04=m04,expected_conflicts=conflicts,
        wrong_key_all=wrong_key_all,actual_unsupported_outputs=unsupported,
        actual_membership_checks=membership,correction_categories=dict(corrections),technical_detection_tasks=technical_detection,
        post_errors=errors,denominators=DENOMINATORS,physical_calls=0,original_M3_conclusion_unchanged=True,
        population_FPR=None,new_source_confirmation=False,automatic_next_candidate=False,
        claim_ceiling='Seen correlated and calibration records only; fixed CPU development evidence, no new physical observations or population FPR bound')


def compact(result,output):
    try:evaluation=read(Path(output)/'evaluation.json')
    except Exception:evaluation={}
    return dict(status=result['status'],candidate_status=evaluation.get('candidate_status'),checks=evaluation.get('checks'),
        denominators=DENOMINATORS,correction_categories=evaluation.get('correction_categories'),
        membership_checks=evaluation.get('actual_membership_checks'),physical_calls=0,output=str(output))
