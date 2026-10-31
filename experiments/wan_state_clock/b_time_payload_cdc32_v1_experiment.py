"""One consumed c01 received-media read; all plans persist before E41/cache access."""
from pathlib import Path
import json
import time
from . import b_time_payload_cdc32_v1_records as records


def _empty(plan, reason):
    return dict(input_id=plan['input_id'],query_id=plan['query_id'],key_id='key00',mapping_scope=plan['mapping_scope'],
        status='NOT_ENTERED',existence_test='NOT_EVALUATED',source_authenticated=False,planned_slots=32,
        slots=[dict(slot=8*g['group']+b,group=g['group'],bit=b,value=None,status='NOT_ENTERED',cause_set=[reason],
                    selected_refs=g['references'],supporting_refs=[],missing_refs=g['references'],conflict_refs=[],
                    technical_failure=True,conditional_candidate_ids=[],failure_details={}) for g in plan['groups'] for b in range(8)],
        used_evidence_ids=[],counts=dict(planned=32,returned=0,failed=0,not_entered=32))


def _empty_final(input_id, arm, reason):
    return dict(input_id=input_id,arm=arm,status='NOT_ENTERED',planned_slots=32,existence_test='NOT_EVALUATED',source_authenticated=False,
        slots=[dict(slot=i,value=None,status='NOT_ENTERED',cause_set=[reason],selected_refs=[],supporting_refs=[],missing_refs=[],conflict_refs=[]) for i in range(32)])


def _flatten(decoded):
    result={}
    for region in ('R0','R1'):
        for row in decoded['regions'][region]['local_cells']:
            result.setdefault(row['evidence_id'],{})[decoded['key']]={str(b['local_bit']):b for b in row['bits']}
    return result


def run(config,output=None,backend_factory=None,evaluator=None):
    output=Path(output or config['output']).expanduser().resolve()
    state=records.output_state(output)
    if state=='CONSUMED':return records.recover(output)
    if state=='CONFLICT':return dict(status='OUTPUT_CONFLICT',physical_calls_this_invocation=0)
    check=records.preflight(config,output)
    if check['status']!='PREFLIGHT_READY':return check
    output.mkdir(parents=True,exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED',budgets=records.budgets(),plan=records.plan(),
                denominators=records.denominators(),limits=records.LIMITS,source=str(config['source']),
                recovery_layout=records.recovery_layout(),rerun='READ_ONLY',scope='same-chain development'),stream,indent=2)
    except FileExistsError:return records.recover(output)
    journal=records.Journal(output);start=time.monotonic();errors={};backend=None;interrupted=False
    plans={};locals_={};finals={};receipts={};cache_index={};physical={};resources={}
    from main.tube_state import b_time_payload_cdc32_v1 as method
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    source=Path(config['source']).expanduser().resolve()
    for view,length in zip(records.VIEWS,records.LENGTHS):
        plans[view]=method.blind_plans(f'input{records.VIEWS.index(view)}',length,dict(status='NOT_ENTERED'))
        for u in records.units(view,length):
            physical.setdefault(u['id'],dict(status='NOT_STARTED',raw_bit_slots=160,vote_slots=1280,
                encoder_input_support=list(range(u['start'],u['start']+41)),cache_file=None))
    try:
        # Construction starts the RSS watcher only; model loading remains after every plan.
        if backend_factory is None:
            from runtime.wan.b_time_payload_cdc32_v1 import Backend
            backend_factory=Backend
        backend=backend_factory(config['model_id']);journal.check=backend.check
        for view,length in zip(records.VIEWS,records.LENGTHS):
            try:
                def freeze():
                    value=records.load(source/'receiver/received'/view/'submission.json')
                    value=method.blind_plans(f'input{records.VIEWS.index(view)}',length,value)
                    records.dump(output/'plans'/f'{view}.json',value)
                    return records.load(output/'plans'/f'{view}.json')
                plans[view]=journal.call(view+'/plan',freeze)
            except Exception as error:
                if type(error).__name__=='ResourceExceeded':raise
                errors[view+'/plan']=str(error)
                plans[view]=method.blind_plans(f'input{records.VIEWS.index(view)}',length,dict(status='UNREADABLE_SUBMISSION',error=str(error)))
        # Complete all five views/three arms even when a submission was unreadable.
        records.dump(output/'plans.json',plans);plans=records.load(output/'plans.json')
        records.dump(output/'raw_bank.json',physical)
        def load_model():
            value=backend.load();records.dump(output/'model.json',value);return value
        journal.call('model_load',load_model)
        for view,length in zip(records.VIEWS,records.LENGTHS):
            pixels=None
            path=source/'receiver/received'/view/'received.mp4'
            try:
                def probe():
                    value=backend.probe(path,length);records.dump(output/'views'/view/'probe.json',value);return value
                journal.call(view+'/probe',probe)
                pixels=journal.call(view+'/decode',lambda:backend.decode(path,length))
            except Exception as error:
                if type(error).__name__=='ResourceExceeded':raise
                errors[view+'/media']=str(error)
            for unit in records.units(view,length):
                uid=unit['id'];prefix=view+f"/q{unit['query']}/p{unit['phase']}";folder=output/'bank'/uid
                if uid in receipts:continue  # Actual-unit aliases never create a second encoding.
                if pixels is None:
                    receipts[uid]=dict(status='DEPENDENCY_UNRESOLVED',error='received media unavailable')
                    physical[uid].update(receipts[uid]);continue
                latent=projection=None
                try:
                    latent=journal.call(prefix+'/E41',lambda:backend.encode(pixels,unit['start'],length))
                    def project():
                        value=backend.project(latent)
                        # DCT coefficients are retained in the decoded original cache as well.
                        return value
                    projection=journal.call(prefix+'/project',project);latent=None
                    def bits():
                        value=backend.bits(projection,uid)
                        records.dump(folder/'bits.json',value)
                        records.dump(folder/'cache.json',_flatten(value))
                        return value
                    journal.call(prefix+'/bits',bits)
                    cache_index[uid]=folder/'cache.json';receipts[uid]=dict(status='RETURNED',cache_file=str(folder/'cache.json'))
                except Exception as error:
                    receipts[uid]=dict(status=next((journal.rows[prefix+'/'+op]['status'] for op in ('bits','project','E41')
                        if journal.rows[prefix+'/'+op]['status']!='NOT_STARTED'),'FAILED'),error=str(error))
                    if type(error).__name__=='ResourceExceeded':raise
                    errors[prefix]=str(error)
                finally:
                    latent=projection=None
                    physical[uid].update(receipts.get(uid,dict(status='INTERRUPTED')))
                    records.dump(output/'raw_bank.json',physical)
            pixels=None
            memo={};loaded={}
            def cache(eid,key,bit):
                uid=eid.rsplit('/row',1)[0]
                if uid not in loaded:loaded[uid]=records.load(cache_index[uid])
                return loaded[uid][eid][key][str(bit)]
            for arm in records.ARMS:
                local=[]
                for q,plan in enumerate(plans[view][arm]):
                    attached=method.attach_receipts(plan,receipts)
                    def read():
                        value=cdc.read_time_local(attached,cache,memo=memo)
                        records.dump(output/'local'/view/arm/f'q{q}.json',value);return value
                    value=journal.call(f'{view}/{arm}/q{q}/read',read)
                    local.append(value);locals_.setdefault(view,{}).setdefault(arm,[]).append(value)
                def aggregate():
                    value=cdc.aggregate_time(local);records.dump(output/'final'/view/f'{arm}.json',value);return value
                finals.setdefault(view,{})[arm]=journal.call(f'{view}/{arm}/aggregate',aggregate)
    except BaseException as error:
        interrupted=not isinstance(error,Exception);errors['attempt']=f'{type(error).__name__}: {error}'
    def terminal_metadata():
        journal.terminalize('fixed attempt ended; no retries or later operation grants')
        # Metadata terminalization is not a logical read/aggregate and never opens a cache.
        for view in records.VIEWS:
            for arm in records.ARMS:
                completed=locals_.setdefault(view,{}).setdefault(arm,[])
                for q in range(len(completed),3):completed.append(_empty(plans[view][arm][q],'OPERATION_NOT_RETURNED'))
                if arm not in finals.setdefault(view,{}):
                    value=_empty_final(f'input{records.VIEWS.index(view)}',arm,'AGGREGATE_NOT_RETURNED')
                    value['slots']=[dict(row,selected_refs=[r for local in completed for r in local['slots'][row['slot']]['selected_refs']]) for row in value['slots']]
                    finals[view][arm]=value
        for uid,row in physical.items():
            if row['status']=='NOT_STARTED':row.update(status='DEPENDENCY_UNRESOLVED',error='not entered before attempt terminated')
    blind_saved=False
    try:
        terminal_metadata()
        records.dump(output/'raw_bank.json',physical)
        records.dump(output/'blind_terminal.json',dict(status='ALL_BLIND_SLOTS_TERMINAL',plans=plans,local=locals_,final=finals,
            raw_bank=physical,rows=journal.rows,errors=dict(errors),denominators=records.denominators()))
        blind_saved=True
        if backend is not None:backend.check()
    except BaseException as error:
        interrupted |= not isinstance(error,Exception)
        errors['terminal_save']=f'{type(error).__name__}: {error}'
    finally:
        # Keep the same RSS watcher alive through the large final save, also on failure.
        if backend is not None:
            try:resources=backend.close()
            except Exception as error:errors['close']=str(error)
    journal.cpu_seconds=time.monotonic()-journal.started-journal.non_cpu_seconds
    if journal.cpu_seconds>1800:errors['cpu_resources']='CPU receive/save limit observed AFTER_BLIND_TERMINAL_SAVE'
    if resources.get('error') or resources.get('peak_resident_bytes',0)>8*1024**3:
        errors['resources']=resources.get('error') or 'RSS limit exceeded'
    resources.update(cpu_receive_save_seconds=journal.cpu_seconds,E41_seconds=journal.encode_seconds,limits=records.LIMITS,
        completion_errors=dict(errors),measurement_end='AFTER_BLIND_TERMINAL_SAVE_AND_MONITOR_CLOSE',
        time_enforcement='codec/RSS polled during codec; model/E41/CPU elapsed checked after return; no hard interruption guarantee for model/E41')
    records.dump(output/'resources.json',resources)
    evaluation_status='EVIDENCE_INCOMPLETE'
    try:
        if not blind_saved:raise ValueError('blind terminal was not saved; truth/evaluator not opened')
        if evaluator is None:
            from .b_time_payload_cdc32_v1_evaluation import evaluate
            evaluator=evaluate
        evaluation=evaluator(output,source/'receiver/private/attack_truth.json')
        if errors and evaluation.get('status')!='COUNTEREXAMPLE':
            evaluation.update(status='EVIDENCE_INCOMPLETE',completion_errors=dict(errors))
        records.dump(output/'evaluation.json',evaluation);evaluation_status=evaluation['status']
    except Exception as error:errors['evaluation']=str(error)
    result=dict(status='INTERRUPTED' if interrupted else 'FIXED_ATTEMPT_TERMINATED',errors=errors,
        evaluation_status=evaluation_status,denominators=records.denominators(),budgets=records.budgets(),
        wall_seconds=time.monotonic()-start,scientific_pass=None,
        claim_ceiling='same-chain conditional development only; native failures remain; no QIM losslessness, identity authentication or general reliability claim')
    records.dump(output/'result.json',result);return result
