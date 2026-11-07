"""Fixed stage workers; time -> all plans -> all identity finals -> claims -> post."""
from pathlib import Path
import time
from main.tube_state import b_paper_cohort_v1 as p
from . import b_paper_cohort_records_v1 as rec


class Events:
    def __init__(self,output,stage):
        self.path=Path(output)/'stages'/stage/'internal_calls.json';self.phase=None
        self.rows={k:dict(limit=v,started=0,returned=0) for k,v in p.INNER.items()}
        rec.write(self.path,self.rows)
    def __call__(self,kind,returned):
        row=self.rows[kind]
        if not returned:
            limit=4200 if kind=='K' and self.phase=='solver' else row['limit']
            if row['started']>=limit: raise RuntimeError('fixed inner budget consumed: '+kind)
            row['started']+=1
        else:
            if row['returned']>=row['started']: raise RuntimeError('inner return without start')
            row['returned']+=1
        rec.write(self.path,self.rows)


def save_return(path,value): rec.write(path,value);return value


def receive(output,scratch,journal,backend):
    import numpy as np
    from main.tube_state.b_joint_time_relative_validation_v1 import construction_maps
    maps=construction_maps()  # Editor only; receiver receives pixels and anonymous IDs.
    rec.write(output/'private/attack_truth.json',maps)
    for arm in p.ARMS:
        pixels=backend.quantize(backend.preclip(arm));path=output/'media'/arm/'first.mp4'
        stem=arm+'/first/';path.parent.mkdir(parents=True,exist_ok=True)
        journal.call(stem+'encode',lambda:backend.encode(pixels,path))
        probe=journal.call(stem+'probe',lambda:backend.probe(path,181));rec.write(path.with_suffix('.probe.json'),probe)
        first=journal.call(stem+'decode_RGB',lambda:backend.decode(path,181));del pixels
        for item in (i for i in p.inputs() if i['arm']==arm):
            uid,n=item['id'],item['length'];prefix='time/'+uid+'/'
            path=output/'media'/uid/'received.mp4';path.parent.mkdir(parents=True,exist_ok=True)
            edited=first[maps[item['view']]]
            journal.call(prefix+'encode',lambda:backend.encode(edited,path))
            probe=journal.call(prefix+'probe',lambda:backend.probe(path,n));rec.write(path.with_suffix('.probe.json'),probe)
            rgb=journal.call(prefix+'decode_RGB',lambda:backend.decode(path,n))
            # RGB readback is an actual budgeted call. Identity has its own fixed codec pass.
            del rgb,edited
            y=journal.call(prefix+'decode_Y',lambda:backend.decode_y(path,n,probe))
            folder=output/'blind'/uid
            packet=journal.call(prefix+'hard_receive',lambda:backend.hard_receive(y,folder,uid))
            del y
            if len(packet['nodes'])!=n: raise ValueError('fixed received node count')
            journal.call(prefix+'relative_compare',lambda:save_return(folder/'relative.json',backend.relative(packet,uid)))
            del packet
        del first


def finalize_raw(output,operations):
    from .b_relative_time_payload_records_v1 import _decoded_index
    slots=rec.read(output/'raw_terminal_slots.json')['slots']
    for uid,keys in slots.items():
        for key,rows in keys.items():
            op=operations[uid+'/'+key];indexed={};errors={};file_error=None
            if op['status']=='RETURNED':
                try: indexed,errors=_decoded_index(rec.read(output/'bank'/uid/(key+'.json')),uid,key)
                except (OSError,ValueError,TypeError,KeyError) as error: file_error=str(error)
            for row in rows:
                row.update(operation_status=op['status'],status=op['status'],value=None)
                if op['status']!='RETURNED': continue
                addr=(row['local_row'],row['region'],row['local_bit']);bit=indexed.get(addr)
                error=file_error or errors.get(addr) or ('missing bit' if bit is None else None)
                if bit:
                    value=bit.get('value');status=bit.get('status')
                    available=status in ('AVAILABLE','RETURNED','READ','CONDITIONAL_READ')
                    missing=status in ('FAILED','NOT_ENTERED','UNRESOLVED','UNSUPPORTED','INTERRUPTED')
                    valid=(type(bit.get('count')) is int and bit['count']==8 and
                        ((available and type(value) is int and value in (0,1)) or (missing and value is None)))
                    if not valid: error=error or 'illegal bit/count/status'
                    row.update(bit_status=status,count=bit.get('count'),original_bit_value=value)
                if error: row.update(status='UNRESOLVED_RECEIPT',error=error)
                else: row.update(value=bit['value'],status=bit['status'])
    rec.write(output/'raw_terminal_slots.json',dict(slots=slots,planned_slots=57600,status='ALL_RAW_TERMINAL'))


def identity(output,journal,backend_factory):
    from main.tube_state import b_time_payload_cdc32_v1 as previous
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    from main.tube_state import qmatch_v1 as qmatch
    from main.tube_state.qmatch_given_crop_v1 import evidence_from_aggregate
    from .qmatch_given_crop_v1 import header_only,cache_from_payload
    from .b_relative_time_payload_worker_v1 import quality_pair
    config=rec.read(output/'public.json');plans={};raw=rec.read(output/'raw_bank.json');backend=None
    def relative(uid):
        try: return rec.read(output/'blind'/uid/'relative.json')
        except (OSError,ValueError) as error: return dict(input_error=str(error))
    # Freeze every strategy before model_load / bit values, including borrowed NEW -> P_PAIRED.
    for item in p.inputs():
        uid=item['id']
        new='input'+str(10+p.VIEWS.index(item['view']))
        plans[uid]=journal.call(f'identity/{uid}/plan',lambda:save_return(output/'plans'/f'{uid}.json',
            p.freeze(item,relative(uid),relative(new) if item['arm']=='P0' else None)))
    rec.write(output/'plans_frozen.json',dict(status='ALL_PLANS_SAVED',saved_at=time.time(),inputs=15,strategies=40))
    technical={};evidence_files={}
    try:
        backend=backend_factory(config['model_id'])
        journal.call('identity/model_load',lambda:save_return(output/'identity_model.json',backend.load()))
        quality=rec.read(output/'quality.json')
        for view in p.VIEWS:
            pixels_by_arm={}
            for item in (i for i in p.inputs() if i['view']==view):
                stem,n=item['id'],item['length'];path=output/'media'/stem/'received.mp4'
                journal.call(f'identity/{stem}/probe',lambda:save_return(output/'identity_media'/stem/'probe.json',backend.probe(path,n)))
                pixels=journal.call(f'identity/{stem}/decode_RGB',lambda:backend.decode(path,n));pixels_by_arm[item['arm']]=pixels
                cache={};tech=[]
                for bank in p.bank(item):
                    uid=bank['view_id'];folder=output/'bank'/uid
                    latent=journal.call(uid+'/E41',lambda:backend.encode(pixels,bank['start'],n))
                    raw[uid].update(E41='RETURNED',status='ENCODED');rec.write(output/'raw_bank.json',raw)
                    projection=journal.call(uid+'/project',lambda:save_return(folder/'projection.json',backend.project(latent)));del latent
                    raw[uid]['project']='RETURNED'
                    for key in p.KEYS:
                        def bits(key=key):
                            decoded=backend.bits(projection,uid,key);rec.write(folder/(key+'.json'),decoded)
                            entry=dict(key_id=key);header=header_only(decoded,entry)
                            if header['view_id']!=uid: raise ValueError('actual decoded view address mismatch')
                            flat,errors=cache_from_payload(decoded,entry,header);tech.extend(errors)
                            rec.write(folder/(key+'_cache.json'),flat)
                            for eid,keyed in flat.items(): cache.setdefault(eid,{}).update(keyed)
                            return decoded['status']
                        journal.call(uid+'/'+key,bits);raw[uid]['keys'][key]='RETURNED'
                    raw[uid]['status']='RETURNED';rec.write(output/'raw_bank.json',raw);del projection
                technical[stem]=tech
                for strategy,group in plans[stem]['strategies'].items():
                    key=p.key_for(strategy);local=[];memo={}
                    receipts={uid:dict(row,status='RETURNED' if row['keys'].get(key)=='RETURNED' else 'TECHNICAL_UNREAD') for uid,row in raw.items()}
                    for q,plan in enumerate(group):
                        attached=previous.attach_receipts(plan,receipts)
                        for g in attached['groups']:
                            for ref in g['references']:
                                ops=[ref['view_id']+'/'+op for op in ('E41','project',key)]
                                ref.update(source_package=config['source'],original_operations=ops,
                                    original_operation_statuses={op:journal.rows[op]['status'] for op in ops},
                                    original_bits_file=str(Path('bank')/ref['view_id']/(key+'.json')))
                        local.append(journal.call(f'{stem}/{strategy}/q{q}/local',lambda:save_return(
                            output/'local'/stem/strategy/f'q{q}.json',cdc.read_time_local(attached,cache,memo=memo))))
                    final=journal.call(f'{stem}/{strategy}/aggregate',lambda:save_return(
                        output/'final'/stem/(strategy+'.json'),cdc.aggregate_time(local)))
                    if strategy.startswith(('B_','K_')):
                        unit=dict(input_id=stem,media_id=stem,source_package=config['source'],arm=item['arm'],view=view,key_id=key)
                        errors=[] if plans[stem]['time']['status']=='READY' else [dict(kind='TIME_INPUT',reason=plans[stem]['time']['status'])]
                        ev=evidence_from_aggregate(final,unit,input_errors=errors,technical_errors=tech)
                        ev['geometry_scope']='BLIND_RELATIVE_TIME_CONDITIONAL'
                        ep=output/'evidence'/stem/(key+'.json');rec.write(ep,ev);evidence_files[(stem,key)]=ep
                cache.clear()
            for pair,a,b,scope in [('NEW_P0','NEW','P0','TIME_PACKET_INCREMENT'),('P0_OFF','P0','OFF','IDENTITY_TRAJECTORY_DIFFERENCE'),('NEW_OFF','NEW','OFF','TOTAL_DIFFERENCE')]:
                address=f'quality/{pair}/{view}'
                value=journal.call(address,lambda:quality_pair(pixels_by_arm[a],pixels_by_arm[b]));value['scope']=scope
                quality[f'{pair}/{view}']=value;rec.write(output/'quality.json',quality)
            pixels_by_arm.clear()
        rec.write(output/'identity_finals_frozen.json',dict(status='ALL_IDENTITY_FINALS_SAVED',saved_at=time.time(),finals=40))
        decisions=rec.read(output/'decisions.json')
        for i,d in enumerate(decisions):
            stem,key,claim=d['input_id'],d['key_id'],d['claim_id'];address=f'{stem}/{key}/{claim}'
            def invoke():
                d.update(status='STARTED',comparisons=None);rec.write(output/'decisions.json',decisions)
                return qmatch.match_claim(rec.read(evidence_files[(stem,key)]),claim,p.query_bits(claim))
            value=journal.call(address,invoke)
            decisions[i]=dict(d,**{k:v for k,v in value.items() if k not in d},status='RETURNED',
                              verdict=value['verdict'],comparisons=value['comparisons'])
            rec.write(output/'decisions.json',decisions)
    finally:
        if backend is not None: rec.write(output/'identity_backend_resources.json',backend.close())


def worker(stage,output,scratch,*,generation_factory=None,time_factory=None,identity_factory=None,evaluator=None):
    output,scratch=Path(output),Path(scratch)
    if stage not in p.STAGES: raise ValueError('unknown stage')
    if rec.read(output/'attempt.json')['status']!='RUNNING': return rec.recover(output)
    folder=output/'stages'/stage;folder.mkdir(parents=True,exist_ok=True)
    try:
        with (folder/'started.json').open('x') as f: f.write('{"consumed":true}')
    except FileExistsError: return rec.recover(output)
    rec.begin_io()
    journal=rec.Journal(output);events=Events(output,stage);backend=None;failure=None;interrupted=False;tick=time.monotonic()
    try:
        if stage=='identity':
            if (output/'dependency_failure.json').is_file(): failure=rec.read(output/'dependency_failure.json')['failure']
            else:
                if identity_factory is None:
                    from .b_paper_cohort_backend_v1 import identity_backend
                    identity_factory=identity_backend
                identity(output,journal,identity_factory)
        else:
            config=rec.read(output/('private/config.json' if stage in ('prepare','solve') else 'public.json'))
            if stage=='replay': config=dict(config,source=dict(id=config['source']))
            if generation_factory is None or time_factory is None:
                from .b_paper_cohort_backend_v1 import GenerationBackend,TimeBackend
                generation_factory=generation_factory or GenerationBackend;time_factory=time_factory or TimeBackend
            backend=(time_factory if stage=='receive' else generation_factory)(config,output,scratch,events,rec.write)
            def call(key,fn): events.phase=key;return journal.call(key,fn)
            if stage=='prepare':
                call('generation_load',backend.generation_load)
                call('generate_OFF',lambda:backend.generate_arm('OFF'));call('generate_P0',lambda:backend.generate_arm('P0'))
                call('capture_vae_load',backend.vae_load);call('decode_OFF',backend.decode_off)
                call('capture',backend.capture);call('head_prepare',backend.head)
            elif stage=='solve':
                for key,fn in [('q0',backend.q0),('target',backend.target),('solver',backend.solve),('prediction',backend.predict),('handoff',backend.handoff)]:call(key,fn)
            elif stage=='replay':
                call('replay_vae_load',backend.vae_load);call('native_replay',backend.replay)
            else: receive(output,scratch,journal,backend)
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}';interrupted=isinstance(error,(KeyboardInterrupt,InterruptedError))
    finally:
        if backend is not None:
            try: rec.write(folder/'backend_resources.json',backend.close())
            except BaseException as error:
                failure=failure or f'close: {type(error).__name__}: {error}'
                interrupted=interrupted or isinstance(error,(KeyboardInterrupt,InterruptedError))
    if stage=='identity':
        try:
            finalize_raw(output,journal.rows);rec.close_blind(output,failure)
            if not interrupted:
                if evaluator is None:
                    from .b_paper_cohort_evaluation_v1 import post_once
                    evaluator=post_once
                evaluator(output)
        except BaseException as error:
            failure=failure or f'post/closure: {type(error).__name__}: {error}'
            interrupted=interrupted or isinstance(error,(KeyboardInterrupt,InterruptedError))
    result=dict(stage=stage,status='COMPLETED' if failure is None else 'STOPPED_FIXED_ATTEMPT',failure=failure,
                interrupted=interrupted,wall_seconds=time.monotonic()-tick)
    rec.write(folder/'result.json',result);rec.save_io(folder/'io_counts.json');return result
