"""GPU-1/GPU-2 independent consumed stages. Never execute local CPU work here."""
from pathlib import Path
from . import b_joint_time_packet_a_physical_v1_records as rec


def wait(reason):return dict(status='WAITING_INPUT',reason=reason,attempt_consumed=False,physical_calls_this_invocation=0)


def gpu1(config,output,*,backend_factory=None):
    output=Path(output).expanduser().resolve()
    if (output/'attempt.json').exists():return rec.recover(output)
    cpu=Path(config['cpu_handoff'])
    try:handoff=rec.load(cpu/'cpu_handoff.json')
    except (OSError,ValueError):return wait('local CPU stage-P terminal handoff not available')
    if handoff.get('stage_terminal') is not True:return wait('local CPU stage-P is not terminal')
    if not rec.reserve(output,'gpu1',config):return rec.recover(output)
    journal=rec.Journal(output,'gpu1');errors={};sources={s:dict(status='NOT_STARTED') for s in rec.SOURCES};backend=None
    try:
        if backend_factory is None:
            from runtime.wan.b_joint_time_packet_a_native_v1 import Backend
            backend_factory=Backend
        backend=backend_factory(config.get('model_id',rec.MODEL_ID))
        eligible=handoff.get('source_eligibility',{})
        if any(eligible.get(s) for s in rec.SOURCES):
            rec.dump(output/'model.json',journal.call('model_load',backend.load))
        capture=None
        if eligible.get('e02'):
            try:
                capture=journal.call('e02/capture',lambda:backend.capture(config['e02_terminal'],output/'e02/host',lambda k,d:journal.event('e02',k,d)))
            except Exception as error:errors['e02/capture']=type(error).__name__+': '+str(error)
        for source in rec.SOURCES:
            try:
                if not eligible.get(source):raise ValueError('CPU source has no compliant native handoff; no re-solve')
                if source=='e02':
                    if capture is None:raise ValueError('planned e02 host capture unavailable')
                    baseline,features=capture['base'],capture['features']
                else:baseline,features=config['n01_base'],config['n01_features']
                value=journal.call(source+'/replay',lambda:backend.replay(cpu/source,baseline,features,output/source,
                    lambda k,d:journal.event(source,k,d),rec.dump))
                sources[source]=dict(value,base=str(baseline),preclip=source+'/NEW_preclip.npy')
            except Exception as error:
                sources[source]=dict(status='UNRESOLVED',error=type(error).__name__+': '+str(error));errors[source]=sources[source]['error']
    except BaseException as error:errors['stage']=type(error).__name__+': '+str(error)
    finally:
        if backend is not None:
            try:rec.dump(output/'resources.json',backend.close())
            except Exception as error:errors['close']=str(error)
    return journal.finish(errors,sources=sources,next='LOCAL_CPU_MEDIA',codec=0,solver=0,DiT=0,training=0)


def gpu2(config,output,*,backend_factory=None):
    output=Path(output).expanduser().resolve()
    if (output/'attempt.json').exists():return rec.recover(output)
    cpu=Path(config['cpu_media']);prior=rec.terminal(cpu)
    if prior is None:return wait('local CPU-media terminal receipt not available; successful RGB for every source is not required')
    if not rec.reserve(output,'gpu2',config):return rec.recover(output)
    journal=rec.Journal(output,'gpu2');errors={};backend=None
    units=rec.identity_units();bank={u['id']:dict(u) for u in units};all_rows=[]
    from main.tube_state import b_joint_time_packet_a_identity_v1 as identity
    rec.dump(output/'raw_bank.json',bank)  # All addresses frozen before even old cache access.
    try:
        if backend_factory is None:
            from runtime.wan.b_joint_time_packet_a_native_v1 import Backend
            backend_factory=Backend
        backend=backend_factory(config.get('model_id',rec.MODEL_ID));model_error=None
        try:rec.dump(output/'model.json',journal.call('model_load',backend.load))
        except Exception as error:model_error=type(error).__name__+': '+str(error);errors['model_load']=model_error
        pixels=None;loaded_view=None
        for unit in units:
            uid=unit['id'];rows=identity.empty(unit,'NOT_ENTERED');decoded=None
            try:
                if (unit['source'],unit['arm'])==('n01','OLD'):
                    def reuse():
                        value=rec.load(Path(config['n01_old_bank'])/unit['old_id']/'bits.json')
                        rec.dump(output/'bank'/uid/'bits.json',value)
                        return value,identity.addressed(value,unit,unit['old_id'])
                    decoded,rows=journal.call(uid+'/reuse',reuse)
                else:
                    if model_error is not None:raise RuntimeError(model_error)
                    key=f"{unit['source']}/{unit['arm']}/{unit['view']}"
                    media=prior.get('rgb',{}).get(key,{})
                    if media.get('status')!='SAVED':raise ValueError('actual submitted received RGB unavailable')
                    if loaded_view!=key:
                        import numpy as np
                        pixels=np.load(rec.package_file(cpu,media['file']),mmap_mode='r',allow_pickle=False);loaded_view=key
                    if pixels.shape!=(unit['frames'],320,512,3):raise ValueError('submitted received RGB length mismatch')
                    latent=projection=None
                    try:
                        if hasattr(backend,'check_encode_budget'):backend.check_encode_budget(unit['source'])
                        latent=journal.call(uid+'/E41',lambda:backend.encode(pixels,unit['start'],unit['source']))
                        projection=journal.call(uid+'/project',lambda:backend.project(latent));latent=None
                        def bits():
                            value=backend.bits(projection,uid);rec.dump(output/'bank'/uid/'bits.json',value);return value
                        decoded=journal.call(uid+'/bits',bits)
                        rows=identity.addressed(decoded,unit,uid)
                    finally:latent=projection=None
                rec.dump(output/'bank'/uid/'bits.json',decoded)
                bank[uid].update(status='RETURNED',file='bank/'+uid+'/bits.json')
            except Exception as error:
                message=type(error).__name__+': '+str(error);errors[uid]=message
                bank[uid].update(status='UNRESOLVED',error=message);rows=identity.empty(unit,message)
            all_rows.extend(rows)
            rec.dump(output/'bank'/uid/'addressed.json',rows);rec.dump(output/'raw_bank.json',bank)
    except BaseException as error:errors['stage']=type(error).__name__+': '+str(error)
    finally:
        existing={tuple(x[k] for k in ('source','arm','view','query','phase')) for x in all_rows}
        for unit in units:
            if tuple(unit[k] for k in ('source','arm','view','query','phase')) not in existing:
                rows=identity.empty(unit,'STAGE_ENDED_WITHOUT_READ');all_rows.extend(rows)
                rec.dump(output/'bank'/unit['id']/'addressed.json',rows)
                bank[unit['id']].update(status='UNRESOLVED',error='stage ended without read')
        rec.dump(output/'addressed_raw_bits.json',all_rows);rec.dump(output/'raw_bank.json',bank)
        if backend is not None:
            try:rec.dump(output/'resources.json',backend.close())
            except Exception as error:errors['close']=str(error)
    return journal.finish(errors,bank=bank,raw_rows=38400,next='LOCAL_POST',no_formal_aggregator=True,codec=0,DiT=0,training=0)
