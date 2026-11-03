"""Local CPU stage after GPU-1. Fixed media calls, raw streams and 30 blind views."""
from pathlib import Path
from . import b_joint_time_packet_a_physical_v1_records as rec
from .b_joint_time_packet_a_gpu_v1 import wait


def construction_maps():
    """Private fixture construction only. Never passed into the received-only reader."""
    return dict(FULL=list(range(181)),CROP=list(range(38,139)),
                DELETE1=[i for i in range(181) if i!=44],
                REPEAT1=list(range(45))+list(range(44,181)),
                DELETE3=[i for i in range(181) if i not in (43,44,45)])


def run(config,output,*,backend_factory=None):
    output=Path(output).expanduser().resolve()
    if (output/'attempt.json').exists():return rec.recover(output)
    gpu=Path(config['gpu1']);prior=rec.terminal(gpu)
    if prior is None:return wait('GPU-1 terminal receipt not available')
    if not rec.reserve(output,'cpu_media',config):return rec.recover(output)
    journal=rec.Journal(output,'cpu_media');errors={};backend=None;rgb={};raws={};layers={};times={}
    for unit in rec.time_units():
        times[unit['id']]=dict(unit,status='UNRESOLVED',reason='not entered')
    def failure(key,error):
        errors[key]=type(error).__name__+': '+str(error)
        return dict(status='UNRESOLVED',error=errors[key])
    try:
        import numpy as np
        if backend_factory is None:
            from runtime.wan.b_joint_time_packet_a_media_v1 import Backend
            backend_factory=Backend
        backend=backend_factory()
        mappings=construction_maps();rec.dump(output/'private/attack_truth.json',dict(views=mappings))
        for source in rec.SOURCES:
            native=None;first=None
            try:
                source_receipt=prior.get('sources',{}).get(source,{})
                if source_receipt.get('status')!='NATIVE_MEASURED':raise ValueError('GPU-1 source native output unavailable')
                native=np.load(gpu/source_receipt['preclip'],mmap_mode='r',allow_pickle=False)
                for layer in ('preclip','clip','rgb8'):
                    key=f'{source}/NEW/layer/{layer}'
                    try:
                        value=journal.call(key,lambda:backend.rgb_raw(native,layer))
                        path=output/'layers'/source/'NEW'/(layer+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez(path,**value);layers[key]=dict(status='SAVED',file=str(path.relative_to(output)))
                    except Exception as error:layers[key]=failure(key,error)
                pixels=backend.quantize(native)
                path=output/'media'/source/'NEW/first.mp4';path.parent.mkdir(parents=True,exist_ok=True)
                journal.call(source+'/NEW/first/encode',lambda:backend.encode(pixels,path));pixels=None
                probe=journal.call(source+'/NEW/first/probe',lambda:backend.probe(path,181))
                rec.dump(path.with_suffix('.probe.json'),probe)
                first=journal.call(source+'/NEW/first/decode_RGB',lambda:backend.decode(path,181))
                np.save(path.with_suffix('.RGB.npy'),first,allow_pickle=False)
                value=journal.call(source+'/NEW/layer/firstRGB',lambda:backend.rgb_raw(first,'firstRGB'))
                dest=output/'layers'/source/'NEW/firstRGB.npz';np.savez(dest,**value)
                layers[source+'/NEW/layer/firstRGB']=dict(status='SAVED',file=str(dest.relative_to(output)))
            except Exception as error:failure(source+'/NEW/first',error)
            for view,length in zip(rec.VIEWS,rec.LENGTHS):
                key=f'{source}/NEW/{view}';path=output/'media'/source/'NEW'/view/'received.mp4'
                try:
                    if first is None:raise ValueError('first-generation decoded RGB unavailable')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    journal.call(key+'/encode',lambda:backend.encode(first[mappings[view]],path))
                    probe=journal.call(key+'/probe',lambda:backend.probe(path,length))
                    rec.dump(path.with_suffix('.probe.json'),probe)
                except Exception as error:
                    rgb[key]=failure(key,error);continue
                # Independent RGB/Y decodes share the one probe. One failure does not suppress the other.
                try:
                    pixels=journal.call(key+'/decode_RGB',lambda:backend.decode(path,length))
                    dest=path.with_suffix('.RGB.npy');np.save(dest,pixels,allow_pickle=False);pixels=None
                    rgb[key]=dict(status='SAVED',file=str(dest.relative_to(output)),frames=length)
                except Exception as error:rgb[key]=failure(key+'/RGB',error)
                try:
                    y=journal.call(key+'/decode_Y',lambda:backend.decode_y(path,length,probe))
                    dest=path.with_suffix('.Y.npy');np.save(dest,y,allow_pickle=False);y=None
                    raws[key]=dict(native_y=str(dest))
                except Exception as error:failure(key+'/Y',error)
            native=first=None
        # The five planned e02 OLD decodes supply only identity input; old time raw is reused.
        for view,length in zip(rec.VIEWS,rec.LENGTHS):
            key='e02/OLD/'+view
            try:
                path=Path(config['e02_old_media'][view])
                probe=journal.call(key+'/probe',lambda:backend.probe(path,length))
                rec.dump(output/'media/e02/OLD'/view/'probe.json',probe)
                pixels=journal.call(key+'/decode_RGB',lambda:backend.decode(path,length))
                dest=output/'media/e02/OLD'/view/'received.RGB.npy';np.save(dest,pixels,allow_pickle=False);pixels=None
                rgb[key]=dict(status='SAVED',file=str(dest.relative_to(output)),frames=length)
            except Exception as error:rgb[key]=failure(key,error)
        for source in rec.SOURCES:
            for arm in ('OLD','BASE'):
                for layer in rec.LAYERS:
                    key=f'{source}/{arm}/layer/{layer}'
                    try:
                        def reuse_layer():
                            with np.load(config['layer_raw'][key],allow_pickle=False) as saved:
                                value={k:saved[k] for k in ('q','d0','d1','soft','mask')}
                            if any(v.shape!=(181,5,2,63) for v in value.values()):raise ValueError('saved full181 raw geometry required')
                            if value['mask'].dtype!=np.bool_:raise ValueError('saved boolean observation mask required')
                            dest=output/'layers'/source/arm/(layer+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                            np.savez(dest,**value);return dict(status='SAVED_REUSE',file=str(dest.relative_to(output)))
                        layers[key]=journal.call(key,reuse_layer)
                    except Exception as error:layers[key]=dict(failure(key,error),status='MISSING_OR_UNREADABLE_NO_REFILL')
        for unit in rec.time_units():
            key=unit['id']
            try:
                if unit['arm']=='NEW':
                    if key not in raws:raise ValueError('actual new second-generation native Y unavailable')
                    raw_path=None;native_y=raws[key]['native_y']
                else:raw_path=config['time_raw'][key];native_y=None
                value=journal.call(key+'/receive',lambda:backend.receive(raw_path,output/'received'/key,
                    unit['coding'],unit['frames'],rec.dump,native_y=native_y))
                times[key].update(status=value['status'],file='received/'+key+'/scores.json')
            except Exception as error:times[key].update(failure(key+'/receive',error))
            rec.dump(output/'time_units.json',times)
    except BaseException as error:failure('stage',error)
    finally:
        for unit in rec.time_units():
            if unit['arm']=='NEW':rgb.setdefault(unit['id'],dict(status='UNRESOLVED',error='stage ended without RGB'))
            if times[unit['id']]['status']!='BLIND_SAVED':
                rec.dump(output/'received'/unit['id']/'unresolved.json',dict(unit,status='UNRESOLVED',formal_output=None,
                    nodes=[dict(received_frame=i,status='UNRESOLVED',formal_output=None) for i in range(unit['frames'])],
                    edges=[dict(received_edge=i,status='UNRESOLVED',formal_output=None) for i in range(unit['frames']-1)],
                    cause=times[unit['id']].get('error','not entered')))
        for source in rec.SOURCES:
            for arm in rec.ARMS:
                for layer in rec.LAYERS:layers.setdefault(f'{source}/{arm}/layer/{layer}',dict(status='UNRESOLVED',error='not entered'))
        rec.dump(output/'time_units.json',times);rec.dump(output/'layer_streams.json',layers)
        if backend is not None:
            try:rec.dump(output/'resources.json',backend.close())
            except Exception as error:failure('close',error)
    return journal.finish(errors,rgb=rgb,time_units=times,layers=layers,next='USER_GPU_2',truth_opened_for_evaluation=False,
                          model_load=0,E41=0,DiT=0,training=0)
