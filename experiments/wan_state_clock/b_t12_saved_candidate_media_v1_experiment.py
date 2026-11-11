"""Fixed saved candidate, five views and original two-key B/K/U receiver."""
from pathlib import Path
import copy
import shutil
import time
from . import b_t12_saved_candidate_media_v1_records as rec


def run(config,*,backend_factory=None,preflight_check=rec.preflight):
    output=rec.validate_paths(config)
    if not (output/'attempt.json').exists():
        preflight=preflight_check(config)
        rec.write(output.with_name(output.name+'-preflight.json'),preflight)
        if preflight['status']!='INPUTS_READABLE': return preflight
    book=rec.Book(config);config=book.config;output=book.root
    if book.rows['post']['status']!='NOT_STARTED':
        try: return rec.read(output/'result.json')
        finally: book.close()
    import numpy as np
    import torch
    from main.tube_state import b_paper_cohort_v1 as protocol
    from main.tube_state import b_time_payload_cdc32_v1 as previous
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    from main.tube_state import qmatch_v1 as qmatch
    from main.tube_state.qmatch_given_crop_v1 import evidence_from_aggregate
    from main.tube_state.b_joint_time_relative_validation_v1 import construction_maps
    from runtime.wan.qmatch_given_crop_v1 import header_only,cache_from_payload
    from runtime.wan.b_relative_time_payload_worker_v1 import quality_pair
    backend=None;interrupted=False;began=time.monotonic()
    def loads(returned,value):
        rows=rec.read(output/'model_loads.json')
        if not returned:
            token=len(rows);rows.append(dict(status='STARTED',started_at=time.time()))
            rec.write(output/'model_loads.json',rows);return token
        rows[value['token']].update(value);rec.write(output/'model_loads.json',rows)
    def head_event(kind,returned):
        if kind!='head_replay': raise ValueError('unexpected physical call: '+kind)
        rows=rec.read(output/'native_calls.json')
        eligible=[str(s) for s in rec.STARTS if rows[str(s)]['status']==('STARTED' if returned else 'NOT_STARTED')]
        if not eligible: raise ValueError('native46 consumed or return without start')
        row=rows[eligible[0]]
        row.update(status='RETURNED' if returned else 'STARTED',**({'returned':1} if returned else {'started':1}))
        rec.write(output/'native_calls.json',rows)
    def saved_array(key,path,fn,deps=(),tensor=False):
        target=output/path;target.parent.mkdir(parents=True,exist_ok=True)
        load=(lambda:torch.load(target,map_location='cpu',weights_only=True)) if tensor else (lambda:np.load(target,mmap_mode='r',allow_pickle=False))
        save=(lambda v:torch.save(v.detach().cpu(),target)) if tensor else (lambda v:np.save(target,v,allow_pickle=False))
        return book.call(key,fn,load=load,save=save,dependencies=deps)
    def js(key,path,fn,deps=()): return book.json(key,path,fn,deps)
    def media_encode(key,path,fn,deps):
        def exists():
            if not path.is_file(): raise FileNotFoundError(str(path))
            return dict(file=str(path))
        return book.call(key,fn,load=exists,dependencies=deps)
    raw_path=output/'raw_bank.json'
    if raw_path.exists():
        try:
            saved_raw=rec.read(raw_path)
            if not isinstance(saved_raw,dict) or any(not isinstance(row,dict) or not isinstance(row.get('keys'),dict) for row in saved_raw.values()):
                raise ValueError('saved raw bank must map view addresses to records with key receipts')
        except (OSError,ValueError,TypeError) as error:
            diagnostics=output/'recovery';diagnostics.mkdir(exist_ok=True);stamp=str(time.time_ns())
            shutil.copyfile(raw_path,diagnostics/f'raw_bank_invalid_{stamp}.json')
            rec.write(diagnostics/f'raw_bank_error_{stamp}.json',dict(error=f'{type(error).__name__}: {error}',
                action='rebuild derived index from fixed banks and saved operation/artifact reads; no consumed physical retry'))
    raw={b['view_id']:dict(b,status='NOT_STARTED',
        E41=book.observation_status(b['view_id']+'/E41'),project=book.observation_status(b['view_id']+'/project'),
        keys={key:book.observation_status(b['view_id']+'/'+key) for key in ('key00','key01')})
        for item in rec.items() for b in rec.banks(item)}
    try:
        if backend_factory is None:
            from runtime.wan.b_t12_saved_candidate_media_v1 import Backend
            backend_factory=Backend
        backend=backend_factory(config['model_id'],loads)
        native=js('native','native/result.json',lambda:backend.native(config['candidate'],config['host'],
            Path(config['original'])/'private/head.pt',output/'native',head_event,rec.write))
        pixels=saved_array('quantize','cache/rgb8.npy',lambda:backend.quantize(output/'native/preclip.npy'),('native',))
        first=output/'media/first.mp4'
        media_encode('first/encode',first,lambda:backend.encode(pixels,first),('quantize',))
        js('first/probe','media/first.probe.json',lambda:backend.probe(first,181),('first/encode',))
        first_rgb=saved_array('first/RGB','cache/first_rgb.npy',lambda:backend.decode(first,181,'RGB'),('first/probe',))
        del pixels
        maps=construction_maps()  # Editor only. Never passed to the receiver.
        for item in rec.items():
            uid,n=item['id'],item['length'];path=output/'media'/uid/'received.mp4'
            media_encode(uid+'/encode',path,lambda:backend.encode(first_rgb[maps[item['view']]],path),('first/RGB',))
            js(uid+'/probe',f'media/{uid}/received.probe.json',lambda:backend.probe(path,n),(uid+'/encode',))
            rgb=saved_array(uid+'/RGB',f'cache/{uid}/rgb.npy',lambda:backend.decode(path,n,'RGB'),(uid+'/probe',));del rgb
            y=saved_array(uid+'/Y',f'cache/{uid}/y.npy',lambda:backend.decode(path,n,'Y'),(uid+'/probe',))
            rawfile=output/f'blind/{uid}/readout.npz';rawfile.parent.mkdir(parents=True,exist_ok=True)
            def raw_load():
                with np.load(rawfile,allow_pickle=False) as data: return {k:data[k] for k in data.files}
            observed=book.call(uid+'/read',lambda:backend.read(y),load=raw_load,save=lambda v:np.savez(rawfile,**v),dependencies=(uid+'/Y',))
            packet=js(uid+'/hard',f'blind/{uid}/packets.json',lambda:backend.hard(observed),(uid+'/read',))
            js(uid+'/relative',f'blind/{uid}/relative.json',lambda:backend.relative(packet,uid),(uid+'/hard',))
            del y,observed,packet
        del first_rgb,maps
        # All five received-only plans are frozen before the first E41 / bit read.
        plans={}
        for item in rec.items():
            uid=item['id']
            def freeze():
                try: relative=rec.read(output/f'blind/{uid}/relative.json') if book.available(uid+'/relative') else {}
                except (OSError,ValueError): relative={}
                return protocol.freeze(item,relative)
            plans[uid]=js(uid+'/plan',f'plans/{uid}.json',freeze)
            frozen=plans[uid]
            if frozen is not None and (not isinstance(frozen,dict) or not isinstance(frozen.get('time'),dict)
                    or 'status' not in frozen['time'] or not isinstance(frozen.get('strategies'),dict)):
                book.rows[uid+'/plan']['artifact_error']='saved plan time/strategies unavailable';book.save();plans[uid]=None
        rec.write(output/'plans_frozen.json',dict(status='ALL_PLANS_TERMINAL',count=5,truth_opened=False))
        for item in rec.items():
            uid,n=item['id'],item['length'];cache={};tech=[]
            try: rgb=np.load(output/f'cache/{uid}/rgb.npy',mmap_mode='r',allow_pickle=False) if book.available(uid+'/RGB') else None
            except (OSError,ValueError) as error:
                book.rows[uid+'/RGB']['artifact_error']=str(error);book.save();rgb=None
            for bank in rec.banks(item):
                bid=bank['view_id'];folder=f'bank/{bid}'
                model_available=True
                if book.rows[bid+'/E41']['status']=='NOT_STARTED' and book.available(uid+'/RGB'):
                    try: backend.prepare_payload()
                    except Exception as error:
                        model_available=False
                        book.rows[bid+'/E41'].update(dependencies_unavailable=['shared_model_load'],dependency_error=str(error));book.save()
                latent=saved_array(bid+'/E41',folder+'/latent.pt',lambda:backend.encode_view(rgb,bank['start']),(uid+'/RGB',),True) if model_available else None
                raw[bid]['E41']=book.observation_status(bid+'/E41')
                projection=js(bid+'/project',folder+'/projection.json',lambda:backend.project(latent),(bid+'/E41',));del latent
                raw[bid]['project']=book.observation_status(bid+'/project')
                for key in ('key00','key01'):
                    decoded=js(bid+'/'+key,folder+'/'+key+'.json',lambda:backend.bits(projection,bid,key),(bid+'/project',))
                    raw[bid]['keys'][key]=book.observation_status(bid+'/'+key)
                    if decoded is not None:
                        try:
                            header=header_only(decoded,dict(key_id=key))
                            if header['view_id']!=bid: raise ValueError('decoded view address differs from actual E41')
                            flat,errors=cache_from_payload(decoded,dict(key_id=key),header);tech.extend(errors)
                            for eid,keyed in flat.items(): cache.setdefault(eid,{}).update(keyed)
                        except (ValueError,KeyError,TypeError) as error: tech.append(dict(kind='RAW_RECEIPT',view=bid,key=key,error=str(error)))
                raw[bid]['status']='RETURNED' if raw[bid]['E41']==raw[bid]['project']=='RETURNED' and all(v=='RETURNED' for v in raw[bid]['keys'].values()) else 'INCOMPLETE'
                rec.write(raw_path,raw)
            frozen=plans[uid]
            if frozen is not None:
                for strategy,group in frozen['strategies'].items():
                    key=protocol.key_for(strategy);local=[];memo={}
                    receipts={bid:dict(row,status='RETURNED' if row.get('E41')==row.get('project')==row['keys'].get(key)=='RETURNED' else 'TECHNICAL_UNREAD') for bid,row in raw.items()}
                    for q,plan in enumerate(group):
                        attached=previous.attach_receipts(plan,receipts)
                        for group_row in attached['groups']:
                            for ref in group_row['references']:
                                operations=[ref['view_id']+'/'+op for op in ('E41','project',key)]
                                ref.update(source_package='paper-joint-t12/run02',original_operations=operations,
                                    original_operation_statuses={op:book.observation_status(op) for op in operations},
                                    original_bits_file=str(Path('bank')/ref['view_id']/(key+'.json')))
                        local.append(js(f'{uid}/{strategy}/q{q}',f'local/{uid}/{strategy}/q{q}.json',
                            lambda:cdc.read_time_local(attached,cache,memo=memo),(uid+'/plan',)))
                    final=js(f'{uid}/{strategy}/aggregate',f'final/{uid}/{strategy}.json',lambda:cdc.aggregate_time(local),
                        tuple(f'{uid}/{strategy}/q{q}' for q in range(3)))
                    if strategy in ('B_NEW','K_NEW') and final is not None:
                        unit=dict(input_id=uid,media_id=uid,source_package='paper-joint-t12/run02',arm='NEW',view=item['view'],key_id=key)
                        errors=[] if frozen['time']['status']=='READY' else [dict(kind='TIME_INPUT',reason=frozen['time']['status'])]
                        evidence_final=copy.deepcopy(final) if isinstance(final,dict) else {}
                        for slot in evidence_final.get('slots',[]) if isinstance(evidence_final.get('slots'),list) else []:
                            if not isinstance(slot,dict): continue
                            for ref in slot.get('supporting_refs',[]) if isinstance(slot.get('supporting_refs'),list) else []:
                                if not isinstance(ref,dict): continue
                                ref['original_operation_statuses']={op:book.observation_status(op) for op in ref.get('original_operations',[]) if op in book.rows}
                        ev=evidence_from_aggregate(evidence_final,unit,input_errors=errors,technical_errors=tech)
                        ev['geometry_scope']='BLIND_RELATIVE_TIME_CONDITIONAL';rec.write(output/f'evidence/{uid}/{key}.json',ev)
            # Same candidate RGB reused; only the five original P0 references may be decoded.
            ref=Path(config['original'])/'media'/item['reference']/'received.mp4'
            js(uid+'/ref/probe',f'reference/{uid}/probe.json',lambda:backend.probe(ref,n))
            ref_rgb=saved_array(uid+'/ref/RGB',f'reference/{uid}/rgb.npy',lambda:backend.decode(ref,n,'RGB'),(uid+'/ref/probe',))
            js(uid+'/quality',f'quality/{uid}.json',lambda:dict(quality_pair(rgb,ref_rgb),reference=item['reference'],
                scope='TIME_PACKET_INCREMENT_CANDIDATE_P0',SSIM='NOT_MEASURED',LPIPS='NOT_MEASURED'),(uid+'/RGB',uid+'/ref/RGB'))
            del rgb,ref_rgb,cache
        rec.write(output/'identity_finals_frozen.json',dict(status='ALL_IDENTITY_FINALS_TERMINAL',planned=15))
        for item in rec.items():
            uid=item['id']
            for key,strategy in (('key00','B_NEW'),('key01','K_NEW')):
                for claim in ('q0','q1'):
                    js(f'{uid}/{key}/{claim}',f'claims/{uid}/{key}/{claim}.json',
                        lambda:qmatch.match_claim(rec.read(output/f'evidence/{uid}/{key}.json'),claim,protocol.query_bits(claim)),
                        (f'{uid}/{strategy}/aggregate',))
        book.close_blind()
        from .b_t12_saved_candidate_media_v1_evaluation import evaluate
        js('post','evaluation_summary.json',lambda:evaluate(output,config,book.rows))
    except (KeyboardInterrupt,SystemExit): interrupted=True;raise
    finally:
        if backend is not None:
            try: rec.write(output/f'resources/{time.time_ns()}.json',backend.close())
            except Exception as error: rec.write(output/'close_error.json',dict(error=str(error)))
        rec.write(output/'invocation.json',dict(wall_seconds=time.monotonic()-began,interrupted=interrupted,resource_hard_caps=None))
        result=book.finish();book.close()
    return result
