"""Card89 user-only E6/FFT6 completion with saved Card88 prefixes/280 reducers."""
from pathlib import Path, PurePosixPath
from collections import Counter
import json
import stat
import time
import zipfile
import torch
from main.tube_state import b_real_four_phase_v1 as physical
from main.tube_state import b_phase_centered_saved_v1 as cpu
from runtime.wan import b_real_local_wan_v1 as prior

dump = prior.dump
DEFAULT_MODEL_ID = prior.DEFAULT_MODEL_ID
KEY_IDS = physical.KEY_IDS
# Preparation/execution cache map only. Numerical functions receive no arm/origin.
SAVED = [('u00', 'c00'), ('u01', 'c01'), ('u05', 'c02'), ('u06', 'c03')]
NEW = [('u02', 'P0', 62), ('u03', 'P0', 63), ('u04', 'P0', 64),
       ('u07', 'PM', 62), ('u08', 'PM', 63), ('u09', 'PM', 64)]
BINDINGS = {'c00':['u00', 'u01', 'u02', 'u03'], 'c01':['u01', 'u02', 'u03', 'u04'],
            'c02':['u05', 'u06', 'u07', 'u08'], 'c03':['u06', 'u07', 'u08', 'u09']}


def fixed_denominator():
    return dict(VAE_load=1, cached_RGB_loads=2, E=6, FFT=6, nominal_encoder_chunks=66,
        decoder_chunks=0, D=0, DiT=0, writer=0, codec=0, MP4_decode=0, gradients=0,
        saved_prefix_observations=4, new_E41_observations=6, logical_phase_views=16,
        saved_Q_vote_tables=8, new_Q_vote_tables=12, unique_Q_vote_tables=20, logical_keyed_tables=32,
        new_ROI_dots=21120, dot_dimension=768, new_signed_votes=115200, new_zero_masks=115200,
        new_raw_bits=960, new_half_means=816, new_half_terms=8160, new_raw_means=408,
        new_raw_terms=4080, new_direct_means=1224, new_direct_terms=12240,
        new_centered_scores=408, new_min_statistics=408, new_reducers=408,
        new_message_rows=13056, new_nonempty_group_bit_counters=5856, new_candidate_vote_terms=3916800,
        new_candidate_cell_rows=4080, new_candidate_source_rows=17952,
        saved_candidates_reused=280, unique_candidates=688, unique_message_rows=22016,
        queries=8, hypotheses_per_query=137, logical_hypotheses=1096, logical_message_rows=35072,
        per_variant_output_message_slots=256, per_variant_output_source_rows=352, ledger_slots=len(units()))


def protocol():
    return dict(card=89, scope='one already-seen host actual phase observation feasibility only',
        phase_offsets=[0, 1, 2, 3], E_frames=41, read_latents=list(range(1, 11)),
        source_starts={'p0':[1, 35], 'p1_to_p3':[2, 35]}, variants=list(cpu.VARIANTS),
        saved_prefix='same-reset E45 rows1..10; not fresh E41; no prior280 score/reducer replay',
        no_cross_input_decision=True, no_threshold=True, no_true_message_selection=True,
        writer_changed=False, gamma=1.0, net_bits=32, resource_hard_caps=None,
        fixed_denominator=fixed_denominator(), identity_status='UNCALIBRATED',
        phase_status='UNCALIBRATED', FPR=None, trusted_full_messages=None, scientific_pass=None)


def units():
    u={'model':'vae_load', 'post/truth':'post_truth_read', 'post/metadata':'post_metadata_read',
       'post/card88':'post_saved_comparison_read', 'evaluation':'post_evaluation'}
    for arm in ('P0', 'PM'):
        u['load/'+arm]='cached_RGB_load'
    for key in KEY_IDS:
        u['book/'+key]='public_codebook'
    for uid, _ in SAVED:
        for key in KEY_IDS:
            for kind in ('Q', 'votes', 'table'):
                u[f'reuse_{kind}/{uid}/{key}']='saved_blind_'+kind
    for uid, _, _ in NEW:
        u['prepare/'+uid]='received_RGB_preparation'
        u['E/'+uid]='posterior_encode'
        u['FFT/'+uid]='shared_payload_fft'
        for key in KEY_IDS:
            for kind in ('Q', 'votes', 'score', 'reduce'):
                u[f'{kind}/{uid}/{key}']=dict(Q='new_direction_observation', votes='new_keyed_vote_read',
                    score='new_centered_scores', reduce='new_conditional_reduction')[kind]
    for case in BINDINGS:
        for key in KEY_IDS:
            u[f'family/{case}/{key}']='logical_family_binding'
    return u


class Journal(prior.Journal):
    def __init__(self, output, backend):
        self.output, self.backend = output, backend
        self.data = {k:dict(kind=v, status='NOT_ENTERED') for k,v in units().items()}
        self.save()


class WanBackend(prior.WanBackend):
    def encode_received(self, pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        if tuple(pixels.shape) != (41, 320, 512, 3) or pixels.dtype != torch.uint8:
            raise ValueError('actual41-frame RGB uint8 view required')
        z = reencode_rgb24_readback(self.vae, pixels.float()/255).detach().cpu().float()
        physical.validate_local(z)
        return z


def prepare_received(pixels, start):
    if tuple(pixels.shape) != (181, 320, 512, 3) or pixels.dtype != torch.uint8:
        raise ValueError('actual cached181-frame RGB uint8 required')
    if start < 0 or start+41 > len(pixels):
        raise ValueError('complete41-frame prefix view required; no padding')
    return pixels[start:start+41].clone()


def read_member(archive, member):
    name = PurePosixPath(member)
    if name.is_absolute() or '..' in name.parts or '\\' in member:
        raise ValueError('unsafe sidecar member path')
    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(member)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('unsafe sidecar symbolic link')
        return json.loads(bundle.read(info))


def validate_saved(kind, value):
    if kind == 'Q':
        if len(value['roi_q']) != 10 or len(value['pooled_q']) != 10:
            raise ValueError('saved ten-row Q prefix required')
        for roi, pooled in zip(value['roi_q'], value['pooled_q']):
            if len(roi) != 4 or len(pooled) != 44 or any(len(x) != 44 for x in roi):
                raise ValueError('saved prefix Q shape10x4x44 required')
            if any(not cpu.finite_number(x) for row in roi for x in row) or any(not cpu.finite_number(x) for x in pooled):
                raise ValueError('finite saved prefix Q required')
    elif kind == 'votes':
        signed, zeros = value['signed_votes'], value['zero_coefficient_mask']
        if len(signed) != 4 or len(zeros) != 4 or len(value['local_cells']) != 10:
            raise ValueError('saved prefix vote shape required')
        for a,b in zip(signed, zeros):
            if len(a) != 10 or len(b) != 10:
                raise ValueError('ten saved vote times required')
            for x,y in zip(a,b):
                if len(x) != 240 or len(y) != 240 or any(type(v) is not int or v not in (-1,1) for v in x) or any(type(v) is not int or v not in (0,1) for v in y):
                    raise ValueError('valid240 signed/zero entries required')
                if any(z and v != -1 for v,z in zip(x,y)):
                    raise ValueError('zero coefficient must have strict-negative sign vote')
        for row in value['local_cells']:
            if len(row['bits']) != 8 or any(type(x['decoded']) is not int or x['decoded'] not in (0,1) for x in row['bits']):
                raise ValueError('saved raw8-bit local rows required')
    else:
        if [x['source_start'] for x in value['scores']] != list(range(1,36)) or [x['source_start'] for x in value['candidates']] != list(range(1,36)):
            raise ValueError('all35 original saved candidate slots required')
        for row in value['scores']:
            if row['status'] == 'RETURNED' and any(not cpu.finite_number(row[v]) for v in cpu.VARIANTS):
                raise ValueError('finite returned saved score required')
        for row in value['candidates']:
            if len(row['cells']) != 10 or len(row['source_occupancy']) != 44 or len(row['decoded_bits']) != 32 or len(row['segments']) != 4 or any(len(x['bits']) != 8 for x in row['segments']):
                raise ValueError('fixed saved10/44/32 candidate rows required')
    return value


def execution_artifacts(output):
    output=Path(output)
    names=('attempt.json','protocol.json','ledger.json','checkpoint.json','counts.json','numeric_completion.json','result.json','evaluation.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ('blind','features','preparation','references','evaluated'):
        root=output/folder
        if root.is_dir():
            found.extend(str(p.relative_to(output)) for p in root.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def unavailable_q(status='NOT_ENTERED', root=None):
    return dict(status=status, unavailable_from=root, roi_q=[[[None]*44 for _ in range(4)] for _ in range(10)],
        pooled_q=[[None]*44 for _ in range(10)],retained_rows=list(range(1,11)),truth_inputs=False)


def unavailable_votes(status='NOT_ENTERED', root=None):
    return dict(status=status,unavailable_from=root,signed_votes=None,zero_coefficient_mask=None,
        local_cells=[dict(local_row=r,bits=[dict(slot=b,decoded=None,status=status,unavailable_from=root) for b in range(8)]) for r in range(1,11)],
        raw_bit_slots=80,truth_inputs=False,message_inputs=False)


def run(source_root, output, saved_sidecar, truth_path, model_id=DEFAULT_MODEL_ID, *, backend=None, sidecar_reader=read_member):
    output, source_root = Path(output), Path(source_root)
    found = execution_artifacts(output)
    if found:
        result = prior.shared.existing_attempt(output, found)
        result['fixed_denominator'] = fixed_denominator()
        return result
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(card=89, started_unix=time.time(), source_root=str(source_root),
                scope='one fixed E6/FFT6 user attempt; any failure consumes it'), stream)
            prior.shared.flush_file(stream)
    except FileExistsError:
        return prior.shared.existing_attempt(output, ['attempt.json'])
    backend = backend or WanBackend(model_id)
    ledger = Journal(output, backend)
    counters, deps, books, pixels = Counter(), {}, {}, {}
    tables = {uid+'/'+key:cpu.blank_table(range(1,36) if any(uid == a for a,_ in SAVED) else range(2,36))
              for uid in [f'u{i:02}' for i in range(10)] for key in KEY_IDS}
    qs={n:unavailable_q() for n in tables}
    votes={n:unavailable_votes() for n in tables}
    families={}
    for case,views in BINDINGS.items():
        for key in KEY_IDS:
            refs={p:uid+'/'+key for p,uid in enumerate(views)}
            families[case+'/'+key]=physical.bind_family({p:tables[n] for p,n in refs.items()},refs)
    metadata = {f'c{i:02}':dict(arm='P0' if i < 2 else 'PM', start=60+i%2, stop=105+i%2,
        variant=i%2, observation_ids=BINDINGS[f'c{i:02}']) for i in range(4)}
    result = dict(status='RUNNING', fixed_denominator=fixed_denominator(), protocol=protocol(), scientific_pass=None)

    def checkpoint():
        result.update(counts=ledger.counts(), actual_new_CPU=dict(counters))
        dump(output/'checkpoint.json', result)
    def call(slot, fn, dependencies=()):
        deps[slot] = list(dependencies)
        if any(ledger.data[d]['status'] != 'RETURNED' for d in dependencies):
            return None
        return ledger.call(slot, fn)
    def record(slot, fn, path, dependencies=()):
        def execute():
            value=fn()
            dump(output/path, value)
            return value
        return call(slot, execute, dependencies)
    def why(slot):
        def visit(s, seen):
            if s in seen:
                return None
            seen.add(s)
            if ledger.data[s]['status'] in ('FAILED','INTERRUPTED','STARTED'):
                return s
            return next((r for d in deps.get(s,[]) if (r:=visit(d,seen))), None)
        root=visit(slot,set())
        if root is None:
            root=next((s for s,v in ledger.data.items() if v['status']=='INTERRUPTED'),None)
        return ('UPSTREAM_'+ledger.data[root]['status'],root) if root else ('NOT_ENTERED',slot)
    def save_table(name):
        dump(output/'blind/unique'/name/'table.json', tables[name])

    for uid,_,_ in NEW:
        deps['prepare/'+uid]=['load/'+next(a for u,a,_ in NEW if u==uid)]
        deps['E/'+uid]=['model','prepare/'+uid]
        deps['FFT/'+uid]=['E/'+uid]
        for key in KEY_IDS:
            n=uid+'/'+key
            deps['Q/'+n]=['E/'+uid,'book/'+key]
            deps['score/'+n]=['Q/'+n]
            deps['votes/'+n]=['FFT/'+uid]
            deps['reduce/'+n]=['votes/'+n]
    dump(output/'protocol.json', protocol())
    dump(output/'preparation/sidecar.json', metadata)
    for name in tables:
        save_table(name)
        dump(output/'blind/unique'/name/'Q.json',qs[name])
        dump(output/'blind/unique'/name/'votes.json',votes[name])
    for name,value in families.items():
        dump(output/'blind/families'/(name+'.json'),value)
    checkpoint()
    try:
        for uid, old_case in SAVED:
            for key in KEY_IDS:
                name=uid+'/'+key
                for kind, dest in [('Q',qs),('votes',votes),('table',tables)]:
                    value=record('reuse_'+kind+'/'+name,
                        lambda kind=kind,old_case=old_case,key=key:validate_saved(kind,sidecar_reader(saved_sidecar,f'blind/{kind if kind != "table" else "tables"}/{old_case}/{key}.json')),
                        Path('blind/unique')/name/(kind+'.json'))
                    if value is not None:
                        dest[name]=value
                checkpoint()
        for key, secret in zip(KEY_IDS, physical.KEYS):
            value=call('book/'+key,lambda secret=secret:physical.carrier.codebook(secret))
            if value is not None:
                books[key]=value
        for arm in ('P0','PM'):
            value=call('load/'+arm,lambda arm=arm:backend.load_pixels(source_root/'media'/arm/'readback.pt'))
            if value is not None:
                pixels[arm]=value
        result['model']=call('model',backend.load)
        for uid, arm, start in NEW:
            received=call('prepare/'+uid,lambda arm=arm,start=start:prepare_received(pixels[arm],start),deps['prepare/'+uid])
            def encode():
                z=backend.encode_received(received)
                path=output/'features'/(uid+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(z,path)
                return z
            z=call('E/'+uid,encode,deps['E/'+uid])
            for key in KEY_IDS:
                name=uid+'/'+key
                value=record('Q/'+name,lambda key=key:physical.observe(z,books[key]),Path('blind/unique')/name/'Q.json',deps['Q/'+name])
                if value is not None:
                    qs[name]=value
                call('score/'+name,lambda name=name:cpu.score(qs[name],tables[name],counters),deps['score/'+name])
                save_table(name)
            shared_spectrum=call('FFT/'+uid,lambda:physical.spectrum(z),deps['FFT/'+uid])
            for key,secret in zip(KEY_IDS,physical.KEYS):
                name=uid+'/'+key
                value=record('votes/'+name,lambda secret=secret:physical.read_spectrum(shared_spectrum,secret),Path('blind/unique')/name/'votes.json',deps['votes/'+name])
                if value is not None:
                    votes[name]=value
                call('reduce/'+name,lambda name=name:cpu.reduce_all(votes[name],tables[name],counters),deps['reduce/'+name])
                save_table(name)
            checkpoint()
    except KeyboardInterrupt:
        result['fatal_error']='KeyboardInterrupt'
    except Exception as error:
        result['fatal_error']=f'{type(error).__name__}: {error}'
    for uid, _ in SAVED:
        for key in KEY_IDS:
            name=uid+'/'+key
            bad=next((f'reuse_{kind}/{name}' for kind in ('Q','votes','table') if ledger.data[f'reuse_{kind}/{name}']['status']!='RETURNED'),None)
            if bad:
                status,root=why(bad)
                cpu.mark_unfinished(tables[name],'score',status,root)
                cpu.mark_unfinished(tables[name],'reduce',status,root)
                tables[name].update(status='INCOMPLETE',unavailable_from=root)
            for kind,values,make in [('Q',qs,unavailable_q),('votes',votes,unavailable_votes)]:
                if ledger.data[f'reuse_{kind}/{name}']['status']!='RETURNED':
                    status,root=why(f'reuse_{kind}/{name}')
                    values[name]=make(status,root)
                dump(output/'blind/unique'/name/(kind+'.json'),values[name])
            save_table(name)
    for uid,_,_ in NEW:
        for key in KEY_IDS:
            name=uid+'/'+key
            causes={}
            for kind in ('score','reduce'):
                slot=kind+'/'+name
                if ledger.data[slot]['status']!='RETURNED':
                    status,root=why(slot)
                    cpu.mark_unfinished(tables[name],kind,status,root)
                    causes[kind]=root
            tables[name]['status']='INCOMPLETE' if causes else 'NUMERIC_COMPLETE_UNCALIBRATED'
            if causes:
                tables[name]['unavailable_from']=causes
            for kind,values,make in [('Q',qs,unavailable_q),('votes',votes,unavailable_votes)]:
                if ledger.data[kind+'/'+name]['status']!='RETURNED':
                    status,root=why(kind+'/'+name)
                    values[name]=make(status,root)
                dump(output/'blind/unique'/name/(kind+'.json'),values[name])
            save_table(name)
    for case, views in BINDINGS.items():
        for key in KEY_IDS:
            name=case+'/'+key
            references={p:uid+'/'+key for p,uid in enumerate(views)}
            value=record('family/'+name,lambda references=references:physical.bind_family({p:tables[n] for p,n in references.items()},references),Path('blind/families')/(name+'.json'))
            if value is not None:
                families[name]=value
            else:
                status,root=why('family/'+name)
                families[name].update(status='INCOMPLETE_PHASE_FAMILY',unavailable_from=root,error_status=status)
                dump(output/'blind/families'/(name+'.json'),families[name])
    dump(output/'counts.json',dict(counters))
    dump(output/'numeric_completion.json',dict(completed_unix=time.time(),blind_queries=8,
        logical_hypotheses=1096,logical_message_rows=35072,unique_candidates=688,
        all_fixed_blind_rows_persisted=True,truth_read=False,metadata_read=False,card88_post_read=False,
        completed_families=sum(x['status']=='COMPLETE_UNCALIBRATED' for x in families.values()),actual_new_CPU=dict(counters)))
    truth=record('post/truth',lambda:json.loads(Path(truth_path).read_text()),Path('references/truth.json'))
    post_meta=record('post/metadata',lambda:json.loads((output/'preparation/sidecar.json').read_text()),Path('references/metadata.json'))
    old=record('post/card88',lambda:sidecar_reader(saved_sidecar,'post/card88_evaluation.json'),Path('references/card88_evaluation.json'))
    from runtime.wan.b_real_four_phase_evaluation_v1 import evaluate
    evaluated=record('evaluation',lambda:evaluate(families,tables,votes,post_meta,truth,old),Path('evaluation.json'))
    all_complete=len(families)==8 and all(x['status']=='COMPLETE_UNCALIBRATED' for x in families.values())
    post_ok=evaluated is not None and evaluated['status']=='POST_ONLY'
    result.update(status='EXECUTION_COMPLETE' if all_complete and post_ok else 'NUMERIC_COMPLETE_POST_UNAVAILABLE' if all_complete else 'INCOMPLETE',
        counts=ledger.counts(),calls=ledger.data,actual_new_CPU=dict(counters),identity_status='UNCALIBRATED',
        phase_status='UNCALIBRATED',trusted_full_messages=None,FPR=None,scientific_pass=None,
        evaluation_status=evaluated['status'] if evaluated is not None else 'UNAVAILABLE',
        actual_new_observation=dict(E_returned=sum(v['status']=='RETURNED' for n,v in ledger.data.items() if n.startswith('E/')),
            FFT_returned=sum(v['status']=='RETURNED' for n,v in ledger.data.items() if n.startswith('FFT/')),
            Q_tables_returned=sum(v['status']=='RETURNED' for n,v in ledger.data.items() if n.startswith('Q/')),
            ROI_dots_in_returned_Q_tables=1760*sum(v['status']=='RETURNED' for n,v in ledger.data.items() if n.startswith('Q/')),
            raw_local_bit_counters_in_returned_vote_tables=80*sum(v['status']=='RETURNED' for n,v in ledger.data.items() if n.startswith('votes/'))))
    if torch.cuda.is_available():
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(output/'result.json',result)
    checkpoint()
    return result
