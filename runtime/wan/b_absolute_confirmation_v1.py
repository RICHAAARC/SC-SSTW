"""Card99 once-only random-role physical confirmation package. Import is stdlib only."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import time
from main.tube_state import b_absolute_confirmation_v1 as method
from main.tube_state import b_phase_centered_saved_v1 as arithmetic
from runtime.wan.b_fixed_detection_confidence_v1 import dump,flush,execution_artifacts as previous_artifacts,existing_attempt as previous_existing,Journal as PreviousJournal

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_absolute_confirmation_v1.json'
BUDGET=ROOT/'experiments/wan_state_clock/configs/b_absolute_confirmation_v1_budget.json'
KEYS={'key00':'watermark','key01':'watermark-wrong'}
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Absolute-Compatibility-V1/fixed-four-unseen-v1'


def execution_artifacts(output):
    output = Path(output)
    names = ('attempt.json', 'result.json', 'checkpoint.json', 'ledger.json', 'thresholds.json', 'evaluation.json', 'protocol.json', 'roles.json', 'role_draw_receipt.json')
    found = [n+s for n in names for s in ('', '.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ('hosts', 'receipts', 'markers', 'blind', 'private', 'decisions', 'baselines'):
        root = output/folder
        if root.is_dir():
            found.extend(str(p.relative_to(output)) for p in root.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def existing_attempt(output,found=None):
    out=previous_existing(output,found)
    for n in ('roles.json','role_draw_receipt.json'):
        try:
            if (Path(output)/n).is_file():out['saved'][n]=json.loads((Path(output)/n).read_text())
        except (OSError,ValueError) as e:out['read_errors'][n]=str(e)
    return out


def load_config(path=CONFIG):
    cfg=json.loads(Path(path).read_text())
    if [h['id'] for h in cfg['hosts']]!=[f'b97-s{i:02}' for i in range(1,5)] or [h['seed'] for h in cfg['hosts']]!=list(range(2026102201,2026102205)):
        raise ValueError('Card99 fixed four source roster required')
    if any('role' in h or not isinstance(h['prompt'],str) or not h['prompt'] or h['negative_prompt']!='' for h in cfg['hosts']):raise ValueError('unassigned four prompts required')
    if cfg['origins']!=list(method.ORIGINS) or cfg['writer']['key']!=KEYS['key00'] or cfg['writer']['gamma']!=1.0:raise ValueError('fixed origins/key/gamma required')
    if cfg['generation']!=dict(height=320,width=512,frames=181,fps=8,steps=50,guidance_scale=5.0,max_sequence_length=512):raise ValueError('fixed native media/generation semantics required')
    if len(cfg['writer']['bits'])!=32 or any(type(x) is not int or x not in (0,1) for x in cfg['writer']['bits']):raise ValueError('32 binary writer bits required')
    return cfg


def assign_roles(config,output,picker=None):
    """Future real run only. Tests inject a fixed selector; preparation never draws."""
    receipt=dict(slot='role_draw',kind='uniform_role_draw',status='STARTED',attempts=1,started_unix=time.time())
    dump(output/'role_draw_receipt.json',receipt)
    try:
        pairs=list(combinations([h['id'] for h in config['hosts']],2))
        if picker is None:
            import secrets
            picker=secrets.choice
        pair=tuple(picker(pairs))
        if pair not in pairs:raise ValueError('role selector did not return one of six unordered pairs')
        assignment={h['id']:'calibration' if h['id'] in pair else 'confirmation' for h in config['hosts']}
        roles=dict(status='PERSISTED_BEFORE_OBSERVATIONS',calibration=list(pair),confirmation=[h for h in assignment if h not in pair],
            assignment=assignment,pair_count=6,redraw=False,saved_unix=time.time())
        dump(output/'roles.json',roles)
        receipt['status']='RETURNED'
        return roles
    except BaseException as e:
        receipt.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        receipt['finished_unix']=time.time();dump(output/'role_draw_receipt.json',receipt)


def units(config):
    u={'role_draw':'uniform_role_draw','calibration':'supervised_score_only_calibration','evaluation':'post_confirmation_evaluation'}
    for host in config['hosts']:
        h=host['id'];payload=host['role']=='confirmation'
        for label,kind in [('generation_load','generation_pipeline_load'),('generate/OFF','source_50step_trajectory'),('generate/P0','source_50step_trajectory'),('construct','PM_construct'),('vae_load','VAE_load')]:u[h+'/'+label]=kind
        for key in KEYS:u[h+'/book/'+key]='public_codebook'
        for arm in method.ARMS:
            for label,kind in [('D','ordinary_D181'),('preclip','preclip_record'),('quantize','rgb8_quantize'),('encode_mp4','MP4_encode'),('probe','MP4_probe'),('decode_mp4','MP4_decode'),('load_RGB','RGB_cache_load')]:u[f'{h}/{arm}/{label}']=kind
            for oi in range(6):
                for p in range(4):
                    v=f'{h}/{arm}/r{oi}/p{p}'
                    u[v+'/prepare']='E41_view_preparation';u[v+'/E']='E41'
                    if payload:u[v+'/FFT']='receiver_FFT'
                    for key in KEYS:
                        for label in ('Q','score'):u[f'{v}/{key}/{label}']=label
                        if payload:u[f'{v}/{key}/votes']='votes'
        for pair in ('OFF_to_P0','P0_to_PM'):
            for layer in ('preclip','rgb8','mp4'):u[f'{h}/quality/{pair}/{layer}']='paired_quality'
        for q in method.query_metadata():
            u[h+'/family/'+q]='score_family_binding'
            if payload:
                u[h+'/B0/'+q]='B0_unique_max_reference'
                u[h+'/decision/'+q]='absolute_compatibility_read'
    return u


class Journal(PreviousJournal):
    def __init__(self,output,config,backend):
        self.output,self.backend=Path(output),backend
        self.data={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units(config).items()};self.events={}
        self.data['role_draw']=json.loads((self.output/'role_draw_receipt.json').read_text())
        self.save()
    def call(self,slot,function,dependencies=()):
        if self.data[slot]['status']=='NOT_ENTERED' and all(self.data[d]['status']=='RETURNED' for d in dependencies):self.data[slot]['attempts']=1
        return super().call(slot,function,dependencies)


def score_table(starts):
    t=arithmetic.blank_table(starts)
    del t['candidates']  # No all-H payload reductions or payload-completeness gate.
    return t


def run_host(host, config, output, backend, journal):
    h = host['id']
    directory = output/'hosts'/h
    directory.mkdir(parents=True, exist_ok=True)
    counters, terminals, books = Counter(), {}, {}
    def call(name, function, deps=()):
        return journal.call(h+'/'+name, function, [h+'/'+d for d in deps])
    def record(name, function, rel, deps=()):
        def execute():
            value = function()
            dump(directory/rel, value)
            return value
        return call(name, execute, deps)
    def tensor(rel, value):
        backend.save_tensor(directory/rel, value)
        return value
    phase_tables, local_bits = {}, {}
    families = {qid: method.empty_family() for qid in method.query_metadata()}
    dump(directory/'preparation.json', dict(host=host, queries=method.query_metadata()))
    try:
        for key, secret in KEYS.items():
            book = call('book/'+key, lambda secret=secret: backend.book(secret))
            if book is not None:
                books[key] = book
        loaded = record('generation_load', lambda: backend.generation_load(host, config), 'private/generation_load.json')
        try:
            for arm in ('OFF', 'P0'):
                def generate(arm=arm):
                    traces = []
                    def step(row):
                        traces.append(row)
                        dump(directory/'private/traces'/(arm+'.json'), traces)
                    z, receipt = backend.generate(arm, config['writer'], lambda k, done: journal.event(h, k, done), step)
                    tensor('private/'+arm+'_terminal.pt', z)
                    dump(directory/'private'/(arm+'_generation.json'), receipt)
                    return z
                z = call('generate/'+arm, generate, ['generation_load'])
                if z is not None:
                    terminals[arm] = z
        finally:
            backend.release_generation()
        def construct():
            z, receipt = backend.construct(terminals['P0'], books['key00'], config['writer']['gamma'])
            tensor('private/PM_terminal.pt', z)
            dump(directory/'private/construction.json', receipt)
            return z
        pm = call('construct', construct, ['generate/P0', 'book/key00'])
        if pm is not None:
            terminals['PM'] = pm
        vae = record('vae_load', backend.load_vae, 'private/vae.json') if terminals else None
        try:
            # Materialize each arm once; no full181 E or latent reader.
            for arm in method.ARMS:
                def decode(arm=arm):
                    value = backend.decode(terminals[arm])
                    tensor('media/'+arm+'/preclip.pt', value['preclip'])
                    tensor('media/'+arm+'/float_rgb.pt', value['float_rgb'])
                    return value
                deps = ['vae_load', 'construct' if arm == 'PM' else 'generate/'+arm]
                decoded = call(arm+'/D', decode, deps)
                record(arm+'/preclip', lambda: backend.preclip(decoded['preclip']), 'quality/'+arm+'_preclip.json', [arm+'/D'])
                rgb = call(arm+'/quantize', lambda: tensor('media/'+arm+'/rgb8.pt', backend.quantize(decoded['float_rgb'])), [arm+'/D'])
                path = directory/'media'/arm/'video.mp4'
                record(arm+'/encode_mp4', lambda: backend.save_mp4(rgb, path), 'media/'+arm+'/codec.json', [arm+'/quantize'])
                probe = record(arm+'/probe', lambda: backend.probe(path), 'media/'+arm+'/probe.json', [arm+'/encode_mp4'])
                call(arm+'/decode_mp4', lambda: tensor('media/'+arm+'/readback.pt', backend.read_mp4(path, probe)), [arm+'/probe'])
                del decoded, rgb
                pixels = call(arm+'/load_RGB', lambda: backend.load_pixels(directory/'media'/arm/'readback.pt'), [arm+'/decode_mp4'])
                for oi, origin in enumerate(method.ORIGINS):
                    for p in range(4):
                        prefix = f'{arm}/r{oi}/p{p}'
                        view = call(prefix+'/prepare', lambda: backend.prepare_view(pixels, origin, p), [arm+'/load_RGB'])
                        z = call(prefix+'/E', lambda: tensor('features/'+prefix+'.pt', backend.encode_view(view)), [prefix+'/prepare', 'vae_load'])
                        tables = {key: score_table(range(1 if p == 0 else 2, 36)) for key in KEYS}
                        for key in KEYS:
                            phase_tables[arm, oi, key, p] = tables[key]
                        for key in KEYS:
                            rel = 'blind/'+prefix+'/'+key
                            q = record(prefix+'/'+key+'/Q', lambda key=key: backend.observe(z, books[key]), rel+'/Q.json', [prefix+'/E', 'book/'+key])
                            call(prefix+'/'+key+'/score', lambda key=key: backend.score(q, tables[key], counters), [prefix+'/'+key+'/Q'])
                            slot = h+'/'+prefix+'/'+key+'/score'
                            tables[key]['score_call_status'] = journal.data[slot]['status']
                            if journal.data[slot]['status'] != 'RETURNED':
                                arithmetic.mark_unfinished(tables[key], 'score', journal.data[slot]['status'], slot)
                            dump(directory/(rel+'/table.json'), tables[key])
                        if host['role']=='confirmation':
                            spec = call(prefix+'/FFT', lambda: backend.spectrum(z), [prefix+'/E'])
                            for key, secret in KEYS.items():
                                rel = 'blind/'+prefix+'/'+key
                                votes = record(prefix+'/'+key+'/votes', lambda secret=secret: backend.votes(spec, secret), rel+'/votes.json', [prefix+'/FFT'])
                                if votes is not None:
                                    local_bits[arm, oi, key, p] = [bit.get('decoded') for row in votes['local_cells'] for bit in row['bits']]
                            del spec
                        del z, view, tables
                del pixels
        finally:
            backend.release_vae()
        for pair in ('OFF_to_P0', 'P0_to_PM'):
            left, right = pair.split('_to_')
            for layer, label in [('preclip', 'preclip'), ('rgb8', 'rgb8'), ('mp4', 'readback')]:
                ld = left+'/D' if layer == 'preclip' else left+('/quantize' if layer == 'rgb8' else '/decode_mp4')
                rd = right+'/D' if layer == 'preclip' else right+('/quantize' if layer == 'rgb8' else '/decode_mp4')
                record('quality/'+pair+'/'+layer, lambda: backend.quality(
                    backend.load_tensor(directory/'media'/left/(label+'.pt')),
                    backend.load_tensor(directory/'media'/right/(label+'.pt')), pair+'/'+layer),
                    'quality/'+pair+'/'+layer+'.json', [ld, rd])
    finally:
        # Bind every slot even after an interrupt; missing observations stay fixed rows.
        for (arm, oi, key, p), table in phase_tables.items():
            slot=f'{h}/{arm}/r{oi}/p{p}/{key}/score'
            table['score_call_status']=journal.data[slot]['status']
            if table['score_call_status']!='RETURNED':
                arithmetic.mark_unfinished(table,'score',table['score_call_status'],slot)
                dump(directory/f'blind/{arm}/r{oi}/p{p}/{key}/table.json',table)
        for qid,meta in method.query_metadata().items():
            oi=method.ORIGINS.index(meta['origin'])
            tables={p:phase_tables.get((meta['arm'],oi,meta['key'],p),{}) for p in range(4)}
            raw={str(p):local_bits.get((meta['arm'],oi,meta['key'],p),[None]*80) for p in range(4)} if host['role']=='confirmation' else {}
            family=method.bind(tables,raw)
            def save_family():
                dump(directory/'blind/families'/(qid+'.json'),family)
                return family
            saved=call('family/'+qid,save_family)
            families[qid]=saved or method.empty_family('FAMILY_SAVE_FAILED')
        dump(directory/'cpu_counts.json',dict(counters))
        dump(directory/'blind_complete.json',dict(status='SCORES_COMPLETE' if all(f['score_complete'] for f in families.values()) else 'SCORE_INCOMPLETE',
            saved_unix=time.time(),query_denominator=36,hypothesis_denominator=4932,
            payload_expected=host['role']=='confirmation',semantic_truth_not_read_by_receiver=True))
        journal.save()
    return families


def run_b0(host,families,thresholds,output,backend,journal):
    h=host['id'];out={};counters=Counter()
    for qid,family in families.items():
        def reference():
            selected,reason=method.unique_max(family)
            if selected is None:value=method.b0_shell(family,reason=reason)
            else:
                meta=method.query_metadata()[qid]
                oi=method.ORIGINS.index(meta['origin']);p=selected['phase']
                prefix=f"{h}/{meta['arm']}/r{oi}/p{p}/{meta['key']}"
                if journal.data[prefix+'/votes']['status']!='RETURNED':value=method.b0_shell(family,reason='TECHNICAL_UNRESOLVED')
                else:
                    path=output/'hosts'/h/'blind'/meta['arm']/f'r{oi}'/f'p{p}'/meta['key']/'votes.json'
                    with path.open() as stream:votes=json.load(stream)
                    reduced=backend.reduce_candidate(votes,selected['source_start'],counters)
                    value=method.b0_shell(family,selected,reduced)
            gated=method.gate_b0(value,thresholds)
            dump(output/'baselines'/h/(qid+'.json'),value)
            dump(output/'baselines'/h/(qid+'.gated.json'),gated)
            return dict(ungated=value,gated=gated)
        out[qid]=journal.call(h+'/B0/'+qid,reference) or dict(ungated=method.b0_shell(family),gated=method.b0_shell(family))
    dump(output/'hosts'/h/'B0_counts.json',dict(counters))
    return out


def run(output=None,*,config_path=CONFIG,backend=None,role_picker=None):
    output=Path(output or DEFAULT_OUTPUT)
    found=execution_artifacts(output)
    if found:return existing_attempt(output,found)
    config=load_config(config_path);budget=json.loads(BUDGET.read_text())
    output.mkdir(parents=True,exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(card=99,started_unix=time.time(),scope='one complete attempt; failures/interruption consumed; no retry'),stream);flush(stream)
    except FileExistsError:return existing_attempt(output)
    result=dict(status='RUNNING',card=99,scientific_PASS=None,new_physical_run_entered=False,default_changed=False,A_changed=False,
        fixed_confirmation_denominators=budget['fixed_confirmation_denominators'],source_slots={h['id']:'NOT_ENTERED' for h in config['hosts']})
    dump(output/'result.json',result)
    journal=None;thresholds=dict(status='CALIBRATION_UNRESOLVED',tau=None,L=None);families={};decisions={};baselines={}
    try:
        roles=assign_roles(config,output,role_picker)
        config=dict(config,hosts=[dict(h,role=roles['assignment'][h['id']]) for h in config['hosts']])
        journal=Journal(output,config,backend)
        dump(output/'protocol.json',dict(card=99,config=config,planned_budget=budget,source_sampling='fixed four unseen tuples; uniform random two calibration',population_FPR_guarantee=None))
        if backend is None:
            from runtime.wan.b_absolute_confirmation_backend_v1 import WanBackend
            backend=WanBackend(config['model_id']);journal.backend=backend
        result['new_physical_run_entered']=True
        for host in config['hosts']:
            if host['role']=='calibration':
                result['source_slots'][host['id']]='ENTERED'
                families[host['id']]=run_host(host,config,output,backend,journal)
        dump(output/'markers/calibration_blind_complete.json',dict(saved_unix=time.time(),hosts=roles['calibration'],confirmation_sourceprep_entered=False))
        thresholds=journal.call('calibration',lambda:method.calibrate({h:families[h] for h in roles['calibration']})) or thresholds
        thresholds['frozen_unix']=time.time() if thresholds['status']=='FROZEN' else None
        dump(output/'thresholds.json',thresholds)  # Actual save BEFORE any confirmation sourceprep.
        if thresholds['status']=='FROZEN':
            for host in config['hosts']:
                if host['role']=='confirmation':
                    result['source_slots'][host['id']]='ENTERED'
                    families[host['id']]=run_host(host,config,output,backend,journal)
                    baselines[host['id']]=run_b0(host,families[host['id']],thresholds,output,backend,journal)
        for host in config['hosts']:
            if host['role']!='confirmation':continue
            h=host['id'];families.setdefault(h,{q:method.empty_family('NOT_ENTERED_CALIBRATION_UNRESOLVED') for q in method.query_metadata()})
            baselines.setdefault(h,{q:dict(ungated=method.b0_shell(families[h][q]),gated=method.b0_shell(families[h][q])) for q in method.query_metadata()})
            decisions[h]={}
            for qid,family in families[h].items():
                def decide():
                    value=method.decision(family,thresholds)
                    dump(output/'decisions'/h/(qid+'.json'),value)
                    return value
                decisions[h][qid]=journal.call(h+'/decision/'+qid,decide)
        slots=[h+'/'+label+'/'+q for h in roles['confirmation'] for label in ('decision','B0') for q in method.query_metadata()]
        unsaved=[s for s in slots if journal.data[s]['status']!='RETURNED']
        if len(slots)!=144 or unsaved:
            result['post_status']='NOT_ENTERED_UNSAVED_BLIND_OUTPUTS';journal.data['evaluation']['unavailable_from']=unsaved
        else:
            dump(output/'markers/confirmation_blind_complete.json',dict(saved_unix=time.time(),new_decisions=72,B0_decisions=72,
                all_blind_outputs_saved=True,confirmation_message_truth_not_read=True,threshold_freeze=thresholds['frozen_unix']))
            def post():
                # Read actual files before making the semantic message argument available.
                saved={h:{} for h in roles['confirmation']};base={h:{} for h in roles['confirmation']}
                for h in roles['confirmation']:
                    for q in method.query_metadata():
                        saved[h][q]=json.loads((output/'decisions'/h/(q+'.json')).read_text())
                        base[h][q]=dict(ungated=json.loads((output/'baselines'/h/(q+'.json')).read_text()),
                            gated=json.loads((output/'baselines'/h/(q+'.gated.json')).read_text()))
                dump(output/'post_truth_access.json',dict(opened_unix=time.time(),all_blind_outputs_saved=True))
                from runtime.wan.b_absolute_confirmation_evaluation_v1 import evaluate
                value=evaluate(saved,base,{h:families[h] for h in roles['confirmation']},thresholds,config['writer']['bits'],roles)
                dump(output/'evaluation.json',value)
                return dict(status=value['status'])
            journal.call('evaluation',post)
        technical=sum(d['detection'] in ('UNRESOLVED_SCORE','CALIBRATION_UNRESOLVED') or any(m['status']=='TECHNICAL_UNRESOLVED' for m in d['message']) for ds in decisions.values() for d in ds.values() if d is not None)
        baseline_technical=sum(b['ungated']['status']=='TECHNICAL_UNRESOLVED' for bs in baselines.values() for b in bs.values())
        result.update(technical_queries=technical,technical_B0_queries=baseline_technical,
            compatibility_counts={k:sum(d['counts'][k] for ds in decisions.values() for d in ds.values() if d is not None)
                for k in ('member_slots','numeric_member_comparisons','group_slots','message_slots','cached_single_cell_bit_accesses','selected_canonical_cells')})
        complete=(thresholds['status']=='FROZEN' and technical==0 and baseline_technical==0 and all(row['status']=='RETURNED' for row in journal.data.values()))
        result['status']='COMPLETED_FINITE_ROSTER_EMPIRICAL_ONLY' if complete else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    except BaseException as e:
        result.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
    finally:
        if journal is not None:
            result.update(counts=journal.counts(),physical_events=journal.events)
            journal.save()
            result['new_physical_run_entered']=any(v['kind']=='generation_pipeline_load' and v['status']!='NOT_ENTERED' for v in journal.data.values())
        result['threshold_status']=thresholds['status']
        dump(output/'result.json',result)
    return result
