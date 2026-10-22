"""Card101 fixed user-run pipeline. All physical work is entered only by user's new attempt."""
from pathlib import Path
import json,time
from collections import Counter
from main.tube_state import b_absolute_confirmation_v1 as layout
from runtime.wan import b_absolute_confirmation_v1 as prior

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_background_normalized_match_v1.json'
BUDGET=ROOT/'experiments/wan_state_clock/configs/b_background_normalized_match_v1_budget.json'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
KEYS=prior.KEYS
dump=prior.dump
flush=prior.flush
def execution_artifacts(output):
    output=Path(output)
    names=('attempt.json','result.json','ledger.json','thresholds.json','background.json','evaluation.json','protocol.json','roles.json','role_draw_receipt.json','blind_complete.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ('hosts','receipts','markers','decisions','baselines','numeric'):
        root=output/folder
        if root.is_dir():found.extend(str(p.relative_to(output)) for p in root.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def existing_attempt(output,found=None):
    value=prior.existing_attempt(output,found)
    try:
        p=Path(output)/'background.json'
        if p.is_file():value['saved']['background.json']=json.loads(p.read_text())
    except (OSError,ValueError) as e:value['read_errors']['background.json']=str(e)
    return value


def load_config(path=CONFIG):
    cfg=json.loads(Path(path).read_text())
    if [h['id'] for h in cfg['hosts']]!=[f'b101-s{i:02}' for i in range(1,5)] or [h['seed'] for h in cfg['hosts']]!=list(range(2026102205,2026102209)):
        raise ValueError('fixed Card101 unseen source tuple roster required')
    if any('role' in h or not h['prompt'] or h['negative_prompt']!='' for h in cfg['hosts']):raise ValueError('unassigned source prompts required')
    if cfg['origins']!=list(layout.ORIGINS) or cfg['writer']['key']!=KEYS['key00'] or cfg['writer']['gamma']!=1:raise ValueError('fixed crop/key/gamma')
    if cfg['generation']!=dict(height=320,width=512,frames=181,fps=8,steps=50,guidance_scale=5.0,max_sequence_length=512):raise ValueError('fixed native media/generation')
    if len(cfg['writer']['bits'])!=32 or any(type(x) is not int or x not in (0,1) for x in cfg['writer']['bits']):raise ValueError('net32 binary writer')
    return cfg


def units(config):
    u={k:v for k,v in prior.units(config).items() if v!='score'}
    u['background']='background_fit'
    for host in config['hosts']:
        for q in layout.query_metadata():u[host['id']+'/score_family/'+q]='normalized_family_score'
    return u


class Journal(prior.Journal):
    def __init__(self,out,config,backend):
        self.output,self.backend=Path(out),backend;self.events={}
        self.data={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units(config).items()}
        self.data['role_draw']=json.loads((self.output/'role_draw_receipt.json').read_text());self.save()


def collect_host(host, config, output, backend, journal):
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
    observations, local_bits = {}, {}

    dump(directory/'preparation.json', dict(host=host, queries=layout.query_metadata()))
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
            for arm in layout.ARMS:
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
                for oi, origin in enumerate(layout.ORIGINS):
                    for p in range(4):
                        prefix = f'{arm}/r{oi}/p{p}'
                        view = call(prefix+'/prepare', lambda: backend.prepare_view(pixels, origin, p), [arm+'/load_RGB'])
                        z = call(prefix+'/E', lambda: tensor('features/'+prefix+'.pt', backend.encode_view(view)), [prefix+'/prepare', 'vae_load'])
                        for key in KEYS:
                            rel = 'blind/'+prefix+'/'+key
                            observations[arm,oi,key,p] = record(prefix+'/'+key+'/Q',lambda key=key:backend.observe(z,books[key]),rel+'/Q.json',[prefix+'/E','book/'+key])
                        if host['role']=='confirmation':
                            spec = call(prefix+'/FFT', lambda: backend.spectrum(z), [prefix+'/E'])
                            for key, secret in KEYS.items():
                                rel = 'blind/'+prefix+'/'+key
                                votes = record(prefix+'/'+key+'/votes', lambda secret=secret: backend.votes(spec, secret), rel+'/votes.json', [prefix+'/FFT'])
                                if votes is not None:
                                    local_bits[arm, oi, key, p] = [bit.get('decoded') for row in votes['local_cells'] for bit in row['bits']]
                            del spec
                        del z, view
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
        dump(directory/'raw_blind_complete.json',dict(saved_unix=time.time(),Q_slots=144,
            Q_returned=sum(journal.data[f'{h}/{arm}/r{oi}/p{p}/{key}/Q']['status']=='RETURNED' for arm in layout.ARMS for oi in range(6) for p in range(4) for key in KEYS),
            payload_expected=host['role']=='confirmation',new_score_not_yet_computed=True))
        journal.save()
    bit_families={q:{str(p):local_bits.get((meta['arm'],layout.ORIGINS.index(meta['origin']),meta['key'],p),[None]*80) for p in range(4)} for q,meta in layout.query_metadata().items()} if host['role']=='confirmation' else {}
    return dict(Q=observations,bits=bit_families)



def family_bank(collection,qid):
    meta=layout.query_metadata()[qid];oi=layout.ORIGINS.index(meta['origin'])
    return {str(p):collection['Q'].get((meta['arm'],oi,meta['key'],p)) for p in range(4)}


def fit_calibration_background(collections,cal_hosts):
    from main.tube_state import b_background_normalized_match_v1 as method
    return method.fit_background([[{key:{str(p):collections[h]['Q'].get(('OFF',oi,key,p)) for p in range(4)}
        for key in method.KEY_IDS} for oi in range(6)] for h in cal_hosts])


def score_host(host,collection,background,out,journal):
    from main.tube_state import b_background_normalized_match_v1 as method
    h=host['id'];families={}
    for q,meta in layout.query_metadata().items():
        def score():
            value=method.score_family(family_bank(collection,q),background['models'][meta['key']])
            value['phase_local_bits']=collection['bits'].get(q,{})
            return value
        value=journal.call(h+'/score_family/'+q,score) or method.empty_family('SCORE_CALL_FAILED')
        def save():
            dump(out/'hosts'/h/'blind/families'/(q+'.json'),value)
            return value
        families[q]=journal.call(h+'/family/'+q,save) or method.empty_family('FAMILY_SAVE_FAILED')
    dump(out/'hosts'/h/'cpu_counts.json',dict(normalized_candidate_slots=4932,feature_support_per_h=40,physical_Q_tables=144,physical_Q_values=253440))
    return families


def b0_reference(host,families,thresholds,out,backend,journal):
    from main.tube_state import b_background_normalized_match_v1 as method
    aliased={q:method.compatibility_input(f) for q,f in families.items()}
    return prior.run_b0(host,aliased,dict(status=thresholds['status'],tau=thresholds['tau_N']),out,backend,journal)


def run(output=None,*,config_path=CONFIG,backend=None,role_picker=None):
    out=Path(output or DEFAULT_OUTPUT)
    found=execution_artifacts(out)
    if found:return existing_attempt(out,found)
    config=load_config(config_path);budget=json.loads(BUDGET.read_text())
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as stream:
            json.dump(dict(card=101,started_unix=time.time(),scope='one fixed random-role task, no retries/resume'),stream);flush(stream)
    except FileExistsError:return existing_attempt(out)
    result=dict(status='RUNNING',card=101,CFAR=False,role_randomization_rank_guarantee=None,scientific_PASS=None,
        source_slots={h['id']:'NOT_ENTERED' for h in config['hosts']},default_changed=False,A_changed=False,writer_changed=False)
    journal=None;background=dict(status='BACKGROUND_UNAVAILABLE');thresholds=dict(status='CALIBRATION_UNRESOLVED',tau_N=None)
    collections={};families={};baselines={};decisions={}
    try:
        from main.tube_state import b_background_normalized_match_v1 as method
        roles=prior.assign_roles(config,out,role_picker)
        config=dict(config,hosts=[dict(h,role=roles['assignment'][h['id']]) for h in config['hosts']])
        journal=Journal(out,config,backend)
        dump(out/'protocol.json',dict(card=101,config=config,planned_budget=budget,
            working_model='one Q40 regularized positive normalized match',CFAR=False,role_rank_guarantee=None))
        if backend is None:
            from runtime.wan.b_background_normalized_backend_v1 import WanBackend
            backend=WanBackend(config['model_id']);journal.backend=backend
        for host in config['hosts']:
            if host['role']=='calibration':
                result['source_slots'][host['id']]='ENTERED'
                collections[host['id']]=collect_host(host,config,out,backend,journal)
        dump(out/'markers/calibration_raw_blind_complete.json',dict(saved_unix=time.time(),hosts=roles['calibration'],confirmation_sourceprep_entered=False))
        def fit():
            value=fit_calibration_background(collections,roles['calibration'])
            dump(out/'background.json',value);return value
        background=journal.call('background',fit) or background
        if background['status']=='BACKGROUND_FROZEN':
            for host in config['hosts']:
                if host['role']=='calibration':families[host['id']]=score_host(host,collections[host['id']],background,out,journal)
            dump(out/'markers/calibration_blind_complete.json',dict(saved_unix=time.time(),background_saved=True,confirmation_sourceprep_entered=False))
            def calibrate():
                value=method.calibrate_null({h:families[h] for h in roles['calibration']})
                value['frozen_unix']=time.time() if value['status']=='FROZEN' else None
                dump(out/'thresholds.json',value);return value
            thresholds=journal.call('calibration',calibrate) or thresholds
        else:
            dump(out/'background.json',background);dump(out/'thresholds.json',thresholds)
            journal.data['calibration']['unavailable_from']=['background']
        if thresholds['status']=='FROZEN':
            for host in config['hosts']:
                if host['role']=='confirmation':
                    h=host['id'];result['source_slots'][h]='ENTERED'
                    collections[h]=collect_host(host,config,out,backend,journal)
                    families[h]=score_host(host,collections[h],background,out,journal)
                    baselines[h]=b0_reference(host,families[h],thresholds,out,backend,journal)
            for h in roles['confirmation']:
                decisions[h]={}
                for q,family in families[h].items():
                    def decide():
                        value=method.decide(family,thresholds);dump(out/'decisions'/h/(q+'.json'),value);return value
                    decisions[h][q]=journal.call(h+'/decision/'+q,decide)
            blind_slots=[h+'/'+label+'/'+q for h in roles['confirmation'] for label in ('B0','decision') for q in layout.query_metadata()]
            unsaved=[s for s in blind_slots if journal.data[s]['status']!='RETURNED']
            if not unsaved:
                dump(out/'markers/confirmation_blind_complete.json',dict(saved_unix=time.time(),all_blind_outputs_saved=True,new_decisions=72,B0_pairs=72,threshold_freeze=thresholds['frozen_unix']))
                def post():
                    saved={h:{} for h in roles['confirmation']};base={h:{} for h in roles['confirmation']}
                    for h in roles['confirmation']:
                        for q in layout.query_metadata():
                            saved[h][q]=json.loads((out/'decisions'/h/(q+'.json')).read_text())
                            base[h][q]=dict(ungated=json.loads((out/'baselines'/h/(q+'.json')).read_text()),
                                gated=json.loads((out/'baselines'/h/(q+'.gated.json')).read_text()))
                    dump(out/'post_truth_access.json',dict(opened_unix=time.time(),all_actual_blind_saves=True))
                    from runtime.wan.b_background_normalized_evaluation_v1 import evaluate_user
                    value=evaluate_user(saved,base,{h:families[h] for h in roles['confirmation']},thresholds,config['writer']['bits'],roles)
                    dump(out/'evaluation.json',value);return value['status']
                journal.call('evaluation',post)
            else:journal.data['evaluation']['unavailable_from']=unsaved
        else:
            for slot,row in journal.data.items():
                if any(slot.startswith(h+'/') for h in roles['confirmation']) or row['kind']=='post_confirmation_evaluation':
                    row['unavailable_from']=['BACKGROUND_OR_60_NULL_FAMILIES_UNAVAILABLE']
        tech=sum(d is None or d['detection'] in ('UNRESOLVED_SCORE','CALIBRATION_UNRESOLVED') or any(b['status']=='TECHNICAL_UNRESOLVED' for b in d['message']) for ds in decisions.values() for d in ds.values())
        b0tech=sum(b['ungated']['status']=='TECHNICAL_UNRESOLVED' for bs in baselines.values() for b in bs.values())
        complete=all(r['status']=='RETURNED' for r in journal.data.values()) and all(f['status']=='SCORES_COMPLETE' for fs in families.values() for f in fs.values()) and tech==0 and b0tech==0
        result.update(status='COMPLETED_BACKGROUND_MATCH_EMPIRICAL_ONLY' if complete else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED',technical_queries=tech,technical_B0_queries=b0tech)
    except BaseException as e:result.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
    finally:
        if journal is not None:
            not_entered=sum(v['kind']=='absolute_compatibility_read' and v['status']=='NOT_ENTERED' for v in journal.data.values())
            result.update(decision_not_entered=not_entered,message_slots_not_entered=32*not_entered)
            journal.save();result.update(counts=journal.counts(),physical_events=journal.events,
                new_physical_run_entered=any(v['kind']=='generation_pipeline_load' and v['status']!='NOT_ENTERED' for v in journal.data.values()))
        result.update(background_status=background['status'],threshold_status=thresholds['status'])
        dump(out/'result.json',result)
    return result
